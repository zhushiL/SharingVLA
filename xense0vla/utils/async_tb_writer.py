"""Non-blocking TensorBoard writer backed by a daemon thread.

Wraps ``SummaryWriter`` so that actual file I/O runs on a background
thread.  The training loop only pays for ``.item()`` (memcpy, GPU
already synced) and ``queue.put()`` (microseconds).
"""

import logging
import queue
import threading

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class AsyncTBWriter:

    _QUEUE_WARN_THRESHOLD = 500

    def __init__(self, log_dir: str):
        self._writer = SummaryWriter(log_dir=log_dir)
        self._queue: queue.Queue = queue.Queue()
        self._warn_logged = False
        self._closed = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ---- public API (called from main thread) ----

    def add_scalar(self, tag, scalar_value, global_step):
        if self._closed:
            return
        self._maybe_warn_queue_size()
        self._queue.put(("add_scalar", (tag, scalar_value, global_step)))

    def add_histogram(self, tag, values, global_step):
        if self._closed:
            return
        self._queue.put(("add_histogram", (tag, values, global_step)))

    def add_histogram_from_counts(self, tag, counts_cpu, global_step):
        """Queue histogram write from a CPU counts tensor [num_experts].

        The ``repeat_interleave`` prep runs on the background thread so
        the main thread is not blocked.
        """
        if self._closed:
            return
        self._queue.put(("_hist_from_counts", (tag, counts_cpu, global_step)))

    def add_expert_bar(self, tag, counts_cpu, global_step, title=None):
        """Queue an exact per-expert bar chart (matplotlib figure) from a CPU
        counts tensor [num_experts].
        """
        if self._closed:
            return
        self._queue.put(("_expert_bar", (tag, counts_cpu, global_step, title)))

    def flush(self, timeout: float = 30.0):
        """Block until all queued writes are done, then flush the event file on the worker thread."""
        if self._closed or not self._thread.is_alive():
            return
        done = threading.Event()
        self._queue.put(("_flush", (done,)))
        if not done.wait(timeout=timeout):
            logger.warning("AsyncTBWriter: flush timed out after %.1fs", timeout)

    def close(self, timeout: float = 10.0):
        if self._closed:
            return
        self._closed = True
        if self._thread.is_alive():
            self._queue.put(None)  # sentinel to flush and stop worker
            self._thread.join(timeout=timeout)
        if not self._thread.is_alive():
            self._writer.close()
        else:
            logger.warning("AsyncTBWriter: worker thread did not exit within %.1fs", timeout)

    # ---- internals ----

    def _maybe_warn_queue_size(self):
        qsize = self._queue.qsize()
        if qsize > self._QUEUE_WARN_THRESHOLD and not self._warn_logged:
            logger.warning(
                "AsyncTBWriter: queue backlog = %d, NFS may be slow", qsize
            )
            self._warn_logged = True
        elif qsize <= self._QUEUE_WARN_THRESHOLD // 2:
            self._warn_logged = False

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                try:
                    self._writer.flush()
                except Exception as e:
                    logger.warning("AsyncTBWriter: final flush failed: %s", e)
                self._queue.task_done()
                break
            method, args = item
            try:
                if method == "_flush":
                    done_event = args[0]
                    try:
                        self._writer.flush()
                    finally:
                        done_event.set()
                elif method == "_hist_from_counts":
                    self._do_histogram_from_counts(*args)
                elif method == "_expert_bar":
                    self._do_expert_bar(*args)
                else:
                    getattr(self._writer, method)(*args)
            except Exception as e:
                logger.warning("AsyncTBWriter: %s failed: %s", method, e)
            self._queue.task_done()

    def _do_expert_bar(self, tag, counts_cpu, global_step, title=None):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        counts = counts_cpu.detach().float().numpy()
        n = counts.shape[0]
        fig = Figure(figsize=(10, 4))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.bar(range(n), counts)
        avg = counts.mean()
        ax.axhline(avg, color="r", linestyle="--", linewidth=1, label=f"avg={avg:.0f}")
        ax.set_title(title or f"Expert load (step {global_step})")
        ax.set_xlabel("Expert ID")
        ax.set_ylabel("Token count")
        ax.legend(loc="upper right", fontsize=8)
        self._writer.add_figure(tag, fig, global_step)

    def _do_histogram_from_counts(self, tag, counts_cpu, global_step):
        total = counts_cpu.sum().item()
        if total <= 0:
            return
        max_hist_events = 100000
        if total > max_hist_events:
            counts_cpu = counts_cpu * (max_hist_events / total)
        counts_long = counts_cpu.round().long().clamp(min=0)
        num_experts = counts_long.numel()
        indices = torch.repeat_interleave(
            torch.arange(num_experts), counts_long
        )
        if indices.numel() > 0:
            # Integer-aligned bin edges [-0.5, 0.5, ..., num_experts-0.5] so each
            # expert id maps to exactly one bucket. The default 'tensorflow' bins
            # are exponentially spaced and merge high-index experts (2-3 per bucket
            # past id ~11), producing fake peaks/holes in the per-expert view.
            edges = np.arange(num_experts + 1, dtype=np.float64) - 0.5
            self._writer.add_histogram(tag, indices, global_step, bins=edges)
