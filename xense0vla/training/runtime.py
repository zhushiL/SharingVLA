"""CUDA stage timing and checkpoint diagnostics."""

from typing import Dict, List

import torch


def _cuda_stage_start(
    stage_events: Dict[str, List[tuple[torch.cuda.Event, torch.cuda.Event]]],
    name: str,
    enabled: bool,
) -> None:
    if not enabled:
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    stage_events.setdefault(name, []).append((start, end))


def _cuda_stage_end(
    stage_events: Dict[str, List[tuple[torch.cuda.Event, torch.cuda.Event]]],
    name: str,
    enabled: bool,
) -> None:
    if enabled:
        stage_events[name][-1][1].record()


def _elapsed_cuda_stages(
    stage_events: Dict[str, List[tuple[torch.cuda.Event, torch.cuda.Event]]],
    stage_names: List[str],
) -> List[float]:
    return [sum(start.elapsed_time(end) for start, end in stage_events.get(name, ())) / 1000.0 for name in stage_names]


def _format_checkpoint_load_failures(failures: List[tuple[str, str]]) -> str:
    return "\n".join(f"  - {path}: {error}" for path, error in failures)
