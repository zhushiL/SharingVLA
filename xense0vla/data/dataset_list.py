"""Dataset-list parsing and deterministic frame subsets shared by training and stats."""

import hashlib
import math
import shlex
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetEntry:
    robot: str
    path: str
    sampling_ratio: float = 1.0


def read_dataset_list(task_files, robot_names=None):
    """Read ``robot sampling_ratio=0.41 path`` lines, skipping disabled entries.

    Legacy unquoted paths with spaces remain supported. Quote paths containing
    ``#``. Old two-column lists and trailing-ratio lists remain supported.
    Relative paths use the cwd; the ``sampling_ratio=`` token is reserved.
    """
    entries = []
    for task_file in task_files.split(","):
        task_file = task_file.strip()
        if not task_file.lower().endswith(".txt"):
            raise ValueError(f"Dataset list must be a .txt file: {task_file}")
        with open(task_file, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                location = f"{task_file}:{line_number}"
                try:
                    fields = shlex.split(line, comments=True)
                except ValueError as error:
                    raise ValueError(f"{location}: {error}") from error
                if not fields:
                    continue
                expected = f"{location}: expected 'robot_name sampling_ratio=0.41 path'."
                if len(fields) < 2:
                    raise ValueError(expected)
                ratio = 1.0
                ratio_positions = [i for i, part in enumerate(fields[1:], 1) if part.startswith("sampling_ratio=")]
                if len(ratio_positions) > 1 or (ratio_positions and ratio_positions[0] not in (1, len(fields) - 1)):
                    raise ValueError(expected)
                if ratio_positions:
                    value = fields.pop(ratio_positions[0]).split("=", 1)[1]
                    try:
                        ratio = float(value)
                    except ValueError as error:
                        raise ValueError(f"{location}: invalid sampling_ratio {value!r}.") from error
                    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
                        raise ValueError(f"{location}: sampling_ratio must be finite and in [0, 1], got {value!r}.")
                if len(fields) < 2:
                    raise ValueError(expected)
                robot, path = fields[0], " ".join(fields[1:])
                if not path:
                    raise ValueError(f"{location}: dataset path must not be empty.")
                if ratio == 0 or (robot_names is not None and robot not in robot_names):
                    continue
                entries.append(DatasetEntry(robot, path, ratio))
    if not entries:
        raise ValueError("The dataset lists contain no enabled datasets matching the requested robots.")
    return entries


def get_all_tasks(task_files, robot_names=None, *, with_sampling_ratios=False):
    """Keep the legacy two-list return value unless ratios are explicitly requested."""
    entries = read_dataset_list(task_files, robot_names)
    result = ([entry.robot for entry in entries], [entry.path for entry in entries])
    if with_sampling_ratios:
        return (*result, [entry.sampling_ratio for entry in entries])
    return result


def format_dataset_entry(robot, path, sampling_ratio=1.0):
    """Preserve paths and full ratio precision when rewriting a filtered list."""
    return f"{shlex.quote(robot)} sampling_ratio={sampling_ratio!r} {shlex.quote(path)}"


def sample_frame_indices(num_frames, ratio, *, seed=42, dataset_id=""):
    """Return sorted unique native row indices, or None for the full dataset.

    Selection is fixed across epochs/ranks and independent of list order or the
    global RNG. Only sample starts are filtered; action/history queries still
    address the original dataset and retain episode boundaries and continuity.
    """
    if num_frames <= 0:
        raise ValueError(f"Dataset {dataset_id!r} has no frames.")
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("Active dataset sampling_ratio must be finite and in (0, 1].")
    if seed < 0:
        raise ValueError("sampling_seed must be non-negative.")
    count = max(1, int(num_frames * ratio))
    if count == num_frames:
        return None
    identity = int.from_bytes(hashlib.sha256(dataset_id.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng([seed, identity])
    indices = rng.choice(num_frames, size=count, replace=False)
    indices.sort()
    return indices
