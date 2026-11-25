"""Resumable distributed state and portable deployment exports."""

import json
import os
import random
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.checkpoint import CheckpointException

from xense0vla.training.runtime import _format_checkpoint_load_failures
from xense0vla.utils import helper
from xense0vla.utils.hf_checkpoint import export_hf_checkpoint
from xense0vla.utils.runtime_assets import TRAINING_CONFIG_NAME


logger = helper.create_logger(__name__)

_SINGLE_HF_WEIGHT_FILENAMES = (
    "model.safetensors",
    "pytorch_model.bin",
)
_SHARDED_HF_INDEX_FILENAMES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def _has_hf_weights(directory: Path) -> bool:
    if any((directory / name).is_file() for name in _SINGLE_HF_WEIGHT_FILENAMES):
        return True
    for index_name in _SHARDED_HF_INDEX_FILENAMES:
        index_path = directory / index_name
        if not index_path.is_file():
            continue
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        shard_names = {str(name) for name in weight_map.values() if isinstance(name, str) and name}
        if not shard_names:
            return False
        if all((directory / shard_name).is_file() for shard_name in shard_names):
            return True
    return False


def _is_complete_export_dir(directory: Path) -> bool:
    return (
        (directory / "config.json").is_file()
        and (directory / TRAINING_CONFIG_NAME).is_file()
        and _has_hf_weights(directory)
    )


def is_complete_hf_export(checkpoint_dir: str | Path, *, standalone: bool) -> bool:
    """Return True only when ``hf_ckpt`` contains a complete deployable bundle."""
    hf_dir = Path(checkpoint_dir) / "hf_ckpt"
    if not _is_complete_export_dir(hf_dir):
        return False
    if standalone:
        return _is_complete_export_dir(hf_dir / "action_policy")
    return (hf_dir / "vlm_backbone" / "config.json").is_file()


def find_resume_checkpoints(train):
    if train.load_checkpoint_path:
        return [Path(train.load_checkpoint_path)]
    if not train.enable_resume:
        return []
    root = Path(train.output_dir) / "checkpoints"
    prefix = "global_step_"
    checkpoints = {int(p.name.removeprefix(prefix)): p for p in root.glob(prefix + "*")
                   if p.is_dir() and p.name.removeprefix(prefix).isdigit()}
    candidates = [checkpoints[step] for step in sorted(checkpoints, reverse=True)]
    if dist.is_initialized():
        shared = [candidates]
        dist.broadcast_object_list(shared, src=0)
        return shared[0]
    return candidates


def build_for_resume(train, candidates, *, standalone):
    """Reconstruct the saved architecture without fetching its original pretrained components."""
    from xense0vla.models import build_foundation_model, build_processor
    from xense0vla.models.configuration_vla import VLAConfig
    from xense0vla.models.module_utils import build_empty_policy, materialize_dcp_model
    from xense0vla.utils.runtime_assets import resolve_backbone_source

    # A completed export travels with its DCP. The run's model_assets also exists
    # if training was interrupted between the DCP write and the deployment export.
    exports = [p / "hf_ckpt" for p in candidates]
    assets = next((p for p in [*exports, candidates[0].parent.parent / "model_assets"]
                   if (p / "config.json").is_file()), None)
    if assets is None:
        raise FileNotFoundError(f"No saved model configuration for DCP resume at {candidates[0]}.")
    config = VLAConfig.from_pretrained(assets, local_files_only=True)
    if config.standalone_action_policy != standalone:
        raise ValueError("The DCP architecture does not match the selected training entry.")
    processor = None
    if not standalone:
        config.tokenizer_path = resolve_backbone_source(assets)
        processor = build_processor(config.tokenizer_path)

    if train.allow_partial_checkpoint:
        # A partial DCP can leave weights untouched. Seed them from a completed
        # local hf_ckpt containing weight files so every tensor is defined.
        weight_export = next(
            (p for p in exports if (p / "config.json").is_file() and _has_hf_weights(p)),
            None,
        )
        if weight_export is None:
            raise FileNotFoundError(
                "allow_partial_checkpoint=True requires a completed hf_ckpt with weight files "
                f"alongside the DCP candidates; checked {[str(p) for p in exports]}."
            )
        model = build_foundation_model(config, str(weight_export), train.parameter_dtype, train.init_device)
    else:
        model = build_empty_policy(config, getattr(torch, train.parameter_dtype))
        model._restore_from_dcp = True
        if train.init_device == "cuda" or (train.init_device == "cpu" and train.global_rank == 0):
            materialize_dcp_model(model, train.init_device)
    return model, processor


class TrainingCheckpoint:
    def __init__(self, *, args, model, optimizer, lr_scheduler, train_dataloader,
                 model_assets, writer, checkpointer):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.model_assets = model_assets
        self.writer = writer
        self.Checkpointer = checkpointer
        self._export_gloo_group = None
        self.last_restored_path: Optional[Path] = None

    def is_saved_in_output_dir(self, step: int) -> bool:
        """Return True when the restored checkpoint is already the step directory under output_dir."""
        if self.last_restored_path is None or step <= 0:
            return False
        expected = (
            Path(self.args.train.save_checkpoint_path) / f"global_step_{step}"
        ).expanduser().resolve()
        return self.last_restored_path.expanduser().resolve() == expected

    def has_complete_export(self, step: int) -> bool:
        """Check on rank 0 whether ``global_step_<step>/hf_ckpt`` is complete and broadcast the result."""
        if step <= 0:
            return False
        standalone = bool(
            any(getattr(asset, "standalone_action_policy", False) for asset in self.model_assets)
        )
        checkpoint_dir = Path(self.args.train.save_checkpoint_path) / f"global_step_{step}"
        if self.args.train.global_rank == 0:
            complete = [is_complete_hf_export(checkpoint_dir, standalone=standalone)]
        else:
            complete = [False]
        if dist.is_initialized():
            dist.broadcast_object_list(complete, src=0)
        return bool(complete[0])

    def _get_export_sync_group(self):
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return None
        if self._export_gloo_group is None:
            try:
                self._export_gloo_group = dist.new_group(
                    backend="gloo",
                    timeout=timedelta(hours=2),
                )
            except Exception as exc:
                logger.warning_rank0(
                    f"Falling back to default process group for export sync ({exc!r})."
                )
                return None
        return self._export_gloo_group

    def export_deployment_checkpoint(self, checkpoint_path: str) -> None:
        """Export BF16 model-only weights and fail every rank together."""

        failure = [None]
        local_error = None
        hf_path = None
        sync_group = self._get_export_sync_group()
        if self.args.train.global_rank == 0:
            try:
                hf_path = export_hf_checkpoint(
                    checkpoint_path=checkpoint_path,
                    output_dir=self.args.train.output_dir,
                    model_assets=self.model_assets,
                )
            except Exception as exc:
                local_error = exc
                failure[0] = repr(exc)
        if dist.is_initialized():
            dist.broadcast_object_list(failure, src=0, group=sync_group)
        if failure[0] is not None:
            raise RuntimeError(
                f"Failed to export deployment hf_ckpt from {checkpoint_path}: "
                f"{failure[0]}"
            ) from local_error
        logger.info_rank0(
            f"BF16 deployment checkpoint saved at {hf_path} successfully!"
        )

    def save(self, step: int) -> None:
        """Save resumable DCP state, then always export its standalone hf_ckpt."""

        helper.empty_cache()
        checkpoint_path = os.path.join(
            self.args.train.save_checkpoint_path,
            f"global_step_{step}",
        )
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "extra_state": {
                "global_step": step,
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "train_dataloader": self.train_dataloader.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
            },
        }
        if self.args.train.global_rank == 0:
            self.writer.flush()
        self.Checkpointer.save(
            self.args.train.save_checkpoint_path,
            state,
            global_steps=step,
        )
        dist.barrier()
        logger.info_rank0(
            f"Distributed training checkpoint saved at {checkpoint_path} successfully!"
        )
        self.export_deployment_checkpoint(checkpoint_path)

    def restore(self, candidates):
        if not candidates:
            logger.info_rank0("Starting training from scratch.")
            return 0, 0, 0

        failures = []
        for path in candidates:
            state = {"model": self.model, "optimizer": self.optimizer, "extra_state": {}}
            error = None
            try:
                self.Checkpointer.load(path, state, allow_partial_load=self.args.train.allow_partial_checkpoint)
                extra = state["extra_state"]
                global_step = extra["global_step"]
                epoch, step = divmod(global_step, self.args.train.train_steps)
                self.lr_scheduler.load_state_dict(extra["lr_scheduler"])
                if step > 0 and self.args.train.resume_dataloader_state:
                    self.train_dataloader.load_state_dict(extra["train_dataloader"])
                if extra.get("torch_rng_state") is not None:
                    torch.set_rng_state(extra["torch_rng_state"])
                if extra.get("cuda_rng_state") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state(extra["cuda_rng_state"])
                if extra.get("python_rng_state") is not None:
                    random.setstate(extra["python_rng_state"])
                if extra.get("numpy_rng_state") is not None:
                    np.random.set_state(extra["numpy_rng_state"])
            except (Exception, CheckpointException) as exc:
                error = repr(exc)

            # A missing per-rank dataloader/RNG file must send every rank to
            # the same next candidate before any new DCP collective starts.
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, error)
            if any(errors):
                detail = next(message for message in errors if message)
                failures.append((str(path), detail))
                logger.warning_rank0(f"Failed to restore {path}: {detail}")
                continue
            self.last_restored_path = Path(path)
            logger.info_rank0(f"Restored distributed checkpoint {path}.")
            return epoch, step, global_step

        raise RuntimeError(f"Failed to restore the requested checkpoints:\n{_format_checkpoint_load_failures(failures)}")
