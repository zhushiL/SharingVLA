"""Resumable distributed state and portable deployment exports."""

import os
import re

import torch
import torch.distributed as dist

from xense0vla.training.runtime import _format_checkpoint_load_failures
from xense0vla.utils import helper
from xense0vla.utils.hf_checkpoint import export_hf_checkpoint


logger = helper.create_logger(__name__)


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

    def export_deployment_checkpoint(self, checkpoint_path: str) -> None:
        """Export BF16 model-only weights and fail every rank together."""

        failure = [None]
        local_error = None
        hf_path = None
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
        dist.broadcast_object_list(failure, src=0)
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

    def restore(self):
        start_epoch, start_step, global_step = 0, 0, 0
        load_checkpoint_path = None
        candidates = []
        if self.args.train.load_checkpoint_path or self.args.train.enable_resume:
            if self.args.train.load_checkpoint_path:
                load_checkpoint_path = self.args.train.load_checkpoint_path
                candidates = [load_checkpoint_path]
            elif self.args.train.enable_resume:
                checkpoint_dir = f'{self.args.train.output_dir}/checkpoints'
                if os.path.exists(checkpoint_dir):
                    pattern = re.compile(r"global_step_(\d+)")
                    tmp = []
                    for dirname in os.listdir(checkpoint_dir):
                        match = pattern.fullmatch(dirname)
                        if match:
                            step = int(match.group(1))
                            tmp.append((step, os.path.join(checkpoint_dir, dirname)))
                    tmp.sort(key=lambda x: x[0], reverse=True)
                    candidates = [p for _, p in tmp]
                if candidates:
                    load_checkpoint_path = candidates[0]
                else:
                    logger.info_rank0(f"No checkpoints in {self.args.train.output_dir} now!")
        if candidates:
            loaded = False
            checkpoint_failures = []
            for candidate_index, cp in enumerate(candidates):
                state = {"model": self.model, "optimizer": self.optimizer, "extra_state": {}}  # cannot be None
                try:
                    self.Checkpointer.load(cp, state, allow_partial_load=getattr(self.args.train, 'allow_partial_checkpoint', False))
                    global_step = state["extra_state"]["global_step"]
                    start_epoch = global_step // self.args.train.train_steps
                    start_step = global_step % self.args.train.train_steps
                    self.lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
                    if start_step > 0 and self.args.train.resume_dataloader_state:
                        self.train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
                    torch.set_rng_state(state["extra_state"]["torch_rng_state"])
                    if start_step == 0:  # resume at the end of epoch
                        iter(self.train_dataloader)  # clear resume state and prefetch data
                    dist.barrier()
                    logger.info_rank0(f"Load distributed checkpoint from {cp} successfully!")
                    loaded = True
                    break
                except Exception as e:
                    checkpoint_failures.append((cp, repr(e)))
                    retry_message = (
                        " Trying the next older checkpoint."
                        if candidate_index + 1 < len(candidates)
                        else ""
                    )
                    logger.warning_rank0(
                        f"Failed to load checkpoint {cp}: {repr(e)}.{retry_message}"
                    )
                    continue
            if not loaded:
                failure_summary = _format_checkpoint_load_failures(checkpoint_failures)
                raise RuntimeError(
                    "Every requested or discovered checkpoint failed to load; "
                    "refusing to start from scratch in an existing run directory.\n"
                    f"{failure_summary}"
                )
        else:
            logger.info_rank0("Starting training from scratch.")

        return start_epoch, start_step, global_step
