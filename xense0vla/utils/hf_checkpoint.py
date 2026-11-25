"""Synchronous export of a standalone VLA deployment checkpoint."""

from __future__ import annotations

import os
import shutil
from typing import Any, Sequence

import torch

from xense0vla.checkpoint import ckpt_to_state_dict
from xense0vla.models import save_model_weights
from xense0vla.utils import helper
from xense0vla.utils.runtime_assets import bundle_runtime_assets


def export_hf_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    model_assets: Sequence[Any],
) -> str:
    """Export only model weights and deployment assets into ``hf_ckpt``.

    The distributed checkpoint remains the source of truth for exact training
    resume and contains optimizer/extra state. This export deliberately reads
    only its model state, converting floating-point deployment tensors to BF16.
    Resumable DCP tensors and non-floating buffers retain their original dtype.
    """

    checkpoint_path = os.path.abspath(checkpoint_path)
    final_dir = os.path.join(checkpoint_path, "hf_ckpt")
    tmp_dir = os.path.join(
        checkpoint_path,
        f".hf_ckpt.tmp.{os.getpid()}",
    )
    backup_dir = os.path.join(checkpoint_path, ".hf_ckpt.previous")

    # Recover a previous publish interrupted between moving the old artifact
    # aside and installing the new one.
    if os.path.exists(backup_dir):
        if os.path.exists(final_dir):
            shutil.rmtree(backup_dir)
        else:
            os.replace(backup_dir, final_dir)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    state_dict = None
    try:
        # ckpt_to_state_dict reads checkpoint_path/model only. Optimizer,
        # scheduler, RNG and dataloader state never enter the HF artifact.
        raw_state_dict = ckpt_to_state_dict(checkpoint_path=checkpoint_path)
        # Convert tensors in place by popping from raw_state_dict so FP32 and
        # BF16 copies are never simultaneously resident in rank-0 CPU RAM.
        state_dict = {}
        for name in list(raw_state_dict.keys()):
            tensor = raw_state_dict.pop(name)
            state_dict[name] = (
                tensor.detach().to(device="cpu", dtype=torch.bfloat16)
                if tensor.is_floating_point()
                else tensor.detach().cpu()
            )
            del tensor
        del raw_state_dict
        save_model_weights(
            tmp_dir,
            state_dict,
            save_dtype=None,
            model_assets=model_assets,
        )
        bundle_runtime_assets(output_dir, tmp_dir)
        config = next(
            (asset for asset in model_assets if getattr(asset, "model_type", None) == "vla"),
            None,
        )
        if config is not None and config.standalone_action_policy:
            # Publish the transferable subdirectory with identical standalone keys and assets.
            # Hard links keep one payload on disk while both paths can be moved independently.
            action_dir = os.path.join(tmp_dir, "action_policy")
            def link_or_copy(source, destination):
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)

            shutil.copytree(tmp_dir, action_dir, copy_function=link_or_copy,
                            ignore=shutil.ignore_patterns("action_policy"))

        # Keep the previous complete artifact recoverable until the new staging
        # directory has been published successfully.
        if os.path.exists(final_dir):
            os.replace(final_dir, backup_dir)
        try:
            os.replace(tmp_dir, final_dir)
        except BaseException:
            if os.path.exists(backup_dir) and not os.path.exists(final_dir):
                os.replace(backup_dir, final_dir)
            raise
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        return final_dir
    finally:
        del state_dict
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        helper.empty_cache()


__all__ = ["export_hf_checkpoint"]
