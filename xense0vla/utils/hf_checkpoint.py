"""Synchronous export of a standalone VLA deployment checkpoint."""

from __future__ import annotations

import json
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
        state_dict = ckpt_to_state_dict(checkpoint_path=checkpoint_path)
        # Convert only the deployment copy, never the live model or DCP state.
        state_dict = {
            name: tensor.detach().to(device="cpu", dtype=torch.bfloat16) if tensor.is_floating_point() else tensor
            for name, tensor in state_dict.items()
        }
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
            # Export the complete trainable policy separately so mid_train and
            # action_posttrain never need to initialize a language backbone.
            prefix = "model.action_policy."
            action_state = {
                name[len(prefix) :]: tensor for name, tensor in state_dict.items() if name.startswith(prefix)
            }
            if not action_state:
                raise ValueError("Standalone checkpoint contains no action policy parameters.")
            action_dir = os.path.join(tmp_dir, "action_policy")
            save_model_weights(action_dir, action_state, save_dtype=None)
            vision_prefix = "model.backbone.visual."
            vision_state = {
                name[len(vision_prefix) :]: tensor
                for name, tensor in state_dict.items()
                if name.startswith(vision_prefix)
            }
            if not vision_state:
                raise ValueError("Standalone checkpoint contains no Qwen vision parameters.")
            save_model_weights(os.path.join(action_dir, "vision_encoder"), vision_state, save_dtype=None)
            with open(os.path.join(action_dir, "config.json"), "w", encoding="utf-8") as handle:
                json.dump(config.to_dict(), handle, indent=2)

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
