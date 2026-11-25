"""Flow matching for a standalone wrist policy or a language-conditioned VLA."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel

from .action_policy import ActionPolicy, ActionPolicyBlock
from .configuration_vla import VLAConfig
from .policy_utils import action_joint_mask, sample_beta, sample_clean_prefix


class FlowMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.action_policy = ActionPolicy(config)
        self.backbone = None
        if not config.standalone_action_policy:
            from .qwen_backbone import QwenBackbone

            self.backbone = QwenBackbone(config)
        if config.freeze_action_policy:
            self.action_policy.requires_grad_(False)
            if config.unfreeze_mot_layers:
                self.action_policy.layers[2::3].requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_action_policy:
            self.action_policy.eval()
            if self.config.unfreeze_mot_layers:
                self.action_policy.layers[2::3].train(mode)
        return self

    def encode_observations(
        self,
        wrist_images,
        head_images=None,
        image_grid_thw=None,
        lang_tokens=None,
        lang_masks=None,
        subtask_tokens=None,
        subtask_masks=None,
    ):
        spatial_tokens = self.action_policy.encode_spatial_tokens(wrist_images)
        zero = spatial_tokens.new_zeros((), dtype=torch.float32)
        if self.backbone is None:
            return spatial_tokens, None, None, None, zero
        batch, count = wrist_images.shape[0], self.config.history_memory + 1
        if head_images is None or head_images.shape[:2] != (batch, count):
            raise ValueError("VLA requires history_memory + 1 preprocessed head images.")
        if image_grid_thw is None or image_grid_thw.shape != (batch, count, 3):
            raise ValueError("Each head frame requires its own Qwen image_grid_thw.")
        encoded = self.backbone.embed_images(head_images, image_grid_thw.reshape(-1, 3))
        encoded = encoded.reshape(batch, count, 64, -1)
        if lang_tokens is None or lang_masks is None:
            raise ValueError("The full model requires task tokens and task masks.")
        (
            hidden,
            padding,
            positions,
            prefix_length,
            prefix_mask,
            position_start,
            subtask_gather_pos,
        ) = self.backbone.embed_prefix(
            encoded,
            lang_tokens,
            lang_masks,
            subtask_tokens,
            subtask_masks,
        )
        need_outputs = subtask_tokens is not None
        outputs, cache = self.backbone.encode_prefix(
            hidden,
            padding,
            positions,
            prefix_length,
            need_outputs=need_outputs,
        )
        language_loss = zero
        if need_outputs:
            # Fixed-shape gather + ignore_index=-100 avoids dynamic boolean
            # indexing GPU syncs, torch.compile graph breaks, and unannotated-batch
            # special cases while keeping checkpointed vocab-sized chunks bounded.
            hidden_dim = outputs.shape[-1]
            subtask_hidden = outputs.gather(
                1,
                subtask_gather_pos.unsqueeze(-1).expand(-1, -1, hidden_dim),
            ).reshape(-1, hidden_dim)
            selected = subtask_masks.bool()
            labels = subtask_tokens.masked_fill(~selected, -100).reshape(-1)
            target_count = selected.sum().clamp(min=1)

            def chunk_loss(hidden_chunk, label_chunk):
                logits = self.backbone.qwen35.lm_head(hidden_chunk)
                return F.cross_entropy(logits.float(), label_chunk, reduction="sum", ignore_index=-100)

            loss_sum = zero
            for start in range(0, labels.shape[0], 128):
                args = (subtask_hidden[start : start + 128], labels[start : start + 128])
                if self.training and torch.is_grad_enabled():
                    loss_sum = loss_sum + checkpoint(chunk_loss, *args, use_reentrant=False)
                else:
                    loss_sum = loss_sum + chunk_loss(*args)
            language_loss = loss_sum / target_count
        return spatial_tokens, cache, prefix_mask, position_start, language_loss

    def forward(
        self,
        wrist_images,
        actions,
        head_images=None,
        lang_tokens=None,
        lang_masks=None,
        subtask_tokens=None,
        subtask_masks=None,
        image_grid_thw=None,
        noise=None,
        time=None,
        action_valid_mask=None,
        clean_prefix=None,
    ):
        if noise is None:
            noise = torch.randn_like(actions)
        if noise.shape != actions.shape:
            raise ValueError("noise must match actions.")
        if time is None:
            time = sample_beta(1.5, 1.0, actions.shape[0], actions.device) * 0.999 + 0.001
        if time.shape != (actions.shape[0],):
            raise ValueError("time must have shape [batch].")
        if action_valid_mask is None:
            action_valid_mask = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
        elif action_valid_mask.shape != actions.shape[:2]:
            raise ValueError("action_valid_mask must have shape [batch, horizon].")
        if clean_prefix is None:
            clean_prefix = sample_clean_prefix(action_valid_mask, self.config.rtc_max_delay_steps)
        action_time = torch.where(clean_prefix, 0.0, time.float()[:, None])
        noisy = (
            action_time[..., None] * noise.float() + (1.0 - action_time[..., None]) * actions.float()
        ).to(actions.dtype)
        if self.training and self.config.use_subtask_loss:
            if subtask_tokens is None or subtask_masks is None:
                raise ValueError("This training stage requires current-frame subtask targets.")
        else:
            subtask_tokens = subtask_masks = None
        spatial_tokens, cache, prefix_mask, position_start, language_loss = self.encode_observations(
            wrist_images,
            head_images,
            image_grid_thw,
            lang_tokens,
            lang_masks,
            subtask_tokens,
            subtask_masks,
        )
        velocity = self.action_policy(
            noisy, action_time, spatial_tokens, cache, prefix_mask, position_start, action_valid_mask
        )
        target = noise - actions
        losses = (
            F.mse_loss(velocity, target, reduction="none")
            if self.config.loss_type == "fm"
            else F.l1_loss(velocity, target, reduction="none")
        )
        metrics = {
            "rtc/delay_mean": clean_prefix.sum(dim=1).float().mean().detach(),
            "rtc/prefix_fraction": clean_prefix.float().mean().detach(),
        }
        return losses, language_loss, metrics, ~clean_prefix

    @torch.no_grad()
    def sample_actions(
        self,
        wrist_images,
        head_images=None,
        state=None,
        lang_tokens=None,
        lang_masks=None,
        noise=None,
        image_grid_thw=None,
        action_prefix=None,
        prefix_lengths=None,
    ):
        batch = wrist_images.shape[0]
        device = wrist_images.device
        dtype = self.action_policy.action_in_projection.weight.dtype
        expected = (batch, self.config.chunk_size, self.config.max_action_dim)
        if noise is None:
            noise = torch.randn(expected, device=device, dtype=torch.float32)
        else:
            if noise.shape != expected:
                raise ValueError(f"noise must have shape {expected}.")
            noise = noise.to(device=device, dtype=torch.float32)
        if (action_prefix is None) != (prefix_lengths is None):
            raise ValueError("action_prefix and prefix_lengths must be supplied together.")
        clean_prefix = torch.zeros(expected[:2], device=device, dtype=torch.bool)
        action_prefix_fp32 = None
        if action_prefix is not None:
            if not self.config.rtc_training_enabled:
                raise ValueError("Prefix-conditioned inference requires an RTC-trained checkpoint.")
            if action_prefix.shape != expected or prefix_lengths.shape != (batch,):
                raise ValueError("Invalid action_prefix or prefix_lengths shape.")
            prefix_lengths = prefix_lengths.to(device=device, dtype=torch.long)
            if (
                not torch.compiler.is_compiling()
                and ((prefix_lengths < 0) | (prefix_lengths > self.config.rtc_max_delay_steps)).any()
            ):
                raise ValueError("Requested delay exceeds the trained RTC range.")
            action_prefix_fp32 = action_prefix.to(device=device, dtype=torch.float32)
            clean_prefix = torch.arange(self.config.chunk_size, device=device)[None] < prefix_lengths[:, None]
        # Spatial encoder, VLM, and spatial-token KV caches each run once.
        spatial_tokens, cache, prefix_mask, position_start, _ = self.encode_observations(
            wrist_images,
            head_images,
            image_grid_thw,
            lang_tokens,
            lang_masks,
        )
        cached_kv, action_embeddings, action_mot_mask = self.action_policy.prefill_spatial_tokens(
            spatial_tokens,
            cache,
            prefix_mask,
            position_start,
        )
        # Keep ODE time and Euler accumulator in FP32 to prevent BF16 high-frequency
        # sinusoidal phase distortion and per-step accumulation drift.
        current = (
            torch.where(clean_prefix[..., None], action_prefix_fp32, noise)
            if action_prefix_fp32 is not None
            else noise
        )
        step_size = 1.0 / self.config.num_steps
        for index in range(self.config.num_steps):
            time = torch.full(
                expected[:2],
                1.0 - index * step_size,
                device=device,
                dtype=torch.float32,
            )
            if action_prefix_fp32 is not None:
                time = time.masked_fill(clean_prefix, 0.0)
            velocity = self.action_policy.denoise_step(
                current.to(dtype),
                time,
                cached_kv,
                action_embeddings,
                action_mot_mask,
            )
            current = current - velocity.float() * step_size
            if action_prefix_fp32 is not None:
                current = torch.where(clean_prefix[..., None], action_prefix_fp32, current)
        return current


class VLAModel(PreTrainedModel):
    config_class = VLAConfig
    name = "torch_vla"
    main_input_name = "wrist_images"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules = ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock", "ActionPolicyBlock", "Block"]

    def __init__(self, config, eval=False):
        super().__init__(config)
        self.model = FlowMatching(config)
        self.post_init()
        if eval:
            self.eval()

    def _init_weights(self, module):
        std = float(getattr(self.config, "initializer_range", 0.02) or 0.02)
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.GroupNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, ActionPolicyBlock):
            nn.init.normal_(module.modulation, std=module.modulation.shape[-1] ** -0.5)
        if isinstance(module, ActionPolicy):
            nn.init.normal_(module.output_modulation, std=module.output_modulation.shape[-1] ** -0.5)

    def get_optim_params(self):
        return self.parameters()

    def forward(
        self,
        wrist_images,
        actions,
        head_images=None,
        lang_tokens=None,
        lang_masks=None,
        subtask_tokens=None,
        subtask_masks=None,
        joint_mask=None,
        noise=None,
        time=None,
        image_grid_thw=None,
        clean_prefix=None,
        **kwargs,
    ):
        joint_mask = action_joint_mask(actions, joint_mask, self.config.action_dim)
        losses, language_loss, metrics, postfix = self.model(
            wrist_images,
            actions,
            head_images,
            lang_tokens,
            lang_masks,
            subtask_tokens,
            subtask_masks,
            image_grid_thw,
            noise,
            time,
            joint_mask.any(dim=-1),
            clean_prefix,
        )
        effective = joint_mask & postfix[..., None]
        masked = losses * effective
        action_loss = masked.sum() / effective.sum().clamp(min=1)
        batch_losses = masked.sum(dim=(1, 2)) / effective.sum(dim=(1, 2)).clamp(min=1)
        total = self.config.action_loss_weight * action_loss + self.config.language_loss_weight * language_loss
        return total, action_loss, language_loss, {"batch_mean_losses": batch_losses.detach(), **metrics}, {}

    def sample_actions(self, *args, **kwargs):
        return self.model.sample_actions(*args, **kwargs)


__all__ = ["VLAModel", "FlowMatching", "ActionPolicy"]
