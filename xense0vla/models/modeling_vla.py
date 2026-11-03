"""Qwen3.5 language supervision and an independently deployable action policy."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, PreTrainedModel
from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNorm,
    Qwen3_5VisionModel,
    apply_rotary_pos_emb,
    repeat_kv,
)

from xense0vla.models.loader import VLAWeightLoader

from .action_policy import ActionPolicy, ActionPolicyBlock
from .configuration_vla import VLAConfig
from .policy_utils import sample_beta, sample_rtc_delays
from .qwen35_mot import apply_qwen35_mot_patch, qwen35_layer_type


class PolicyWeightLoader(VLAWeightLoader):
    """Map only the visual tower for standalone initialization, or all Qwen weights."""

    def get_vlm_submodule(self, model):
        backbone = model.model.backbone
        self.standalone = model.config.standalone_action_policy
        self.tied = bool(backbone.vlm_config.tie_word_embeddings)
        return backbone.visual if self.standalone else backbone.qwen35

    def map_ckpt_key(self, key, load_vlm_only, load_full_checkpoint):
        if not load_vlm_only:
            return key
        if key.startswith("mtp.") or key.startswith("model.mtp."):
            return None
        if self.standalone:
            if key.startswith("model.visual."):
                return "model.backbone.visual." + key.removeprefix("model.visual.")
            return None
        if self.tied and key.startswith("lm_head."):
            return None
        return "model.backbone.qwen35." + key


class QwenBackbone(nn.Module):
    """The standalone variant constructs the Qwen visual model only."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        apply_qwen35_mot_patch()
        if config.backbone_config is not None:
            vlm_config = Qwen3_5Config(**config.backbone_config)
        elif config.tokenizer_path:
            vlm_config = AutoConfig.from_pretrained(config.tokenizer_path, trust_remote_code=False)
        else:
            raise ValueError("Provide tokenizer_path/vlm_path or an embedded backbone_config.")
        if vlm_config.model_type != "qwen3_5":
            raise ValueError("This policy requires a Qwen3.5 backbone.")
        vlm_config._attn_implementation = config.backbone_attention_implementation
        vlm_config.text_config._attn_implementation = config.backbone_attention_implementation
        vlm_config.vision_config._attn_implementation = config.vit_attn_implementation
        self.vlm_config = vlm_config
        config.backbone_config = vlm_config.to_dict()
        if config.standalone_action_policy:
            self.visual = Qwen3_5VisionModel._from_config(vlm_config.vision_config)
        else:
            self.qwen35 = Qwen3_5ForConditionalGeneration._from_config(vlm_config)
            num_full = sum(kind == "full_attention" for kind in vlm_config.text_config.layer_types)
            if num_full == 0 or config.action_policy_num_layers % num_full:
                raise ValueError(
                    "The 18 action layers require an evenly dividing number of VLM full-attention layers."
                )

    @property
    def vision_model(self):
        return self.visual if self.config.standalone_action_policy else self.qwen35.model.visual

    @property
    def language_model(self):
        if self.config.standalone_action_policy:
            raise AttributeError("An independent action policy has no language model.")
        return self.qwen35.model.language_model

    def embed_images(self, pixel_values: Tensor, image_grid_thw: Tensor) -> Tensor:
        if pixel_values.ndim != 4:
            raise ValueError(f"Qwen image input must be preprocessed [B,N,256,1536], got {pixel_values.shape}.")
        pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        visual = self.vision_model
        num_images = image_grid_thw.shape[0]
        visual_input_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else visual.dtype
        output = visual(pixel_values.to(dtype=visual_input_dtype), grid_thw=image_grid_thw, return_dict=True)
        image_embeds = output.pooler_output
        if num_images <= 0 or image_embeds.shape[0] != num_images * 64:
            raise ValueError(
                "Every 256x256 camera image must produce exactly 64 Qwen3.5 "
                f"visual tokens; got {image_embeds.shape[0]} output tokens for "
                f"{num_images} images."
            )
        # VLA enforces one fixed 256x256 grid for every RGB camera.
        # Reshape directly instead of copying the CUDA grid to Python with
        # ``tolist()`` and splitting hundreds of same-sized chunks each step.
        return image_embeds.reshape(num_images, 64, -1)

    def embed_prefix(self, head_tokens, lang_tokens, lang_masks, subtask_tokens=None, subtask_masks=None):
        """Two head images, 72 task slots, and optionally 100 shifted target slots."""
        batch, _, patches, width = head_tokens.shape
        if patches != 64 or head_tokens.shape[1] != 2:
            raise ValueError("The VLM requires two head images with 64 visual tokens each.")
        if lang_tokens.shape != (batch, 72) or lang_masks.shape != (batch, 72):
            raise ValueError("The VLM task prompt must occupy exactly 72 slots.")
        embedding = self.language_model.embed_tokens
        device = head_tokens.device
        start_ids = torch.full((batch, 1), self.vlm_config.vision_start_token_id, device=device, dtype=torch.long)
        end_ids = torch.full((batch, 1), self.vlm_config.vision_end_token_id, device=device, dtype=torch.long)
        start, end = embedding(start_ids), embedding(end_ids)
        images = [torch.cat([start, head_tokens[:, index], end], dim=1) for index in range(2)]
        parts = [*images, embedding(lang_tokens)]
        masks = [torch.ones(batch, 132, dtype=torch.bool, device=device), lang_masks.bool()]
        prefix_length = 132 + 72
        if subtask_tokens is not None:
            if subtask_masks is None or subtask_tokens.shape != (batch, 100) or subtask_masks.shape != (batch, 100):
                raise ValueError("Subtask targets and masks must occupy exactly 100 slots.")
            # A regular language EOS/BOS starts the prediction region. This is
            # teacher forcing: target i is predicted from target <i, never itself.
            bos = self.vlm_config.text_config.bos_token_id
            if bos is None:
                bos = self.vlm_config.text_config.eos_token_id
            if isinstance(bos, (list, tuple)):
                bos = bos[0]
            if bos is None:
                raise ValueError("Qwen must define a BOS or EOS token for shifted language supervision.")
            shifted = torch.cat([torch.full_like(subtask_tokens[:, :1], bos), subtask_tokens[:, :-1]], dim=1)
            parts.append(embedding(shifted))
            masks.append(subtask_masks.bool())
        hidden = torch.cat(parts, dim=1)
        padding = torch.cat(masks, dim=1)
        positions = torch.zeros(3, batch, hidden.shape[1], device=device, dtype=torch.long)
        coordinates = torch.arange(8, device=device)
        for index in range(2):
            slot, offset = index * 66, index * 10
            positions[:, :, slot] = offset
            positions[0, :, slot + 1 : slot + 65] = offset + 1
            positions[1, :, slot + 1 : slot + 65] = coordinates[:, None].expand(-1, 8).reshape(-1) + offset + 1
            positions[2, :, slot + 1 : slot + 65] = coordinates[None, :].expand(8, -1).reshape(-1) + offset + 1
            positions[:, :, slot + 65] = offset + 9
        tail_positions = padding[:, 132:].long().cumsum(dim=-1) + 19
        tail_positions = tail_positions.masked_fill(~padding[:, 132:], 0)
        positions[:, :, 132:] = tail_positions[None]
        return hidden, padding, positions, prefix_length

    def encode_prefix(self, hidden, padding, positions, prefix_length):
        """One differentiable VLM pass; retain only image/task KV for MoT."""
        embeddings = self.language_model.rotary_emb(hidden, positions)
        length = hidden.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
        allowed = causal[None, None] & padding[:, None, None, :]
        prefix_kv = []
        for layer in self.language_model.layers:
            if qwen35_layer_type(layer) == "linear_attention":
                hidden = layer(hidden, mot_mode="linear", attention_mask=padding)
            else:
                query, key, value, gate = layer(hidden, mot_mode="project")
                query, key = apply_rotary_pos_emb(query, key, *embeddings)
                # Slice before exposing the cache. No target text, including its
                # padding or the shifted first token, reaches the action stream.
                prefix_kv.append((key[:, :, :prefix_length], value[:, :, :prefix_length]))
                attended = (
                    F.scaled_dot_product_attention(
                        query,
                        repeat_kv(key, layer.self_attn.num_key_value_groups),
                        repeat_kv(value, layer.self_attn.num_key_value_groups),
                        attn_mask=allowed,
                        dropout_p=0.0,
                        scale=layer.self_attn.scaling,
                    )
                    .transpose(1, 2)
                    .reshape(hidden.shape[0], length, -1)
                )
                hidden = layer(hidden, (attended, gate), mot_mode="finish")
        return self.language_model.norm(hidden), prefix_kv


class FlowMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = QwenBackbone(config)
        self.action_policy = ActionPolicy(config, self.backbone.vlm_config)
        if config.freeze_vision_encoder:
            self.backbone.vision_model.requires_grad_(False)
        if config.freeze_action_policy:
            self.action_policy.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.backbone.vision_model.eval()
        if self.config.freeze_action_policy:
            self.action_policy.eval()
        return self

    def encode_observations(
        self,
        images,
        img_masks,
        state,
        image_grid_thw,
        lang_tokens=None,
        lang_masks=None,
        subtask_tokens=None,
        subtask_masks=None,
    ):
        batch = state.shape[0]
        count = 2 if self.config.standalone_action_policy else 4
        if images.shape[:2] != (batch, count) or img_masks.shape != (batch, count):
            raise ValueError(f"Expected {count} image slots (history/current head then wrists for full policy).")
        if state.shape != (batch, self.config.max_state_dim):
            raise ValueError("State shape must match max_state_dim.")
        if image_grid_thw is None or image_grid_thw.numel() != batch * count * 3:
            raise ValueError("Each image needs its Qwen image_grid_thw.")
        encoded = self.backbone.embed_images(images, image_grid_thw.reshape(-1, 3))
        encoded = encoded.reshape(batch, count, 64, -1)
        environment = self.action_policy.encode_environment(encoded[:, -2:], state, img_masks[:, -2:])
        zero = environment.new_zeros((), dtype=torch.float32)
        if self.config.standalone_action_policy:
            return environment, None, None, None, zero
        if not img_masks[:, :2].bool().all():
            raise ValueError("Both historical and current head frames must be valid.")
        if lang_tokens is None or lang_masks is None:
            raise ValueError("The full model requires task tokens and task masks.")
        hidden, padding, positions, prefix_length = self.backbone.embed_prefix(
            encoded[:, :2],
            lang_tokens,
            lang_masks,
            subtask_tokens,
            subtask_masks,
        )
        outputs, cache = self.backbone.encode_prefix(hidden, padding, positions, prefix_length)
        language_loss = zero
        if subtask_tokens is not None:
            # Qwen's vocabulary is large. Select supervised tokens first, then
            # checkpoint bounded LM-head/CE chunks so their vocab-sized logits
            # are not all retained for backward. The VLM itself still runs once.
            selected = subtask_masks.bool()
            supervised_hidden = outputs[:, prefix_length:][selected]
            labels = subtask_tokens[selected]
            target_count = labels.numel()
            if target_count == 0:
                # Keep an LM-head graph on unannotated distributed batches,
                # while ignore_index yields exactly zero loss and gradients.
                supervised_hidden = outputs[:1, prefix_length]
                labels = torch.full((1,), -100, dtype=torch.long, device=outputs.device)
            language_loss = outputs.sum() * 0.0

            def chunk_loss(hidden_chunk, label_chunk):
                logits = self.backbone.qwen35.lm_head(hidden_chunk)
                return F.cross_entropy(logits.float(), label_chunk, reduction="sum", ignore_index=-100)

            for start in range(0, labels.numel(), 128):
                args = (supervised_hidden[start : start + 128], labels[start : start + 128])
                if self.training and torch.is_grad_enabled():
                    language_loss = language_loss + checkpoint(chunk_loss, *args, use_reentrant=False)
                else:
                    language_loss = language_loss + chunk_loss(*args)
            language_loss = language_loss / max(target_count, 1)
        position_start = positions[:, :, :prefix_length].amax(dim=(0, 2)) + 1
        return environment, cache, padding[:, :prefix_length], position_start, language_loss

    def forward(
        self,
        images,
        img_masks,
        state,
        actions,
        lang_tokens=None,
        lang_masks=None,
        subtask_tokens=None,
        subtask_masks=None,
        image_grid_thw=None,
        noise=None,
        time=None,
        action_valid_mask=None,
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
        delays = torch.zeros(actions.shape[0], device=actions.device, dtype=torch.long)
        if self.config.rtc_training_enabled:
            valid_prefix = action_valid_mask.long().cumprod(dim=1).sum(dim=1).sub(1).clamp(min=0)
            delays = sample_rtc_delays(valid_prefix, self.config.rtc_max_delay_steps)
        clean_prefix = torch.arange(actions.shape[1], device=actions.device)[None] < delays[:, None]
        action_time = torch.where(clean_prefix, 0.0, time[:, None]).to(actions.dtype)
        noisy = action_time[..., None] * noise + (1 - action_time[..., None]) * actions
        if self.training and self.config.use_subtask_loss:
            if subtask_tokens is None or subtask_masks is None:
                raise ValueError("This training stage requires current-frame subtask targets.")
        else:
            subtask_tokens = subtask_masks = None
        environment, cache, prefix_mask, position_start, language_loss = self.encode_observations(
            images,
            img_masks,
            state,
            image_grid_thw,
            lang_tokens,
            lang_masks,
            subtask_tokens,
            subtask_masks,
        )
        velocity = self.action_policy(
            noisy, action_time, environment, cache, prefix_mask, position_start, action_valid_mask
        )
        target = noise - actions
        losses = (
            F.mse_loss(velocity, target, reduction="none")
            if self.config.loss_type == "fm"
            else F.l1_loss(velocity, target, reduction="none")
        )
        metrics = {
            "rtc/delay_mean": delays.float().mean().detach(),
            "rtc/prefix_fraction": clean_prefix.float().mean().detach(),
        }
        return losses, language_loss, metrics, ~clean_prefix

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        state,
        lang_tokens=None,
        lang_masks=None,
        noise=None,
        image_grid_thw=None,
        action_prefix=None,
        prefix_lengths=None,
    ):
        batch = state.shape[0]
        expected = (batch, self.config.chunk_size, self.config.max_action_dim)
        if noise is None:
            noise = torch.randn(expected, device=state.device, dtype=state.dtype)
        elif noise.shape != expected:
            raise ValueError(f"noise must have shape {expected}.")
        if (action_prefix is None) != (prefix_lengths is None):
            raise ValueError("action_prefix and prefix_lengths must be supplied together.")
        clean_prefix = torch.zeros(expected[:2], device=state.device, dtype=torch.bool)
        if action_prefix is not None:
            if not self.config.rtc_training_enabled:
                raise ValueError("Prefix-conditioned inference requires an RTC-trained checkpoint.")
            if action_prefix.shape != expected or prefix_lengths.shape != (batch,):
                raise ValueError("Invalid action_prefix or prefix_lengths shape.")
            prefix_lengths = prefix_lengths.to(device=state.device, dtype=torch.long)
            if ((prefix_lengths < 0) | (prefix_lengths > self.config.rtc_max_delay_steps)).any():
                raise ValueError("Requested delay exceeds the trained RTC range.")
            action_prefix = action_prefix.to(device=state.device, dtype=state.dtype)
            clean_prefix = torch.arange(self.config.chunk_size, device=state.device)[None] < prefix_lengths[:, None]
        # Vision/environment and VLM each run once, with no subtask region.
        environment, cache, prefix_mask, position_start, _ = self.encode_observations(
            images,
            img_masks,
            state,
            image_grid_thw,
            lang_tokens,
            lang_masks,
        )
        current = torch.where(clean_prefix[..., None], action_prefix, noise) if action_prefix is not None else noise
        for index in range(self.config.num_steps):
            time = torch.full(
                expected[:2], 1.0 - index / self.config.num_steps, device=state.device, dtype=state.dtype
            )
            time = time.masked_fill(clean_prefix, 0)
            velocity = self.action_policy(current, time, environment, cache, prefix_mask, position_start)
            current = current - velocity / self.config.num_steps
            if action_prefix is not None:
                current = torch.where(clean_prefix[..., None], action_prefix, current)
        return current


class VLAModel(PreTrainedModel):
    config_class = VLAConfig
    name = "torch_vla"
    main_input_name = "images"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules = ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock", "ActionPolicyBlock"]

    @classmethod
    def get_weight_loader(cls):
        return PolicyWeightLoader()

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
        elif isinstance(module, Qwen3_5RMSNorm):
            nn.init.zeros_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            module._reset_parameters()
        if isinstance(module, ActionPolicyBlock):
            nn.init.normal_(module.modulation, std=module.modulation.shape[-1] ** -0.5)
        if isinstance(module, ActionPolicy):
            nn.init.normal_(module.output_modulation, std=module.output_modulation.shape[-1] ** -0.5)

    def get_optim_params(self):
        return self.parameters()

    def load_additional_pretrained_weights(self):
        if self.config.action_policy_path:
            return self.load_action_policy(self.config.action_policy_path)
        return {}

    def load_action_policy(self, path):
        """Load an action_policy/ export together with its required visual tower."""
        from xense0vla.models.module_utils import _load_state_dict

        source = Path(path)
        vision_source = source / "vision_encoder"
        if not vision_source.is_dir():
            raise FileNotFoundError(f"Expected action_policy/ with vision_encoder/ at {source}")
        state, vision_state = {}, {}
        for iterator in _load_state_dict(str(source)):
            state.update(dict(iterator))
        for iterator in _load_state_dict(str(vision_source)):
            vision_state.update(dict(iterator))
        self.model.backbone.vision_model.load_state_dict(vision_state, strict=True)
        self.model.action_policy.load_state_dict(state, strict=True)
        return {
            "action_policy": {"source": str(source), "loaded_tensors": len(state), "vision_tensors": len(vision_state)}
        }

    def save_action_policy(self, path, state_dict=None):
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = self.model.action_policy.state_dict() if state_dict is None else state_dict
        if any(key.startswith("model.action_policy.") for key in state):
            state = {
                key.removeprefix("model.action_policy."): value
                for key, value in state.items()
                if key.startswith("model.action_policy.")
            }
        save_file(
            {
                key: value.detach()
                .to(device="cpu", dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
                .contiguous()
                for key, value in state.items()
            },
            str(path / "model.safetensors"),
        )
        # Preserve the actual visual tower even when this run freezes weights
        # that were fine-tuned by an earlier run.
        vision_dir = path / "vision_encoder"
        vision_dir.mkdir(exist_ok=True)
        vision_state = self.model.backbone.vision_model.state_dict()
        if state_dict is not None:
            for prefix in ("model.backbone.visual.", "model.backbone.qwen35.model.visual."):
                exported = {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}
                if exported:
                    vision_state = exported
                    break
        save_file(
            {
                key: value.detach()
                .to(device="cpu", dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
                .contiguous()
                for key, value in vision_state.items()
            },
            str(vision_dir / "model.safetensors"),
        )
        self.config.save_pretrained(path)

    def forward(
        self,
        images,
        img_masks,
        state,
        actions,
        lang_tokens=None,
        lang_masks=None,
        subtask_tokens=None,
        subtask_masks=None,
        joint_mask=None,
        noise=None,
        time=None,
        image_grid_thw=None,
        **kwargs,
    ):
        if joint_mask is not None and joint_mask.shape != actions.shape:
            raise ValueError("joint_mask must match the action tensor.")
        losses, language_loss, metrics, postfix = self.model(
            images,
            img_masks,
            state,
            actions,
            lang_tokens,
            lang_masks,
            subtask_tokens,
            subtask_masks,
            image_grid_thw,
            noise,
            time,
            joint_mask.any(dim=-1) if joint_mask is not None else None,
        )
        if joint_mask is None:
            joint_mask = torch.ones_like(losses, dtype=torch.bool)
            joint_mask[:, :, self.config.action_dim :] = False
        effective = joint_mask.bool() & postfix[..., None]
        masked = losses * effective
        action_loss = masked.sum() / effective.sum().clamp(min=1)
        batch_losses = masked.sum(dim=(1, 2)) / effective.sum(dim=(1, 2)).clamp(min=1)
        total = self.config.action_loss_weight * action_loss + self.config.language_loss_weight * language_loss
        return total, action_loss, language_loss, {"batch_mean_losses": batch_losses.detach(), **metrics}, {}

    def sample_actions(self, *args, **kwargs):
        return self.model.sample_actions(*args, **kwargs)


__all__ = ["VLAModel", "FlowMatching", "QwenBackbone", "ActionPolicy"]
