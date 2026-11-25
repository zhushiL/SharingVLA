"""Dual-wrist action policy with a spatial_encoder and optional language conditioning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from depth_anything_3.model.dinov2.dinov2 import DinoV2
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .configuration_vla import ActionPolicyConfig
from .policy_utils import create_sinusoidal_pos_embedding


class SpatialVisionEncoder(DinoV2):
    """Multi-view Transformer backbone of the spatial_encoder.

    Action policy pretraining initializes it from DA3-LARGE weights. The final
    block attends across both wrists and returns patch features for each view.
    """

    def __init__(self):
        super().__init__(name="vitl", out_layers=[23], alt_start=8, qknorm_start=8, rope_start=8, cat_token=False)
        self.gradient_checkpointing = False
        self._block_checkpointing_installed = False
        self._checkpointed_block_count = 0
        self._install_block_checkpointing()

    def _resolve_transformer_blocks(self) -> list[nn.Module]:
        candidates = (
            getattr(self, "blocks", None),
            getattr(getattr(self, "pretrained", None), "blocks", None),
        )
        for container in candidates:
            if not isinstance(container, nn.ModuleList) or len(container) == 0:
                continue
            blocks: list[nn.Module] = []
            for item in container:
                if isinstance(item, nn.Sequential):
                    blocks.extend(
                        child for child in item if not isinstance(child, nn.Identity)
                    )
                else:
                    blocks.append(item)
            if blocks:
                return blocks
        return []

    def _install_block_checkpointing(self) -> None:
        blocks = self._resolve_transformer_blocks()
        if not blocks:
            return
        encoder = self

        def _make_wrapped_forward(orig_forward):
            def _wrapped_forward(*args, **kwargs):
                if encoder.gradient_checkpointing and encoder.training and torch.is_grad_enabled():
                    return checkpoint(orig_forward, *args, use_reentrant=False, **kwargs)
                return orig_forward(*args, **kwargs)

            return _wrapped_forward

        for block in blocks:
            block.forward = _make_wrapped_forward(block.forward)
        self._checkpointed_block_count = len(blocks)
        self._block_checkpointing_installed = True

    def forward(self, wrists: Tensor) -> Tensor:
        if wrists.ndim != 5 or wrists.shape[1:] != (2, 3, 224, 224):
            raise ValueError("Expected normalized left/right wrist RGB images of shape [B,2,3,224,224].")
        layers, _ = super().forward(wrists)
        patches, _camera_tokens = layers[0]
        return patches


def apply_rotary(value, cos, sin):
    width = cos.shape[-1]
    rotary, rest = value.split([width, value.shape[-1] - width], dim=-1)
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat([-second, first], dim=-1)
    return torch.cat([rotary * cos + rotated * sin, rest], dim=-1)


class ActionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.groups = config.num_attention_heads // config.num_key_value_heads
        query_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, 2 * query_width, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_width, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_width, bias=False)
        self.o_proj = nn.Linear(query_width, config.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden, embeddings, mask, prefix, *, return_kv: bool = False):
        batch, length = hidden.shape[:2]
        query, gate = self.q_proj(hidden).view(batch, length, -1, 2 * self.head_dim).chunk(2, dim=-1)
        key = self.k_proj(hidden).view(batch, length, -1, self.head_dim)
        value = self.v_proj(hidden).view_as(key).transpose(1, 2)
        query = apply_rotary(self.q_norm(query).to(value.dtype).transpose(1, 2), *embeddings)
        key = apply_rotary(self.k_norm(key).to(value.dtype).transpose(1, 2), *embeddings)
        self_kv = (key, value) if return_kv else None
        if prefix is not None:
            key = torch.cat([prefix[0].to(key.dtype), key], dim=2)
            value = torch.cat([prefix[1].to(value.dtype), value], dim=2)
        attn_mask = None if mask is None else (mask[:, None] if mask.ndim == 3 else mask)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            enable_gqa=True,
        ).transpose(1, 2).reshape(batch, length, -1)
        output = self.o_proj(attended * torch.sigmoid(gate.reshape_as(attended)))
        if return_kv:
            return output, self_kv
        return output


class ActionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, value):
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


class ActionPolicyBlock(nn.Module):
    """Full attention with Wan's shared time projection plus a layer offset table.

    The six modulation channels are attention shift/scale/gate and MLP
    shift/scale/gate, matching Wan2.2's wan/modules/model.py.
    """

    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.self_attn = ActionAttention(config)
        self.mlp = ActionMLP(config)
        self.input_layernorm = nn.LayerNorm(width, elementwise_affine=False, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(width, elementwise_affine=False, eps=config.rms_norm_eps)
        self.modulation = nn.Parameter(torch.randn(1, 6, width) / width**0.5)

    def forward(self, hidden, time_modulation, position_embeddings, allowed_mask, prefix_kv=None, *, return_kv: bool = False):
        modulation = time_modulation.float() + self.modulation.float().unsqueeze(1)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = modulation.unbind(dim=2)
        normalized = (self.input_layernorm(hidden).float() * (1 + scale_attn) + shift_attn).to(hidden.dtype)
        attn_out = self.self_attn(normalized, position_embeddings, allowed_mask, prefix_kv, return_kv=return_kv)
        if return_kv:
            attended, self_kv = attn_out
        else:
            attended, self_kv = attn_out, None
        hidden = (hidden.float() + gate_attn * attended.float()).to(hidden.dtype)
        normalized = (self.post_attention_layernorm(hidden).float() * (1 + scale_mlp) + shift_mlp).to(hidden.dtype)
        hidden = (hidden.float() + gate_mlp * self.mlp(normalized).float()).to(hidden.dtype)
        if return_kv:
            return hidden, self_kv
        return hidden


class ActionPolicy(nn.Module):
    """Complete transferable policy, including all observation and action projections."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        arch = ActionPolicyConfig(**config.action_config)
        self.arch = arch
        width = arch.hidden_size
        self.spatial_encoder = SpatialVisionEncoder()
        # Downsample the paired wrist features before projecting to the action width.
        self.spatial_projection = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 2048),
            nn.SiLU(),
            nn.Conv2d(2048, width, kernel_size=1),
        )
        self.action_in_projection = nn.Linear(config.max_action_dim, width)
        self.time_embedding = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        self.layers = nn.ModuleList(ActionPolicyBlock(arch) for _ in range(arch.num_hidden_layers))
        self.output_norm = nn.LayerNorm(width, elementwise_affine=False, eps=arch.rms_norm_eps)
        self.output_modulation = nn.Parameter(torch.randn(1, 2, width) / width**0.5)
        self.action_out_projection = nn.Linear(width, config.max_action_dim)

    def encode_spatial_tokens(self, wrist_images):
        """Run the full spatial_encoder, from the two wrist views to 64 spatial tokens."""
        spatial_trainable = any(param.requires_grad for param in self.spatial_encoder.parameters())
        if not spatial_trainable:
            with torch.no_grad():
                patches = self.spatial_encoder(wrist_images)
        elif (
            self.gradient_checkpointing
            and self.spatial_encoder.training
            and not getattr(self.spatial_encoder, "_block_checkpointing_installed", False)
        ):
            patches = checkpoint(self.spatial_encoder, wrist_images, use_reentrant=False)
        else:
            prev_ckpt = getattr(self.spatial_encoder, "gradient_checkpointing", False)
            if self.gradient_checkpointing and not prev_ckpt:
                self.spatial_encoder.gradient_checkpointing = True
            try:
                patches = self.spatial_encoder(wrist_images)
            finally:
                if self.gradient_checkpointing and not prev_ckpt:
                    self.spatial_encoder.gradient_checkpointing = prev_ckpt
        # [B, view, patch, channel] -> [B, left_channels + right_channels, H, W]
        paired = patches.permute(0, 1, 3, 2).reshape(patches.shape[0], 2048, 16, 16)
        return self.spatial_projection(paired).flatten(2).transpose(1, 2)

    def rotary_embeddings(self, positions, dtype):
        # Compute in FP32 even when parameters are BF16.
        indices = torch.arange(0, self.arch.rotary_dim, 2, device=positions.device, dtype=torch.float32)
        frequency = self.arch.rope_theta ** (-indices / self.arch.rotary_dim)
        angles = positions.float()[:, None, :, None] * frequency
        angles = torch.cat([angles, angles], dim=-1)
        return angles.cos().to(dtype), angles.sin().to(dtype)

    @torch.no_grad()
    def prefill_spatial_tokens(
        self,
        spatial_tokens,
        prefix_kv=None,
        prefix_mask=None,
        position_start=None,
    ):
        """Precompute invariant 18-layer KV caches for the 64 spatial tokens once per ODE solve."""
        dtype = self.action_in_projection.weight.dtype
        spatial_hidden = spatial_tokens.to(dtype)
        batch, spatial_length, width = spatial_hidden.shape
        device = spatial_hidden.device
        horizon = self.config.chunk_size

        spatial_time = torch.zeros((1, 1), device=device, dtype=torch.float32)
        spatial_sinusoidal = create_sinusoidal_pos_embedding(spatial_time, width, 4e-3, 4.0, device)
        spatial_time_embedding = self.time_embedding(spatial_sinusoidal.to(dtype))
        spatial_time_modulation = self.time_projection(spatial_time_embedding).unflatten(-1, (6, width))

        if position_start is None:
            position_start = torch.zeros(batch, dtype=torch.long, device=device)
        all_positions = position_start[:, None] + torch.arange(spatial_length + horizon, device=device)[None]
        spatial_cos, spatial_sin = self.rotary_embeddings(all_positions[:, :spatial_length], dtype)
        action_embeddings = self.rotary_embeddings(all_positions[:, spatial_length:], dtype)

        spatial_mot_mask = None
        action_mot_mask = None
        if prefix_kv is not None:
            if len(prefix_kv) != len(self.layers) // 3:
                raise ValueError("Each third action layer requires its own VLM KV pair.")
            if prefix_mask is None:
                raise ValueError("MoT prefix KV requires a padding mask.")
            prefix_bool = prefix_mask.bool()
            spatial_valid = torch.ones((batch, spatial_length), dtype=torch.bool, device=device)
            action_valid = torch.ones((batch, horizon), dtype=torch.bool, device=device)
            spatial_mot_mask = torch.cat([prefix_bool, spatial_valid], dim=1)[:, None, None, :]
            action_mot_mask = torch.cat([prefix_bool, spatial_valid, action_valid], dim=1)[:, None, None, :]

        cached_kv = []
        for index, layer in enumerate(self.layers):
            vlm_cache = prefix_kv[index // 3] if prefix_kv is not None and index % 3 == 2 else None
            mask = spatial_mot_mask if vlm_cache is not None else None
            spatial_hidden, (spatial_k, spatial_v) = layer(
                spatial_hidden,
                spatial_time_modulation,
                (spatial_cos, spatial_sin),
                mask,
                vlm_cache,
                return_kv=True,
            )
            if vlm_cache is not None:
                cached_k = torch.cat([vlm_cache[0].to(spatial_k.dtype), spatial_k], dim=2)
                cached_v = torch.cat([vlm_cache[1].to(spatial_v.dtype), spatial_v], dim=2)
                cached_kv.append((cached_k, cached_v))
            else:
                cached_kv.append((spatial_k, spatial_v))
        return tuple(cached_kv), action_embeddings, action_mot_mask

    def denoise_step(
        self,
        noisy_actions,
        timestep,
        cached_kv,
        action_embeddings,
        action_mot_mask=None,
    ):
        """Run one flow-matching denoising step over only the action tokens using cached spatial/VLM KV."""
        batch, horizon, _ = noisy_actions.shape
        if horizon != self.config.chunk_size or noisy_actions.shape[-1] != self.config.max_action_dim:
            raise ValueError("Action tensor must match the configured chunk_size and max_action_dim.")
        if timestep.ndim == 1:
            timestep = timestep[:, None].expand(-1, horizon)
        if timestep.shape != noisy_actions.shape[:2]:
            raise ValueError("timestep must have shape [batch] or [batch, horizon].")
        hidden = self.action_in_projection(noisy_actions)
        sinusoidal = create_sinusoidal_pos_embedding(timestep, hidden.shape[-1], 4e-3, 4.0, hidden.device)
        time_embedding = self.time_embedding(sinusoidal.to(hidden.dtype))
        time_modulation = self.time_projection(time_embedding).unflatten(-1, (6, hidden.shape[-1]))
        for index, layer in enumerate(self.layers):
            mask = action_mot_mask if (action_mot_mask is not None and index % 3 == 2) else None
            hidden = layer(hidden, time_modulation, action_embeddings, mask, cached_kv[index])
        output_condition = time_embedding.float().unsqueeze(2) + self.output_modulation.float().unsqueeze(1)
        shift, scale = output_condition.unbind(dim=2)
        hidden = (self.output_norm(hidden).float() * (1 + scale) + shift).to(hidden.dtype)
        return self.action_out_projection(hidden)

    def forward(
        self,
        noisy_actions,
        timestep,
        spatial_tokens,
        prefix_kv=None,
        prefix_mask=None,
        position_start=None,
        action_valid_mask=None,
    ):
        batch, horizon, _ = noisy_actions.shape
        if horizon != self.config.chunk_size or noisy_actions.shape[-1] != self.config.max_action_dim:
            raise ValueError("Action tensor must match the configured chunk_size and max_action_dim.")
        if timestep.ndim == 1:
            timestep = timestep[:, None].expand(-1, horizon)
        if timestep.shape != noisy_actions.shape[:2]:
            raise ValueError("timestep must have shape [batch] or [batch, horizon].")
        action_hidden = self.action_in_projection(noisy_actions)
        hidden = torch.cat([spatial_tokens.to(action_hidden.dtype), action_hidden], dim=1)
        spatial_length = spatial_tokens.shape[1]
        # Spatial tokens are observations, hence always at clean flow time.
        all_times = torch.cat([timestep.new_zeros((batch, spatial_length), dtype=torch.float32), timestep.float()], dim=1)
        sinusoidal = create_sinusoidal_pos_embedding(all_times, hidden.shape[-1], 4e-3, 4.0, hidden.device)
        time_embedding = self.time_embedding(sinusoidal.to(hidden.dtype))
        time_modulation = self.time_projection(time_embedding).unflatten(-1, (6, hidden.shape[-1]))
        if action_valid_mask is None:
            action_valid_mask = torch.ones(batch, horizon, dtype=torch.bool, device=hidden.device)
        padding = torch.cat(
            [
                torch.ones(batch, spatial_length, dtype=torch.bool, device=hidden.device),
                action_valid_mask.bool(),
            ],
            dim=1,
        )
        allowed = padding[:, None, :].expand(-1, hidden.shape[1], -1)
        # Keep spatial features independent of action noise, even in deeper layers.
        allowed = allowed.clone()
        allowed[:, :spatial_length, spatial_length:] = False
        joint_mask = None
        if prefix_kv is not None:
            if len(prefix_kv) != len(self.layers) // 3:
                raise ValueError("Each third action layer requires its own VLM KV pair.")
            if prefix_mask is None:
                raise ValueError("MoT prefix KV requires a padding mask.")
            joint_mask = torch.cat([prefix_mask[:, None, :].expand(-1, hidden.shape[1], -1), allowed], dim=-1)
        if position_start is None:
            position_start = torch.zeros(batch, dtype=torch.long, device=hidden.device)
        positions = position_start[:, None] + torch.arange(hidden.shape[1], device=hidden.device)[None]
        embeddings = self.rotary_embeddings(positions, hidden.dtype)
        for index, layer in enumerate(self.layers):
            cache = prefix_kv[index // 3] if prefix_kv is not None and index % 3 == 2 else None
            mask = joint_mask if cache is not None else allowed
            if self.gradient_checkpointing and torch.is_grad_enabled():
                hidden = checkpoint(layer, hidden, time_modulation, embeddings, mask, cache, use_reentrant=False)
            else:
                hidden = layer(hidden, time_modulation, embeddings, mask, cache)
        action_out = hidden[:, -horizon:]
        action_time_embedding = time_embedding[:, -horizon:]
        output_condition = action_time_embedding.float().unsqueeze(2) + self.output_modulation.float().unsqueeze(1)
        shift, scale = output_condition.unbind(dim=2)
        action_out = (self.output_norm(action_out).float() * (1 + scale) + shift).to(action_out.dtype)
        return self.action_out_projection(action_out)


__all__ = ["ActionPolicy", "ActionPolicyBlock", "SpatialVisionEncoder"]
