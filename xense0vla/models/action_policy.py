"""Independent wrist/state-conditioned flow policy with optional MoT prefix KV."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5MLP,
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .policy_utils import create_sinusoidal_pos_embedding


class EnvironmentEncoder(nn.Module):
    """Fuse aligned left/right patches, then attend to the robot state."""

    def __init__(self, visual_width, config):
        super().__init__()
        width = config.action_expert_hidden_size
        fused_width = 2 * width
        self.wrist_projection = nn.Linear(visual_width, width)
        self.state_projection = nn.Linear(config.max_state_dim, fused_width)
        self.image_norm = nn.LayerNorm(fused_width)
        self.self_attention = nn.MultiheadAttention(fused_width, config.environment_num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(fused_width)
        self.state_norm = nn.LayerNorm(fused_width)
        self.cross_attention = nn.MultiheadAttention(fused_width, config.environment_num_heads, batch_first=True)
        self.output_projection = nn.Linear(fused_width, width)

    def forward(self, wrist_tokens: Tensor, state: Tensor, wrist_mask: Tensor) -> Tensor:
        if wrist_tokens.ndim != 4 or wrist_tokens.shape[1] != 2:
            raise ValueError("wrist_tokens must be [batch, left/right, patches, visual_width].")
        if wrist_mask.shape != wrist_tokens.shape[:2]:
            raise ValueError("wrist_mask must be [batch, 2].")
        if not wrist_mask.bool().all():
            raise ValueError("Both wrist cameras are required by the independent action policy.")
        tokens = self.wrist_projection(wrist_tokens)
        # This is feature concatenation, preserving the action vector's left/right order.
        tokens = torch.cat([tokens[:, 0], tokens[:, 1]], dim=-1)
        normalized = self.image_norm(tokens)
        tokens = tokens + self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        state_tokens = self.state_norm(self.state_projection(state.to(tokens.dtype)))[:, None]
        tokens = (
            tokens + self.cross_attention(self.cross_norm(tokens), state_tokens, state_tokens, need_weights=False)[0]
        )
        return self.output_projection(tokens)


class ActionPolicyBlock(nn.Module):
    """Full attention with Wan's shared time projection plus a layer offset table.

    The six modulation channels are attention shift/scale/gate and MLP
    shift/scale/gate, matching Wan2.2's wan/modules/model.py.
    """

    def __init__(self, text_config, layer_idx):
        super().__init__()
        width = text_config.hidden_size
        self.self_attn = Qwen3_5Attention(text_config, layer_idx)
        self.mlp = Qwen3_5MLP(text_config, text_config.intermediate_size)
        self.input_layernorm = nn.LayerNorm(width, elementwise_affine=False, eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(width, elementwise_affine=False, eps=text_config.rms_norm_eps)
        self.modulation = nn.Parameter(torch.randn(1, 6, width) / width**0.5)
        self.layer_type = "full_attention"

    def forward(self, hidden, time_modulation, position_embeddings, allowed_mask, prefix_kv=None):
        modulation = time_modulation.float() + self.modulation.float().unsqueeze(1)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = modulation.unbind(dim=2)
        normalized = (self.input_layernorm(hidden).float() * (1 + scale_attn) + shift_attn).to(hidden.dtype)
        attn = self.self_attn
        batch, length = hidden.shape[:2]
        q_gate = attn.q_proj(normalized).view(batch, length, -1, attn.head_dim * 2)
        query, output_gate = q_gate.chunk(2, dim=-1)
        query = attn.q_norm(query).transpose(1, 2)
        key = attn.k_norm(attn.k_proj(normalized).view(batch, length, -1, attn.head_dim)).transpose(1, 2)
        value = attn.v_proj(normalized).view(batch, length, -1, attn.head_dim).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        if prefix_kv is not None:
            prefix_key, prefix_value = prefix_kv
            key = torch.cat([prefix_key.to(key.dtype), key], dim=2)
            value = torch.cat([prefix_value.to(value.dtype), value], dim=2)
        attended = (
            F.scaled_dot_product_attention(
                query,
                repeat_kv(key, attn.num_key_value_groups),
                repeat_kv(value, attn.num_key_value_groups),
                attn_mask=allowed_mask[:, None],
                dropout_p=0.0,
                scale=attn.scaling,
            )
            .transpose(1, 2)
            .reshape(batch, length, -1)
        )
        attended = attended * torch.sigmoid(output_gate.reshape(batch, length, -1))
        hidden = (hidden.float() + gate_attn * attn.o_proj(attended).float()).to(hidden.dtype)
        normalized = (self.post_attention_layernorm(hidden).float() * (1 + scale_mlp) + shift_mlp).to(hidden.dtype)
        return (hidden.float() + gate_mlp * self.mlp(normalized).float()).to(hidden.dtype)


class ActionPolicy(nn.Module):
    """Complete transferable policy, including all observation and action projections."""

    def __init__(self, config, backbone_config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        text_config = deepcopy(backbone_config.text_config)
        text_config.hidden_size = config.action_expert_hidden_size
        text_config.intermediate_size = config.action_expert_intermediate_size
        text_config.num_hidden_layers = config.action_policy_num_layers
        text_config.layer_types = ["full_attention"] * config.action_policy_num_layers
        self.text_config = text_config
        width = text_config.hidden_size
        self.environment = EnvironmentEncoder(backbone_config.vision_config.out_hidden_size, config)
        self.action_in_projection = nn.Linear(config.max_action_dim, width)
        self.time_embedding = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        self.layers = nn.ModuleList(
            ActionPolicyBlock(text_config, index) for index in range(config.action_policy_num_layers)
        )
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(text_config)
        self.output_norm = nn.LayerNorm(width, elementwise_affine=False, eps=text_config.rms_norm_eps)
        self.output_modulation = nn.Parameter(torch.randn(1, 2, width) / width**0.5)
        self.action_out_projection = nn.Linear(width, config.max_action_dim)

    def encode_environment(self, wrist_tokens, state, wrist_mask):
        return self.environment(wrist_tokens, state, wrist_mask)

    def forward(
        self,
        noisy_actions,
        timestep,
        environment,
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
        hidden = torch.cat([environment.to(action_hidden.dtype), action_hidden], dim=1)
        env_length = environment.shape[1]
        # Environment tokens are observations, hence always at clean flow time.
        all_times = torch.cat([timestep.new_zeros(batch, env_length), timestep], dim=1)
        sinusoidal = create_sinusoidal_pos_embedding(all_times, hidden.shape[-1], 4e-3, 4.0, hidden.device)
        time_embedding = self.time_embedding(sinusoidal.to(hidden.dtype))
        time_modulation = self.time_projection(time_embedding).unflatten(-1, (6, hidden.shape[-1]))
        if action_valid_mask is None:
            action_valid_mask = torch.ones(batch, horizon, dtype=torch.bool, device=hidden.device)
        padding = torch.cat(
            [
                torch.ones(batch, env_length, dtype=torch.bool, device=hidden.device),
                action_valid_mask.bool(),
            ],
            dim=1,
        )
        allowed = padding[:, None, :].expand(-1, hidden.shape[1], -1)
        # Keep environment features independent of action noise, even in deeper layers.
        allowed = allowed.clone()
        allowed[:, :env_length, env_length:] = False
        if prefix_kv:
            if len(self.layers) % len(prefix_kv):
                raise ValueError("Action layers must divide evenly over the VLM full-attention KV entries.")
            if prefix_mask is None:
                raise ValueError("MoT prefix KV requires a padding mask.")
            allowed = torch.cat([prefix_mask[:, None, :].expand(-1, hidden.shape[1], -1), allowed], dim=-1)
        if position_start is None:
            position_start = torch.zeros(batch, dtype=torch.long, device=hidden.device)
        positions = position_start[:, None] + torch.arange(hidden.shape[1], device=hidden.device)[None]
        embeddings = self.rotary_emb(hidden, positions)
        for index, layer in enumerate(self.layers):
            cache = prefix_kv[index * len(prefix_kv) // len(self.layers)] if prefix_kv else None
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(layer, hidden, time_modulation, embeddings, allowed, cache, use_reentrant=False)
            else:
                hidden = layer(hidden, time_modulation, embeddings, allowed, cache)
        output_condition = time_embedding.float().unsqueeze(2) + self.output_modulation.float().unsqueeze(1)
        shift, scale = output_condition.unbind(dim=2)
        hidden = (self.output_norm(hidden).float() * (1 + scale) + shift).to(hidden.dtype)
        return self.action_out_projection(hidden[:, -horizon:])


__all__ = ["ActionPolicy", "ActionPolicyBlock", "EnvironmentEncoder"]
