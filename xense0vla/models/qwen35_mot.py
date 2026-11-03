"""FSDP-safe Qwen3.5 projections for differentiable MoT prefix KV."""

from __future__ import annotations

from types import MethodType
from typing import Optional

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    torch_causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)


def qwen35_layer_type(layer):
    value = getattr(layer, "layer_type", getattr(layer, "block_type", None))
    if value not in {"linear_attention", "full_attention"}:
        raise ValueError(f"Unsupported Qwen3.5 layer type: {value!r}")
    return value


def _mot_decoder_forward(self, hidden_states, position_embeddings=None, *, mot_mode=None, **kwargs):
    if mot_mode is None:
        return type(self)._vla_original_forward(
            self, hidden_states, position_embeddings=position_embeddings, **kwargs
        )
    if mot_mode == "linear":
        if qwen35_layer_type(self) != "linear_attention":
            raise ValueError("linear MoT operation requires a linear-attention layer.")
        hidden_states = hidden_states + self.linear_attn(
            hidden_states=self.input_layernorm(hidden_states),
            cache_params=None,
            attention_mask=kwargs.get("attention_mask"),
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
    if qwen35_layer_type(self) != "full_attention":
        raise ValueError("MoT projection/finish requires full attention.")
    if mot_mode == "project":
        normalized = self.input_layernorm(hidden_states)
        shape = normalized.shape[:-1]
        dim = self.self_attn.head_dim
        query, gate = self.self_attn.q_proj(normalized).view(*shape, -1, dim * 2).chunk(2, dim=-1)
        query = self.self_attn.q_norm(query).transpose(1, 2)
        key = self.self_attn.k_norm(self.self_attn.k_proj(normalized).view(*shape, -1, dim)).transpose(1, 2)
        value = self.self_attn.v_proj(normalized).view(*shape, -1, dim).transpose(1, 2)
        return query, key, value, gate.reshape(*shape, -1)
    if mot_mode == "finish":
        attention, gate = position_embeddings
        hidden_states = hidden_states + self.self_attn.o_proj(attention * torch.sigmoid(gate))
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
    raise ValueError(f"Unknown MoT operation {mot_mode!r}.")


def _torch_rms_norm_gated_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
    **_kwargs,
) -> torch.Tensor:
    """Device-independent equivalent of Qwen3.5's gated RMSNorm."""

    input_dtype = hidden_states.dtype
    normalized = hidden_states.float()
    variance = normalized.square().mean(dim=-1, keepdim=True)
    epsilon = getattr(self, "eps", getattr(self, "variance_epsilon", 1e-6))
    normalized = normalized * torch.rsqrt(variance + epsilon)
    if self.weight is not None:
        normalized = normalized * self.weight.float()
    if gate is not None:
        normalized = normalized * torch.nn.functional.silu(gate.float())
    return normalized.to(input_dtype)


def _device_safe_gated_delta_forward(
    self: Qwen3_5GatedDeltaNet,
    hidden_states: torch.Tensor,
    *args,
    **kwargs,
):
    """Use Qwen3.5's PyTorch kernels whenever the input is not on CUDA.

    Transformers selects the optional causal-conv1d/FLA implementations when
    those packages are installed, but the selected kernels are CUDA-only.  A
    CPU model would otherwise try to launch them and fail before the documented
    PyTorch fallbacks can run. Transformers 5.14.1 also skips its padding-state
    mask for batch size one, so apply the supplied 2D mask here for identical
    training and single-sample deployment semantics.
    """

    original_forward = type(self)._vla_original_forward
    attention_mask = kwargs.get("attention_mask")
    if attention_mask is None and len(args) >= 2:
        attention_mask = args[1]
    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError(
                "Qwen3.5 Gated DeltaNet attention_mask must match hidden states "
                f"[batch, sequence], got {tuple(attention_mask.shape)} for "
                f"{tuple(hidden_states.shape)}."
            )
        hidden_states = (
            hidden_states
            * attention_mask.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )[:, :, None]
        )
    if hidden_states.device.type == "cuda":
        return original_forward(self, hidden_states, *args, **kwargs)

    kernel_names = (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "recurrent_gated_delta_rule",
    )
    original_kernels = {name: getattr(self, name) for name in kernel_names}
    norm_had_instance_forward = "forward" in self.norm.__dict__
    original_norm_forward = self.norm.__dict__.get("forward")

    self.causal_conv1d_fn = None
    self.causal_conv1d_update = torch_causal_conv1d_update
    self.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
    self.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
    self.norm.forward = MethodType(_torch_rms_norm_gated_forward, self.norm)
    try:
        return original_forward(self, hidden_states, *args, **kwargs)
    finally:
        for name, kernel in original_kernels.items():
            setattr(self, name, kernel)
        if norm_had_instance_forward:
            self.norm.forward = original_norm_forward
        else:
            del self.norm.forward


def apply_qwen35_mot_patch():
    """Install explicit split forwards while leaving ordinary Qwen forwards unchanged."""
    if not hasattr(Qwen3_5DecoderLayer, "_vla_original_forward"):
        Qwen3_5DecoderLayer._vla_original_forward = Qwen3_5DecoderLayer.forward
        Qwen3_5DecoderLayer.forward = _mot_decoder_forward
    if not hasattr(Qwen3_5GatedDeltaNet, "_vla_original_forward"):
        Qwen3_5GatedDeltaNet._vla_original_forward = Qwen3_5GatedDeltaNet.forward
        Qwen3_5GatedDeltaNet.forward = _device_safe_gated_delta_forward


__all__ = ["apply_qwen35_mot_patch", "qwen35_layer_type"]
