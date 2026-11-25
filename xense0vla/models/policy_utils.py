"""Small tensor helpers shared by the Qwen3.5 VLA policy."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def create_sinusoidal_pos_embedding(
    time: Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Encode scalar diffusion times with arbitrary leading dimensions.

    Ordinary flow matching supplies ``[batch]`` times.  Training-time RTC
    additionally supplies one time per action token, ``[batch, horizon]``.
    """

    if dimension % 2:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim < 1:
        raise ValueError("time must have at least one dimension")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction
    radians = (2.0 * math.pi / period) * time.float().unsqueeze(-1)
    return torch.cat([torch.sin(radians), torch.cos(radians)], dim=-1)


def sample_beta(alpha: float, beta: float, batch_size: int, device: torch.device | str) -> Tensor:
    """Sample an exact Beta distribution on the requested device."""

    alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_tensor = torch.as_tensor(beta, dtype=torch.float32, device=device)
    distribution = torch.distributions.Beta(alpha_tensor, beta_tensor)
    return distribution.sample((batch_size,))


def sample_rtc_delays(max_valid_prefix: Tensor, max_delay_steps: int) -> Tensor:
    """Uniformly sample each item's delay from 0 through its valid maximum.

    The caller leaves at least one valid action after the clean prefix. Zero
    delay has exactly the same probability as every other allowed delay.
    """
    if max_valid_prefix.ndim != 1 or max_delay_steps < 1:
        raise ValueError("Expected a [batch] prefix bound and a positive maximum delay.")
    upper = max_valid_prefix.long().clamp(min=0, max=max_delay_steps)
    return torch.floor(torch.rand(upper.shape, device=upper.device) * (upper + 1)).long()


def action_joint_mask(actions: Tensor, joint_mask: Tensor | None, action_dim: int) -> Tensor:
    if joint_mask is None:
        valid = torch.arange(actions.shape[-1], device=actions.device) < action_dim
        return valid.expand_as(actions)
    if joint_mask.shape != actions.shape:
        raise ValueError("joint_mask must match the action tensor.")
    return joint_mask.bool()


def sample_clean_prefix(valid_steps: Tensor, max_delay: int) -> Tensor:
    if max_delay == 0:
        return torch.zeros_like(valid_steps, dtype=torch.bool)
    # Keep at least one valid target after the contiguous clean prefix.
    last_delay = valid_steps.long().cumprod(dim=1).sum(dim=1).sub(1)
    delays = sample_rtc_delays(last_delay, max_delay)
    return torch.arange(valid_steps.shape[1], device=valid_steps.device) < delays[:, None]


__all__ = ["create_sinusoidal_pos_embedding", "sample_beta", "sample_rtc_delays", "action_joint_mask", "sample_clean_prefix"]
