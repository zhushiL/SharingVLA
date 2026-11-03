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


__all__ = ["create_sinusoidal_pos_embedding", "sample_beta", "sample_rtc_delays"]
