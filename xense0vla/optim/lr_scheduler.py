# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from typing import TYPE_CHECKING, Literal

from torch.optim.lr_scheduler import LambdaLR

from ..utils import logging


if TYPE_CHECKING:
    from torch.optim import Optimizer


logger = logging.get_logger(__name__)


def build_lr_scheduler(
    optimizer: "Optimizer",
    train_steps: int,
    lr: float = 1e-3,
    lr_decay_style: Literal["constant", "linear", "cosine", "two_stage"] = "constant",
    lr_decay_ratio: float = 1.0,
    lr_warmup_ratio: float = 0.0,
    lr_min: float = 1e-7,
    lr_start: float = 0.0,
):
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}.")
    if lr_min < 0 or lr_min > lr:
        raise ValueError(f"lr_min ({lr_min}) must lie in [0, lr={lr}].")
    if not 0.0 <= lr_warmup_ratio <= 1.0:
        raise ValueError(f"lr_warmup_ratio ({lr_warmup_ratio}) must lie in [0, 1].")
    if not 0.0 < lr_decay_ratio <= 1.0:
        raise ValueError(f"lr_decay_ratio ({lr_decay_ratio}) must lie in (0, 1].")
    if lr_decay_style == "cosine" and lr_warmup_ratio > lr_decay_ratio:
        raise ValueError(
            f"lr_warmup_ratio ({lr_warmup_ratio}) cannot exceed lr_decay_ratio ({lr_decay_ratio})."
        )
    lr_warmup_steps = int(train_steps * lr_warmup_ratio)
    if lr_decay_style == "constant":
        return get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            lr_start=lr_start,
            init_lr=lr,
        )

    if lr_decay_style == "linear":
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=train_steps,
            init_lr=lr,
            min_lr=lr_min,
            lr_start=lr_start,
        )

    if lr_decay_style == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=train_steps,
            init_lr=lr,
            lr_decay_ratio=lr_decay_ratio,
            min_lr=lr_min,
            lr_start=lr_start,
        )

    if lr_decay_style == "two_stage":
        decay_steps = max(lr_warmup_steps, int(train_steps * lr_decay_ratio))
        decay_ratio = (lr_min / lr) if lr_min > 0 else 0.2
        return get_two_stage_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            init_lr=lr,
            decay_steps=decay_steps,
            decay_ratio=decay_ratio,
            lr_start=lr_start,
        )

    raise ValueError(f"Unknown learning rate decay style: {lr_decay_style}.")


def get_constant_schedule_with_warmup(
    optimizer: "Optimizer",
    num_warmup_steps: int,
    init_lr: float,
    last_epoch: int = -1,
    lr_start: float = 0.0,
):
    """
    Creates a schedule with a constant learning rate preceded by a linear warmup period
    without wasting step 0 at lr=0.
    """

    def _lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return (lr_start + (init_lr - lr_start) * (current_step + 1) / max(1, num_warmup_steps)) / init_lr

        return 1.0

    return LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)


def get_two_stage_constant_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int = 100,
    init_lr: float = 0.0,
    decay_steps: int = 10_000,
    decay_ratio: float = 0.2,
    last_epoch: int = -1,
    lr_start: float = 0.0,
):
    """
    Two-stage constant learning rate schedule with warmup following OpenVLA-OFT.
    """

    def _lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            warmup_lr = lr_start + (init_lr - lr_start) * ((current_step + 1) / max(1, num_warmup_steps))
            return warmup_lr / init_lr

        if current_step >= decay_steps:
            return decay_ratio

        return 1.0

    return LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: "Optimizer",
    num_warmup_steps: int,
    num_training_steps: int,
    init_lr: float,
    last_epoch: int = -1,
    min_lr: float = 1e-7,
    lr_start: float = 0.0,
):
    """
    Creates a schedule with a learning rate that decreases linearly from init_lr to min_lr
    after a linear warmup period.
    """

    def _lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return (lr_start + (init_lr - lr_start) * (current_step + 1) / max(1, num_warmup_steps)) / init_lr

        min_lr_ratio = min_lr / init_lr
        return max(
            min_lr_ratio,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: "Optimizer",
    num_warmup_steps: int,
    num_training_steps: int,
    init_lr: float,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    lr_decay_ratio: float = 1.0,
    min_lr: float = 1e-7,
    lr_start: float = 0.0,
):
    """
    Creates a schedule with a learning rate that decreases following a cosine curve from
    init_lr to min_lr after a linear warmup period.
    """

    def lr_lambda(current_step: int):
        lr_decay_steps = int(num_training_steps * lr_decay_ratio)
        if current_step < num_warmup_steps:
            return (lr_start + (init_lr - lr_start) * (current_step + 1) / max(1, num_warmup_steps)) / init_lr

        min_lr_ratio = min_lr / init_lr
        if current_step >= lr_decay_steps:
            return min_lr_ratio

        progress = float(current_step - num_warmup_steps) / float(max(1, lr_decay_steps - num_warmup_steps))
        progress = min(1.0, max(0.0, progress))
        factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        factor = factor * (1 - min_lr_ratio) + min_lr_ratio
        return max(0.0, factor)

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
