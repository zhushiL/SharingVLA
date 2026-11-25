# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

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


from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

from ..utils.import_utils import is_torch_npu_available
from .muon import DistributedMuon, split_muon_adamw_params


_VLM_PARAMETER_PATH = "model.backbone."
_SPATIAL_PARAMETER_PATH = "model.action_policy.spatial_encoder."
_NO_DECAY_NAME_PATTERNS: Tuple[str, ...] = (
    "bias",
    "norm",
    "modulation",
    "a_log",
    "dt_bias",
    "embed_tokens",
    "embedding",
)


def _collect_no_decay_fqns(model: "torch.nn.Module") -> set:
    fqns: set = set()
    norm_types = (torch.nn.LayerNorm, torch.nn.RMSNorm, torch.nn.GroupNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.Embedding)
    for module_name, module in model.named_modules():
        if isinstance(module, norm_types) or "norm" in module.__class__.__name__.lower():
            for pname, _ in module.named_parameters(recurse=False):
                fqns.add(f"{module_name}.{pname}" if module_name else pname)
    return fqns


def _is_no_decay_param(name: str, param: torch.Tensor, no_decay_fqns: set) -> bool:
    if param.ndim < 2 or name in no_decay_fqns or name.endswith(".bias"):
        return True
    lname = name.lower()
    return any(pat in lname for pat in _NO_DECAY_NAME_PATTERNS)


def _split_named_params_by_module_lr(
    params: Sequence[torch.Tensor],
    names: Sequence[str],
    base_lr: float,
    vlm_lr: Optional[float],
    spatial_encoder_lr: Optional[float],
    *,
    weight_decay: float = 0.0,
    no_decay_fqns: Optional[set] = None,
) -> List[Dict[str, Any]]:
    if len(params) != len(names):
        raise ValueError("Parameter and FQN lists must have the same length.")

    no_decay_set = no_decay_fqns or set()
    buckets: Dict[Tuple[str, bool], List[torch.Tensor]] = {
        ("base", True): [],
        ("base", False): [],
        ("vlm", True): [],
        ("vlm", False): [],
        ("spatial_encoder", True): [],
        ("spatial_encoder", False): [],
    }
    lr_by_bucket = {
        "base": base_lr,
        "vlm": vlm_lr if vlm_lr is not None else base_lr,
        "spatial_encoder": spatial_encoder_lr if spatial_encoder_lr is not None else base_lr,
    }
    for param, name in zip(params, names):
        if vlm_lr is not None and _VLM_PARAMETER_PATH in name:
            bucket = "vlm"
        elif spatial_encoder_lr is not None and _SPATIAL_PARAMETER_PATH in name:
            bucket = "spatial_encoder"
        else:
            bucket = "base"
        use_decay = weight_decay > 0.0 and not _is_no_decay_param(name, param, no_decay_set)
        buckets[(bucket, use_decay)].append(param)

    groups: List[Dict[str, Any]] = []
    for bucket in ("base", "vlm", "spatial_encoder"):
        for use_decay in (True, False):
            group_params = buckets[(bucket, use_decay)]
            if group_params:
                suffix = "" if use_decay or weight_decay == 0.0 else "_no_decay"
                groups.append(
                    {
                        "params": group_params,
                        "lr": lr_by_bucket[bucket],
                        "weight_decay": weight_decay if use_decay else 0.0,
                        "group_name": f"{bucket}{suffix}",
                    }
                )
    return groups


def _build_module_lr_param_groups(
    model: "torch.nn.Module",
    base_lr: float,
    vlm_lr: Optional[float],
    spatial_encoder_lr: Optional[float],
    weight_decay: float = 0.0,
) -> List[Dict[str, Any]]:
    no_decay_fqns = _collect_no_decay_fqns(model)
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    names = [name for name, _ in trainable]
    params = [param for _, param in trainable]
    return _split_named_params_by_module_lr(
        params,
        names,
        base_lr,
        vlm_lr,
        spatial_encoder_lr,
        weight_decay=weight_decay,
        no_decay_fqns=no_decay_fqns,
    )


# https://github.com/meta-llama/llama-recipes/blob/v0.0.4/src/llama_recipes/policies/anyprecision_optimizer.py
class AnyPrecisionAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        use_kahan_summation=True,
        momentum_dtype=torch.bfloat16,
        variance_dtype=torch.bfloat16,
        compensation_buffer_dtype=torch.bfloat16,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "use_kahan_summation": use_kahan_summation,
            "momentum_dtype": momentum_dtype,
            "variance_dtype": variance_dtype,
            "compensation_buffer_dtype": compensation_buffer_dtype,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """

        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            use_kahan_summation = group["use_kahan_summation"]

            momentum_dtype = group["momentum_dtype"]
            variance_dtype = group["variance_dtype"]
            compensation_buffer_dtype = group["compensation_buffer_dtype"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("AnyPrecisionAdamW does not support sparse gradients.")

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)

                    # momentum - EMA of gradient values
                    state["exp_avg"] = torch.zeros_like(p, dtype=momentum_dtype)

                    # variance uncentered - EMA of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=variance_dtype)

                    # optional Kahan summation - accumulated error tracker
                    if use_kahan_summation:
                        state["compensation"] = torch.zeros_like(p, dtype=compensation_buffer_dtype)

                # Main processing
                # update the steps for each param group update
                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad = p.grad

                if weight_decay:  # weight decay, AdamW style
                    p.data.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # update momentum
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # update uncentered variance

                bias_correction1 = 1 - beta1**step  # adjust using bias1
                step_size = lr / bias_correction1

                denom_correction = (1 - beta2**step) ** 0.5  # adjust using bias2 and avoids math import
                centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(eps, alpha=1)

                if use_kahan_summation:  # lr update to compensation
                    compensation = state["compensation"]
                    compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)

                    # update weights with compensation (Kahan summation)
                    # save error back to compensation for next iteration
                    temp_buffer = p.detach().clone()
                    p.data.add_(compensation)
                    compensation.add_(temp_buffer.sub_(p.data))
                else:  # usual AdamW updates
                    p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)


def build_optimizer(
    model: "torch.nn.Module",
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    fused: bool = False,
    optimizer_type: str = "adamw",
    param_groups: Optional[Sequence[Dict[str, Any]]] = None,
    vlm_lr: Optional[float] = None,
    spatial_encoder_lr: Optional[float] = None,
) -> "torch.optim.Optimizer":
    if param_groups is None:
        param_groups = _build_module_lr_param_groups(
            model, lr, vlm_lr, spatial_encoder_lr, weight_decay=weight_decay
        )
    elif vlm_lr is not None or spatial_encoder_lr is not None:
        special_lr_names = [
            name
            for name, value in (("vlm_lr", vlm_lr), ("spatial_encoder_lr", spatial_encoder_lr))
            if value is not None
        ]
        special_lr_label = " and ".join(special_lr_names)
        raise ValueError(
            f"{special_lr_label} cannot be combined with explicit param_groups."
        )

    if optimizer_type == "adamw":
        foreach = False if is_torch_npu_available() else (not fused)
        fused = False if is_torch_npu_available() else fused
        optim = AdamW(param_groups, lr, betas, eps, weight_decay, fused=fused, foreach=foreach)
    elif optimizer_type == "anyprecision_adamw":
        optim = AnyPrecisionAdamW(param_groups, lr, betas, eps, weight_decay)
    elif optimizer_type == "muon":
        raise ValueError(
            "optimizer_type='muon' must go through build_muon_optimizer(model, args_train, ...) "
            "so that non-2D parameters can be routed to AdamW."
        )
    else:
        raise ValueError("Only adamw and anyprecision_adamw are supported as optimizers.")

    return optim


class CombinedOptimizer(Optimizer):
    """Drive several inner optimizers as if they were a single one."""

    def __init__(self, optimizers: Sequence[Optimizer]):
        if not optimizers:
            raise ValueError("CombinedOptimizer needs at least one inner optimizer.")
        self.optimizers: List[Optimizer] = list(optimizers)
        self.defaults = {}

    @property
    def param_groups(self):
        groups: List[Dict[str, Any]] = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    @param_groups.setter
    def param_groups(self, value):
        # LR schedulers mutate the shared param-group dicts; reassignment is a no-op.
        pass

    @property
    def state(self):
        merged: Dict[Any, Any] = {}
        for opt in self.optimizers:
            merged.update(opt.state)
        return merged

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)


def build_muon_optimizer(
    model: "torch.nn.Module",
    args_train,
    lr: float,
    weight_decay: float = 0.0,
    adamw_betas: Tuple[float, float] = (0.9, 0.95),
    adamw_eps: float = 1e-8,
) -> "torch.optim.Optimizer":
    """Build DistributedMuon for matrix-like weights plus AdamW fallback groups."""
    muon_params, adamw_params, muon_names, adamw_names = split_muon_adamw_params(
        model,
        no_decay_modules=None,
        no_decay_params=None,
        extra_adamw_name_patterns=getattr(args_train, "muon_exclude_name_patterns", None) or None,
    )

    no_decay_fqns = _collect_no_decay_fqns(model)
    vlm_lr = getattr(args_train, "vlm_lr", None)
    spatial_encoder_lr = getattr(args_train, "spatial_encoder_lr", None)
    muon_groups = _split_named_params_by_module_lr(
        muon_params,
        muon_names,
        lr,
        vlm_lr,
        spatial_encoder_lr,
        weight_decay=weight_decay,
        no_decay_fqns=no_decay_fqns,
    )
    adamw_groups = _split_named_params_by_module_lr(
        adamw_params,
        adamw_names,
        lr,
        vlm_lr,
        spatial_encoder_lr,
        weight_decay=weight_decay,
        no_decay_fqns=no_decay_fqns,
    )

    if not muon_groups:
        raise RuntimeError(
            "build_muon_optimizer: no Muon-eligible 2D parameters were found. "
            "Use build_optimizer(optimizer_type='adamw') instead."
        )

    muon_opt = DistributedMuon(
        muon_groups,
        lr=lr,
        weight_decay=weight_decay,
        momentum=float(getattr(args_train, "muon_momentum", 0.95)),
        nesterov=bool(getattr(args_train, "muon_nesterov", True)),
        ns_steps=int(getattr(args_train, "muon_ns_steps", 5)),
        adjust_lr_fn=getattr(args_train, "muon_adjust_lr_fn", "match_rms_adamw"),
    )

    inner_opts: List[Optimizer] = [muon_opt]
    if adamw_groups:
        fused = False if is_torch_npu_available() else bool(getattr(args_train, "fused_optimizer", False))
        foreach = False if is_torch_npu_available() else (not fused)
        adamw_opt = AdamW(
            adamw_groups,
            lr=lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=weight_decay,
            fused=fused,
            foreach=foreach,
        )
        inner_opts.append(adamw_opt)

    return CombinedOptimizer(inner_opts)
