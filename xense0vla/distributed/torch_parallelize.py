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


import os
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import create_selective_checkpoint_contexts

from ..models.module_utils import initialize_model_weights, load_model_weights
from ..utils import logging
from .checkpoint import CheckpointFunction
from .fsdp import (
    init_fsdp_fn,
    parallel_init_fsdp_fn,
    parallel_load_safetensors,
)
from .parallel_state import get_parallel_state


from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor.parallel import parallelize_module


logger = logging.get_logger(__name__)

def verbose_fsdp_grouping(model, prefix="", depth=0):
    indent = "    " * depth

    for name, child in model.named_children():
        if isinstance(child, FullyShardedDataParallel):
            module_names = [m_name for m_name, _ in child.named_modules()][1:]  # [1:] excludes self
            strategy = child.sharding_strategy
            logger.debug_rank0(f"{indent}├── [FSDP Group] {prefix}{name}")
            logger.debug_rank0(
                f"{indent}│   ├── Sharding Strategy: {strategy}, Mixed Precision: {child.mixed_precision}"
            )
            logger.debug_rank0(f"{indent}│   └── Contains Modules: {module_names}")

            verbose_fsdp_grouping(child, prefix=f"{prefix}{name}.", depth=depth + 1)
        else:
            verbose_fsdp_grouping(child, prefix=f"{prefix}{name}.", depth=depth)


def build_parallelize_model(
    model: "nn.Module",
    weights_path: Optional[str] = None,
    sharding_plan: Optional[Dict[str, Any]] = None,
    enable_full_shard: bool = True,
    enable_mixed_precision: bool = True,
    enable_gradient_checkpointing: bool = True,
    basic_modules: Optional[List[str]] = None,
    fsdp_llm_blocks: bool = True,
    ignore_norm: bool = False,
    vlm_fsdp: bool = False,
    parameter_dtype: Optional[str] = None,
    **kwargs,
) -> "nn.Module":
    """
    Applies parallel strategies to the model.
    """
    parallel_state = get_parallel_state()

    deferred_load = getattr(model, "_deferred_weight_load", None)
    if weights_path is None and deferred_load is not None:
        weights_path = deferred_load["weights_path"]

    if not parallel_state.fsdp_enabled:
        if kwargs.get("init_device") != "cuda":
            raise ValueError("Only FSDP training supports `init_device=cpu` or `init_device=meta`.")
        if kwargs.pop("enable_fsdp_offload", False):
            raise ValueError("Only FSDP training supports `enable_fsdp_offload`.")

    if parameter_dtype is not None:
        if parameter_dtype not in {"float32", "bfloat16"}:
            raise ValueError("parameter_dtype must be float32 or bfloat16.")
        model = model.to(dtype=getattr(torch, parameter_dtype))
    elif enable_mixed_precision:
        # Preserve the historical FP32 optimizer default for other callers.
        model = model.float()

    if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        logger.info_rank0("Enable gradient checkpointing.")
        use_reentrant = kwargs.pop("enable_reentrant", False)
        if use_reentrant:
            torch.utils.checkpoint.CheckpointFunction = CheckpointFunction

        ops_to_save = kwargs.pop("ops_to_save", None)
        gradient_checkpointing_kwargs = {"use_reentrant": use_reentrant}
        if ops_to_save is not None:
            context_fn = partial(create_selective_checkpoint_contexts, ops_to_save)
            gradient_checkpointing_kwargs["context_fn"] = context_fn
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    if parallel_state.tp_enabled:
        logger.info_rank0("Apply tensor parallel to the model.")
        model = parallelize_module(
            model,
            device_mesh=parallel_state.tp_mesh,
        )

    if parallel_state.fsdp_enabled:
        logger.info_rank0(f"Apply data parallel to the model: {parallel_state.dp_mode}.")
        if parallel_state.dp_mode == "fsdp2":
            if kwargs.get("init_device") == "cpu":
                raise ValueError(
                    "FSDP2 does not support rank-0-only CPU initialization in this loader. "
                    "Use init_device='cuda' or 'meta'."
                )

            def merge_ignored_params(fsdp_options: Dict[str, Any], params: set) -> None:
                if not params:
                    return
                ignored_params = set(fsdp_options.get("ignored_params", set()))
                ignored_params.update(params)
                fsdp_options["ignored_params"] = ignored_params

            def module_local_fsdp_kwargs(module: "nn.Module", fsdp_options: Dict[str, Any]) -> Dict[str, Any]:
                ignored_params = fsdp_options.get("ignored_params", None)
                if not ignored_params:
                    return fsdp_options

                local_params = set(module.parameters())
                local_ignored_params = ignored_params.intersection(local_params)
                module_kwargs = dict(fsdp_options)
                if local_ignored_params:
                    module_kwargs["ignored_params"] = local_ignored_params
                else:
                    module_kwargs.pop("ignored_params", None)
                return module_kwargs

            fsdp_kwargs = {
                "mesh": parallel_state.fsdp_mesh,
                "reshard_after_forward": enable_full_shard,
                **kwargs.pop("fsdp_kwargs", {}),
            }
            if enable_mixed_precision:
                logger.info_rank0("Enable BF16 mixed precision training.")
                mp_policy = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    output_dtype=torch.bfloat16,
                )
                fsdp_kwargs["mp_policy"] = mp_policy
            if ignore_norm:
                norm_parameters = {
                    parameter
                    for module in model.modules()
                    if "norm" in module.__class__.__name__.lower()
                    for parameter in module.parameters(recurse=False)
                }
                merge_ignored_params(fsdp_kwargs, norm_parameters)

            ignored_module_classes = getattr(
                model, "get_ignore_modules_in_mixed_precision", lambda: ()
            )() or ()
            ignore_modules_in_mixed_precision = tuple(ignored_module_classes)
            basic_module_names = set(basic_modules or ())

            def apply_fsdp_to_decoder_blocks(module: "nn.Module") -> None:
                if (
                    module.__class__.__name__ in basic_module_names
                    or module.__class__ in ignore_modules_in_mixed_precision
                ):
                    logger.debug(f"Apply FSDP2 to {module.__class__.__name__}.")
                    module_fsdp_kwargs = module_local_fsdp_kwargs(module, fsdp_kwargs)
                    if module.__class__ in ignore_modules_in_mixed_precision:
                        module_fsdp_kwargs = {
                            key: value
                            for key, value in module_fsdp_kwargs.items()
                            if key != "mp_policy"
                        }
                    fully_shard(module, **module_fsdp_kwargs)

            if basic_module_names:
                model.apply(apply_fsdp_to_decoder_blocks)
            elif fsdp_llm_blocks:
                block_names = set(getattr(model, "_no_split_modules", ()))
                blocks = [
                    module
                    for module in model.modules()
                    if module is not model and module.__class__.__name__ in block_names
                ]
                if not blocks:
                    raise RuntimeError(
                        "fsdp_llm_blocks=True but the model exposes no matching _no_split_modules."
                    )
                logger.info_rank0(f"Applying FSDP2 to {len(blocks)} model blocks.")
                for block in blocks:
                    fully_shard(block, **module_local_fsdp_kwargs(block, fsdp_kwargs))

            if vlm_fsdp and not basic_module_names:
                loader_factory = getattr(type(model), "get_weight_loader", None)
                if loader_factory is None:
                    raise TypeError("vlm_fsdp=True requires the model to provide get_weight_loader().")
                vlm = loader_factory().get_vlm_submodule(model)
                block_names = set(
                    getattr(vlm, "_no_split_modules", ())
                    or getattr(model, "_no_split_modules", ())
                )
                blocks = [
                    module
                    for module in vlm.modules()
                    if module is not vlm and module.__class__.__name__ in block_names
                ]
                if not blocks:
                    raise RuntimeError(
                        "vlm_fsdp=True but no VLM blocks match the model's _no_split_modules."
                    )
                logger.info_rank0(f"Applying FSDP2 to {len(blocks)} VLM blocks.")
                for block in blocks:
                    fully_shard(block, **module_local_fsdp_kwargs(block, fsdp_kwargs))
            elif vlm_fsdp:
                logger.info_rank0(
                    "vlm_fsdp is already covered by module-level FSDP; "
                    "skipping duplicate VLM wrapping."
                )

            fully_shard(model, **fsdp_kwargs)

            if kwargs.get("init_device") == "meta":
                if weights_path is None:
                    # shard init empty model with fsdp2
                    model.to_empty(device="cuda")
                    initialize_model_weights(model)
                else:
                    from torch.distributed.tensor import distribute_tensor

                    load_kwargs = deferred_load or {}
                    load_model_weights(
                        model,
                        weights_path,
                        "cuda",
                        dtensor_factory=distribute_tensor,
                        load_vlm_only=bool(load_kwargs.get("load_vlm_only", False)),
                        load_full_checkpoint=bool(load_kwargs.get("load_full_checkpoint", False)),
                    )
                    if hasattr(model, "_deferred_weight_load"):
                        delattr(model, "_deferred_weight_load")

        elif parallel_state.dp_mode == "fsdp1":
            basic_module_names = set(basic_modules or ())
            wrap_policy = (
                partial(
                    lambda_auto_wrap_policy,
                    lambda_fn=lambda module: module.__class__.__name__ in basic_module_names,
                )
                if basic_module_names
                else None
            )

            # set fsdp/hsdp sharding strategy
            if parallel_state.fsdp_mesh.ndim > 1 and parallel_state.fsdp_mesh.size() > 1:
                strategy = ShardingStrategy.HYBRID_SHARD
            else:
                strategy = ShardingStrategy.FULL_SHARD

            fsdp_kwargs = {
                "auto_wrap_policy": wrap_policy,
                "device_id": torch.cuda.current_device(),
                "sharding_strategy": strategy if enable_full_shard else ShardingStrategy.NO_SHARD,
                "use_orig_params": True,
            }

            fsdp_kwargs["device_mesh"] = parallel_state.fsdp_mesh

            fsdp_kwargs.update(kwargs.pop("fsdp_kwargs", {}))

            if enable_mixed_precision:
                logger.info_rank0("Enable BF16 mixed precision training.")
                mixed_precision = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32,
                )
                if hasattr(model, "get_ignore_modules_in_mixed_precision"):
                    mixed_precision._module_classes_to_ignore += model.get_ignore_modules_in_mixed_precision()

                fsdp_kwargs["mixed_precision"] = mixed_precision

            if kwargs.get("init_device") == "cpu":
                logger.info_rank0("Enable rank0-only initialization.")
                fsdp_kwargs["sync_module_states"] = True
                if parallel_state.global_rank != 0:
                    fsdp_kwargs["param_init_fn"] = init_fsdp_fn(model, device="cuda")
            elif kwargs.get("init_device") == "meta":
                # assert weights_path is not None, "`weights_path` must be provided when `init_device=meta` for fsdp1."

                logger.info_rank0("Enable meta initialization.")
                provided_state_dict = kwargs.get("state_dict", {})
                if weights_path is None and not provided_state_dict:
                    raise ValueError(
                        "FSDP1 meta initialization requires weights_path or state_dict. "
                        "Use init_device='cuda' to initialize a dense model from scratch."
                    )
                if deferred_load is not None and deferred_load.get("load_vlm_only", False):
                    raise ValueError(
                        "FSDP1 meta initialization cannot map a standalone VLM checkpoint "
                        "into the composite dense model. Use init_device='cuda'/'cpu' or FSDP2."
                    )
                if weights_path and not os.path.isdir(weights_path):
                    raise ValueError(
                        "FSDP1 meta initialization requires a local Transformers checkpoint "
                        "directory. Use init_device='cuda'/'cpu' or FSDP2 for Hub model IDs."
                    )
                shard_states = (
                    parallel_load_safetensors(weights_path)
                    if weights_path
                    else provided_state_dict
                )
                fsdp_kwargs["param_init_fn"] = parallel_init_fsdp_fn(model, shard_states)

            if kwargs.pop("enable_fsdp_offload", False):
                logger.info_rank0("Enable offloading for parameters & gradients & optimizer states.")
                fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

            if kwargs.pop("enable_forward_prefetch", False):
                fsdp_kwargs["forward_prefetch"] = True
            else:
                fsdp_kwargs["forward_prefetch"] = False
                fsdp_kwargs["backward_prefetch"] = None

            # FULLY_SHARD first
            model = FullyShardedDataParallel(model, **fsdp_kwargs)

            verbose_fsdp_grouping(model)
        else:
            ddp_kwargs = {"device_ids": [parallel_state.local_rank]}
            if enable_mixed_precision:
                logger.info_rank0("Enable BF16 mixed precision training.")
                mixed_precision = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.bfloat16,
                )
                ddp_kwargs["mixed_precision"] = mixed_precision

            model = DDP(model, **ddp_kwargs)

    return model
