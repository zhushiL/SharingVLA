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


import json
import os
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file


try:
    from diffusers.utils import SAFE_WEIGHTS_INDEX_NAME as DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME
    from diffusers.utils import SAFETENSORS_WEIGHTS_NAME as DIFFUSERS_SAFETENSORS_WEIGHTS_NAME
except Exception:
    DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
    DIFFUSERS_SAFETENSORS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
from torch import distributed as dist
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils.hub import cached_file, get_checkpoint_shard_files

from ..utils import logging
from ..utils.helper import empty_cache, get_dtype_size


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin


    ModelAssets = Union[GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin]


logger = logging.get_logger(__name__)

VLM_BACKBONE_SUBDIR = "vlm_backbone"


@contextmanager
def init_empty_weights():
    """
    A context manager under which models are initialized with all parameters on the meta device.

    Borrowed from: https://github.com/huggingface/accelerate/blob/v1.0.0rc1/src/accelerate/big_modeling.py#L57
    """
    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(module: "nn.Module", name: str, param: "nn.Parameter"):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to("meta"), **kwargs)

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter


@dataclass
class StateDictIterator:
    filepath: str

    def __iter__(self) -> Generator[Tuple[str, "torch.Tensor"], None, None]:
        if self.filepath.endswith(".safetensors"):
            with safe_open(self.filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)

        else:
            state_dict = torch.load(self.filepath, map_location="cpu", weights_only=True, mmap=True)
            for key in state_dict.keys():
                yield key, state_dict[key]


def _load_state_dict(weights_path: str, **kwargs) -> List["StateDictIterator"]:
    """
    Loads (sharded) state dict in transformers' format.
    """
    cache_kwargs = {"_raise_exceptions_for_missing_entries": False, **kwargs}
    resolved_weight_file = cached_file(weights_path, SAFE_WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(weights_path, SAFE_WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_weight_file, **kwargs)
        return [StateDictIterator(shard_file) for shard_file in shard_files]

    resolved_weight_file = cached_file(weights_path, DIFFUSERS_SAFETENSORS_WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(weights_path, DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_weight_file, **kwargs)
        return [StateDictIterator(shard_file) for shard_file in shard_files]

    resolved_weight_file = cached_file(weights_path, WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(weights_path, WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_weight_file, **kwargs)
        return [StateDictIterator(shard_file) for shard_file in shard_files]

    raise ValueError(f"Cannot find checkpoint files in {weights_path}.")


def _find_submodule(module: "nn.Module", name: str) -> Tuple["nn.Module", str]:
    """
    Finds the leaf module according to the name.
    """
    pieces = name.split(".")
    for piece in pieces[:-1]:
        if not hasattr(module, piece):
            raise ValueError(f"Cannot find {piece} in {module}.")

        module = getattr(module, piece)

    return module, pieces[-1]


def _dispatch_parameter(
    module: "nn.Module",
    name: str,
    tensor: "torch.Tensor",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
) -> None:
    """
    Assigns parameter to an empty model.

    NOTE: FSDP module must use in-place operators.
    """
    module, name = _find_submodule(module, name)
    orig_tensor = module._parameters[name].data
    if tensor.shape != orig_tensor.shape:
        raise ValueError(f"Checkpoint shape {tuple(tensor.shape)} differs from {name} shape {tuple(orig_tensor.shape)}.")
    tensor = tensor.to(device=orig_tensor.device, dtype=orig_tensor.dtype)
    if hasattr(orig_tensor, "device_mesh"):
        if dtensor_factory is None:
            raise ValueError("Loading sharded parameters requires dtensor_factory.")

        device_mesh = getattr(orig_tensor, "device_mesh")
        placements = getattr(orig_tensor, "placements")
        module._parameters[name].data.copy_(dtensor_factory(tensor, device_mesh, placements))
    else:  # not dtensor
        module._parameters[name].data.copy_(tensor)


def cast_model_parameters_dtype(model: "nn.Module", dtype: "torch.dtype") -> "nn.Module":
    """Cast floating-point parameters and persistent buffers while keeping non-persistent FP32 buffers intact.

    Calling ``model.to(dtype=torch.bfloat16)`` truncates constructor-computed
    non-persistent RoPE buffers (such as Qwen3.5 ``inv_freq``) to 7 mantissa
    bits, causing severe phase error during BF16 inference.
    """
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is not None and param.is_floating_point() and param.dtype != dtype:
                module._parameters[name] = nn.Parameter(
                    param.data.to(dtype=dtype),
                    requires_grad=param.requires_grad,
                )
        for name, buf in list(module._buffers.items()):
            if (
                buf is not None
                and buf.is_floating_point()
                and name not in module._non_persistent_buffers_set
                and buf.dtype != dtype
            ):
                module._buffers[name] = buf.to(dtype=dtype)
    return model


def _dispatch_buffer(
    module: "nn.Module",
    name: str,
    buffer: "torch.Tensor",
) -> None:
    """
    Assigns buffer to an empty model, preserving non-persistent FP32 precision.
    """
    module, name = _find_submodule(module, name)
    orig_tensor = module._buffers[name].data
    if name in module._non_persistent_buffers_set and buffer.is_floating_point():
        module._buffers[name] = buffer.to(device=orig_tensor.device)
    else:
        module._buffers[name] = buffer.to(orig_tensor)


def initialize_model_weights(model: "nn.Module") -> None:
    """Initialize newly allocated weights with the initializer of their owning HF model."""

    from transformers import PreTrainedModel

    def visit(module: "nn.Module", init_fn) -> None:
        if isinstance(module, PreTrainedModel):
            init_fn = module._init_weights
        for child in module.children():
            visit(child, init_fn)
        if init_fn is not None:
            init_fn(module)

    visit(model, None)


def build_empty_policy(config, dtype):
    from transformers.initialization import no_init_weights
    from .modeling_vla import VLAModel

    with init_empty_weights(), no_init_weights():
        model = cast_model_parameters_dtype(VLAModel(config), dtype)
    if model.model.backbone is not None:
        _tie_module_embeddings(model.model.backbone.qwen35)
    return model


def _tie_module_embeddings(module: "nn.Module") -> None:
    if not getattr(getattr(module, "config", None), "tie_word_embeddings", False):
        return
    try:
        input_embeddings = module.get_input_embeddings()
        output_embeddings = module.get_output_embeddings()
        if input_embeddings is not None and output_embeddings is not None:
            output_embeddings._parameters["weight"] = input_embeddings._parameters["weight"]
    except (AttributeError, NotImplementedError) as exc:
        logger.debug(f"Could not tie embeddings for {type(module).__name__}: {exc}")


def materialize_dcp_model(model, device):
    """Allocate checkpoint destinations while retaining the constructor's computed buffers."""
    buffers = dict(model.named_buffers())
    model.to_empty(device=device)
    for name, tensor in buffers.items():
        _dispatch_buffer(model, name, tensor)
    if model.model.backbone is not None:
        _tie_module_embeddings(model.model.backbone.qwen35)


def _checkpoint_sources(model, path, complete):
    """Source path, input prefix, destination prefix, and explicitly excluded keys."""
    if complete:
        return [(path, "", "", ())]
    if model.config.standalone_action_policy:
        return [(path, "model.backbone.", "model.action_policy.spatial_encoder.",
                 ("model.head.", "model.cam_enc.", "model.cam_dec."))]
    qwen = model.model.backbone.qwen35
    skips = ("mtp.", "model.mtp.")
    if qwen.config.tie_word_embeddings:
        skips += ("lm_head.",)
    return [
        (path, "", "model.backbone.qwen35.", skips),
        (model.config.action_policy_path, "model.action_policy.", "model.action_policy.", ()),
    ]


@torch.no_grad()
def load_model_weights(
    model,
    weights_path: str,
    init_device="cuda",
    dtensor_factory=None,
    load_full_checkpoint: bool = False,
):
    """Strict, streaming initialization from a complete policy or pretrained components."""
    qwen = model.model.backbone.qwen35 if model.model.backbone is not None else None
    _tie_module_embeddings(qwen)
    # to_empty destroys storage, including constructor-computed non-persistent RoPE buffers.
    buffers = {name: tensor.clone() for name, tensor in model.named_buffers()}
    persistent = {
        f"{prefix}.{key}" if prefix else key
        for prefix, module in model.named_modules()
        for key, value in module.named_buffers(recurse=False)
        if key not in module._non_persistent_buffers_set
    }
    parameters = set(dict(model.named_parameters()))
    aliases = set(dict(model.named_parameters(remove_duplicate=False))) - parameters
    model.to_empty(device=init_device)
    if model.config.standalone_action_policy and not load_full_checkpoint:
        initialize_model_weights(model)

    for path, source, target, skipped in _checkpoint_sources(model, weights_path, load_full_checkpoint):
        expected = {name for name in parameters | persistent if name.startswith(target)}
        if not expected:
            raise ValueError(f"No model tensors match checkpoint target {target!r}.")
        for shard in _load_state_dict(path):
            for key, tensor in shard:
                if key.startswith(skipped):
                    continue
                if not key.startswith(source):
                    raise KeyError(f"Unexpected key {key!r} in {path}.")
                name = target + key[len(source):]
                if name in aliases:
                    continue
                if name not in expected:
                    raise KeyError(f"Unexpected or duplicate tensor {name!r} from {path}.")
                expected.remove(name)
                if name in parameters:
                    _dispatch_parameter(model, name, tensor, dtensor_factory)
                else:
                    if tensor.shape != buffers[name].shape:
                        raise ValueError(f"Buffer {name} has incompatible checkpoint shape {tensor.shape}.")
                    buffers[name] = tensor
        if expected:
            raise KeyError(f"Checkpoint {path} is missing {sorted(expected)}.")
    for name, tensor in buffers.items():
        _dispatch_buffer(model, name, tensor)
    _tie_module_embeddings(qwen)


def _get_shard_info(
    state_dict: Dict[str, "torch.Tensor"],
    save_dtype: Optional[Union[str, "torch.dtype"]],
    shard_size: int,
    safe_serialization: bool,
) -> Tuple[bool, int, Dict[str, str]]:
    """
    Gets the shard information, should be executed at rank 0.
    """
    current_size, total_size = 0, 0
    current_shard, shard_list = [], []
    for name, tensor in state_dict.items():
        if isinstance(save_dtype, str):
            dtype = getattr(torch, save_dtype)
        elif isinstance(save_dtype, torch.dtype):
            dtype = save_dtype
        else:
            dtype = tensor.dtype
        tensor_size = tensor.numel() * get_dtype_size(dtype)  # dtensor's numel == tensor's numel
        if current_size != 0 and current_size + tensor_size > shard_size:
            total_size += current_size
            shard_list.append(current_shard)
            current_size = 0
            current_shard = []

        current_size += tensor_size
        current_shard.append(name)

    if current_size != 0:
        total_size += current_size
        shard_list.append(current_shard)

    weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
    num_shards = len(shard_list)
    weight_map = OrderedDict()
    is_sharded = None
    if num_shards == 1:
        is_sharded = False
        for name in shard_list[0]:
            weight_map[name] = weights_name
    else:
        is_sharded = True
        for shard_idx, shard in enumerate(shard_list):
            prefix, extension = weights_name.rsplit(".", maxsplit=1)
            file_name = f"{prefix}-{shard_idx + 1:05d}-of-{num_shards:05d}.{extension}"
            for name in shard:
                weight_map[name] = file_name

    return is_sharded, total_size, weight_map


def _save_state_dict(
    state_dict: Dict[str, "torch.Tensor"],
    path_to_save: "os.PathLike",
    safe_serialization: bool,
) -> None:
    """
    Save function.
    """
    if os.path.exists(path_to_save):
        os.remove(path_to_save)
    if safe_serialization:
        # Keep explicit tied keys in portable exports. Safetensors requires
        # distinct contiguous storage even when two model keys are aliases.
        tensors = {}
        storage_pointers = set()
        for name, tensor in state_dict.items():
            tensor = tensor.contiguous()
            storage = (tensor.device, tensor.untyped_storage().data_ptr())
            if storage in storage_pointers:
                tensor = tensor.clone()
            storage_pointers.add(storage)
            tensors[name] = tensor
        save_file(tensors, path_to_save, metadata={"format": "pt"})
    else:
        torch.save(state_dict, path_to_save)


@torch.no_grad()
def save_model_weights(
    output_dir: Union[str, "os.PathLike"],
    state_dict: Dict[str, "torch.Tensor"],
    global_rank: Optional[int] = None,
    save_dtype: Optional[Union[str, "torch.dtype"]] = None,
    shard_size: int = 5_000_000_000,
    safe_serialization: bool = True,
    model_assets: Optional[Sequence["ModelAssets"]] = None,
) -> None:
    """
    Saves full model weights. The model parameters should be either tensor or
    DTensor. By default, each tensor keeps its source dtype; ``save_dtype`` is
    only for callers that explicitly require a uniform conversion.

    If global_rank is given, it will assume it is executed on all ranks.
    """

    os.makedirs(output_dir, exist_ok=True)
    is_sharded, total_size, weight_map = _get_shard_info(state_dict, save_dtype, shard_size, safe_serialization)
    full_state_dict = OrderedDict()
    prev_file_name = None
    for name, tensor in state_dict.items():
        if hasattr(tensor.data, "full_tensor"):  # dtensor
            tensor = tensor.data.full_tensor()
        else:
            tensor = tensor.data

        if save_dtype:
            tensor = tensor.to(dtype=getattr(torch, save_dtype) if isinstance(save_dtype, str) else save_dtype)

        if prev_file_name is not None and weight_map[name] != prev_file_name:
            if global_rank is None or global_rank == 0:
                _save_state_dict(full_state_dict, os.path.join(output_dir, prev_file_name), safe_serialization)
                full_state_dict = OrderedDict()

            empty_cache()
            if global_rank is not None and dist.is_initialized():  # avoid process hanging
                torch.cuda.synchronize()
                dist.barrier()

        if global_rank is None or global_rank == 0:
            full_state_dict[name] = tensor.detach().cpu()

        prev_file_name = weight_map[name]
        del tensor

    if global_rank is None or global_rank == 0:
        if len(full_state_dict):
            _save_state_dict(full_state_dict, os.path.join(output_dir, prev_file_name), safe_serialization)

        if is_sharded:
            index = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }

            index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            with open(os.path.join(output_dir, index_file), "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)

            logger.info(f"Model weight splits saved in {output_dir}.")
        else:
            logger.info(f"Model weights saved at {os.path.join(output_dir, prev_file_name)}.")

        if model_assets is not None:
            save_model_assets(output_dir, model_assets)


def save_model_assets(output_dir: Union[str, "os.PathLike"], model_assets: Sequence["ModelAssets"]):
    """Save portable model configuration and, for a complete VLA, its language processor."""
    config = next(asset for asset in model_assets if getattr(asset, "model_type", None) == "vla")
    portable = deepcopy(config)
    portable.vlm_path = portable.action_policy_path = portable.spatial_encoder_path = None
    portable.tokenizer_path = None if portable.standalone_action_policy else VLM_BACKBONE_SUBDIR
    portable.save_pretrained(output_dir)
    if not portable.standalone_action_policy:
        from transformers.models.qwen3_5 import Qwen3_5Config

        processor = next((asset for asset in model_assets if getattr(asset, "tokenizer", None) is not None), None)
        if processor is None:
            raise ValueError("VLA export requires the Qwen processor and tokenizer.")
        backbone_dir = os.path.join(output_dir, VLM_BACKBONE_SUBDIR)
        if not isinstance(portable.backbone_config, dict):
            raise ValueError("VLA export requires a serialized backbone_config dict.")
        Qwen3_5Config(**portable.backbone_config).save_pretrained(backbone_dir)
        processor.save_pretrained(backbone_dir)
