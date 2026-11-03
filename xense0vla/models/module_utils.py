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
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Literal, Optional, Sequence, Tuple, Union

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
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils.hub import cached_file, get_checkpoint_shard_files

from ..utils import logging
from ..utils.helper import empty_cache, get_dtype_size


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from .loader import VLAWeightLoader

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
    prefix: str = ''

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
    tensor = tensor.to(orig_tensor)
    if hasattr(orig_tensor, "device_mesh"):  # dtensor
        if orig_tensor.device.type == "cpu":
            raise ValueError("Cannot load dtensor on CPU.")

        device_mesh = getattr(orig_tensor, "device_mesh")
        placements = getattr(orig_tensor, "placements")
        module._parameters[name].data.copy_(dtensor_factory(tensor, device_mesh, placements))
    else:  # not dtensor
        module._parameters[name].data.copy_(tensor)


def _dispatch_buffer(
    module: "nn.Module",
    name: str,
    buffer: "torch.Tensor",
) -> None:
    """
    Assigns buffer to an empty model.
    """
    module, name = _find_submodule(module, name)
    orig_tensor = module._buffers[name].data
    module._buffers[name] = buffer.to(orig_tensor)


def initialize_model_weights(model: "nn.Module") -> None:
    """Initialize a materialized model with the closest owning HF initializer.

    VLA pre-training builds the composite model on ``meta`` and loads only the
    VLM checkpoint.  Initializing before dispatching checkpoint tensors makes
    every new dense expert/head parameter valid, while the subsequent load
    overwrites all pretrained VLM tensors.  Nested ``PreTrainedModel`` objects
    keep their own special initialization (notably Qwen3.5 Gated DeltaNet
    ``A_log``/``dt_bias`` and zero-centered RMSNorm weights).
    """

    from transformers import PreTrainedModel

    def visit(module: "nn.Module", init_fn) -> None:
        if isinstance(module, PreTrainedModel):
            init_fn = module._init_weights
        for child in module.children():
            visit(child, init_fn)
        if init_fn is not None:
            init_fn(module)

    visit(model, None)


def _get_weight_loader(model) -> "VLAWeightLoader":
    loader_factory = getattr(type(model), "get_weight_loader", None)
    if loader_factory is None:
        return None
    return loader_factory()


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


@torch.no_grad()
def load_model_weights(
    model: Union["nn.Module", "PreTrainedModel"],
    weights_path: str,
    init_device: Literal["cpu", "cuda"] = "cuda",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
    load_vlm_only: bool = False,
    load_full_checkpoint: bool = False,
) -> None:
    """
    Loads pre-trained model states in transformers' format.
    """
    if load_vlm_only == load_full_checkpoint:
        raise ValueError(
            "Exactly one strict initialization mode must be selected: "
            "Qwen-only initialization or complete checkpoint loading."
        )

    weight_loader = _get_weight_loader(model)
    if weight_loader is None:
        raise TypeError(
            f"{type(model).__name__} must implement get_weight_loader() for "
            "strict VLA checkpoint loading."
        )

    # Constructors run under no_init_weights(), which can skip HF's normal
    # tying step. Establish aliases before determining strict expected keys.
    _tie_module_embeddings(model)
    _tie_module_embeddings(weight_loader.get_vlm_submodule(model))
    parameter_names = {name for name, _ in model.named_parameters()}
    # HF state_dict includes tied aliases, while named_parameters omits them.
    # Require/load the canonical tensor exactly once and restore tying below.
    parameter_aliases = {
        name for name, _ in model.named_parameters(remove_duplicate=False)
        if name not in parameter_names
    }
    # ``init_empty_weights`` leaves buffers materialized, so preserve their
    # constructor-computed values (for example rotary frequencies) before
    # ``to_empty`` materializes the parameters and discards existing storage.
    buffer_dict = {name: buffer.clone() for name, buffer in model.named_buffers()}

    vlm_param_names = weight_loader.get_vlm_para_fullnames(model)
    logger.info_rank0(f"VLM submodule contains {len(vlm_param_names)} parameters.")

    model.to_empty(device=init_device)
    if load_full_checkpoint:
        # Loading a complete checkpoint is strict: every parameter must
        # be present and shape-compatible below. Initializing billions of
        # parameters immediately before overwriting all of them only increases
        # startup time and memory traffic.
        logger.info_rank0("Skipping parameter initialization for strict full-checkpoint loading.")
    else:
        initialize_model_weights(model)

    if load_full_checkpoint:
        logger.info_rank0("Loading the complete policy checkpoint.")
    elif load_vlm_only:
        logger.info_rank0("Loading Qwen weights and initializing the action policy.")

    # Load checkpoint weight iterators
    state_dict_iterators = _load_state_dict(weights_path)
    for state_dict_iterator in tqdm(
        state_dict_iterators, desc="Loading checkpoint shards", disable=int(os.getenv("LOCAL_RANK", "-1")) > 0
    ):
        for ckpt_key, tensor in state_dict_iterator:
            name = weight_loader.map_ckpt_key(
                ckpt_key,
                load_vlm_only=load_vlm_only,
                load_full_checkpoint=load_full_checkpoint,
            )
            if name is None or name in parameter_aliases:
                continue
            if name in buffer_dict:
                buffer_dict[name] = tensor.clone()
            elif name in parameter_names:
                parameter_names.remove(name)
                _dispatch_parameter(model, name, tensor, dtensor_factory)
            else:
                mode = "full-checkpoint loading" if load_full_checkpoint else "VLM-only initialization"
                error_msg = f"Unexpected key {name!r} found during {mode}."
                logger.info_rank0(error_msg)
                raise KeyError(error_msg)

        del state_dict_iterator
        empty_cache()

    for name, buffer in buffer_dict.items():
        _dispatch_buffer(model, name, buffer)

    missing_vlm_parameters = parameter_names & vlm_param_names
    if load_vlm_only and missing_vlm_parameters:
        raise KeyError(
            "The VLM source checkpoint did not initialize every Qwen3.5 parameter; "
            f"missing {sorted(missing_vlm_parameters)}."
        )
    if load_vlm_only and hasattr(model, "load_additional_pretrained_weights"):
        model.load_additional_pretrained_weights()
    if load_full_checkpoint and parameter_names:
        raise KeyError(
            "The complete checkpoint is missing model parameters: "
            f"{sorted(parameter_names)}."
        )
    elif parameter_names:
        logger.info_rank0(f"Initialized {len(parameter_names)} parameters not present in the source checkpoint.")

    # we should tie embeddings after loading weights because to_empty() leads to untied weights,
    # except for fsdp1 (custom init) and fsdp2 (swap tensor) contexts.
    _tie_module_embeddings(model)
    _tie_module_embeddings(weight_loader.get_vlm_submodule(model))


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
    """Save regular HF assets plus a portable Qwen3.5 backbone bundle.

    VLA's root ``config.json`` is the composite policy config, while
    model construction also needs Qwen3.5's original ``config.json``.  The
    latter therefore lives in ``vlm_backbone/`` together with the sole copy of
    the processor/tokenizer files.  The checkpoint root remains reserved for
    composite-model assets and weights.
    """

    from transformers import AutoConfig

    os.makedirs(output_dir, exist_ok=True)
    assets = tuple(model_assets)
    vla_config = next(
        (asset for asset in assets if getattr(asset, "model_type", None) == "vla"),
        None,
    )
    processor = next(
        (
            asset
            for asset in assets
            if asset is not vla_config and getattr(asset, "image_processor", None) is not None
        ),
        None,
    )

    for model_asset in assets:
        if vla_config is not None and model_asset is not vla_config:
            # VLA reserves the root for its composite config. The
            # processor is persisted once in the canonical backbone bundle.
            continue
        if hasattr(model_asset, "save_pretrained"):
            asset_to_save = model_asset
            if model_asset is vla_config:
                asset_to_save = deepcopy(model_asset)
                asset_to_save.vlm_path = None
                asset_to_save.tokenizer_path = VLM_BACKBONE_SUBDIR
                asset_to_save.vlm_backbone_subdir = VLM_BACKBONE_SUBDIR
                asset_to_save.action_policy_path = None
            asset_to_save.save_pretrained(output_dir)
        else:
            logger.warning(f"Model asset {model_asset} should implement `save_pretrained`.")

    if vla_config is None:
        return
    if processor is None:
        raise ValueError(
            "A VLA HF checkpoint must include its processor/tokenizer in `model_assets`."
        )

    vlm_source = (
        getattr(vla_config, "vlm_path", None)
        or getattr(vla_config, "tokenizer_path", None)
    )
    if not vlm_source:
        raise ValueError("Cannot package VLA without its Qwen3.5 config source.")

    qwen_config = AutoConfig.from_pretrained(vlm_source, trust_remote_code=False)
    if getattr(qwen_config, "model_type", None) != "qwen3_5":
        raise ValueError(
            "VLA checkpoint packaging requires a Qwen3.5 config; "
            f"got {getattr(qwen_config, 'model_type', None)!r} from {vlm_source!r}."
        )

    backbone_dir = os.path.join(output_dir, VLM_BACKBONE_SUBDIR)
    os.makedirs(backbone_dir, exist_ok=True)
    qwen_config.save_pretrained(backbone_dir)
    processor.save_pretrained(backbone_dir)
