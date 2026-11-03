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


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py

import inspect
from abc import ABC, abstractmethod
from typing import Optional

import torch.nn as nn
from transformers import PreTrainedModel
from transformers.initialization import no_init_weights

from ..utils import logging
from .module_utils import _tie_module_embeddings, init_empty_weights, load_model_weights


logger = logging.get_logger(__name__)


class BaseModelLoader(ABC):
    def __init__(self):
        pass

    def load_model(self, model_config, **kwargs):
        raise NotImplementedError


class CustomizedModelingLoader(BaseModelLoader):
    def __init__(self, model_cls: PreTrainedModel):
        super().__init__()
        self.model_cls = model_cls  # model class from code_path

    def _instantiate_model(self, init_kwargs: dict):
        config = init_kwargs["config"]
        constructor_kwargs = {"config": config}
        signature = inspect.signature(self.model_cls.__init__)
        for name, parameter in signature.parameters.items():
            if name in {"self", "config"}:
                continue
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            if name in init_kwargs:
                constructor_kwargs[name] = init_kwargs[name]
            elif hasattr(config, name):
                constructor_kwargs[name] = getattr(config, name)
            elif parameter.default is inspect.Parameter.empty:
                raise TypeError(
                    f"Cannot construct {self.model_cls.__name__}: required argument "
                    f"{name!r} is absent from both loader kwargs and the model config."
                )

        model = self.model_cls(**constructor_kwargs)
        torch_dtype = init_kwargs.get("torch_dtype")
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        # no_init_weights() skips Hugging Face's automatic tying. Establish
        # shared parameters before deferred/meta models are wrapped by FSDP.
        _tie_module_embeddings(model)
        loader_factory = getattr(type(model), "get_weight_loader", None)
        if loader_factory is not None:
            _tie_module_embeddings(loader_factory().get_vlm_submodule(model))
        return model

    def load_model(self, init_kwargs: dict, **kwargs):
        init_device = kwargs.pop("init_device", "cuda")
        weights_path = kwargs.pop("weights_path", None)
        empty_init = kwargs.pop("empty_init", False)
        load_vlm_only = bool(kwargs.pop("load_vlm_only", False))
        load_full_checkpoint = bool(kwargs.pop("load_full_checkpoint", False))

        if load_vlm_only == load_full_checkpoint:
            raise ValueError(
                "Exactly one strict initialization mode must be selected: "
                "Qwen-only initialization or complete checkpoint loading."
            )
        if weights_path is None:
            raise ValueError("Strict VLA initialization requires `weights_path`.")

        logger.info_rank0(
            f"Loading model from customized modeling.\n"
            f"init_device: {init_device}\n"
            f"empty_init: {empty_init}\n"
            f"weights_path: {weights_path}"
        )

        if load_vlm_only and not hasattr(self.model_cls, "get_weight_loader"):
            raise TypeError(
                f"{self.model_cls.__name__} cannot load a standalone VLM checkpoint "
                "because it does not implement get_weight_loader()."
            )
        with init_empty_weights(), no_init_weights():
            model = self._instantiate_model(init_kwargs)

        if not empty_init:
            load_model_weights(
                model,
                weights_path,
                init_device,
                load_vlm_only=load_vlm_only,
                load_full_checkpoint=load_full_checkpoint,
            )
        elif init_device == "meta":
            model._deferred_weight_load = {
                "weights_path": weights_path,
                "load_vlm_only": load_vlm_only,
                "load_full_checkpoint": load_full_checkpoint,
            }

        return model


def get_loader(model_config):
    from .configuration_vla import VLAConfig
    from .modeling_vla import VLAModel

    if not isinstance(model_config, VLAConfig):
        raise TypeError("Expected VLAConfig.")
    return CustomizedModelingLoader(model_cls=VLAModel)


class VLAWeightLoader(ABC):
    """
    Base class for VLA weight loaders
    """
    def get_vlm_para_fullnames(self, model)-> set[str]:
        vlm_mod = self.get_vlm_submodule(model)
        prefix = self.get_submodule_prefix(model, vlm_mod)
        return {f"{prefix}.{n}" for n, _ in vlm_mod.named_parameters()}


    def map_ckpt_key(self, key, load_vlm_only: bool, load_full_checkpoint: bool)-> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def get_vlm_submodule(self, model: nn.Module)-> nn.Module:
        ...

    def get_submodule_prefix(self, model, submodule)->str:
        for name, mod in model.named_modules():
            if mod is submodule:
                return name
        raise ValueError(f"Submodule {type(submodule).__name__} not found in model")
