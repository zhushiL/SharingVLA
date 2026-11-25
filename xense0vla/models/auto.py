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


from typing import TYPE_CHECKING, Literal, Optional

import torch
from transformers import (
    AutoProcessor,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.utils import logging as transformers_logging

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from .module_utils import build_empty_policy, load_model_weights


if TYPE_CHECKING:
    from transformers import ProcessorMixin

logger = logging.get_logger(__name__)


def _load_processor_without_info_logs(processor_path: str) -> "ProcessorMixin":
    """Load a processor without dumping routine Transformers configuration."""

    previous_verbosity = transformers_logging.get_verbosity()
    try:
        transformers_logging.set_verbosity_warning()
        return AutoProcessor.from_pretrained(
            processor_path,
            padding_side="right",
            trust_remote_code=False,
        )
    finally:
        transformers_logging.set_verbosity(previous_verbosity)


def build_processor(processor_path: str) -> "ProcessorMixin":
    """Load the processor from the exact source selected by the caller."""

    return _load_processor_without_info_logs(processor_path)


def build_foundation_model(
    config: PretrainedConfig,
    weights_path: Optional[str] = None,
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16",
    init_device: Literal["cpu", "cuda", "meta"] = "cuda",
) -> "PreTrainedModel":
    """Construct an independent or composed policy and load its selected weights."""

    if getattr(config, "model_type", None) != "vla":
        raise TypeError(
            "build_foundation_model only supports VLA configs; "
            f"got model_type={getattr(config, 'model_type', None)!r}."
        )

    # The same fully qualified keys are used by standalone and composed policies.
    full_checkpoint = weights_path is not None
    if config.standalone_action_policy and not full_checkpoint:
        if bool(config.spatial_encoder_path) == bool(config.action_policy_path):
            raise ValueError("Provide either spatial_encoder_path for pretraining or a full action_policy_path.")
        weights_path = config.action_policy_path or config.spatial_encoder_path
        full_checkpoint = config.action_policy_path is not None
    elif not full_checkpoint:
        if not config.vlm_path or not config.action_policy_path:
            raise ValueError("Assembling a VLA requires vlm_path and action_policy_path.")
        weights_path = config.vlm_path

    logger.info_rank0(f"Initializing policy from {weights_path!r}.")
    model = build_empty_policy(config, getattr(torch, torch_dtype))
    if init_device == "meta":
        model._deferred_weight_load = {"weights_path": weights_path, "load_full_checkpoint": full_checkpoint}
    elif init_device != "cpu" or get_parallel_state().global_rank <= 0:
        load_model_weights(model, weights_path, init_device, load_full_checkpoint=full_checkpoint)
    return model
