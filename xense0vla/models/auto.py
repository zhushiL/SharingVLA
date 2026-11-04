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
from .loader import get_loader


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
    """
    Build VLA through its registered strict checkpoint loader.
    """

    if getattr(config, "model_type", None) != "vla":
        raise TypeError(
            "build_foundation_model only supports VLA configs; "
            f"got model_type={getattr(config, 'model_type', None)!r}."
        )

    full_checkpoint = weights_path is not None
    source_weights_path = weights_path if full_checkpoint else config.vlm_path
    if not source_weights_path:
        raise ValueError("Provide a complete weights_path or a Qwen vlm_path.")
    logger.info_rank0(f"Initializing from {source_weights_path!r} (complete checkpoint: {full_checkpoint}).")

    loader = get_loader(config)
    init_kwargs = {
        "config": config,
        "torch_dtype": getattr(torch, torch_dtype),
    }

    if (init_device == "cpu" and get_parallel_state().global_rank > 0) or init_device == "meta":
        empty_init = True
    else:
        empty_init = False
    model = loader.load_model(
        init_kwargs=init_kwargs,
        weights_path=source_weights_path,
        empty_init=empty_init,
        init_device=init_device,
        load_vlm_only=not full_checkpoint,
        load_full_checkpoint=full_checkpoint,
    )
    return model


class VisionProcessor:
    """Image-only processor bundle; never constructs or loads a tokenizer."""

    def __init__(self, image_processor):
        self.image_processor = image_processor

    def save_pretrained(self, path):
        return self.image_processor.save_pretrained(path)


def build_image_processor(processor_path: str, *, local_files_only: bool = False) -> VisionProcessor:
    from transformers import AutoImageProcessor

    return VisionProcessor(AutoImageProcessor.from_pretrained(
        processor_path, trust_remote_code=False, local_files_only=local_files_only,
    ))
