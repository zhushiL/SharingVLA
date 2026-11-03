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


"""Helper utils"""

import gc
import logging as builtin_logging
import os
import sys
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import psutil
import torch
import transformers
from torch import nn
from transformers import enable_full_determinism
from transformers import set_seed as set_seed_func

from . import logging
from .import_utils import is_torch_npu_available


if is_torch_npu_available():
    import torch_npu  # noqa: F401 # type: ignore
    from torch_npu.contrib import transfer_to_npu  # noqa: F401 # type: ignore


logger = logging.get_logger(__name__)

CACHE_DIR = os.path.expanduser(os.getenv("CACHE_DIR", os.path.join("~/.cache", "xense0vla")))


def set_seed(seed: int, full_determinism: bool = False) -> None:
    """
    Sets a manual seed on all devices.
    """
    if full_determinism:
        enable_full_determinism(seed)
    else:
        set_seed_func(seed)


def create_logger(name: Optional[str] = None) -> "logging._Logger":
    """
    Creates a pretty logger for the third-party program.
    """
    logger = builtin_logging.getLogger(name)
    formatter = builtin_logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
    )
    handler = builtin_logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(builtin_logging.INFO)
    logger.propagate = False
    return logger


def enable_third_party_logging() -> None:
    """
    Enables explicit logger of the third-party libraries.
    """
    transformers.logging.set_verbosity_info()
    transformers.logging.enable_default_handler()
    transformers.logging.enable_explicit_format()


def print_device_mem_info(prompt: str = "VRAM usage") -> None:
    """
    Logs VRAM info.
    """
    memory_allocated = torch.cuda.memory_allocated() / (1024**3)
    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    logger.info_rank0(f"{prompt}: cur {memory_allocated:.2f}GB, max {max_memory_allocated:.2f}GB.")


def print_cpu_memory_info():
    cpu_usage = psutil.cpu_percent(interval=1)  # 1-second interval
    logger.info_rank0(f"CPU Usage: {cpu_usage}%")

    memory_info = psutil.virtual_memory()
    logger.info_rank0(f"Total Memory: {memory_info.total / (1024**3):.2f} GB")
    logger.info_rank0(f"Available Memory: {memory_info.available / (1024**3):.2f} GB")
    logger.info_rank0(f"Used Memory: {memory_info.used / (1024**3):.2f} GB")
    logger.info_rank0(f"Memory Usage: {memory_info.percent}%")


def empty_cache() -> None:
    """
    Collects system memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_cache_dir(path: Optional[str] = None) -> str:
    """
    Returns the cache directory for the given path.
    """
    if path is None:
        return CACHE_DIR

    path = os.path.normpath(path)
    if not os.path.splitext(path)[-1]:  # is a dir
        path = os.path.join(path, "")

    path = os.path.split(os.path.dirname(path))[-1]
    return os.path.join(CACHE_DIR, path, "")  # must endswith os.path.sep


@lru_cache
def get_dtype_size(dtype: "torch.dtype") -> int:
    """
    Taken from https://github.com/huggingface/safetensors/blob/v0.4.5/bindings/python/py_src/safetensors/torch.py#L350
    """
    _float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    _float8_e5m2 = getattr(torch, "float8_e5m2", None)
    _SIZE = {
        torch.int64: 8,
        torch.float32: 4,
        torch.int32: 4,
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.int16: 2,
        torch.uint8: 1,
        torch.int8: 1,
        torch.bool: 1,
        torch.float64: 8,
        _float8_e4m3fn: 1,
        _float8_e5m2: 1,
    }
    return _SIZE[dtype]


def unwrap_model(model: "nn.Module") -> "nn.Module":
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Taken from: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/modeling_utils.py#L4808
    """
    if hasattr(model, "module"):
        return unwrap_model(getattr(model, "module"))
    else:
        return model


def print_example(example: Dict[str, "torch.Tensor"], rank: int) -> None:
    """
    Logs a single example to screen.
    """
    for key, value in example.items():
        if isinstance(value, torch.Tensor):
            logger.info(
                f"[rank {rank}]: {key}'s shape: {value.shape}, "
                f"device: {value.device}, dtype: {value.dtype}, numel: {value.numel()}"
            )
        else:
            logger.info(f"[rank {rank}]: {key}: {value}")
        # logger.info(f"[rank {rank}]: {key}'s shape: {value.shape}, device: {value.device},")


def dict2device(input_dict: dict):
    """
    Move a dict of Tensor to GPUs.
    """
    output_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            output_dict[k] = v.cuda()
        elif isinstance(v, dict):
            output_dict[k] = dict2device(v)
        else:
            output_dict[k] = v
    return output_dict


def make_list(item):
    if isinstance(item, List) or isinstance(item, np.ndarray):
        return item
    return [item]


def create_profiler(
    start_step: int, end_step: int, trace_dir: str, record_shapes: bool, profile_memory: bool, with_stack: bool
):
    """
    Creates a profiler to record the CPU and CUDA activities. Default export to trace.json.
    Profile steps in [start_step, end_step).

    Args:
        start_step (int): The step to start recording.
        end_step (int): The step to end recording.
        trace_dir (str): The path to save the profiling result.
        record_shapes (bool): Whether to record the shapes of the tensors.
        profile_memory (bool): Whether to profile the memory usage.
        with_stack (bool): Whether to include the stack trace.
    """

    def handler_fn(p):
        torch.profiler.tensorboard_trace_handler(trace_dir)(p)
        logger.info(f"Profiling result saved at {trace_dir}.")

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]

    warmup = 0 if start_step == 1 else 1
    wait = start_step - warmup - 1
    active = end_step - start_step
    logger.info(f"build profiler schedule - wait: {wait}, warmup: {warmup}, active: {active}.")
    profiler = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=1,
        ),
        on_trace_ready=handler_fn,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_modules=True,
        with_stack=with_stack,
    )
    return profiler
