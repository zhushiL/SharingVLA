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
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from ..utils.logging import get_logger
from pathlib import Path

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
)
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
logger = get_logger(__name__)

_EXTRA_STATE_FORMAT = "extra_state_rank_{}.pt"
_MODEL_DIR = "model"
_OPTIMIZER_DIR = "optimizer"
_EXTRA_STATE_DIR = "extra_state"


class ModelState(Stateful):
    """Wrap a model for torch distributed checkpointing."""

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def state_dict(self):
        return {"model": get_model_state_dict(model=self.model)}

    def load_state_dict(self, state_dict):
        set_model_state_dict(model=self.model, model_state_dict=state_dict["model"])


class OptimizerState(Stateful):
    """Wrap an optimizer for torch distributed checkpointing."""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        optimizers = getattr(self.optimizer, "optimizers", self.optimizer)
        optimizer_state_dict = get_optimizer_state_dict(model=self.model, optimizers=optimizers)
        return {"optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        optimizers = getattr(self.optimizer, "optimizers", self.optimizer)
        set_optimizer_state_dict(
            model=self.model,
            optimizers=optimizers,
            optim_state_dict=state_dict["optim"],
        )


def build_checkpointer(dist_backend: str = "fsdp2"):
    """Return the DCP implementation for the configured parallel backend."""
    if dist_backend not in {"ddp", "fsdp1", "fsdp2"}:
        raise ValueError(f"Unsupported distributed backend: {dist_backend}")
    return DistributedCheckpointer


class CheckpointerBase(ABC):
    """Base class for checkpointer"""

    @abstractmethod
    def save(
        cls,
        path: str,
        state: Dict[str, Any],
    ):
        return

    @abstractmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
        allow_partial_load: bool = False,
    ):
        return


class DistributedCheckpointer(CheckpointerBase):
    """
    Distributed checkpointer for torch.distributed.checkpoint
    """

    @classmethod
    def save(
        cls,
        path: str,
        state: Dict[str, Any],
        global_steps: int = None,
        save_async=False,
    ) -> None:
        """
        save training state to distributed checkpoint

        args:
            path: path to save checkpoint
            state: state to save
            global_steps: global steps
            save_async: whether to save asynchronously
        return:
            None
        """

        checkpoint_dir = f"{path}/global_step_{global_steps}" if global_steps else path
        os.makedirs(checkpoint_dir, exist_ok=True)

        if "model" not in state:
            raise ValueError("Model must be provided to save a distributed checkpoint.")

        if save_async:
            model_dir = os.path.join(checkpoint_dir, _MODEL_DIR)
            dcp.async_save(
                state_dict={"state": ModelState(state["model"])},
                storage_writer=FileSystemWriter(
                    model_dir,
                    thread_count=16,
                    single_file_per_rank=True,
                    sync_files=False,
                ),
            )
            if "optimizer" in state:
                optimizer_dir = os.path.join(checkpoint_dir, _OPTIMIZER_DIR)
                dcp.async_save(
                    state_dict={"state": OptimizerState(model=state["model"], optimizer=state["optimizer"])},
                    storage_writer=FileSystemWriter(
                        optimizer_dir,
                        thread_count=16,
                        single_file_per_rank=True,
                        sync_files=False,
                    ),
                )
        else:
            def safe_create_writer(output_dir):
                tmp_path = Path(output_dir) / ".metadata.tmp"
                if tmp_path.exists():
                    print(f"Warning: removing existing tmp file: {tmp_path}")
                    tmp_path.unlink()  # remove .metadata.tmp
                return FileSystemWriter(
                    output_dir,
                    thread_count=16,
                    single_file_per_rank=True,
                    sync_files=False,
                )
            model_dir = os.path.join(checkpoint_dir, _MODEL_DIR)
            storage_writer = safe_create_writer(model_dir)
            dcp.save(
                state_dict={"state": ModelState(state["model"])},
                storage_writer=storage_writer,
            )
            if "optimizer" in state:
                optimizer_dir = os.path.join(checkpoint_dir, _OPTIMIZER_DIR)
                dcp.save(
                    state_dict={"state": OptimizerState(model=state["model"], optimizer=state["optimizer"])},
                    storage_writer=FileSystemWriter(
                        optimizer_dir,
                        thread_count=16,
                        single_file_per_rank=True,
                        sync_files=False,
                    ),
                )
                # dist.barrier()

        if "extra_state" in state:
            extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
            os.makedirs(extra_state_dir, exist_ok=True)
            extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(dist.get_rank()))
            torch.save(
                state["extra_state"],
                extra_state_path,
            )

        logger.info_rank0(f"Saved checkpoint to {checkpoint_dir}")

    @classmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
        process_group=None,
        allow_partial_load: bool = False,
    ) -> Dict[str, Any]:
        """
        load training state from distributed checkpoint
        args:
            path: path to load checkpoint
            state: state to load, "model" are required,  "optimizer" and "extra_state" are optional
            allow_partial_load: if True, skip missing keys in checkpoint (useful for model structure changes)

        return:
            state: state loaded
        """
        checkpoint_dir = path

        if state is None:
            raise ValueError("State dict must be provided to load a distributed checkpoint.")

        if "model" not in state:
            raise ValueError("Model must be provided to load a distributed checkpoint.")

        model_planner = DefaultLoadPlanner(allow_partial_load=True) if allow_partial_load else None
        optimizer_planner = DefaultLoadPlanner(allow_partial_load=True) if allow_partial_load else None

        if "optimizer" in state:
            model_dir = os.path.join(checkpoint_dir, _MODEL_DIR)
            dcp.load(
                state_dict={"state": ModelState(state["model"])},
                storage_reader=FileSystemReader(model_dir),
                planner=model_planner,
                process_group=process_group,
            )

            optimizer_dir = os.path.join(checkpoint_dir, _OPTIMIZER_DIR)
            try:
                dcp.load(
                    state_dict={"state": OptimizerState(model=state["model"], optimizer=state["optimizer"])},
                    storage_reader=FileSystemReader(optimizer_dir),
                    planner=optimizer_planner,
                    process_group=process_group,
                )
            except Exception as exc:
                if not allow_partial_load:
                    raise RuntimeError(
                        f"Failed to load optimizer state from {optimizer_dir}."
                    ) from exc
                logger.warning_rank0(
                    "Partial checkpoint loading is enabled; optimizer state from "
                    f"{optimizer_dir} was skipped after load failure: {exc!r}. "
                    "Training will continue without restored optimizer buffers; "
                    "this is not an exact resume."
                )
        else:
            model_dir = os.path.join(checkpoint_dir, _MODEL_DIR)
            dcp.load(
                state_dict={"state": ModelState(state["model"])},
                storage_reader=FileSystemReader(model_dir),
                planner=model_planner,
                process_group=process_group,
            )

        if "extra_state" in state:
            extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
            os.makedirs(extra_state_dir, exist_ok=True)
            extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(dist.get_rank()))
            state["extra_state"] = torch.load(
                extra_state_path,
            )

        logger.info_rank0(f"Loaded checkpoint from {checkpoint_dir}")

        return state
