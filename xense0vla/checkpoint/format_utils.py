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


from pathlib import Path

from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict


def ckpt_to_state_dict(checkpoint_path):
    """Read model weights from a DCP training checkpoint on one process."""
    model_dir = Path(checkpoint_path) / "model"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Missing DCP model directory: {model_dir}")
    state_dict = {}
    _load_state_dict(
        state_dict,
        storage_reader=FileSystemReader(model_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return state_dict["state"]["model"]
