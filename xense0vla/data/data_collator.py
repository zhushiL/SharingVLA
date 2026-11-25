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


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data._utils.collate import default_collate


@dataclass
class DataCollator(ABC):
    """
    Used in dataloader as a collate_fn.
    """

    @abstractmethod
    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        """
        Converts a list of features to batched tensor dict.
        """
        ...


class CollatePipeline:
    def __init__(self, data_collators: Optional[Union[Callable, List[Callable]]] = None):
        """
        Args:
            data_collators: a list of data collators or a single data collator
        """

        if not isinstance(data_collators, (list, tuple)):
            data_collators = [data_collators]
        self.data_collators = data_collators

    def __call__(self, batch: Sequence[Dict[str, Any]]):
        """
        process data batch through data collators.

        Args:
            batch: the original input data batch

        Returns:
            batch: the processed data batch

        """
        for data_collator in self.data_collators:
            batch = data_collator(batch)
        return batch


@dataclass
class MakeMicroBatchCollator(DataCollator):
    """
    Data collator to build micro batches, used in mapping dataloader.
    """

    num_micro_batch: int
    internal_data_collator: "DataCollator"

    def __call__(self, features: Sequence[Tuple[Dict[str, "torch.Tensor"]]]) -> List[Dict[str, "torch.Tensor"]]:
        if not features:
            raise ValueError("MakeMicroBatchCollator received an empty batch.")
        features_list = list(features)
        if isinstance(features_list[0], list):
            features_list = [item[0] for item in features_list]  # 1-to-N inverse transform

        num_chunks = min(max(1, int(self.num_micro_batch)), len(features_list))
        base, remainder = divmod(len(features_list), num_chunks)
        micro_batches = []
        offset = 0
        for chunk_idx in range(num_chunks):
            size = base + (1 if chunk_idx < remainder else 0)
            micro_batches.append(self.internal_data_collator(features_list[offset : offset + size]))
            offset += size

        return micro_batches


@dataclass
class PolicyDataCollator(DataCollator):
    """Stack fixed-shape robot observations, targets and masks into one batch."""

    def __call__(self, features):
        if not features or any(feature.keys() != features[0].keys() for feature in features):
            raise ValueError("Policy samples must be non-empty and have identical feature keys.")
        return default_collate(features)
