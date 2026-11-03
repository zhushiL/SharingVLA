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


from typing import TYPE_CHECKING

from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from ..distributed.parallel_state import get_parallel_state
from .data_collator import CollatePipeline, MakeMicroBatchCollator, PolicyDataCollator

if TYPE_CHECKING:
    from torch.utils.data import Dataset


class DistributedDataloader(StatefulDataLoader):
    dataset: "Dataset"
    sampler: "StatefulDistributedSampler"

    def set_epoch(self, epoch: int) -> None:
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
        elif hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


def build_dataloader(
    dataset,
    micro_batch_size,
    global_batch_size,
    dataloader_batch_size,
    collate_fn=None,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=False,
    seed=0,
):
    """Random distributed sampling with resumable state and gradient accumulation."""
    parallel = get_parallel_state()
    num_micro_batch = global_batch_size // (micro_batch_size * parallel.dp_size)
    if collate_fn is None:
        collate_fn = PolicyDataCollator()
    if isinstance(collate_fn, list):
        collate_fn = CollatePipeline(collate_fn)
    collate_fn = MakeMicroBatchCollator(num_micro_batch, collate_fn)
    sampler = (
        None
        if isinstance(dataset, IterableDataset)
        else StatefulDistributedSampler(
            dataset,
            num_replicas=parallel.dp_size,
            rank=parallel.dp_rank,
            shuffle=True,
            seed=seed,
        )
    )
    return DistributedDataloader(
        dataset,
        batch_size=dataloader_batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers else None,
        persistent_workers=persistent_workers if num_workers else False,
    )
