import json
import numpy as np
import os
import sys
import random
import tempfile
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import torch.multiprocessing as mp
import torch
import torch.distributed as dist

from xense0vla.data.dataset_list import format_dataset_entry, get_all_tasks
from xense0vla.data.dataset import build_vla_dataset
from xense0vla.utils.normalize import (
    RunningStats,
    RunningStatsState,
)
from xense0vla.utils import helper
from xense0vla.utils.arguments import parse_args
import xense0vla.utils.normalize as normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xense0vla.training.arguments import PolicyTrainingArguments, VLADataArguments

if TYPE_CHECKING:
    from xense0vla.data.robot_dataset import LeRobotDataset

logger = helper.create_logger(__name__)

@dataclass
class NormComputeDataArguments(VLADataArguments):
    data_ratio_for_norm_compute: float = field(
        default=1.0,
        metadata={"help": "data ratio for norm compute."},
    )
    robot_name: str = field(
        default=None,
        metadata={"help": "robot name to compute norm."},
    )
    norm_path: str = field(
        default=None,
        metadata={"help": "Path to save norm stats."},
    )
    norm_merge_chunk_dim: bool = field(
        default=True,
        metadata={"help": "If merge chunk dim of action for norm compute."},
    )


@dataclass
class Arguments:
    data: "NormComputeDataArguments" = field(default_factory=NormComputeDataArguments)
    train: "PolicyTrainingArguments" = field(default_factory=PolicyTrainingArguments)


def collate_dict(batch_list):
    """
    Convert [{ 'a': t1, 'b': t2 }, { 'a': t3, 'b': t4 }]
    into { 'a': tensor([t1, t3]), 'b': tensor([t2, t4]) }
    """
    keys = batch_list[0].keys()
    batch = {}
    for key in keys:
        # If it is a Tensor, stack them together
        if isinstance(batch_list[0][key], torch.Tensor):
            batch[key] = torch.stack([item[key] for item in batch_list])
    return batch

# Worker logic - use an initializer to avoid passing the entire dataset with every task
_global_dataset = None

def init_worker(dataset):
    global _global_dataset
    _global_dataset = dataset

def worker_fn(indices):
    global _global_dataset
    # 1. Fetch each dict one by one
    samples = [_global_dataset[i] for i in indices]
    # 2. Collate the list of dicts into a single large dict (Batch)
    batch = collate_dict(samples)
    return batch

def get_batch_indices(target_ids, batch_size):
    return [target_ids[i:i + batch_size] for i in range(0, len(target_ids), batch_size)]


def _update_stats_from_batch(batch, stats, state_norm_keys, acton_norm_keys, delta_norm, norm_merge_chunk_dim):
    for key in state_norm_keys:
        values = np.asarray(batch[key])
        stats[key].update(values.reshape(-1, values.shape[-1]))
    valid_action_mask = None
    if "action_is_pad" in batch:
        valid_action_mask = ~np.asarray(batch["action_is_pad"], dtype=bool)
    for key in acton_norm_keys:
        raw_values = np.asarray(batch[key])
        if not delta_norm[key] or norm_merge_chunk_dim:
            if valid_action_mask is not None and raw_values.ndim == 3 and valid_action_mask.shape == raw_values.shape[:2]:
                values = raw_values[valid_action_mask]
            else:
                values = raw_values.reshape(-1, raw_values.shape[-1])
        else:
            values = raw_values.reshape(raw_values.shape[0], -1)
        if values.shape[0] > 0:
            stats[key].update(values)


def compute_norm(dataset, batch_size, stats, state_norm_keys, acton_norm_keys, delta_norm, ratio,
                 rank=0, world_size=1, num_workers=8, norm_merge_chunk_dim=False):
    if ratio < 1:
        num_step = int(len(dataset)*ratio)
        # Fix the seed so every rank samples the same global subset; slicing it
        # afterwards yields a deterministic, disjoint partition.
        random.seed(42)
        data_ids = random.sample(range(len(dataset)), num_step)
    else:
        data_ids = list(range(len(dataset)))

    # Strided slicing by rank offsets the imbalance caused by differing sub-dataset sizes.
    data_ids = data_ids[rank::world_size]

    all_batch_indices = get_batch_indices(data_ids, batch_size)
    total_batches = len(all_batch_indices) # Total number of batches

    if num_workers <= 0:
        init_worker(dataset)
        pbar = tqdm(
            (worker_fn(indices) for indices in all_batch_indices),
            total=total_batches,
            unit="batch",
            ncols=100,
            disable=(rank != 0),
            desc=f"rank{rank}",
        )
        for batch in pbar:
            _update_stats_from_batch(batch, stats, state_norm_keys, acton_norm_keys, delta_norm, norm_merge_chunk_dim)
        del dataset
        return

    mp.set_start_method('fork', force=True)
    # With the initializer each worker initializes the dataset only once, avoiding
    # passing it along with every task.
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(dataset,)) as pool:
        results_generator = pool.imap_unordered(worker_fn, all_batch_indices)

        pbar = tqdm(
            results_generator,
            total=total_batches,
            unit="batch",
            ncols=100,
            disable=(rank != 0),
            desc=f"rank{rank}",
        )
        for batch in pbar:
            _update_stats_from_batch(batch, stats, state_norm_keys, acton_norm_keys, delta_norm, norm_merge_chunk_dim)

    del pool
    del dataset

def get_norm_stats(stats, delta_norm, chunk_size, norm_merge_chunk_dim=False):
    assert stats is not None
    norm_stats = {}
    for key, state in stats.items():
        _chunk_size = chunk_size if (key in delta_norm and delta_norm[key]==True) and not norm_merge_chunk_dim else None
        norm_stats[key] = state.get_statistics(chunk_size=_chunk_size)
    return norm_stats


def _init_dataset_worker(
    args, data_names
) -> 'LeRobotDataset':
    
    args.data.chunk_size = args.train.chunk_size
    dataset = build_vla_dataset(dataset_config=args.data, 
                                model_config=None, 
                                config=None, 
                                processor=None, 
                                do_nomalize = False,
                                return_item = True,
                                disabled_image_features = True)
    
    return dataset

if __name__ == "__main__":

    args = parse_args(Arguments)
    if not args.data.norm_path:
        raise ValueError("data.norm_path must be specified before computing normalization statistics.")

    # Use CPU gloo backend so multiprocessing fork workers never inherit a CUDA/NCCL context.
    if args.train.world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="gloo", timeout=timedelta(hours=2))
    rank = args.train.global_rank
    world_size = args.train.world_size

    logger.info(f"Process rank: {rank}, world size: {world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))

    logger.info_rank0("Prepare data")
    stats = None

    robot_name = args.data.robot_name.split(',') if args.data.robot_name is not None else None
    
    if args.data.data_name == 'multi':
        data_names, repo_ids, sampling_ratios = get_all_tasks(
            args.data.train_path, robot_name, with_sampling_ratios=True
        )
        if robot_name is None:
            assert len(set(data_names)) == 1
        else:
            for data_name in set(data_names):
                assert data_name in robot_name
    else:
        data_names, repo_ids = [args.data.data_name], [args.data.train_path]
        sampling_ratios = [1.0]
        args.data.data_name = 'multi'

    tmp_filename = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f"norm_datasets_rank{rank}_pid{os.getpid()}_",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as f:
            tmp_filename = f.name
            for robot, task, sampling_ratio in zip(data_names, repo_ids, sampling_ratios):
                f.write(format_dataset_entry(robot, task, sampling_ratio) + "\n")
        args.data.train_path = tmp_filename
        dataset = _init_dataset_worker(args, data_names)
    finally:
        if tmp_filename and os.path.exists(tmp_filename):
            os.remove(tmp_filename)
    if rank == 0:
        print(f"===========\nProcessing {len(dataset._datasets)} lerobot datasets\n===========")
    assert len(list(set([' '.join(_datasets.state_features+_datasets.action_features) for _datasets in dataset._datasets]))) == 1
    first_sub = dataset._datasets[0].feature_transform.action_subtract_state
    first_rel = dataset._datasets[0].feature_transform.action_relative_type
    for sub_dataset in dataset._datasets[1:]:
        if (
            sub_dataset.feature_transform.action_subtract_state != first_sub
            or sub_dataset.feature_transform.action_relative_type != first_rel
        ):
            raise ValueError(
                "All datasets in a combined normalization run must share identical "
                "action_subtract_state and action_relative_type settings."
            )

    state_norm_keys = dataset._datasets[0].state_features
    acton_norm_keys = dataset._datasets[0].action_features
    delta_norm = first_sub
    stats = {key: normalize.RunningStats() for key in acton_norm_keys+state_norm_keys}
    chunk_size = args.data.chunk_size
    
    ratio = args.data.data_ratio_for_norm_compute
    compute_norm(dataset, args.train.micro_batch_size, stats, state_norm_keys, acton_norm_keys,
                 delta_norm, ratio=ratio, rank=rank, world_size=world_size,
                 num_workers=args.data.num_workers, norm_merge_chunk_dim=args.data.norm_merge_chunk_dim)

    # Cross-rank merge: each rank serializes its local stats and all_gather_object's
    # them to all ranks; rank0 performs the actual merge and persists the result.
    if world_size > 1:
        local_state = {
            k: (v.get_state().model_dump() if v._count > 0 else None)
            for k, v in stats.items()
        }
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_state)
        if rank == 0:
            merged = {}
            for key in stats.keys():
                objs = []
                for shard in gathered:
                    if shard is None or shard.get(key) is None:
                        continue
                    objs.append(RunningStats.from_state(RunningStatsState(**shard[key])))
                if not objs:
                    raise RuntimeError(f"No rank produced any data for key={key!r}")
                merged[key] = RunningStats.merge(objs)
            stats = merged
        dist.barrier()

    if rank == 0:
        norm_stats = get_norm_stats(stats, delta_norm, chunk_size, args.data.norm_merge_chunk_dim)
        output_path = Path(args.data.norm_path)
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats, stats[state_norm_keys[0]]._count)

    if world_size > 1:
        dist.destroy_process_group()
