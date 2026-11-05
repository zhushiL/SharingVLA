# Copyright 2026 Robbyant Team and/or its affiliates
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
from pathlib import Path

import torch
from torch.utils.data import Dataset


from lerobot.datasets.lerobot_dataset import LeRobotDataset as BaseLeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from ..utils import logging
from .feature_transform import FeatureTransform
from .video_utils import decode_video_frames


logger = logging.get_logger(__name__)


def _to_relative_indices(dataset, query_indices):
    index_map = getattr(dataset, "_absolute_to_relative_idx", None)
    if index_map is None:
        return query_indices
    return [index_map[idx] for idx in query_indices]


def _get_task_name(tasks, task_idx):
    if hasattr(tasks, "iloc"):
        return tasks.iloc[task_idx].name
    return tasks[task_idx]


def _resolve_lerobot_location(repo_id):
    repo_path = Path(repo_id).expanduser()
    if repo_path.exists():
        return repo_path.name, repo_path
    return repo_id, None


class LeRobotDataset(BaseLeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        load_image: bool = True,
        camera_keys=None,
        load_task: bool = True,
        load_subtask: bool = True,
        **kwargs,
    ):
        super().__init__(repo_id, **kwargs)
        self.load_image = load_image
        self.load_task = load_task
        self.load_subtask = load_subtask
        self.selected_camera_keys = set(
            camera_keys if camera_keys is not None else (key for key in self.meta.camera_keys if "tactile" not in key)
        )
        if any("tactile" in key for key in self.selected_camera_keys):
            raise ValueError("Tactile cameras are not supported by this policy.")
        # Embedded image columns are lazy Arrow columns; dropping unused cameras
        # before indexing also prevents their image decoder from running.
        excluded = set(self.meta.camera_keys) - self.selected_camera_keys
        columns = getattr(self.hf_dataset, "column_names", [])
        if excluded.intersection(columns):
            self.hf_dataset = self.hf_dataset.select_columns([key for key in columns if key not in excluded])

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """
        Query dataset for indices across keys, skipping video keys.

        Tries column-first [key][indices] for speed, falls back to row-first.

        Args:
            query_indices: Dict mapping keys to index lists to retrieve

        Returns:
            Dict with stacked tensors of queried data (video keys excluded)
        """
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            # Map absolute indices to relative indices if needed
            relative_indices = _to_relative_indices(self, q_idx)
            result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            if vid_key not in self.selected_camera_keys:
                continue
            # LeRobot stores episode videos in shared files.
            ep = self.meta.episodes[ep_idx]
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            query_ts = [from_timestamp + ts for ts in query_ts]

            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    def __getitems__(self, indices):
        """Use the same camera whitelist and history queries for batched access."""
        return [self[index] for index in indices]

    def __getitem__(self, idx) -> dict:
        # Ensure dataset is loaded when we actually need to read from it
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(item["index"].item(), ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0 and self.load_image:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None and self.load_image:
            image_keys = self.selected_camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])
        # Add task as a string
        if self.load_task:
            task_idx = item["task_index"].item()
            item["task"] = _get_task_name(self.meta.tasks, task_idx)

        # Match LeRobot's generic subtask interface. Empty text represents an
        # unannotated frame and is masked by prepare_subtask, including EOS.
        if self.load_subtask and "subtask_index" in item:
            subtasks = getattr(self.meta, "subtasks", None)
            if subtasks is None:
                raise ValueError("subtask_index requires meta/subtasks.parquet and LeRobot 0.4.4.")
            subtask_idx = int(item["subtask_index"].item())
            if not 0 <= subtask_idx < len(subtasks):
                raise ValueError(f"Invalid subtask_index {subtask_idx}; expected [0, {len(subtasks)}).")
            item["subtask"] = subtasks.iloc[subtask_idx].name

        return item


class VLADataset(Dataset):
    def __init__(
        self,
        repo_id,
        data_name,
        dataset_config,
        robot_config_root,
        config=None,
        processor=None,
        video_backend="torchcodec",
        chunk_size=20,
        image_size=(224, 224),
        do_nomalize=True,
        return_item=False,
        disabled_image_features=False,
        feature_transform=None,
        transform=None,
        image_augment=False,
        return_raw_images=False,
    ):
        if do_nomalize and config is None:
            raise ValueError("VLADataset requires a model config; pass model.config via build_vla_dataset.")

        self.processor = processor
        self.config = config
        self.chunk_size = chunk_size
        self.data_name = data_name
        self.disabled_image_features = disabled_image_features
        self.return_raw_images = return_raw_images
        load_image = True if do_nomalize else False

        if feature_transform is None:
            robot_config = os.path.join(robot_config_root, f"{data_name}.yaml")
            # Normalization belongs to the run-level data configuration.
            norm_stats_path = getattr(dataset_config, "norm_stats_file", None) or None
            self.feature_transform = FeatureTransform(
                robot_config,
                dataset_config,
                self.config,
                processor,
                disabled_image_features,
                do_nomalize,
                chunk_size=chunk_size,
                return_item_befor_padding=return_item,
                image_augment=image_augment,
                return_raw_images=return_raw_images,
                norm_stats_path=norm_stats_path,
                image_size=image_size,
            )
        else:
            self.feature_transform = feature_transform

        self.action_features = self.feature_transform.actions
        self.state_features = self.feature_transform.states
        self.image_features = self.feature_transform.images

        lerobot_repo_id, lerobot_root = _resolve_lerobot_location(repo_id)
        self.dataset_meta = LeRobotDatasetMetadata(repo_id=lerobot_repo_id, root=lerobot_root)
        available_camera_keys = set(self.dataset_meta.camera_keys)
        required_camera_sources = (
            set() if disabled_image_features else set(self.feature_transform.org_features.get("images", []))
        )
        missing_camera_sources = sorted(required_camera_sources - available_camera_keys)
        if missing_camera_sources:
            raise ValueError(
                "The robot image mapping references source features missing "
                f"from dataset metadata: {missing_camera_sources}."
            )
        merged_delta = {**self.get_delta_timestamps(), **self.get_video_delta_timestamps()}
        load_subtask = (
            self.feature_transform.use_subtask_loss and not self.feature_transform.action_only and not return_item
        )

        self.dataset = LeRobotDataset(
            repo_id=lerobot_repo_id,
            root=lerobot_root,
            # Raw sources must remain at native resolution until logical
            # multi-source cameras have been composed by FeatureTransform.
            image_transforms=None,
            delta_timestamps=merged_delta,
            video_backend=video_backend,
            load_image=load_image and not disabled_image_features,
            camera_keys=self.feature_transform.org_model_images,
            load_task=not self.feature_transform.action_only,
            load_subtask=load_subtask,
        )

        if load_subtask:
            if (
                "subtask_index" not in self.dataset.hf_dataset.column_names
                or getattr(self.dataset.meta, "subtasks", None) is None
            ):
                raise ValueError(
                    "Subtask supervision requires LeRobot 0.4.4, a subtask_index data column and "
                    "meta/subtasks.parquet. See docs/data.md for the subtask format."
                )
        self.return_item = return_item
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def get_features(self):
        features = set()
        for feature_category, _features in self.feature_transform.org_features.items():
            if len(self.feature_transform.actions_convert_from_state) > 0 and feature_category == "actions":
                continue
            features.update(_features)
        features.update(self.feature_transform.feature_to_keep)
        features = [x for x in list(features) if x not in ["action_is_pad", "task", "subtask"]]
        return features

    def get_delta_timestamps(self, return_indices=False):
        delta_timestamps = {}
        fps = None if return_indices else self.dataset_meta.fps
        if not len(self.feature_transform.actions_convert_from_state) > 0:
            for action_feature in self.feature_transform.org_features["actions"]:
                delta_timestamps[action_feature] = [t / fps if fps else t for t in range(self.chunk_size)]
        else:
            for state_feature in self.feature_transform.org_features["states"]:
                delta_timestamps[state_feature] = [t / fps if fps else t for t in range(self.chunk_size + 1)]
        return delta_timestamps

    def get_video_delta_timestamps(self):
        """Sample the head at [t-offset, t], clamped by LeRobot within the episode."""
        if self.disabled_image_features or self.feature_transform.action_only:
            return {}
        fps = float(self.dataset_meta.fps)
        if fps <= 0:
            raise ValueError(f"Temporal head-camera sampling requires positive FPS, got {fps}.")
        offsets = [-self.feature_transform.head_history_offset / fps, 0.0]
        return dict.fromkeys(self.feature_transform.org_head_model_images, offsets)

    def check_lerobot_item(self, item):
        # if state or action is a 0-d tensor, convert it to 1-d tensor
        if not len(self.feature_transform.actions_convert_from_state) > 0:
            for action_feature in self.feature_transform.org_features["actions"]:
                if len(item[action_feature].shape) == 1:
                    item[action_feature] = item[action_feature].unsqueeze(-1)

            for state_feature in self.feature_transform.org_features["states"]:
                if len(item[state_feature].shape) == 0:
                    item[state_feature] = item[state_feature].unsqueeze(-1)
        else:
            for state_feature in self.feature_transform.org_features["states"]:
                if len(item[state_feature].shape) == 1:
                    item[state_feature] = item[state_feature].unsqueeze(-1)
        return item

    def transform_raw_item(self, raw_item):
        """Apply the existing per-sample VLA transform to a decoded raw row."""

        raw_item = self.check_lerobot_item(raw_item)
        item = self.feature_transform.apply(raw_item)
        if self.transform is not None:
            item = self.transform(
                item, raw_item, self.feature_transform.feature_config.images, self.feature_transform.key_mapping
            )
        return item

    def getitem(self, idx):
        return self.transform_raw_item(self.dataset[idx])

    def __getitems__(self, indices: list[int]):
        return [self[index] for index in indices]

    def __getitem__(self, idx):
        item = self.getitem(idx)
        return item
