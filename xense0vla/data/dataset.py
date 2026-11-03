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



from torch.utils.data import Dataset

from ..utils import logging
from .multi_dataset import MultiVLADataset
from .robot_dataset import VLADataset


logger = logging.get_logger(__name__)

try:
    import datasets.features.features as features

    _OLD_GENERATE_FROM_DICT = features.generate_from_dict

    def _new_generate_from_dict(obj):
        if isinstance(obj, dict) and obj.get("_type") == "List":
            obj["_type"] = "Sequence"
        return _OLD_GENERATE_FROM_DICT(obj)

    features.generate_from_dict = _new_generate_from_dict
except (ImportError, AttributeError):
    # If datasets or the function doesn't exist, do nothing.
    pass


def build_vla_dataset(
    dataset_config,
    model_config,
    config,
    processor,
    do_nomalize=True,
    return_item=False,
    disabled_image_features=False,
    return_raw_images=False,
) -> "Dataset":
    if model_config is None:
        assert processor is None, "processor and model_config can only be None when computing norm"
    data_name = dataset_config.data_name
    repo_id = dataset_config.train_path
    robot_config_root = dataset_config.robot_config_root
    chunk_size = dataset_config.chunk_size
    prompt_type = getattr(dataset_config, "prompt_type", "global")
    img_size = getattr(dataset_config, "img_size", 256)
    image_augment = bool(getattr(dataset_config, "image_augment", False))
    if data_name == "multi":
        dataset = MultiVLADataset(
            repo_file=repo_id,
            dataset_config=dataset_config,
            robot_config_root=robot_config_root,
            config=config,
            processor=processor,
            video_backend=getattr(dataset_config, "video_backend", "torchcodec"),
            image_size=(img_size, img_size),
            chunk_size=chunk_size,
            disabled_image_features=disabled_image_features,
            prompt_type=prompt_type,
            do_nomalize=do_nomalize,
            return_item=return_item,
            image_augment=image_augment,
            return_raw_images=return_raw_images,
        )
    else:
        dataset = VLADataset(
            repo_id=repo_id,
            data_name=data_name,
            robot_config_root=robot_config_root,
            dataset_config=dataset_config,
            config=config,
            processor=processor,
            video_backend=getattr(dataset_config, "video_backend", "torchcodec"),
            image_size=(img_size, img_size),
            do_nomalize=do_nomalize,
            return_item=return_item,
            chunk_size=chunk_size,
            disabled_image_features=disabled_image_features,
            image_augment=image_augment,
            return_raw_images=return_raw_images,
        )

    return dataset
