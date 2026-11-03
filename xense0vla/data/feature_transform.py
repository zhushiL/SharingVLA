import ast
import json
import os
from collections import OrderedDict
from typing import List

import torch
import torch.nn.functional as F
import yaml
from pydantic import BaseModel

from ..utils import logging as logging_utils
from .ee_pose_transform import (
    _is_pose_relative_type,
    absolute_pose,
    relative_pose,
)
from .transforms import (
    Normalizer,
    concatenate_visuals,
    prepare_action,
    prepare_images,
    prepare_language,
    prepare_state,
    prepare_subtask,
    resize_visual,
)


logger = logging_utils.get_logger(__name__)


class FeatureInfo(BaseModel):
    joints: List[str] | None = None
    images: List[str] | None = None
    joints_max_dim: dict | None = None

    def update_info(self, data_config):
        joints_info = data_config.joints
        cameras = [camera for camera in (data_config.cameras or []) if "tactile" not in camera]
        self.images = ["observation.images." + image for image in cameras]

        joints = []
        joints_max_dim = {}
        for s in joints_info:
            joint_info = ast.literal_eval(s) if isinstance(s, str) else dict(s)
            joint = next(iter(joint_info.keys()))
            if joint_info[joint] == 0:
                continue

            joints.append(joint)
            joints_max_dim.update(joint_info)
        self.joints = joints
        self.joints_max_dim = joints_max_dim


class FeatureTransform:
    def __init__(
        self,
        robot_config_path,
        data_config,
        model_config,
        processor,
        disabled_image_features=False,
        do_nomalize=True,
        chunk_size=20,
        return_item_befor_padding=False,
        norm_stats_path=None,
        return_raw_images=False,
        image_augment=False,
        image_size=(256, 256),
    ):

        assert os.path.exists(robot_config_path), f"{robot_config_path} does not exist."
        with open(robot_config_path, "r") as f:
            robot_config = yaml.safe_load(f)
        f.close()

        if do_nomalize and not norm_stats_path:
            raise ValueError(
                "Normalization is enabled but no norm_stats_path was provided. "
                "Set data.norm_stats_file explicitly; robot YAML norm_stats entries "
                "are not used."
            )
        if "norm_stats" in robot_config:
            raise ValueError(
                "Robot YAML norm_stats is no longer supported. Move the path to "
                "data.norm_stats_file and remove the robot-level field."
            )

        self.feature_config = FeatureInfo()
        if getattr(data_config, "joints", None) is not None:
            self.feature_config.update_info(data_config)

        def config_value(name, default=None):
            if isinstance(model_config, dict):
                return model_config.get(name, default)
            return getattr(model_config, name, default)

        self.action_only = bool(config_value("standalone_action_policy", False))
        self.use_subtask_loss = bool(config_value("use_subtask_loss", False))
        self.head_history_offset = int(config_value("head_history_offset", 30))
        if self.head_history_offset < 0:
            raise ValueError("head_history_offset must be non-negative.")
        configured_images = self.feature_config.images or [
            "observation.images.camera_top",
            "observation.images.camera_wrist_left",
            "observation.images.camera_wrist_right",
        ]
        # Camera order is structural: head, left wrist, right wrist. Standalone
        # policies neither query nor decode the head camera.
        if self.action_only and len(configured_images) == 2:
            configured_images = [None, *configured_images]
        if len(configured_images) != 3:
            raise ValueError("Expected [left wrist, right wrist] for action policy or [head, left wrist, right wrist] for VLA.")
        self.head_model_image = None if self.action_only else configured_images[0]
        self.model_images = list(configured_images[1:] if self.action_only else configured_images)
        active_images = set(self.model_images)
        robot_config["images"] = [
            entry
            for entry in robot_config.get("images", [])
            if (entry if isinstance(entry, str) else next(iter(entry))) in active_images
        ]
        if not return_item_befor_padding:
            self.check_robot_config(robot_config)

        self.model_config = model_config
        self.tokenizer = getattr(processor, "tokenizer", None)
        self.processor = processor

        self.chunk_size = chunk_size
        self.return_item_befor_padding = return_item_befor_padding

        # disabled_image_features: keep the image features or not when getting lerobot item
        self.disabled_image_features = disabled_image_features
        self.return_raw_images = return_raw_images
        self.image_size = image_size

        if not disabled_image_features:
            self.image_augment = image_augment

        # keep the self.feature_to_keep in lerobot item when convert to new item
        self.feature_to_keep = {
            "timestamp",
            "frame_index",
            "episode_index",
            "task_index",
            "action_is_pad",
            "task",
            "subtask",
        }

        target_features = {"states": [], "actions": [], "images": []}
        org_features = {"states": set(), "actions": set(), "images": set()}
        self.get_feature_mapping(robot_config, target_features, org_features)
        self.states = target_features["states"]
        self.actions = target_features["actions"]
        self.images = target_features["images"]
        if not self.disabled_image_features:
            missing = set(self.model_images) - set(self.images)
            if missing:
                raise ValueError(f"Robot configuration is missing required cameras: {sorted(missing)}")
        self.org_features = org_features
        self.org_model_images = self.get_origin_image_features(self.model_images)
        self.org_head_model_images = self.get_origin_image_features(
            [self.head_model_image] if self.head_model_image is not None else []
        )
        self.output_image_keys = (
            self.model_images
            if self.action_only
            else [self.head_model_image + ".history", self.head_model_image, *self.model_images[1:]]
        )

        self.normalizer = self.get_normalizer(norm_stats_path, do_nomalize, data_config)

    def get_origin_image_features(self, target_images):
        """Resolve native image keys needed by the cameras sent to the model."""
        origin_images = []
        for target_image in target_images:
            if target_image not in self.images:
                # Unified multi-robot configs may reserve a model-camera slot
                # that this robot does not provide; prepare_images zero-pads it.
                continue
            convert_info = self.key_mapping.get(target_image)
            if convert_info is None:
                sources = [target_image]
            else:
                origin_keys = convert_info["origin_keys"]
                if isinstance(origin_keys, str):
                    sources = [origin_keys]
                elif isinstance(origin_keys, list):
                    sources = origin_keys
                else:
                    raise ValueError(f"Unsupported image origin mapping for {target_image!r}: {origin_keys!r}")
            for source in sources:
                if source not in origin_images:
                    origin_images.append(source)
        return origin_images

    def get_normalizer(self, norm_stats_path, do_nomalize, data_config):

        if not do_nomalize:
            return None

        action_state_norm_type = {k: v for d in data_config.norm_type for k, v in ast.literal_eval(d).items()}

        if not norm_stats_path:
            raise ValueError("Normalization is enabled but no norm_stats_path was provided.")
        norm_type = {}
        for feature in self.actions:
            base_name = feature.split("action.")[-1]
            assert base_name in action_state_norm_type, f"{feature} does not have predefined norm type."
            norm_type[feature] = action_state_norm_type[base_name]

        for feature in self.states:
            base_name = feature.split("observation.state.")[-1]
            assert base_name in action_state_norm_type, f"{feature} does not have predefined norm type."
            norm_type[feature] = action_state_norm_type[base_name]

        for feature in self.images:
            norm_type[feature] = "identity"

        with open(norm_stats_path) as f:
            norm_stats = json.load(f)
        f.close()

        normalizer = Normalizer(
            norm_stats=norm_stats["norm_stats"],
            norm_type=norm_type,
        )

        return normalizer

    def check_robot_config(self, robot_config):

        for feature_category, features_convert_info in robot_config.items():
            assert isinstance(features_convert_info, list)
            for feature_convert_info in features_convert_info:
                if isinstance(feature_convert_info, dict):
                    assert len(feature_convert_info.keys()) == 1

                if feature_category == "actions":
                    assert isinstance(feature_convert_info, dict)
                    action_feature = list(feature_convert_info.keys())[0].split("action.")[-1]
                    if action_feature not in self.feature_config.joints:
                        raise ValueError(
                            f"{action_feature} in the robot config is not included among the predefined features in the training config: {self.feature_config.joints}"
                        )

                elif feature_category == "states":
                    assert isinstance(feature_convert_info, dict) or isinstance(feature_convert_info, str)
                    if isinstance(feature_convert_info, dict):
                        state_feature = list(feature_convert_info.keys())[0]
                    else:
                        state_feature = feature_convert_info
                    state_feature = state_feature.split("observation.state.")[-1]
                    if state_feature not in self.feature_config.joints:
                        raise ValueError(
                            f"{state_feature} in the robot config is not included among the predefined features in the training config: {self.feature_config.joints}"
                        )

                elif feature_category == "images":
                    assert isinstance(feature_convert_info, dict) or isinstance(feature_convert_info, str)
                    if isinstance(feature_convert_info, dict):
                        image_feature = list(feature_convert_info.keys())[0]
                    else:
                        image_feature = feature_convert_info
                    if image_feature not in self.feature_config.images:
                        raise ValueError(
                            f"{image_feature} in the robot config is not included among the predefined features in the training config: {self.feature_config.images}"
                        )

    def get_feature_mapping(self, robot_config, target_features, org_features):
        to_convert_features = {}
        reverse_convert_features = {}
        action_subtract_state = {}
        action_relative_type = {}
        actions_convert_from_state = set()
        for feature_category, features_convert_info in robot_config.items():
            assert isinstance(features_convert_info, list)

            for feature_convert_info in features_convert_info:
                if isinstance(feature_convert_info, str):
                    self.feature_to_keep.add(feature_convert_info)
                    target_features[feature_category].append(feature_convert_info)
                    org_features[feature_category].add(feature_convert_info)

                elif isinstance(feature_convert_info, dict):
                    target_feature = next(iter(feature_convert_info.keys()))
                    target_features[feature_category].append(target_feature)

                    if feature_category == "actions":
                        assert isinstance(feature_convert_info, dict)
                        assert "subtract_state" in feature_convert_info[target_feature]
                        action_subtract_state[target_feature] = feature_convert_info[target_feature].pop(
                            "subtract_state"
                        )
                        if "velocity" in target_feature or "effector.position" in target_feature:
                            assert not action_subtract_state[target_feature], (
                                f"{target_feature} cannot be subtracted from state"
                            )

                        action_relative_type[target_feature] = feature_convert_info[target_feature].pop(
                            "relative_type", "local"
                        )
                        if action_relative_type[target_feature] not in {None, "world", "local"}:
                            raise ValueError("relative_type must be 'world', 'local' or null.")
                        if_convert_from_state = feature_convert_info[target_feature].pop("convert_from_state", False)
                        if if_convert_from_state:
                            actions_convert_from_state.add(target_feature)

                        if "origin_keys" not in feature_convert_info[target_feature] and not if_convert_from_state:
                            self.feature_to_keep.add(target_feature)

                    if "origin_keys" in feature_convert_info[target_feature]:
                        if isinstance(feature_convert_info[target_feature]["origin_keys"], list):
                            origin_key_items = feature_convert_info[target_feature]["origin_keys"]
                            if feature_category == "images":
                                if len(origin_key_items) < 2 or not all(
                                    isinstance(origin_key, str) for origin_key in origin_key_items
                                ):
                                    raise ValueError(
                                        f"{target_feature}.origin_keys must contain at least two image-key strings."
                                    )
                                combine = feature_convert_info[target_feature].get("combine", "concat_height")
                                if combine != "concat_height":
                                    raise ValueError(f"Unsupported combine mode {combine!r} for {target_feature}.")
                                to_convert_features[target_feature] = feature_convert_info[target_feature]
                                org_features["images"].update(origin_key_items)
                                continue

                            ordered_origin_keys = OrderedDict()
                            for item in origin_key_items:
                                if not isinstance(item, dict):
                                    raise ValueError(f"{target_feature}.origin_keys must be a list of slice mappings.")
                                for k, v in item.items():
                                    while k in ordered_origin_keys:
                                        k = k + "*"
                                    ordered_origin_keys[k] = v

                            feature_convert_info[target_feature]["origin_keys"] = ordered_origin_keys
                            to_convert_features.update(feature_convert_info)

                            if feature_category in ["actions", "states"]:
                                org_features[feature_category].update(
                                    [k.split("*")[0] for k in ordered_origin_keys.keys()]
                                )

                            target_start_id = 0
                            for org_key, info in ordered_origin_keys.items():
                                org_info = info.copy()
                                org_key = org_key.split("*")[0]
                                if org_key not in reverse_convert_features:
                                    reverse_convert_features[org_key] = []

                                if "start" in org_info:
                                    org_info["target_key"] = target_feature
                                    org_info["target_start"] = target_start_id
                                    org_info["target_end"] = target_start_id + org_info["end"] - org_info["start"]
                                    target_start_id = org_info["target_end"]
                                else:
                                    org_info["target_key"] = target_feature
                                reverse_convert_features[org_key].append(org_info)

                        if isinstance(feature_convert_info[target_feature]["origin_keys"], str):
                            to_convert_features[target_feature] = feature_convert_info[target_feature]
                            if feature_category in ["actions", "states"]:
                                reverse_convert_features[feature_convert_info[target_feature]["origin_keys"]] = {
                                    "target_key": target_feature
                                }
                            org_features[feature_category].add(feature_convert_info[target_feature]["origin_keys"])

        to_convert_state = {}
        if len(actions_convert_from_state) > 0:
            for action in list(actions_convert_from_state):
                assert action.replace("action.", "observation.state.") in target_features["states"], (
                    f"{action} can not converted from {action.replace('action.', 'observation.state.')}"
                )
                to_convert_state[action] = action.replace("action.", "observation.state.")
        self.actions_convert_from_state = to_convert_state

        if len(self.actions_convert_from_state) > 0:
            # To keep time alignment, if any action is converted from state, all actions must be converted from state
            assert len(self.actions_convert_from_state) == len(target_features["actions"])

        self.action_subtract_state = action_subtract_state
        self.action_relative_type = action_relative_type
        for feature_category, feature in org_features.items():
            if feature_category == "actions":
                if len(feature) == 0 and len(self.actions_convert_from_state) > 0:
                    org_features["actions"] == []
                    continue
            if len(feature) == 0:
                org_features[feature_category] = target_features[feature_category]
            else:
                org_features[feature_category] = list(feature)

        self.key_mapping = to_convert_features
        self.key_reverse_mapping = reverse_convert_features

    def convert_features(self, item, w_action):
        out_item = {}
        for target_key, convert_info in self.key_mapping.items():
            if self.disabled_image_features and target_key in self.images:
                continue
            if not w_action and "action" in target_key:
                continue
            if isinstance(convert_info["origin_keys"], str):
                origin_key = convert_info["origin_keys"]
                if origin_key not in item:
                    raise KeyError(f"Missing source feature {origin_key!r} required by {target_key!r}.")
                out_item[target_key] = item[origin_key]
                continue

            if target_key in self.images and isinstance(convert_info["origin_keys"], list):
                origin_keys = convert_info["origin_keys"]
                missing_keys = [origin_key for origin_key in origin_keys if origin_key not in item]
                if missing_keys:
                    raise KeyError(f"Missing source images {missing_keys!r} required by {target_key!r}.")
                out_item[target_key] = concatenate_visuals(
                    [item[origin_key] for origin_key in origin_keys],
                    mode=convert_info.get("combine", "concat_height"),
                )
                continue

            assert isinstance(convert_info["origin_keys"], OrderedDict)
            concat_list = []
            convert_success = True
            for origin_key, origin_info in convert_info["origin_keys"].items():
                origin_key = origin_key.split("*")[0]
                if origin_key not in item:
                    convert_success = False
                    break
                origin_data = item.get(origin_key)[..., origin_info["start"] : origin_info["end"]]
                concat_list.append(origin_data)
            if convert_success:
                out_item[target_key] = torch.cat(concat_list, dim=-1)
            del concat_list

        for feature in self.feature_to_keep:
            if feature in item:
                out_item[feature] = item[feature]

        if not self.disabled_image_features:
            for image_feature in self.images:
                if image_feature in out_item:
                    # Multi-source cameras have already been composed above;
                    # resize the resulting logical view exactly once.
                    out_item[image_feature] = resize_visual(out_item[image_feature], self.image_size)

        if len(self.actions_convert_from_state) > 0 and w_action:
            for action_feature in self.actions:
                state_feature = self.actions_convert_from_state[action_feature]
                assert state_feature in out_item and len(out_item[state_feature].shape) == 2
                out_item[action_feature] = out_item[state_feature][1:].clone()
                out_item[state_feature] = out_item[state_feature][0].clone()
            for state_feature in self.states:
                if len(out_item[state_feature].shape) == 2:
                    out_item[state_feature] = out_item[state_feature][0]

        del item
        return out_item

    def reverse_features(self, item):
        if len(self.actions_convert_from_state) > 0:
            for action_feature, state_feature in self.actions_convert_from_state.items():
                item[state_feature] = torch.cat([item[state_feature].unsqueeze(0), item[action_feature]], dim=0)

                item.pop(action_feature)

        out_item = {}

        for target_key, convert_info in self.key_reverse_mapping.items():
            if isinstance(convert_info, dict) and convert_info["target_key"] in item:
                out_item[target_key] = item[convert_info["target_key"]]
                continue

            if isinstance(convert_info, list):
                convert_info = sorted(convert_info, key=lambda x: x["end"])
                concat_list = []
                convert_success = True
                for _convert_info in convert_info:
                    if _convert_info["target_key"] not in item:
                        raise ValueError(
                            f"{_convert_info['target_key']} is not contained in robot config as target feature"
                        )
                    concat_list.append(
                        item[_convert_info["target_key"]][
                            ..., _convert_info["target_start"] : _convert_info["target_end"]
                        ]
                    )
                if convert_success:
                    out_item[target_key] = torch.cat(concat_list, dim=-1)

        for feature in self.feature_to_keep:
            if feature in item:
                out_item[feature] = item[feature]
        del item
        return out_item

    def apply(self, item, policy_eval=False, training=True):
        w_action = not policy_eval
        if w_action:
            item["action_is_pad"] = (
                item[f"{self.org_features['actions'][0]}_is_pad"]
                if not len(self.actions_convert_from_state) > 0
                else item[f"{self.org_features['states'][0]}_is_pad"][1:]
            )
        else:
            item["action_is_pad"] = torch.zeros(self.chunk_size)
        item = self.convert_features(item, w_action=w_action)

        for action_feature in self.actions:
            if self.action_subtract_state[action_feature] and w_action:
                state_feature = action_feature.replace("action.", "observation.state.")
                if not (action_feature in item and state_feature in item):
                    raise ValueError(f"{action_feature} or/and {state_feature} are not in the item")
                relative_type = self.action_relative_type.get(action_feature)
                if _is_pose_relative_type(relative_type):
                    assert "end.position" in action_feature
                    item[action_feature] = relative_pose(
                        item[action_feature],
                        item[state_feature],
                        relative_type=relative_type,
                    )
                else:
                    item[action_feature] -= item[state_feature]

        if self.normalizer is not None:
            item = self.normalizer.normalize(item)

        if self.return_item_befor_padding:
            return item

        batch_dict = self.pad_and_concat(item, w_action)

        state = prepare_state(batch_dict, self.model_config.max_state_dim)
        actions = prepare_action(batch_dict, self.model_config.max_action_dim)
        image_processor = getattr(self.processor, "image_processor", None)
        if not self.disabled_image_features:
            images, img_masks, raw_images, image_grid_thw = prepare_images(
                image_processor,
                batch_dict,
                image_keys=self.output_image_keys,
                train=self.image_augment and not policy_eval,
                return_raw_images=self.return_raw_images,
                return_image_grid_thw=image_processor is not None,
            )
        else:
            images, img_masks, raw_images, image_grid_thw = [], [], None, None

        language = {}
        if not self.action_only:
            lang_tokens, lang_masks = prepare_language(self.model_config, self.tokenizer, batch_dict)
            language.update(lang_tokens=lang_tokens, lang_masks=lang_masks)
            if self.use_subtask_loss and training and not policy_eval:
                subtask_tokens, subtask_masks = prepare_subtask(
                    self.model_config, self.tokenizer, batch_dict["subtask"]
                )
                language.update(subtask_tokens=subtask_tokens, subtask_masks=subtask_masks)
        action_is_pad = batch_dict["action_is_pad"].to(dtype=torch.bool)

        state_joint_mask = batch_dict["state_joint_mask"]
        assert self.model_config.max_state_dim >= state_joint_mask.shape[-1], (
            f"max_action_dim is smaller than the state joint dimension: {self.model_config.max_action_dim} < {state_joint_mask.shape[-1]}"
        )
        state_joint_mask = F.pad(
            state_joint_mask, (0, self.model_config.max_state_dim - state_joint_mask.shape[-1])
        ).to(dtype=torch.bool)

        action_joint_mask = batch_dict["action_joint_mask"]
        assert self.model_config.max_action_dim >= action_joint_mask.shape[-1], (
            f"max_action_dim is smaller than the action joint dimension: {self.model_config.max_action_dim} < {action_joint_mask.shape[-1]}"
        )
        action_joint_mask = F.pad(
            action_joint_mask, (0, self.model_config.max_action_dim - action_joint_mask.shape[-1])
        ).to(dtype=torch.bool)

        chunk_joint_mask = batch_dict["chunk_joint_mask"]
        assert self.model_config.max_action_dim >= chunk_joint_mask.shape[-1], (
            f"max_action_dim is smaller than the action joint dimension: {self.model_config.max_action_dim} < {chunk_joint_mask.shape[-1]}"
        )
        chunk_joint_mask = chunk_joint_mask.to(dtype=torch.bool) & ~action_is_pad[:, None]
        chunk_joint_mask = F.pad(chunk_joint_mask, (0, self.model_config.max_action_dim - chunk_joint_mask.shape[-1]))

        del batch_dict
        batch_dict = {
            "images": images,
            "img_masks": img_masks,
            "state": state,
            "actions": actions,
            "action_is_pad": action_is_pad,
            "joint_mask": chunk_joint_mask,
            "state_joint_mask": state_joint_mask,
            "action_joint_mask": action_joint_mask,
        }
        batch_dict.update(language)
        if image_grid_thw is not None:
            batch_dict["image_grid_thw"] = image_grid_thw
        if self.return_raw_images:
            batch_dict["raw_images"] = raw_images
        return batch_dict

    def prepare_action_prefix(self, observation, action_prefix):
        """Re-anchor absolute native actions and map them to model action space.

        TT-RTC prefixes are actions that were planned from an older
        observation but are still committed while the new request is in
        flight.  The wire format is the robot's native absolute action vector;
        this method converts it using the *current* state, applies the same
        normalization as training, and pads it to ``max_action_dim`` without
        reprocessing any images.
        """

        if self.actions_convert_from_state:
            raise ValueError(
                "RTC action prefixes are not supported for robot configs whose "
                "actions are synthesized from future states."
            )
        native_action_keys = list(self.org_features.get("actions") or [])
        if len(native_action_keys) != 1:
            raise ValueError(
                f"RTC action-prefix transport requires exactly one native action key, got {native_action_keys}."
            )
        action_prefix = torch.as_tensor(action_prefix, dtype=torch.float32)
        if action_prefix.ndim != 2 or action_prefix.shape[0] == 0:
            raise ValueError(
                f"action_prefix must have shape [prefix_steps, native_action_dim], got {tuple(action_prefix.shape)}."
            )

        source = {
            key: torch.as_tensor(value)
            for key, value in observation.items()
            if key in self.org_features.get("states", ())
        }
        source[native_action_keys[0]] = action_prefix

        def convert_numeric_feature(target_key):
            convert_info = self.key_mapping.get(target_key)
            if convert_info is None:
                if target_key not in source:
                    raise KeyError(f"Missing native feature {target_key!r} for RTC prefix.")
                return source[target_key]
            origin_keys = convert_info["origin_keys"]
            if isinstance(origin_keys, str):
                if origin_keys not in source:
                    raise KeyError(f"Missing native feature {origin_keys!r} for RTC prefix.")
                return source[origin_keys]
            if not isinstance(origin_keys, OrderedDict):
                raise TypeError(f"Unsupported numeric mapping for RTC feature {target_key!r}.")
            values = []
            for origin_key, origin_info in origin_keys.items():
                origin_key = origin_key.split("*")[0]
                if origin_key not in source:
                    raise KeyError(f"Missing native feature {origin_key!r} for RTC prefix.")
                values.append(source[origin_key][..., origin_info["start"] : origin_info["end"]])
            return torch.cat(values, dim=-1)

        states = {key: convert_numeric_feature(key) for key in self.states}
        actions = {key: convert_numeric_feature(key) for key in self.actions}
        for action_key, value in actions.items():
            if not self.action_subtract_state[action_key]:
                continue
            state_key = action_key.replace("action.", "observation.state.")
            if state_key not in states:
                raise KeyError(f"RTC prefix conversion requires current state {state_key!r}.")
            relative_type = self.action_relative_type.get(action_key)
            if _is_pose_relative_type(relative_type):
                actions[action_key] = relative_pose(
                    value,
                    states[state_key],
                    relative_type=relative_type,
                )
            else:
                actions[action_key] = value - states[state_key]

        if self.normalizer is not None:
            actions = self.normalizer.normalize(actions)

        padded_parts = []
        for joint_name in self.feature_config.joints:
            action_key = f"action.{joint_name}"
            reserved_width = self.feature_config.joints_max_dim[joint_name]
            if action_key not in self.actions:
                padded_parts.append(
                    torch.zeros(
                        action_prefix.shape[0],
                        reserved_width,
                        dtype=action_prefix.dtype,
                        device=action_prefix.device,
                    )
                )
                continue
            value = torch.as_tensor(
                actions[action_key],
                dtype=action_prefix.dtype,
                device=action_prefix.device,
            )
            pad_width = reserved_width - value.shape[-1]
            if pad_width < 0:
                raise ValueError(f"RTC action feature {action_key!r} exceeds its reserved width.")
            padded_parts.append(F.pad(value, (0, pad_width)))
        model_prefix = torch.cat(padded_parts, dim=-1)
        if model_prefix.shape[-1] > self.model_config.max_action_dim:
            raise ValueError(
                "RTC action prefix exceeds model max_action_dim: "
                f"{model_prefix.shape[-1]} > {self.model_config.max_action_dim}."
            )
        return F.pad(
            model_prefix,
            (0, self.model_config.max_action_dim - model_prefix.shape[-1]),
        )

    def unapply(self, item):
        if not self.return_item_befor_padding:
            item = self.reverse_pad_and_concat(item)

        if self.normalizer is not None:
            item = self.normalizer.unnormalize(item)

        for action_feature in self.actions:
            if self.action_subtract_state[action_feature]:
                state_feature = action_feature.replace("action.", "observation.state.")
                relative_type = self.action_relative_type.get(action_feature)
                if _is_pose_relative_type(relative_type):
                    assert "end.position" in action_feature
                    item[action_feature] = absolute_pose(
                        item[action_feature],
                        item[state_feature],
                        relative_type=relative_type,
                    )
                else:
                    item[action_feature] += item[state_feature]

        item = self.reverse_features(item)
        return item

    def reverse_pad_and_concat(self, item):
        reverse_item = {}

        # In policy_eval, model output `actions` is always padded to max_action_dim
        # Pad mask with False at the tail so `actions[:, mask]` selects only the
        # real joint dims — the padded region is False and contributes nothing.
        state_joint_mask = item["state_joint_mask"]
        assert state_joint_mask.shape[-1] == item["state"].shape[-1]

        action_joint_mask = item["action_joint_mask"]
        assert action_joint_mask.shape[-1] == item["actions"].shape[-1]

        state = item["state"][state_joint_mask]
        action = item["actions"][:, action_joint_mask]

        for k in self.feature_config.joints:
            state_key = f"observation.state.{k}"
            if state_key in self.states:
                joint_dim = self.normalizer.norm_stats[state_key]["mean"].shape[-1]
                reverse_item[state_key] = state[:joint_dim]
                state = state[joint_dim:]
            del state_key

            action_key = f"action.{k}"
            if action_key in self.actions:
                joint_dim = self.normalizer.norm_stats[action_key]["mean"].shape[-1]
                reverse_item[action_key] = action[:, :joint_dim]
                action = action[:, joint_dim:]
            del action_key
        return reverse_item

    def pad_and_concat(self, item, w_action=True):
        images = {}

        def _to_uint8(img):
            # Keep images in uint8 [0, 255] before the Qwen-VL image processor, whose
            # `do_rescale=True` divides by 255. lerobot decodes mp4 (video dtype) cameras
            # already in uint8, but decodes parquet-embedded PNG (image dtype) cameras into
            # float32 [0, 1]; feeding the latter as-is would be rescaled a second time and
            # collapse the input to ~-1. Convert float [0, 1]-ish back to uint8, leave uint8 alone.
            if not torch.is_tensor(img) or not img.is_floating_point():
                return img
            try:
                max_val = float(img.max())
            except RuntimeError:
                max_val = 0.0
            if max_val <= 2.0:
                img = (img * 255).round().clamp(0, 255).to(torch.uint8)
            return img

        for image_key in self.model_images:
            if image_key not in item:
                continue
            image = _to_uint8(item[image_key])
            if image_key == self.head_model_image:
                if image.ndim != 4 or image.shape[0] != 2:
                    raise ValueError(
                        f"Head camera {image_key!r} requires [history, current] with shape "
                        f"[2,C,H,W], got {tuple(image.shape)}."
                    )
                images[image_key + ".history"] = image[0]
                images[image_key] = image[1]
            else:
                if image.ndim != 3:
                    raise ValueError(f"Wrist camera {image_key!r} requires [C,H,W], got {tuple(image.shape)}.")
                images[image_key] = image

        actions, action_joints_pad = [], []
        states, state_joints_pad = [], []

        for k in self.feature_config.joints:
            state_key = f"observation.state.{k}"

            if state_key in self.states:
                pad_len = self.feature_config.joints_max_dim[k] - item[state_key].shape[-1]
                assert pad_len >= 0, f"pad_len is negative: {pad_len}"
                states.append(F.pad(item[state_key], (0, pad_len)))
                state_joints_pad.append(F.pad(torch.ones(item[state_key].shape), (0, pad_len)))
            else:
                states.append(torch.zeros(self.feature_config.joints_max_dim[k]))
                state_joints_pad.append(torch.zeros(self.feature_config.joints_max_dim[k]))
            del state_key

            action_key = f"action.{k}"
            if action_key in self.actions and w_action:
                pad_len = self.feature_config.joints_max_dim[k] - item[action_key].shape[-1]
                assert pad_len >= 0, f"pad_len is negative: {pad_len}"
                actions.append(F.pad(item[action_key], (0, pad_len)))
                action_joints_pad.append(F.pad(torch.ones(item[action_key].shape[-1]), (0, pad_len)))
            elif action_key in self.actions and not w_action:
                assert action_key in self.normalizer.norm_stats, (
                    f"{action_key} not in norm keys: {self.normalizer.norm_stats.keys()}"
                )
                actions.append(torch.zeros(self.chunk_size, self.feature_config.joints_max_dim[k]))
                pad_len = (
                    self.feature_config.joints_max_dim[k] - self.normalizer.norm_stats[action_key]["mean"].shape[-1]
                )
                assert pad_len >= 0, f"pad_len is negative: {pad_len}"
                action_joints_pad.append(
                    F.pad(torch.ones(self.normalizer.norm_stats[action_key]["mean"].shape[-1]), (0, pad_len))
                )
            else:
                actions.append(torch.zeros(self.chunk_size, self.feature_config.joints_max_dim[k]))
                action_joints_pad.append(torch.zeros(self.feature_config.joints_max_dim[k]))
            del action_key

        action_joint_mask = torch.cat(action_joints_pad, dim=-1).to(dtype=torch.bool)
        state_joint_mask = torch.cat(state_joints_pad, dim=-1).to(dtype=torch.bool)
        state = torch.cat(states, dim=-1).to(torch.float32)
        action = torch.cat(actions, dim=-1).to(torch.float32)
        chunk_joint_mask = action_joint_mask.clone().unsqueeze(0).repeat(self.chunk_size, 1)

        batch_dict = {
            "image": images,
            "state": state,
            "action": action,
            "action_is_pad": item["action_is_pad"],
            "chunk_joint_mask": chunk_joint_mask,
            "action_joint_mask": action_joint_mask,
            "state_joint_mask": state_joint_mask,
        }
        if not self.action_only:
            batch_dict["prompt"] = [item["task"]]
            batch_dict["subtask"] = item.get("subtask")
        return batch_dict
