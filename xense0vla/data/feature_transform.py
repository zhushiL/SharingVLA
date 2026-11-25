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
    history_offsets,
    prepare_action,
    prepare_images,
    prepare_language,
    prepare_state,
    prepare_subtask,
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
        self.head_offsets = history_offsets(config_value("history_memory", 3), config_value("history_stride", 30))
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
        if not self.feature_config.images:
            self.feature_config.images = list(self.model_images)
        active_images = set(self.model_images)
        robot_config["images"] = [
            entry
            for entry in robot_config.get("images", [])
            if (entry if isinstance(entry, str) else next(iter(entry))) in active_images
        ]
        if not return_item_befor_padding:
            joints = self.feature_config.joints
            if not joints or len(set(joints)) != len(joints):
                raise ValueError("data.joints must contain a non-empty list of unique features.")
            self.action_widths_per_arm = []
            for name in joints:
                width = self.feature_config.joints_max_dim[name]
                if not isinstance(width, int) or width <= 0 or width % 2:
                    raise ValueError(f"data.joints width for {name!r} must be positive and even for two arms.")
                self.action_widths_per_arm.append(width // 2)
            self.check_robot_config(robot_config)

        self.model_config = model_config
        self.tokenizer = getattr(processor, "tokenizer", None)
        self.processor = processor

        self.chunk_size = chunk_size
        self.return_item_befor_padding = return_item_befor_padding

        # disabled_image_features: keep the image features or not when getting lerobot item
        self.disabled_image_features = disabled_image_features
        self.return_raw_images = return_raw_images

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
        self.normalizer = self.get_normalizer(norm_stats_path, do_nomalize, data_config)

    def get_origin_image_features(self, target_images):
        """Resolve native image keys needed by the cameras sent to the model."""
        origin_images = []
        for target_image in target_images:
            if target_image not in self.images:
                # Statistics-only transforms also work with camera-free datasets.
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

        action_state_norm_type = {
            k: v
            for d in data_config.norm_type
            for k, v in (ast.literal_eval(d) if isinstance(d, str) else dict(d)).items()
        }

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

        with open(norm_stats_path, encoding="utf-8") as f:
            norm_stats = json.load(f)

        stats_map = norm_stats["norm_stats"]
        required_stat_keys = [
            key
            for key in (*self.actions, *self.states)
            if norm_type.get(key, "identity") not in {"identity", "sincos"}
        ]
        missing_stat_keys = [key for key in required_stat_keys if key not in stats_map]
        if missing_stat_keys:
            raise KeyError(
                f"Normalization stats file {norm_stats_path!r} is missing required keys: {missing_stat_keys}."
            )

        normalizer = Normalizer(
            norm_stats=stats_map,
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

                        default_relative_type = "local" if "end.position" in target_feature else None
                        action_relative_type[target_feature] = feature_convert_info[target_feature].pop(
                            "relative_type", default_relative_type
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
                    org_features["actions"] = []
                    continue
            if len(feature) == 0:
                org_features[feature_category] = target_features[feature_category]
            else:
                org_features[feature_category] = list(feature)

        self.key_mapping = to_convert_features
        self.key_reverse_mapping = reverse_convert_features
        self._mapped_feature_widths = {}
        for target_key, convert_info in to_convert_features.items():
            origin_keys = convert_info.get("origin_keys")
            if isinstance(origin_keys, OrderedDict) and all(
                isinstance(info, dict) and "start" in info and "end" in info for info in origin_keys.values()
            ):
                self._mapped_feature_widths[target_key] = sum(
                    int(info["end"]) - int(info["start"]) for info in origin_keys.values()
                )

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
            for origin_key, origin_info in convert_info["origin_keys"].items():
                origin_key = origin_key.split("*")[0]
                if origin_key not in item:
                    raise KeyError(f"Missing source feature {origin_key!r} required by {target_key!r}.")
                source_tensor = item[origin_key]
                start = int(origin_info["start"])
                end = int(origin_info["end"])
                if start < 0 or end <= start or source_tensor.shape[-1] < end:
                    raise ValueError(
                        f"Feature slice [{start}:{end}] for {target_key!r} exceeds source "
                        f"{origin_key!r} width {source_tensor.shape[-1]}."
                    )
                concat_list.append(source_tensor[..., start:end])
            out_item[target_key] = torch.cat(concat_list, dim=-1)
            del concat_list

        for feature in self.feature_to_keep:
            if feature in item:
                out_item[feature] = item[feature]

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

        raw_states = None
        if policy_eval:
            # Preserve exact unnormalized, unclipped state tensors so unapply()
            # reconstructs absolute poses from the same raw reference frame used
            # during training and RTC prefix conversion.
            raw_states = {
                state_feature: (
                    item[state_feature].detach().clone()
                    if isinstance(item[state_feature], torch.Tensor)
                    else torch.as_tensor(item[state_feature]).clone()
                )
                for state_feature in self.states
                if state_feature in item
            }

        if self.normalizer is not None:
            item = self.normalizer.normalize(item)

        if self.return_item_befor_padding:
            if raw_states is not None:
                item["raw_states"] = raw_states
            return item

        batch_dict = self.pad_and_concat(item, w_action)

        state = prepare_state(batch_dict, self.model_config.max_state_dim)
        actions = prepare_action(batch_dict, self.model_config.max_action_dim)
        visual_inputs = {}
        if not self.disabled_image_features:
            visual_inputs = prepare_images(
                item, self.model_images[-2:], self.head_model_image, self.model_config, self.processor,
                augment=self.image_augment and not policy_eval, return_raw=self.return_raw_images,
            )

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
            **visual_inputs,
            "state": state,
            "actions": actions,
            "action_is_pad": action_is_pad,
            "joint_mask": chunk_joint_mask,
            "state_joint_mask": state_joint_mask,
            "action_joint_mask": action_joint_mask,
        }
        if raw_states is not None:
            batch_dict["raw_states"] = raw_states
        batch_dict.update(language)
        return batch_dict

    def _split_and_pad_action_feature(self, value, name):
        width = self.feature_config.joints_max_dim[name]
        actual_width = value.shape[-1]
        if actual_width > width:
            raise ValueError(f"Action feature {name!r} has width {actual_width}, above its reserved width {width}.")
        if actual_width % 2:
            raise ValueError(f"Action feature {name!r} must have an even width for two arms.")
        # Keep a separate arm axis so padding lands in the correct arm's slots.
        arms = value.unflatten(-1, (2, actual_width // 2))
        if actual_width == width:
            return arms
        return F.pad(arms, (0, (width - actual_width) // 2))

    def _pack_action_features(self, features, *, empty):
        """Pack named features in joints order within each arm. Missing features get padding."""
        parts = [
            self._split_and_pad_action_feature(features.get(f"action.{name}", empty), name)
            for name in self.feature_config.joints
        ]
        return torch.cat(parts, dim=-1).flatten(-2)

    def _unpack_action_features(self, packed):
        arm_width = sum(self.action_widths_per_arm)
        arms = packed[..., :2 * arm_width].unflatten(-1, (2, arm_width))
        return arms.split(self.action_widths_per_arm, dim=-1)

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

        actions = {
            key: torch.as_tensor(value, dtype=action_prefix.dtype, device=action_prefix.device)
            for key, value in actions.items()
        }
        model_prefix = self._pack_action_features(actions, empty=action_prefix[..., :0])
        if model_prefix.shape[-1] > self.model_config.max_action_dim:
            raise ValueError(
                "RTC action prefix exceeds model max_action_dim: "
                f"{model_prefix.shape[-1]} > {self.model_config.max_action_dim}."
            )
        return F.pad(
            model_prefix,
            (0, self.model_config.max_action_dim - model_prefix.shape[-1]),
        )

    def _feature_width(self, feature_key: str) -> int:
        if self.normalizer is not None and feature_key in self.normalizer.norm_stats:
            return int(self.normalizer.norm_stats[feature_key]["mean"].shape[-1])
        if feature_key in self._mapped_feature_widths:
            return self._mapped_feature_widths[feature_key]
        base_name = feature_key.split("observation.state.")[-1].split("action.")[-1]
        if self.feature_config.joints_max_dim and base_name in self.feature_config.joints_max_dim:
            return int(self.feature_config.joints_max_dim[base_name])
        raise KeyError(f"Unable to determine feature width for {feature_key!r}.")

    def unapply(self, item):
        raw_states = item.get("raw_states")
        if not self.return_item_befor_padding:
            item = self.reverse_pad_and_concat(item)

        if self.normalizer is not None:
            item = self.normalizer.unnormalize(item)

        if isinstance(raw_states, dict):
            for state_feature, raw_value in raw_states.items():
                if state_feature in self.states:
                    item[state_feature] = raw_value

        for action_feature in self.actions:
            if self.action_subtract_state[action_feature]:
                state_feature = action_feature.replace("action.", "observation.state.")
                relative_type = self.action_relative_type.get(action_feature)
                anchor_state = item[state_feature]
                if isinstance(item[action_feature], torch.Tensor):
                    anchor_state = torch.as_tensor(
                        anchor_state,
                        dtype=item[action_feature].dtype,
                        device=item[action_feature].device,
                    )
                if _is_pose_relative_type(relative_type):
                    assert "end.position" in action_feature
                    item[action_feature] = absolute_pose(
                        item[action_feature],
                        anchor_state,
                        relative_type=relative_type,
                    )
                else:
                    item[action_feature] = item[action_feature] + anchor_state

        item = self.reverse_features(item)
        return item

    def reverse_pad_and_concat(self, item):
        reverse_item = {}

        # The masks exclude unused feature slots and trailing model padding.
        state_joint_mask = item["state_joint_mask"]
        assert state_joint_mask.shape[-1] == item["state"].shape[-1]

        action_joint_mask = item["action_joint_mask"]
        assert action_joint_mask.shape[-1] == item["actions"].shape[-1]

        state = item["state"][state_joint_mask]
        action_parts = self._unpack_action_features(item["actions"])
        action_masks = self._unpack_action_features(action_joint_mask)

        for k, action, mask in zip(self.feature_config.joints, action_parts, action_masks, strict=True):
            state_key = f"observation.state.{k}"
            if state_key in self.states:
                joint_dim = self._feature_width(state_key)
                reverse_item[state_key] = state[:joint_dim]
                state = state[joint_dim:]
            del state_key

            action_key = f"action.{k}"
            if action_key in self.actions:
                reverse_item[action_key] = action[..., mask]
        return reverse_item

    def pad_and_concat(self, item, w_action=True):
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

        state = torch.cat(states, dim=-1).to(torch.float32)
        state_joint_mask = torch.cat(state_joints_pad, dim=-1).to(dtype=torch.bool)

        actions, action_masks = {}, {}
        for action_key in self.actions:
            if w_action:
                value = item[action_key]
            else:
                width = self._feature_width(action_key)
                value = torch.zeros(self.chunk_size, width)
            actions[action_key] = value
            action_masks[action_key] = value.new_ones((value.shape[-1],), dtype=torch.bool)

        action = self._pack_action_features(actions, empty=torch.empty(self.chunk_size, 0)).to(torch.float32)
        action_joint_mask = self._pack_action_features(action_masks, empty=torch.empty(0, dtype=torch.bool))
        chunk_joint_mask = action_joint_mask.clone().unsqueeze(0).repeat(self.chunk_size, 1)

        batch_dict = {
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
