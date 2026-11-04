import json
import logging
import os
import random
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch import Tensor
from transformers import AutoConfig, AutoProcessor
from transformers.initialization import no_init_weights

from xense0vla.data.feature_transform import FeatureTransform
from xense0vla.data.transforms import canonicalize_visual
from xense0vla.models import (
    init_empty_weights,
)
from xense0vla.models import (
    load_model_weights as load_hf_model_weights,
)
from xense0vla.models.auto import build_image_processor
from xense0vla.models.configuration_vla import VLAConfig
from xense0vla.models.modeling_vla import VLAModel
from xense0vla.utils.runtime_assets import (
    resolve_backbone_source,
    resolve_manifest_asset_path,
    resolve_training_manifest,
)


def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


VLM_BACKBONE_SUBDIR = "vlm_backbone"
logger = logging.getLogger(__name__)


def _normalization_width(normalizer, key):
    norm_stats = getattr(normalizer, "norm_stats", None)
    if not isinstance(norm_stats, Mapping):
        return None
    feature_stats = norm_stats.get(key)
    if not isinstance(feature_stats, Mapping):
        return None
    shape = getattr(feature_stats.get("mean"), "shape", None)
    if not shape:
        return None
    width = int(shape[-1])
    return width if width > 0 else None


def _infer_action_output_dim(feature_transform):
    """Infer the flat deployment response width for a single native action key."""

    org_features = getattr(feature_transform, "org_features", None)
    if not isinstance(org_features, Mapping):
        return None
    action_keys = list(org_features.get("actions") or [])
    if len(action_keys) != 1:
        return None

    action_key = action_keys[0]
    reverse_mapping = getattr(feature_transform, "key_reverse_mapping", {})
    reverse = (
        reverse_mapping.get(action_key)
        if isinstance(reverse_mapping, Mapping)
        else None
    )
    if isinstance(reverse, list) and reverse:
        widths = []
        for item in reverse:
            if not isinstance(item, Mapping):
                return None
            if "start" in item and "end" in item:
                width = int(item["end"]) - int(item["start"])
            else:
                width = _normalization_width(
                    feature_transform.normalizer,
                    item.get("target_key"),
                )
            if width is None or width <= 0:
                return None
            widths.append(width)
        return sum(widths)

    if isinstance(reverse, Mapping):
        return _normalization_width(
            feature_transform.normalizer,
            reverse.get("target_key"),
        )
    return _normalization_width(feature_transform.normalizer, action_key)


def _require_checkpoint_root_file(
    checkpoint_dir: Path,
    file_name: str,
    *,
    description: str,
) -> Path:
    candidate = checkpoint_dir / file_name
    if not candidate.is_file():
        raise FileNotFoundError(
            f"A standalone VLA checkpoint requires {description} at "
            f"{str(candidate)!r}."
        )
    resolved = candidate.resolve()
    if resolved.parent != checkpoint_dir:
        raise ValueError(
            f"Checkpoint {description} must resolve directly inside "
            f"{str(checkpoint_dir)!r}, got {str(resolved)!r}."
        )
    return resolved


class PolicyPreprocessMixin:
    @staticmethod
    def _to_device_image_grid_thw(image_grid_thw, device):
        if image_grid_thw is None:
            return None
        return image_grid_thw.to(device=device, dtype=torch.long)

    @torch.no_grad()
    def sample_actions_batch(
        self,
        observation: dict[str, Tensor],
        use_bf16: bool = False,
        sample_actions_fn=None,
        action_prefix: Tensor | None = None,
        prefix_lengths: Tensor | None = None,
    ) -> Tensor:
        """Run one model forward for a batch of already-transformed observations.

        Single-sample inference in ``select_action`` builds a leading batch
        dimension just before calling ``sample_actions``. For Robocasa vector
        envs we already collate that dimension in the websocket server, so this
        method keeps the tensors batched and returns normalized action chunks
        with shape ``(B, chunk, joint_max_dim)``.
        """
        self.eval()
        device = "cuda"
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        s1 = time.time()

        images = observation["images"]
        img_masks = observation["img_masks"]
        lang_tokens = observation.get("lang_tokens")
        lang_masks = observation.get("lang_masks")
        state = observation["state"]
        image_grid_thw = observation.get("image_grid_thw", None)

        has_batch_dim = img_masks.ndim >= 2
        if not has_batch_dim:
            images = images.unsqueeze(0)
            img_masks = img_masks.unsqueeze(0)
        if lang_tokens is not None and lang_tokens.ndim == 1:
            lang_tokens = lang_tokens.unsqueeze(0)
            lang_masks = lang_masks.unsqueeze(0)
        if state.ndim == 1:
            state = state.unsqueeze(0)

        if sample_actions_fn is None:
            raise ValueError("sample_actions_fn is required.")
        model_images = images.to(device=device)
        if model_images.is_floating_point():
            model_images = model_images.to(dtype=dtype)

        actions = sample_actions_fn(
            images=model_images,
            img_masks=img_masks.to(device=device),
            lang_tokens=None if lang_tokens is None else lang_tokens.to(device=device),
            lang_masks=None if lang_masks is None else lang_masks.to(device=device),
            state=state.to(dtype=dtype, device=device),
            image_grid_thw=self._to_device_image_grid_thw(image_grid_thw, device),
            action_prefix=(
                None
                if action_prefix is None
                else action_prefix.to(dtype=dtype, device=device)
            ),
            prefix_lengths=(
                None
                if prefix_lengths is None
                else prefix_lengths.to(dtype=torch.long, device=device)
            ),
        )

        delta_time = time.time() - s1
        logger.info("sample_actions batch=%s cost %.3f s", actions.shape[0], delta_time)
        return actions.to(dtype=torch.float32, device="cpu")

class InferenceModel(PolicyPreprocessMixin, VLAModel):
    """Inference policy with feature normalization helpers."""


class RobotPolicy:
    """Reusable checkpoint policy with normalization, history, RTC and action chunks."""

    expected_standalone = None

    def __init__(
        self,
        path_to_pi_model="",
        use_length=-1,
        chunk_ret=False,
        use_bf16=True,
        use_fp32=False,
        use_compile=False,
        robot_name=None,
    ) -> None:
        if use_bf16 == use_fp32:
            raise ValueError("Exactly one of use_bf16 and use_fp32 must be enabled.")
        if not torch.cuda.is_available():
            raise RuntimeError("VLA deployment requires a CUDA GPU.")
        if use_bf16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support BF16 inference; use FP32 instead.")
        if use_length != -1 and use_length <= 0:
            raise ValueError(f"use_length must be -1 or a positive integer, got {use_length}.")
        self.use_length = use_length
        self.chunk_ret = chunk_ret

        self.use_compile = use_compile
        self.use_bf16 = use_bf16
        self.use_fp32 = use_fp32

        self.vla = self.load_vla(path_to_pi_model)
        self._prepare_model_for_inference()

        if self.use_length == -1:
            self.use_length = self.config.chunk_size
        if self.use_length > self.config.chunk_size:
            raise ValueError(
                f"use_length cannot exceed the checkpoint chunk size "
                f"({self.config.chunk_size}), got {self.use_length}."
            )

        self.global_step = 0
        self._chunk_step = 0
        self.last_action_chunk = None
        self.last_normalized_action_chunk = None
        self._cached_batch_size = None
        self._head_histories = []
        self.action_key: str = "action"

        selected_robot = robot_name or getattr(self.data_config, "data_name", None)
        if not selected_robot or selected_robot == "multi":
            raise ValueError(
                "robot_name is required when the checkpoint manifest does not select one robot."
            )
        self.reset(selected_robot)

    def get_metadata(self) -> dict:
        """Describe the active policy and the deployment features it supports."""

        feature_transform = self.vla.feature_transform
        action_output_dim = _infer_action_output_dim(feature_transform)
        if action_output_dim is None:
            raise RuntimeError(
                "Cannot publish a flat action_output_dim for the selected robot config."
            )
        rtc_enabled = bool(getattr(self.config, "rtc_training_enabled", False))
        image_height, image_width = feature_transform.image_size
        if image_height != image_width:
            raise RuntimeError(
                "Deployment client expects square RGB model inputs, got "
                f"{feature_transform.image_size}."
            )
        return {
            "protocol_version": 2 if rtc_enabled else 1,
            "policy_name": "action_policy" if self.config.standalone_action_policy else "vla",
            "chunk_ret": bool(self.chunk_ret),
            "use_length": int(self.use_length),
            "checkpoint_chunk_size": int(self.config.chunk_size),
            "action_output_dim": int(action_output_dim),
            "vision_input_size": int(image_height),
            "head_history_offset": int(self.config.head_history_offset),
            "rtc_mode": (
                "training_time_prefix_conditioned"
                if rtc_enabled
                else "async_receding_horizon"
            ),
            "action_chunk_timebase": "request_observation",
            "head_history_timebase": "client_control_ticks",
            "prefix_conditioned": rtc_enabled,
            "rtc_training_max_delay_steps": int(
                getattr(self.config, "rtc_max_delay_steps", 0)
            ),
        }

    def _prepare_model_for_inference(self) -> None:
        meta_parameters = [name for name, value in self.vla.named_parameters() if value.is_meta]
        if meta_parameters:
            raise RuntimeError(
                "Checkpoint loading left parameters on the meta device: "
                f"{meta_parameters[:5]}."
            )
        non_cuda_parameters = [
            name
            for name, value in self.vla.named_parameters()
            if value.device.type != "cuda"
        ]
        if non_cuda_parameters:
            raise RuntimeError(
                "Checkpoint loading left parameters outside CUDA: "
                f"{non_cuda_parameters[:5]}."
            )
        self.vla.eval()
        self.sample_actions_fn = self.vla.model.sample_actions
        if self.use_compile:
            self.sample_actions_fn = torch.compile(self.vla.model.sample_actions)

    def load_vla(self, path_to_pi_model) -> VLAModel:
        checkpoint_dir = Path(path_to_pi_model).expanduser().resolve()
        if not checkpoint_dir.is_dir():
            raise NotADirectoryError(
                f"model_path must be a complete HF checkpoint directory, got {str(checkpoint_dir)!r}."
            )
        print(f"loading model from: {checkpoint_dir}")

        training_config_path = resolve_training_manifest(checkpoint_dir)
        with open(training_config_path, "r", encoding="utf-8") as f:
            training_config = yaml.safe_load(f)

        if not isinstance(training_config, dict):
            raise ValueError(f"Invalid training config in {str(training_config_path)!r}.")
        composite_config_path = _require_checkpoint_root_file(
            checkpoint_dir,
            "config.json",
            description="authoritative model config",
        )
        with open(composite_config_path, "r", encoding="utf-8") as f:
            serialized_config = json.load(f)
        if serialized_config.get("model_type") != "vla":
            raise ValueError(
                f"{str(composite_config_path)!r} is not a VLA composite config."
            )
        config = VLAConfig.from_dict(serialized_config)

        if self.expected_standalone is not None and config.standalone_action_policy != self.expected_standalone:
            raise ValueError(f"{type(self).__name__} cannot load standalone_action_policy={config.standalone_action_policy!r}.")

        if getattr(config, "vlm_family", None) != "qwen3_5":
            raise ValueError(
                "The composite checkpoint config is not a Qwen3.5 VLA config; "
                f"got vlm_family={getattr(config, 'vlm_family', None)!r}."
            )
        backbone_subdir = getattr(config, "vlm_backbone_subdir", None)
        if backbone_subdir != VLM_BACKBONE_SUBDIR:
            raise ValueError(
                "A standalone VLA checkpoint must declare "
                f"vlm_backbone_subdir={VLM_BACKBONE_SUBDIR!r}, got "
                f"{backbone_subdir!r}."
            )
        base_model_path = resolve_backbone_source(checkpoint_dir)
        bundled_backbone_path = Path(base_model_path)

        qwen_config = AutoConfig.from_pretrained(
            base_model_path,
            trust_remote_code=False,
            local_files_only=True,
        )
        if getattr(qwen_config, "model_type", None) != "qwen3_5":
            raise ValueError(
                f"{base_model_path!r} has model_type={getattr(qwen_config, 'model_type', None)!r}; "
                "VLA deployment requires Qwen3.5."
            )
        config.vlm_path = None
        config.tokenizer_path = base_model_path

        if config.standalone_action_policy:
            self.processor = build_image_processor(str(bundled_backbone_path), local_files_only=True)
        else:
            self.processor = AutoProcessor.from_pretrained(
                str(bundled_backbone_path), padding_side="right",
                trust_remote_code=False, local_files_only=True,
            )
        self.language_tokenizer = getattr(self.processor, "tokenizer", None)
        data_section = training_config.get("data")
        if not isinstance(data_section, dict):
            raise ValueError(f"Training config {str(training_config_path)!r} has no data section.")
        data_section = dict(data_section)
        data_section["norm_stats_file"] = resolve_manifest_asset_path(
            data_section.get("norm_stats_file"),
            training_config_path,
        )
        data_section["robot_config_root"] = resolve_manifest_asset_path(
            data_section.get("robot_config_root"),
            training_config_path,
        )
        norm_stats_path = Path(data_section["norm_stats_file"])
        if not norm_stats_path.is_file():
            raise FileNotFoundError(
                f"Bundled norm_stats_file is not a file: {str(norm_stats_path)!r}."
            )
        robot_config_root = Path(data_section["robot_config_root"])
        if not robot_config_root.is_dir():
            raise NotADirectoryError(
                "Bundled robot_config_root is not a directory: "
                f"{str(robot_config_root)!r}."
            )
        data_config = SimpleNamespace(**data_section)

        target_dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        load_started_at = time.perf_counter()
        print(f"Initializing model directly on CUDA as {target_dtype} ...")
        with init_empty_weights(), no_init_weights():
            self.vla = InferenceModel(config, eval=True)
        # Dtype-only conversion keeps parameters on meta. The loader then
        # materializes each checkpoint tensor directly on CUDA in its final
        # inference dtype, avoiding a full FP32 CPU model and a second transfer.
        self.vla = self.vla.to(dtype=target_dtype)
        load_hf_model_weights(
            self.vla,
            str(checkpoint_dir),
            init_device="cuda",
            load_full_checkpoint=True,
        )

        self.vla.feature_transform = None
        self.data_config = data_config
        self.config = config

        self.norm_stats_path = data_config.norm_stats_file

        torch.cuda.synchronize()
        print(f"Model initialized on CUDA in {time.perf_counter() - load_started_at:.1f}s.")

        return self.vla

    def reset(self, robo_name) -> None:
        self.global_step = 0
        self._chunk_step = 0
        self.last_action_chunk = None
        self.last_normalized_action_chunk = None
        self._cached_batch_size = None
        self._head_histories = []

        configured_robot = getattr(self.data_config, "data_name", None)
        if configured_robot and configured_robot != "multi" and robo_name != configured_robot:
            raise ValueError(
                f"This checkpoint is packaged for robot {configured_robot!r}, got {robo_name!r}."
            )
        if Path(robo_name).name != robo_name or robo_name in {"", ".", ".."}:
            raise ValueError(f"Invalid robot name: {robo_name!r}.")

        robot_config_root_value = getattr(self.data_config, "robot_config_root", None)
        if not robot_config_root_value:
            raise ValueError("The checkpoint manifest is missing data.robot_config_root.")
        robot_config_root = Path(robot_config_root_value).resolve()
        if not robot_config_root.is_dir():
            raise NotADirectoryError(
                f"Bundled robot_config_root is not a directory: {str(robot_config_root)!r}."
            )
        robot_config_path = (robot_config_root / f"{robo_name}.yaml").resolve()
        try:
            robot_config_path.relative_to(robot_config_root)
        except ValueError as exc:
            raise ValueError(
                f"Robot config for {robo_name!r} escapes bundled robot_config_root."
            ) from exc
        if not robot_config_path.is_file():
            raise FileNotFoundError(
                f"Robot config does not exist: {str(robot_config_path)!r}."
            )

        image_size = getattr(self.data_config, 'img_size', 256)
        feature_transform = FeatureTransform(
            str(robot_config_path),
            self.data_config,
            self.config,
            self.processor,
            chunk_size=self.config.chunk_size,
            norm_stats_path=self.norm_stats_path,
            image_size=(image_size, image_size),
        )
        # Load data processors
        self.vla.feature_transform = feature_transform
        self.action_key = feature_transform.org_features["actions"]

    def _update_head_histories(self, observations):
        """Accept explicit [t-offset, t] pairs, or keep a control-tick history.

        Chunk clients that skip observations must send the explicit pair so
        the offset remains measured in camera/control frames, not requests.
        Episode warmup repeats the earliest available frame.
        """
        if self.config.standalone_action_policy:
            return observations
        offset = int(self.config.head_history_offset)
        if len(self._head_histories) != len(observations):
            self._head_histories = [deque(maxlen=offset + 1) for _ in observations]
        source_key = self.vla.feature_transform.org_model_images[0]
        prepared = []
        for raw, history in zip(observations, self._head_histories):
            observation = dict(raw)
            if source_key not in observation:
                raise KeyError(f"Missing head camera {source_key!r}.")
            visual = canonicalize_visual(observation[source_key])
            if visual.ndim == 4:
                if visual.shape[0] != 2:
                    raise ValueError("Head camera clips must contain [history, current] frames.")
                observation[source_key] = visual
                history.append(visual[-1].clone())
            elif visual.ndim == 3:
                history.append(visual.clone())
                observation[source_key] = torch.stack([history[0], history[-1]])
            else:
                raise ValueError("Head camera must be CHW/HWC or a two-frame clip.")
            prepared.append(observation)
        return prepared

    def _unapply_batched_actions(self, transformed_observations, actions):
        action_chunk = {}
        for action in self.action_key:
            action_chunk[action] = []

        for transformed, action in zip(transformed_observations, actions):
            single = dict(transformed)
            single['actions'] = action.to(dtype=torch.float32, device='cpu')
            if self.use_bf16 and 'state' in single:
                single['state'] = single['state'].to(dtype=torch.float32)
            data = self.vla.feature_transform.unapply(single)

            for action in self.action_key:
                # keep action keys after unapply
                value = data[action]
                if isinstance(value, torch.Tensor):
                    value = value.float().cpu().numpy()
                else:
                    value = np.asarray(value, dtype=np.float32)
                action_chunk[action].append(value)

        for action_key in action_chunk.keys():
            action_chunk[action_key] = np.stack(action_chunk[action_key], axis=0)

        return action_chunk

    def _prepare_model_input(self, observation):
        # not modify input observation
        observation = dict(observation)
        for k, v in list(observation.items()):
            if isinstance(v, np.ndarray):
                observation[k] = torch.from_numpy(v)
        observation =  self.vla.feature_transform.apply(observation, policy_eval=True)
        if self.use_bf16:
            observation['state'] = observation['state'].to(torch.bfloat16)
        return observation

    def _prepare_rtc_prefix_batch(self, observations):
        """Validate wire prefixes and map them into normalized model space."""

        rtc_enabled = bool(getattr(self.config, "rtc_training_enabled", False))
        cleaned = []
        model_prefixes = []
        prefix_lengths = []
        native_prefixes = []
        for raw_observation in observations:
            observation = dict(raw_observation)
            raw_prefix = observation.pop("rtc_action_prefix", None)
            declared_length = observation.pop("rtc_prefix_length", None)
            if (raw_prefix is None) != (declared_length is None):
                raise ValueError(
                    "rtc_action_prefix and rtc_prefix_length must be supplied together."
                )
            if raw_prefix is None:
                prefix_length = 0
                native_prefix = None
                model_prefix = torch.zeros(
                    self.config.chunk_size,
                    self.config.max_action_dim,
                    dtype=torch.float32,
                )
            else:
                if not rtc_enabled:
                    raise ValueError(
                        "The client supplied an RTC prefix to a checkpoint that was "
                        "not trained with rtc_training.enabled=true."
                    )
                if isinstance(declared_length, bool) or not isinstance(
                    declared_length, (int, np.integer)
                ):
                    raise TypeError("rtc_prefix_length must be an integer.")
                prefix_length = int(declared_length)
                raw_prefix = np.asarray(raw_prefix)
                native_action_dim = _infer_action_output_dim(
                    self.vla.feature_transform
                )
                expected_shape = (prefix_length, native_action_dim)
                if native_action_dim is None or raw_prefix.shape != expected_shape:
                    raise ValueError(
                        f"rtc_action_prefix must have shape {expected_shape}, "
                        f"got {raw_prefix.shape}."
                    )
                if not 1 <= prefix_length <= self.config.rtc_max_delay_steps:
                    raise ValueError(
                        "rtc_prefix_length must stay within the checkpoint's trained "
                        f"range [1, {self.config.rtc_max_delay_steps}], got {prefix_length}."
                    )
                if not np.issubdtype(raw_prefix.dtype, np.number) or not np.isfinite(
                    raw_prefix
                ).all():
                    raise ValueError("rtc_action_prefix must contain finite numeric actions.")
                native_prefix = np.ascontiguousarray(
                    raw_prefix.astype(np.float32, copy=False)
                )
                converted = self.vla.feature_transform.prepare_action_prefix(
                    observation,
                    torch.as_tensor(raw_prefix, dtype=torch.float32),
                )
                model_prefix = torch.zeros(
                    self.config.chunk_size,
                    self.config.max_action_dim,
                    dtype=torch.float32,
                )
                model_prefix[:prefix_length] = converted
            cleaned.append(observation)
            model_prefixes.append(model_prefix)
            prefix_lengths.append(prefix_length)
            native_prefixes.append(native_prefix)
        # RTC checkpoints always receive the same tensor signature, including
        # warmup with all-zero prefix lengths. This lets torch.compile build
        # the prefix-conditioned graph before the control loop starts instead
        # of recompiling on the first live request.
        if not rtc_enabled:
            return cleaned, None, None, native_prefixes
        return (
            cleaned,
            torch.stack(model_prefixes, dim=0),
            torch.tensor(prefix_lengths, dtype=torch.long),
            native_prefixes,
        )

    @staticmethod
    def _pad_and_stack_tensors(values):
        shapes = [tuple(value.shape) for value in values]
        if len(set(shapes)) == 1:
            return torch.stack(values, dim=0)

        if all(value.ndim == 1 for value in values):
            max_len = max(value.shape[0] for value in values)
            fill_value = False if values[0].dtype == torch.bool else 0
            padded = []
            for value in values:
                out = torch.full(
                    (max_len,),
                    fill_value,
                    dtype=value.dtype,
                    device=value.device,
                )
                out[: value.shape[0]] = value
                padded.append(out)
            return torch.stack(padded, dim=0)

        raise ValueError(f"Cannot batch tensors with different shapes: {shapes}")
    def _infer_batch(self, observations, return_normalized=False):
        if not isinstance(observations, (list, tuple)) or len(observations) == 0:
            raise ValueError("batch observation must be a non-empty list")
        (
            observations,
            action_prefix,
            prefix_lengths,
            native_prefixes,
        ) = self._prepare_rtc_prefix_batch(observations)
        applied = [self._prepare_model_input(obs) for obs in observations] # bsize, dict{key }
        batch_observation = {}
        for key in applied[0].keys():
            values = [item[key] for item in applied]
            if isinstance(values[0], torch.Tensor):
                # Pad different 1d tokens to the max length: language, language mask
                batch_observation[key] = self._pad_and_stack_tensors(values)
            else:
                batch_observation[key] = values

        actions = self.vla.sample_actions_batch(
            batch_observation,
            self.use_bf16,
            sample_actions_fn=self.sample_actions_fn,
            action_prefix=action_prefix,
            prefix_lengths=prefix_lengths,
        )

        unnormalized_actions = self._unapply_batched_actions(applied, actions)
        # The native prefix supplied by the client is the authoritative clean
        # plan. Mapping absolute poses through relative pose normalization and
        # BF16 model space need not be bitwise reversible, so publish those
        # committed steps verbatim while the model uses their converted form
        # to condition the newly generated postfix.
        if any(prefix is not None for prefix in native_prefixes):
            native_action_keys = list(
                self.vla.feature_transform.org_features.get("actions") or []
            )
            if len(native_action_keys) != 1:
                raise RuntimeError(
                    "RTC native-prefix restoration requires exactly one action key."
                )
            native_action_key = native_action_keys[0]
            for batch_index, native_prefix in enumerate(native_prefixes):
                if native_prefix is not None:
                    unnormalized_actions[native_action_key][
                        batch_index, : native_prefix.shape[0]
                    ] = native_prefix
        if return_normalized:
            return unnormalized_actions, actions
        return unnormalized_actions

    def infer(self, observation, return_normalized=False):
        """Generates an action with the VLA policy."""
        if 'reset' in observation and observation['reset']:
            if "path_to_pi_model" in observation:
                raise ValueError("Remote model reload is disabled; restart the server with --model_path.")
            if "robo_name" not in observation:
                raise ValueError("A reset request must include robo_name.")
            self.reset(robo_name=observation['robo_name'])
            return dict(action=None)

        is_batch = 'batch' in observation
        observations = observation['batch'] if is_batch else [observation]
        batch_size = len(observations)
        # Record every control tick, including ticks served from an action chunk.
        observations = self._update_head_histories(observations)
        should_forward = (
            self.chunk_ret
            or self.last_action_chunk is None
            or self._cached_batch_size != batch_size
            or (return_normalized and self.last_normalized_action_chunk is None)
            or self._chunk_step >= self.use_length
        )

        if should_forward:
            if return_normalized:
                unnormalized_actions, normalized_actions = self._infer_batch(
                    observations,
                    return_normalized=True,
                )
            else:
                unnormalized_actions = self._infer_batch(observations)
                normalized_actions = None
            for output_key in unnormalized_actions:
                unnormalized_actions[output_key] = unnormalized_actions[output_key][:, :self.use_length]
            if normalized_actions is not None:
                normalized_actions = normalized_actions[:, :self.use_length]

            self.last_action_chunk = unnormalized_actions  # always (B, chunk, dim)
            self.last_normalized_action_chunk = normalized_actions
            self._cached_batch_size = batch_size
            self._chunk_step = 0

        if self.chunk_ret:
            action = self.last_action_chunk               # (B, chunk, dim)
            normalized_action = self.last_normalized_action_chunk
        else:
            step_idx = self._chunk_step
            action = {}
            for action_key in self.last_action_chunk.keys():
                action[action_key] = self.last_action_chunk[action_key][:, step_idx]
            normalized_action = (
                self.last_normalized_action_chunk[:, step_idx]
                if self.last_normalized_action_chunk is not None
                else None
            )

        if not is_batch:
            for action_key in action:
                action[action_key] = action[action_key][0]
            if normalized_action is not None:
                normalized_action = normalized_action[0]

        result = action
        if return_normalized:
            result = dict(result)
            result["_normalized_actions"] = normalized_action

        self.global_step += 1
        if not self.chunk_ret:
            self._chunk_step += 1

        return result

