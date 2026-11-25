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
from transformers import AutoProcessor
from transformers.initialization import no_init_weights

from xense0vla.data.feature_transform import FeatureTransform
from xense0vla.data.transforms import canonicalize_visual, resize_rgb_uint8
from xense0vla.models import (
    cast_model_parameters_dtype,
    init_empty_weights,
)
from xense0vla.models import (
    load_model_weights as load_hf_model_weights,
)
from xense0vla.models.configuration_vla import VLAConfig
from xense0vla.models.modeling_vla import VLAModel
from xense0vla.utils.runtime_assets import (
    resolve_backbone_source,
    resolve_manifest_asset_path,
    resolve_training_manifest,
)


def set_seed_everywhere(seed: int, *, deterministic: bool = False):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


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
    @torch.no_grad()
    def sample_actions_batch(
        self,
        observation: dict[str, Tensor],
        use_bf16: bool = False,
        sample_actions_fn=None,
        action_prefix: Tensor | None = None,
        prefix_lengths: Tensor | None = None,
    ) -> Tensor:
        """Sample normalized action chunks from a transformed observation or batch."""
        device = next(self.parameters()).device
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        s1 = time.perf_counter()
        single = observation["wrist_images"].ndim == 4
        inputs = {}
        for key in ("wrist_images", "head_images", "image_grid_thw", "lang_tokens", "lang_masks"):
            if key in observation:
                value = observation[key]
                if single:
                    value = value.unsqueeze(0)
                inputs[key] = value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)

        if sample_actions_fn is None:
            raise ValueError("sample_actions_fn is required.")
        if action_prefix is not None:
            if single and action_prefix.ndim == 2:
                action_prefix = action_prefix.unsqueeze(0)
            action_prefix = action_prefix.to(dtype=torch.float32, device=device)
        if prefix_lengths is not None:
            if single and prefix_lengths.ndim == 0:
                prefix_lengths = prefix_lengths.unsqueeze(0)
            prefix_lengths = prefix_lengths.to(dtype=torch.long, device=device)
        actions = sample_actions_fn(
            **inputs,
            action_prefix=action_prefix,
            prefix_lengths=prefix_lengths,
        )

        actions_cpu = actions.to(dtype=torch.float32, device="cpu")
        delta_time = time.perf_counter() - s1
        logger.info("sample_actions batch=%s cost %.3f s", actions_cpu.shape[0], delta_time)
        return actions_cpu

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
        return {
            "protocol_version": 2 if rtc_enabled else 1,
            "policy_name": "action_policy" if self.config.standalone_action_policy else "vla",
            "chunk_ret": bool(self.chunk_ret),
            "use_length": int(self.use_length),
            "checkpoint_chunk_size": int(self.config.chunk_size),
            "action_output_dim": int(action_output_dim),
            "wrist_image_size": self.config.wrist_image_size,
            "head_image_size": self.config.head_image_size if not self.config.standalone_action_policy else None,
            "history_memory": self.config.history_memory if not self.config.standalone_action_policy else 0,
            "history_stride": self.config.history_stride,
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
        meta_tensors = [
            name
            for name, value in (*self.vla.named_parameters(), *self.vla.named_buffers())
            if value.is_meta
        ]
        if meta_tensors:
            raise RuntimeError(
                "Checkpoint loading left tensors on the meta device: "
                f"{meta_tensors[:5]}."
            )
        non_cuda_tensors = [
            name
            for name, value in (*self.vla.named_parameters(), *self.vla.named_buffers())
            if value.device.type != "cuda"
        ]
        if non_cuda_tensors:
            raise RuntimeError(
                "Checkpoint loading left tensors outside CUDA: "
                f"{non_cuda_tensors[:5]}."
            )
        self.vla.eval()
        self.sample_actions_fn = self.vla.model.sample_actions
        if self.use_compile:
            # Compile the fixed-shape denoising step (action tokens + static KV
            # tuple) instead of the full sample_actions wrapper, avoiding DA3 /
            # GatedDeltaNet graph breaks.
            self.vla.model.action_policy.denoise_step = torch.compile(
                self.vla.model.action_policy.denoise_step,
                mode="reduce-overhead",
            )

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

        self.processor = None
        if not config.standalone_action_policy:
            config.tokenizer_path = resolve_backbone_source(checkpoint_dir)
            self.processor = AutoProcessor.from_pretrained(
                config.tokenizer_path, padding_side="right", trust_remote_code=False, local_files_only=True
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
        # Parameter-only dtype conversion keeps parameters on meta while
        # preserving non-persistent FP32 RoPE buffers (Qwen3.5 inv_freq, etc.).
        cast_model_parameters_dtype(self.vla, target_dtype)
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

        feature_transform = FeatureTransform(
            str(robot_config_path),
            self.data_config,
            self.config,
            self.processor,
            chunk_size=self.config.chunk_size,
            norm_stats_path=self.norm_stats_path,
        )
        # Load data processors
        self.vla.feature_transform = feature_transform
        self.action_key = feature_transform.org_features["actions"]

    def _update_head_histories(self, observations, *, build_clips: bool = True):
        """Use explicit chronological clips, or sample a per-control-tick buffer."""
        if self.config.standalone_action_policy:
            return observations
        transform = self.vla.feature_transform
        offsets = transform.head_offsets
        keys = transform.org_head_model_images
        if len(self._head_histories) != len(observations):
            self._head_histories = [
                {key: deque(maxlen=1 - offsets[0]) for key in keys} for _ in observations
            ]
        prepared = []
        for raw, histories in zip(observations, self._head_histories):
            observation = dict(raw) if build_clips else None
            for key in keys:
                visual = canonicalize_visual(raw[key])
                history = histories[key]
                if visual.ndim == 4 and visual.shape[0] != len(offsets):
                    raise ValueError(f"Expected {len(offsets)} head frames for {key!r}, got {visual.shape[0]}.")
                # A single-source head can be stored at model resolution. Composite
                # heads must retain native coordinates until FeatureTransform joins them.
                if len(keys) == 1:
                    visual = resize_rgb_uint8(visual, self.config.head_image_size)
                history.append((visual[-1] if visual.ndim == 4 else visual).clone())
                if build_clips:
                    if visual.ndim == 3:
                        last = len(history) - 1
                        visual = torch.stack([history[max(0, last + offset)] for offset in offsets])
                    observation[key] = visual
            if build_clips:
                prepared.append(observation)
        return prepared if build_clips else observations

    def _unapply_batched_actions(self, transformed_observations, actions):
        action_chunk = {}
        for action in self.action_key:
            action_chunk[action] = []

        for transformed, action in zip(transformed_observations, actions):
            single = dict(transformed)
            single['actions'] = action.to(dtype=torch.float32, device='cpu')
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
        return self.vla.feature_transform.apply(observation, policy_eval=True)

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
                if prefix_length >= self.use_length:
                    raise ValueError(
                        f"rtc_prefix_length ({prefix_length}) must be strictly smaller than "
                        f"use_length ({self.use_length}) so new postfix actions are returned."
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
        reset_flag = observation.get("reset") if isinstance(observation, Mapping) else None
        if isinstance(reset_flag, (bool, np.bool_)) and bool(reset_flag):
            if "path_to_pi_model" in observation:
                raise ValueError("Remote model reload is disabled; restart the server with --model_path.")
            if "robo_name" not in observation:
                raise ValueError("A reset request must include robo_name.")
            self.reset(robo_name=observation['robo_name'])
            return dict(action=None)

        is_batch = 'batch' in observation
        observations = observation['batch'] if is_batch else [observation]
        batch_size = len(observations)
        has_rtc_prefix = any(
            isinstance(obs, Mapping)
            and (obs.get("rtc_action_prefix") is not None or obs.get("rtc_prefix_length") is not None)
            for obs in observations
        )
        if has_rtc_prefix and not self.chunk_ret:
            raise ValueError(
                "RTC prefix-conditioned requests require chunk_ret=True; single-step cached mode is unsupported."
            )
        should_forward = (
            self.chunk_ret
            or self.last_action_chunk is None
            or self._cached_batch_size != batch_size
            or (return_normalized and self.last_normalized_action_chunk is None)
            or self._chunk_step >= self.use_length
        )
        # Record every control tick, including ticks served from an action chunk,
        # but skip stacking full clips on ticks that will reuse the cached chunk.
        observations = self._update_head_histories(observations, build_clips=should_forward)

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
            # Copy dict so non-batch slicing or caller metadata never mutates the cache.
            action = dict(self.last_action_chunk)         # (B, chunk, dim)
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

