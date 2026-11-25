from collections import OrderedDict
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from xense0vla.utils import logging as logging_utils


logger = logging_utils.get_logger(__name__)

IGNORE_INDEX = -100
_TOKEN_CACHE_MAX_SIZE = 4096
_LANGUAGE_TOKEN_CACHE: OrderedDict[tuple, tuple[Tensor, Tensor]] = OrderedDict()
_SUBTASK_TOKEN_CACHE: OrderedDict[tuple, tuple[Tensor, Tensor]] = OrderedDict()


def _clamp_value(value, *, min_val: float, max_val: float):
    if isinstance(value, torch.Tensor):
        return torch.clamp(value, min=min_val, max=max_val)
    return np.clip(value, min_val, max_val)


def _as_debug_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def dict_apply(func, d):
    """
    Apply a function to all values in a dictionary recursively.
    If the value is a dictionary, it will apply the function to its values.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            dict_apply(func, value)
        else:
            d[key] = func(value)
    return d


class Normalizer:
    def __init__(
        self,
        norm_stats: Dict[str, Dict[str, np.ndarray]],
        norm_type: Dict[str, str] | None = None,
    ):
        self.norm_stats = {
            key: {stat_name: np.asarray(stat_val, dtype=np.float32) for stat_name, stat_val in stats.items()}
            for key, stats in norm_stats.items()
        }
        self._torch_norm_stats = {
            key: {stat_name: torch.from_numpy(stat_val) for stat_name, stat_val in stats.items()}
            for key, stats in self.norm_stats.items()
        }
        self.norm_type = norm_type or {}

    def _get_stat(self, key: str, stat_name: str, value):
        if isinstance(value, torch.Tensor):
            stat = self._torch_norm_stats[key][stat_name]
            if stat.device != value.device or stat.dtype != torch.float32:
                stat = stat.to(device=value.device, dtype=torch.float32)
        else:
            stat = self.norm_stats[key][stat_name]

        # Action stats may use a longer horizon than the active action chunk.
        # When training with a smaller chunk_size, align stats to the actual
        # action horizon instead of relying on broadcasting.
        if getattr(stat, "ndim", None) == 2 and getattr(value, "ndim", None) == 2:
            if stat.shape[-1] != value.shape[-1]:
                raise ValueError(
                    f"Normalization dim mismatch for {key}.{stat_name}: "
                    f"stat shape {stat.shape}, value shape {tuple(value.shape)}"
                )
            if value.shape[0] > stat.shape[0]:
                raise ValueError(
                    f"Normalization horizon mismatch for {key}.{stat_name}: "
                    f"stat horizon {stat.shape[0]}, value horizon {value.shape[0]}. "
                    "Please recompute norm stats with a larger chunk_size."
                )
            if stat.shape[0] != value.shape[0]:
                stat = stat[: value.shape[0]]

        return stat

    def normalize(self, data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        normalized_data = {}
        for key, value in data.items():
            if key in self.norm_stats:
                norm_type = self.norm_type.get(key, "identity")
                val = value.float() if isinstance(value, torch.Tensor) and value.is_floating_point() else value
                if norm_type == "meanstd":
                    mean = self._get_stat(key, "mean", val)
                    std = self._get_stat(key, "std", val)
                    normalized_value = (val - mean) / (std + 1e-6)
                elif norm_type == "bounds_98":
                    low = self._get_stat(key, "q02", val)
                    high = self._get_stat(key, "q98", val)
                    normalized_value = (val - low) / (high - low + 1e-6) * 2.0 - 1.0
                    normalized_value = _clamp_value(normalized_value, min_val=-1.5, max_val=1.5)
                elif norm_type == "bounds_99":
                    low = self._get_stat(key, "q01", val)
                    high = self._get_stat(key, "q99", val)
                    normalized_value = (val - low) / (high - low + 1e-6) * 2.0 - 1.0
                    normalized_value = _clamp_value(normalized_value, min_val=-1.5, max_val=1.5)
                elif norm_type == "bounds_98_woclip":
                    low = self._get_stat(key, "q02", val)
                    high = self._get_stat(key, "q98", val)
                    normalized_value = (val - low) / (high - low + 1e-6) * 2.0 - 1.0
                elif norm_type == "bounds_99_woclip":
                    low = self._get_stat(key, "q01", val)
                    high = self._get_stat(key, "q99", val)
                    normalized_value = (val - low) / (high - low + 1e-6) * 2.0 - 1.0
                elif norm_type == "std":
                    std = self._get_stat(key, "std", val)
                    normalized_value = val / (std + 1e-6)
                elif norm_type == "minmax":
                    min_val = self._get_stat(key, "min", val)
                    max_val = self._get_stat(key, "max", val)
                    normalized_value = (val - min_val) / (max_val - min_val + 1e-6) * 2.0 - 1.0
                    normalized_value = _clamp_value(normalized_value, min_val=-1.0, max_val=1.0)
                elif norm_type == "minmax_woclip":
                    min_val = self._get_stat(key, "min", val)
                    max_val = self._get_stat(key, "max", val)
                    normalized_value = (val - min_val) / (max_val - min_val + 1e-6) * 2.0 - 1.0
                elif norm_type == "sincos":
                    if not key.startswith("observation.state."):
                        raise ValueError(f"sincos normalization is only allowed for state keys, got {key}")
                    if isinstance(val, torch.Tensor):
                        normalized_value = torch.cat([torch.cos(val), torch.sin(val)], dim=-1)
                    else:
                        normalized_value = np.concatenate([np.cos(val), np.sin(val)], axis=-1)
                elif norm_type == "identity":
                    normalized_value = value
                else:
                    raise ValueError(
                        f"Unknown normalization type: {norm_type}. Supported types are 'meanstd', 'minmax', 'sincos', and 'identity'."
                    )
                normalized_data[key] = normalized_value
            else:
                # If the key is not in norm_stats, we assume no normalization is needed
                normalized_data[key] = value
        return normalized_data

    def unnormalize(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Unnormalize the given data using stored normalization statistics.

        Args:
            data (Dict[str, np.ndarray]): Dictionary of normalized arrays to unnormalize.

        Returns:
            Dict[str, np.ndarray]: Dictionary of unnormalized arrays.
        """
        unnormalized_data = {}
        for key, value in data.items():
            if key in self.norm_stats:
                norm_type = self.norm_type.get(key, "identity")
                if norm_type == "meanstd":
                    mean = self._get_stat(key, "mean", value)
                    std = self._get_stat(key, "std", value)
                    unnormalized_value = value * (std + 1e-6) + mean
                elif norm_type == "bounds_98" or norm_type == "bounds_98_woclip":
                    low = self._get_stat(key, "q02", value)
                    high = self._get_stat(key, "q98", value)
                    unnormalized_value = ((value + 1.0) / 2.0) * (high - low + 1e-6) + low
                elif norm_type == "bounds_99" or norm_type == "bounds_99_woclip":
                    low = self._get_stat(key, "q01", value)
                    high = self._get_stat(key, "q99", value)
                    unnormalized_value = ((value + 1.0) / 2.0) * (high - low + 1e-6) + low
                elif norm_type == "std":
                    std = self._get_stat(key, "std", value)
                    unnormalized_value = value * (std + 1e-6)
                elif norm_type == "minmax" or norm_type == "minmax_woclip":
                    min_val = self._get_stat(key, "min", value)
                    max_val = self._get_stat(key, "max", value)
                    # Reverse: (x + 1)/2 * (max-min+eps) + min
                    unnormalized_value = (value + 1) / 2.0 * (max_val - min_val + 1e-6) + min_val
                elif norm_type == "sincos":
                    if not key.startswith("observation.state."):
                        raise ValueError(f"sincos normalization is only allowed for state keys, got {key}")
                    if value.shape[-1] % 2 != 0:
                        raise ValueError(f"Cannot unnormalize sincos state with odd last dim for {key}: {value.shape}")
                    half_dim = value.shape[-1] // 2
                    cos_value = value[..., :half_dim]
                    sin_value = value[..., half_dim:]
                    if isinstance(value, torch.Tensor):
                        unnormalized_value = torch.atan2(sin_value, cos_value)
                    else:
                        unnormalized_value = np.arctan2(sin_value, cos_value)
                elif norm_type == "identity":
                    unnormalized_value = value
                else:
                    raise ValueError(
                        f"Unknown normalization type: {norm_type}. Supported types are 'meanstd', 'minmax', 'sincos', and 'identity'."
                    )
                unnormalized_data[key] = unnormalized_value
            else:
                # If no normalization was applied, return as-is
                unnormalized_data[key] = value
        return unnormalized_data


def resize_with_pad_item(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 3:
        raise ValueError(f"(c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[1:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img.unsqueeze(0), size=(resized_height, resized_width), mode="bilinear", align_corners=False
    ).squeeze(0)

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def canonicalize_visual(visual: Tensor | np.ndarray) -> Tensor:
    """Convert one image or video clip to channel-first torch layout.

    Supported layouts are ``HWC``, ``CHW``, ``THWC``, and ``TCHW``. A 2-D
    grayscale image is also accepted and becomes ``1HW``.
    """
    if not isinstance(visual, torch.Tensor):
        visual = torch.as_tensor(visual)

    channel_sizes = (1, 3, 4)
    if visual.ndim == 2:
        return visual.unsqueeze(0).contiguous()
    if visual.ndim == 3:
        if visual.shape[0] in channel_sizes:
            return visual.contiguous()
        if visual.shape[-1] in channel_sizes:
            return visual.permute(2, 0, 1).contiguous()
    elif visual.ndim == 4:
        if visual.shape[1] in channel_sizes:
            return visual.contiguous()
        if visual.shape[-1] in channel_sizes:
            return visual.permute(0, 3, 1, 2).contiguous()

    raise ValueError(f"Expected visual data in HWC, CHW, THWC, or TCHW layout, got shape {tuple(visual.shape)}")


def concatenate_visuals(
    visuals: Sequence[Tensor | np.ndarray],
    mode: str = "concat_height",
) -> Tensor:
    """Compose raw visual sources before any logical-camera resizing."""
    if mode != "concat_height":
        raise ValueError(f"Unsupported visual combine mode {mode!r}; expected 'concat_height'.")
    if len(visuals) < 2:
        raise ValueError("concat_height requires at least two visual sources.")

    canonical = [canonicalize_visual(visual) for visual in visuals]
    reference = canonical[0]
    for index, visual in enumerate(canonical[1:], start=1):
        if visual.ndim != reference.ndim:
            raise ValueError(
                "Visual sources must have the same rank before concatenation: "
                f"source 0 is {tuple(reference.shape)}, source {index} is {tuple(visual.shape)}"
            )
        if visual.shape[:-2] != reference.shape[:-2] or visual.shape[-1] != reference.shape[-1]:
            raise ValueError(
                "concat_height requires matching frame/channel/width dimensions: "
                f"source 0 is {tuple(reference.shape)}, source {index} is {tuple(visual.shape)}"
            )
        if visual.dtype != reference.dtype or visual.device != reference.device:
            raise ValueError("Visual sources must have the same dtype and device before concatenation.")

    # CHW -> C,(sum H),W and TCHW -> T,C,(sum H),W.
    return torch.cat(canonical, dim=-2)


def resize_visual(visual: Tensor | np.ndarray, size: int | Sequence[int]) -> Tensor:
    """Resize a channel-first image/video while preserving its value scale and dtype."""
    visual = canonicalize_visual(visual)
    if isinstance(size, int):
        target_size = (size, size)
    else:
        target_size = tuple(int(value) for value in size)
    if len(target_size) != 2 or min(target_size) <= 0:
        raise ValueError(f"Image size must contain two positive integers, got {size!r}.")
    if tuple(visual.shape[-2:]) == target_size:
        return visual

    squeeze = visual.ndim == 3
    batched = visual.unsqueeze(0) if squeeze else visual
    original_dtype = visual.dtype
    resized = F.interpolate(
        batched.to(torch.float32),
        size=target_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    if original_dtype == torch.uint8:
        resized = resized.round().clamp(0, 255).to(original_dtype)
    elif visual.is_floating_point():
        resized = resized.to(original_dtype)
    else:
        resized = resized.round().to(original_dtype)
    return resized.squeeze(0) if squeeze else resized


def _visual_hw(visual: Tensor) -> tuple[int, int]:
    if visual.ndim == 3:
        return int(visual.shape[1]), int(visual.shape[2])
    if visual.ndim == 4:
        return int(visual.shape[2]), int(visual.shape[3])
    raise ValueError(f"Expected image (C,H,W) or video (T,C,H,W), got {tuple(visual.shape)}")


def sample_visual_augmentation_params(
    reference: Tensor,
) -> dict:
    """Sample one replayable augmentation config for all views/frames in a sample."""
    _visual_hw(reference)
    device = reference.device
    return {
        "brightness": 0.7 + torch.rand((), device=device) * 0.6,
        "contrast": 0.6 + torch.rand((), device=device) * 0.8,
        "saturation": 0.5 + torch.rand((), device=device),
    }


def apply_visual_augmentation(
    visual: Tensor,
    params: dict,
) -> Tensor:
    """Apply the same sampled brightness, contrast and saturation to every frame.

    Accepts images with shape (C,H,W) and video clips with shape (T,C,H,W).
    The output keeps the input shape and dtype, so it can be reused before image
    or video processors.
    """
    if visual.ndim == 3:
        visual_bchw = visual.unsqueeze(0)
        squeeze = True
    elif visual.ndim == 4:
        visual_bchw = visual
        squeeze = False
    else:
        raise ValueError(f"Expected image (C,H,W) or video (T,C,H,W), got {tuple(visual.shape)}")

    orig_dtype = visual.dtype
    image = visual_bchw.to(torch.float32)
    if orig_dtype == torch.uint8 or image.max() > 1.0:
        image = image / 255.0

    brightness = params["brightness"].to(device=image.device, dtype=image.dtype)
    contrast = params["contrast"].to(device=image.device, dtype=image.dtype)
    saturation = params["saturation"].to(device=image.device, dtype=image.dtype)
    image = image * brightness
    mean = image.mean(dim=[1, 2, 3], keepdim=True)
    image = (image - mean) * contrast + mean
    gray = image.mean(dim=1, keepdim=True)
    image = gray + (image - gray) * saturation
    image = image.clamp(0.0, 1.0)

    if orig_dtype == torch.uint8:
        image = (image * 255.0).round().to(torch.uint8)
    else:
        image = image.to(orig_dtype)
    return image.squeeze(0) if squeeze else image


def history_offsets(memory: int, stride: int) -> list[int]:
    """Frame offsets from oldest history to current observation (inclusive)."""
    if type(memory) is not int or memory < 0 or type(stride) is not int or stride < 1:
        raise ValueError("history_memory must be a non-negative integer and history_stride a positive integer.")
    return list(range(-memory * stride, 1, stride))


def resize_rgb_uint8(visual: Tensor, size: int) -> Tensor:
    visual = canonicalize_visual(visual)
    if visual.shape[-3] != 3:
        raise ValueError(f"Expected an RGB camera observation, got {tuple(visual.shape)}.")
    if visual.is_floating_point():
        # Detect [0,1] vs [0,255] BEFORE bilinear+antialias interpolation, which
        # can slightly overshoot 1.0 (e.g. 1.0000001) on bright pixels.
        is_unit_scale = bool(visual.numel() > 0 and visual.amax() <= 1.0 + 1e-3)
        visual = resize_visual(visual, size)
        if is_unit_scale:
            visual = visual.clamp(0.0, 1.0) * 255.0
        return visual.round().clamp(0, 255).to(torch.uint8)
    if visual.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 or floating point RGB, got {visual.dtype}.")
    return resize_visual(visual, size)


def prepare_images(item, wrist_keys, head_key, config, processor, *, augment=False, return_raw=False):
    """Prepare the wrist pair for the spatial_encoder and optional head images for Qwen."""
    wrists = torch.stack([resize_rgb_uint8(item[key], config.wrist_image_size) for key in wrist_keys])
    if wrists.ndim != 4 or wrists.shape[0] != 2:
        raise ValueError("Expected two current wrist images ordered left, right.")
    params = sample_visual_augmentation_params(wrists) if augment else None
    if params is not None:
        wrists = apply_visual_augmentation(wrists, params)
    pixels = wrists.float().div_(255)
    mean = pixels.new_tensor(config.wrist_image_mean).view(3, 1, 1)
    std = pixels.new_tensor(config.wrist_image_std).view(3, 1, 1)
    result = {"wrist_images": (pixels - mean) / std}
    if return_raw:
        result["raw_wrist_images"] = wrists

    if head_key is not None:
        head = resize_rgb_uint8(item[head_key], config.head_image_size)
        # Video and embedded image backends can differ on the singleton time axis.
        if head.ndim == 3:
            head = head.unsqueeze(0)
        count = config.history_memory + 1
        if head.shape[0] != count:
            raise ValueError(f"Expected {count} chronological head frames, got {head.shape[0]}.")
        if params is not None:
            head = apply_visual_augmentation(head, params)
        processed = processor.image_processor(images=list(head.unbind()), do_resize=False, return_tensors="pt")
        patches = processed["pixel_values"]
        result["head_images"] = patches.reshape(count, -1, patches.shape[-1])
        result["image_grid_thw"] = processed["image_grid_thw"].reshape(count, 3)
        if return_raw:
            result["raw_head_images"] = head
    return result


def prepare_state(observation: dict[str, Tensor], max_state_dim):
    """Pad the state to the maximum state dimension.

    Args:
        observation (dict[str, Tensor])

    Returns:
        state (torch.Tensor): (*b, max_state_dim) padded state tensor
    """
    state = observation["state"]
    state = F.pad(state, (0, max_state_dim - state.shape[-1]))
    return state


def prepare_action(observation: dict[str, Tensor], max_action_dim):
    """Pad the action to the maximum action dimension.

    Args:
        observation (dict[str, Tensor])

    Returns:
        action (torch.Tensor): (*b, n, max_action_dim) padded action tensor
        action_dim (int): the actual dimension of the action before padding
    """
    action = observation["action"]
    action = F.pad(action, (0, max_action_dim - action.shape[-1]))
    return action


def prepare_language(config, language_tokenizer, observation: dict[str, Tensor]):
    """Tokenize a prompt with the Qwen3.5 chat template.

    If ``lang_tokens`` and ``lang_masks`` are provided, use them directly.

    Args:
        observation (dict[str, Tensor])

    Returns:
        lang_tokens (torch.Tensor): (*b, l) language tokens
        lang_masks (torch.Tensor): (*b, l) masks for language tokens, True if token is present, False if missing
    """
    lang_tokens = observation.get("lang_tokens", None)
    lang_masks = observation.get("lang_masks", None)
    prompt = observation.get("prompt", None)

    # either provide `prompt` or (`lang_tokens`, `lang_masks`)
    if prompt is None and (lang_tokens is None or lang_masks is None):
        raise ValueError("Either 'prompt' or ('lang_tokens', 'lang_masks') must be provided in the observation.")

    if prompt is not None and (lang_tokens is None or lang_masks is None):
        max_length = int(config.tokenizer_max_length)
        cache_key = None
        if isinstance(prompt, (list, tuple)) and len(prompt) == 1 and isinstance(prompt[0], str):
            cache_key = (id(language_tokenizer), max_length, prompt[0])
            cached = _LANGUAGE_TOKEN_CACHE.get(cache_key)
            if cached is not None:
                _LANGUAGE_TOKEN_CACHE.move_to_end(cache_key)
                return cached[0].clone(), cached[1].clone()
        formatted_prompt = [
            language_tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=False,
            )
            for p in prompt
        ]
        tokenized_prompt = language_tokenizer(
            formatted_prompt,
            padding="max_length",
            padding_side="right",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"]
        lang_masks = tokenized_prompt["attention_mask"]
        out_tokens = lang_tokens.squeeze(0)
        out_masks = lang_masks.to(device=lang_tokens.device, dtype=torch.bool).squeeze(0)
        if cache_key is not None:
            _LANGUAGE_TOKEN_CACHE[cache_key] = (out_tokens.clone(), out_masks.clone())
            if len(_LANGUAGE_TOKEN_CACHE) > _TOKEN_CACHE_MAX_SIZE:
                _LANGUAGE_TOKEN_CACHE.popitem(last=False)
        return out_tokens, out_masks

    return lang_tokens.squeeze(0), lang_masks.to(device=lang_tokens.device, dtype=torch.bool).squeeze(0)


def prepare_subtask(config, language_tokenizer, subtask):
    """Right-pad the current subtask target, reserving its final token for EOS.

    No BOS/chat prefix is inserted: the model appends these tokens immediately
    after the fixed 72-token task region and shifts labels for teacher forcing.
    """
    if language_tokenizer is None:
        raise ValueError("Subtask supervision requires a language tokenizer.")
    length = int(getattr(config, "subtask_max_length", 100))
    eos = language_tokenizer.eos_token_id
    pad = language_tokenizer.pad_token_id
    if eos is None or pad is None:
        raise ValueError("The language tokenizer must define EOS and padding tokens.")
    if length < 1:
        raise ValueError("subtask_max_length must be positive.")
    # Missing labels retain action supervision without contributing an EOS target.
    if subtask is None or (isinstance(subtask, str) and not subtask.strip()):
        return torch.full((length,), pad, dtype=torch.long), torch.zeros(length, dtype=torch.bool)
    if not isinstance(subtask, str):
        raise TypeError("A subtask annotation must be a string or None.")
    cache_key = (id(language_tokenizer), length, eos, pad, subtask)
    cached = _SUBTASK_TOKEN_CACHE.get(cache_key)
    if cached is not None:
        _SUBTASK_TOKEN_CACHE.move_to_end(cache_key)
        return cached[0].clone(), cached[1].clone()
    tokens = torch.full((length,), pad, dtype=torch.long)
    mask = torch.zeros(length, dtype=torch.bool)
    encoded = language_tokenizer(subtask, add_special_tokens=False, truncation=True, max_length=length - 1)[
        "input_ids"
    ]
    encoded = list(encoded)[: length - 1] + [eos]
    tokens[: len(encoded)] = torch.tensor(encoded, dtype=torch.long)
    mask[: len(encoded)] = True
    _SUBTASK_TOKEN_CACHE[cache_key] = (tokens.clone(), mask.clone())
    if len(_SUBTASK_TOKEN_CACHE) > _TOKEN_CACHE_MAX_SIZE:
        _SUBTASK_TOKEN_CACHE.popitem(last=False)
    return tokens, mask
