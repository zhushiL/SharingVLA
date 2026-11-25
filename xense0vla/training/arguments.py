"""Shared runtime arguments and complete VLA arguments; YAML files remain independent."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from xense0vla.utils.arguments import DataArguments, ModelArguments, TrainingArguments


@dataclass
class PolicyTrainingArguments(TrainingArguments):
    """Optimization and runtime settings shared by both entry points."""

    fused_optimizer: bool = field(
        default=False,
        metadata={"help": "Use PyTorch's fused CUDA AdamW implementation."},
    )
    performance_log_steps: int = field(
        default=50,
        metadata={
            "help": (
                "Log cross-rank data-wait and CUDA-stage timings every N steps; "
                "0 disables the extra timing collective."
            )
        },
    )
    empty_cache_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Number of steps between CUDA allocator cache flushes; "
                "0 keeps the cache for stable training throughput."
            )
        },
    )
    action_loss_weight: float = field(default=1.0)
    action_dim: int = field(
        default=7,
        metadata={"help": "Action dimension."},
    )
    max_action_dim: int = field(
        default=32,
        metadata={"help": "Action dimension after padding."},
    )
    max_state_dim: int = field(
        default=32,
        metadata={"help": "State dimension after padding."},
    )
    chunk_size: int = field(
        default=20,
        metadata={"help": "Number of action tokens predicted per sample."},
    )
    num_steps: int = field(
        default=10,
        metadata={"help": ("Number of Euler denoising steps used during action sampling.")},
    )
    loss_type: str = field(
        default="fm",
        metadata={"help": "Which loss to use."},
    )
    rtc_training: Optional[Dict[str, Any]] = field(
        default_factory=dict,
        metadata={
            "help": ("RTC clean-prefix settings. Enable explicitly; integer delays are sampled uniformly.")
        },
    )
    decayed_max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum norm for the decayed gradients."},
    )
    stable_train_steps: int = field(
        default=100000,
        metadata={
            "help": "Training steps for stable training, after this step, the decayed_max_grad_norm will be applied."
        },
    )
    resume_dataloader_state: bool = field(
        default=True,
        metadata={"help": "Whether to resume dataloader."},
    )
    parameter_dtype: Literal["float32", "bfloat16"] = field(
        default="float32",
        metadata={
            "help": "Training parameter/AdamW-state dtype; defaults to FP32 with BF16 autocast. Deployment weights export as BF16."
        },
    )


@dataclass
class VLATrainingArguments(PolicyTrainingArguments):
    """Language supervision and backbone options for complete VLA training."""

    vlm_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Optional VLM learning rate; defaults to train.lr."},
    )
    freeze_action_policy: bool = field(default=True)
    unfreeze_mot_layers: bool = field(default=False)
    use_subtask_loss: bool = field(
        default=False,
        metadata={"help": "Optional in mid_train and post_train; absent for standalone action training."},
    )
    subtask_max_length: int = field(default=100)
    language_loss_weight: float = field(default=1.0)
    history_memory: Optional[int] = field(default=None, metadata={"help": "Historical head frames. Defaults to 3 or the full VLA checkpoint value."})
    history_stride: Optional[int] = field(default=None, metadata={"help": "Head-frame interval. Defaults to 30 or the full VLA checkpoint value."})
    tokenizer_max_length: int = field(
        default=72,
        metadata={"help": "Maximum length of the tokenizer."},
    )
    vlm_fsdp: bool = field(
        default=False,
        metadata={"help": "Whether to apply FSDP2 for VLM."},
    )


@dataclass
class PolicyDataArguments(DataArguments):
    sampling_seed: int = field(
        default=42,
        metadata={"help": "Fixed seed for per-dataset frame subsets; shared by training and normalization."},
    )
    persistent_workers: bool = field(
        default=True,
        metadata={"help": "Keep DataLoader workers alive across iterator recreation."},
    )
    source_name: str = field(
        default=None,
        metadata={"help": "Source name of dataset."},
    )
    robot_config_root: str = field(
        default=None,
        metadata={"help": "Path to get all robot configs."},
    )
    joints: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The order of joints and their dim"},
    )
    cameras: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Logical cameras decoded and preprocessed by the data pipeline."},
    )
    norm_type: Optional[List[str]] = field(default=None, metadata={"help": "Normalization type."})
    norm_stats_file: str = field(
        default=None,
        metadata={"help": "Path to the normalization stats file."},
    )
    video_backend: Literal["torchcodec", "pyav"] = field(default="torchcodec")


@dataclass
class VLADataArguments(PolicyDataArguments):
    prompt_type: Literal["global", "subtask", "both"] = field(
        default="global",
        metadata={"help": "Global task input; subtask is a separate supervised target."},
    )


@dataclass
class VLAArguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "VLADataArguments" = field(default_factory=VLADataArguments)
    train: "VLATrainingArguments" = field(default_factory=VLATrainingArguments)


def validate_training_configuration(args, *, standalone: bool) -> None:
    """Validate independent architecture, loss and RTC options before model allocation."""
    if getattr(args.train, "rmpad", False) or getattr(args.train, "rmpad_with_pos_ids", False):
        raise ValueError(
            "VLA requires regular padded batches; set both rmpad and rmpad_with_pos_ids to false."
        )
    use_subtask = bool(getattr(args.train, "use_subtask_loss", False))
    if standalone and use_subtask:
        raise ValueError("Standalone action policy training has no subtask objective.")
    if hasattr(args.train, "use_subtask_loss"):
        args.train.use_subtask_loss = use_subtask
    if standalone and getattr(args.train, "freeze_action_policy", False):
        raise ValueError("Standalone training must train its action policy.")
    if getattr(args.train, "tokenizer_max_length", 72) != 72 or getattr(args.train, "subtask_max_length", 100) != 100:
        raise ValueError("The task and subtask regions must contain exactly 72 and 100 token slots.")
    memory = getattr(args.train, "history_memory", None)
    stride = getattr(args.train, "history_stride", None)
    if (memory is not None and memory < 0) or (stride is not None and stride < 1):
        raise ValueError("history_memory must be non-negative and history_stride positive.")
    if getattr(args.train, "unfreeze_mot_layers", False) and not args.train.freeze_action_policy:
        raise ValueError("unfreeze_mot_layers applies when freeze_action_policy is enabled.")
    for name in ("lr", "vlm_lr", "spatial_encoder_lr"):
        value = getattr(args.train, name, None)
        if value is not None and value <= 0:
            raise ValueError(f"train.{name} must be positive.")
    if getattr(args.data, "prompt_type", "global") != "global":
        raise ValueError("The task prompt must be global; subtask labels are separate targets.")
    expected = ["camera_wrist_left", "camera_wrist_right"]
    if not standalone:
        expected.insert(0, "camera_top")
    if list(args.data.cameras or expected) != expected:
        raise ValueError(f"cameras must preserve this order: {expected}.")
    rtc = dict(args.train.rtc_training or {})
    allowed_rtc_keys = {"enabled", "max_delay_steps", "delay_sampling"}
    unknown_rtc_keys = set(rtc) - allowed_rtc_keys
    if unknown_rtc_keys:
        if any(key in unknown_rtc_keys for key in ("delay_weights", "zero_delay_probability")):
            raise ValueError("RTC samples every integer delay uniformly; sampling weights are unsupported.")
        raise ValueError(f"Unknown rtc_training fields: {sorted(unknown_rtc_keys)}")
    if rtc.get("delay_sampling", "uniform") != "uniform":
        raise ValueError("RTC samples every integer delay uniformly; sampling weights are unsupported.")
    rtc["enabled"] = bool(rtc.get("enabled", False))
    rtc["delay_sampling"] = "uniform"
    if rtc["enabled"]:
        max_delay = rtc.get("max_delay_steps", min(8, args.train.chunk_size - 1))
        if isinstance(max_delay, bool) or not isinstance(max_delay, int):
            raise TypeError("rtc_training.max_delay_steps must be an integer.")
        if not 1 <= max_delay < args.train.chunk_size:
            raise ValueError("rtc_training.max_delay_steps must lie in [1, chunk_size).")
        rtc["max_delay_steps"] = max_delay
    else:
        rtc["max_delay_steps"] = 0
    args.train.rtc_training = rtc
