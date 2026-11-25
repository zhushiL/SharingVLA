"""Configuration for the independent action policy and Qwen3.5 MoT model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from transformers import PretrainedConfig


@dataclass(frozen=True)
class ActionPolicyConfig:
    """Architecture of the standalone flow transformer and its attention interface."""

    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    rotary_dim: int = 64

    def __post_init__(self):
        if self.num_hidden_layers != 18 or self.hidden_size != 1024:
            raise ValueError("The wrist-conditioned policy requires 18 layers of width 1024.")
        if min(self.intermediate_size, self.num_attention_heads, self.num_key_value_heads, self.head_dim) <= 0:
            raise ValueError("Action attention and feed-forward dimensions must be positive.")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Action query heads must divide evenly into KV groups.")
        if not 0 < self.rotary_dim <= self.head_dim or self.rotary_dim % 2 or min(self.rope_theta, self.rms_norm_eps) <= 0:
            raise ValueError("Invalid action rotary embedding or normalization configuration.")


class VLAConfig(PretrainedConfig):
    model_type = "vla"
    is_composition = True
    has_no_defaults_at_init = True

    def __init__(
        self,
        vlm_path=None,
        tokenizer_path=None,
        standalone_action_policy=False,
        action_policy_path=None,
        spatial_encoder_path=None,
        freeze_action_policy=None,
        unfreeze_mot_layers=False,
        action_config=None,
        wrist_image_size=224,
        head_image_size=256,
        wrist_image_mean=(0.485, 0.456, 0.406),
        wrist_image_std=(0.229, 0.224, 0.225),
        action_dim=14,
        max_action_dim=14,
        max_state_dim=14,
        chunk_size=20,
        tokenizer_max_length=72,
        subtask_max_length=100,
        history_memory=3,
        history_stride=30,
        use_subtask_loss=False,
        language_loss_weight=1.0,
        action_loss_weight=1.0,
        loss_type="fm",
        action_expert_hidden_size=1024,
        action_expert_intermediate_size=3072,
        joint_attention_backend="sdpa",
        backbone_attention_implementation="sdpa",
        vit_attn_implementation=None,
        rtc_training=None,
        num_steps=10,
        use_compile=False,
        backbone_config=None,
        **kwargs,
    ):
        # Only model parameters and Transformers metadata belong in config.json.
        metadata_fields = set(PretrainedConfig().to_dict()) | {
            "tie_word_embeddings", "attn_implementation", "_commit_hash",
        }
        unknown = set(kwargs) - metadata_fields
        if unknown:
            raise ValueError(f"Unknown VLA configuration fields: {sorted(unknown)}")
        if kwargs.pop("model_type", "vla") != "vla":
            raise ValueError("model_type must be 'vla'.")
        if kwargs.get("architectures") not in (None, ["VLAModel"]):
            raise ValueError("architectures must be ['VLAModel'].")
        if kwargs.pop("tie_word_embeddings", False):
            raise ValueError("VLA does not tie policy embeddings.")
        super().__init__(tie_word_embeddings=False, **kwargs)
        self.standalone_action_policy = bool(standalone_action_policy)
        self.use_subtask_loss = bool(use_subtask_loss)
        if self.standalone_action_policy and self.use_subtask_loss:
            raise ValueError("An independent action policy has no language supervision.")
        if freeze_action_policy is None:
            freeze_action_policy = not self.standalone_action_policy
        if unfreeze_mot_layers and (self.standalone_action_policy or not freeze_action_policy):
            raise ValueError("unfreeze_mot_layers requires a VLA with freeze_action_policy enabled.")
        if self.standalone_action_policy and freeze_action_policy:
            raise ValueError("Independent action-policy training must train its action policy.")
        if tokenizer_max_length != 72 or subtask_max_length != 100:
            raise ValueError("The task and subtask regions must have 72 and 100 tokens respectively.")
        if type(history_memory) is not int or history_memory < 0:
            raise ValueError("history_memory must be a non-negative integer.")
        if type(history_stride) is not int or history_stride < 1:
            raise ValueError("history_stride must be a positive integer.")
        if (wrist_image_size, head_image_size) != (224, 256):
            raise ValueError("The policy expects 224px wrist images and 256px head images.")
        if len(wrist_image_mean) != 3 or len(wrist_image_std) != 3 or any(v <= 0 for v in wrist_image_std):
            raise ValueError("Wrist normalization needs three RGB means and three positive standard deviations.")
        if not 0 < action_dim <= max_action_dim:
            raise ValueError("Expected 0 < action_dim <= max_action_dim.")
        if min(max_state_dim, chunk_size, num_steps) <= 0:
            raise ValueError("State dimension, action horizon and denoising steps must be positive.")
        if loss_type not in {"fm", "L1_fm"}:
            raise ValueError("loss_type must be 'fm' or 'L1_fm'.")
        if joint_attention_backend != "sdpa":
            raise ValueError("MoT joint attention currently requires sdpa.")
        if language_loss_weight < 0 or action_loss_weight < 0:
            raise ValueError("Loss weights must be non-negative.")
        self.architectures = ["VLAModel"]
        self.vlm_path = vlm_path
        self.tokenizer_path = tokenizer_path or vlm_path
        self.backbone_config = backbone_config
        self.action_policy_path = action_policy_path
        self.spatial_encoder_path = spatial_encoder_path
        self.unfreeze_mot_layers = bool(unfreeze_mot_layers)
        self.wrist_image_size, self.head_image_size = wrist_image_size, head_image_size
        self.wrist_image_mean, self.wrist_image_std = list(wrist_image_mean), list(wrist_image_std)
        self.freeze_action_policy = bool(freeze_action_policy)
        self.action_dim, self.max_action_dim, self.max_state_dim = action_dim, max_action_dim, max_state_dim
        self.chunk_size = chunk_size
        self.tokenizer_max_length, self.subtask_max_length = tokenizer_max_length, subtask_max_length
        self.history_memory, self.history_stride = history_memory, history_stride
        self.language_loss_weight, self.action_loss_weight = float(language_loss_weight), float(action_loss_weight)
        if action_config is None:
            action_config = dict(hidden_size=action_expert_hidden_size, intermediate_size=action_expert_intermediate_size)
        self.action_config = asdict(ActionPolicyConfig(**action_config))
        self.joint_attention_backend = joint_attention_backend
        self.backbone_attention_implementation = backbone_attention_implementation
        self.vit_attn_implementation = vit_attn_implementation or backbone_attention_implementation
        self.loss_type, self.num_steps, self.use_compile = loss_type, num_steps, use_compile
        rtc = dict(rtc_training or {})
        allowed_rtc_keys = {"enabled", "max_delay_steps", "delay_sampling"}
        unknown_rtc_keys = set(rtc) - allowed_rtc_keys
        if unknown_rtc_keys:
            if any(key in unknown_rtc_keys for key in ("delay_weights", "zero_delay_probability")):
                raise ValueError("RTC samples all allowed delays uniformly; sampling weights are not supported.")
            raise ValueError(f"Unknown rtc_training fields: {sorted(unknown_rtc_keys)}")
        if rtc.get("delay_sampling", "uniform") != "uniform":
            raise ValueError("RTC delay_sampling must be uniform.")
        enabled = bool(rtc.get("enabled", False))
        if enabled:
            max_delay = rtc.get("max_delay_steps", min(8, chunk_size - 1))
            if isinstance(max_delay, bool) or not isinstance(max_delay, int):
                raise TypeError("RTC max_delay_steps must be an integer.")
            if not 1 <= max_delay < chunk_size:
                raise ValueError("Enabled RTC max_delay_steps must be in [1, chunk_size).")
        else:
            max_delay = 0
        self.rtc_training = {
            "enabled": enabled,
            "max_delay_steps": max_delay,
            "delay_sampling": "uniform",
        }

    @property
    def rtc_training_enabled(self):
        return self.rtc_training["enabled"]

    @property
    def rtc_max_delay_steps(self):
        return self.rtc_training["max_delay_steps"]


__all__ = ["ActionPolicyConfig", "VLAConfig"]
