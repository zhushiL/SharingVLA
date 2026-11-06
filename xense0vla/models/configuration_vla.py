"""Configuration for the independent action policy and Qwen3.5 MoT model."""

from __future__ import annotations

from transformers import PretrainedConfig


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
        freeze_vision_encoder=True,
        freeze_action_policy=False,
        action_dim=14,
        max_action_dim=14,
        max_state_dim=14,
        chunk_size=20,
        tokenizer_max_length=72,
        subtask_max_length=100,
        head_history_offset=30,
        use_subtask_loss=False,
        language_loss_weight=1.0,
        action_loss_weight=1.0,
        loss_type="fm",
        action_expert_hidden_size=1024,
        action_expert_intermediate_size=3072,
        action_policy_num_layers=18,
        environment_num_heads=8,
        use_state=False,
        joint_attention_backend="sdpa",
        backbone_attention_implementation="sdpa",
        vit_attn_implementation=None,
        rtc_training=None,
        num_steps=10,
        use_compile=False,
        backbone_config=None,
        vlm_family="qwen3_5",
        vlm_backbone_subdir="vlm_backbone",
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
        if vlm_family != "qwen3_5" or vlm_backbone_subdir != "vlm_backbone":
            raise ValueError("VLA requires Qwen3.5 assets in vlm_backbone/.")
        super().__init__(tie_word_embeddings=False, **kwargs)
        self.standalone_action_policy = bool(standalone_action_policy)
        self.use_subtask_loss = bool(use_subtask_loss)
        if self.standalone_action_policy and self.use_subtask_loss:
            raise ValueError("An independent action policy has no language supervision.")
        if self.standalone_action_policy and freeze_action_policy:
            raise ValueError("Independent action-policy training must train its action policy.")
        if tokenizer_max_length != 72 or subtask_max_length != 100:
            raise ValueError("The task and subtask regions must have 72 and 100 tokens respectively.")
        if head_history_offset < 0:
            raise ValueError("head_history_offset must be non-negative.")
        if action_policy_num_layers != 18:
            raise ValueError("The action policy must contain 18 full-attention layers.")
        if action_expert_hidden_size <= 0 or action_expert_hidden_size % 2:
            raise ValueError("action_expert_hidden_size must be positive and even.")
        if action_expert_intermediate_size <= 0 or environment_num_heads <= 0:
            raise ValueError("Action MLP width and environment attention head count must be positive.")
        if 2 * action_expert_hidden_size % environment_num_heads:
            raise ValueError("The environment width must be divisible by environment_num_heads.")
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
        self.vlm_family = vlm_family
        self.vlm_backbone_subdir = vlm_backbone_subdir
        self.vlm_path = vlm_path
        self.tokenizer_path = tokenizer_path or vlm_path
        self.backbone_config = backbone_config
        self.action_policy_path = action_policy_path
        self.freeze_vision_encoder = bool(freeze_vision_encoder)
        self.freeze_action_policy = bool(freeze_action_policy)
        self.action_dim, self.max_action_dim, self.max_state_dim = action_dim, max_action_dim, max_state_dim
        self.chunk_size = chunk_size
        self.tokenizer_max_length, self.subtask_max_length = tokenizer_max_length, subtask_max_length
        self.head_history_offset = head_history_offset
        self.language_loss_weight, self.action_loss_weight = float(language_loss_weight), float(action_loss_weight)
        self.action_expert_hidden_size = action_expert_hidden_size
        self.action_expert_intermediate_size = action_expert_intermediate_size
        self.action_policy_num_layers = action_policy_num_layers
        self.environment_num_heads = environment_num_heads
        self.use_state = bool(use_state)
        self.joint_attention_backend = joint_attention_backend
        self.backbone_attention_implementation = backbone_attention_implementation
        self.vit_attn_implementation = vit_attn_implementation or backbone_attention_implementation
        self.loss_type, self.num_steps, self.use_compile = loss_type, num_steps, use_compile
        rtc = dict(rtc_training or {})
        if any(key in rtc for key in ("delay_weights", "zero_delay_probability")):
            raise ValueError("RTC samples all allowed delays uniformly; sampling weights are not supported.")
        if rtc.get("delay_sampling", "uniform") != "uniform":
            raise ValueError("RTC delay_sampling must be uniform.")
        enabled = bool(rtc.get("enabled", False))
        max_delay = rtc.get("max_delay_steps", min(8, chunk_size - 1) if enabled else 0)
        if isinstance(max_delay, bool) or not isinstance(max_delay, int):
            raise TypeError("RTC max_delay_steps must be an integer.")
        if enabled and not 1 <= max_delay < chunk_size:
            raise ValueError("Enabled RTC max_delay_steps must be in [1, chunk_size).")
        if max_delay < 0:
            raise ValueError("RTC max_delay_steps must be non-negative.")
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


__all__ = ["VLAConfig"]
