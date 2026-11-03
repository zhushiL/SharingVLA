"""Standalone wrist/state policy training, with no language-model inputs."""

from dataclasses import dataclass, field
from typing import Literal, Optional

from xense0vla.models.auto import build_image_processor
from xense0vla.training.arguments import PolicyDataArguments, PolicyTrainingArguments, validate_training_configuration
from xense0vla.training.recipe import build_policy_model


@dataclass
class ActionModelArguments:
    vlm_path: str
    action_policy_path: Optional[str] = None
    action_expert_hidden_size: int = 1024
    action_expert_intermediate_size: int = 3072
    joint_attention_backend: Literal["sdpa"] = "sdpa"
    vit_attn_implementation: str = "sdpa"

    def __post_init__(self):
        if not self.vlm_path:
            raise ValueError("vlm_path is required to load the Qwen vision encoder.")


@dataclass
class ActionPolicyArguments:
    model: ActionModelArguments = field(default_factory=ActionModelArguments)
    data: PolicyDataArguments = field(default_factory=PolicyDataArguments)
    train: PolicyTrainingArguments = field(default_factory=PolicyTrainingArguments)


class ActionPolicyTraining:
    """Own the standalone model, image processor and action-only batch contract."""

    arguments = ActionPolicyArguments
    name = "action_policy"

    def validate(self, args):
        validate_training_configuration(args, standalone=True)

    def build_model(self, args):
        return build_policy_model(args, standalone_action_policy=True)

    def build_processor(self, args):
        return build_image_processor(args.model.vlm_path)

    def forward(self, model, batch):
        if any(key in batch for key in ("lang_tokens", "lang_masks", "subtask_tokens", "subtask_masks")):
            raise ValueError("Standalone action batches must not contain task or subtask tokens.")
        if batch["img_masks"].shape[-1] != 2:
            raise ValueError("Standalone action batches require exactly left and right wrist images.")
        return model(**batch)
