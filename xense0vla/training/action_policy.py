"""Standalone dual-wrist action policy training."""

from dataclasses import dataclass, field
from typing import Optional

from xense0vla.training.arguments import PolicyDataArguments, PolicyTrainingArguments, validate_training_configuration
from xense0vla.training.recipe import build_policy_model


@dataclass
class ActionModelArguments:
    spatial_encoder_path: Optional[str] = None
    action_policy_path: Optional[str] = None
    action_expert_intermediate_size: int = 3072

    def __post_init__(self):
        if bool(self.spatial_encoder_path) == bool(self.action_policy_path):
            raise ValueError("Choose one initialization source: spatial_encoder_path or action_policy_path.")


@dataclass
class ActionTrainingArguments(PolicyTrainingArguments):
    spatial_encoder_lr: Optional[float] = field(
        default=None, metadata={"help": "LR for the pretrained spatial_encoder backbone (defaults to one tenth of lr)."}
    )


@dataclass
class ActionPolicyArguments:
    model: ActionModelArguments = field(default_factory=ActionModelArguments)
    data: PolicyDataArguments = field(default_factory=PolicyDataArguments)
    train: ActionTrainingArguments = field(default_factory=ActionTrainingArguments)


class ActionPolicyTraining:
    """Own the standalone model, image processor and action-only batch contract."""

    arguments = ActionPolicyArguments
    name = "action_policy"

    def validate(self, args):
        validate_training_configuration(args, standalone=True)
        if args.model.action_policy_path:
            if args.train.spatial_encoder_lr is not None:
                raise ValueError("Action post-training uses train.lr for the entire policy.")
        else:
            if args.train.spatial_encoder_lr is None:
                args.train.spatial_encoder_lr = 0.1 * args.train.lr
            if args.train.spatial_encoder_lr >= args.train.lr:
                raise ValueError("Action policy pretraining requires spatial_encoder_lr < lr for the pretrained backbone.")

    def build_model(self, args):
        return build_policy_model(args, standalone_action_policy=True)

    def build_processor(self, args):
        return None

    def forward(self, model, batch):
        if any(key in batch for key in ("lang_tokens", "lang_masks", "subtask_tokens", "subtask_masks")):
            raise ValueError("Standalone action batches must not contain task or subtask tokens.")
        return model(**batch)
