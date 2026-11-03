"""Complete VLA training with task conditioning and optional subtask loss."""

from xense0vla.models import build_processor
from xense0vla.training.arguments import VLAArguments, validate_training_configuration
from xense0vla.training.recipe import build_policy_model


class VLATraining:
    arguments = VLAArguments
    name = "vla"

    def validate(self, args):
        validate_training_configuration(args, standalone=False)

    def build_model(self, args):
        return build_policy_model(args, standalone_action_policy=False)

    def build_processor(self, args):
        return build_processor(args.model.tokenizer_path)

    def forward(self, model, batch):
        if "lang_tokens" not in batch or "lang_masks" not in batch:
            raise ValueError("VLA batches require task tokens and their mask.")
        if batch["img_masks"].shape[-1] != 4:
            raise ValueError("VLA batches require two head frames and two wrist images.")
        return model(**batch)
