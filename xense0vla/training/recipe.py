"""Model construction shared by the two training recipes."""

from inspect import signature

from xense0vla.models import build_foundation_model
from xense0vla.models.configuration_vla import VLAConfig


def build_policy_model(args, *, standalone_action_policy):
    model_fields = signature(VLAConfig.__init__).parameters
    values = {key: value for key, value in {**vars(args.model), **vars(args.train)}.items() if key in model_fields}
    config = VLAConfig(**values, standalone_action_policy=standalone_action_policy)
    return build_foundation_model(
        config=config,
        weights_path=getattr(args.model, "model_path", None),
        torch_dtype=args.train.parameter_dtype,
        init_device=args.train.init_device,
    )
