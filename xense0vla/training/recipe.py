"""Model construction shared by standalone policy and VLA training."""

from inspect import signature

from xense0vla.models import build_foundation_model
from xense0vla.models.configuration_vla import VLAConfig


def build_policy_model(args, *, standalone_action_policy):
    model_path = getattr(args.model, "model_path", None)
    action_path = args.model.action_policy_path
    values = {**vars(args.model), **vars(args.train)}
    fields = signature(VLAConfig.__init__).parameters
    values = {name: value for name, value in values.items() if name in fields and value is not None}
    values["standalone_action_policy"] = standalone_action_policy

    if model_path or action_path:
        source = VLAConfig.from_pretrained(model_path or action_path)
        if source.standalone_action_policy != bool(action_path):
            raise ValueError("model_path requires a complete VLA. action_policy_path requires a standalone policy.")
        if args.train.max_action_dim != source.max_action_dim:
            raise ValueError("max_action_dim must match the checkpoint action projections.")
        # The export owns the neural architecture and its input preprocessing.
        # Losses, freezing, sampling and RTC are selected by the new training stage.
        values.update(
            action_config=source.action_config,
            wrist_image_size=source.wrist_image_size,
            head_image_size=source.head_image_size,
            wrist_image_mean=source.wrist_image_mean,
            wrist_image_std=source.wrist_image_std,
        )
        if model_path:
            values["backbone_config"] = source.backbone_config
            values.setdefault("history_memory", source.history_memory)
            values.setdefault("history_stride", source.history_stride)

    config = VLAConfig(**values)
    return build_foundation_model(
        config=config,
        weights_path=model_path,
        torch_dtype=args.train.parameter_dtype,
        init_device=args.train.init_device,
    )
