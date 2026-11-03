"""WebSocket CLI; import robot policies from xense0vla.inference for local use."""

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xense0vla.inference import ActionPolicy, VLAPolicy  # noqa: E402
from xense0vla.inference.policy import RobotPolicy, set_seed_everywhere  # noqa: E402


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

try:
    from .websocket_policy_server import WebsocketPolicyServer
except ImportError:
    from deploy.websocket_policy_server import WebsocketPolicyServer

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Launch the Qwen3.5 VLA WebSocket policy server")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "Standalone hf_ckpt directory containing VLA weights, "
            "configs, the vlm_backbone processor/tokenizer bundle, "
            "normalization statistics, robot configs, and vla_cli.yaml."
        ),
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="WebSocket bind address (use 127.0.0.1 for local-only access)",
    )

    parser.add_argument(
        "--use_length",
        type=int,
        default=-1,
        help=(
            "Number of actions returned/consumed from each model chunk; "
            "-1 uses the checkpoint's complete chunk_size"
        ),
    )

    parser.add_argument(
        "--chunk_ret",
        type=str2bool,
        default=True,
        help="Return the action chunk (true) or one action per request (false)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8006,
        help="WebSocket server port",
    )

    parser.add_argument(
        "--robot_name",
        default=None,
        help="Robot config name; defaults to data.data_name in the checkpoint manifest",
    )

    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Inference dtype; bfloat16 is recommended on supported GPUs",
    )

    parser.add_argument(
        "--use_compile",
        "--use-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the Qwen3.5 action inference path after eager loading.",
    )

    parser.add_argument("--policy", choices=("auto", "action_policy", "vla"), default="auto")
    args = parser.parse_args()

    set_seed_everywhere(42)
    policy_class = {"auto": RobotPolicy, "action_policy": ActionPolicy, "vla": VLAPolicy}[args.policy]
    model = policy_class(
        args.model_path,
        use_length=args.use_length,
        chunk_ret=args.chunk_ret,
        use_bf16=args.dtype == "bfloat16",
        use_fp32=args.dtype == "float32",
        use_compile=args.use_compile,
        robot_name=args.robot_name,
    )
    model_server = WebsocketPolicyServer(
        model,
        host=args.host,
        port=args.port,
        metadata=model.get_metadata(),
    )
    model_server.serve_forever()


if __name__ == "__main__":
    main()
