"""Exercise the real training CLIs and exported robot policies on one CUDA GPU.

Uses the complete Qwen3.5-2B and actual robot data. Both training entries use
the default FP32 parameters/AdamW states and BF16 autocast computation.
Full VLA training requires more than 32 GiB unless memory is offloaded/sharded.
Each stage runs in its own process and exports resumable plus deployment weights.
These one-step artifacts are smoke tests, not trained robot controllers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STAGES = {
    "action_pretrain": ("action_policy", "configs/policy_action/pretrain.yaml"),
    "action_posttrain": ("action_policy", "configs/policy_action/post_train.yaml"),
    "mid_train": ("vla", "configs/policy_vla/mid_train.yaml"),
    "post_train": ("vla", "configs/policy_vla/post_train.yaml"),
}


def checkpoint(output, stage, step=1):
    return output / stage / "checkpoints" / f"global_step_{step}" / "hf_ckpt"


def verify_deployment(args):
    import numpy as np
    import torch

    from xense0vla.data.robot_dataset import VLADataset
    from xense0vla.inference import ActionPolicy, VLAPolicy

    stage = args.verify_stage
    standalone = stage.startswith("action_")
    policy_class = ActionPolicy if standalone else VLAPolicy
    policy = policy_class(str(checkpoint(args.output_dir, stage)), chunk_ret=True)
    robot_name = policy.data_config.data_name
    if standalone:
        assert policy.language_tokenizer is None
        assert not hasattr(policy.vla.model.backbone, "qwen35")
        assert not (checkpoint(args.output_dir, stage) / "vlm_backbone/tokenizer.json").exists()
    assert all(not p.requires_grad for p in policy.vla.model.backbone.vision_model.parameters())
    dataset = VLADataset(
        str(args.data_path), robot_name, policy.data_config, policy.data_config.robot_config_root,
        config=policy.config, processor=policy.processor, video_backend="pyav",
        chunk_size=policy.config.chunk_size, image_size=(256, 256),
        feature_transform=policy.vla.feature_transform, return_item=True,
    )
    raw = dataset.dataset[args.sample_index]
    # Robot observations must not require ground-truth actions or subtask labels.
    observation = {key: value for key, value in raw.items()
                   if not key.startswith("action") and key != "subtask"}
    if standalone:
        observation.pop("task", None)
    prefix_events = []
    if not standalone:
        encode_prefix = policy.vla.model.backbone.encode_prefix

        def traced(hidden, padding, positions, prefix_length):
            result = encode_prefix(hidden, padding, positions, prefix_length)
            prefix_events.append({"slots": hidden.shape[1], "kv_lengths": [k.shape[2] for k, _ in result[1]]})
            return result

        policy.vla.model.backbone.encode_prefix = traced
    result = policy.infer(observation)
    action = np.asarray(result["action"])
    assert action.shape == (50, 20) and np.isfinite(action).all()
    if not standalone:
        assert prefix_events == [{"slots": 204, "kv_lengths": [204] * 6}]
    rtc = policy.config.rtc_training_enabled
    if rtc:
        prefix = action[:2].copy()
        second = policy.infer({**observation, "rtc_action_prefix": prefix, "rtc_prefix_length": 2})
        np.testing.assert_array_equal(second["action"][:2], prefix)
        assert np.isfinite(second["action"]).all()
    policy.reset(robot_name)
    assert policy.last_action_chunk is None and not policy._head_histories
    report = {
        "policy_class": policy_class.__name__, "sample_index": args.sample_index,
        "episode_index": int(raw["episode_index"]), "frame_index": int(raw["frame_index"]),
        "camera_keys": sorted(dataset.dataset.selected_camera_keys),
        "action_shape": list(action.shape), "finite": True,
        "tokenizer_loaded": policy.language_tokenizer is not None,
        "vlm_prefix_calls": prefix_events, "rtc_prefix_preserved": rtc,
        "reset_passed": True, "device": torch.cuda.get_device_name(),
    }
    (args.output_dir / f"{stage}_deployment.json").write_text(json.dumps(report, indent=2))


def run_logged(command, path, env):
    print(f"Running {path.name}", flush=True)
    with path.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); inspect {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/training_entries_gpu")
    parser.add_argument("--data-path", type=Path, default=ROOT.parent / "data/bi_rebot_09_pick_and_place_1023-annotated")
    parser.add_argument("--model-path", type=Path, default=ROOT / "pretrained_models/Qwen3.5-2B")
    parser.add_argument("--stages", choices=tuple(STAGES), nargs="+", default=list(STAGES))
    parser.add_argument("--parameter-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--verify-only", action="store_true", help="Verify previously exported checkpoints without retraining.")
    parser.add_argument("--verify-stage", choices=tuple(STAGES), help=argparse.SUPPRESS)
    parser.add_argument("--sample-index", type=int, default=165)
    parser.add_argument("--robot-config-root", type=Path, help="Optional robot mappings for the validation dataset; the source configs are unchanged.")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.data_path = args.data_path.resolve()
    args.model_path = args.model_path.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.verify_stage:
        verify_deployment(args)
        return
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "4",
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    report = {}
    for stage in args.stages:
        kind, config = STAGES[stage]
        dtype = args.parameter_dtype
        log = args.output_dir / f"{stage}.log"
        if not args.verify_only:
            if checkpoint(args.output_dir, stage).exists():
                raise FileExistsError(f"Use a fresh output directory or --verify-only: {checkpoint(args.output_dir, stage)}")
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1",
                       f"tasks/train_{kind}.py", config, "--data.train_path", str(args.data_path),
                       "--train.output_dir", str(args.output_dir / stage), "--train.max_steps", "1",
                       "--train.micro_batch_size", "1", "--train.parameter_dtype", dtype,
                       "--train.seed", "127", "--data.num_workers", "0", "--data.video_backend", "pyav",
                       "--train.performance_log_steps", "1"]
            if args.robot_config_root:
                command += ["--data.robot_config_root", str(args.robot_config_root.resolve())]
            if stage != "post_train":
                command += ["--model.vlm_path", str(args.model_path)]
            if stage in {"action_posttrain", "mid_train"}:
                command += ["--model.action_policy_path", str(checkpoint(args.output_dir, "action_pretrain") / "action_policy")]
            elif stage == "post_train":
                command += ["--model.model_path", str(checkpoint(args.output_dir, "mid_train")), "--train.use_subtask_loss", "true"]
            run_logged(command, log, env)
        if args.verify_only:
            import yaml

            from xense0vla.utils.runtime_assets import TRAINING_CONFIG_NAME

            manifest = yaml.safe_load((checkpoint(args.output_dir, stage) / TRAINING_CONFIG_NAME).read_text())
            dtype = manifest["train"]["parameter_dtype"]
        text = log.read_text()
        matches = re.findall(r"Total_Loss ([\d.e+-]+), Action_Loss ([\d.e+-]+), Language_Loss ([\d.e+-]+).*?PeakAllocated ([\d.]+)GiB", text)
        if not matches:
            raise AssertionError(f"Missing finite optimizer-step metrics: {log}")
        total, action, language, peak = map(float, matches[-1])
        assert all(math.isfinite(v) for v in (total, action, language, peak))
        if kind == "action_policy":
            assert language == 0
        else:
            assert language > 0, "The selected sample must exercise real subtask supervision."
        command = [sys.executable, str(Path(__file__).resolve()), "--verify-stage", stage,
                   "--output-dir", str(args.output_dir), "--data-path", str(args.data_path),
                   "--sample-index", str(args.sample_index)]
        run_logged(command, args.output_dir / f"{stage}_deployment.log", env)
        report[stage] = {"entrypoint": f"tasks/train_{kind}.py", "parameter_dtype": dtype,
                         "total_loss": total, "action_loss": action, "language_loss": language,
                         "peak_allocated_gib": peak, "deployment": json.loads(
                             (args.output_dir / f"{stage}_deployment.json").read_text())}
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
        print(f"Passed {stage}: train, export, reload, robot inference", flush=True)


if __name__ == "__main__":
    main()
