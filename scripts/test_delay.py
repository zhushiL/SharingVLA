"""Measure WebSocket latency and verify RTC prefixes for each configured delay.

This sends recorded observations to an inference server; it does not command a
robot, simulate an asynchronous controller, or evaluate task success.
"""

from __future__ import annotations

import argparse
import json
import math
import http.client
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.websocket_client_policy import WebsocketClientPolicy  # noqa: E402


def load_observation(args, metadata):
    if args.observation:
        with np.load(args.observation, allow_pickle=False) as data:
            observation = {key: data[key] for key in data.files}
        if "task" in observation:
            observation["task"] = str(observation["task"].item())
        return observation

    import yaml
    from xense0vla.data.robot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    root = args.data_path.expanduser().resolve()
    mapping = yaml.safe_load((ROOT / "configs/robot_configs/own_robot.yaml").read_text())
    cameras = [next(iter(item.values()))["origin_keys"] for item in mapping["images"]]
    full = metadata["policy_name"] == "vla"
    dataset_meta = LeRobotDatasetMetadata(root.name, root=root)
    delta = {cameras[0]: [-metadata["head_history_offset"] / dataset_meta.fps, 0.0]} if full else None
    selected = cameras if full else cameras[1:]
    dataset = LeRobotDataset(
        root.name,
        root=root,
        camera_keys=selected,
        delta_timestamps=delta,
        load_task=full,
        load_subtask=False,
        video_backend=args.video_backend,
    )
    raw = dataset[args.sample_index]
    keys = [*selected, "observation.state", *(["task"] if full else [])]
    return {key: raw[key].cpu().numpy() if hasattr(raw[key], "cpu") else raw[key] for key in keys}


def select_delays(metadata, requested=None):
    if not metadata.get("chunk_ret"):
        raise ValueError("Delay tests require a server returning full action chunks (the server default).")
    maximum = int(metadata["rtc_training_max_delay_steps"]) if metadata.get("prefix_conditioned") else 0
    delays = list(range(maximum + 1)) if requested is None else sorted(set(requested))
    if not delays or any(delay < 0 or delay > maximum for delay in delays):
        raise ValueError(f"Delays must lie in [0, {maximum}]; positive delays require an RTC checkpoint.")
    if max(delays) >= int(metadata["use_length"]):
        raise ValueError("Returned chunk length must exceed every tested delay, leaving an action postfix.")
    return delays


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
    }


def validate_response(response, metadata, prefix=None):
    action = np.asarray(response["action"])
    shape = (int(metadata["use_length"]), int(metadata["action_output_dim"]))
    if action.shape != shape or not np.isfinite(action).all():
        raise AssertionError(f"Expected finite actions with shape {shape}, got {action.shape}.")
    if prefix is not None and not np.array_equal(action[: len(prefix)], prefix):
        raise AssertionError("The server changed the committed RTC action prefix.")
    server_ms = float(response["server_timing"]["infer_ms"])
    if not math.isfinite(server_ms) or server_ms < 0:
        raise AssertionError("The server returned invalid inference timing.")
    return action, server_ms


def benchmark(client, observation, *, robot_name="own_robot", control_hz, repeats=20, warmup=2, delays=None):
    if not math.isfinite(control_hz) or control_hz <= 0 or repeats < 1 or warmup < 0:
        raise ValueError("control_hz and repeats must be positive; warmup must be non-negative.")
    metadata = client.get_server_metadata()
    delays = select_delays(metadata, delays)
    if "rtc_action_prefix" in observation or "rtc_prefix_length" in observation:
        raise ValueError("The recorded observation must not include RTC fields; the test supplies them.")
    client.reset(robot_name)
    # Bootstrap a plan in native coordinates; exclude model startup from timings.
    reference, _ = validate_response(client.infer(observation), metadata)
    period_ms = 1000.0 / control_hz
    results = []
    for delay in delays:
        prefix = reference[:delay].copy() if delay else None
        request = dict(observation)
        if prefix is not None:
            request.update(rtc_action_prefix=prefix, rtc_prefix_length=delay)
        roundtrips, server_times = [], []
        for iteration in range(warmup + repeats):
            start = time.perf_counter()
            response = client.infer(request)
            elapsed = (time.perf_counter() - start) * 1000.0
            _, server_ms = validate_response(response, metadata, prefix)
            if iteration >= warmup:
                roundtrips.append(elapsed)
                server_times.append(server_ms)
        timing = summarize(roundtrips)
        required_steps = math.ceil(timing["p95_ms"] / period_ms)
        results.append(
            {
                "delay_steps": delay,
                "prefix_preserved": bool(delay),
                "requests": repeats,
                "roundtrip": timing,
                "server_inference": summarize(server_times),
                "p95_required_delay_steps": required_steps,
                "p95_within_trained_rtc_range": bool(metadata.get("prefix_conditioned"))
                and required_steps <= int(metadata["rtc_training_max_delay_steps"]),
            }
        )
    return {
        "status": "passed",
        "policy": metadata,
        "robot_name": robot_name,
        "control_hz": control_hz,
        "control_period_ms": period_ms,
        "warmup_per_delay": warmup,
        "measurement": "Roundtrip includes serialization, network, preprocessing, GPU inference and response decoding.",
        "scope": "Fixed recorded observation; prefix integrity and latency only, not robot/control performance.",
        "delays": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8006)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-path", type=Path, help="Local LeRobot dataset with the own_robot mapping.")
    source.add_argument("--observation", type=Path, help="NPZ containing native observation keys; no pickled objects.")
    parser.add_argument("--sample-index", type=int, default=165)
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--robot-name", default="own_robot", help="Robot mapping embedded in the served checkpoint.")
    parser.add_argument(
        "--control-hz", type=float, required=True, help="Actual action execution frequency, not request rate."
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--delays", type=int, nargs="+", help="Default: every delay 0..RTC max; only 0 for non-RTC models."
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/delay/report.json")
    args = parser.parse_args()
    # Fail promptly instead of entering the client's connection retry loop when no server is running.
    connection = http.client.HTTPConnection(args.host, args.port, timeout=5)
    try:
        connection.request("GET", "/healthz")
        if connection.getresponse().status != 200:
            raise ConnectionError("The policy server health check failed.")
    finally:
        connection.close()
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    try:
        observation = load_observation(args, client.get_server_metadata())
        report = benchmark(
            client,
            observation,
            robot_name=args.robot_name,
            control_hz=args.control_hz,
            repeats=args.repeats,
            warmup=args.warmup,
            delays=args.delays,
        )
        report["source"] = str(args.observation or args.data_path)
        report["sample_index"] = args.sample_index if args.data_path else None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        for result in report["delays"]:
            print(
                f"delay={result['delay_steps']}: p50={result['roundtrip']['p50_ms']:.1f} ms, "
                f"p95={result['roundtrip']['p95_ms']:.1f} ms, "
                f"p95 needs {result['p95_required_delay_steps']} control steps",
                flush=True,
            )
        print(f"Report: {args.output}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
