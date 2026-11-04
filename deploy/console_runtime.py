"""Local checkpoint inspection, single-worker inference and bounded previews.

All control state belongs to the asyncio loop. Only model operations run in the
single GPU executor; preview encoding has its own executor and latest-only slot.
"""
from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import json
import logging
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image
import yaml

log = logging.getLogger(__name__)


def asset(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("部署资源必须使用 checkpoint 内的相对路径")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.exists():
        raise ValueError(f"缺少 checkpoint 内资源: {value}")
    return path


def inspect_checkpoint(path):
    root = Path(path).expanduser().resolve()
    config = json.loads((root / "config.json").read_text())
    manifest = yaml.safe_load((root / "vla_cli.yaml").read_text())
    if not isinstance(config, dict) or not isinstance(manifest, dict) or not isinstance(manifest.get("data"), dict):
        raise ValueError("模型配置或 vla_cli.yaml 的 data 配置无效")
    if config.get("model_type") != "vla":
        raise ValueError("请选择完整 hf_ckpt 目录，不能选择 action_policy 子目录")
    backbone = asset(root, "vlm_backbone")
    asset(root, "vlm_backbone/config.json")
    # Transformers can bundle the image processor inside processor_config.json
    # instead of exporting a separate preprocessor_config.json.
    processor_path = backbone / "processor_config.json"
    processor = json.loads(processor_path.read_text()) if processor_path.is_file() else {}
    bundled_image_processor = (
        isinstance(processor, dict)
        and isinstance(processor.get("image_processor"), dict)
        and bool(processor["image_processor"])
    )
    if not any(backbone.glob("*preprocessor*.json")) and not bundled_image_processor:
        raise ValueError("vlm_backbone 缺少图像处理配置")
    if not config.get("standalone_action_policy", False) and not (backbone / "tokenizer_config.json").is_file():
        raise ValueError("VLA 缺少本地 tokenizer_config.json")
    indexes = list(root.glob("*.safetensors.index.json"))
    if indexes:
        for index in indexes:
            for shard in set(json.loads(index.read_text())["weight_map"].values()):
                asset(root, shard)
    elif not list(root.glob("*.safetensors")):
        raise ValueError("目录中缺少完整 safetensors 模型权重")
    data = manifest["data"]
    asset(root, data["norm_stats_file"])
    robot_root = asset(root, data["robot_config_root"])
    robots = sorted(p.stem for p in robot_root.glob("*.yaml"))
    default = data.get("data_name")
    if default and default != "multi":
        robots = [r for r in robots if r == default]
    if not robots:
        raise ValueError("没有可用的机器人配置")
    rtc = config.get("rtc_training") or {}
    rtc_enabled = bool(rtc.get("enabled", False))
    max_delay = rtc.get("max_delay_steps", min(8, config["chunk_size"] - 1) if rtc_enabled else 0)
    return {
        "path": str(root), "policy_name": "action_policy" if config.get("standalone_action_policy") else "vla",
        "robots": robots, "robot_name": default if default in robots else robots[0],
        "chunk_size": config.get("chunk_size"), "rtc_enabled": rtc_enabled,
        "rtc_max_delay_steps": max_delay,
    }


def robot_schema(path, robot):
    root = Path(path).resolve()
    data = yaml.safe_load((root / "vla_cli.yaml").read_text())["data"]
    config = yaml.safe_load(asset(root, data["robot_config_root"]).joinpath(f"{robot}.yaml").read_text())
    cameras, states, labels, xyz = [], set(), {}, []
    for entry in config.get("images", []):
        if isinstance(entry, str):
            cameras.append(entry)
        else:
            for key, value in entry.items():
                origin = value.get("origin_keys", key)
                cameras.extend([origin] if isinstance(origin, str) else origin)
    for entry in config.get("states", []):
        if isinstance(entry, str):
            states.add(entry)
            continue
        for target, value in entry.items():
            origin = value.get("origin_keys", target)
            if isinstance(origin, str):
                states.add(origin)
                continue
            for part, mapping in enumerate(origin):
                for key, span in mapping.items():
                    states.add(key)
                    # Native TCP poses are xyz + 6D rotation; expose only the
                    # position indices for the compact UI, never alter model input.
                    width = span["end"] - span["start"]
                    if target == "observation.state.end.position" and width >= 3:
                        stride = 9 if width % 9 == 0 else 3 if width == 3 else None
                        if stride:
                            for start in range(span["start"], span["end"], stride):
                                xyz.append({"key": key, "indices": [start, start + 1, start + 2],
                                            "label": f"TCP {len(xyz) + 1}"})
                    for index in range(span["start"], span["end"]):
                        labels.setdefault(key, {})[str(index)] = f"{target.removeprefix('observation.state.')} · {part + 1} · {index - span['start']}"
    return {"cameras": list(dict.fromkeys(cameras)), "states": sorted(states), "state_labels": labels,
            "state_xyz": xyz}


def load_policy(path, robot, dtype, compile_model):
    # Lazy import keeps initial page startup independent of CUDA/model loading.
    from xense0vla.inference.policy import RobotPolicy, set_seed_everywhere
    set_seed_everywhere(42)
    return RobotPolicy(path, robot_name=robot, chunk_ret=True, use_length=-1,
                       use_bf16=dtype == "bfloat16", use_fp32=dtype == "float32", use_compile=compile_model)


def encode_preview(snapshot):
    obs, info, schema = snapshot
    images, errors = {}, []
    for key in schema["cameras"]:
        if key not in obs:
            continue
        try:
            arr = np.asarray(obs[key])
            frames = [("", arr)] if arr.ndim == 3 else [("", arr[-1]), (" / history", arr[0])]
            for suffix, frame in frames:
                if frame.ndim != 3:
                    raise ValueError("expected RGB image")
                if frame.shape[-1] != 3 and frame.shape[0] == 3:
                    frame = frame.transpose(1, 2, 0)
                if frame.dtype != np.uint8 or frame.shape[-1] != 3:
                    raise ValueError("expected uint8 RGB")
                im = Image.fromarray(frame)
                im.thumbnail((768, 480))
                stream = io.BytesIO()
                im.save(stream, format="JPEG", quality=75)
                images[key + suffix] = "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()
        except (ValueError, TypeError, IndexError) as error:
            errors.append(f"{key}: {error}")
    states = {}
    for key in schema["states"]:
        if key in obs:
            a = np.asarray(obs[key], dtype=float).reshape(-1)
            states[key] = [float(x) if np.isfinite(x) else None for x in a]
    return {**info, "images": images, "states": states, "errors": errors}


class ConsoleRuntime:
    def __init__(self, factory=load_policy, preview_fps=5):
        self.factory = factory
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy")
        self.preview_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preview")
        self.preview_fps = preview_fps
        self.policy = None
        self.phase = "empty"
        self.error = None
        self.metadata = {}
        self.checkpoint = None
        self.schema = {"cameras": [], "states": [], "state_labels": {}}
        self.task = ""
        self.task_version = 0
        self.active_task_version = None
        self.result_task_version = None
        self.infer_ms = None
        self.client = None
        self.session_id = None
        self.request_seq = 0
        self.busy = False
        self.pending_preview = None
        self.preview = None
        self.preview_task = None
        self.generation = 0
        self.last_prefix_length = 0

    async def model_call(self, fn, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(self.worker, partial(fn, *args, **kwargs))

    def set_task(self, task):
        if not isinstance(task, str) or not task.strip() or len(task) > 4096:
            raise ValueError("请输入 1–4096 字符的任务")
        if self.metadata.get("policy_name") == "action_policy":
            raise ValueError("Action Policy 不使用任务文本")
        self.task = task.strip()
        self.task_version += 1
        return self.task_version

    def connection_changed(self, address):
        self.client = str(address) if address else None
        if address:
            self.session_id = uuid.uuid4().hex
            self.clear_observations()

    def clear_observations(self):
        self.generation += 1
        self.pending_preview = self.preview = None
        self.request_seq = 0
        self.active_task_version = self.result_task_version = None
        self.infer_ms = None
        self.last_prefix_length = 0

    def status(self):
        return {"phase": self.phase, "error": self.error, "metadata": self.metadata,
                "checkpoint": self.checkpoint, "schema": self.schema, "task": self.task,
                "task_version": self.task_version, "active_task_version": self.active_task_version,
                "result_task_version": self.result_task_version, "infer_ms": self.infer_ms,
                "client": self.client, "session_id": self.session_id, "request_seq": self.request_seq,
                "busy": self.busy, "rtc_prefix_length": self.last_prefix_length}

    async def load(self, info, robot, dtype="bfloat16", compile_model=False):
        self.phase, self.error = "loading", None
        try:
            self.schema = robot_schema(info["path"], robot)
            self.policy = await self.model_call(self.factory, info["path"], robot, dtype, compile_model)
            self.metadata = {**self.policy.get_metadata(), "task_source": "server", "robot_name": robot,
                             "console_protocol_version": 1, "observation_schema": self.schema}
            if not self.metadata["chunk_ret"] or self.metadata["use_length"] != self.metadata["checkpoint_chunk_size"]:
                raise ValueError("控制台要求完整 chunk 返回")
            self.checkpoint = {**info, "robot_name": robot, "dtype": dtype, "use_compile": compile_model}
            self.clear_observations()
            self.phase = "ready"
        except Exception as error:
            log.exception("Model load failed")
            await self.model_call(self.release_model)
            self.phase, self.error = "error", str(error)
            self.metadata = {}
            raise

    def release_model(self):
        self.policy = None
        import gc
        import sys
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def infer(self, raw):
        if self.phase != "ready" or self.policy is None:
            raise ValueError("模型尚未就绪")
        if not isinstance(raw, dict) or "batch" in raw:
            raise ValueError("控制台仅支持单机器人 observation 字典")
        if self.busy:
            raise ValueError("仅支持一个在途推理请求")
        self.busy = True
        try:
            if raw.get("reset") is True:
                if raw.get("robo_name") != self.metadata["robot_name"]:
                    raise ValueError("reset 的机器人必须与页面选择一致")
                result = await self.model_call(self.policy.infer, raw)
                self.clear_observations()
                return result
            self.request_seq += 1
            full_vla = self.metadata["policy_name"] == "vla"
            version, task = (self.task_version, self.task) if full_vla else (None, None)
            # Copy the dictionary, leaving the wire payload and committed RTC prefix untouched.
            obs = dict(raw)
            obs.pop("task", None)
            if full_vla:
                obs["task"] = task
            info = {"sequence": self.request_seq, "session_id": self.session_id,
                    "observation_epoch": self.generation, "received_at": time.time(), "task_version": version}
            # Scalar metadata is echoed without redefining the client's control-step timebase.
            for key in ("request_id", "observation_tick", "observation_timestamp"):
                value = obs.pop(key, None)
                if value is not None:
                    if not isinstance(value, (int, float, str)) or isinstance(value, bool):
                        raise ValueError(f"{key} 必须是标量")
                    if isinstance(value, float) and not np.isfinite(value):
                        raise ValueError(f"{key} 必须是有限数值")
                    info[key] = value
            self.pending_preview = (dict(obs), info, self.schema)
            if self.preview_task is None:
                self.preview_task = asyncio.create_task(self.preview_loop())
            if full_vla and not task:
                return {"action": None, "status": "task_required", "console": info}
            self.active_task_version = version
            self.last_prefix_length = int(obs.get("rtc_prefix_length", 0))
            started = time.monotonic()
            result = await self.model_call(self.policy.infer, obs)
            self.infer_ms = (time.monotonic() - started) * 1000
            self.result_task_version = version
            self.error = None
            return {**result, "console": {**info, "task": task}, "status": "ok"}
        except Exception as error:
            self.error = str(error)
            raise
        finally:
            self.busy = False
            self.active_task_version = None

    async def preview_loop(self):
        while True:
            if self.pending_preview is not None:
                snapshot, generation = self.pending_preview, self.generation
                self.pending_preview = None
                try:
                    result = await asyncio.get_running_loop().run_in_executor(self.preview_worker, encode_preview, snapshot)
                    if generation == self.generation:
                        self.preview = result
                except Exception:
                    log.exception("Preview encoding failed (inference is unaffected)")
            await asyncio.sleep(1 / self.preview_fps)

    async def close(self):
        if self.preview_task:
            self.preview_task.cancel()
            await asyncio.gather(self.preview_task, return_exceptions=True)
        await self.model_call(self.release_model)
        self.worker.shutdown(wait=True)
        self.preview_worker.shutdown(wait=True)
