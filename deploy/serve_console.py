"""Offline robot console: python -m deploy.serve_console --checkpoint-root outputs."""
from __future__ import annotations

import argparse
import asyncio
import ipaddress
import logging
import os
from pathlib import Path
import socket
import shutil
from urllib.parse import urlsplit

from aiohttp import web
import psutil
import yaml

from .console_runtime import ConsoleRuntime, inspect_checkpoint
from .websocket_policy_server import WebsocketPolicyServer

STATIC = Path(__file__).with_name("static")
log = logging.getLogger(__name__)


def network_addresses(host, web_port, policy_port):
    addresses = [{"interface": "loopback", "ip": "127.0.0.1"}]
    for interface, entries in psutil.net_if_addrs().items():
        for entry in entries:
            if entry.family == socket.AF_INET and not entry.address.startswith("127."):
                addresses.append({"interface": interface, "ip": entry.address})
    for entry in addresses:
        ip = entry["ip"]
        entry.update(page_url=f"http://{ip}:{web_port}", policy_url=f"ws://{ip}:{policy_port}",
                     reachable_bind=host == "0.0.0.0" or host == ip)
    return addresses


class ConsoleService:
    def __init__(self, roots, host="0.0.0.0", web_port=8080, policy_port=8006, runtime=None, restrict_checkpoints=False):
        self.roots = [Path(root).expanduser().resolve() for root in roots]
        self.restrict_checkpoints = restrict_checkpoints
        self.host, self.web_port, self.policy_port = host, web_port, policy_port
        self.runtime = runtime or ConsoleRuntime()
        self.server = None
        self.load_task = None
        self.picker_process = None
        self.picker_busy = False

    def resolve(self, value):
        path = Path(value).expanduser().resolve()
        if self.restrict_checkpoints and not any(path.is_relative_to(root) for root in self.roots):
            raise ValueError("目录不在允许的 checkpoint-root 内，请在启动时添加该根目录")
        if not path.is_dir():
            raise ValueError("目录不存在")
        return path

    def native_picker_command(self, request):
        # The dialog opens on the server desktop, so expose it only to local visitors.
        try:
            local = ipaddress.ip_address(request.remote or "").is_loopback
        except ValueError:
            local = False
        if local and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return shutil.which("zenity")
        return None

    async def choose_folder(self, request):
        command = self.native_picker_command(request)
        if not command:
            raise ValueError("系统选择器不可用，请使用路径选择")
        if self.runtime.phase not in {"empty", "error"}:
            raise ValueError("请先卸载当前模型")
        body = await request.json()
        if self.picker_busy:
            raise ValueError("文件夹选择窗口已打开")
        try:
            initial = self.resolve(body.get("path") or str(self.roots[0]))
        except (ValueError, OSError):
            initial = self.roots[0]
        self.picker_busy = True
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                command, "--file-selection", "--directory", "--title=选择权重文件夹",
                "--filename=" + str(initial) + os.sep,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            self.picker_process = process
            try:
                stdout, _ = await asyncio.wait_for(process.communicate(), timeout=300)
            except asyncio.TimeoutError:
                raise ValueError("选择窗口已超时，请重新打开") from None
            if process.returncode == 1:
                return web.json_response({"path": None})
            if process.returncode != 0:
                raise ValueError("无法打开系统选择器，请使用路径选择")
            selected = stdout.decode("utf-8").removesuffix("\n")
            return web.json_response({"path": str(self.resolve(selected)) if selected else None})
        finally:
            if process and process.returncode is None:
                process.terminate()
                await process.wait()
            self.picker_process = None
            self.picker_busy = False

    async def status(self, request):
        return web.json_response({**self.runtime.status(), "roots": list(dict.fromkeys(str(r) for r in (self.roots if self.restrict_checkpoints else [*self.roots, Path.home(), Path("/")]))),
                                  "addresses": network_addresses(self.host, self.web_port, self.policy_port),
                                  "native_picker_available": bool(self.native_picker_command(request)),
                                  "policy_listening": self.server is not None})

    async def observation(self, request):
        return web.json_response(self.runtime.preview)

    async def browse(self, request):
        path = self.resolve(request.query.get("path", str(self.roots[0])))
        def listing():
            dirs = []
            for p in sorted(path.iterdir(), key=lambda p: p.name.lower()):
                try:
                    if not p.is_dir() or not os.access(p, os.R_OK | os.X_OK):
                        continue
                    if self.restrict_checkpoints and not any(p.resolve().is_relative_to(root) for root in self.roots):
                        continue
                    checkpoint = (p / "vla_cli.yaml").is_file() and (p / "config.json").is_file()
                except OSError:
                    # A protected or disappearing child must not hide its readable siblings.
                    continue
                dirs.append({"name": p.name, "path": str(p), "checkpoint": checkpoint})
                if len(dirs) >= 500:
                    break
            can_go_up = path.parent != path and (not self.restrict_checkpoints or any(path.parent.is_relative_to(root) for root in self.roots))
            parent = str(path.parent) if can_go_up else None
            return {"path": str(path), "parent": parent, "directories": dirs}
        return web.json_response(await asyncio.to_thread(listing))

    async def inspect(self, request):
        body = await request.json()
        path = self.resolve(body["path"])
        return web.json_response(await asyncio.to_thread(inspect_checkpoint, path))

    async def load(self, request):
        rt = self.runtime
        if rt.phase not in {"empty", "error"} or (self.load_task and not self.load_task.done()):
            raise ValueError("请先断开 client 并卸载当前模型")
        body = await request.json()
        path = self.resolve(body["path"])
        # Reserve before awaiting filesystem inspection so double clicks cannot load twice.
        rt.phase, rt.error = "checking", None
        try:
            info = await asyncio.to_thread(inspect_checkpoint, path)
            robot = body.get("robot_name") or info["robot_name"]
            dtype = body.get("dtype", "bfloat16")
            compile_model = body.get("use_compile", False)
            if robot not in info["robots"] or dtype not in {"bfloat16", "float32"} or not isinstance(compile_model, bool):
                raise ValueError("机器人、精度或 compile 配置无效")
        except Exception:
            rt.phase = "empty"
            raise
        self.load_task = asyncio.create_task(self.load_background(info, robot, dtype, compile_model))
        return web.json_response({"status": "loading"}, status=202)

    async def load_background(self, info, robot, dtype, compile_model):
        try:
            await self.runtime.load(info, robot, dtype, compile_model)
            wrapper = WebsocketPolicyServer(self.runtime, self.host, self.policy_port,
                                            self.runtime.metadata, self.runtime.connection_changed)
            self.server = await wrapper.start()
        except Exception as error:
            log.exception("Console model startup failed")
            await self.runtime.model_call(self.runtime.release_model)
            self.runtime.phase, self.runtime.error = "error", str(error)
            self.runtime.metadata = {}
            self.runtime.checkpoint = None

    async def unload(self, request):
        rt = self.runtime
        if rt.client or rt.busy or rt.phase not in {"ready", "error"} or (self.load_task and not self.load_task.done()):
            raise ValueError("请先断开机器人 client，并等待加载或推理结束")
        rt.phase = "unloading"
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        await rt.model_call(rt.release_model)
        rt.phase, rt.error, rt.checkpoint = "empty", None, None
        rt.metadata = {}
        rt.clear_observations()
        return web.json_response({"status": "empty"})

    async def task(self, request):
        body = await request.json()
        version = self.runtime.set_task(body.get("task"))
        return web.json_response({"task_version": version})

    async def cleanup(self, app):
        if self.picker_process and self.picker_process.returncode is None:
            self.picker_process.terminate()
            await self.picker_process.wait()
        if self.load_task:
            await self.load_task
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        await self.runtime.close()


@web.middleware
async def local_api(request, handler):
    # Browser writes must originate from this console, including when served on a LAN IP.
    origin = request.headers.get("Origin")
    if origin and urlsplit(origin).netloc != request.host:
        return web.json_response({"error": "Cross-origin access is disabled"}, status=403)
    try:
        response = await handler(request)
    except (ValueError, KeyError, OSError, TypeError, yaml.YAMLError) as error:
        response = web.json_response({"error": str(error)}, status=400)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
    return response


def create_app(service):
    app = web.Application(middlewares=[local_api], client_max_size=64 * 1024)
    async def index(request):
        return web.FileResponse(STATIC / "index.html")
    app.router.add_get("/", index)
    app.router.add_get("/api/status", service.status)
    app.router.add_get("/api/observation", service.observation)
    app.router.add_get("/api/browse", service.browse)
    app.router.add_post("/api/choose-folder", service.choose_folder)
    app.router.add_post("/api/inspect", service.inspect)
    app.router.add_post("/api/load", service.load)
    app.router.add_post("/api/unload", service.unload)
    app.router.add_post("/api/task", service.task)
    app.router.add_static("/static", STATIC)
    app.on_cleanup.append(service.cleanup)
    return app


def main():
    parser = argparse.ArgumentParser(description="启动离线推理控制台；在页面选择完整 hf_ckpt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--port", type=int, default=8006, help="机器人 WebSocket 端口，加载成功后开放")
    parser.add_argument("--checkpoint-root", action="append", help="目录选择器的快捷目录，可重复；默认当前目录")
    parser.add_argument("--restrict-checkpoints", action="store_true", help="仅允许浏览和加载 checkpoint-root 内的权重；默认可选择任意本机目录")
    parser.add_argument("--preview-fps", type=float, default=5, help="页面预览编码上限，1–15 Hz")
    args = parser.parse_args()
    if not 1 <= args.preview_fps <= 15 or args.port == args.web_port:
        parser.error("preview-fps 必须为 1–15，网页与推理端口必须不同")
    if not all(1 <= port <= 65535 for port in (args.port, args.web_port)):
        parser.error("端口必须为 1–65535")
    # Applied before importing transformers / HF. Never download during console runtime.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    service = ConsoleService(args.checkpoint_root or [Path.cwd()], args.host, args.web_port, args.port,
                             ConsoleRuntime(preview_fps=args.preview_fps), restrict_checkpoints=args.restrict_checkpoints)
    for address in network_addresses(args.host, args.web_port, args.port):
        if address["reachable_bind"]:
            log.info("Console %s | Client %s (%s)", address["page_url"], address["policy_url"], address["interface"])
    web.run_app(create_app(service), host=args.host, port=args.web_port, access_log=None)


if __name__ == "__main__":
    main()
