import asyncio
from concurrent.futures import ThreadPoolExecutor
import hmac
import http
import inspect
import logging
import os
import time

import numpy as np
from .msgpack_numpy import Packer, unpackb
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)

DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024  # 64 MiB
DEFAULT_PING_INTERVAL = 20.0
DEFAULT_PING_TIMEOUT = 20.0


def _is_reset_request(obs) -> bool:
    if not isinstance(obs, dict):
        return False
    flag = obs.get("reset")
    return isinstance(flag, (bool, np.bool_)) and bool(flag)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Accepts reset and inference requests. Policies may implement sync or async infer;
    synchronous infer calls automatically run on a dedicated single-thread executor so
    the asyncio event loop stays responsive for heartbeats and /healthz checks.
    """

    def __init__(
        self,
        policy,
        host: str = "127.0.0.1",
        port: int | None = None,
        metadata: dict | None = None,
        on_connection_change=None,
        api_key: str | None = None,
        max_size: int | None = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._active_connection = None
        self._on_connection_change = on_connection_change
        self._api_key = api_key if api_key else None
        self._max_size = max_size
        self._sync_executor: ThreadPoolExecutor | None = None
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def _get_sync_executor(self) -> ThreadPoolExecutor:
        if self._sync_executor is None:
            self._sync_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ws-policy")
        return self._sync_executor

    def close_executor(self) -> None:
        if self._sync_executor is not None:
            self._sync_executor.shutdown(wait=False)
            self._sync_executor = None

    def serve_forever(self) -> None:
        try:
            asyncio.run(self.run())
        finally:
            self.close_executor()

    async def run(self):
        server = await self.start()
        try:
            async with server:
                await server.serve_forever()
        finally:
            self.close_executor()

    def _process_request(
        self,
        connection: _server.ServerConnection,
        request: _server.Request,
    ) -> _server.Response | None:
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        if self._api_key is not None:
            auth = request.headers.get("Authorization", "")
            expected = f"Api-Key {self._api_key}"
            if not hmac.compare_digest(auth, expected):
                return connection.respond(http.HTTPStatus.UNAUTHORIZED, "Unauthorized\n")
        return None

    async def start(self):
        """Bind only when the policy is ready; caller owns server.close()."""
        return await _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=self._max_size,
            ping_interval=_optional_float_env("WEBSOCKET_PING_INTERVAL", DEFAULT_PING_INTERVAL),
            ping_timeout=_optional_float_env("WEBSOCKET_PING_TIMEOUT", DEFAULT_PING_TIMEOUT),
            process_request=self._process_request,
        )

    async def _call_policy(self, obs):
        infer_fn = self._policy.infer
        if inspect.iscoroutinefunction(infer_fn):
            return await infer_fn(obs)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self._get_sync_executor(), infer_fn, obs)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def _handler(self, websocket: _server.ServerConnection):
        if self._active_connection is not None:
            logger.warning("Rejecting a second policy client from %s", websocket.remote_address)
            await websocket.close(code=1013, reason="Only one policy client is supported.")
            return

        self._active_connection = websocket
        if self._on_connection_change:
            self._on_connection_change(websocket.remote_address)
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()
        try:
            await websocket.send(packer.pack(self._metadata))

            prev_total_time = None
            reset_received = False
            while True:
                start_time = time.monotonic()
                obs = unpackb(await websocket.recv())
                is_reset = _is_reset_request(obs)
                if not reset_received and not is_reset:
                    raise ValueError("The first request on a new connection must be a reset request.")

                infer_time = time.monotonic()
                action = await self._call_policy(obs)
                infer_time = time.monotonic() - infer_time
                reset_received = True

                timing = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    timing["prev_total_ms"] = prev_total_time * 1000
                payload = {**action, "server_timing": timing}

                await websocket.send(packer.pack(payload))
                prev_total_time = time.monotonic() - start_time
        except websockets.ConnectionClosed:
            logger.info(f"Connection from {websocket.remote_address} closed")
        except Exception:
            logger.exception("Inference request from %s failed", websocket.remote_address)
            try:
                await websocket.send(
                    "Inference request failed. See the policy server log for the traceback."
                )
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal policy server error.",
                )
            except websockets.ConnectionClosed:
                pass
        finally:
            if self._active_connection is websocket:
                self._active_connection = None
                if self._on_connection_change:
                    self._on_connection_change(None)


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None


def _optional_float_env(name: str, default: float | None = None) -> float | None:
    value = os.environ.get(name)
    if value is None:
        return default
    stripped = value.strip()
    if not stripped:
        return default
    if stripped.lower() == "none":
        return None
    return float(stripped)
