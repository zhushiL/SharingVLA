import asyncio
import http
import logging
import os
import time

from .msgpack_numpy import Packer, unpackb
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._active_connection = None
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=_optional_float_env("WEBSOCKET_PING_INTERVAL"),
            ping_timeout=_optional_float_env("WEBSOCKET_PING_TIMEOUT"),
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        if self._active_connection is not None:
            logger.warning("Rejecting a second policy client from %s", websocket.remote_address)
            await websocket.close(code=1013, reason="Only one policy client is supported.")
            return

        self._active_connection = websocket
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()
        try:
            await websocket.send(packer.pack(self._metadata))

            prev_total_time = None
            reset_received = False
            while True:
                start_time = time.monotonic()
                obs = unpackb(await websocket.recv())
                is_reset = isinstance(obs, dict) and obs.get("reset") is True
                if not reset_received and not is_reset:
                    raise ValueError("The first request on a new connection must be a reset request.")

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time
                reset_received = True

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
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


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None


def _optional_float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value.lower() == "none":
        return None
    return float(value)
