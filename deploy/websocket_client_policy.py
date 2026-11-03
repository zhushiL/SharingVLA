import logging
import os
import time
from typing import Dict, Optional, Tuple

import websockets.sync.client
from .msgpack_numpy import Packer, unpackb


def _close_quietly(connection) -> None:
    try:
        connection.close()
    except Exception as error:
        logging.warning("Failed to close an incomplete WebSocket connection: %s", error)


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = 8006, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            conn = None
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                ping_interval = _optional_float_env("WEBSOCKET_PING_INTERVAL")
                ping_timeout = _optional_float_env("WEBSOCKET_PING_TIMEOUT")
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=ping_interval,
                    ping_timeout=ping_timeout,
                )
                metadata = unpackb(conn.recv())
                if not isinstance(metadata, dict):
                    raise RuntimeError(f"Invalid policy server metadata: {metadata!r}")
                return conn, metadata
            except (ConnectionRefusedError, OSError):
                if conn is not None:
                    _close_quietly(conn)
                logging.info("Still waiting for server at %s...", self._uri)
                time.sleep(5)
            except BaseException:
                if conn is not None:
                    _close_quietly(conn)
                raise

    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    def reset(self, robo_name: str = "own_robot") -> None:
        response = self.infer(dict(reset=True, robo_name=robo_name))
        if not isinstance(response, dict) or response.get("action", object()) is not None:
            raise RuntimeError(f"Invalid policy reset response: {response!r}")

    def close(self) -> None:
        self._ws.close()

def _optional_float_env(name: str) -> Optional[float]:
    value = os.environ.get(name)
    if value is None or value.lower() == "none":
        return None
    return float(value)
