"""
Shared fixtures for webview server integration tests.

These tests communicate with a running darktable-server instance
over a Unix domain socket using the JSON-RPC protocol.
"""

import json
import os
import socket
import struct
import subprocess
import tempfile
import time

import pytest


def _find_server_binary():
    """Locate darktable-server binary in common build dirs."""
    root = os.path.dirname(os.path.abspath(__file__))
    # Walk up to repo root (src/tests/integration/webview -> root)
    for _ in range(4):
        root = os.path.dirname(root)

    candidates = [
        os.path.join(root, "build", "bin", "darktable-server"),
        os.path.join(root, "build-release", "bin", "darktable-server"),
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


class ServerClient:
    """JSON-RPC client for darktable-server over Unix socket."""

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(socket_path)
        self.sock.settimeout(10.0)
        self._id = 0
        self._buf = b""

    def call(self, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC request and return the parsed response."""
        self._id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._id,
            "method": method,
        }
        if params is not None:
            request["params"] = params

        data = json.dumps(request).encode("utf-8")
        # Length-prefixed framing: 4-byte big-endian length + JSON
        self.sock.sendall(struct.pack(">I", len(data)) + data)
        return self._recv_response()

    def _recv_response(self) -> dict:
        """Read a length-prefixed JSON-RPC response."""
        # Read 4-byte length header
        while len(self._buf) < 4:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Server closed connection")
            self._buf += chunk

        msg_len = struct.unpack(">I", self._buf[:4])[0]
        self._buf = self._buf[4:]

        # Read message body
        while len(self._buf) < msg_len:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Server closed connection")
            self._buf += chunk

        msg = self._buf[:msg_len]
        self._buf = self._buf[msg_len:]
        return json.loads(msg)

    def close(self):
        self.sock.close()


@pytest.fixture
def server_binary():
    """Path to the darktable-server binary. Skip if not built."""
    path = _find_server_binary()
    if path is None:
        pytest.skip("darktable-server binary not found (run cmake --build)")
    return path


@pytest.fixture
def server_socket_path():
    """Temporary Unix socket path for server.

    Uses /tmp directly because macOS limits Unix socket paths to 104 bytes
    and pytest's tmp_path is too long.
    """
    path = tempfile.mktemp(prefix="dt-", suffix=".sock", dir="/tmp")
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def server(server_binary, server_socket_path):
    """Start a darktable-server and yield a connected client.

    Automatically stops the server after the test.
    """
    env = os.environ.copy()
    env["GSETTINGS_SCHEMA_DIR"] = "/opt/homebrew/share/glib-2.0/schemas"

    config_dir = tempfile.mkdtemp(prefix="dt-test-config-")
    proc = subprocess.Popen(
        [server_binary, "--socket", server_socket_path,
         "--core", "--configdir", config_dir, "--disable-opencl"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for socket to appear (darktable init can take a while on first run)
    for _ in range(150):
        if os.path.exists(server_socket_path):
            break
        # Check if process died
        if proc.poll() is not None:
            stdout = proc.stdout.read().decode() if proc.stdout else ""
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            pytest.fail(f"Server exited early (code {proc.returncode}):\n{stderr}\n{stdout}")
        time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("Server did not create socket within 15s")

    client = ServerClient(server_socket_path)
    yield client

    client.close()
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
