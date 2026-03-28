"""Windows relay from localhost:8082 to WSL-local gpu_server.

Нужен как workaround для mirrored WSL networking, когда Docker/Windows не может
напрямую достучаться до WSL-native `gpu_server.py`, но сам WSL localhost:8082 жив.
"""

from __future__ import annotations

import json
import os
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOST = "0.0.0.0"
PORT = int(os.environ.get("WSL_TEI_RELAY_PORT", "18082"))
WSL_DISTRO = os.environ.get("WSL_TEI_RELAY_DISTRO", "Ubuntu-22.04")
UPSTREAM = os.environ.get("WSL_TEI_RELAY_UPSTREAM", "http://127.0.0.1:8082").rstrip("/")


def _run_wsl_curl(method: str, path: str, body: bytes | None = None) -> tuple[int, bytes]:
    """Проксировать HTTP-запрос в WSL через curl."""

    target = f"{UPSTREAM}{path}"
    curl_cmd = [
        "curl",
        "-sS",
        "-X",
        method,
        "-H",
        "Content-Type: application/json",
        target,
    ]
    if body is not None:
        curl_cmd.extend(["--data-binary", "@-"])

    result = subprocess.run(
        ["wsl", "-d", WSL_DISTRO, "-e", "bash", "-lc", " ".join(subprocess.list2cmdline([part]) for part in curl_cmd)],
        input=body,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        error = {
            "error": "wsl relay upstream failed",
            "returncode": result.returncode,
            "stderr": result.stderr.decode("utf-8", "replace"),
        }
        return 502, json.dumps(error, ensure_ascii=False).encode("utf-8")
    return 200, result.stdout


class Handler(BaseHTTPRequestHandler):
    """Минимальный relay-handler для TEI/gpu_server endpoints."""

    server_version = "wsl-tei-relay/1.0"

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send(404, b'{"error":"not found"}')
            return
        status, body = _run_wsl_curl("GET", self.path)
        self._send(status, body)

    def do_POST(self) -> None:
        if self.path not in {"/v1/embeddings", "/embed"}:
            self._send(404, b'{"error":"not found"}')
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        status, response = _run_wsl_curl("POST", self.path, body)
        self._send(status, response)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.serve_forever()
