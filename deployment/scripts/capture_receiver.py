#!/usr/bin/env python3
"""Local-only receiver for browser screenshots captured through Computer Use."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re


OUTPUT = Path(__file__).resolve().parents[1] / "screenshots"
NAME = re.compile(r"^/[a-z0-9][a-z0-9_-]*\.jpg$")
MAX_BYTES = 20 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if not NAME.fullmatch(self.path):
            self.send_error(400, "Invalid screenshot name")
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BYTES:
            self.send_error(413, "Invalid screenshot size")
            return
        payload = self.rfile.read(length)
        if not payload.startswith(b"\xff\xd8\xff"):
            self.send_error(415, "Expected JPEG")
            return
        OUTPUT.mkdir(parents=True, exist_ok=True)
        target = OUTPUT / self.path.removeprefix("/")
        target.write_bytes(payload)
        self.send_response(201)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:4174")
        self.end_headers()
        self.wfile.write(str(target).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:
        print(format % args, flush=True)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 4175), Handler).serve_forever()

