"""Local server for the Level 1 viewer. Standard library only; binds to 127.0.0.1.

Routes:

``GET /``                the page;  ``GET /app.js`` its script
``GET /scene.glb``       the built scene;  ``GET /manifest.json`` its manifest
``GET /state``           the scene state the page draws (``viewer.state.scene_state``)
``POST /command``        ``{"text": "..."}`` -> the engine's result plus the new state
``POST /reset``          rebuild the scene from scratch

    python -m viewer.server --scene datasets/processed/level1/s0011 --port 8765
"""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from awr.config import load_domain_config
from reasoning.command_engine import CommandEngine
from viewer.state import scene_state

__all__ = ["ViewerApp", "make_server", "main"]

STATIC = Path(__file__).parent / "static"
MAX_COMMAND_BYTES = 4096


class ViewerApp:
    """One engine, one built scene, and a lock so commands apply one at a time."""

    def __init__(self, scene_dir: Path, *, domain: str = "thorax") -> None:
        self.scene_dir = scene_dir
        self.domain = domain
        self.manifest: dict[str, Any] = json.loads(
            (scene_dir / "manifest.json").read_text(encoding="utf-8")
        )
        self.lock = threading.Lock()
        self.engine = self._fresh_engine()

    def _fresh_engine(self) -> CommandEngine:
        engine = CommandEngine(config=load_domain_config(domain=self.domain))
        engine.execute("generate a human heart")
        return engine

    def state(self) -> dict[str, Any]:
        """The current state, for the page."""
        with self.lock:
            return scene_state(self.engine, self.manifest)

    def command(self, text: str) -> dict[str, Any]:
        """Run one command through the engine; never raises for a bad command."""
        with self.lock:
            try:
                result = self.engine.execute(text)
                ok, message = bool(result.ok), str(result.message)
            except Exception as exc:  # noqa: BLE001 - reported to the page, not swallowed
                ok, message = False, f"{type(exc).__name__}: {exc}"
            return {"ok": ok, "message": message, "state": scene_state(self.engine, self.manifest)}

    def reset(self) -> dict[str, Any]:
        """Start the scene again."""
        with self.lock:
            self.engine = self._fresh_engine()
            return {"ok": True, "message": "Scene reset.", "state": scene_state(
                self.engine, self.manifest
            )}


def _handler(app: ViewerApp) -> type[BaseHTTPRequestHandler]:
    files = {
        "/": (STATIC / "index.html", "text/html; charset=utf-8"),
        "/app.js": (STATIC / "app.js", "text/javascript; charset=utf-8"),
        "/scene.glb": (app.scene_dir / "scene.glb", "model/gltf-binary"),
        "/manifest.json": (app.scene_dir / "manifest.json", "application/json"),
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "LagnavViewer/0.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            """Quiet: the page polls."""

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def do_GET(self) -> None:  # noqa: N802 - http.server's naming
            """Static files and state."""
            path = self.path.split("?", 1)[0]
            if path == "/state":
                self._json(app.state())
                return
            if path in files:
                file_path, content_type = files[path]
                if file_path.is_file():
                    self._send(HTTPStatus.OK, file_path.read_bytes(), content_type)
                    return
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - http.server's naming
            """Commands and reset."""
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length > MAX_COMMAND_BYTES:
                self._json({"error": "request too large"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            raw = self.rfile.read(length) if length else b""
            if self.path == "/reset":
                self._json(app.reset())
                return
            if self.path != "/command":
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                text = str(json.loads(raw.decode("utf-8") or "{}").get("text", "")).strip()
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                self._json({"error": "expected JSON {\"text\": ...}"}, HTTPStatus.BAD_REQUEST)
                return
            if not text:
                self._json({"error": "empty command"}, HTTPStatus.BAD_REQUEST)
                return
            self._json(app.command(text))

    return Handler


def make_server(app: ViewerApp, port: int = 8765) -> ThreadingHTTPServer:
    """A server on 127.0.0.1 only: the viewer is a local tool, not a service."""
    return ThreadingHTTPServer(("127.0.0.1", port), _handler(app))


def main(argv: Sequence[str] | None = None) -> int:
    """Serve one built scene."""
    parser = argparse.ArgumentParser(description="Lagnav Level 1 viewer.")
    parser.add_argument("--scene", type=Path, required=True, help="folder with scene.glb")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = make_server(ViewerApp(args.scene), args.port)
    print(f"Lagnav viewer on http://127.0.0.1:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
