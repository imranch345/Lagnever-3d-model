"""Local server for the Level 1 viewer. Standard library only; binds to 127.0.0.1.

Routes:

``GET /``                the page;  ``GET /app.js`` its script
``GET /scene.glb``       the built scene;  ``GET /manifest.json`` its manifest
``GET /state``           the scene state the page draws (``viewer.state.scene_state``)
``POST /command``        ``{"text": "..."}`` -> the engine's result plus the new state
``POST /reset``          rebuild the scene from scratch
``POST /sample``         ``{"seed": n}`` -> draw a thorax from the shape model (``--model``)
``POST /original``       back to the real subject's geometry

    python -m viewer.server --scene datasets/processed/level1/s0011 --port 8765
"""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Callable, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from awr.config import load_domain_config
from language.layer import LanguageLayer, Translator, translator_from_environment
from reasoning.command_engine import CommandEngine
from viewer.state import scene_state

__all__ = ["ViewerApp", "make_server", "main"]

STATIC = Path(__file__).parent / "static"
MAX_COMMAND_BYTES = 4096


class ViewerApp:
    """One engine, one built scene, and a lock so commands apply one at a time."""

    def __init__(
        self,
        scene_dir: Path,
        *,
        domain: str = "thorax",
        translator: Translator | None = None,
        sampler: Callable[[int, Path], dict[str, Any]] | None = None,
        samples_dir: Path | None = None,
    ) -> None:
        self.scene_dir = scene_dir
        self.original_dir = scene_dir
        # Draws a thorax from a shape model into a folder and returns its manifest. Injected,
        # so the server needs the shape tooling only when a model is actually loaded.
        self.sampler = sampler
        self.samples_dir = samples_dir or scene_dir.parent / "samples"
        self._next_seed = 1
        self.domain = domain
        self.manifest: dict[str, Any] = json.loads(
            (scene_dir / "manifest.json").read_text(encoding="utf-8")
        )
        self.lock = threading.Lock()
        self.engine = self._fresh_engine()
        # Off unless enabled (LAGNAV_LANGUAGE=claude): the rule parser handles everything it
        # understands for free, and only free-form text the parser cannot read reaches Claude.
        self.translator = (
            translator
            if translator is not None
            else translator_from_environment(self.engine.ontology)
        )

    def _fresh_engine(self) -> CommandEngine:
        engine = CommandEngine(config=load_domain_config(domain=self.domain))
        engine.execute("generate the heart and lungs")
        return engine

    def _state(self) -> dict[str, Any]:
        state = scene_state(self.engine, self.manifest)
        state["language"] = "claude" if self.translator is not None else "rules"
        state["sampling"] = self.sampler is not None
        state["generated"] = self.manifest.get("generated")
        return state

    def sample(self, seed: int | None = None) -> dict[str, Any]:
        """Swap in a generated thorax, keeping the scene state.

        Visibility and opacity stay as they were: the geometry is a view of the AWR, and
        changing the view changes nothing in it.
        """
        with self.lock:
            if self.sampler is None:
                return {
                    "ok": False,
                    "message": "No shape model is loaded; start the viewer with --model.",
                    "state": self._state(),
                }
            if seed is None:
                seed, self._next_seed = self._next_seed, self._next_seed + 1
            manifest = self.sampler(seed, self.samples_dir / f"seed-{seed}")
            self.scene_dir, self.manifest = self.samples_dir / f"seed-{seed}", manifest
            checks = manifest.get("checks", {})
            verdict = (
                "passes the quality checks"
                if checks.get("usable")
                else f"fails: {'; '.join(checks.get('hard', []))}"
            )
            return {
                "ok": True,
                "message": f"Generated thorax, seed {seed}: {verdict}.",
                "state": self._state(),
            }

    def original(self) -> dict[str, Any]:
        """Back to the real subject's geometry."""
        with self.lock:
            self.scene_dir = self.original_dir
            self.manifest = json.loads(
                (self.scene_dir / "manifest.json").read_text(encoding="utf-8")
            )
            return {
                "ok": True,
                "message": f"Real subject {self.manifest.get('subject')}.",
                "state": self._state(),
            }

    def state(self) -> dict[str, Any]:
        """The current state, for the page."""
        with self.lock:
            return self._state()

    def command(self, text: str) -> dict[str, Any]:
        """Run one request through the language layer and the engine; never raises.

        The reply says which path ran ("rules" or "claude") and, for the language model, the
        program it produced and the commands applied, so every change is visible.
        """
        with self.lock:
            program: dict[str, Any] | None = None
            commands: list[str] = []
            source = "rules"
            try:
                result = LanguageLayer(self.engine, self.translator).execute(text)
                ok, message, source = result.ok, result.message, result.source
                program, commands = result.program, result.commands
            except Exception as exc:  # noqa: BLE001 - reported to the page, not swallowed
                ok, message = False, f"{type(exc).__name__}: {exc}"
            return {
                "ok": ok,
                "message": message,
                "source": source,
                "program": program,
                "commands": commands,
                "state": self._state(),
            }

    def language_enabled(self) -> bool:
        """Whether free-form text can reach the language model."""
        return self.translator is not None

    def reset(self) -> dict[str, Any]:
        """Start the scene again."""
        with self.lock:
            self.engine = self._fresh_engine()
            return {"ok": True, "message": "Scene reset.", "state": self._state()}


def _handler(app: ViewerApp) -> type[BaseHTTPRequestHandler]:
    # Resolved per request: the scene folder changes when a sample is drawn.
    files: dict[str, tuple[Callable[[], Path], str]] = {
        "/": (lambda: STATIC / "index.html", "text/html; charset=utf-8"),
        "/app.js": (lambda: STATIC / "app.js", "text/javascript; charset=utf-8"),
        "/scene.glb": (lambda: app.scene_dir / "scene.glb", "model/gltf-binary"),
        "/manifest.json": (lambda: app.scene_dir / "manifest.json", "application/json"),
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
                locate, content_type = files[path]
                file_path = locate()
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
            if self.path == "/original":
                self._json(app.original())
                return
            if self.path == "/sample":
                try:
                    body = json.loads(raw.decode("utf-8") or "{}")
                    seed = body.get("seed")
                    seed = None if seed is None else int(seed)
                except (
                    json.JSONDecodeError,
                    UnicodeDecodeError,
                    AttributeError,
                    ValueError,
                    TypeError,
                ):
                    self._json({"error": 'expected JSON {"seed": n}'}, HTTPStatus.BAD_REQUEST)
                    return
                self._json(app.sample(seed))
                return
            if self.path != "/command":
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                text = str(json.loads(raw.decode("utf-8") or "{}").get("text", "")).strip()
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                self._json({"error": 'expected JSON {"text": ...}'}, HTTPStatus.BAD_REQUEST)
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
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="joint shape model (shapes.build) to draw generated thoraxes from",
    )
    args = parser.parse_args(argv)
    sampler = None
    if args.model is not None:
        from shapes.generate import write_sample_scene
        from shapes.joint import JointModel

        joint = JointModel.load(args.model)
        provenance_path = args.model.parent.parent / "provenance.json"
        provenance = (
            json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance_path.is_file()
            else None
        )

        def sampler(seed: int, out: Path) -> dict[str, Any]:
            return write_sample_scene(joint, out, seed=seed, provenance=provenance)

    server = make_server(ViewerApp(args.scene, sampler=sampler), args.port)
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
