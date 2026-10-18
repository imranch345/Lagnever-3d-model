"""Local server for the Level 1 viewer. Standard library only; binds to 127.0.0.1.

Routes:

``GET /``                the page;  ``GET /app.js`` its script
``GET /scene.glb``       the built scene;  ``GET /manifest.json`` its manifest
``GET /state``           the scene state the page draws (``viewer.state.scene_state``)
``POST /command``        ``{"text": "..."}`` -> the engine's result plus the new state
``POST /reset``          rebuild the scene from scratch
``POST /sample``         ``{"seed": n}`` -> draw a thorax from the shape model (``--model``)
``POST /original``       back to the real subject's geometry

A request such as "make the left ventricle 20% larger" reshapes one structure through the shape
model (``--model`` with a ``validity.json`` beside it; ADR-L2-001). The reply then has
``geometry_changed`` set, so the page reloads the meshes.

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
from awr.schema import Transform
from editing.operations import SetTransform
from language.layer import (
    LanguageLayer,
    Translator,
    load_user_env,
    translator_from_environment,
)
from reasoning.command_engine import CommandEngine
from viewer.state import scene_state

__all__ = ["Resizer", "ViewerApp", "build_app", "make_server", "main"]

STATIC = Path(__file__).parent / "static"
MAX_COMMAND_BYTES = 4096

#: (scene folder, its manifest, entity, factor, output folder) -> (applied, message, manifest).
Resizer = Callable[
    [Path, dict[str, Any], str, float, Path], tuple[bool, str, dict[str, Any] | None]
]


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
        resizer: Resizer | None = None,
    ) -> None:
        self.scene_dir = scene_dir
        self.original_dir = scene_dir
        # Draws a thorax from a shape model into a folder and returns its manifest. Injected,
        # so the server needs the shape tooling only when a model is actually loaded.
        self.sampler = sampler
        # Reshapes one entity through its shape model into a new folder (ADR-L2-001).
        self.resizer = resizer
        self.samples_dir = samples_dir or scene_dir.parent / "samples"
        self._next_seed = 1
        self._resizes = 0
        self.domain = domain
        self.manifest: dict[str, Any] = json.loads(
            (scene_dir / "manifest.json").read_text(encoding="utf-8")
        )
        # The geometry before any resize, restored by reset.
        self.base_dir, self.base_manifest = self.scene_dir, self.manifest
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
        state["resizing"] = self.resizer is not None
        state["generated"] = self.manifest.get("generated")
        state["resized"] = self.manifest.get("resized", [])
        return state

    def _set_geometry(self, scene_dir: Path, manifest: dict[str, Any]) -> str:
        """A new base geometry: sizes in the AWR go back to 1, since this geometry has none."""
        self.scene_dir, self.manifest = scene_dir, manifest
        self.base_dir, self.base_manifest = scene_dir, manifest
        scene, editor = self.engine.scene, self.engine.editor
        if scene is None or editor is None:
            return ""
        resized = [e for e in scene.ids() if scene.get(e).state.transform.scale != (1.0, 1.0, 1.0)]
        for entity in resized:
            before = scene.get(entity).state.transform
            editor.apply(
                SetTransform((entity,), Transform(before.translation, before.rotation)),
                command_text="new geometry",
            )
        if not resized:
            return ""
        return f" Sizes reset for {len(resized)} resized structure{'s' * (len(resized) != 1)}."

    def _resize(self, entity: str, factor: float) -> tuple[bool, str]:
        """The language layer's resize hook: reshape into a new folder, swap on success."""
        if self.resizer is None:
            return False, "Resizing needs a shape model: start the viewer with --model."
        self._resizes += 1
        out = self.samples_dir / f"resize-{self._resizes}"
        ok, message, manifest = self.resizer(self.scene_dir, self.manifest, entity, factor, out)
        if ok and manifest is not None:
            self.scene_dir, self.manifest = out, manifest
        return ok, message

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
            reset_note = self._set_geometry(self.samples_dir / f"seed-{seed}", manifest)
            checks = manifest.get("checks", {})
            verdict = (
                "passes the quality checks"
                if checks.get("usable")
                else f"fails: {'; '.join(checks.get('hard', []))}"
            )
            validity = manifest.get("validity")
            if validity is not None:
                verdict += (
                    " and the validity checks"
                    if validity["passes"]
                    else f"; fails validity: {'; '.join(validity['failures'][:3])}"
                )
            generated = manifest.get("generated", {})
            spread = generated.get("spread")
            if spread is not None:
                tries = generated.get("sampler", {}).get("tries")
                how = f" on try {tries}" if tries else ""
                verdict = f"drawn at {spread:g} SD{how}, {verdict}"
            return {
                "ok": True,
                "message": f"Generated thorax, seed {seed}: {verdict}.{reset_note}",
                "state": self._state(),
            }

    def original(self) -> dict[str, Any]:
        """Back to the real subject's geometry."""
        with self.lock:
            manifest = json.loads((self.original_dir / "manifest.json").read_text(encoding="utf-8"))
            reset_note = self._set_geometry(self.original_dir, manifest)
            return {
                "ok": True,
                "message": f"Real subject {self.manifest.get('subject')}.{reset_note}",
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
            before = self.scene_dir
            try:
                hook = self._resize if self.resizer is not None else None
                result = LanguageLayer(self.engine, self.translator, resize=hook).execute(text)
                ok, message, source = result.ok, result.message, result.source
                program, commands = result.program, result.commands
                if "reset_view" in result.operations:
                    # A typed reset does what the Reset button does: resizes are undone too.
                    message += self._undo_resizes()
            except Exception as exc:  # noqa: BLE001 - reported to the page, not swallowed
                ok, message = False, f"{type(exc).__name__}: {exc}"
            return {
                "ok": ok,
                "message": message,
                "source": source,
                "program": program,
                "commands": commands,
                "geometry_changed": self.scene_dir != before,
                "state": self._state(),
            }

    def language_enabled(self) -> bool:
        """Whether free-form text can reach the language model."""
        return self.translator is not None

    def _undo_resizes(self) -> str:
        """Back to the geometry before any resize, with every size in the AWR back to 1."""
        if self.scene_dir == self.base_dir:
            return ""
        self._set_geometry(self.base_dir, self.base_manifest)
        return " Resizes undone."

    def reset(self) -> dict[str, Any]:
        """Start the scene again."""
        with self.lock:
            self.engine = self._fresh_engine()
            changed = self.scene_dir != self.base_dir
            note = self._undo_resizes()
            return {
                "ok": True,
                "message": "Scene reset." + note,
                "geometry_changed": changed,
                "state": self._state(),
            }


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


def build_app(scene: Path, model: Path | None = None) -> ViewerApp:
    """The viewer for a built scene, with sampling and resizing when a model is given."""
    sampler = None
    resizer: Resizer | None = None
    if model is not None:
        from shapes.generate import write_sample_scene
        from shapes.joint import JointModel

        joint = JointModel.load(model)
        provenance_path = model.parent.parent / "provenance.json"
        provenance = (
            json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance_path.is_file()
            else None
        )

        models_dir = model.parent.parent
        validity_path = models_dir / "validity.json"
        if validity_path.is_file():
            from shapes.generate import write_valid_sample_scene
            from shapes.validity import Limits

            validity = json.loads(validity_path.read_text(encoding="utf-8"))
            limits = Limits.from_dict(validity["limits"])

            # 1 SD, kept only if valid: real-person novelty, and every shown sample passes
            # (ADR-L2-002).
            def sampler(seed: int, out: Path) -> dict[str, Any]:
                return write_valid_sample_scene(
                    joint, out, seed=seed, limits=limits, provenance=provenance
                )

        else:

            def sampler(seed: int, out: Path) -> dict[str, Any]:
                return write_sample_scene(joint, out, seed=seed, provenance=provenance)

        if (models_dir / "validity.json").is_file():
            from shapes.resize import Resizer as ShapeResizer

            shape_resizer = ShapeResizer(models_dir)

            def resizer(
                scene: Path, manifest: dict[str, Any], entity: str, factor: float, out: Path
            ) -> tuple[bool, str, dict[str, Any] | None]:
                outcome = shape_resizer.resize(scene, manifest, entity, factor, out)
                return outcome.ok, outcome.message, outcome.manifest

        else:
            print("No validity.json beside the model: resizing is off.", flush=True)

    # Generated and resized scenes go beside the model, never among the real subjects, where
    # the quality report would count them as scans.
    samples_dir = model.parent.parent / "viewer_scenes" if model is not None else None
    return ViewerApp(scene, sampler=sampler, resizer=resizer, samples_dir=samples_dir)


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
    loaded = load_user_env()
    if loaded:
        print(f"Loaded from ~/.lagnav/anthropic.env: {', '.join(sorted(loaded))}", flush=True)
    app = build_app(args.scene, args.model)
    server = make_server(app, args.port)
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
