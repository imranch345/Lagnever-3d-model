"""Level 1: the viewer's state export and its local server.

The page draws only what ``/state`` says, and changes the scene only through ``/command``
text, so these two are where correctness lives. The server is exercised over real HTTP on an
ephemeral port; it needs no asset extra (the GLB is served as opaque bytes).
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from awr.config import load_domain_config
from reasoning.command_engine import CommandEngine
from viewer.server import MAX_COMMAND_BYTES, ViewerApp, make_server
from viewer.state import DEFAULT_COLOURS, colour_for, scene_state

MANIFEST: dict[str, Any] = {
    "subject": "phantom",
    "entities": {"respiratory.trachea": {}, "heart.left_ventricle": {}},
    "source": {
        "dataset": "TotalSegmentator dataset",
        "version": "1.0",
        "license": "CC BY 4.0",
        "attribution": "Wasserthal et al.",
        "url": "https://zenodo.org/records/6802614",
    },
    "medical_status": "research CT, not validated; not for clinical use",
}


@pytest.fixture()
def engine() -> CommandEngine:
    """A thorax scene at its default level of detail."""
    engine = CommandEngine(config=load_domain_config(domain="thorax"))
    engine.execute("generate a human heart")
    return engine


class TestSceneState:
    """What the page is told."""

    def test_geometry_is_flagged_only_where_a_mesh_exists(self, engine) -> None:
        """Valves are in the scene but must be marked as having no shape."""
        entities = {e["id"]: e for e in scene_state(engine, MANIFEST)["entities"]}
        assert entities["respiratory.trachea"]["has_geometry"] is True
        assert entities["heart.mitral_valve"]["has_geometry"] is False
        assert len(entities) == 53

    def test_state_follows_the_engine(self, engine) -> None:
        """A command's effect shows up in the next state, and nowhere else."""
        before = {e["id"]: e for e in scene_state(engine, MANIFEST)["entities"]}
        engine.execute("make the left ventricle transparent")
        after = {e["id"]: e for e in scene_state(engine, MANIFEST)["entities"]}
        assert after["heart.left_ventricle"]["opacity"] == pytest.approx(0.3)
        changed = [i for i in after if after[i] != before[i]]
        assert changed == ["heart.left_ventricle"]

    def test_credits_and_status_come_from_the_manifest(self, engine) -> None:
        """ADR-L1-004: the page's credits are the manifest's, not hard-coded."""
        state = scene_state(engine, MANIFEST)
        assert state["credits"]["license"] == "CC BY 4.0"
        assert state["credits"]["subject"] == "phantom"
        assert "not validated" in state["medical_status"]

    def test_colours(self) -> None:
        """An engine-set colour wins; otherwise the default; otherwise a neutral."""
        assert colour_for("heart.left_ventricle") == DEFAULT_COLOURS["heart.left_ventricle"]
        assert colour_for("heart.left_ventricle", (1.0, 0.0, 0.5)) == "#ff0080"
        assert colour_for("heart.nothing").startswith("#")

    def test_no_scene_is_an_error_not_an_empty_page(self) -> None:
        """Asking before generating is a bug in the caller."""
        with pytest.raises(RuntimeError):
            scene_state(CommandEngine(config=load_domain_config(domain="thorax")), MANIFEST)


@pytest.fixture()
def server(tmp_path: Path) -> Iterator[str]:
    """A live server on 127.0.0.1 and an ephemeral port, over a fake built scene."""
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    (tmp_path / "scene.glb").write_bytes(b"glTF\x02\x00\x00\x00fake")
    httpd = make_server(ViewerApp(tmp_path), port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        with error:
            return error.code, error.read()


def _post(url: str, body: bytes) -> tuple[int, Any]:
    request = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        with error:
            return error.code, json.loads(error.read())


class TestTheServer:
    """Over real HTTP."""

    def test_it_listens_on_loopback_only(self, server: str) -> None:
        """A local tool, not a service."""
        assert server.startswith("http://127.0.0.1:")

    def test_the_page_and_assets_are_served(self, server: str) -> None:
        """The page, its script, the scene and the manifest."""
        for path, needle in (("/", b"Lagnav"), ("/app.js", b"GLTFLoader"),
                             ("/scene.glb", b"glTF"), ("/manifest.json", b"CC BY 4.0")):
            status, body = _get(server + path)
            assert status == 200, path
            assert needle in body, path

    def test_a_command_changes_state_through_the_engine(self, server: str) -> None:
        """POST /command runs the text and returns the new state."""
        status, body = _post(server + "/command", json.dumps({"text": "hide the trachea"}).encode())
        assert status == 200 and body["ok"] is True
        trachea = next(e for e in body["state"]["entities"] if e["id"] == "respiratory.trachea")
        assert trachea["visible"] is False
        status, raw = _get(server + "/state")
        assert json.loads(raw)["scene_version"] == body["state"]["scene_version"]

    def test_a_command_the_engine_rejects_is_reported_not_raised(self, server: str) -> None:
        """An unknown structure is the engine's answer, not a server error."""
        status, body = _post(server + "/command", json.dumps({"text": "hide the spleen"}).encode())
        assert status == 200 and body["ok"] is False and body["message"]

    def test_reset_rebuilds_the_scene(self, server: str) -> None:
        """After reset, the hidden trachea is back."""
        _post(server + "/command", json.dumps({"text": "hide the trachea"}).encode())
        status, body = _post(server + "/reset", b"")
        trachea = next(e for e in body["state"]["entities"] if e["id"] == "respiratory.trachea")
        assert status == 200 and trachea["visible"] is True

    @pytest.mark.parametrize(
        ("path", "body", "expected"),
        [
            ("/command", b"not json", 400),
            ("/command", json.dumps({"text": "   "}).encode(), 400),
            ("/command", b"[1, 2]", 400),
            ("/nowhere", b"{}", 404),
            ("/command", b"x" * (MAX_COMMAND_BYTES + 1), 413),
        ],
        ids=["bad-json", "empty-text", "not-an-object", "unknown-path", "too-large"],
    )
    def test_bad_requests_are_refused(self, server: str, path: str, body: bytes,
                                      expected: int) -> None:
        """Each with a JSON error, never a stack trace."""
        status, payload = _post(server + path, body)
        assert status == expected
        assert "error" in payload

    def test_unknown_files_are_not_served(self, server: str) -> None:
        """Only the four named files; no directory walking."""
        for path in ("/../pyproject.toml", "/static/app.js", "/viewer/server.py"):
            status, _ = _get(server + path)
            assert status == 404, path
