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

    def test_an_organ_without_geometry_for_its_lod_falls_back(self) -> None:
        """Generated scenes have lung shells but no lobes: the shell is drawn, AWR unchanged."""
        engine = CommandEngine(config=load_domain_config(domain="thorax"))
        engine.execute("generate the heart and lungs")
        modelled = [
            "heart.left_atrium",
            "heart.left_ventricle",
            "heart.right_atrium",
            "heart.right_ventricle",
            "heart.myocardium",
            "respiratory.left_lung",
            "respiratory.right_lung",
        ]
        generated = {"entities": {e: {} for e in modelled}}
        real = {
            "entities": {e: {} for e in [*modelled, "heart", "respiratory.lung_upper_lobe_left"]}
        }

        def flags(manifest: dict[str, Any], entity: str) -> tuple[bool, bool]:
            state = scene_state(engine, manifest)
            e = next(x for x in state["entities"] if x["id"] == entity)
            return e["visible"], e["drawn"]

        # LOD 1: lobes are the chosen representation; generated scenes have none.
        assert flags(generated, "respiratory.left_lung") == (False, True)
        assert flags(real, "respiratory.left_lung") == (False, False)
        assert flags(real, "respiratory.lung_upper_lobe_left") == (True, True)
        assert flags(generated, "heart.left_ventricle") == (True, True)
        # LOD 0: the heart shell is chosen; without its geometry, the chambers stand in.
        engine.execute("show less detail")
        assert flags(generated, "heart.left_ventricle") == (False, True)
        assert flags(real, "heart.left_ventricle") == (False, False)
        # Hidden by request: nothing falls back.
        engine.execute("hide the lungs")
        assert flags(generated, "respiratory.left_lung") == (False, False)

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
        for path, needle in (
            ("/", b"Lagnav"),
            ("/app.js", b"GLTFLoader"),
            ("/scene.glb", b"glTF"),
            ("/manifest.json", b"CC BY 4.0"),
        ):
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
    def test_bad_requests_are_refused(
        self, server: str, path: str, body: bytes, expected: int
    ) -> None:
        """Each with a JSON error, never a stack trace."""
        status, payload = _post(server + path, body)
        assert status == expected
        assert "error" in payload

    def test_unknown_files_are_not_served(self, server: str) -> None:
        """Only the four named files; no directory walking."""
        for path in ("/../pyproject.toml", "/static/app.js", "/viewer/server.py"):
            status, _ = _get(server + path)
            assert status == 404, path


class TestAnimationSchedule:
    """The page plays the engine's semantic clip; the schedule comes with state."""

    def test_nothing_plays_until_asked(self, engine) -> None:
        """No clip in state before 'animate blood flow'."""
        assert scene_state(engine, MANIFEST)["animation"] is None

    def test_the_schedule_covers_the_cycle_with_roles(self, engine) -> None:
        """Phases tile [0, 1] in order, and the ventricles have a role in every phase."""
        engine.execute("animate blood flow")
        clip = scene_state(engine, MANIFEST)["animation"]
        phases = clip["phases"]
        assert phases[0]["start"] == 0.0 and phases[-1]["end"] == pytest.approx(1.0)
        for before, after in zip(phases, phases[1:], strict=False):
            assert before["end"] == pytest.approx(after["start"])
        assert all("heart.left_ventricle" in p["roles"] for p in phases)
        assert clip["semantic_only"] is True

    def test_the_lungs_carry_flow_in_the_thorax(self, engine) -> None:
        """Pulmonary circulation puts the lungs in the clip."""
        engine.execute("animate blood flow")
        carrying = set(scene_state(engine, MANIFEST)["animation"]["carrying_flow"])
        assert {"respiratory.left_lung", "respiratory.right_lung"} <= carrying

    def test_stopping_clears_the_schedule(self, engine) -> None:
        """After 'stop the animation' the page stops playing."""
        engine.execute("animate blood flow")
        engine.execute("stop the animation")
        assert scene_state(engine, MANIFEST)["animation"] is None


class TestTheLanguageLayerInTheServer:
    """Commands route through the language layer; replies say which path ran."""

    def test_rules_by_default(self, tmp_path) -> None:
        """No translator: rules only, and the state says so."""
        (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
        app = ViewerApp(tmp_path, translator=None)
        app.translator = None
        reply = app.command("hide the trachea")
        assert reply["ok"] and reply["source"] == "rules" and reply["program"] is None
        assert reply["state"]["language"] == "rules"

    def test_free_text_shows_the_program(self, tmp_path) -> None:
        """With a translator, the reply carries the program and the commands it ran."""
        from language.program import Program, Step

        class Scripted:
            def translate(self, text: str) -> Program:
                return Program(steps=(Step("hide", ("respiratory.lungs",)),))

        (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
        app = ViewerApp(tmp_path, translator=Scripted())
        reply = app.command("get the lungs out of my way")
        assert reply["ok"] and reply["source"] == "claude"
        assert reply["commands"] == ["hide respiratory.lungs"]
        assert reply["state"]["language"] == "claude"


class TestResizeInTheServer:
    """A resize swaps geometry, the state shows the size, and new geometry resets it."""

    def _app(self, tmp_path: Path) -> tuple[ViewerApp, list[float]]:
        real = tmp_path / "real"
        real.mkdir()
        (real / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
        factors: list[float] = []

        def resizer(
            scene: Path, manifest: dict[str, Any], entity: str, factor: float, out: Path
        ) -> tuple[bool, str, dict[str, Any] | None]:
            factors.append(factor)
            if factor > 1.25:
                return False, "outside real anatomy", None
            out.mkdir(parents=True)
            written = {**manifest, "resized": [*manifest.get("resized", []), {"entity": entity}]}
            (out / "manifest.json").write_text(json.dumps(written), encoding="utf-8")
            return True, f"{entity} resized", written

        def sampler(seed: int, out: Path) -> dict[str, Any]:
            out.mkdir(parents=True, exist_ok=True)
            return {**MANIFEST, "generated": {"seed": seed}}

        app = ViewerApp(real, resizer=resizer, sampler=sampler, samples_dir=tmp_path / "s")
        return app, factors

    @staticmethod
    def _size(reply: dict[str, Any], entity: str = "heart.left_ventricle") -> float:
        return next(e["size"] for e in reply["state"]["entities"] if e["id"] == entity)

    def test_a_resize_swaps_geometry_and_records_the_size(self, tmp_path: Path) -> None:
        """Applied: new folder, size in the state, and the page told to reload."""
        app, factors = self._app(tmp_path)
        reply = app.command("make the left ventricle 20% larger")
        assert reply["ok"] and reply["geometry_changed"] and factors == [1.2]
        assert self._size(reply) == 1.2 and reply["state"]["resizing"] is True
        assert app.scene_dir == tmp_path / "s" / "resize-1"
        assert reply["state"]["resized"] == [{"entity": "heart.left_ventricle"}]

    def test_a_refused_resize_changes_nothing(self, tmp_path: Path) -> None:
        """Not applied: same folder, size 1, the reason in the reply."""
        app, _ = self._app(tmp_path)
        reply = app.command("make the left ventricle 30% larger")
        assert not reply["ok"] and not reply["geometry_changed"]
        assert reply["message"] == "outside real anatomy" and self._size(reply) == 1.0

    def test_reset_and_new_geometry_undo_sizes(self, tmp_path: Path) -> None:
        """Reset restores the geometry before the resizes; a sample resets sizes and says so."""
        app, _ = self._app(tmp_path)
        app.command("make the left ventricle 10% larger")
        reset = app.reset()
        assert reset["geometry_changed"] and app.scene_dir == tmp_path / "real"
        assert self._size(reset) == 1.0
        app.command("make the left ventricle 10% larger")
        sampled = app.sample(3)
        assert "Sizes reset for 1 resized structure." in sampled["message"]
        assert self._size(sampled) == 1.0

    def test_without_a_model_resize_says_how(self, tmp_path: Path) -> None:
        """No resizer: refused with the way to enable it."""
        (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
        reply = ViewerApp(tmp_path).command("make the left ventricle 10% larger")
        assert not reply["ok"] and "--model" in reply["message"]
        assert reply["state"]["resizing"] is False
