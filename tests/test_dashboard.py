"""The dashboard: its status reads files honestly, and its server serves only what it should.

Job states come from log markers, freshness and processes. A Step 21 point is decided only
when all eight branches have results. Documents and evidence are served from whitelists, so
a crafted path cannot reach other files.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from dashboard import status as st
from dashboard.server import make_dashboard_server
from viewer.server import ViewerApp


def _branch(folder: Path, name: str, *, escaped: bool, step: int = 800, seed: int = 2000) -> None:
    payload: dict[str, Any] = {
        "loader_seed": seed,
        "escape": {"escaped": escaped, "permanent_crossing_step": step if escaped else None},
        "final": {"graph_usage_deg": 2.5, "validation_rotation_deg": 21.0},
    }
    if name == "reference":
        payload["reproduces_frozen_seed9"] = {
            "reproduces": True,
            "exact": 30,
            "shared_checkpoints": 30,
        }
    (folder / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def roots(tmp_path: Path) -> st.Roots:
    """A product and a research checkout with just enough files."""
    product, research = tmp_path / "product", tmp_path / "research"
    runs = research / "experiments" / "runs" / "step21"
    for point in (400, 600, 700, 200):
        folder = runs / f"point_{point:04d}"
        folder.mkdir(parents=True)
        if point == 200:
            _branch(folder, "branch_00", escaped=True)
            (folder / "branch_01.log").write_text("training…")
            continue
        _branch(folder, "reference", escaped=True)
        for i in range(8):
            _branch(folder, f"branch_{i:02d}", escaped=not (point == 600 and i == 3), step=800 + i)
        if point == 400:
            _branch(folder, "branch_00_replay", escaped=True)
    (runs / "sweep.log").write_text(
        "=== RESUMED 2026-10-16 10:14 ===\n=== STAGE 1 DONE ===\n[step21] Stage 2: [200, 0]\n",
        encoding="utf-8",
    )
    docs = product / "docs"
    (docs / "evidence").mkdir(parents=True)
    (docs / "RESULTS.md").write_text(
        "| R1 | A claim. | 1.0 vs 2.0 | report |\n| P1 | A milestone. | 17 surfaces | shot |\n",
        encoding="utf-8",
    )
    (docs / "LAB_NOTEBOOK.md").write_text("# Notebook\n\n## Day one\n\n## Day two\n")
    (docs / "evidence" / "shot.jpg").write_bytes(b"\xff\xd8jpeg")
    (docs / "evidence" / "README.md").write_text(
        "| file | date | shows | produced by |\n| --- | --- | --- | --- |\n"
        "| `shot.jpg` | 2026-10-16 | A **scene**. | `cmd` |\n| `missing.png` | x | y | z |\n",
        encoding="utf-8",
    )
    (product / "secret.txt").write_text("not for the dashboard")
    return st.Roots(product=product, research=research)


class TestStatus:
    """What the dashboard is told."""

    def test_step21_points_verdicts_and_running_runs(self, roots: st.Roots) -> None:
        """Decided only with all 8 branches; a split is a split; a fresh log is running."""
        result = st.step21(roots.research)
        points = {p["point"]: p for p in result["points"]}
        assert sorted(points) == [0, 200, 400, 600, 700]
        assert points[400]["verdict"] == "unanimous" and points[400]["reference_reproduces"]
        assert points[600]["verdict"] == "split" and points[600]["escaped"] == 7
        assert points[200]["verdict"] is None and not points[200]["decided"]
        states = {r["name"]: r["state"] for r in points[200]["runs"]}
        assert states["branch_00"] == "done" and states["branch_01"] == "running"
        assert states["branch_02"] == "pending"
        assert points[0]["done"] == 0
        assert result["stage_two"] == [200, 0] and result["stage_one_done"]
        assert result["job"]["state"] == "running"

    def test_a_quiet_log_is_stopped_unless_its_process_lives(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No marker and no recent write: stopped, or running if the process is alive."""
        log = tmp_path / "job.log"
        log.write_text("=== VALIDITY START ===\n")
        old = time.time() - 2 * st.QUIET_SECONDS
        os.utime(log, (old, old))

        def job(alive: bool) -> str:
            monkeypatch.setattr(st, "process_alive", lambda pattern: alive)
            return st._job(
                "v",
                log,
                started="=== VALIDITY START",
                finished="=== CHAIN4 EXITED",
                now=time.time(),
                since="=== VALIDITY START",
                process="shapes.validity",
            )["state"]

        assert job(alive=False) == "stopped" and job(alive=True) == "running"
        log.write_text("=== VALIDITY START ===\nresults\n=== CHAIN4 EXITED ===\n")
        assert job(alive=False) == "done"
        assert st._job("w", tmp_path / "none.log", started=None, finished="x", now=0)["state"] == (
            "waiting"
        )

    def test_results_notebook_and_evidence(self, roots: st.Roots) -> None:
        """Headline rows, newest notebook entries first, and only evidence that exists."""
        rows = st.results_rows(roots.product)
        assert rows["research"][0]["id"] == "R1" and rows["product"][0]["numbers"] == "17 surfaces"
        assert st.notebook_entries(roots.product) == ["Day two", "Day one"]
        assert [e["file"] for e in st.evidence(roots.product)] == ["shot.jpg"]

    def test_paths_are_whitelisted(self, roots: st.Roots) -> None:
        """Bare evidence names only; documents by id only."""
        assert st.evidence_path(roots.product, "shot.jpg") is not None
        for bad in ("../secret.txt", "..%2Fsecret.txt", "/etc/passwd", ".hidden.png", "README"):
            assert st.evidence_path(roots.product, bad) is None, bad
        assert st.doc_path(roots, "notebook") is not None
        assert st.doc_path(roots, "../secret.txt") is None
        assert st.doc_path(roots, "step21-plan") is None  # not in this research checkout

    def test_collect_works_on_an_empty_checkout(self, tmp_path: Path) -> None:
        """Nothing built yet: a complete status with empty parts, not an error."""
        empty = st.collect(st.Roots(tmp_path, tmp_path))
        assert empty["level2a"]["report"] is None and empty["level2a"]["quality"]["total"] == 0
        assert all(j["state"] == "waiting" for j in empty["jobs"][1:])


@pytest.fixture()
def server(roots: st.Roots, tmp_path: Path) -> Iterator[str]:
    """The dashboard over real HTTP on an ephemeral port."""
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "manifest.json").write_text(json.dumps({"subject": "x", "entities": {}}))
    httpd = make_dashboard_server(ViewerApp(scene), roots, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url: str) -> tuple[int, bytes, str]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 - loopback test
            return response.status, response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        with error:
            return error.code, error.read(), error.headers.get("Content-Type", "")


class TestServer:
    """Routes."""

    def test_pages_status_and_the_viewer_beside_them(self, server: str) -> None:
        """The dashboard, its script, the viewer page and the viewer's own state."""
        assert b"Lagnav Dashboard" in _get(f"{server}/")[1]
        assert _get(f"{server}/dashboard.js")[0] == 200
        assert b"/app.js" in _get(f"{server}/viewer")[1]
        assert _get(f"{server}/state")[0] == 200
        code, body, kind = _get(f"{server}/api/status")
        status = json.loads(body)
        assert code == 200 and kind == "application/json"
        assert {"jobs", "step21", "level2a", "results", "evidence", "docs"} <= set(status)

    def test_documents_and_evidence_are_served_from_whitelists(self, server: str) -> None:
        """Known ids and bare names only; everything else is a 404."""
        code, body, kind = _get(f"{server}/api/doc?id=notebook")
        assert code == 200 and b"Day two" in body and kind.startswith("text/markdown")
        assert _get(f"{server}/evidence/shot.jpg")[2] == "image/jpeg"
        for bad in (
            "/api/doc?id=../secret.txt",
            "/api/doc?id=nope",
            "/evidence/..%2Fsecret.txt",
            "/evidence/../secret.txt",
            "/evidence/README",
            "/secret.txt",
        ):
            assert _get(f"{server}{bad}")[0] == 404, bad
