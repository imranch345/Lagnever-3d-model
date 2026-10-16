"""Everything the dashboard shows, read live from files on this machine. Standard library only.

Nothing here runs or imports research code: Step 21 results are read from the JSON each run
writes, and job states are inferred from the markers the chains append to their logs. A job is
*running* when its log changed recently and has no exit marker since its last start, *done*
when the exit marker is there, *stopped* when the log went quiet without one, and *waiting*
when it has not started.

Two roots: ``product`` is this checkout (Level 1 and 2a, docs, evidence); ``research`` is the
checkout where Step 21 runs (``~/code/lagnav-3d``). They can be the same folder.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DOCS",
    "Roots",
    "collect",
    "doc_path",
    "evidence",
    "evidence_path",
    "level2a",
    "notebook_entries",
    "results_rows",
    "step21",
]

#: A log untouched for this long, with no exit marker, counts as stopped.
QUIET_SECONDS = 30 * 60
BRANCHES = tuple(f"branch_{i:02d}" for i in range(8))
STAGE_ONE = (400, 600, 700)
#: Step 20's frozen result at 805 is part of the pattern (unanimous, escape 830-840).
STEP20 = {"point": 805, "escaped": 8, "branches": 8, "verdict": "unanimous"}
#: Frozen seed 9 escapes at 830 < e <= 835 (Step 18).
FROZEN_ESCAPE = (830, 835)
#: Results files the docs tab may open: id -> (root, relative path, title).
DOCS: dict[str, tuple[str, str, str]] = {
    "notebook": ("product", "docs/LAB_NOTEBOOK.md", "Lab notebook"),
    "results": ("product", "docs/RESULTS.md", "Results summary"),
    "level2a": ("product", "docs/LEVEL_2A_REPORT.md", "Level 2a report"),
    "level2-plan": ("product", "docs/LEVEL_2_PLAN.md", "Level 2 plan"),
    "level1-plan": ("product", "docs/LEVEL_1_PLAN.md", "Level 1 plan"),
    "adr-resize": (
        "product",
        "docs/adr/ADR-L2-001-resize-through-the-shape-model.md",
        "ADR-L2-001: resize",
    ),
    "adr-language": (
        "product",
        "docs/adr/ADR-L1-005-language-layer-claude-programs.md",
        "ADR-L1-005: language layer",
    ),
    "step21-plan": ("research", "docs/STEP_21_BASIN_ENTRY_PLAN.md", "Step 21 preregistration"),
    "step20": ("research", "docs/STEP_20_CHECKPOINT_BRANCHING_REPORT.md", "Step 20 report"),
    "evidence": ("product", "docs/evidence/README.md", "Evidence index"),
}
EVIDENCE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".md"})


@dataclass(frozen=True)
class Roots:
    """Where to read from."""

    product: Path
    research: Path

    def get(self, name: str) -> Path:
        """``product`` or ``research``."""
        return self.research if name == "research" else self.product


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _since_last(lines: Sequence[str], marker: str) -> list[str]:
    """The lines from the last one containing ``marker`` (inclusive), or all of them."""
    for index in range(len(lines) - 1, -1, -1):
        if marker in lines[index]:
            return list(lines[index:])
    return list(lines)


def _age(path: Path, now: float) -> float | None:
    try:
        return now - path.stat().st_mtime
    except OSError:
        return None


def process_alive(pattern: str) -> bool:
    """Whether a process whose command line contains ``pattern`` is running (``pgrep -f``)."""
    try:
        found = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return found.returncode == 0 and bool(found.stdout.strip())


def _job(
    name: str,
    log: Path,
    *,
    started: str | None,
    finished: str,
    now: float,
    detail: str = "",
    since: str = "=== RESUMED",
    process: str | None = None,
) -> dict[str, Any]:
    """One job's state from its log's markers and freshness, or its process when quiet.

    Some jobs write nothing until they finish, so a quiet log is not proof of a stop: a job
    whose ``process`` pattern matches a running command is running.
    """
    lines = _lines(log)
    age = _age(log, now)
    current = _since_last(lines, since) if any(since in line for line in lines) else lines
    if not lines or (started is not None and not any(started in line for line in current)):
        state = "waiting"
    elif any(finished in line for line in current):
        state = "done"
    elif (age is not None and age < QUIET_SECONDS) or (process and process_alive(process)):
        state = "running"
    else:
        state = "stopped"
    return {
        "name": name,
        "state": state,
        "detail": detail,
        "updated_seconds_ago": None if age is None else round(age),
        "log": str(log),
    }


# ----------------------------------------------------------------------------------------------
# Step 21


def _branch(folder: Path, name: str, now: float) -> dict[str, Any]:
    """One continuation: done with its outcome, running, stopped, or pending."""
    result = _read_json(folder / f"{name}.json")
    if isinstance(result, Mapping):
        escape = result.get("escape", {})
        final = result.get("final", {})
        entry: dict[str, Any] = {
            "name": name,
            "state": "done",
            "loader_seed": result.get("loader_seed"),
            "escaped": bool(escape.get("escaped")),
            "escape_step": escape.get("permanent_crossing_step"),
            "final_usage_deg": final.get("graph_usage_deg"),
            "final_rotation_deg": final.get("validation_rotation_deg"),
        }
        reproduces = result.get("reproduces_frozen_seed9")
        if isinstance(reproduces, Mapping):
            entry["reproduces"] = bool(reproduces.get("reproduces"))
            entry["exact"] = reproduces.get("exact")
            entry["shared"] = reproduces.get("shared_checkpoints")
        return entry
    age = _age(folder / f"{name}.log", now)
    if age is None:
        return {"name": name, "state": "pending"}
    return {"name": name, "state": "running" if age < QUIET_SECONDS else "stopped"}


def _spread(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    # Population SD, as Step 21's own summary reports it.
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {
        "mean": round(mean, 4),
        "sd": round(var**0.5, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _stage_two(lines: Sequence[str]) -> list[int]:
    for line in reversed(lines):
        found = re.search(r"Stage 2: \[([0-9, ]*)\]", line)
        if found:
            return [int(x) for x in found.group(1).split(",") if x.strip()]
    return []


def step21(research: Path, now: float | None = None) -> dict[str, Any]:
    """Every branch point's runs, verdicts, integrity, and the sweep's state."""
    now = time.time() if now is None else now
    out = research / "experiments" / "runs" / "step21"
    log = out / "sweep.log"
    lines = _lines(log)
    stage_two = _stage_two(lines)
    points: list[dict[str, Any]] = []
    for point in sorted({*STAGE_ONE, *stage_two}, reverse=True):
        folder = out / f"point_{point:04d}"
        names = ["reference", *BRANCHES]
        if point == STAGE_ONE[0]:
            names.append("branch_00_replay")
        runs = [_branch(folder, name, now) for name in names]
        branches = [r for r in runs if r["name"] in BRANCHES]
        done = [b for b in branches if b["state"] == "done"]
        escaped = [b for b in done if b["escaped"]]
        entry: dict[str, Any] = {
            "point": point,
            "stage": 1 if point in STAGE_ONE else 2,
            "runs": runs,
            "done": sum(r["state"] == "done" for r in runs),
            "total": len(runs),
            "escaped": len(escaped),
            "decided": len(done) == len(BRANCHES),
            "verdict": None,
            "escape_steps": _spread([float(b["escape_step"]) for b in escaped]),
            "final_rotation": _spread([float(b["final_rotation_deg"]) for b in escaped]),
        }
        if entry["decided"]:
            entry["verdict"] = (
                "unanimous"
                if len(escaped) == len(BRANCHES)
                else ("none" if not escaped else "split")
            )
        reference = next(r for r in runs if r["name"] == "reference")
        entry["reference_reproduces"] = reference.get("reproduces")
        points.append(entry)
    current = _since_last(lines, "=== RESUMED")
    job = _job(
        "Step 21 sweep",
        log,
        started=None,
        finished="=== CHAIN EXITED ===",
        now=now,
        detail=(f"{sum(p['done'] for p in points)} of {sum(p['total'] for p in points)} runs done"),
        process="experiments.step21.basin_entry",
    )
    run_ages = [
        age
        for p in points
        for r in p["runs"]
        if (age := _age(out / f"point_{p['point']:04d}" / f"{r['name']}.log", now)) is not None
    ]
    if run_ages and job["updated_seconds_ago"] is not None:
        job["updated_seconds_ago"] = round(min(job["updated_seconds_ago"], *run_ages))
    # sweep.log only hears of starts and ends, and a run takes most of an hour; a run whose
    # own log is fresh means the sweep is alive.
    if job["state"] == "stopped" and any(
        r["state"] == "running" for p in points for r in p["runs"]
    ):
        job["state"] = "running"
    summary = _read_json(out / "summary.json")
    return {
        "question": "When is seed 9's escape into graph use fixed? Branch its training at earlier "
        "steps with 8 different data orderings each.",
        "points": points,
        "step20": STEP20,
        "frozen_escape": FROZEN_ESCAPE,
        "stage_two": stage_two,
        "stage_one_done": any("=== STAGE 1 DONE ===" in line for line in lines),
        "stage_two_done": any("=== STAGE 2 DONE ===" in line for line in current),
        "job": job,
        "log_tail": [line for line in lines if line.strip()][-10:],
        "summary": summary if isinstance(summary, Mapping) else None,
    }


# ----------------------------------------------------------------------------------------------
# Level 2a


def _rejections(subjects: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for subject in subjects:
        if subject.get("usable"):
            continue
        kinds = set()
        for finding in subject.get("hard", []):
            if "field of view" in finding:
                kinds.add("cut by the field of view")
            elif finding.startswith("missing"):
                kinds.add("structures missing")
            elif "pieces" in finding:
                kinds.add("in separate pieces")
            else:
                kinds.add("other")
        counts.update(kinds)
    return dict(counts.most_common())


def level2a(product: Path, now: float | None = None) -> dict[str, Any]:
    """Fetch, quality gate, registration, models, validity and samples."""
    now = time.time() if now is None else now
    processed = product / "datasets" / "processed"
    data_log = processed / "level1" / "level2a_data.log"
    build_log = processed / "level2a" / "build.log"
    validity_log = processed / "level2a" / "validity.log"
    lines = _lines(data_log)
    fetched = next((line for line in reversed(lines) if " subjects, " in line), "")
    quality = _read_json(processed / "level1" / "real" / "quality_report.json")
    report = _read_json(processed / "level2a" / "report.json")
    validity = _read_json(processed / "level2a" / "validity.json")
    registrations = processed / "level2a" / "registrations"
    raw = product / "datasets" / "raw" / "totalsegmentator_v1"
    samples = []
    for manifest_path in sorted((processed / "level2a" / "samples").glob("*/manifest.json")):
        manifest = _read_json(manifest_path)
        if isinstance(manifest, Mapping):
            checks = manifest.get("checks", {})
            samples.append(
                {
                    "name": manifest_path.parent.name,
                    "seed": manifest.get("generated", {}).get("seed"),
                    "usable": bool(checks.get("usable")),
                    "hard": checks.get("hard", []),
                }
            )
    subjects = quality.get("subjects", []) if isinstance(quality, Mapping) else []
    on_disk = len([p for p in raw.iterdir() if p.is_dir()]) if raw.is_dir() else 0
    jobs = [
        _job(
            "Fetch 200 subjects",
            data_log,
            started=None,
            finished="=== FETCH DONE ===",
            now=now,
            detail=f"{on_disk} subjects on disk" if on_disk else "",
            since="=== START",
            process="assets.fetch",
        ),
        _job(
            "Mesh build and quality gate",
            data_log,
            started=None,
            finished="=== CHAIN2 EXITED ===",
            now=now,
            detail=(f"{sum(q.get('usable', False) for q in subjects)} of {len(subjects)} usable")
            if subjects
            else "",
            process="assets.quality",
        ),
        _job(
            "Shape-model build",
            build_log,
            started="=== BUILD START",
            finished="=== CHAIN3 EXITED",
            now=now,
            detail=f"{len(list(registrations.glob('*.npz')))} registrations cached"
            if registrations.is_dir()
            else "",
            since="=== BUILD START",
            process="shapes.build",
        ),
        _job(
            "Validity sweep and samples",
            validity_log,
            started="=== VALIDITY START",
            finished="=== CHAIN4 EXITED",
            now=now,
            detail="validity.json written" if validity else "",
            since="=== VALIDITY START",
            process="shapes.validity",
        ),
    ]
    return {
        "subjects_on_disk": on_disk,
        "fetch_line": fetched,
        "quality": {
            "total": len(subjects),
            "usable": sum(bool(q.get("usable")) for q in subjects),
            "chosen": quality.get("chosen") if isinstance(quality, Mapping) else None,
            "rejections": _rejections(subjects),
        },
        "report": report if isinstance(report, Mapping) else None,
        "validity": validity if isinstance(validity, Mapping) else None,
        "samples": samples,
        "jobs": jobs,
    }


# ----------------------------------------------------------------------------------------------
# Documents


def _table_rows(text: str, prefix: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        if line.startswith(prefix):
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def results_rows(product: Path) -> dict[str, list[dict[str, str]]]:
    """The headline tables of RESULTS.md: research R-rows and product P-rows."""
    text = "\n".join(_lines(product / "docs" / "RESULTS.md"))
    research = [
        {"id": r[0], "claim": r[1], "numbers": r[2], "source": r[3]}
        for r in _table_rows(text, "| R")
        if len(r) >= 4
    ]
    product_rows = [
        {"id": r[0], "claim": r[1], "numbers": r[2], "source": r[3]}
        for r in _table_rows(text, "| P")
        if len(r) >= 4
    ]
    return {"research": research, "product": product_rows}


def notebook_entries(product: Path, limit: int = 8) -> list[str]:
    """The latest notebook headings, newest first."""
    headings = [
        line.removeprefix("## ").strip()
        for line in _lines(product / "docs" / "LAB_NOTEBOOK.md")
        if line.startswith("## ")
    ]
    return list(reversed(headings))[:limit]


def evidence(product: Path) -> list[dict[str, str]]:
    """The captioned evidence table, newest first, for files that exist."""
    folder = product / "docs" / "evidence"
    items = []
    for row in _table_rows("\n".join(_lines(folder / "README.md")), "| `"):
        if len(row) < 4:
            continue
        name = row[0].strip("`")
        if (folder / name).is_file():
            items.append({"file": name, "date": row[1], "shows": row[2], "produced": row[3]})
    return list(reversed(items))


def evidence_path(product: Path, name: str) -> Path | None:
    """An evidence file by bare name, or None: no folders, no other suffixes."""
    folder = (product / "docs" / "evidence").resolve()
    if "/" in name or "\\" in name or name.startswith("."):
        return None
    path = (folder / name).resolve()
    if path.parent != folder or path.suffix.lower() not in EVIDENCE_SUFFIXES:
        return None
    return path if path.is_file() else None


def doc_path(roots: Roots, doc_id: str) -> Path | None:
    """A whitelisted document's path, or None."""
    if doc_id not in DOCS:
        return None
    root, relative, _ = DOCS[doc_id]
    path = roots.get(root) / relative
    return path if path.is_file() else None


def collect(roots: Roots, now: float | None = None) -> dict[str, Any]:
    """The whole status, as the dashboard's ``/api/status`` returns it."""
    now = time.time() if now is None else now
    research = step21(roots.research, now)
    shapes = level2a(roots.product, now)
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "jobs": [research["job"], *shapes["jobs"]],
        "step21": research,
        "level2a": shapes,
        "results": results_rows(roots.product),
        "notebook": notebook_entries(roots.product),
        "evidence": evidence(roots.product),
        "docs": [
            {"id": key, "title": title, "available": doc_path(roots, key) is not None}
            for key, (_, _, title) in DOCS.items()
        ],
    }
