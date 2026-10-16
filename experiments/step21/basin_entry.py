"""Step 21: find when seed 9's escape becomes fixed, by branching at earlier steps.

Step 20 held seed 9's full state at step 805 and varied only the subsequent data ordering; all
eight branches escaped. That says the outcome is fixed by 805, not when it became fixed. This step
repeats the same intervention from earlier states and reads off the earliest step from which the
escape is still unanimous. The design is preregistered in ``docs/STEP_21_BASIN_ENTRY_PLAN.md``.

Everything that defines the intervention is Step 20's and is imported rather than copied, so it
cannot drift:

* ``_build`` -- the frozen configuration, the trainer, and Step 16's recorder with the 805-900
  dense window;
* ``continue_run`` -- restore a state, reseed only the loader's generator (or restore the original
  stream for the reference), train to 1200, and apply Step 15's 1.5 degree escape criterion;
* ``BRANCH_SEEDS`` -- loader seeds 2000-2007, the same at every branch point;
* ``save_state`` / ``fingerprint`` / ``ResumableStream`` from ``experiments.step20.state``.

What is new is small:

``capture``
    trains seed 9 once, to step 805, saving a continuable state at every point in
    ``CAPTURE_POINTS``. It must reproduce frozen seed 9, and its 805 state must equal the one Step
    20 branched from -- otherwise nothing downstream is run.

``run``
    launches the reference and the eight branches for each requested point, at most ``MAX_JOBS``
    at once, every one at ``THREADS`` threads, skipping any whose result already exists.

``summarise``
    applies the preregistered verdicts and, once Stage 1 is complete, names the Stage 2 points the
    plan's table selects.

``stage2``
    summarises, refuses to go on unless every Stage 1 integrity check passed (``stage_two_gate``),
    then runs exactly the points the table selected -- or nothing, for a non-monotone Stage 1.

    python -m experiments.step21.basin_entry capture
    python -m experiments.step21.basin_entry run --points 400 600 700 --replay-point 400
    python -m experiments.step21.basin_entry stage2
    python -m experiments.step21.basin_entry summarise
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.step15.seed_modes import GRAPH_USING_DEG
from experiments.step20.branching import BRANCH_POINT as STEP20_BRANCH_POINT
from experiments.step20.branching import BRANCH_SEEDS, SEED, _build, continue_run
from experiments.step20.state import ResumableStream, save_state

__all__ = [
    "CAPTURE_POINTS",
    "MAX_JOBS",
    "STAGE_ONE",
    "STAGE_TWO",
    "STUCK_BELOW_DEG",
    "THREADS",
    "capture",
    "capture_states",
    "classify_branch",
    "main",
    "next_points",
    "reproduction",
    "stage_two_gate",
    "summarise_point",
    "verdict",
    "verify_freezes",
]

#: Every state the plan can need, saved by one capture run. 805 is not a branch point: it exists
#: only to prove the capture is the same state Step 20 branched from.
CAPTURE_POINTS: tuple[int, ...] = (0, 200, 400, 500, 600, 650, 700, 750, STEP20_BRANCH_POINT)

#: The three points Step 20 section 14 named.
STAGE_ONE: tuple[int, ...] = (400, 600, 700)

#: Stage 2, keyed by whether each Stage 1 point (400, 600, 700) was unanimous. Fixed before any
#: run. A pattern missing from this table is non-monotone and runs no Stage 2.
STAGE_TWO: dict[tuple[bool, bool, bool], tuple[int, ...]] = {
    (True, True, True): (200, 0),
    (False, True, True): (500,),
    (False, False, True): (650,),
    (False, False, False): (750,),
}

#: A non-escaping branch below this final usage is "stuck"; at or above it, "in transition".
#: Step 15's non-using seeds sit at 0.09-0.38; seed 3 ended its budget at 1.30.
STUCK_BELOW_DEG = 0.5

#: Part of the frozen protocol since Step 17: torch CPU reductions depend on thread count.
THREADS = 4

#: Step 20 ran three continuations at once and reproduced exactly under that load.
MAX_JOBS = 3

#: Below this, no run is launched and nothing is deleted to make space.
MIN_FREE_BYTES = 2 * 1024**3

OUT = Path("experiments/runs/step21")
CORPUS = Path("datasets/processed/step10_rotated")
RUNS_ROOT = Path("experiments/runs")

#: Frozen seed 9: Step 18's run A (dense 600-900, coarse elsewhere), the trajectory Step 20's
#: capture was verified against.
FROZEN_SEED9 = Path("experiments/runs/step18/trajectory_seed9_onset.json")

#: Step 20's capture record, checksummed in STEP20_FROZEN.sha256. Its fingerprint is the 805
#: state every Step 20 branch started from.
STEP20_CAPTURE = Path("experiments/runs/step20/capture.json")


def state_path(out_dir: Path, point: int) -> Path:
    """Where the capture saves the state at ``point``."""
    return out_dir / f"state_seed{SEED}_step{point}.pt"


def point_dir(out_dir: Path, point: int) -> Path:
    """Where the continuations from ``point`` write their results."""
    return out_dir / f"point_{point:04d}"


# ----------------------------------------------------------------------
# Capture
# ----------------------------------------------------------------------
def capture_states(
    trainer: Any, *, points: Iterable[int], out_dir: Path, recorder: Any = None
) -> dict[int, dict[str, Any]]:
    """Train from step 0 to the last point, saving a continuable state at each point.

    The order inside one step is Step 20's: measure, then save, then train. A state saved at
    ``s`` has therefore seen exactly ``s`` training steps, which is what Step 20's 805 state was,
    and the recorder -- which restores RNG, gradients and mode -- cannot leak into it.
    """
    wanted = sorted(set(points))
    if not wanted or wanted[0] < 0:
        raise ValueError(f"capture points must be non-negative, got {wanted}")
    last = wanted[-1]
    stream = ResumableStream(trainer.train_loader)
    iterator = stream.iterate()
    marks: dict[int, dict[str, Any]] = {}
    for step in range(last + 1):
        if recorder is not None:
            recorder(trainer, step)
        if step in wanted:
            marks[step] = save_state(
                trainer=trainer,
                stream=stream,
                step=step,
                path=state_path(out_dir, step),
                label=f"seed{SEED}_step{step}",
            )
        if step < last:
            trainer.train_step(next(iterator), step)
    return marks


def reproduction(
    rows: Sequence[Mapping[str, Any]], frozen_path: Path = FROZEN_SEED9, *, start: int = 0
) -> dict[str, Any]:
    """Exact agreement with frozen seed 9 at every checkpoint the two share from ``start`` on.

    Exact, not approximate: Steps 17-20 reproduced bit-for-bit at 4 threads, and a tolerance would
    hide the thread-count divergence Step 17 found, which first appeared as a small difference.
    """
    frozen = {
        int(r["step"]): r for r in json.loads(frozen_path.read_text(encoding="utf-8"))["trajectory"]
    }
    ours = {int(r["step"]): r for r in rows}
    shared = sorted(s for s in set(frozen) & set(ours) if s >= start)
    mismatches = [
        {
            "step": s,
            "usage": [ours[s]["graph_usage_deg"], frozen[s]["graph_usage_deg"]],
            "rotation": [ours[s]["validation_rotation_deg"], frozen[s]["validation_rotation_deg"]],
        }
        for s in shared
        if ours[s]["graph_usage_deg"] != frozen[s]["graph_usage_deg"]
        or ours[s]["validation_rotation_deg"] != frozen[s]["validation_rotation_deg"]
    ]
    return {
        "frozen": str(frozen_path),
        "shared_checkpoints": len(shared),
        "exact": len(shared) - len(mismatches),
        "reproduces": bool(shared) and not mismatches,
        "mismatches": mismatches,
    }


def matches_step20(mark: Mapping[str, Any], capture_path: Path = STEP20_CAPTURE) -> dict[str, Any]:
    """Whether a fingerprint equals the frozen Step 20 805 state's, field for field."""
    frozen = json.loads(capture_path.read_text(encoding="utf-8"))["fingerprint"]
    return {"step20_fingerprint": frozen, "identical": dict(mark) == frozen}


def capture(*, corpus_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Train seed 9 once to 805, saving every capture point, and check it before anything else."""
    trainer, recorder, _positions = _build(corpus_dir)
    print(f"[step21] capturing seed {SEED} at {list(CAPTURE_POINTS)}", flush=True)
    marks = capture_states(trainer, points=CAPTURE_POINTS, out_dir=out_dir, recorder=recorder)
    check = reproduction(recorder.rows)
    cross = matches_step20(marks[STEP20_BRANCH_POINT])
    return {
        "experiment_id": "step21-capture",
        "seed": SEED,
        "capture_points": list(CAPTURE_POINTS),
        "states": {str(p): str(state_path(out_dir, p)) for p in CAPTURE_POINTS},
        "fingerprints": {str(p): marks[p] for p in CAPTURE_POINTS},
        "reproduces_frozen_seed9": check,
        "step805_matches_step20": cross,
        "valid": check["reproduces"] and cross["identical"],
        "trajectory": recorder.rows,
        "test_splits_read": [],
    }


# ----------------------------------------------------------------------
# Continuations
# ----------------------------------------------------------------------
def continue_point(
    *, corpus_dir: Path, out_dir: Path, point: int, branch: str, replay: bool = False
) -> Path:
    """Run one continuation from ``point`` with Step 20's ``continue_run``; return its file."""
    loader_seed = None if branch == "reference" else BRANCH_SEEDS[branch]
    report = continue_run(
        corpus_dir=corpus_dir,
        state_path=state_path(out_dir, point),
        out_dir=out_dir,
        branch=branch,
        loader_seed=loader_seed,
    )
    report["experiment_id"] = "step21-reference" if branch == "reference" else "step21-branch"
    report["step21_point"] = point
    report["replay"] = replay
    if branch == "reference":
        report["reproduces_frozen_seed9"] = reproduction(report["trajectory"], start=point)
    target = point_dir(out_dir, point) / (f"{branch}_replay.json" if replay else f"{branch}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return target


def _jobs(points: Sequence[int], replay_point: int | None) -> list[tuple[int, str, bool]]:
    """Every continuation the plan asks for at these points: reference first, then 8 branches."""
    jobs = [(p, name, False) for p in points for name in ("reference", *sorted(BRANCH_SEEDS))]
    if replay_point is not None:
        if replay_point not in points:
            raise ValueError(f"replay point {replay_point} is not among {list(points)}")
        jobs.append((replay_point, "branch_00", True))
    return jobs


def run_points(
    *, points: Sequence[int], replay_point: int | None, out_dir: Path, jobs: int
) -> int:
    """Launch the continuations as subprocesses, at most ``jobs`` at once; return failures."""
    if jobs < 1 or jobs > MAX_JOBS:
        raise ValueError(f"jobs must be between 1 and {MAX_JOBS}, got {jobs}")
    missing = [p for p in points if not state_path(out_dir, p).exists()]
    if missing:
        raise SystemExit(f"no captured state for {missing}; run 'capture' first")
    pending = []
    for point, branch, replay in _jobs(points, replay_point):
        name = f"{branch}_replay" if replay else branch
        if (point_dir(out_dir, point) / f"{name}.json").exists():
            print(f"[step21] skip {point}/{name}: result exists", flush=True)
            continue
        pending.append((point, branch, replay, name))

    env = {**os.environ, "OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS)}
    running: list[tuple[subprocess.Popen[bytes], str, Any]] = []
    failures = 0
    while pending or running:
        while pending and len(running) < jobs:
            if shutil.disk_usage(out_dir).free < MIN_FREE_BYTES:
                raise SystemExit("free disk below 2 GiB; stopping without deleting anything")
            point, branch, replay, name = pending.pop(0)
            log_path = point_dir(out_dir, point) / f"{name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("wb")
            command = [
                sys.executable, "-m", "experiments.step21.basin_entry", "continue",
                "--point", str(point), "--branch", branch, "--out", str(out_dir),
            ]
            if replay:
                command.append("--replay")
            print(f"[step21] start {point}/{name}", flush=True)
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((process, f"{point}/{name}", log))
        time.sleep(5)
        for entry in list(running):
            process, label, log = entry
            if process.poll() is None:
                continue
            log.close()
            running.remove(entry)
            if process.returncode != 0:
                failures += 1
            status = "done" if process.returncode == 0 else f"FAILED ({process.returncode})"
            print(f"[step21] {label}: {status}", flush=True)
    return failures


# ----------------------------------------------------------------------
# Verdicts
# ----------------------------------------------------------------------
def verdict(escaped: int, total: int) -> str:
    """Section 4: unanimous only at total/total; a single non-escape makes a point split."""
    if total <= 0 or not 0 <= escaped <= total:
        raise ValueError(f"invalid count {escaped}/{total}")
    if escaped == total:
        return "unanimous"
    return "none" if escaped == 0 else "split"


def classify_branch(report: Mapping[str, Any]) -> str:
    """Escaped, or for a non-escape, stuck versus in transition at the end of the budget."""
    if report["escape"]["escaped"]:
        return "escaped"
    usage = float(report["final"]["graph_usage_deg"])
    if usage >= GRAPH_USING_DEG:
        # Cannot happen with a permanent-crossing criterion; refuse rather than mislabel.
        raise ValueError(f"final usage {usage} at or above the threshold but not escaped")
    return "stuck" if usage < STUCK_BELOW_DEG else "in_transition"


def next_points(verdicts: Mapping[int, str]) -> tuple[int, ...]:
    """Stage 2 points selected by the plan's table; empty for a non-monotone Stage 1."""
    absent = [p for p in STAGE_ONE if p not in verdicts]
    if absent:
        raise ValueError(f"Stage 1 incomplete, missing {absent}")
    key = tuple(verdicts[p] == "unanimous" for p in STAGE_ONE)
    return STAGE_TWO.get((key[0], key[1], key[2]), ())


def _spread(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def summarise_point(
    branches: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any] | None = None,
    replay: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Primary and secondary outcomes for one branch point, plus its integrity checks."""
    classes = {str(b["branch"]): classify_branch(b) for b in branches}
    escaped = [b for b in branches if b["escape"]["escaped"]]
    everyone = [*branches, *([reference] if reference else []), *([replay] if replay else [])]
    starts = {json.dumps(b["starting_fingerprint"], sort_keys=True) for b in everyone}
    first = {json.dumps(b["first_batches_after_branch"]) for b in branches}
    summary: dict[str, Any] = {
        "branches": len(branches),
        "escaped": len(escaped),
        "verdict": verdict(len(escaped), len(branches)),
        "classes": classes,
        "escape_steps": _spread(
            [float(b["escape"]["permanent_crossing_step"]) for b in escaped]
        ),
        "final_usage_escaped": _spread(
            [float(b["final"]["graph_usage_deg"]) for b in escaped]
        ),
        "final_rotation_escaped": _spread(
            [float(b["final"]["validation_rotation_deg"]) for b in escaped]
        ),
        "distinct_starting_fingerprints": len(starts),
        "distinct_first_batches": len(first),
        "reference_reproduces": (
            bool(reference["reproduces_frozen_seed9"]["reproduces"]) if reference else None
        ),
    }
    if replay is not None:
        original = next(b for b in branches if b["branch"] == replay["branch"])
        summary["replay_identical"] = _measured(original["trajectory"]) == _measured(
            replay["trajectory"]
        )
    return summary


def _measured(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """A trajectory without its wall-clock field, which differs between identical runs."""
    return [{key: value for key, value in row.items() if key != "seconds"} for row in rows]


def stage_two_gate(summary: Mapping[str, Any]) -> list[str]:
    """Every reason Stage 2 must not start; empty only when Stage 1 is complete and valid.

    The plan's section 6 makes a failed reference invalidate everything after it, so a Stage 2
    chosen from an unchecked Stage 1 would be chosen from results nobody may read.
    """
    problems: list[str] = []
    for point in STAGE_ONE:
        entry = summary["points"].get(str(point))
        if entry is None or "verdict" not in entry:
            problems.append(f"{point}: incomplete")
            continue
        if entry["reference_reproduces"] is not True:
            problems.append(f"{point}: reference does not reproduce frozen seed 9")
        if entry["distinct_starting_fingerprints"] != 1:
            problems.append(f"{point}: branches did not all start from one state")
        if entry["distinct_first_batches"] != len(BRANCH_SEEDS):
            problems.append(f"{point}: branch orderings are not all distinct")
    first = summary["points"].get(str(STAGE_ONE[0]), {})
    if first.get("replay_identical") is not True:
        problems.append(f"{STAGE_ONE[0]}: determinism replay missing or not identical")
    changed = [name for name, counts in summary["freezes"].items() if counts["bad"]]
    if changed:
        problems.append(f"freeze manifests changed: {changed}")
    return problems


def verify_freezes(root: Path = RUNS_ROOT) -> dict[str, dict[str, int]]:
    """Every ``*FROZEN*.sha256`` manifest under ``root``, checked file by file."""
    result: dict[str, dict[str, int]] = {}
    for manifest in sorted(root.glob("*/*FROZEN*.sha256")):
        ok = bad = 0
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, name = line.split(maxsplit=1)
            path = Path(name.strip())
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            ok, bad = (ok + 1, bad) if actual == digest else (ok, bad + 1)
        result[str(manifest)] = {"ok": ok, "bad": bad}
    return result


def summarise(out_dir: Path) -> dict[str, Any]:
    """Verdict per point run so far, the monotonicity reading, and the Stage 2 selection."""
    points: dict[int, dict[str, Any]] = {}
    for folder in sorted(out_dir.glob("point_*")):
        point = int(folder.name.removeprefix("point_"))
        load = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json")}
        branches = [load[name] for name in sorted(BRANCH_SEEDS) if name in load]
        if len(branches) < len(BRANCH_SEEDS):
            points[point] = {"incomplete": sorted(load)}
            continue
        points[point] = summarise_point(
            branches, load.get("reference"), load.get("branch_00_replay")
        )

    verdicts = {p: s["verdict"] for p, s in points.items() if "verdict" in s}
    # Step 20's 805 result is part of the pattern: unanimous, 8/8.
    ordered = sorted({**verdicts, STEP20_BRANCH_POINT: "unanimous"}.items())
    monotone = all(
        not (earlier == "unanimous" and later != "unanimous")
        for (_, earlier), (_, later) in zip(ordered, ordered[1:], strict=False)
    )
    stage_one_done = all(p in verdicts for p in STAGE_ONE)
    return {
        "experiment_id": "step21-summary",
        "points": {str(p): s for p, s in sorted(points.items())},
        "pattern_including_step20_805": {str(p): v for p, v in ordered},
        "monotone": monotone,
        "stage_two": list(next_points(verdicts)) if stage_one_done else None,
        "freezes": verify_freezes(),
        "test_splits_read": [],
    }


# ----------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 21 basin entry by branching.")
    parser.add_argument("mode", choices=("capture", "run", "continue", "stage2", "summarise"))
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--points", type=int, nargs="+", default=list(STAGE_ONE))
    parser.add_argument("--replay-point", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=MAX_JOBS)
    parser.add_argument("--point", type=int, default=None)
    parser.add_argument("--branch", default=None)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if os.environ.get("OMP_NUM_THREADS") != str(THREADS) and args.mode in ("capture", "continue"):
        raise SystemExit(f"OMP_NUM_THREADS must be {THREADS}: thread count changes the numbers")

    if args.mode == "capture":
        report = capture(corpus_dir=args.corpus, out_dir=args.out)
        (args.out / "capture.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"reproduces frozen seed 9: {report['reproduces_frozen_seed9']['reproduces']}")
        print(f"805 matches Step 20: {report['step805_matches_step20']['identical']}")
        return 0 if report["valid"] else 1
    if args.mode == "continue":
        if args.point not in CAPTURE_POINTS or args.branch not in ("reference", *BRANCH_SEEDS):
            raise SystemExit("continue needs --point from CAPTURE_POINTS and a known --branch")
        target = continue_point(
            corpus_dir=args.corpus,
            out_dir=args.out,
            point=args.point,
            branch=args.branch,
            replay=args.replay,
        )
        print(f"written to {target}")
        return 0
    if args.mode == "run":
        unknown = [p for p in args.points if p not in CAPTURE_POINTS or p == STEP20_BRANCH_POINT]
        if unknown:
            raise SystemExit(f"not a Step 21 branch point: {unknown}")
        failures = run_points(
            points=args.points, replay_point=args.replay_point, out_dir=args.out, jobs=args.jobs
        )
        return 1 if failures else 0

    report = summarise(args.out)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.mode == "stage2":
        problems = stage_two_gate(report)
        for problem in problems:
            print(f"[step21] Stage 2 blocked: {problem}")
        if problems:
            return 1
        if not report["stage_two"]:
            print("[step21] non-monotone Stage 1: no Stage 2 (plan section 6, outcome C)")
            return 0
        print(f"[step21] Stage 2: {report['stage_two']}", flush=True)
        failures = run_points(
            points=report["stage_two"], replay_point=None, out_dir=args.out, jobs=args.jobs
        )
        return 1 if failures else 0
    for point, entry in report["points"].items():
        shown = entry.get("verdict", f"incomplete {entry.get('incomplete')}")
        print(f"point {point:>4}: {shown}  ({entry.get('escaped', '?')}/8)")
    print(f"monotone: {report['monotone']}   stage two: {report['stage_two']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
