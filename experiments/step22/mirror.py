"""Step 22: is seed 5's non-escape fixed from initialisation, and is it weights or order.

Preregistration: ``docs/STEP_22_MIRROR_PLAN.md`` (with its clarifications before any code).

* **Capture.** Seed 5 is trained to step 800, saving continuable states at 0, 400 and 800.
  Its trajectory must reproduce frozen seed 5 (Step 16) exactly at every shared checkpoint.
* **Arm 1.** From seed 5's step 0: a reference (original stream), eight branches with loader
  seeds 2000-2007 (Step 21's, so ``branch_k`` is the same order as from seed 9's step 0), and a
  replay of ``branch_00``.
* **Arm 2.** From step 0, the crossed swap and its identity controls. ``NwMo`` means seed N's
  weights and global RNG, with seed M's batch order (loader seed M).
* **Stage 2,** only if Arm 1 is not "none": seed 5 from 400 and 800, reference and 8 branches.

Continuation is Step 20's ``continue_run``, with the run seed of the weights it continues.
Verdicts for a non-escaper are none / split / all (plan section 4).

    OMP_NUM_THREADS=4 python -m experiments.step22.mirror capture
    python -m experiments.step22.mirror run --jobs 3
    python -m experiments.step22.mirror stage2 --jobs 3
    python -m experiments.step22.mirror summarise
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.step20.branching import BRANCH_SEEDS, _build, continue_run
from experiments.step21.basin_entry import (
    MAX_JOBS,
    MIN_FREE_BYTES,
    RUNS_ROOT,
    THREADS,
    _measured,
    _spread,
    capture_states,
    classify_branch,
    reproduction,
    state_path,
    verify_freezes,
)

__all__ = [
    "ARM_TWO",
    "CAPTURE_POINTS",
    "STAGE_TWO_POINTS",
    "Run",
    "arm_one",
    "integrity",
    "plan_runs",
    "stage_two_points",
    "summarise",
    "verdict",
]

SEED = 5
SEED_NINE = 9
CAPTURE_POINTS: tuple[int, ...] = (0, 400, 800)
STAGE_TWO_POINTS: tuple[int, ...] = (400, 800)
OUT = Path("experiments/runs/step22")
CORPUS = Path("datasets/processed/step10_rotated")
FROZEN_SEED5 = Path("experiments/runs/step16/trajectory_seed5.json")
STEP21_OUT = Path("experiments/runs/step21")
#: Step 21's reference from seed 9's step 0: what ``ident_9w_9o`` must equal.
STEP21_SEED9_ZERO = STEP21_OUT / "point_0000" / "reference.json"


class Run(dict[str, Any]):
    """One continuation the plan asks for: name, point, weights' seed, loader seed, replay."""


def _run(name: str, point: int, weights: int, loader: int | None, replay: bool = False) -> Run:
    return Run(name=name, point=point, weights=weights, loader=loader, replay=replay)


#: Arm 2, from step 0 (plan section 3). Weights' seed, then the batch order's seed.
ARM_TWO: dict[str, tuple[int, int]] = {
    "cross_5w_9o": (SEED, SEED_NINE),
    "cross_9w_5o": (SEED_NINE, SEED),
    "ident_5w_5o": (SEED, SEED),
    "ident_9w_9o": (SEED_NINE, SEED_NINE),
}


def plan_runs(points: Sequence[int] = (0,), *, arm_two: bool = True) -> list[Run]:
    """Every continuation for these seed-5 points, references first; Arm 2 only at step 0."""
    runs: list[Run] = []
    for point in points:
        runs.append(_run("reference", point, SEED, None))
        runs.extend(_run(name, point, SEED, BRANCH_SEEDS[name]) for name in sorted(BRANCH_SEEDS))
        if point == 0:
            runs.append(_run("branch_00", 0, SEED, BRANCH_SEEDS["branch_00"], replay=True))
    if arm_two and 0 in points:
        runs.extend(_run(name, 0, w, o) for name, (w, o) in ARM_TWO.items())
    return runs


def _file_name(run: Mapping[str, Any]) -> str:
    return f"{run['name']}_replay" if run["replay"] else str(run["name"])


def point_dir(out_dir: Path, point: int) -> Path:
    """Where continuations from a seed-5 point write their results."""
    return out_dir / f"point_{point:04d}"


def _state_for(run: Mapping[str, Any], out_dir: Path) -> Path:
    """Seed 5's states are this step's capture; seed 9's step 0 is Step 21's."""
    if run["weights"] == SEED_NINE:
        return state_path(STEP21_OUT, 0, SEED_NINE)
    return state_path(out_dir, int(run["point"]), SEED)


# ----------------------------------------------------------------------
# Capture and continuation
# ----------------------------------------------------------------------
def capture(*, corpus_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Train seed 5 to 800, saving 0, 400 and 800, and check it against frozen seed 5."""
    trainer, recorder, _positions = _build(corpus_dir, SEED)
    print(f"[step22] capturing seed {SEED} at {list(CAPTURE_POINTS)}", flush=True)
    marks = capture_states(
        trainer, points=CAPTURE_POINTS, out_dir=out_dir, recorder=recorder, seed=SEED
    )
    check = reproduction(recorder.rows, FROZEN_SEED5)
    nine = json.loads((STEP21_OUT / "capture.json").read_text(encoding="utf-8"))
    differs = marks[0] != nine["fingerprints"]["0"]
    return {
        "experiment_id": "step22-capture",
        "seed": SEED,
        "capture_points": list(CAPTURE_POINTS),
        "states": {str(p): str(state_path(out_dir, p, SEED)) for p in CAPTURE_POINTS},
        "fingerprints": {str(p): marks[p] for p in CAPTURE_POINTS},
        "reproduces_frozen_seed5": check,
        "step0_differs_from_seed9": differs,
        "valid": check["reproduces"] and differs,
        "trajectory": recorder.rows,
        "test_splits_read": [],
    }


def continue_one(*, corpus_dir: Path, out_dir: Path, run: Mapping[str, Any]) -> Path:
    """One continuation, built with its weights' seed; returns its result file."""
    report = continue_run(
        corpus_dir=corpus_dir,
        state_path=_state_for(run, out_dir),
        out_dir=out_dir,
        branch=str(run["name"]),
        loader_seed=run["loader"],
        seed=int(run["weights"]),
    )
    report.update(
        experiment_id="step22-continuation",
        step22_point=run["point"],
        weights_seed=run["weights"],
        loader_seed=run["loader"],
        replay=run["replay"],
    )
    if run["name"] == "reference":
        report["reproduces_frozen_seed5"] = reproduction(
            report["trajectory"], FROZEN_SEED5, start=int(run["point"])
        )
    target = point_dir(out_dir, int(run["point"])) / f"{_file_name(run)}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return target


def run_all(*, runs: Sequence[Run], out_dir: Path, jobs: int) -> int:
    """Subprocesses at most ``jobs`` at once, completed results kept; returns failures."""
    if jobs < 1 or jobs > MAX_JOBS:
        raise ValueError(f"jobs must be between 1 and {MAX_JOBS}, got {jobs}")
    pending = []
    for run in runs:
        if not _state_for(run, out_dir).exists():
            raise SystemExit(f"no captured state for {_file_name(run)}; run 'capture' first")
        if (point_dir(out_dir, int(run["point"])) / f"{_file_name(run)}.json").exists():
            print(f"[step22] skip {run['point']}/{_file_name(run)}: result exists", flush=True)
            continue
        pending.append(run)
    env = {**os.environ, "OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS)}
    running: list[tuple[subprocess.Popen[bytes], str, Any]] = []
    failures = 0
    while pending or running:
        while pending and len(running) < jobs:
            if shutil.disk_usage(out_dir).free < MIN_FREE_BYTES:
                raise SystemExit("free disk below 2 GiB; stopping without deleting anything")
            run = pending.pop(0)
            label = f"{run['point']}/{_file_name(run)}"
            log_path = point_dir(out_dir, int(run["point"])) / f"{_file_name(run)}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("wb")
            command = [
                sys.executable,
                "-m",
                "experiments.step22.mirror",
                "continue",
                "--run",
                json.dumps(dict(run)),
                "--out",
                str(out_dir),
            ]
            print(f"[step22] start {label}", flush=True)
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((process, label, log))
        time.sleep(5)
        for entry in list(running):
            process, label, log = entry
            if process.poll() is None:
                continue
            log.close()
            running.remove(entry)
            failures += process.returncode != 0
            status = "done" if process.returncode == 0 else f"FAILED ({process.returncode})"
            print(f"[step22] {label}: {status}", flush=True)
    return failures


# ----------------------------------------------------------------------
# Verdicts and summary
# ----------------------------------------------------------------------
def verdict(escaped: int, total: int) -> str:
    """Plan section 4, for a non-escaper: none at 0, all at total, otherwise split."""
    if total <= 0 or not 0 <= escaped <= total:
        raise ValueError(f"invalid count {escaped}/{total}")
    if escaped == 0:
        return "none"
    return "all" if escaped == total else "split"


def stage_two_points(arm_one_verdict: str) -> tuple[int, ...]:
    """Section 3: Stage 2 runs at 400 and 800 unless Arm 1 is none; C' (all) stops too."""
    return STAGE_TWO_POINTS if arm_one_verdict == "split" else ()


def arm_one(branches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Primary and secondary outcomes over seed 5's eight branches from one point."""
    escaped = [b for b in branches if b["escape"]["escaped"]]
    return {
        "branches": len(branches),
        "escaped": len(escaped),
        "verdict": verdict(len(escaped), len(branches)),
        "classes": {str(b["branch"]): classify_branch(b) for b in branches},
        "escape_steps": _spread([float(b["escape"]["permanent_crossing_step"]) for b in escaped]),
        "final_usage": _spread([float(b["final"]["graph_usage_deg"]) for b in branches]),
        "final_rotation": _spread([float(b["final"]["validation_rotation_deg"]) for b in branches]),
    }


def _same(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return _measured(a["trajectory"]) == _measured(b["trajectory"])


def integrity(
    *,
    capture_report: Mapping[str, Any] | None,
    reference: Mapping[str, Any] | None,
    branches: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any] | None,
    swaps: Mapping[str, Mapping[str, Any]],
    seed9_zero: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Section 7's checks that a summary can make; any False makes the step invalid."""
    starts = {json.dumps(b["starting_fingerprint"], sort_keys=True) for b in [*branches]}
    if reference is not None:
        starts.add(json.dumps(reference["starting_fingerprint"], sort_keys=True))
    checks: dict[str, Any] = {
        "capture_reproduces_frozen_seed5": bool(
            capture_report and capture_report["reproduces_frozen_seed5"]["reproduces"]
        ),
        "seed5_step0_differs_from_seed9": bool(
            capture_report and capture_report["step0_differs_from_seed9"]
        ),
        "reference_reproduces_frozen_seed5": bool(
            reference and reference["reproduces_frozen_seed5"]["reproduces"]
        ),
        "one_starting_state": len(starts) == 1,
        "distinct_first_batches": len(
            {json.dumps(b["first_batches_after_branch"]) for b in branches}
        )
        == len(branches),
        "replay_identical": bool(
            replay is not None
            and _same(replay, next(b for b in branches if b["branch"] == "branch_00"))
        ),
    }
    if "ident_5w_5o" in swaps and reference is not None:
        checks["ident_5w_5o_equals_reference"] = _same(swaps["ident_5w_5o"], reference)
    if "ident_9w_9o" in swaps and seed9_zero is not None:
        checks["ident_9w_9o_equals_step21_reference"] = _same(swaps["ident_9w_9o"], seed9_zero)
    checks["valid"] = all(v for k, v in checks.items())
    return checks


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def summarise(out_dir: Path) -> dict[str, Any]:
    """Arm 1, Arm 2, Stage 2 if run, integrity and the decision's inputs."""
    zero = point_dir(out_dir, 0)
    branches = [b for name in sorted(BRANCH_SEEDS) if (b := _load(zero / f"{name}.json"))]
    reference = _load(zero / "reference.json")
    swaps = {name: s for name in ARM_TWO if (s := _load(zero / f"{name}.json"))}
    checks = integrity(
        capture_report=_load(out_dir / "capture.json"),
        reference=reference,
        branches=branches,
        replay=_load(zero / "branch_00_replay.json"),
        swaps=swaps,
        seed9_zero=_load(STEP21_SEED9_ZERO),
    )
    complete = len(branches) == len(BRANCH_SEEDS)
    primary = arm_one(branches) if complete else {"incomplete": len(branches)}
    stage_two: dict[str, Any] = {}
    for point in STAGE_TWO_POINTS:
        folder = point_dir(out_dir, point)
        found = [b for name in sorted(BRANCH_SEEDS) if (b := _load(folder / f"{name}.json"))]
        if found:
            ref = _load(folder / "reference.json")
            outcome = (
                arm_one(found) if len(found) == len(BRANCH_SEEDS) else {"incomplete": len(found)}
            )
            stage_two[str(point)] = {
                **outcome,
                "reference_reproduces": bool(ref and ref["reproduces_frozen_seed5"]["reproduces"]),
            }
    return {
        "experiment_id": "step22-summary",
        "arm_one_step0": primary,
        "arm_two": {
            name: {
                "weights_seed": s["weights_seed"],
                "loader_seed": s["loader_seed"],
                "escaped": s["escape"]["escaped"],
                "escape_step": s["escape"].get("permanent_crossing_step"),
                "class": classify_branch(s),
                "final_usage_deg": s["final"]["graph_usage_deg"],
                "final_rotation_deg": s["final"]["validation_rotation_deg"],
            }
            for name, s in swaps.items()
        },
        "stage_two_points": list(stage_two_points(str(primary["verdict"]))) if complete else None,
        "stage_two": stage_two,
        "integrity": checks,
        "freezes": verify_freezes(RUNS_ROOT),
        "test_splits_read": [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 22: seed 5 at initialisation, and a swap.")
    parser.add_argument("mode", choices=("capture", "run", "continue", "stage2", "summarise"))
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--jobs", type=int, default=MAX_JOBS)
    parser.add_argument("--run", default=None, help="one run, as JSON (used by 'run')")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if os.environ.get("OMP_NUM_THREADS") != str(THREADS) and args.mode in ("capture", "continue"):
        raise SystemExit(f"OMP_NUM_THREADS must be {THREADS}: thread count changes the numbers")
    if args.mode == "capture":
        report = capture(corpus_dir=args.corpus, out_dir=args.out)
        (args.out / "capture.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"reproduces frozen seed 5: {report['reproduces_frozen_seed5']['reproduces']}")
        print(f"step 0 differs from seed 9: {report['step0_differs_from_seed9']}")
        return 0 if report["valid"] else 1
    if args.mode == "continue":
        run = Run(json.loads(args.run))
        known = {_file_name(r): r for r in plan_runs((0, *STAGE_TWO_POINTS))}
        if _file_name(run) not in known or known[_file_name(run)] != run:
            raise SystemExit(f"not a run the plan declares: {args.run}")
        print(f"written to {continue_one(corpus_dir=args.corpus, out_dir=args.out, run=run)}")
        return 0
    if args.mode == "run":
        return 1 if run_all(runs=plan_runs((0,)), out_dir=args.out, jobs=args.jobs) else 0

    report = summarise(args.out)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.mode == "stage2":
        if not report["integrity"]["valid"]:
            print(f"[step22] invalid, no Stage 2: {report['integrity']}", flush=True)
            return 1
        points = report["stage_two_points"]
        if not points:
            print(f"[step22] Arm 1 is {report['arm_one_step0'].get('verdict')}: no Stage 2")
            return 0
        print(f"[step22] Stage 2: {points}", flush=True)
        stage = plan_runs(points, arm_two=False)
        return 1 if run_all(runs=stage, out_dir=args.out, jobs=args.jobs) else 0
    print(json.dumps({k: report[k] for k in ("arm_one_step0", "arm_two", "integrity")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
