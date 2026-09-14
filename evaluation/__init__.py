"""Evaluation layer.

Two kinds of thing live here, kept clearly apart:

* Checks that are **implemented now** because the representation exists:
  anatomical completeness of the ontology, entity-to-geometry correspondence,
  edit consistency, scene persistence and command-sequence benchmarks.
* Metrics that are **declared but not implemented** because what they measure
  does not exist yet: geometry quality, topology, multi-view consistency,
  text-to-3D compliance and educational appropriateness.

Declared metrics raise :class:`~awr.errors.NotYetImplementedError` rather than
returning a number, so no report can quietly contain a meaningless score.
"""

from evaluation.anatomy import (
    MANDATORY_HEART_ENTITIES,
    REQUIRED_FUNCTIONAL_EDGES,
    evaluate_anatomy,
)
from evaluation.baselines import (
    BASELINES,
    BaselineAdapter,
    BaselineKind,
    CapabilityMatrix,
    ComparisonAxis,
)
from evaluation.benchmarks import (
    HEART_V01_BENCHMARK,
    BenchmarkCase,
    BenchmarkReport,
    run_benchmark,
    run_benchmark_suite,
)
from evaluation.consistency import (
    evaluate_edit_consistency,
    evaluate_identity_stability,
    evaluate_persistence,
)
from evaluation.geometry import DECLARED_GEOMETRY_METRICS, evaluate_correspondence
from evaluation.neural_metrics import EXPERIMENT_METRICS, MetricSpec, primary_metrics

__all__ = [
    "BASELINES",
    "BaselineAdapter",
    "BaselineKind",
    "CapabilityMatrix",
    "ComparisonAxis",
    "DECLARED_GEOMETRY_METRICS",
    "EXPERIMENT_METRICS",
    "MetricSpec",
    "primary_metrics",
    "HEART_V01_BENCHMARK",
    "MANDATORY_HEART_ENTITIES",
    "REQUIRED_FUNCTIONAL_EDGES",
    "BenchmarkCase",
    "BenchmarkReport",
    "evaluate_anatomy",
    "evaluate_correspondence",
    "evaluate_edit_consistency",
    "evaluate_identity_stability",
    "evaluate_persistence",
    "run_benchmark",
    "run_benchmark_suite",
]
