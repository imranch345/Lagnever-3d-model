"""Project-level guarantees: module boundaries, dependencies and determinism."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from awr.paths import repo_root

TIER_ONE_PACKAGES = (
    "awr",
    "reasoning",
    "editing",
    "geometry",
    "generation",
    "animation",
    "evaluation",
    "datasets",
)
"""Packages restricted to the standard library plus PyYAML. See ADR 0010."""

TIER_TWO_PACKAGES = (
    "generation/neural/nn",
    "datasets/synthetic",
    # The Step 7 whole-organ generator evaluates occupancy fields over point arrays, the
    # same kind of work as the tier-two synthetic generator, and belongs on the same side
    # of the boundary. The tier-one rule protects the representation layer, not every
    # file that happens to sit under a tier-one package.
    "datasets/whole_organ",
    "training",
    "visualization",
    "experiments",
)
"""Packages permitted to import PyTorch and NumPy. See ADR 0010."""

CORE_PACKAGES = TIER_ONE_PACKAGES

THIRD_PARTY_ALLOWED = {"yaml"}
HEAVY_DEPENDENCIES = {"torch", "numpy"}


def _is_tier_two(path: Path) -> bool:
    relative = path.relative_to(repo_root()).as_posix()
    return any(relative.startswith(f"{prefix}/") for prefix in TIER_TWO_PACKAGES)


def _python_files() -> list[Path]:
    """Tier-one source files: every package file that is not inside a tier-two package."""
    root = repo_root()
    return [
        path
        for package in TIER_ONE_PACKAGES
        for path in (root / package).rglob("*.py")
        if not _is_tier_two(path)
    ]


def _tier_two_files() -> list[Path]:
    root = repo_root()
    return [
        path
        for prefix in TIER_TWO_PACKAGES
        for path in (root / prefix).rglob("*.py")
    ]


def test_expected_packages_exist() -> None:
    """The architectural separation named in the specification is present."""
    root = repo_root()
    for package in TIER_ONE_PACKAGES:
        assert (root / package / "__init__.py").is_file(), package
    for package in TIER_TWO_PACKAGES:
        assert (root / package / "__init__.py").is_file(), package
    for folder in ("configs", "ontology/heart", "datasets/annotations", "experiments", "tests"):
        assert (root / folder).is_dir(), folder


def test_dependency_tiers_are_respected() -> None:
    """Tier one imports only the standard library and PyYAML, per ADR 0010.

    Stricter than the pre-Step-6 check it replaces: it also asserts that the heavy
    dependencies do not appear anywhere in a tier-one file, including inside a
    function body, so the boundary cannot be worked around with a lazy import.
    """
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        for heavy in HEAVY_DEPENDENCIES:
            assert f"import {heavy}" not in text, f"{path} imports {heavy}"


def test_tier_two_is_where_the_heavy_dependencies_live() -> None:
    """PyTorch is actually used by the prototype, and only there."""
    users = [
        path.name
        for path in _tier_two_files()
        if "import torch" in path.read_text(encoding="utf-8")
    ]
    assert len(users) >= 5, users


def test_core_imports_only_the_standard_library_and_pyyaml() -> None:
    """The AWR core carries no unnecessary dependencies."""
    offenders: list[str] = []
    local = {package.replace("/", ".").split(".")[0] for package in TIER_ONE_PACKAGES}
    local.update({"training", "visualization", "experiments"})
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for name in names:
                if not name or name in local or name in sys.stdlib_module_names:
                    continue
                if name in THIRD_PARTY_ALLOWED:
                    continue
                offenders.append(f"{path.name}: {name}")
    assert offenders == []


def test_only_the_config_module_reads_yaml() -> None:
    """PyYAML is confined to configuration loading."""
    users = [
        path.name
        for path in _python_files()
        if "import yaml" in path.read_text(encoding="utf-8")
    ]
    assert users == ["config.py"]


def test_no_absolute_paths_are_hard_coded() -> None:
    """Nothing in the tree points at a developer's machine."""
    root = repo_root()
    candidates = [
        *_python_files(),
        *(root / "configs").glob("*.yaml"),
        *(root / "ontology").rglob("*.json"),
    ]
    for path in candidates:
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text, path
        assert "C:\\" not in text, path


def test_awr_does_not_depend_on_the_layers_above_it() -> None:
    """The AWR core is independent of reasoning, editing, generation and geometry."""
    forbidden = {"reasoning", "editing", "generation", "geometry", "animation", "evaluation"}
    for path in (repo_root() / "awr").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = (node.module or "").split(".")[0]
            elif isinstance(node, ast.Import):
                module = node.names[0].name.split(".")[0]
            assert module not in forbidden, f"{path.name} imports {module}"


def test_public_classes_and_functions_have_docstrings() -> None:
    """Public API surface is documented."""
    missing: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert ast.get_docstring(tree), f"{path} has no module docstring"
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef | ast.FunctionDef) and not node.name.startswith("_"):
                if ast.get_docstring(node) is None:
                    missing.append(f"{path.name}:{node.name}")
    assert missing == []


def test_session_is_reproducible() -> None:
    """The same command sequence produces the same scene twice over."""
    from reasoning.command_engine import CommandEngine

    commands = (
        "generate a human heart",
        "hide everything except the chambers",
        "make the left ventricle transparent",
        "switch to medical level",
    )
    first, second = (CommandEngine() for _ in range(2))
    first.run(commands)
    second.run(commands)
    assert first.scene is not None and second.scene is not None
    assert first.scene.to_dict() == second.scene.to_dict()


@pytest.mark.parametrize("module", ["experiments.heart_session_demo"])
def test_demo_script_runs(module: str) -> None:
    """The demonstration script runs end to end."""
    result = subprocess.run(
        [sys.executable, "-m", module, "--quiet"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "persistent identity" in result.stdout
