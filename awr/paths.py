"""Repository path resolution.

No absolute path is ever hard-coded. Every data file is located relative to the
repository root, which is discovered from this module's own location (or
overridden with the ``LAGNAV_ROOT`` environment variable, which is useful when
the package is installed outside a checkout).
"""

from __future__ import annotations

import os
from pathlib import Path

from awr.errors import ConfigError

__all__ = ["repo_root", "config_dir", "ontology_dir", "datasets_dir", "resolve_repo_path"]

_ROOT_ENV_VAR = "LAGNAV_ROOT"
_ROOT_MARKERS = ("pyproject.toml", "configs", "ontology")


def repo_root() -> Path:
    """Return the repository root directory.

    Resolution order: ``LAGNAV_ROOT`` environment variable, then the nearest
    ancestor of this file that contains the expected project markers.
    """
    override = os.environ.get(_ROOT_ENV_VAR)
    if override:
        candidate = Path(override).expanduser().resolve()
        if not candidate.is_dir():
            raise ConfigError(f"{_ROOT_ENV_VAR}={override!r} is not a directory.")
        return candidate

    here = Path(__file__).resolve()
    for parent in here.parents:
        if all((parent / marker).exists() for marker in _ROOT_MARKERS):
            return parent
    raise ConfigError(
        "Could not locate the Lagnav repository root. Expected an ancestor directory "
        f"containing {', '.join(_ROOT_MARKERS)}, or the {_ROOT_ENV_VAR} environment variable."
    )


def resolve_repo_path(relative: str | os.PathLike[str]) -> Path:
    """Resolve a repository-relative path to an absolute path."""
    path = Path(relative)
    return path if path.is_absolute() else repo_root() / path


def config_dir() -> Path:
    """Directory holding the YAML configuration files."""
    return repo_root() / "configs"


def ontology_dir(domain: str = "heart") -> Path:
    """Directory holding the versioned ontology data for one domain."""
    return repo_root() / "ontology" / domain


def datasets_dir() -> Path:
    """Root directory for dataset metadata and (future) assets."""
    return repo_root() / "datasets"
