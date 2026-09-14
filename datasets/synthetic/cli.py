"""Command line entry point for tier-0 corpus generation.

python -m datasets.synthetic.cli --scenes 2000 --out datasets/processed/heart_tier0_2k
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from awr.config import load_domain_config
from awr.ontology import load_ontology
from awr.paths import datasets_dir
from datasets.synthetic.store import generate_corpus

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a tier-0 synthetic heart corpus."""
    parser = argparse.ArgumentParser(
        description="Generate the tier-0 synthetic heart corpus (SYNTHETIC_RESEARCH_DATA)."
    )
    parser.add_argument("--scenes", type=int, default=2_000)
    parser.add_argument("--families", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lods", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    config = load_domain_config()
    ontology = load_ontology(
        config.domain.ontology_dir, expected_version=config.domain.ontology_version
    )
    out = args.out or datasets_dir() / "processed" / f"heart_tier0_{args.scenes}"
    manifest = generate_corpus(
        ontology,
        out,
        scenes=args.scenes,
        families=args.families,
        lod_choices=args.lods,
        seed=args.seed,
    )
    print(json.dumps(manifest.to_dict(), indent=2))
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
