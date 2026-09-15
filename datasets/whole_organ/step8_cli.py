"""Command line for generating a Step 8 continuous-arrangement corpus."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from datasets.whole_organ.continuous_corpus import generate_step8_corpus, split_regions

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Generate a Step 8 corpus.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=1200)
    parser.add_argument("--validation-scenes", type=int, default=150)
    parser.add_argument("--test-scenes", type=int, default=150)
    parser.add_argument("--families", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20250915)
    args = parser.parse_args(argv)

    manifest = generate_step8_corpus(
        args.out,
        train_scenes=args.train_scenes,
        validation_scenes=args.validation_scenes,
        test_scenes=args.test_scenes,
        families=args.families,
        seed=args.seed,
    )
    print(json.dumps({"splits": dict(manifest.split_counts)}, indent=2))
    print(f"distinct relation graphs: {manifest.distinct_relation_graphs} of {manifest.scenes}")
    for split, regions in split_regions(args.out).items():
        print(f"  {split:20} {regions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
