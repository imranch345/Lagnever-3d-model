"""Generate the whole-organ synthetic corpus.

    python -m datasets.whole_organ.cli --scenes 1600 --families 40 \
        --out datasets/processed/whole_organ_1600
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from awr.paths import datasets_dir
from datasets.whole_organ.corpus import generate_corpus

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a whole-organ corpus."""
    parser = argparse.ArgumentParser(
        description="Generate the whole-organ corpus (SYNTHETIC_RESEARCH_DATA)."
    )
    parser.add_argument("--scenes", type=int, default=1_600)
    parser.add_argument("--families", type=int, default=40)
    parser.add_argument("--lods", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    out = args.out or datasets_dir() / "processed" / f"whole_organ_{args.scenes}"
    manifest = generate_corpus(
        out, scenes=args.scenes, families=args.families, lod_choices=args.lods
    )
    print(json.dumps(manifest.to_dict(), indent=2))
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
