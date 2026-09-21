"""Command line for generating the Step 10 Change 2 rotated corpus.

By default the corpus is **derived** from the Change 1 corpus: every stored field is carried
over — parameters, centroids, relationship graphs, point counts, presence, levels of detail,
split and family assignment — and only the frame measurement is replaced, so the two corpora
are the same 1,950 organs measured two ways. That is the tightest control available on a
change of target, and it is what `--fresh` gives up: a freshly generated corpus draws its own
adjacency point clouds, and the parent's were drawn in hash order before that was fixed, so
its relationship graphs would differ from the parent's for reasons unrelated to rotation.

Being the same organs does **not** make the two corpora's numbers comparable. The placement
distribution moved, so Change 2 computes its own placement-blind floor, and the Change 1
floor of 0.1605 stays with Change 1.

    python -m datasets.whole_organ.step10_cli --out datasets/processed/step10_rotated
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from awr.paths import repo_root
from datasets.whole_organ.continuous_corpus import (
    derive_rotated_corpus,
    generate_step8_corpus,
    split_regions,
)
from training.manifest import git_commit

__all__ = ["main"]

#: The corpus this one is measured from, scene for scene.
PARENT_CORPUS_ID = "step8-continuous-1950-40"


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Generate the Step 10 rotated corpus.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=1200)
    parser.add_argument("--validation-scenes", type=int, default=150)
    parser.add_argument("--test-scenes", type=int, default=150)
    parser.add_argument("--families", type=int, default=40)
    parser.add_argument(
        "--seed",
        type=int,
        default=20250915,
        help="the parent corpus's seed; changing it breaks the scene-for-scene pairing",
    )
    parser.add_argument("--corpus-id", default=None)
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("datasets/processed/step8_continuous"),
        help="derive from this corpus, re-measuring its frames and changing nothing else",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="generate from scratch instead of deriving; the relationship graphs will then "
        "be this generator's rather than the parent's",
    )
    args = parser.parse_args(argv)

    scenes = args.train_scenes + args.validation_scenes + 4 * args.test_scenes
    if args.fresh:
        manifest = generate_step8_corpus(
            args.out,
            train_scenes=args.train_scenes,
            validation_scenes=args.validation_scenes,
            test_scenes=args.test_scenes,
            families=args.families,
            seed=args.seed,
            rotations=True,
            corpus_id=args.corpus_id or f"step10-rotated-{scenes}-{args.families}",
            parent_corpus_id=PARENT_CORPUS_ID,
            generator_commit=git_commit(repo_root()),
        )
    else:
        manifest = derive_rotated_corpus(
            args.parent,
            args.out,
            corpus_id=args.corpus_id,
            generator_commit=git_commit(repo_root()),
            declared_parent_seed=args.seed,
        )
    print(json.dumps({"corpus_id": manifest.corpus_id, "splits": dict(manifest.split_counts)}))
    print(f"distinct relation graphs: {manifest.distinct_relation_graphs} of {manifest.scenes}")
    for split, angles in manifest.rotation_distribution["splits"].items():
        print(
            f"  {split:<18} mean {angles['mean_deg']:6.2f} deg  "
            f"median {angles['median_deg']:6.2f}  max {angles['max_deg']:7.2f}  "
            f">30 deg {angles['fraction_above_30_deg']:.2f}"
        )
    for split, regions in split_regions(args.out).items():
        print(f"  {split:<18} {regions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
