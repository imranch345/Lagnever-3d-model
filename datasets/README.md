# Datasets

No dataset is bundled with this repository and nothing here downloads data.

## Rules

1. **Licence first.** An asset may only enter `datasets/raw/` after its licence and
   source have been recorded in a `Provenance` block. `datasets/schema.py` refuses
   to construct a `Provenance` without both.
2. **No scraping.** Anatomical atlases, scan archives and model libraries are not
   collected opportunistically. Each source is reviewed for terms of use first,
   and the review is recorded in the sample metadata.
3. **Anatomy is anchored to the ontology.** A sample's parts are ontology entity
   ids, never free text, so a mesh part and `heart.left_ventricle` can be checked
   against each other.
4. **Binary assets are never committed.** `.gitignore` excludes `datasets/raw/`
   and `datasets/processed/` along with common mesh and volume formats.

## Layout

| Path                    | Contents                                                     |
| ----------------------- | ------------------------------------------------------------ |
| `datasets/raw/`         | Licensed source assets. Empty, git-ignored.                   |
| `datasets/processed/`   | Derived tensors and normalised assets. Empty, git-ignored.    |
| `datasets/annotations/` | Sample metadata as JSON. Committed. Currently one synthetic example. |

## Current state

`datasets/annotations/heart_sample_0001.json` is **synthetic metadata**. It has no
image, no mesh and no measurement behind it. It exists so the schema has a worked
example and so validation has something to run against.

Acquisition and licensing of real data is a separate R&D step.
