# ADR 0004: Prototype-hub alignment anchored on the entity codebook

**Status:** PROPOSED (unvalidated)
**Date:** Step 5
**Relates to:** `generation/neural/multimodal.py`

## Context

Text, images, 3D geometry, segmentation masks and graph nodes all have to land in
the same place when they refer to the same structure. The default answer in the
literature is pairwise contrastive learning between modalities.

We are in an unusual position: the correspondence those methods spend data
discovering is already written down. Every sample can carry a persistent entity id
because the AWR exists.

## Decision

The entity identity embedding table is the hub. Every modality encoder projects
into a shared alignment space and is trained to land on the correct entity
prototype, with hard negatives drawn from the ontology: the contralateral
structure first, then siblings, then same-type entities, then spatial neighbours.
Cross-modal instance pairs remain as a secondary objective where genuinely paired
data exists.

## Alternatives considered

**Pairwise contrastive learning (image-text style).** Needs only pairing, is well
understood, and works extremely well at scale. Rejected as the primary mechanism
because it needs large paired corpora we do not have and may not be able to
licence, because it aligns appearance rather than anatomy, and because it ignores
labels we already hold. This is ablation A6. Revisit when a large licensed paired
corpus exists.

**Cross-modal transformer with masked modelling.** Rich interaction, graceful with
missing modalities. Deferred: expensive, and it gives no explicit anchor, so when
alignment fails there is nothing to inspect.

**Distillation from a pretrained vision-language model.** Cheap, inherits broad
visual knowledge. Deferred: inherits the teacher's appearance bias and its licence
constraints, and the anatomical precision of general models here is unverified.

## Consequences

* Far more sample efficient in the small-data regime the project starts in.
* The shared space has an inspectable basis: one direction per entity.
* Adding a modality is adding a spoke; existing modalities are undisturbed.
* Every training sample must carry an entity label.
* Prototypes average away within-entity variation, and they can collapse between
  anatomically similar siblings. The ontology-derived hard negatives exist for
  exactly that risk.

## How this could be wrong

If within-entity variation matters more than between-entity separation, prototypes
are the wrong abstraction and pairwise objectives will win. A6 measures it.
