"""Step 10: the spatial parent of each whole-organ entity.

Why this module exists
----------------------

Step 10 Change 1 asks for placement predicted relative to a parent, and gives
``left_ventricle -> mitral_valve`` as the example. That tree **does not exist in AWR**.
The ontology hierarchy is taxonomic: all 20 whole-organ entities are depth-2 leaves under
category nodes (chambers, valves, great_vessels, septa, wall_layers), those category nodes
carry no geometry, and no ``contains`` relation holds between any two of the 20. Composing
placement along the taxonomic tree would therefore compose against a parent that has no
frame, which is not a hierarchy in the sense Change 1 needs.

The spatial hierarchy the brief describes does exist, but in the **generator's
construction order** rather than in the ontology. ``field._build_annuli`` derives each
atrioventricular annulus from a chamber pair; ``field._build_tubes`` starts each great
artery from its own valve's annulus and each great vein from its atrium's surface, and
branches the pulmonary arteries off the pulmonary trunk. Those are real geometric
dependencies: move the parent and the child moves with it.

This module writes that dependency down explicitly, and keeps the taxonomic tree available
beside it, so an experiment can distinguish "hierarchy as AWR has it" from "hierarchy as
the geometry has it" instead of assuming the second and reporting it as the first.

The three tables
----------------

``taxonomic``
    Every entity a root. This is what the AWR hierarchy reduces to once the geometry-free
    category nodes are dropped, so it is the honest control: parent-relative placement over
    it is identical to the Step 9 global target, and any gain over it is attributable to
    hierarchy rather than to the rest of the Step 10 changes.

``spatial``
    The construction dependency, fixed, in the **non-transposed** convention. Fixed is the
    point: see the leakage note below.

``spatial_oracle``
    The construction dependency for the scene actually generated, so the two arterial
    valves follow ``transpose``. Diagnostic only, and labelled as leaking, because a
    per-scene table hands the model the arrangement.

Leakage
-------

``transpose`` is one of the arrangement axes the held-out splits generalise over. A parent
table that follows it tells the model which arrangement it is looking at, before the model
has predicted anything. That is the same class of defect as the Step 8 ground-truth
placement leak, so the default is ``spatial``, which commits to one convention and is
therefore wrong for transposed scenes in a way the model cannot exploit.

Convention for a child with two parents
---------------------------------------

An atrioventricular annulus is built from an atrium **and** a ventricle, symmetrically. No
single parent is forced by the construction, so the ventricle is chosen: it matches the
brief's example and the usual reading of an AV valve as the ventricular inlet.
``SPATIAL_ALTERNATIVE`` holds the atrium-parented variant so the choice can be tested
rather than assumed. The septa depend on their chamber pair just as symmetrically, but with
no comparable convention to appeal to, so they are left as roots rather than given an
arbitrary parent.
"""

from __future__ import annotations

from collections.abc import Mapping

from datasets.whole_organ.field import WHOLE_ORGAN_ENTITIES

__all__ = [
    "ROOT",
    "TAXONOMIC_PARENT",
    "SPATIAL_PARENT",
    "SPATIAL_ALTERNATIVE",
    "HIERARCHIES",
    "parent_of",
    "parent_slots",
    "topological_order",
    "depth_of",
    "describe",
]

#: The parent slot of an entity with no parent.
ROOT = -1


#: Construction dependencies from ``field._build_annuli`` and ``field._build_tubes``.
#: Arterial valves use the non-transposed pairing; see the leakage note in the docstring.
SPATIAL_PARENT: dict[str, str] = {
    # annuli, from the chamber pair they are built between
    "heart.mitral_valve": "heart.left_ventricle",
    "heart.tricuspid_valve": "heart.right_ventricle",
    "heart.aortic_valve": "heart.left_ventricle",
    "heart.pulmonary_valve": "heart.right_ventricle",
    # great arteries, started from their own valve's annulus
    "heart.aorta": "heart.aortic_valve",
    "heart.pulmonary_trunk": "heart.pulmonary_valve",
    "heart.pulmonary_arteries": "heart.pulmonary_trunk",
    # great veins, started from their atrium's surface
    "heart.superior_vena_cava": "heart.right_atrium",
    "heart.inferior_vena_cava": "heart.right_atrium",
    "heart.pulmonary_veins": "heart.left_atrium",
}

#: The AV annuli parented to the atrium instead, to test the convention above.
SPATIAL_ALTERNATIVE: dict[str, str] = {
    **SPATIAL_PARENT,
    "heart.mitral_valve": "heart.left_atrium",
    "heart.tricuspid_valve": "heart.right_atrium",
}

#: AWR's hierarchy once the geometry-free category nodes are dropped: no parents at all.
TAXONOMIC_PARENT: dict[str, str] = {}

HIERARCHIES: dict[str, dict[str, str]] = {
    "taxonomic": TAXONOMIC_PARENT,
    "spatial": SPATIAL_PARENT,
    "spatial_alternative": SPATIAL_ALTERNATIVE,
}


def parent_of(entity_id: str, hierarchy: str = "spatial") -> str | None:
    """The parent entity id, or ``None`` when the entity is a root."""
    return HIERARCHIES[hierarchy].get(entity_id)


def _transposed(table: dict[str, str]) -> dict[str, str]:
    swapped = dict(table)
    swapped["heart.aortic_valve"] = "heart.right_ventricle"
    swapped["heart.pulmonary_valve"] = "heart.left_ventricle"
    return swapped


def parent_slots(
    hierarchy: str = "spatial",
    slot_of: Mapping[str, int] | None = None,
    *,
    slots: int | None = None,
    transpose: bool = False,
) -> tuple[int, ...]:
    """Parent slot per entity slot, ``ROOT`` for a root and for every unrelated slot.

    ``slot_of`` must be the **batch builder's** mapping, which is keyed on the full
    ontology (``ontology.ids()``, 64 slots), not on the 20 whole-organ entities. Those two
    orderings differ, and indexing a 64-slot batch with a position in
    ``WHOLE_ORGAN_ENTITIES`` silently addresses a different entity. Passing ``None`` uses
    the whole-organ ordering, which is correct only for tables built in that same ordering.

    ``transpose`` selects the oracle table for a transposed scene. It leaks the arrangement
    and exists for the diagnostic arm only; the default is the fixed convention.
    """
    table = HIERARCHIES[hierarchy]
    if transpose:
        table = _transposed(table)
    if slot_of is None:
        slot_of = {entity_id: index for index, entity_id in enumerate(WHOLE_ORGAN_ENTITIES)}
    width = len(slot_of) if slots is None else slots
    parents = [ROOT] * width
    for entity_id, parent_id in table.items():
        child_slot = slot_of.get(entity_id)
        parent_slot = slot_of.get(parent_id)
        if child_slot is None or parent_slot is None:
            raise KeyError(
                f"{entity_id!r} or its parent {parent_id!r} is missing from the slot map; "
                "the hierarchy would silently address the wrong entity."
            )
        parents[child_slot] = parent_slot
    return tuple(parents)


def topological_order(parents: tuple[int, ...]) -> tuple[int, ...]:
    """Slot order with every parent before its children.

    Raises ``ValueError`` on a cycle, which would otherwise show up as a composition that
    silently used a stale parent transform.
    """
    order: list[int] = []
    placed = set()
    remaining = set(range(len(parents)))
    while remaining:
        ready = sorted(
            slot
            for slot in remaining
            if parents[slot] == ROOT or parents[slot] in placed
        )
        if not ready:
            raise ValueError(f"cycle in parent table among slots {sorted(remaining)}")
        for slot in ready:
            order.append(slot)
            placed.add(slot)
            remaining.discard(slot)
    return tuple(order)


def depth_of(parents: tuple[int, ...]) -> tuple[int, ...]:
    """Depth per slot, roots at 0."""
    depths = [0] * len(parents)
    for slot in topological_order(parents):
        parent = parents[slot]
        depths[slot] = 0 if parent == ROOT else depths[parent] + 1
    return tuple(depths)


def describe(
    hierarchy: str = "spatial", slot_of: Mapping[str, int] | None = None
) -> dict[str, object]:
    """Shape of a hierarchy, for the manifest and the depth analysis."""
    parents = parent_slots(hierarchy, slot_of)
    depths = depth_of(parents)
    histogram: dict[int, int] = {}
    for depth in depths:
        histogram[depth] = histogram.get(depth, 0) + 1
    return {
        "hierarchy": hierarchy,
        "slots": len(parents),
        "entities": len(WHOLE_ORGAN_ENTITIES),
        "parented": sum(1 for parent in parents if parent != ROOT),
        "max_depth": max(depths),
        "depth_histogram": {str(key): histogram[key] for key in sorted(histogram)},
    }
