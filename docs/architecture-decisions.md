# Architecture decisions

Decisions taken while building the Step 4 foundation, with the reasoning behind
them. Each one is reversible; each one is testable.

## AD-1 Entity identity is opaque, not a tree path

`heart.left_ventricle` keeps its id even though it sits under
`heart.chambers` in the tree. Ids look namespaced for readability only and are
never derived from the hierarchy.

*Why:* re-parenting an entity would otherwise change its identity, which breaks
the core promise that an edit can never change an entity id. It also lets the
presentation tree be reorganised without invalidating stored scenes, annotations
or geometry correspondence.

## AD-2 Identity and state are separate objects

`AnatomicalEntity` is frozen and holds identity and semantics.
`SceneEntity.state` is mutable and holds visibility, opacity, material,
transform, animation and geometry reference. All the fields the specification
lists are reachable from a `SceneEntity`.

*Why:* editing code physically cannot write to identity. The invariant is
enforced by the type system rather than by review.

## AD-3 Organisational hierarchy is not anatomical partonomy

`hierarchy.json` is a presentation tree (how a learner navigates). Anatomical
containment that differs from it is declared separately as `part_of` edges, for
example `fossa_ovalis part_of interatrial_septum` while the fossa is filed under
internal structures.

*Why:* conflating the two forces a single tree to serve navigation and anatomy at
once, and one of the two always loses. The structure graph derives `part_of` from
the tree automatically, so the two can never drift apart.

## AD-4 Groups exist and are never renderable

`heart.chambers`, `heart.valves`, `heart.great_vessels`, `heart.septa`,
`heart.wall_layers`, `heart.internal_structures` are organisational entities with
`renderable=False`. Commands that name a group expand to its renderable
descendants.

*Why:* "show the valves" must act on four valves. Making the group itself
drawable would introduce an object that has state but no anatomy.

## AD-5 LOD controls visibility, never existence

Every ontology entity is instantiated at every LOD. `min_lod`/`max_lod` decide
default visibility only.

*Why:* this is the specification's core principle, that the internal
representation is richer than the visible output. It also means switching from
school to medical level reveals structures rather than rebuilding the scene.

## AD-6 A LOD change recomputes visibility and clears manual overrides, but never touches material state

`SetLod` reapplies the LOD policy to every renderable entity. Opacity, material,
transform and animation survive untouched.

*Why:* the alternative, honouring manual visibility indefinitely, makes "show
more detail" unpredictable after a few edits. Keeping opacity means "make the
left ventricle transparent" survives a level change, which is what a user
expects. `VisibilitySource` records which rule set each value came from, and a
test pins the behaviour.

## AD-7 Relation types are declared in code, edges in data

The vocabulary (graph, symmetry, inverse, flow direction, medium) lives in
`awr/relationships.py`; `relationships.json` holds only edges, validated against
the registry at load time.

*Why:* strong typing for the vocabulary, easy authoring for the data, and an
unregistered relation name becomes a load error instead of a string that silently
means nothing.

## AD-8 Symmetric and inverse relations are stored once and resolved at query time

`connects_to` is declared in one direction; `superior_to` has `inferior_to` as a
declared inverse. Queries resolve both.

*Why:* storing both directions doubles the data and invites contradictions where
one direction is edited and the other is not.

## AD-9 Functional edges carry a medium and a granularity

Relation types declare whether they carry `blood` or an electrical `impulse`.
Edges declare `summary` (chamber-level) or `detailed` (through valves).

*Why:* blood flow and conduction share the functional graph but must never be
traversed as one path, and parallel chamber-level and valve-level routes would be
double-counted. Traversal code filters on declared properties rather than
matching relation names.

## AD-10 No geometry is generated; component slots are reserved instead

`SymbolicGeometryDecoder` allocates deterministic component ids with no payload.

*Why:* a random or placeholder mesh would make visual output look like progress
while being anatomically meaningless, and would invite the rest of the system to
depend on it. Reserving slots makes the correspondence real, persisted and
testable today.

## AD-11 Unimplemented research components raise rather than return

`NotYetImplementedError` is raised by every declared-but-undesigned component:
neural interfaces, geometry representations, future editing operations, geometry
metrics.

*Why:* a stub that returns a plausible value is worse than no stub. A raise makes
accidental dependence impossible and keeps the roadmap in typed code.

## AD-12 Ambiguity is never resolved by guessing

The resolver raises with candidates for "ventricle" or "septum", and refuses
containment matching for short abbreviation-shaped tokens, so "AV" does not
silently become the atrioventricular node when the aortic valve may be meant.

*Why:* in a medical context a confident wrong answer is the worst failure mode.

## AD-13 The parser resolves nothing

`ParsedCommand` carries unresolved phrases. Resolution happens in the resolver;
scene knowledge lives in the engine.

*Why:* it is the seam a language model will be dropped into. If the parser knew
about entities or the scene, swapping it would ripple through the system.

## AD-14 Every state change is an operation, and history is committed in one place

`SceneEditor.apply` is the only path that writes history. A no-op does not bump
the scene version.

*Why:* history becomes a complete, auditable log rather than a partial one, and
edit-consistency evaluation can compare what an operation declared against what
actually changed.

## AD-15 Determinism by construction

Scene ids are content-derived, component ids follow ontology order, no timestamps
are written into scenes, and iteration order follows declaration order.

*Why:* the same session must produce byte-identical output across runs, which a
test enforces. Reproducibility now is what will make the neural stack's effects
measurable later.

## AD-16 Configuration holds the numbers, the ontology holds the anatomy, code holds neither

"Transparent" means 0.3 because `configs/heart.yaml` says so; "medical level"
means LOD 4 for the same reason; cardiac phase fractions likewise.

*Why:* tuning must not require code changes, and anatomy must not live in Python.

## AD-17 One runtime dependency

The core imports only the standard library. PyYAML is confined to
`awr/config.py`. A test enforces both rules.

*Why:* an R&D foundation that is hard to install is hard to iterate on, and the
heavy dependencies belong to the step that introduces models.

## Additions to the specified structure

Four modules were added to the layout in the specification, each for a stated
reason:

| Addition                   | Reason                                                                 |
| -------------------------- | ---------------------------------------------------------------------- |
| `awr/errors.py`            | A typed exception hierarchy is needed for "useful errors, never silent corruption". |
| `awr/paths.py`, `awr/config.py` | Repository-relative path resolution and typed config loading, keeping absolute paths and magic numbers out of code. |
| `awr/ontology.py`, `awr/lod.py` | The ontology data needs a strict typed loader, and the LOD policy needs a home outside the scene. |
| `reasoning/explanation.py` | "Explain what is happening" needs a graph verbaliser; keeping it separate leaves room for a learned explainer. |
| `generation/neural_interfaces.py` | A single place for the 3D latent, material and animation interfaces Step 5 will define. |
