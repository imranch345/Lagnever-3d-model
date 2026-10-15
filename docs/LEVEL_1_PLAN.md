# Level 1 — a language-controlled 3D heart and lungs built from real CT

**Status:** plan. Written 2026-10-15, before any Level 1 code.
**Track:** product. This is separate from the research steps, which continue to their own end
and do not block this plan.
**Not medical.** Shapes come from public research CT. Nothing here is validated for clinical
use, and the viewer will say so.

---

## 1. Goal and definition of done

A person types a request in plain words and a 3D heart and lungs, built from real CT, respond:

> "Show the heart and lungs" → "Hide the lungs" → "Make the left ventricle transparent" →
> "Show only the chambers" → "Explain the right atrium"

Level 1 is done when all of the following hold:

1. **Real geometry.** Each visible structure is a mesh derived from a real CT segmentation, with
   its source and license recorded.
2. **Identity is permanent.** Every mesh belongs to exactly one AWR entity id
   (`heart.left_ventricle`, `respiratory.lung_upper_lobe_left`, …). No edit changes that link.
3. **Edits are edits.** Show, hide, isolate, opacity, level of detail and explanation change
   entity state. They never regenerate the scene, and untouched entities do not change.
4. **It runs locally.** A browser page draws the scene, and a local server applies commands
   through the existing `CommandEngine`.
5. **Structures with no mesh say so.** Valves, conduction and coronaries are not in the data.
   They remain in the AWR and in explanations, and the viewer marks them as "no geometry yet",
   never as a fake shape.
6. **Tests cover it.** That includes the mesh pipeline on synthetic volumes, the label-to-entity
   map, and a scripted end-to-end session.

## 2. What exists, and what is missing

**Exists** (Steps 4–6): the AWR with permanent entity ids, three typed graphs and scene memory;
Heart Ontology v0.1 (42 entities); a deterministic `CommandEngine` that already handles show,
hide, isolate, opacity, LOD, explain and reset; reserved geometry component slots
(`geometry/correspondence.py`); and OBJ, PLY and PNG exporters for research views.

**Missing:**

| piece | status |
| --- | --- |
| geometry from real anatomy | none: `SymbolicGeometry` holds slots with no payload |
| lungs | not in any ontology; the README puts the rest of the body out of scope |
| a viewer | none |
| free-text language | rule parser only |

## 3. Data

**Primary: TotalSegmentator dataset v1.0** ([Zenodo 6802614](https://zenodo.org/records/6802614)).

* 1204 CT scans, 104 labelled structures, **CC BY 4.0**, so attribution is required.
* One 28.4 GB zip. Zenodo serves byte ranges (checked: a 100-byte request returned
  `206 PARTIAL_CONTENT`). So the zip's index plus only the needed label files can be fetched
  without downloading the whole archive. **Estimated download: 100–150 MB for 20 subjects.**
  It needs the user's permission first.

The labels Level 1 uses, from the v1 `total` class map:

| TotalSegmentator label | id | AWR entity | note |
| --- | ---: | --- | --- |
| `heart_myocardium` | 44 | `heart.myocardium` | |
| `heart_atrium_left` | 45 | `heart.left_atrium` | |
| `heart_ventricle_left` | 46 | `heart.left_ventricle` | |
| `heart_atrium_right` | 47 | `heart.right_atrium` | |
| `heart_ventricle_right` | 48 | `heart.right_ventricle` | |
| `pulmonary_artery` | 49 | `heart.pulmonary_trunk` | label covers trunk and main branches |
| `aorta` | 7 | `heart.aorta` | whole aorta; cropped to the thorax |
| `inferior_vena_cava` | 8 | `heart.inferior_vena_cava` | cropped to the thorax |
| `lung_upper_lobe_left` | 13 | `respiratory.lung_upper_lobe_left` | new entity |
| `lung_lower_lobe_left` | 14 | `respiratory.lung_lower_lobe_left` | new entity |
| `lung_upper_lobe_right` | 15 | `respiratory.lung_upper_lobe_right` | new entity |
| `lung_middle_lobe_right` | 16 | `respiratory.lung_middle_lobe_right` | new entity |
| `lung_lower_lobe_right` | 17 | `respiratory.lung_lower_lobe_right` | new entity |
| `trachea` | 43 | `respiratory.trachea` | new entity |

**Not in the data:** valves, superior vena cava (v1 has none), coronaries, conduction system,
pericardium and septa as separate parts. These stay symbolic (§1, item 5).

**Alternatives kept for later.** The KCL four-chamber heart mesh cohort
([Zenodo 3890034](https://zenodo.org/records/3890034), CC BY 4.0, 24 hearts, 0.8–1.3 GB each)
has valve planes and finer meshes. TotalSegmentator v2's `heartchambers_highres` model needs a
license for non-commercial use, and commercial use needs the author's permission, so it is
avoided.

## 4. Architecture

```
TotalSegmentator labels (.nii.gz)
  → mesh pipeline     marching cubes per label, crop, smooth, decimate, millimetres,
                      one shared patient frame
  → asset manifest    entity id → mesh file, source label, subject, license, triangle count
  → GLB scene         one node per entity, node name = AWR entity id
  → local server      text → CommandEngine → AWR scene state (JSON)
  → browser viewer    three.js: loads the GLB, applies visibility, opacity, material and LOD
                      from the AWR state; never edits geometry
  → language layer    later: free text → validated AWR program (ADR 0007), behind the same
                      interface as the rule parser
```

Geometry stays a *view* of the AWR, which is the project's founding rule. The viewer reads
state; only the engine writes it.

## 5. Decisions to record as ADRs

1. **A `thorax` domain.** `ontology/thorax/` composes Heart Ontology v0.1, unchanged, with a new
   respiratory ontology. The research experiments keep loading `ontology/heart` exactly as now.
   This needs a small loader extension: an ontology may `include` another by id and version.
2. **Real-data meshes are reference geometry, not generated geometry.** They fill the
   `MeshRepresentation` slot. A future learned decoder (Level 2) replaces them behind the same
   interface.
3. **A third dependency tier for assets.** `nibabel`, `scikit-image` and `trimesh` are confined
   to a new `assets` extra (extending ADR 0010). The AWR core stays standard library plus
   PyYAML.
4. **Attribution travels with the data.** Every manifest entry and the viewer's credits panel
   name the dataset and its license.

## 6. Milestones

| week | milestone | needs the user? |
| --- | --- | --- |
| 1 | This plan; ADRs; respiratory ontology v0.1 and the `thorax` domain; label map with tests | no |
| 1–2 | Mesh pipeline, tested on synthetic volumes with known shapes | no |
| 2 | Fetch the label subset for 20 subjects; build meshes; quality report; pick one subject | **yes: download** |
| 3–4 | Viewer and local server; engine commands for lungs; scripted session test | no |
| 5–6 | Language layer: an LLM emits AWR programs, validated before execution | **yes: API cost** |
| 7–8 | Polish: LOD ladder for the thorax, explanations, credits panel, demo script | no |

**Quality checks before a subject is used:** each mesh is a single main component, plausible
in volume, with no holes above a threshold. Chambers lie inside the myocardium's bounding box.
Lungs lie on the correct side. The checks are automated and reported, not eyeballed.

## 7. Risks

* **Segmentation quality.** v1 labels were produced with model help and refined. Some subjects
  will have broken parts, which is why there are 20 candidates and automated checks.
* **Coordinate frames.** CT orientation varies. Everything is resampled to one frame, and a test
  pins left and right.
* **Scope creep into medicine.** The viewer and README say "research CT, not validated" on
  every page.
* **Disk.** About 21 GB is free. The subset approach keeps Level 1 well under 1 GB.

## 8. Level 2, for orientation only

A learned model that generates new, plausible hearts and lungs. Per-entity implicit fields
(ADR 0003) are trained on hundreds of these subjects and conditioned on the AWR. It needs a GPU,
which means a cost decision, and it starts only after Level 1 works end to end.

---

## Status, 2026-10-15

*Appended; the plan above is unchanged.*

| milestone | state |
| --- | --- |
| Plan, ADRs, thorax domain, label map | done |
| Mesh pipeline, on a synthetic phantom | done |
| Byte-range fetcher and quality report | done; fetch tested only against a local server |
| Fetch 20 subjects, build them, choose one | done: 32.6 MB fetched; 15 of 20 usable; s0029 chosen (see below) |
| Viewer and local server | done; checked in the browser on the phantom |
| Language layer | decided: the Claude API, when it is needed; not built yet |
| Polish | partly done: pulmonary circulation, cardiac-cycle playback |

### What changed from the plan, and why

* **Engine fixes moved earlier.** The browser check showed "show only the lungs" drawing the
  LOD-0 lung shells and the lobes on top of each other. So `CommandEngine._drawable_targets`
  now resolves shell against parts by active LOD. "Hide the heart" now hides its parts, and
  "generate the heart and lungs" builds the thorax.
* **Connectedness is counted in voxels, not surfaces.** A hollow myocardium has two surfaces
  and is still one structure. The quality report's hard check uses mask pieces.
* **The field-of-view check exempts the trachea** and the cropped vessels, which leave the
  thorax by nature.
* **Pulmonary circulation was added** once the flow model had tests (ADR-L1-001, decision 4).
  The viewer then gained cardiac-cycle playback, which plays the engine's semantic clip slower
  than life.
* **`AnatomyType` stays frozen.** The research feature extractor sizes an input from it.

### What the first real subjects taught the checks

The first build of 20 real subjects (TotalSegmentator v1, 32.6 MB fetched) passed none of the
quality checks. Reading the reasons showed two errors in the checks, not in the data:

* **Chamber sides were judged against the heart's centre.** "Left" and "right" in chamber names
  mean the two circulations. The heart is rotated: the right ventricle is anterior and reaches
  left of the midline, and the left atrium is posterior and close to it. Every subject
  "failed" for the left atrium. Chambers are now judged against their partner (left
  ventricle left of right ventricle, left atrium left of right atrium), and lungs and lobes
  against the body midline at the trachea. The phantom had hidden this because its chambers
  were placed with the same naive idea.
* **Watertightness was measured after decimation.** Full-resolution lung surfaces are closed;
  decimating to 20,000 faces opened small holes. Quality is now measured before decimation,
  and the display mesh's own closure is recorded as information only.

Several subjects still fail for genuine reasons: scans cut above the heart, and structures
missing from the field of view.

### The first real scene

After those fixes, 15 of the 20 subjects pass every hard check. The five that fail do so for
genuine reasons:

* scans cut above the heart, with the trachea or pulmonary trunk missing (s0038, s0058,
  s0089);
* an aorta split by the field of view (s0057);
* a heart and left lung cut at the left and inferior faces (s0067).

The chosen subject is **s0029**. It is the first usable subject with no dropped specks:
17 closed surfaces, and all 11 side checks pass. In the browser, an anterior view shows the
right lung's three lobes on screen left, the trachea above, and with the lungs hidden the
aortic arch, pulmonary trunk and chambers in their expected places.
