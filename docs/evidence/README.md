# Evidence

Screenshots of the system working, captured from the running viewer in the browser, each
with what it shows and how to reproduce it. New captures are added here as results arrive.

Real-CT images show meshes derived from the **TotalSegmentator dataset v1.0** (Wasserthal J.
et al., Zenodo 10.5281/zenodo.6802614, **CC BY 4.0**). They are research data, not validated,
and not for clinical use. Phantom images show simple shapes, not anatomy.

| file | date | shows | produced by |
| --- | --- | --- | --- |
| `2026-10-15_phantom_viewer_first_light.jpg` | 2026-10-15 | First end-to-end run of the viewer, on the synthetic phantom. 17 entity meshes loaded by entity id, banner and credits drawn, "no geometry yet" tags for valves. | commit `d392d92`; `python -m viewer.server --scene datasets/processed/level1/phantom` |
| `2026-10-15_phantom_cardiac_cycle_ejection.png` | 2026-10-15 | The engine's semantic cardiac cycle played in the viewer, 3× slower, during ventricular ejection: ventricles contract, the aorta glows. | commit `6d3c593`; "Animate blood flow" |
| `2026-10-15_real_ct_s0029_heart_and_lungs.jpg` | 2026-10-15 | **First real-CT scene.** Subject s0029 in anterior view: right lung's three lobes (screen left), left lung's two, trachea, heart and vessels. | commits `a01548f`, `0d36a9c`; `python -m viewer.server --scene datasets/processed/level1/real/s0029` |
| `2026-10-15_real_ct_s0029_lungs_hidden.jpg` | 2026-10-15 | The same scene after the typed command "hide the lungs": aortic arch, pulmonary trunk, chambers, IVC. The command hid 5 lobes, and the lung shells were already hidden at LOD 1. | as above |
| `2026-10-15_real_ct_s0029_phone_layout.jpg` | 2026-10-15 | The viewer at phone width (375 px): the scene re-frames and the controls stack below it. | commit `0d36a9c` |
| `2026-10-16_real_ct_s0029_lod0_organ_shells.png` | 2026-10-16 | Level of detail 0 on real CT: heart shell, two lung shells and the trachea, with no lobes. The engine draws an organ as its shell or as its parts, never both. | commit `5823afa`; "show less detail" |
