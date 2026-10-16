# Probe: registering real lungs (2026-10-16)

**Question.** Before the 200-subject shape build, do the registration settings validated on
left ventricles also fit lungs within the build's 2 mm threshold?

**Setup.**
- Reference: s0029. Targets: s0037, s0040, s0057 (TotalSegmentator v1.0, CC BY 4.0).
- Each subject was rigidly aligned to the reference by its heart shell.
- Fit is `mean_from_target`: the mean distance from target surface samples to the fitted
  template, in mm.

**Result 1.** With the chambers' settings (2000 template faces, default stages, no placement):

| subject | left lung | right lung |
| --- | --- | --- |
| s0037 | 2.69 | **17.90** |
| s0040 | 1.39 | 2.42 |
| s0057 | 1.91 | 2.79 |

4 of 6 lungs miss 2 mm. The template sits on the target (0.8 mm template to target), but
does not reach all of it. Non-rigid ICP only pulls towards surface within 15 mm, and lungs
differ in height by more than that.

**Fix 1: placement.** A per-axis scale and shift that matches robust extents (1st to 99th
percentile) is applied before non-rigid ICP. It fixed s0037's right lung (17.90 → 1.89).
Chambers stay at 0.21–0.50 mm.

**Result 2: lung variants, all with placement.** Each line lists the six lungs in this
order: s0037 L, s0040 L, s0057 L, s0037 R, s0040 R, s0057 R.

| template faces | stages | mean_from_target (mm) | mean template to target (mm) |
| --- | --- | --- | --- |
| 2000 | default | 2.35 1.44 1.89 1.89 1.33 2.22 | 0.81 |
| 4000 | default | 1.88 1.20 1.66 1.77 1.22 2.19 | 0.88 |
| 2000 | loose | 2.13 1.18 1.61 1.64 1.06 1.91 | 0.06 |
| **4000** | **loose** | **1.51 0.84 1.23 1.43 0.86 1.76** | **0.09** |

**Decision.** Lungs use 4000 faces and the loose stages (`LOOSE_STEPS`, `LARGE_ORGANS`).
Chambers keep the validated defaults. A looser fit could let correspondence slide in ways
that surface distance cannot show, so it is not applied where it is not needed.

**Remaining.**
- In every variant, 3–8% of the target lung surface stays more than 5 mm from the fit (max
  16–37 mm). Most of those points lie outside the fitted surface.
- They are thin wedges and recesses that a smooth template does not enter. The threshold
  passes, but this is a known limit of the lung models.
- Lung display meshes are not watertight after decimation; their quality is measured before
  decimation.

**Reproduce.** `shapes.register.placement`, `LOOSE_STEPS`; the script follows.

```python
"""Lungs only: template face budget and a looser final stage."""
import time
from pathlib import Path
import numpy as np
from shapes.register import DEFAULT_STEPS, load_scene, make_template, register, rigid_to, fit_error

root = Path("datasets/processed/level1/real")
ref = load_scene(root / "s0029")
subjects = {s: load_scene(root / s) for s in ("s0037", "s0040", "s0057")}
mats = {s: rigid_to(ref["heart"], sc["heart"], seed=0)[0] for s, sc in subjects.items()}
LOOSE = DEFAULT_STEPS[:-1] + ((0.005, 0.0, 0.0, 10), (0.002, 0.0, 0.0, 10))
variants = {
    "2000 default": (2000, DEFAULT_STEPS),
    "4000 default": (4000, DEFAULT_STEPS),
    "2000 loose": (2000, LOOSE),
    "4000 loose": (4000, LOOSE),
}
for name, (faces, steps) in variants.items():
    rows = []
    t = time.time()
    for e in ("respiratory.left_lung", "respiratory.right_lung"):
        tpl = make_template(ref[e], faces)
        for s, sc in subjects.items():
            target = sc[e].copy(); target.apply_transform(mats[s])
            v = register(tpl, target, steps=steps)
            err = fit_error(v, tpl.faces, target)
            rows.append((err["mean_from_target"], err["mean_to_target"], err["max_from_target"]))
    r = np.array(rows)
    print(f"{name:14s} from {np.round(r[:,0],2)} to-mean {r[:,1].mean():.2f} max {np.round(r[:,2],0)} {time.time()-t:.0f}s", flush=True)
```
