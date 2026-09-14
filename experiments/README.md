# Experiments

Scratch space for R&D runs. Nothing here is part of the library.

| Script                   | What it does                                                        |
| ------------------------ | ------------------------------------------------------------------- |
| `heart_session_demo.py`  | Runs the canonical command session against one persistent AWR scene and prints identity, history and validation reports. |

Run the demo:

```bash
python -m experiments.heart_session_demo
python -m experiments.heart_session_demo --quiet --save /tmp/heart_scene.json
```

Run outputs (`experiments/runs/`) are git-ignored.
