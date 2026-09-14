"""Make the repository root importable when running tests from a checkout.

The project uses flat top-level packages (``awr``, ``reasoning``, ...). Installing
with ``pip install -e .`` is the normal path; this file means ``pytest`` also works
in a bare checkout with no install step.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
