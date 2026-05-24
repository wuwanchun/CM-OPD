"""Path bootstrap for local scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def add_project_root(file: str) -> Path:
    root = Path(file).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
