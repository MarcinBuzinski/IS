"""Užtikrina, kad testai importuotų naują paketą, ne archyvą „senas bandymas“."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "senas bandymas"


def _same(path: str) -> bool:
    try:
        return Path(path).resolve() == ARCHIVE.resolve()
    except OSError:
        return False


sys.path[:] = [path for path in sys.path if path and not _same(path)]
root = str(ROOT)
sys.path[:] = [path for path in sys.path if path != root]
sys.path.insert(0, root)
