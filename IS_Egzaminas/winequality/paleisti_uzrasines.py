"""Paleidžia JupyterLab naršyklėje su šio projekto užrašinėmis."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    nbdir = root / "notebooks"
    nbdir.mkdir(exist_ok=True)
    exe = Path(sys.executable)
    return subprocess.call(
        [
            str(exe),
            "-m",
            "jupyterlab",
            "--notebook-dir",
            str(nbdir),
            "--ip",
            "127.0.0.1",
            "--port",
            "8888",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
