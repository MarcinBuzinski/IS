"""Download UCI Wine Quality CSVs and verify SHA-256."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from winequality.config import DATA_RAW, RED_SHA256, RED_URL, WHITE_SHA256, WHITE_URL


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, expected: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_of(dest) == expected:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    digest = sha256_of(tmp)
    if digest != expected:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 mismatch for {dest.name}: got {digest}, expected {expected}")
    tmp.replace(dest)
    return dest


def fetch_raw() -> dict[str, Path]:
    """Return paths to red and white CSVs, downloading if needed."""
    red = _download(RED_URL, DATA_RAW / "winequality-red.csv", RED_SHA256)
    white = _download(WHITE_URL, DATA_RAW / "winequality-white.csv", WHITE_SHA256)
    return {"red": red, "white": white}


if __name__ == "__main__":
    paths = fetch_raw()
    for name, path in paths.items():
        print(f"{name}: {path}  sha256={sha256_of(path)}")
