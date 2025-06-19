import hashlib
from pathlib import Path


def _sha256sum(self, filename: Path) -> str:
    """
    Compute SHA-256 checksum.
    """
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()