"""
Data versioning — tracks which snapshot of IRS data the model was trained on.

Every download run writes a version manifest (JSON) to data/raw/VERSION.json.
The training pipeline reads it and embeds the version in model metadata,
so you always know which data produced which model.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import DATA_DIR, MODELS_DIR

logger = logging.getLogger(__name__)

RAW_DIR = DATA_DIR / "raw"
VERSION_PATH = RAW_DIR / "VERSION.json"


def _file_sha256(path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def stamp_download(files: list[str] | None = None) -> dict:
    """
    Write a VERSION.json manifest after a successful download.

    Parameters
    ----------
    files : list[str] | None
        Filenames inside data/raw/ to include.  Defaults to
        ``["bmf.parquet", "revocations.parquet"]``.

    Returns
    -------
    dict
        The manifest that was written.
    """
    if files is None:
        files = ["bmf.parquet", "revocations.parquet"]

    checksums: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for name in files:
        path = RAW_DIR / name
        if path.exists():
            checksums[name] = _file_sha256(path)
            sizes[name] = path.stat().st_size
        else:
            logger.warning("Version stamp: %s not found, skipping", name)

    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "checksums": checksums,
        "sizes_bytes": sizes,
    }

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    VERSION_PATH.write_text(json.dumps(manifest, indent=2))
    logger.info("Data version manifest written to %s", VERSION_PATH)
    return manifest


def load_version() -> dict | None:
    """
    Load the current data version manifest.

    Returns None if no VERSION.json exists (data has never been downloaded).
    """
    if not VERSION_PATH.exists():
        return None
    return json.loads(VERSION_PATH.read_text())


def embed_in_metadata() -> None:
    """
    Embed the current data version into the model metadata file.

    Called after training so the model metadata records exactly which
    data snapshot produced it.
    """
    version = load_version()
    if version is None:
        logger.warning("No data version manifest found — cannot embed in metadata")
        return

    metadata_path = MODELS_DIR / "metadata.json"
    if not metadata_path.exists():
        logger.warning("Model metadata not found at %s", metadata_path)
        return

    meta = json.loads(metadata_path.read_text())
    meta["data_version"] = version
    metadata_path.write_text(json.dumps(meta, indent=2))
    logger.info("Data version embedded in model metadata")
