"""Tests for the data versioning module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.version import _file_sha256, load_version, stamp_download


@pytest.fixture
def raw_dir(tmp_path):
    """Create a temporary raw data directory with fake parquet files."""
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)

    # Write deterministic fake files
    (raw / "bmf.parquet").write_bytes(b"fake-bmf-data-for-hashing")
    (raw / "revocations.parquet").write_bytes(b"fake-revocations-data")
    return raw


class TestFileSha256:
    def test_deterministic_hash(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        h1 = _file_sha256(f)
        h2 = _file_sha256(f)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"aaa")
        f2.write_bytes(b"bbb")
        assert _file_sha256(f1) != _file_sha256(f2)

    def test_returns_hex_string(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"test")
        h = _file_sha256(f)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest length


class TestStampDownload:
    def test_creates_version_file(self, raw_dir, monkeypatch):
        import src.data.version as mod
        monkeypatch.setattr(mod, "RAW_DIR", raw_dir)
        monkeypatch.setattr(mod, "VERSION_PATH", raw_dir / "VERSION.json")

        stamp_download(files=["bmf.parquet", "revocations.parquet"])
        assert (raw_dir / "VERSION.json").exists()

    def test_manifest_has_expected_keys(self, raw_dir, monkeypatch):
        import src.data.version as mod
        monkeypatch.setattr(mod, "RAW_DIR", raw_dir)
        monkeypatch.setattr(mod, "VERSION_PATH", raw_dir / "VERSION.json")

        manifest = stamp_download(files=["bmf.parquet"])
        assert "downloaded_at" in manifest
        assert "checksums" in manifest
        assert "sizes_bytes" in manifest
        assert "bmf.parquet" in manifest["checksums"]

    def test_skips_missing_files(self, raw_dir, monkeypatch):
        import src.data.version as mod
        monkeypatch.setattr(mod, "RAW_DIR", raw_dir)
        monkeypatch.setattr(mod, "VERSION_PATH", raw_dir / "VERSION.json")

        manifest = stamp_download(files=["bmf.parquet", "nonexistent.csv"])
        assert "nonexistent.csv" not in manifest["checksums"]
        assert "bmf.parquet" in manifest["checksums"]


class TestLoadVersion:
    def test_returns_none_when_no_file(self, tmp_path, monkeypatch):
        import src.data.version as mod
        monkeypatch.setattr(mod, "VERSION_PATH", tmp_path / "VERSION.json")
        assert load_version() is None

    def test_loads_existing_manifest(self, raw_dir, monkeypatch):
        import src.data.version as mod
        version_path = raw_dir / "VERSION.json"
        monkeypatch.setattr(mod, "RAW_DIR", raw_dir)
        monkeypatch.setattr(mod, "VERSION_PATH", version_path)

        stamp_download(files=["bmf.parquet"])
        loaded = load_version()
        assert loaded is not None
        assert "downloaded_at" in loaded
