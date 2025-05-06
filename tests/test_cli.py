"""Tests for the CLI module."""

import subprocess
import sys


class TestCLI:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "nonprofit-risk-model" in result.stdout

    def test_download_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "download", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--force" in result.stdout

    def test_train_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "train", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--sample" in result.stdout

    def test_serve_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "serve", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--port" in result.stdout

    def test_predict_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "predict", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "ein" in result.stdout

    def test_no_command_fails(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.cli"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
