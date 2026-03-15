"""Tests for the data validation module."""

import pandas as pd
import pytest

from src.data.validate import ValidationResult, validate_processed, validate_raw


class TestValidationResult:
    def test_initial_state(self):
        result = ValidationResult(stage="test")
        assert result.ok
        assert result.checks_passed == 0
        assert result.checks_failed == 0

    def test_pass_increments(self):
        result = ValidationResult(stage="test")
        result._pass("check 1")
        assert result.checks_passed == 1
        assert result.ok

    def test_fail_increments(self):
        result = ValidationResult(stage="test")
        result._fail("check 1 failed")
        assert result.checks_failed == 1
        assert not result.ok
        assert "check 1 failed" in result.errors

    def test_mixed_results(self):
        result = ValidationResult(stage="test")
        result._pass("ok")
        result._fail("not ok")
        assert result.checks_passed == 1
        assert result.checks_failed == 1
        assert not result.ok


class TestValidateRaw:
    def test_returns_validation_result(self):
        result = validate_raw()
        assert isinstance(result, ValidationResult)
        assert result.stage == "raw"


class TestValidateProcessed:
    def test_returns_validation_result(self):
        result = validate_processed()
        assert isinstance(result, ValidationResult)
        assert result.stage == "processed"
