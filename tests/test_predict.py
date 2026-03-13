"""
Unit tests for the prediction module (src/models/predict.py).

The trained model is mocked throughout so these tests run without
IRS data or a serialised model file.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.predict import (
    _build_feature_row,
    _get_shap_explanation,
    _label_from_score,
    predict_risk,
    warmup,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "asset_code_usd", "income_code_usd", "revenue_amount",
    "subsection_code", "foundation_code", "ntee_major",
    "years_since_ruling", "years_since_filing",
    "filing_req_code", "deductibility_code", "state",
]


def _make_mock_model(prob: float = 0.65):
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1 - prob, prob]])
    return model


def _make_mock_explainer(n_features: int = 11, base: float = 0.04):
    explainer = MagicMock()
    # shap_values returns array shape (1, n_features)
    shap_vals = np.zeros((1, n_features))
    shap_vals[0, 7] = 0.15   # years_since_filing — big positive
    shap_vals[0, 0] = -0.06  # asset_code_usd — protective
    explainer.shap_values.return_value = shap_vals
    explainer.expected_value = base
    return explainer


def _make_mock_metadata() -> dict:
    return {
        "feature_names": FEATURE_NAMES,
        "n_train": 1_800_000,
        "xgb_best_iteration": 312,
        "cv_metrics": {"roc_auc": 0.87},
    }


# ── _label_from_score ─────────────────────────────────────────────────────────

class TestLabelFromScore:
    def test_low(self):
        assert _label_from_score(0.10) == "low"

    def test_medium_lower_bound(self):
        assert _label_from_score(0.25) == "medium"

    def test_medium_upper_bound(self):
        assert _label_from_score(0.54) == "medium"

    def test_high(self):
        assert _label_from_score(0.55) == "high"

    def test_zero_is_low(self):
        assert _label_from_score(0.0) == "low"

    def test_one_is_high(self):
        assert _label_from_score(1.0) == "high"


# ── _build_feature_row ────────────────────────────────────────────────────────

class TestBuildFeatureRow:
    def test_returns_dataframe(self):
        row = _build_feature_row(
            asset_code_usd=250_000, income_code_usd=75_000, revenue_amount=80_000,
            subsection_code=3, foundation_code=15, ntee_major="P",
            years_since_ruling=10.0, years_since_filing=0.5,
            filing_req_code=1, deductibility_code=1, state="CA",
            feature_names=FEATURE_NAMES,
        )
        assert isinstance(row, pd.DataFrame)
        assert len(row) == 1

    def test_columns_match_feature_names(self):
        row = _build_feature_row(
            asset_code_usd=0, income_code_usd=0, revenue_amount=0,
            subsection_code=3, foundation_code=None, ntee_major="Z",
            years_since_ruling=5.0, years_since_filing=1.0,
            filing_req_code=1, deductibility_code=None, state="UNK",
            feature_names=FEATURE_NAMES,
        )
        assert list(row.columns) == FEATURE_NAMES

    def test_none_codes_become_minus_one(self):
        row = _build_feature_row(
            asset_code_usd=0, income_code_usd=0, revenue_amount=0,
            subsection_code=3, foundation_code=None, ntee_major="Z",
            years_since_ruling=5.0, years_since_filing=1.0,
            filing_req_code=1, deductibility_code=None, state="UNK",
            feature_names=FEATURE_NAMES,
        )
        assert row["foundation_code"].iloc[0] == -1
        assert row["deductibility_code"].iloc[0] == -1

    def test_categorical_columns(self):
        row = _build_feature_row(
            asset_code_usd=0, income_code_usd=0, revenue_amount=0,
            subsection_code=3, foundation_code=None, ntee_major="B",
            years_since_ruling=5.0, years_since_filing=1.0,
            filing_req_code=1, deductibility_code=None, state="NY",
            feature_names=FEATURE_NAMES,
        )
        assert str(row["ntee_major"].dtype) == "category"
        assert str(row["state"].dtype) == "category"


# ── _get_shap_explanation ─────────────────────────────────────────────────────

class TestGetSHAPExplanation:
    def _run(self, n_features=11):
        explainer = _make_mock_explainer(n_features=n_features)
        row = pd.DataFrame([{f: 0 for f in FEATURE_NAMES}])
        row["ntee_major"] = row["ntee_major"].astype("category")
        row["state"] = row["state"].astype("category")
        return _get_shap_explanation(explainer, row, FEATURE_NAMES)

    def test_returns_dict_with_keys(self):
        result = self._run()
        assert "top_risk_drivers" in result
        assert "top_protective_factors" in result
        assert "base_risk" in result

    def test_risk_drivers_are_positive(self):
        result = self._run()
        for factor in result["top_risk_drivers"]:
            assert factor["contribution"] > 0

    def test_protective_factors_are_negative(self):
        result = self._run()
        for factor in result["top_protective_factors"]:
            assert factor["contribution"] < 0

    def test_at_most_three_risk_drivers(self):
        result = self._run()
        assert len(result["top_risk_drivers"]) <= 3

    def test_at_most_two_protective_factors(self):
        result = self._run()
        assert len(result["top_protective_factors"]) <= 2

    def test_base_risk_is_float(self):
        result = self._run()
        assert isinstance(result["base_risk"], float)

    def test_labels_populated(self):
        result = self._run()
        for factor in result["top_risk_drivers"]:
            assert factor["label"]  # non-empty string


# ── predict_risk — no model (heuristic fallback) ──────────────────────────────

class TestPredictRiskHeuristic:
    def test_returns_dict(self):
        result = predict_risk(ein="12-3456789", name="Test Org", use_model=False)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        result = predict_risk(ein="12-3456789", name="Test Org", use_model=False)
        for key in ("ein", "name", "risk_score", "risk_label", "risk_flags",
                    "model_probability", "heuristic_score", "model_available"):
            assert key in result

    def test_model_not_available_when_disabled(self):
        result = predict_risk(ein="12-3456789", name="Test Org", use_model=False)
        assert result["model_available"] is False
        assert result["model_probability"] is None

    def test_score_in_range(self):
        result = predict_risk(ein="12-3456789", name="Test Org", use_model=False)
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_suspicious_name_raises_score(self):
        clean = predict_risk(ein="01-0000001", name="Community Health Center", use_model=False)
        risky = predict_risk(ein="01-0000002", name="Global Emergency Relief Fund", use_model=False)
        assert risky["heuristic_score"] > clean["heuristic_score"]

    def test_stale_filing_raises_score(self):
        fresh = predict_risk(ein="01-0000003", name="Test Org",
                              years_since_filing=0.5, use_model=False)
        stale = predict_risk(ein="01-0000004", name="Test Org",
                              years_since_filing=5.0, use_model=False)
        assert stale["heuristic_score"] > fresh["heuristic_score"]

    def test_no_explanation_without_model(self):
        result = predict_risk(ein="12-3456789", name="Test Org",
                               use_model=False, explain=True)
        assert result.get("explanation") is None

    def test_ein_preserved_in_output(self):
        result = predict_risk(ein="53-0196605", name="Red Cross", use_model=False)
        assert result["ein"] == "53-0196605"

    def test_risk_flags_is_list(self):
        result = predict_risk(ein="12-3456789", name="Test Org", use_model=False)
        assert isinstance(result["risk_flags"], list)


# ── predict_risk — with mocked model ─────────────────────────────────────────

class TestPredictRiskWithModel:
    def _patched_load(self, prob: float = 0.65):
        """Context manager that patches _load_model to return fake model/meta/explainer."""
        model = _make_mock_model(prob)
        meta = _make_mock_metadata()
        explainer = _make_mock_explainer()
        return patch(
            "src.models.predict._load_model",
            return_value=(model, meta, explainer),
        )

    def test_model_probability_used(self):
        with self._patched_load(0.70):
            result = predict_risk(ein="12-3456789", name="Test Org")
        assert result["model_available"] is True
        assert result["model_probability"] == pytest.approx(0.70, abs=0.001)

    def test_risk_score_blended(self):
        """Blended score = 0.75 * model_prob + 0.25 * heuristic."""
        with self._patched_load(0.80):
            result = predict_risk(ein="12-3456789", name="Test Org", use_model=True)
        assert 0.0 <= result["risk_score"] <= 1.0
        # Model dominates (75 % weight)
        assert result["risk_score"] > 0.5

    def test_explain_false_no_explanation(self):
        with self._patched_load():
            result = predict_risk(ein="12-3456789", name="Test Org", explain=False)
        assert result.get("explanation") is None

    def test_explain_true_returns_explanation(self):
        with self._patched_load():
            result = predict_risk(ein="12-3456789", name="Test Org", explain=True)
        exp = result.get("explanation")
        assert exp is not None
        assert "top_risk_drivers" in exp
        assert "base_risk" in exp

    def test_file_not_found_falls_back_to_heuristic(self):
        with patch("src.models.predict._load_model",
                   side_effect=FileNotFoundError("no model")):
            result = predict_risk(ein="12-3456789", name="Test Org")
        assert result["model_available"] is False
        assert result["model_probability"] is None

    def test_result_label_consistent_with_score(self):
        with self._patched_load(0.10):
            result = predict_risk(ein="12-3456789", name="Test Org")
        # With low prob, blended score should be low
        if result["risk_score"] < 0.25:
            assert result["risk_label"] == "low"


# ── warmup ─────────────────────────────────────────────────────────────────────

class TestWarmup:
    def test_warmup_returns_false_without_model(self):
        with patch("src.models.predict._load_model",
                   side_effect=FileNotFoundError("no model")):
            assert warmup() is False

    def test_warmup_returns_true_with_model(self):
        mock_model = _make_mock_model()
        mock_meta = _make_mock_metadata()
        mock_explainer = _make_mock_explainer()
        with patch("src.models.predict._load_model",
                   return_value=(mock_model, mock_meta, mock_explainer)):
            assert warmup() is True


# ── Thread-safety smoke test ───────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_predictions_no_exception(self):
        """Hammer predict_risk from multiple threads; none should raise."""
        errors = []

        def worker():
            try:
                predict_risk(ein="12-3456789", name="Concurrent Org", use_model=False)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors in concurrent calls: {errors}"
