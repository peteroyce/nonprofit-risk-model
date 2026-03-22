"""
Integration tests for the FastAPI endpoints.

All calls to predict_risk() are mocked so these tests run without a
trained model or IRS data files.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.conftest import (
    HIGH_RISK_RESULT,
    LOW_RISK_RESULT,
    MOCK_RESULT,
    MOCK_RESULT_WITH_EXPLANATION,
)


# ── /v1/health ────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        res = client.get("/v1/health")
        assert res.status_code == 200

    def test_response_shape(self, client):
        body = client.get("/v1/health").json()
        assert "status" in body
        assert "model_available" in body
        assert "uptime_seconds" in body
        assert "version" in body

    def test_status_is_ok(self, client):
        body = client.get("/v1/health").json()
        assert body["status"] == "ok"

    def test_legacy_health_redirect(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        assert "redirect" in res.json()


# ── POST /v1/predict ──────────────────────────────────────────────────────────

class TestPredict:
    def test_minimal_input_returns_200(self, client, mock_predict_single, minimal_input):
        res = client.post("/v1/predict", json=minimal_input)
        assert res.status_code == 200

    def test_response_has_required_fields(self, client, mock_predict_single, minimal_input):
        body = client.post("/v1/predict", json=minimal_input).json()
        for field in ("ein", "name", "risk_score", "risk_label", "risk_flags",
                      "heuristic_score", "model_available"):
            assert field in body, f"Missing field: {field}"

    def test_full_input_returns_200(self, client, mock_predict_single, full_input):
        res = client.post("/v1/predict", json=full_input)
        assert res.status_code == 200

    def test_explain_false_no_explanation(self, client, mock_predict_single, minimal_input):
        body = client.post("/v1/predict?explain=false", json=minimal_input).json()
        assert body.get("explanation") is None

    def test_explain_true_returns_explanation(self, client, minimal_input):
        with patch("src.api.main.predict_risk", return_value=MOCK_RESULT_WITH_EXPLANATION):
            body = client.post("/v1/predict?explain=true", json=minimal_input).json()
        exp = body.get("explanation")
        assert exp is not None
        assert "top_risk_drivers" in exp
        assert "top_protective_factors" in exp
        assert "base_risk" in exp

    def test_predict_called_with_correct_params(self, client, mock_predict_single, minimal_input):
        client.post("/v1/predict?explain=false", json=minimal_input)
        call_kwargs = mock_predict_single.call_args.kwargs
        assert call_kwargs["ein"] == minimal_input["ein"]
        assert call_kwargs["name"] == minimal_input["name"]
        assert call_kwargs["explain"] is False

    def test_explain_param_forwarded(self, client, minimal_input):
        with patch("src.api.main.predict_risk", return_value=MOCK_RESULT) as m:
            client.post("/v1/predict?explain=true", json=minimal_input)
        assert m.call_args.kwargs["explain"] is True

    def test_missing_ein_returns_422(self, client, mock_predict_single):
        res = client.post("/v1/predict", json={"name": "No EIN Corp"})
        assert res.status_code == 422

    def test_missing_name_returns_422(self, client, mock_predict_single):
        res = client.post("/v1/predict", json={"ein": "12-3456789"})
        assert res.status_code == 422

    def test_negative_asset_returns_422(self, client, mock_predict_single, minimal_input):
        payload = {**minimal_input, "asset_code_usd": -1000.0}
        res = client.post("/v1/predict", json=payload)
        assert res.status_code == 422

    def test_predict_500_on_exception(self, client, minimal_input):
        with patch("src.api.main.predict_risk", side_effect=RuntimeError("model exploded")):
            res = client.post("/v1/predict", json=minimal_input)
        assert res.status_code == 500
        assert "model exploded" in res.json()["detail"]

    def test_risk_score_in_range(self, client, mock_predict_single, minimal_input):
        body = client.post("/v1/predict", json=minimal_input).json()
        assert 0.0 <= body["risk_score"] <= 1.0

    def test_risk_label_values(self, client, minimal_input):
        for result in (
            {**MOCK_RESULT, "risk_label": "low"},
            {**MOCK_RESULT, "risk_label": "medium"},
            {**MOCK_RESULT, "risk_label": "high"},
        ):
            with patch("src.api.main.predict_risk", return_value=result):
                body = client.post("/v1/predict", json=minimal_input).json()
            assert body["risk_label"] in ("low", "medium", "high")


# ── Input validation ──────────────────────────────────────────────────────────

class TestInputValidation:
    def test_ein_with_dash_accepted(self, client, mock_predict_single):
        res = client.post("/v1/predict", json={"ein": "53-0196605", "name": "Red Cross"})
        assert res.status_code == 200

    def test_ein_without_dash_accepted(self, client, mock_predict_single):
        res = client.post("/v1/predict", json={"ein": "530196605", "name": "Red Cross"})
        assert res.status_code == 200

    def test_ein_too_short_rejected(self, client, mock_predict_single):
        res = client.post("/v1/predict", json={"ein": "1234", "name": "Bad Org"})
        assert res.status_code == 422

    def test_ein_non_numeric_rejected(self, client, mock_predict_single):
        res = client.post("/v1/predict", json={"ein": "ABCDEFGHI", "name": "Bad Org"})
        assert res.status_code == 422

    def test_ein_empty_rejected(self, client, mock_predict_single):
        res = client.post("/v1/predict", json={"ein": "", "name": "Bad Org"})
        assert res.status_code == 422

    def test_state_uppercased(self, client, mock_predict_single):
        res = client.post("/v1/predict", json={"ein": "12-3456789", "name": "Org", "state": "ca"})
        assert res.status_code == 200

    def test_invalid_state_rejected(self, client, mock_predict_single):
        res = client.post("/v1/predict", json={"ein": "12-3456789", "name": "Org", "state": "XYZ"})
        assert res.status_code == 422

    def test_ein_normalized_to_dash_format(self, client, mock_predict_single):
        client.post("/v1/predict", json={"ein": "530196605", "name": "Test"})
        call_kwargs = mock_predict_single.call_args.kwargs
        assert call_kwargs["ein"] == "53-0196605"


# ── POST /v1/predict/batch ────────────────────────────────────────────────────

class TestPredictBatch:
    def _batch_payload(self, n: int = 2) -> dict:
        return {
            "nonprofits": [
                {"ein": f"12-{i:07d}", "name": f"Org {i}"}
                for i in range(1, n + 1)
            ]
        }

    def test_batch_two_orgs(self, client, mock_predict_single):
        res = client.post("/v1/predict/batch", json=self._batch_payload(2))
        assert res.status_code == 200
        body = res.json()
        assert body["total_requested"] == 2
        assert body["total_scored"] == 2
        assert body["total_failed"] == 0
        assert len(body["results"]) == 2

    def test_batch_partial_failure(self, client, minimal_input):
        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("bad data")
            return MOCK_RESULT

        payload = {"nonprofits": [
            {"ein": "11-1111111", "name": "Org A"},
            {"ein": "22-2222222", "name": "Org B"},
            {"ein": "33-3333333", "name": "Org C"},
        ]}
        with patch("src.api.main.predict_risk", side_effect=side_effect):
            body = client.post("/v1/predict/batch", json=payload).json()

        assert body["total_scored"] == 2
        assert body["total_failed"] == 1
        assert len(body["errors"]) == 1
        assert body["errors"][0]["ein"] == "22-2222222"

    def test_batch_empty_list_returns_422(self, client, mock_predict_single):
        res = client.post("/v1/predict/batch", json={"nonprofits": []})
        assert res.status_code == 422

    def test_batch_respects_explain_param(self, client):
        with patch("src.api.main.predict_risk", return_value=MOCK_RESULT) as m:
            client.post(
                "/v1/predict/batch?explain=true",
                json={"nonprofits": [{"ein": "12-3456789", "name": "Test"}]},
            )
        assert m.call_args.kwargs["explain"] is True

    def test_batch_response_fields(self, client, mock_predict_single):
        body = client.post("/v1/predict/batch", json=self._batch_payload(1)).json()
        for field in ("results", "errors", "total_requested", "total_scored", "total_failed"):
            assert field in body


# ── POST /v1/predict/compare ──────────────────────────────────────────────────

class TestPredictCompare:
    def _compare_payload(self) -> dict:
        return {
            "nonprofits": [
                {"ein": "11-1111111", "name": "Safe Org"},
                {"ein": "99-9999999", "name": "Suspicious Relief Fund"},
            ]
        }

    def test_compare_returns_200(self, client):
        results = [LOW_RISK_RESULT, HIGH_RISK_RESULT]
        with patch("src.api.main.predict_risk", side_effect=results):
            res = client.post("/v1/predict/compare", json=self._compare_payload())
        assert res.status_code == 200

    def test_compare_ranked_descending(self, client):
        results = [LOW_RISK_RESULT, HIGH_RISK_RESULT]
        with patch("src.api.main.predict_risk", side_effect=results):
            body = client.post("/v1/predict/compare", json=self._compare_payload()).json()
        ranked = body["ranked"]
        scores = [r["risk_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_compare_highest_and_lowest_correct(self, client):
        results = [LOW_RISK_RESULT, HIGH_RISK_RESULT]
        with patch("src.api.main.predict_risk", side_effect=results):
            body = client.post("/v1/predict/compare", json=self._compare_payload()).json()
        assert body["highest_risk"]["ein"] == HIGH_RISK_RESULT["ein"]
        assert body["lowest_risk"]["ein"] == LOW_RISK_RESULT["ein"]

    def test_compare_total_count(self, client):
        results = [LOW_RISK_RESULT, HIGH_RISK_RESULT]
        with patch("src.api.main.predict_risk", side_effect=results):
            body = client.post("/v1/predict/compare", json=self._compare_payload()).json()
        assert body["total"] == 2

    def test_compare_single_org_returns_422(self, client, mock_predict_single):
        res = client.post("/v1/predict/compare", json={
            "nonprofits": [{"ein": "12-3456789", "name": "Only One"}]
        })
        assert res.status_code == 422

    def test_compare_all_fail_returns_422(self, client):
        with patch("src.api.main.predict_risk", side_effect=RuntimeError("crash")):
            res = client.post("/v1/predict/compare", json=self._compare_payload())
        assert res.status_code == 422


# ── GET /v1/model/features ────────────────────────────────────────────────────

class TestModelFeatures:
    def test_returns_404_without_model(self, client):
        """No model file exists in the test environment."""
        res = client.get("/v1/model/features")
        assert res.status_code == 404

    def test_returns_metadata_when_model_present(self, client, tmp_path):
        import json
        meta = {
            "shap_importance": {"years_since_filing": 0.12, "asset_code_usd": 0.08},
            "feature_names": ["years_since_filing", "asset_code_usd"],
            "n_features": 2,
            "cv_metrics": {"roc_auc": 0.87},
            "n_train": 10000,
            "xgb_best_iteration": 243,
        }
        fake_path = tmp_path / "metadata.json"
        fake_path.write_text(json.dumps(meta))

        with patch("src.api.main.METADATA_PATH", fake_path):
            body = client.get("/v1/model/features").json()

        assert "feature_importance" in body
        assert "cv_metrics" in body
        assert body["n_features"] == 2
