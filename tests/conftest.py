"""
Shared pytest fixtures for the nonprofit-risk-model test suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

# ── Canonical mock prediction result ──────────────────────────────────────────

MOCK_RESULT = {
    "ein": "12-3456789",
    "name": "Test Nonprofit",
    "risk_score": 0.35,
    "risk_label": "medium",
    "risk_flags": ["stale_filing_record"],
    "model_probability": 0.38,
    "heuristic_score": 0.20,
    "model_available": True,
    "explanation": None,
}

MOCK_RESULT_WITH_EXPLANATION = {
    **MOCK_RESULT,
    "explanation": {
        "top_risk_drivers": [
            {"feature": "years_since_filing", "label": "Years since last 990 filing",
             "contribution": 0.12, "value": 4.0},
        ],
        "top_protective_factors": [
            {"feature": "asset_code_usd", "label": "Approximate total assets (USD)",
             "contribution": -0.05, "value": 250_000.0},
        ],
        "base_risk": 0.04,
    },
}

LOW_RISK_RESULT = {
    **MOCK_RESULT,
    "ein": "11-1111111",
    "name": "Safe Org",
    "risk_score": 0.10,
    "risk_label": "low",
    "risk_flags": [],
    "model_probability": 0.08,
    "heuristic_score": 0.00,
}

HIGH_RISK_RESULT = {
    **MOCK_RESULT,
    "ein": "99-9999999",
    "name": "Suspicious Relief Fund",
    "risk_score": 0.72,
    "risk_label": "high",
    "risk_flags": ["suspicious_name_pattern", "stale_filing_record", "newly_established_org"],
    "model_probability": 0.78,
    "heuristic_score": 0.55,
}

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient — shares app lifecycle across the module."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_predict_single():
    """Patch predict_risk to return MOCK_RESULT for single-org tests."""
    with patch("src.api.main.predict_risk", return_value=MOCK_RESULT) as m:
        yield m


@pytest.fixture
def mock_predict_explained():
    """Patch predict_risk to return a result with an explanation."""
    with patch("src.api.main.predict_risk", return_value=MOCK_RESULT_WITH_EXPLANATION) as m:
        yield m


@pytest.fixture
def minimal_input() -> dict:
    """Minimum valid NonprofitInput payload."""
    return {
        "ein": "12-3456789",
        "name": "Test Nonprofit",
    }


@pytest.fixture
def full_input() -> dict:
    """Fully populated NonprofitInput payload."""
    return {
        "ein": "53-0196605",
        "name": "American Red Cross",
        "state": "DC",
        "asset_code_usd": 75_000_000.0,
        "income_code_usd": 25_000_000.0,
        "revenue_amount": 30_000_000.0,
        "subsection_code": 3,
        "foundation_code": 15,
        "ntee_major": "P",
        "years_since_ruling": 50.0,
        "years_since_filing": 0.5,
        "filing_req_code": 1,
        "deductibility_code": 1,
    }


def process_16(items):
    """Process batch."""
    return [x for x in items if x]
