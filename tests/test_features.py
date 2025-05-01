"""Unit tests for feature engineering and scoring logic."""

import pytest
from src.features.engineering import (
    RiskFlags,
    blend_scores,
    compute_heuristic_score,
    extract_risk_flags,
)


class TestExtractRiskFlags:
    def test_suspicious_name_detected(self):
        flags = extract_risk_flags(name="Global Emergency Relief Fund")
        assert flags.suspicious_name is True

    def test_clean_name_not_flagged(self):
        flags = extract_risk_flags(name="American Red Cross")
        assert flags.suspicious_name is False

    def test_stale_filing_flag(self):
        flags = extract_risk_flags(name="Test Org", years_since_filing=5.0)
        assert flags.stale_filing is True

    def test_recent_filing_not_flagged(self):
        flags = extract_risk_flags(name="Test Org", years_since_filing=1.0)
        assert flags.stale_filing is False

    def test_new_org_flag(self):
        flags = extract_risk_flags(name="Test Org", years_since_ruling=1.0)
        assert flags.new_organization is True

    def test_established_org_not_flagged(self):
        flags = extract_risk_flags(name="Test Org", years_since_ruling=10.0)
        assert flags.new_organization is False

    def test_high_risk_foundation_code(self):
        flags = extract_risk_flags(name="Test Org", foundation_code=17)
        assert flags.high_risk_foundation is True

    def test_normal_foundation_code(self):
        flags = extract_risk_flags(name="Test Org", foundation_code=15)
        assert flags.high_risk_foundation is False

    def test_missing_ntee_flagged(self):
        flags = extract_risk_flags(name="Test Org", ntee_code="")
        assert flags.missing_ntee is True

    def test_valid_ntee_not_flagged(self):
        flags = extract_risk_flags(name="Test Org", ntee_code="P20")
        assert flags.missing_ntee is False

    def test_flag_count(self):
        flags = extract_risk_flags(
            name="Global Emergency Relief Fund",
            years_since_filing=5.0,
            years_since_ruling=1.0,
        )
        assert flags.flag_count == 3

    def test_to_list(self):
        flags = extract_risk_flags(name="Test Org", years_since_filing=5.0)
        assert "stale_filing_record" in flags.to_list()


class TestComputeHeuristicScore:
    def test_no_flags_zero_score(self):
        flags = RiskFlags()
        assert compute_heuristic_score(flags) == 0.0

    def test_all_flags_score_leq_one(self):
        flags = RiskFlags(
            suspicious_name=True,
            high_risk_deductibility=True,
            high_risk_foundation=True,
            stale_filing=True,
            new_organization=True,
            missing_ntee=True,
        )
        score = compute_heuristic_score(flags)
        assert 0.0 <= score <= 1.0

    def test_suspicious_name_contributes_most(self):
        flags_name = RiskFlags(suspicious_name=True)
        flags_ntee = RiskFlags(missing_ntee=True)
        assert compute_heuristic_score(flags_name) > compute_heuristic_score(flags_ntee)


class TestBlendScores:
    def test_model_dominates_by_default(self):
        result = blend_scores(model_prob=0.8, heuristic_score=0.0, model_weight=0.75)
        assert result == pytest.approx(0.6, abs=0.01)

    def test_equal_scores_return_same(self):
        result = blend_scores(model_prob=0.5, heuristic_score=0.5)
        assert result == pytest.approx(0.5, abs=0.001)

    def test_bounded_zero_one(self):
        result = blend_scores(model_prob=1.0, heuristic_score=1.0)
        assert 0.0 <= result <= 1.0
