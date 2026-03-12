"""
Feature engineering helpers.

Also provides a `score_nonprofit()` function that computes a risk score
for a *single* nonprofit given its known attributes — used by the API
before the ML model inference step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Heuristic risk flags (rule-based, used to supplement model predictions)
# ---------------------------------------------------------------------------

SUSPICIOUS_NAME_PATTERNS = [
    r"\brelief\b.*\bfund\b",
    r"\bemergency\b.*\brelief\b",
    r"\bcovid\b.*\bfund\b",
    r"\bdisaster\b.*\bfund\b",
    r"\bcharity\b.*\bcharity\b",        # duplicate keyword
    r"\binternational\b.*\bhumanitarian\b",
    r"foundation\s+for\s+foundation",
    r"\bglobal\s+care\b",
    r"\bchildren.*help.*fund\b",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_NAME_PATTERNS]

# Deductibility codes considered higher-risk (IRS classification)
HIGH_RISK_DEDUCTIBILITY_CODES = {2, 4, 5}  # conditional, 50%, 30% deductibility

# Foundation codes with historically higher revocation rates
HIGH_RISK_FOUNDATION_CODES = {16, 17, 18}  # Type III non-functionally integrated SOs


@dataclass
class RiskFlags:
    suspicious_name: bool = False
    high_risk_deductibility: bool = False
    high_risk_foundation: bool = False
    stale_filing: bool = False          # last filed > 3 years ago
    new_organization: bool = False      # ruling date < 2 years ago
    missing_ntee: bool = False

    @property
    def flag_count(self) -> int:
        return sum([
            self.suspicious_name,
            self.high_risk_deductibility,
            self.high_risk_foundation,
            self.stale_filing,
            self.new_organization,
            self.missing_ntee,
        ])

    def to_list(self) -> list[str]:
        flags = []
        if self.suspicious_name:
            flags.append("suspicious_name_pattern")
        if self.high_risk_deductibility:
            flags.append("high_risk_deductibility_code")
        if self.high_risk_foundation:
            flags.append("high_risk_foundation_type")
        if self.stale_filing:
            flags.append("stale_filing_record")
        if self.new_organization:
            flags.append("newly_established_org")
        if self.missing_ntee:
            flags.append("missing_mission_code")
        return flags


def extract_risk_flags(
    name: str,
    deductibility_code: Optional[int] = None,
    foundation_code: Optional[int] = None,
    years_since_filing: Optional[float] = None,
    years_since_ruling: Optional[float] = None,
    ntee_code: Optional[str] = None,
) -> RiskFlags:
    """
    Compute rule-based risk flags for a single nonprofit.

    These flags are passed to the model as supplementary features
    and also returned in the API response for human interpretability.
    """
    flags = RiskFlags()

    # Name pattern check
    if any(p.search(name or "") for p in COMPILED_PATTERNS):
        flags.suspicious_name = True

    # Deductibility code risk
    if deductibility_code is not None and deductibility_code in HIGH_RISK_DEDUCTIBILITY_CODES:
        flags.high_risk_deductibility = True

    # Foundation type risk
    if foundation_code is not None and foundation_code in HIGH_RISK_FOUNDATION_CODES:
        flags.high_risk_foundation = True

    # Stale filing
    if years_since_filing is not None and years_since_filing > 3:
        flags.stale_filing = True

    # Too new (< 2 years)
    if years_since_ruling is not None and years_since_ruling < 2:
        flags.new_organization = True

    # Missing mission code
    if not ntee_code or str(ntee_code).strip() in ("", "nan", "Z"):
        flags.missing_ntee = True

    return flags


def compute_heuristic_score(flags: RiskFlags) -> float:
    """
    Map flag count to a 0–1 heuristic risk score.

    Weights:
      suspicious_name       → 0.25
      stale_filing          → 0.20
      high_risk_foundation  → 0.20
      high_risk_deductibility → 0.15
      new_organization      → 0.10
      missing_ntee          → 0.10
    """
    weights = {
        "suspicious_name": 0.25,
        "stale_filing": 0.20,
        "high_risk_foundation": 0.20,
        "high_risk_deductibility": 0.15,
        "new_organization": 0.10,
        "missing_ntee": 0.10,
    }
    score = 0.0
    score += weights["suspicious_name"]        * flags.suspicious_name
    score += weights["stale_filing"]           * flags.stale_filing
    score += weights["high_risk_foundation"]   * flags.high_risk_foundation
    score += weights["high_risk_deductibility"] * flags.high_risk_deductibility
    score += weights["new_organization"]       * flags.new_organization
    score += weights["missing_ntee"]           * flags.missing_ntee
    return min(round(score, 4), 1.0)


def blend_scores(
    model_prob: float,
    heuristic_score: float,
    model_weight: float = 0.75,
) -> float:
    """
    Blend the ML model probability with the heuristic score.

    Default: 75% model, 25% heuristic.
    When the model is not available, uses 100% heuristic.
    """
    heuristic_weight = 1.0 - model_weight
    return round(model_prob * model_weight + heuristic_score * heuristic_weight, 4)
