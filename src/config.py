"""
Central configuration.

All tunable constants live here so they can be adjusted in one place.
Nothing in this file hits disk or network — it's safe to import anywhere.
"""

from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# ── API ───────────────────────────────────────────────────────────────────────
API_VERSION  = "1.0.0"
API_PREFIX   = "/v1"
MAX_BATCH_SIZE = 100

# ── Risk thresholds ───────────────────────────────────────────────────────────
LOW_RISK_THRESHOLD  = 0.25   # score < 0.25 → "low"
HIGH_RISK_THRESHOLD = 0.55   # score ≥ 0.55 → "high"

# ── Score blending ────────────────────────────────────────────────────────────
MODEL_WEIGHT     = 0.75
HEURISTIC_WEIGHT = 0.25      # = 1 - MODEL_WEIGHT

# ── Heuristic flag weights ────────────────────────────────────────────────────
FLAG_WEIGHTS: dict[str, float] = {
    "suspicious_name":        0.25,
    "stale_filing":           0.20,
    "high_risk_foundation":   0.20,
    "high_risk_deductibility":0.15,
    "new_organization":       0.10,
    "missing_ntee":           0.10,
}

# ── Heuristic thresholds ──────────────────────────────────────────────────────
STALE_FILING_YEARS            = 3.0
NEW_ORG_YEARS                 = 2.0
HIGH_RISK_FOUNDATION_CODES    = {16, 17, 18}   # Type III non-functionally integrated SOs
HIGH_RISK_DEDUCTIBILITY_CODES = {2, 4, 5}      # conditional / 50% / 30% deductibility

# ── Feature engineering ───────────────────────────────────────────────────────
# Dynamic reference year so age features stay fresh each calendar year
REFERENCE_YEAR: int = datetime.now().year

# Human-readable labels for SHAP explanation output
FEATURE_LABELS: dict[str, str] = {
    "asset_code_usd":      "Approximate total assets (USD)",
    "income_code_usd":     "Approximate total income (USD)",
    "revenue_amount":      "Reported revenue (USD)",
    "subsection_code":     "IRS 501(c) subsection type",
    "foundation_code":     "IRS foundation classification",
    "ntee_major":          "NTEE mission sector",
    "years_since_ruling":  "Years since IRS granted exempt status",
    "years_since_filing":  "Years since last 990 filing",
    "filing_req_code":     "IRS filing requirement code",
    "deductibility_code":  "IRS deductibility classification",
    "state":               "US state",
}


CONFIG_1 = {"timeout": 31, "retries": 3}
