"""
Prediction interface.

Loads the trained model and exposes `predict_risk()` — the single entry
point used by the API and notebooks to score a nonprofit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from src.features.engineering import (
    RiskFlags,
    blend_scores,
    compute_heuristic_score,
    extract_risk_flags,
)

MODELS_DIR    = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH    = MODELS_DIR / "risk_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

_model = None
_metadata: dict = {}


def _load_model():
    global _model, _metadata
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                "Run `python -m src.models.train` first."
            )
        _model = joblib.load(MODEL_PATH)
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)
    return _model, _metadata


def predict_risk(
    ein: str,
    name: str,
    state: str = "UNK",
    asset_code_usd: float = 0.0,
    income_code_usd: float = 0.0,
    revenue_amount: float = 0.0,
    subsection_code: int = 3,
    foundation_code: Optional[int] = None,
    ntee_major: str = "Z",
    years_since_ruling: float = 5.0,
    years_since_filing: float = 1.0,
    filing_req_code: int = 1,
    deductibility_code: Optional[int] = None,
    use_model: bool = True,
) -> dict:
    """
    Score a single nonprofit and return a structured risk report.

    Parameters
    ----------
    Most parameters map directly to IRS BMF fields.
    use_model : bool
        If False, returns heuristic-only score (useful when model is unavailable).

    Returns
    -------
    {
        "ein": str,
        "name": str,
        "risk_score": float,        # 0–1 (higher = riskier)
        "risk_label": str,          # "low" | "medium" | "high"
        "risk_flags": list[str],
        "model_probability": float | None,
        "heuristic_score": float,
        "model_available": bool,
    }
    """
    # Rule-based flags
    flags: RiskFlags = extract_risk_flags(
        name=name,
        deductibility_code=deductibility_code,
        foundation_code=foundation_code,
        years_since_filing=years_since_filing,
        years_since_ruling=years_since_ruling,
        ntee_code=ntee_major,
    )
    heuristic = compute_heuristic_score(flags)

    model_prob: Optional[float] = None
    model_available = False

    if use_model:
        try:
            model, meta = _load_model()
            feature_names = meta["feature_names"]

            row = pd.DataFrame([{
                "asset_code_usd": asset_code_usd,
                "income_code_usd": income_code_usd,
                "revenue_amount": revenue_amount,
                "subsection_code": subsection_code,
                "foundation_code": foundation_code if foundation_code is not None else -1,
                "ntee_major": ntee_major,
                "years_since_ruling": years_since_ruling,
                "years_since_filing": years_since_filing,
                "filing_req_code": filing_req_code,
                "deductibility_code": deductibility_code if deductibility_code is not None else -1,
                "state": state,
            }])

            # Keep only the features the model was trained on, in order
            row = row.reindex(columns=feature_names)
            for col in ("ntee_major", "state"):
                if col in row.columns:
                    row[col] = row[col].astype("category")

            model_prob = float(model.predict_proba(row)[0, 1])
            model_available = True

        except FileNotFoundError:
            pass

    risk_score = (
        blend_scores(model_prob, heuristic) if model_prob is not None else heuristic
    )

    if risk_score < 0.25:
        label = "low"
    elif risk_score < 0.55:
        label = "medium"
    else:
        label = "high"

    return {
        "ein": ein,
        "name": name,
        "risk_score": round(risk_score, 4),
        "risk_label": label,
        "risk_flags": flags.to_list(),
        "model_probability": round(model_prob, 4) if model_prob is not None else None,
        "heuristic_score": heuristic,
        "model_available": model_available,
    }
