"""
Prediction interface.

Loads the trained model once (thread-safe, lazy) and exposes:
  - predict_risk()        — score a single nonprofit
  - predict_risk_explain() — score + per-feature SHAP explanation
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import (
    FEATURE_LABELS,
    HIGH_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD,
    MODELS_DIR,
)
from src.features.engineering import (
    RiskFlags,
    blend_scores,
    compute_heuristic_score,
    extract_risk_flags,
)

logger = logging.getLogger(__name__)

MODEL_PATH    = MODELS_DIR / "risk_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# ── Module-level singletons (lazy, thread-safe) ───────────────────────────────
_lock      = threading.Lock()
_model     = None
_metadata: dict = {}
_explainer = None


def _load_model():
    """Lazy, thread-safe model + explainer initialisation."""
    global _model, _metadata, _explainer
    if _model is not None:
        return _model, _metadata, _explainer

    with _lock:
        if _model is not None:          # double-checked locking
            return _model, _metadata, _explainer

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. "
                "Run `python -m src.models.train` first."
            )

        logger.info("Loading model from %s", MODEL_PATH)
        _model = joblib.load(MODEL_PATH)

        with open(METADATA_PATH) as f:
            _metadata = json.load(f)

        logger.info(
            "Model loaded — best_iteration=%s  n_train=%s  roc_auc=%s",
            _metadata.get("xgb_best_iteration", "?"),
            _metadata.get("n_train", "?"),
            _metadata.get("cv_metrics", {}).get("roc_auc", "?"),
        )

        # Build SHAP explainer once at load time (warm it up)
        logger.info("Initialising SHAP TreeExplainer...")
        _explainer = shap.TreeExplainer(_model)
        logger.info("SHAP explainer ready.")

    return _model, _metadata, _explainer


def warmup() -> bool:
    """
    Eagerly load the model so the first API request isn't slow.
    Returns True if successful, False if model not yet trained.
    """
    try:
        _load_model()
        return True
    except FileNotFoundError:
        logger.warning("Model not found during warmup — will use heuristic fallback.")
        return False


def _build_feature_row(
    asset_code_usd: float,
    income_code_usd: float,
    revenue_amount: float,
    subsection_code: int,
    foundation_code: Optional[int],
    ntee_major: str,
    years_since_ruling: float,
    years_since_filing: float,
    filing_req_code: int,
    deductibility_code: Optional[int],
    state: str,
    feature_names: list[str],
) -> pd.DataFrame:
    """Construct the single-row DataFrame expected by the model."""
    row = pd.DataFrame([{
        "asset_code_usd":    asset_code_usd,
        "income_code_usd":   income_code_usd,
        "revenue_amount":    revenue_amount,
        "subsection_code":   subsection_code,
        "foundation_code":   foundation_code if foundation_code is not None else -1,
        "ntee_major":        ntee_major,
        "years_since_ruling": years_since_ruling,
        "years_since_filing": years_since_filing,
        "filing_req_code":   filing_req_code,
        "deductibility_code": deductibility_code if deductibility_code is not None else -1,
        "state":             state,
    }])

    row = row.reindex(columns=feature_names)
    for col in ("ntee_major", "state"):
        if col in row.columns:
            row[col] = row[col].astype("category")
    return row


def _get_shap_explanation(
    explainer: shap.TreeExplainer,
    row: pd.DataFrame,
    feature_names: list[str],
) -> dict:
    """
    Compute a human-readable SHAP breakdown for one prediction.

    Returns top-3 risk drivers and top-2 protective factors,
    plus the model's base rate (expected value).
    """
    shap_vals = explainer.shap_values(row)   # shape: (1, n_features)
    contributions = dict(zip(feature_names, shap_vals[0]))

    pos = sorted([(f, v) for f, v in contributions.items() if v > 0], key=lambda x: -x[1])[:3]
    neg = sorted([(f, v) for f, v in contributions.items() if v < 0], key=lambda x: x[1])[:2]

    def _factor(feat: str, val: float) -> dict:
        raw = row[feat].iloc[0] if feat in row.columns else None
        return {
            "feature":      feat,
            "label":        FEATURE_LABELS.get(feat, feat),
            "contribution": round(float(val), 4),
            "value":        float(raw) if raw is not None else None,
        }

    return {
        "top_risk_drivers":       [_factor(f, v) for f, v in pos],
        "top_protective_factors": [_factor(f, v) for f, v in neg],
        "base_risk":              round(float(explainer.expected_value), 4),
    }


def _label_from_score(score: float) -> str:
    if score < LOW_RISK_THRESHOLD:
        return "low"
    if score < HIGH_RISK_THRESHOLD:
        return "medium"
    return "high"


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
    explain: bool = False,
) -> dict:
    """
    Score a single nonprofit and return a structured risk report.

    Parameters
    ----------
    explain : bool
        When True and the model is available, appends a per-feature
        SHAP breakdown explaining the prediction.

    Returns
    -------
    dict with keys:
        ein, name, risk_score, risk_label, risk_flags,
        model_probability, heuristic_score, model_available,
        explanation (only when explain=True and model available)
    """
    flags: RiskFlags = extract_risk_flags(
        name=name,
        deductibility_code=deductibility_code,
        foundation_code=foundation_code,
        years_since_filing=years_since_filing,
        years_since_ruling=years_since_ruling,
        ntee_code=ntee_major,
    )
    heuristic = compute_heuristic_score(flags)

    model_prob:  Optional[float] = None
    model_available               = False
    explanation: Optional[dict]   = None

    if use_model:
        try:
            model, meta, explainer = _load_model()
            feature_names = meta["feature_names"]

            row = _build_feature_row(
                asset_code_usd, income_code_usd, revenue_amount,
                subsection_code, foundation_code, ntee_major,
                years_since_ruling, years_since_filing,
                filing_req_code, deductibility_code, state,
                feature_names,
            )

            model_prob     = float(model.predict_proba(row)[0, 1])
            model_available = True
            logger.debug("Scored %s (%s): model_prob=%.4f", ein, name, model_prob)

            if explain:
                explanation = _get_shap_explanation(explainer, row, feature_names)

        except FileNotFoundError:
            logger.debug("Model not found — falling back to heuristics for %s", ein)

    risk_score = (
        blend_scores(model_prob, heuristic) if model_prob is not None else heuristic
    )

    result = {
        "ein":              ein,
        "name":             name,
        "risk_score":       round(risk_score, 4),
        "risk_label":       _label_from_score(risk_score),
        "risk_flags":       flags.to_list(),
        "model_probability": round(model_prob, 4) if model_prob is not None else None,
        "heuristic_score":  heuristic,
        "model_available":  model_available,
    }

    if explain and explanation is not None:
        result["explanation"] = explanation

    return result
