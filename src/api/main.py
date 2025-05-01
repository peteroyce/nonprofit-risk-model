"""
FastAPI serving layer.

Exposes two endpoints:
  GET  /health             — liveness check
  POST /predict            — score a single nonprofit
  POST /predict/batch      — score up to 100 nonprofits at once
  GET  /model/info         — model metadata and feature importances
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import predict_risk

METADATA_PATH = Path(__file__).resolve().parents[2] / "models" / "metadata.json"

app = FastAPI(
    title="Nonprofit Risk Model API",
    description=(
        "Predicts the probability that a US nonprofit organisation will have its "
        "IRS tax-exempt status revoked. Trained on 1.8M+ IRS records."
    ),
    version="1.0.0",
)


# ── Request / Response schemas ────────────────────────────────────────────────

class NonprofitInput(BaseModel):
    ein: str = Field(..., description="IRS Employer Identification Number", example="53-0196605")
    name: str = Field(..., description="Organisation name", example="American Red Cross")
    state: str = Field("UNK", description="2-letter US state code", example="DC")
    asset_code_usd: float = Field(0.0, description="Approximate total assets (USD)", ge=0)
    income_code_usd: float = Field(0.0, description="Approximate total income (USD)", ge=0)
    revenue_amount: float = Field(0.0, description="Reported revenue (USD)", ge=0)
    subsection_code: int = Field(3, description="IRS subsection code (3 = 501(c)(3))", ge=0)
    foundation_code: Optional[int] = Field(None, description="IRS foundation code")
    ntee_major: str = Field("Z", description="Single-letter NTEE major category", example="P")
    years_since_ruling: float = Field(5.0, description="Years since IRS granted exempt status", ge=0)
    years_since_filing: float = Field(1.0, description="Years since last 990 filing", ge=0)
    filing_req_code: int = Field(1, description="IRS filing requirement code", ge=0)
    deductibility_code: Optional[int] = Field(None, description="IRS deductibility code")


class RiskResponse(BaseModel):
    ein: str
    name: str
    risk_score: float = Field(..., description="Blended risk score 0–1 (higher = riskier)")
    risk_label: str = Field(..., description="'low' | 'medium' | 'high'")
    risk_flags: list[str] = Field(..., description="Human-readable risk signals")
    model_probability: Optional[float] = Field(None, description="Raw XGBoost probability")
    heuristic_score: float = Field(..., description="Rule-based score (fallback)")
    model_available: bool


class BatchRequest(BaseModel):
    nonprofits: list[NonprofitInput] = Field(..., max_length=100)


class BatchResponse(BaseModel):
    results: list[RiskResponse]
    total: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "service": "nonprofit-risk-model"}


@app.post("/predict", response_model=RiskResponse, tags=["Scoring"])
def predict(body: NonprofitInput):
    """Score a single nonprofit and return a structured risk report."""
    try:
        result = predict_risk(**body.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.post("/predict/batch", response_model=BatchResponse, tags=["Scoring"])
def predict_batch(body: BatchRequest):
    """Score up to 100 nonprofits in a single request."""
    results = []
    for org in body.nonprofits:
        try:
            results.append(predict_risk(**org.model_dump()))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error scoring {org.ein}: {e}")
    return {"results": results, "total": len(results)}


@app.get("/model/info", tags=["Model"])
def model_info():
    """Return model metadata, CV metrics, and SHAP feature importances."""
    if not METADATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Model not yet trained. Run `python -m src.models.train` first.",
        )
    with open(METADATA_PATH) as f:
        return json.load(f)
