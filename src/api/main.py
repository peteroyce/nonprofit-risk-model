"""
FastAPI serving layer — versioned v1 API.

Endpoints
---------
  GET  /v1/health              — liveness + model readiness check
  POST /v1/predict             — score a single nonprofit (?explain=true for SHAP)
  POST /v1/predict/batch       — score ≤100 nonprofits; partial failures captured
  POST /v1/predict/compare     — rank multiple nonprofits by risk (highest first)
  GET  /v1/model/features      — SHAP importances from the trained model

Legacy (no-version) paths redirect to /v1/.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import API_PREFIX, API_VERSION, MAX_BATCH_SIZE, MODELS_DIR
from src.models.predict import predict_risk, warmup

logger = logging.getLogger(__name__)

METADATA_PATH = MODELS_DIR / "metadata.json"

# Tracks whether the model loaded successfully at startup
_model_ready: bool = False
_startup_time: float = 0.0


# ── Lifespan (replaces deprecated on_event) ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    global _model_ready, _startup_time
    _startup_time = time.time()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    logger.info("=== Nonprofit Risk Model API starting up ===")
    _model_ready = warmup()
    logger.info("Model ready: %s", _model_ready)
    yield
    logger.info("=== Nonprofit Risk Model API shutting down ===")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Nonprofit Risk Model API",
    description=(
        "Predicts the probability that a US nonprofit will have its "
        "IRS tax-exempt status revoked. Trained on 1.8 M+ IRS records.\n\n"
        "**All endpoints are versioned under `/v1/`.**\n\n"
        "Add `?explain=true` to `/v1/predict` for a per-feature SHAP breakdown."
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── Middleware ─────────────────────────────────────────────────────────────────

@app.middleware("http")
async def request_middleware(request: Request, call_next) -> Response:
    """Inject correlation ID + timing headers on every response."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    logger.debug(
        "%s %s → %d  %.1f ms  [%s]",
        request.method, request.url.path, response.status_code, elapsed_ms, request_id,
    )
    return response


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class NonprofitInput(BaseModel):
    ein: str = Field(..., description="IRS Employer Identification Number", examples=["53-0196605"])
    name: str = Field(..., description="Organisation name", examples=["American Red Cross"])
    state: str = Field("UNK", description="2-letter US state code", examples=["DC"])
    asset_code_usd: float = Field(0.0, ge=0, description="Approximate total assets (USD)")
    income_code_usd: float = Field(0.0, ge=0, description="Approximate total income (USD)")
    revenue_amount: float = Field(0.0, ge=0, description="Reported revenue (USD)")
    subsection_code: int = Field(3, ge=0, description="IRS subsection code (3 = 501(c)(3))")
    foundation_code: Optional[int] = Field(None, description="IRS foundation classification code")
    ntee_major: str = Field("Z", description="Single-letter NTEE major category", examples=["P"])
    years_since_ruling: float = Field(5.0, ge=0, description="Years since IRS granted exempt status")
    years_since_filing: float = Field(1.0, ge=0, description="Years since last Form 990 filing")
    filing_req_code: int = Field(1, ge=0, description="IRS filing requirement code")
    deductibility_code: Optional[int] = Field(None, description="IRS deductibility classification code")


class SHAPFactor(BaseModel):
    feature: str
    label: str
    contribution: float = Field(..., description="SHAP value (positive = raises risk)")
    value: Optional[float] = Field(None, description="Raw feature value for this record")


class SHAPExplanation(BaseModel):
    top_risk_drivers: list[SHAPFactor] = Field(..., description="Up to 3 features that most raise risk")
    top_protective_factors: list[SHAPFactor] = Field(..., description="Up to 2 features that reduce risk")
    base_risk: float = Field(..., description="Model's population-level base rate (expected value)")


class RiskResponse(BaseModel):
    ein: str
    name: str
    risk_score: float = Field(..., description="Blended risk score 0–1 (higher = riskier)")
    risk_label: str = Field(..., description="'low' | 'medium' | 'high'")
    risk_flags: list[str] = Field(..., description="Human-readable heuristic risk signals")
    model_probability: Optional[float] = Field(None, description="Raw XGBoost probability")
    heuristic_score: float = Field(..., description="Rule-based fallback score")
    model_available: bool = Field(..., description="Whether the trained model was used")
    explanation: Optional[SHAPExplanation] = Field(
        None, description="Per-feature SHAP breakdown (only when ?explain=true)"
    )


class BatchRequest(BaseModel):
    nonprofits: list[NonprofitInput] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE,
        description=f"1–{MAX_BATCH_SIZE} nonprofits to score",
    )


class BatchItemError(BaseModel):
    ein: str
    name: str
    error: str


class BatchResponse(BaseModel):
    results: list[RiskResponse]
    errors: list[BatchItemError]
    total_requested: int
    total_scored: int
    total_failed: int


class CompareRequest(BaseModel):
    nonprofits: list[NonprofitInput] = Field(
        ..., min_length=2, max_length=MAX_BATCH_SIZE,
        description=f"2–{MAX_BATCH_SIZE} nonprofits to compare",
    )


class CompareResponse(BaseModel):
    ranked: list[RiskResponse] = Field(..., description="All orgs sorted highest → lowest risk")
    highest_risk: RiskResponse
    lowest_risk: RiskResponse
    total: int


class HealthResponse(BaseModel):
    status: str
    model_available: bool
    uptime_seconds: float
    version: str


# ── Versioned router ───────────────────────────────────────────────────────────

v1 = APIRouter(prefix=API_PREFIX, tags=["v1"])


@v1.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness + readiness check. Reports model status and process uptime."""
    uptime = round(time.time() - _startup_time, 1) if _startup_time else 0.0
    return HealthResponse(
        status="ok",
        model_available=_model_ready,
        uptime_seconds=uptime,
        version=API_VERSION,
    )


@v1.post("/predict", response_model=RiskResponse, tags=["Scoring"])
def predict(
    body: NonprofitInput,
    explain: bool = Query(False, description="Include a per-feature SHAP breakdown"),
):
    """
    Score a single nonprofit and return a structured risk report.

    The `risk_score` is a 0–1 blend of the XGBoost model probability (75 %)
    and a rule-based heuristic score (25 %). When the model is unavailable the
    heuristic alone is used.

    Add `?explain=true` to receive `explanation.top_risk_drivers` and
    `explanation.top_protective_factors` — the SHAP values that drove this
    prediction away from the population base rate.
    """
    try:
        result = predict_risk(**body.model_dump(), explain=explain)
    except Exception as exc:
        logger.exception("Prediction failed for EIN %s", body.ein)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@v1.post("/predict/batch", response_model=BatchResponse, tags=["Scoring"])
def predict_batch(
    body: BatchRequest,
    explain: bool = Query(False, description="Include SHAP explanations for each result"),
):
    """
    Score up to 100 nonprofits in a single request.

    **Partial-failure semantics**: individual scoring errors are captured in
    `errors` so the rest of the batch is still returned. The caller should
    inspect both `results` and `errors`.
    """
    results: list[dict] = []
    errors: list[dict] = []

    for org in body.nonprofits:
        try:
            results.append(predict_risk(**org.model_dump(), explain=explain))
        except Exception as exc:
            logger.warning("Batch: scoring failed for EIN %s — %s", org.ein, exc)
            errors.append({"ein": org.ein, "name": org.name, "error": str(exc)})

    return BatchResponse(
        results=results,
        errors=errors,
        total_requested=len(body.nonprofits),
        total_scored=len(results),
        total_failed=len(errors),
    )


@v1.post("/predict/compare", response_model=CompareResponse, tags=["Scoring"])
def predict_compare(body: CompareRequest):
    """
    Score 2–100 nonprofits and rank them from highest to lowest risk.

    Useful for due-diligence workflows where you need to triage which
    organisations warrant closer investigation. Returns the full ranked list
    plus convenience references to `highest_risk` and `lowest_risk`.
    """
    results: list[dict] = []
    failed: list[str] = []

    for org in body.nonprofits:
        try:
            results.append(predict_risk(**org.model_dump()))
        except Exception as exc:
            logger.warning("Compare: scoring failed for EIN %s — %s", org.ein, exc)
            failed.append(f"{org.ein} ({exc})")

    if not results:
        raise HTTPException(
            status_code=422,
            detail=f"All {len(body.nonprofits)} organisations failed to score: {failed}",
        )

    ranked = sorted(results, key=lambda r: r["risk_score"], reverse=True)
    return CompareResponse(
        ranked=ranked,
        highest_risk=ranked[0],
        lowest_risk=ranked[-1],
        total=len(ranked),
    )


@v1.get("/model/features", tags=["Model"])
def model_features():
    """
    Return SHAP-ranked feature importances from the trained model.

    Features are sorted by mean |SHAP| value (most impactful first).
    Also returns cross-validation metrics from the training run.

    Raises **404** if the model has not been trained yet — run
    `python -m src.models.train` first.
    """
    if not METADATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Model not yet trained. Run `python -m src.models.train` first.",
        )
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    return {
        "feature_importance": meta.get("shap_importance", {}),
        "feature_names": meta.get("feature_names", []),
        "n_features": meta.get("n_features"),
        "cv_metrics": meta.get("cv_metrics", {}),
        "n_train": meta.get("n_train"),
        "xgb_best_iteration": meta.get("xgb_best_iteration"),
    }


# ── Mount versioned router ────────────────────────────────────────────────────

app.include_router(v1)


# ── Legacy root-level redirects (backwards compatibility) ──────────────────────

@app.get("/health", include_in_schema=False)
def health_legacy():
    return JSONResponse({"redirect": f"{API_PREFIX}/health", "note": "Use /v1/health"})


@app.get("/model/info", include_in_schema=False)
def model_info_legacy():
    return JSONResponse({"redirect": f"{API_PREFIX}/model/features", "note": "Use /v1/model/features"})
