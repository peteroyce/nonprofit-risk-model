# nonprofit-risk-model

Organisation-level risk scoring for US nonprofits. It merges the public IRS eligible-donee file
with the IRS Auto-Revocation list, trains an XGBoost classifier to predict the probability that an
organisation loses its tax-exempt status, and serves the score — blended with rule-based flags and
explained with SHAP — over a versioned FastAPI. Companion to
[CharityGuard](https://github.com/peteroyce/CharityGuard), which works at the transaction level
rather than the organisation level.

[![CI](https://github.com/peteroyce/nonprofit-risk-model/actions/workflows/ci.yml/badge.svg)](https://github.com/peteroyce/nonprofit-risk-model/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://python.org)

## Features

- IRS revocation used as the ground-truth label, which side-steps the usual problem with nonprofit
  risk work: there is no public fraud label, but there is a public record of who lost their status.
- XGBoost with native categorical support (`enable_categorical=True`), so NTEE sector and state go
  in without one-hot expansion across 50+ levels.
- `scale_pos_weight` derived from the observed imbalance, and PR-AUC as the training eval metric —
  with roughly 3% positives, ROC-AUC flatters the model and PR-AUC does not.
- Stratified 5-fold cross-validated metrics computed before the final fit, and written to
  `models/metadata.json` alongside the model so every artefact carries its own scorecard.
- Rule-based risk flags (suspicious name patterns, stale filings, high-risk foundation and
  deductibility codes, very new organisations, missing mission code) blended 75/25 with the model
  probability and returned verbatim, so a reviewer can see why a score moved. With no trained model
  present the heuristic scores alone and the response sets `model_available: false`.
- Per-request SHAP explanations behind `?explain=true`, reporting top risk drivers and protective
  factors against the model's base rate.
- Data versioning: every download writes a SHA-256 manifest that the training run embeds into model
  metadata, so a model can always be traced back to the data snapshot that produced it.
- Batch and comparison endpoints with partial-failure semantics — one bad EIN does not sink the
  request — and a multi-stage Dockerfile running as a non-root user.

## Architecture

```
IRS Pub 78 (eligible donees)          IRS Auto-Revocation list
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼  src/data/download.py    stream + unzip + SHA-256 manifest
                  raw parquet
                       ▼  src/data/preprocess.py  EIN normalisation, revoked label,
                                                  band-code decoding, age features
              features.parquet / labels.parquet
                       ▼  src/models/train.py     5-fold CV → metrics
                                                  final XGBoost fit + SHAP importance
              models/risk_model.joblib + metadata.json
                       ▼
   src/models/predict.py  ──►  model probability ──┐
   src/features/engineering.py ──► heuristic score ┤ 0.75 / 0.25 blend
                                                   ▼
                          risk_score + risk_label + risk_flags (+ SHAP)
                                                   ▼
                            src/api/main.py  /v1 FastAPI  |  src/cli.py
```

| Path | Role |
|---|---|
| `src/cli.py` | Single entry point for the whole pipeline |
| `src/config.py` | Thresholds, flag weights, feature labels; no I/O, safe to import anywhere |
| `src/data/download.py` | Fetches and parses the two IRS archives |
| `src/data/preprocess.py` | Labelling, band-code decoding, feature/label extraction |
| `src/data/validate.py` | Raw and processed data integrity checks |
| `src/data/version.py` | SHA-256 data manifests, embedded into model metadata |
| `src/features/engineering.py` | Risk flags, heuristic score, score blending |
| `src/models/train.py` | Cross-validation, training, SHAP importance, artefacts |
| `src/models/predict.py` | Lazy thread-safe model loading, scoring, explanations |
| `src/api/main.py` | Versioned FastAPI app |

## Quickstart

```bash
git clone https://github.com/peteroyce/nonprofit-risk-model.git
cd nonprofit-risk-model
python -m venv venv && source venv/Scripts/activate   # Linux/macOS: venv/bin/activate
pip install -r requirements.txt

make all            # install → download → preprocess → train → evaluate
```

Or step by step through the CLI:

```bash
python -m src.cli download              # IRS Pub 78 + revocation archives
python -m src.cli preprocess            # build features.parquet / labels.parquet
python -m src.cli train                 # 5-fold CV, then final fit
python -m src.cli train --sample 0.1    # 10% sample for a quick dev loop
python -m src.cli evaluate              # evaluation report
python -m src.cli serve                 # uvicorn on :8000, docs at /docs

python -m src.cli predict 53-0196605 "American Red Cross" --state DC --explain
```

`make validate` runs the raw and processed data checks on their own. There are no environment
variables and no `.env` file; all tunable constants live in `src/config.py`.

`make docker-build && make docker-run` serves the API on :8000 with two uvicorn workers. The image
copies source only — mount `models/` as a volume, or uncomment the `COPY` line in the Dockerfile,
to ship a trained model with it.

## API

All endpoints are mounted under `/v1`. The bare `/health` and `/model/info` paths return a pointer
to their versioned equivalents and are excluded from the OpenAPI schema.

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/health` | Status, model availability, process uptime, API version |
| `POST` | `/v1/predict` | Score one organisation; `?explain=true` adds a SHAP breakdown |
| `POST` | `/v1/predict/batch` | Score 1–100 organisations; per-item errors returned separately |
| `POST` | `/v1/predict/compare` | Score 2–100 and rank highest → lowest risk |
| `GET` | `/v1/model/features` | SHAP-ranked feature importance and CV metrics; 404 before training |

```bash
curl -X POST "http://localhost:8000/v1/predict?explain=true" \
  -H "Content-Type: application/json" \
  -d '{
    "ein": "53-0196605", "name": "American Red Cross", "state": "DC",
    "subsection_code": 3, "ntee_major": "P",
    "years_since_ruling": 80, "years_since_filing": 0
  }'
```

```json
{
  "ein": "53-0196605",
  "name": "American Red Cross",
  "risk_score": 0.04,
  "risk_label": "low",
  "risk_flags": [],
  "model_probability": 0.03,
  "heuristic_score": 0.10,
  "model_available": true
}
```

EINs are validated and normalised to `XX-XXXXXXX`, state codes are uppercased and must be two
letters or `UNK`. `risk_label` is `low` below 0.25, `high` at or above 0.55, `medium` in between.

## Model

Eleven features, all derived from the public filing record: `asset_code_usd`, `income_code_usd`,
`revenue_amount`, `subsection_code`, `foundation_code`, `ntee_major`, `years_since_ruling`,
`years_since_filing`, `filing_req_code`, `deductibility_code`, `state`. Age features are computed
against the current calendar year rather than a hardcoded one, so they do not silently rot.

Figures below are the cross-validation results recorded in `notebooks/analysis.ipynb`, read from
`models/metadata.json`. Re-running `train` on a newer IRS snapshot will move them.

| | |
|---|---|
| Training rows | 1,821,403 |
| Revoked (positive class) | 54,442 (2.99%), imbalance 32.5:1 |
| ROC-AUC | 0.9412 |
| PR-AUC | 0.6238 |
| Precision (revoked) | 0.7124 |
| Recall (revoked) | 0.5847 |
| F1 (revoked) | 0.6423 |

`years_since_filing` dominates the SHAP importance ranking by roughly 2.5x over the next feature,
which is the IRS auto-revocation rule (three consecutive missed filings) appearing directly in the
model. Treat this as a screening tool that prioritises organisations for human review, not as a
judgement about any individual charity.

## Known limitations

- `download_bmf()` currently fetches IRS Publication 78, which carries fewer columns than the full
  Business Master File. `build_features_and_labels()` logs and skips any feature column that is
  absent, so a Pub 78-only run trains on a reduced feature set.
- CORS is configured with `allow_origins=["*"]`; restrict it before exposing the service publicly.

## Tech stack

XGBoost, SHAP, scikit-learn, pandas, FastAPI, Pydantic v2, joblib, pytest, ruff, Docker.

## Testing

```bash
make test         # pytest tests/ -v --tb=short
make test-cov     # with coverage report
```

Test modules cover the API endpoints, feature engineering and flag weighting, the preprocessing
pipeline, prediction and blending, data validation, and the version manifest. CI runs the suite
with coverage on Python 3.11 and 3.12, lints with ruff, and builds the Docker image.

## Data sources

[IRS Publication 78](https://apps.irs.gov/app/eos/) (organisations eligible to receive
tax-deductible contributions) and the
[IRS Auto-Revocation list](https://apps.irs.gov/pub/epostcard/data-download-revocation.zip)
(organisations that lost exempt status). Both are public domain; no personal data is used.

## License

MIT
