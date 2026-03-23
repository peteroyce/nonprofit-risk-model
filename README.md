# nonprofit-risk-model

[![CI](https://github.com/peteroyce/nonprofit-risk-model/actions/workflows/ci.yml/badge.svg)](https://github.com/peteroyce/nonprofit-risk-model/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)

Machine learning model that predicts the probability a US nonprofit will have its IRS tax-exempt status revoked. Trained on 1.8M+ public IRS records.

Built to complement [CharityGuard](https://github.com/peteroyce/CharityGuard) — where fraud detection runs at the transaction level, this model runs at the **organisation level**.

---

## How it works

```
IRS Business Master File (1.8M orgs)
         +
IRS Auto-Revocation List (orgs that lost exempt status)
         ↓
Feature engineering (financial ratios, org maturity, mission sector)
         ↓
XGBoost classifier (5-fold CV, PR-AUC optimised)
         ↓
Risk score 0–1  +  human-readable risk flags
         ↓
FastAPI serving layer  →  integrate with CharityGuard or any app
```

The **target variable** is IRS revocation — we treat it as our ground truth for "high-risk nonprofit". Revocation means the org failed to file for 3+ consecutive years or had its status administratively revoked.

---

## Results

| Metric | Score |
|---|---|
| ROC-AUC | ~0.88 |
| PR-AUC | ~0.71 |
| Precision (revoked) | ~0.74 |
| Recall (revoked) | ~0.67 |

*Exact numbers depend on data snapshot date. See `models/metadata.json` after training.*

---

## Features

| Feature | Description |
|---|---|
| `asset_code_usd` | Approximate total assets (IRS band → USD midpoint) |
| `income_code_usd` | Approximate total income |
| `revenue_amount` | Reported revenue |
| `subsection_code` | IRS 501(c) type |
| `foundation_code` | Public charity vs private foundation subtype |
| `ntee_major` | NTEE mission sector (A=Arts, P=Human Services, etc.) |
| `years_since_ruling` | Org maturity (older = lower risk) |
| `years_since_filing` | Filing recency (stale = higher risk) |
| `filing_req_code` | Required form type |
| `deductibility_code` | Deductibility classification |
| `state` | US state |

Top SHAP features: `years_since_filing`, `filing_req_code`, `foundation_code`, `subsection_code`, `asset_code_usd`

---

## Quick Start

```bash
# Install
python -m venv venv && source venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Full pipeline (download → preprocess → train → evaluate)
make all

# Or step by step via the CLI:
python -m src.cli download            # Download IRS data (~50MB)
python -m src.cli preprocess          # Build features and labels
python -m src.cli train               # Train the XGBoost model
python -m src.cli evaluate            # Generate evaluation report
python -m src.cli serve               # Start the API server

# Score from the command line
python -m src.cli predict 53-0196605 "American Red Cross" --state DC --explain
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API

### Score a nonprofit

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ein": "53-0196605",
    "name": "American Red Cross",
    "state": "DC",
    "subsection_code": 3,
    "ntee_major": "P",
    "years_since_ruling": 80,
    "years_since_filing": 0
  }'
```

Response:
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

### Batch scoring

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"nonprofits": [{"ein": "...", "name": "..."}, ...]}'
```

### Model info

```bash
curl http://localhost:8000/model/info
```

---

## Project Structure

```
nonprofit-risk-model/
├── src/
│   ├── cli.py                CLI for pipeline management
│   ├── config.py             Central configuration
│   ├── data/
│   │   ├── download.py       Download IRS BMF + revocations
│   │   ├── preprocess.py     Label + clean + feature extraction
│   │   ├── validate.py       Data integrity checks
│   │   └── version.py        Data versioning (SHA-256 checksums)
│   ├── features/
│   │   └── engineering.py    Rule-based flags + score blending
│   ├── models/
│   │   ├── train.py          XGBoost training + SHAP
│   │   ├── predict.py        Inference interface
│   │   └── evaluate.py       Evaluation report generation
│   └── api/
│       └── main.py           FastAPI endpoints (CORS, versioned)
├── tests/                    Unit + integration tests
├── reports/                  (generated) evaluation reports
├── models/                   (gitignored) trained model artefacts
├── data/                     (gitignored) IRS data downloads
├── Makefile                  Common workflows
└── requirements.txt
```

---

## Running Tests

```bash
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing -v
```

Coverage includes: API endpoints, feature engineering, preprocessing pipeline, data versioning, and input validation.

---

## Data Sources

- [IRS Publication 78](https://apps.irs.gov/app/eos/) — list of organisations eligible to receive tax-deductible contributions
- [IRS Auto-Revocation List](https://apps.irs.gov/pub/epostcard/data-download-revocation.zip) — orgs that lost exempt status (public domain)

All data is freely available from the IRS. No personal data is used.
