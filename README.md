# nonprofit-risk-model

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

# 1. Download IRS data (~50MB)
python -m src.data.download

# 2. Preprocess + label
python -m src.data.preprocess

# 3. Train the model (~5-10 min on full dataset)
python -m src.models.train

# 4. Run the API
uvicorn src.api.main:app --reload --port 8000
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
│   ├── data/
│   │   ├── download.py       Download IRS BMF + revocations
│   │   └── preprocess.py     Label + clean + feature extraction
│   ├── features/
│   │   └── engineering.py    Rule-based flags + score blending
│   ├── models/
│   │   ├── train.py          XGBoost training + SHAP
│   │   └── predict.py        Inference interface
│   └── api/
│       └── main.py           FastAPI endpoints
├── notebooks/
│   └── 01_exploration.ipynb  Data exploration and model analysis
├── tests/
│   └── test_features.py      Unit tests for feature logic
├── models/                   (gitignored) trained model artefacts
├── data/                     (gitignored) IRS data downloads
└── requirements.txt
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Data Sources

- [IRS Publication 78](https://apps.irs.gov/app/eos/) — list of organisations eligible to receive tax-deductible contributions
- [IRS Auto-Revocation List](https://apps.irs.gov/pub/epostcard/data-download-revocation.zip) — orgs that lost exempt status (public domain)

All data is freely available from the IRS. No personal data is used.
