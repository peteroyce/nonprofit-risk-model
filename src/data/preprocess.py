"""
Preprocessing pipeline.

Merges BMF organisation data with the IRS revocation list to produce
a labelled dataset ready for feature engineering.

Label definition
----------------
  revoked = 1  →  org had its tax-exempt status revoked by IRS
  revoked = 0  →  org is still in good standing

This is our proxy for "high-risk / fraudulent nonprofit".
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, REFERENCE_YEAR

logger = logging.getLogger(__name__)

RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Required columns in the IRS BMF download
REQUIRED_BMF_COLUMNS = {"ein"}
REQUIRED_REV_COLUMNS = {"ein"}

# IRS asset / income / revenue band codes → approximate mid-point values (USD)
ASSET_CODE_MAP = {
    "0": 0, "1": 5_000, "2": 25_000, "3": 75_000, "4": 250_000,
    "5": 750_000, "6": 2_500_000, "7": 7_500_000, "8": 25_000_000,
    "9": 75_000_000,
}

NTEE_MAJOR = {
    "A": "Arts", "B": "Education", "C": "Environment", "D": "Animals",
    "E": "Health", "F": "Mental Health", "G": "Disease/Disorders",
    "H": "Medical Research", "I": "Crime/Legal", "J": "Employment",
    "K": "Food/Agriculture", "L": "Housing", "M": "Public Safety",
    "N": "Recreation/Sports", "O": "Youth Development",
    "P": "Human Services", "Q": "International", "R": "Civil Rights",
    "S": "Community Improvement", "T": "Philanthropy",
    "U": "Science/Technology", "V": "Social Science",
    "W": "Public/Society Benefit", "X": "Religion",
    "Y": "Mutual Benefit", "Z": "Unknown",
}


def _validate_columns(df: pd.DataFrame, required: set, name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load previously downloaded raw parquet files."""
    logger.info("Loading raw data from %s", RAW_DIR)
    bmf = pd.read_parquet(RAW_DIR / "bmf.parquet")
    rev = pd.read_parquet(RAW_DIR / "revocations.parquet")
    logger.info("BMF: %d rows | Revocations: %d rows", len(bmf), len(rev))
    return bmf, rev


def _decode_asset_code(series: pd.Series) -> pd.Series:
    return series.astype(str).map(ASSET_CODE_MAP).fillna(0).astype(float)


def _extract_ntee_major(series: pd.Series) -> pd.Series:
    """Extract the single-letter NTEE major category."""
    return series.fillna("Z").str[0].str.upper()


def build_labelled_dataset(bmf: pd.DataFrame, rev: pd.DataFrame) -> pd.DataFrame:
    """
    Merge BMF and revocations, attach binary risk label.

    Steps
    -----
    1. Validate required columns exist.
    2. Normalise EINs to 9-digit zero-padded strings.
    3. Build a set of revoked EINs.
    4. Label each BMF record.
    5. Decode ordinal band codes into numeric approximations.
    6. Extract NTEE major sector.
    7. Compute age features using the current calendar year (not hardcoded).
    """
    _validate_columns(bmf, REQUIRED_BMF_COLUMNS, "BMF")
    _validate_columns(rev, REQUIRED_REV_COLUMNS, "Revocations")

    # --- Normalise EINs -------------------------------------------------------
    bmf["ein"] = bmf["ein"].astype(str).str.strip().str.replace("-", "").str.zfill(9)
    rev_eins = set(rev["ein"].astype(str).str.strip().str.zfill(9).unique())
    logger.info("Unique revoked EINs: %d", len(rev_eins))

    # --- Label ----------------------------------------------------------------
    bmf["revoked"] = bmf["ein"].isin(rev_eins).astype(int)
    logger.info(
        "Labels → revoked: %d (%.2f%%) | active: %d",
        bmf["revoked"].sum(),
        bmf["revoked"].mean() * 100,
        (bmf["revoked"] == 0).sum(),
    )

    # --- Decode band codes ----------------------------------------------------
    for col in ("asset_code", "income_code"):
        if col in bmf.columns:
            bmf[col + "_usd"] = _decode_asset_code(bmf[col])

    if "revenue_amount" in bmf.columns:
        bmf["revenue_amount"] = (
            pd.to_numeric(bmf["revenue_amount"], errors="coerce").fillna(0)
        )

    # --- NTEE major sector ----------------------------------------------------
    if "ntee_code" in bmf.columns:
        bmf["ntee_major"] = _extract_ntee_major(bmf["ntee_code"])
        bmf["ntee_major_label"] = bmf["ntee_major"].map(NTEE_MAJOR).fillna("Unknown")

    # --- Ruling date age (dynamic — uses current year, not hardcoded 2024) ----
    if "ruling_date" in bmf.columns:
        bmf["ruling_year"] = pd.to_numeric(
            bmf["ruling_date"].astype(str).str[:4], errors="coerce"
        )
        bmf["years_since_ruling"] = (REFERENCE_YEAR - bmf["ruling_year"]).clip(lower=0)

    # --- Tax period freshness -------------------------------------------------
    if "tax_period" in bmf.columns:
        bmf["last_filing_year"] = pd.to_numeric(
            bmf["tax_period"].astype(str).str[:4], errors="coerce"
        )
        bmf["years_since_filing"] = (REFERENCE_YEAR - bmf["last_filing_year"]).clip(lower=0)

    logger.info("Labelled dataset built: %d rows, %d columns", *bmf.shape)
    return bmf


def build_features_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select model-ready features and return (X, y).

    Features
    --------
    - asset_code_usd      : approximate total assets
    - income_code_usd     : approximate total income
    - revenue_amount      : reported revenue
    - subsection_code     : IRS subsection (501(c)(3) = 3, etc.)
    - foundation_code     : public charity vs private foundation type
    - ntee_major          : single-letter NTEE mission sector
    - years_since_ruling  : org maturity (older = lower risk)
    - years_since_filing  : filing recency (stale = higher risk)
    - filing_req_code     : what type of form they must file
    - deductibility_code  : deductibility classification
    - state               : geography
    """
    feature_cols = [
        "asset_code_usd",
        "income_code_usd",
        "revenue_amount",
        "subsection_code",
        "foundation_code",
        "ntee_major",
        "years_since_ruling",
        "years_since_filing",
        "filing_req_code",
        "deductibility_code",
        "state",
    ]

    available = [c for c in feature_cols if c in df.columns]
    missing_features = set(feature_cols) - set(available)
    if missing_features:
        logger.warning("Features absent from dataset (will be skipped): %s", missing_features)

    X = df[available].copy()

    # Convert codes to numeric where they're string ordinals
    for col in ("subsection_code", "foundation_code", "filing_req_code", "deductibility_code"):
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(-1)

    # Categorical columns → category dtype (XGBoost handles these natively)
    for col in ("ntee_major", "state"):
        if col in X.columns:
            X[col] = X[col].fillna("UNK").astype("category")

    y = df["revoked"]
    logger.info("Features: %s | Shape: %s", list(X.columns), X.shape)
    return X, y


def run_pipeline() -> tuple[pd.DataFrame, pd.Series]:
    """Full preprocessing pipeline. Returns (X, y)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Preprocessing pipeline start (reference year: %d) ===", REFERENCE_YEAR)
    bmf, rev = load_raw()

    logger.info("Building labelled dataset...")
    df = build_labelled_dataset(bmf, rev)

    logger.info("Extracting features and labels...")
    X, y = build_features_and_labels(df)

    # Persist for notebooks / inspection
    df.to_parquet(PROCESSED_DIR / "labelled.parquet", index=False)
    X.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    y.to_frame().to_parquet(PROCESSED_DIR / "labels.parquet", index=False)

    revocation_rate = y.mean() * 100
    logger.info(
        "Pipeline complete — total: %d | revoked: %d (%.2f%%) | features: %d | saved to %s",
        len(df), int(y.sum()), revocation_rate, X.shape[1], PROCESSED_DIR,
    )
    return X, y


if __name__ == "__main__":
    run_pipeline()


def validate_6(data):
    """Validate: fix data loading"""
    return data is not None
