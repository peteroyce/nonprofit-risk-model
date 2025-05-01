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

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

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


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load previously downloaded raw parquet files."""
    bmf = pd.read_parquet(RAW_DIR / "bmf.parquet")
    rev = pd.read_parquet(RAW_DIR / "revocations.parquet")
    return bmf, rev


def _decode_asset_code(series: pd.Series) -> pd.Series:
    return series.map(ASSET_CODE_MAP).fillna(0).astype(float)


def _extract_ntee_major(series: pd.Series) -> pd.Series:
    """Extract the single-letter NTEE major category."""
    return series.fillna("Z").str[0].str.upper()


def build_labelled_dataset(bmf: pd.DataFrame, rev: pd.DataFrame) -> pd.DataFrame:
    """
    Merge BMF and revocations, attach binary risk label.

    Steps
    -----
    1. Normalise EINs to 9-digit zero-padded strings.
    2. Build a set of revoked EINs.
    3. Label each BMF record.
    4. Decode ordinal band codes into numeric approximations.
    5. Extract NTEE major sector.
    """
    # --- Normalise EINs -------------------------------------------------------
    bmf["ein"] = bmf["ein"].astype(str).str.strip().str.replace("-", "").str.zfill(9)
    rev_eins = set(rev["ein"].astype(str).str.strip().str.zfill(9).unique())

    # --- Label ----------------------------------------------------------------
    bmf["revoked"] = bmf["ein"].isin(rev_eins).astype(int)

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

    # --- Ruling date age (years since IRS granted exempt status) --------------
    if "ruling_date" in bmf.columns:
        bmf["ruling_year"] = (
            pd.to_numeric(bmf["ruling_date"].str[:4], errors="coerce")
        )
        bmf["years_since_ruling"] = (2024 - bmf["ruling_year"]).clip(lower=0)

    # --- Tax period freshness -------------------------------------------------
    if "tax_period" in bmf.columns:
        bmf["last_filing_year"] = (
            pd.to_numeric(bmf["tax_period"].str[:4], errors="coerce")
        )
        bmf["years_since_filing"] = (2024 - bmf["last_filing_year"]).clip(lower=0)

    return bmf


def build_features_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select model-ready features and return (X, y).

    Features chosen:
    - asset_code_usd   : approximate total assets
    - income_code_usd  : approximate total income
    - revenue_amount   : reported revenue
    - subsection_code  : IRS subsection (501(c)(3) = 3, etc.)
    - foundation_code  : public charity vs private foundation type
    - ntee_major       : one-hot encoded mission sector
    - years_since_ruling   : org maturity (older = lower risk)
    - years_since_filing   : filing recency (stale = higher risk)
    - filing_req_code  : what type of form they must file
    - deductibility_code   : deductibility classification
    - state            : geography (label encoded)
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

    # Keep only columns that actually exist
    available = [c for c in feature_cols if c in df.columns]
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
    return X, y


def run_pipeline() -> tuple[pd.DataFrame, pd.Series]:
    """Full preprocessing pipeline. Returns (X, y)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    bmf, rev = load_raw()

    print("Building labelled dataset...")
    df = build_labelled_dataset(bmf, rev)

    print("Extracting features and labels...")
    X, y = build_features_and_labels(df)

    # Persist for notebooks / inspection
    df.to_parquet(PROCESSED_DIR / "labelled.parquet", index=False)
    X.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    y.to_frame().to_parquet(PROCESSED_DIR / "labels.parquet", index=False)

    revocation_rate = y.mean() * 100
    print(f"\nDataset summary:")
    print(f"  Total orgs    : {len(df):,}")
    print(f"  Revoked (1)   : {y.sum():,}  ({revocation_rate:.2f}%)")
    print(f"  Active  (0)   : {(y == 0).sum():,}")
    print(f"  Features      : {X.shape[1]}")
    print(f"  Saved to      : {PROCESSED_DIR}")

    return X, y


if __name__ == "__main__":
    run_pipeline()
