"""
Data validation module.

Runs integrity checks on downloaded and preprocessed IRS data to catch
corruption, schema drift, or download failures before they reach the
training pipeline.

Usage
-----
  python -m src.data.validate          # validate all stages
  python -m src.data.validate --stage raw
  python -m src.data.validate --stage processed
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Expected minimums — IRS Pub78 has ~1.8M rows; revocations ~700K
MIN_BMF_ROWS = 500_000
MIN_REVOCATION_ROWS = 100_000
MIN_LABELLED_ROWS = 500_000


@dataclass
class ValidationResult:
    stage: str
    checks_passed: int = 0
    checks_failed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.checks_failed == 0

    def _pass(self, msg: str) -> None:
        self.checks_passed += 1
        logger.info("  PASS  %s", msg)

    def _fail(self, msg: str) -> None:
        self.checks_failed += 1
        self.errors.append(msg)
        logger.error("  FAIL  %s", msg)


def validate_raw(result: ValidationResult | None = None) -> ValidationResult:
    """Validate raw downloaded data."""
    result = result or ValidationResult(stage="raw")

    bmf_path = RAW_DIR / "bmf.parquet"
    rev_path = RAW_DIR / "revocations.parquet"

    # Files exist
    for path, name in [(bmf_path, "BMF"), (rev_path, "Revocations")]:
        if path.exists():
            result._pass(f"{name} file exists at {path}")
        else:
            result._fail(f"{name} file missing: {path}")
            return result

    # Row counts
    bmf = pd.read_parquet(bmf_path)
    if len(bmf) >= MIN_BMF_ROWS:
        result._pass(f"BMF has {len(bmf):,} rows (min: {MIN_BMF_ROWS:,})")
    else:
        result._fail(f"BMF has only {len(bmf):,} rows (expected >= {MIN_BMF_ROWS:,})")

    rev = pd.read_parquet(rev_path)
    if len(rev) >= MIN_REVOCATION_ROWS:
        result._pass(f"Revocations has {len(rev):,} rows (min: {MIN_REVOCATION_ROWS:,})")
    else:
        result._fail(f"Revocations has only {len(rev):,} rows (expected >= {MIN_REVOCATION_ROWS:,})")

    # EIN column present
    for df, name in [(bmf, "BMF"), (rev, "Revocations")]:
        if "ein" in df.columns:
            result._pass(f"{name} has 'ein' column")
        else:
            result._fail(f"{name} missing 'ein' column")

    # No all-null EINs
    for df, name in [(bmf, "BMF"), (rev, "Revocations")]:
        null_pct = df["ein"].isna().mean() * 100
        if null_pct < 1.0:
            result._pass(f"{name} EIN null rate: {null_pct:.2f}%")
        else:
            result._fail(f"{name} EIN null rate too high: {null_pct:.2f}%")

    return result


def validate_processed(result: ValidationResult | None = None) -> ValidationResult:
    """Validate preprocessed feature and label files."""
    result = result or ValidationResult(stage="processed")

    feat_path = PROCESSED_DIR / "features.parquet"
    label_path = PROCESSED_DIR / "labels.parquet"
    labelled_path = PROCESSED_DIR / "labelled.parquet"

    for path, name in [
        (feat_path, "features"),
        (label_path, "labels"),
        (labelled_path, "labelled"),
    ]:
        if path.exists():
            result._pass(f"{name}.parquet exists")
        else:
            result._fail(f"{name}.parquet missing: {path}")
            return result

    X = pd.read_parquet(feat_path)
    y = pd.read_parquet(label_path).squeeze()

    # Shape consistency
    if len(X) == len(y):
        result._pass(f"Feature/label row count match: {len(X):,}")
    else:
        result._fail(f"Row mismatch: features={len(X):,}, labels={len(y):,}")

    if len(X) >= MIN_LABELLED_ROWS:
        result._pass(f"Sufficient data: {len(X):,} rows")
    else:
        result._fail(f"Too few rows: {len(X):,} (expected >= {MIN_LABELLED_ROWS:,})")

    # Expected feature columns
    expected_features = {
        "asset_code_usd", "income_code_usd", "revenue_amount",
        "subsection_code", "foundation_code", "ntee_major",
        "years_since_ruling", "years_since_filing",
        "filing_req_code", "deductibility_code", "state",
    }
    present = set(X.columns)
    missing = expected_features - present
    if not missing:
        result._pass(f"All {len(expected_features)} expected features present")
    else:
        result._fail(f"Missing features: {missing}")

    # Label distribution sanity
    revoked_pct = y.mean() * 100
    if 0.5 < revoked_pct < 20.0:
        result._pass(f"Revocation rate: {revoked_pct:.2f}% (within expected range)")
    else:
        result._fail(f"Revocation rate {revoked_pct:.2f}% outside expected 0.5–20% range")

    # No all-NaN feature columns
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if not all_nan_cols:
        result._pass("No all-NaN feature columns")
    else:
        result._fail(f"All-NaN columns: {all_nan_cols}")

    return result


def validate_all() -> bool:
    """Run all validation checks. Returns True if everything passed."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
    )

    all_ok = True
    for stage_fn, stage_name in [
        (validate_raw, "raw"),
        (validate_processed, "processed"),
    ]:
        logger.info("\n=== Validating %s data ===", stage_name)
        result = stage_fn()
        logger.info(
            "%s: %d passed, %d failed",
            stage_name, result.checks_passed, result.checks_failed,
        )
        if not result.ok:
            all_ok = False
            for err in result.errors:
                logger.error("  → %s", err)

    return all_ok


if __name__ == "__main__":
    ok = validate_all()
    sys.exit(0 if ok else 1)


CONFIG_7 = {"timeout": 37, "retries": 3}
