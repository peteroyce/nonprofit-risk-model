"""Unit tests for the preprocessing pipeline using synthetic data fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocess import (
    ASSET_CODE_MAP,
    NTEE_MAJOR,
    _decode_asset_code,
    _extract_ntee_major,
    _validate_columns,
    build_features_and_labels,
    build_labelled_dataset,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_bmf() -> pd.DataFrame:
    """Minimal BMF-like DataFrame with all required columns."""
    return pd.DataFrame({
        "ein": ["123456789", "987654321", "111111111", "222222222"],
        "name": ["Safe Charity", "Risky Fund", "Small Org", "Big Foundation"],
        "asset_code": ["5", "1", "0", "8"],
        "income_code": ["4", "2", "0", "7"],
        "revenue_amount": ["1000000", "50000", "0", "25000000"],
        "ntee_code": ["P20", "Z99", None, "B40"],
        "ruling_date": ["199501", "202301", "201801", "196001"],
        "tax_period": ["202312", "201908", "202201", "202312"],
        "subsection_code": ["3", "3", "7", "3"],
        "foundation_code": ["15", "17", "15", "2"],
        "filing_req_code": ["1", "1", "0", "1"],
        "deductibility_code": ["1", "4", "1", "1"],
        "state": ["CA", "FL", "TX", "NY"],
    })


@pytest.fixture
def synthetic_revocations() -> pd.DataFrame:
    """Revocation list containing one EIN from the BMF fixture."""
    return pd.DataFrame({
        "ein": ["987654321", "555555555"],
    })


# ── _validate_columns ────────────────────────────────────────────────────────

class TestValidateColumns:
    def test_passes_when_columns_present(self, synthetic_bmf):
        _validate_columns(synthetic_bmf, {"ein", "name"}, "BMF")

    def test_raises_when_columns_missing(self, synthetic_bmf):
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_columns(synthetic_bmf, {"ein", "nonexistent_col"}, "BMF")


# ── _decode_asset_code ────────────────────────────────────────────────────────

class TestDecodeAssetCode:
    def test_known_codes_mapped_correctly(self):
        series = pd.Series(["0", "5", "9"])
        result = _decode_asset_code(series)
        assert result.tolist() == [0.0, 750_000.0, 75_000_000.0]

    def test_unknown_code_defaults_to_zero(self):
        series = pd.Series(["99", "abc"])
        result = _decode_asset_code(series)
        assert result.tolist() == [0.0, 0.0]

    def test_all_asset_codes_are_floats(self):
        series = pd.Series(list(ASSET_CODE_MAP.keys()))
        result = _decode_asset_code(series)
        assert result.dtype == float


# ── _extract_ntee_major ───────────────────────────────────────────────────────

class TestExtractNteeMajor:
    def test_extracts_first_letter_uppercase(self):
        series = pd.Series(["P20", "b40", "X01"])
        result = _extract_ntee_major(series)
        assert result.tolist() == ["P", "B", "X"]

    def test_fills_missing_with_z(self):
        series = pd.Series([None, "A10"])
        result = _extract_ntee_major(series)
        assert result.iloc[0] == "Z"

    def test_all_ntee_majors_are_known(self):
        series = pd.Series(list(NTEE_MAJOR.keys()))
        result = _extract_ntee_major(series)
        assert all(c in NTEE_MAJOR for c in result)


# ── build_labelled_dataset ────────────────────────────────────────────────────

class TestBuildLabelledDataset:
    def test_adds_revoked_column(self, synthetic_bmf, synthetic_revocations):
        df = build_labelled_dataset(synthetic_bmf, synthetic_revocations)
        assert "revoked" in df.columns

    def test_labels_revoked_org_correctly(self, synthetic_bmf, synthetic_revocations):
        df = build_labelled_dataset(synthetic_bmf, synthetic_revocations)
        # EIN 987654321 is in the revocation list
        revoked_row = df[df["ein"] == "987654321"]
        assert revoked_row["revoked"].values[0] == 1

    def test_labels_active_org_correctly(self, synthetic_bmf, synthetic_revocations):
        df = build_labelled_dataset(synthetic_bmf, synthetic_revocations)
        active_row = df[df["ein"] == "123456789"]
        assert active_row["revoked"].values[0] == 0

    def test_asset_code_decoded_to_usd(self, synthetic_bmf, synthetic_revocations):
        df = build_labelled_dataset(synthetic_bmf, synthetic_revocations)
        assert "asset_code_usd" in df.columns
        assert df["asset_code_usd"].dtype == float

    def test_income_code_decoded(self, synthetic_bmf, synthetic_revocations):
        df = build_labelled_dataset(synthetic_bmf, synthetic_revocations)
        assert "income_code_usd" in df.columns

    def test_ntee_major_extracted(self, synthetic_bmf, synthetic_revocations):
        df = build_labelled_dataset(synthetic_bmf, synthetic_revocations)
        assert "ntee_major" in df.columns
        assert df.loc[df["ein"] == "123456789", "ntee_major"].values[0] == "P"

    def test_years_since_ruling_computed(self, synthetic_bmf, synthetic_revocations):
        df = build_labelled_dataset(synthetic_bmf, synthetic_revocations)
        assert "years_since_ruling" in df.columns
        # All values should be non-negative
        assert (df["years_since_ruling"] >= 0).all()

    def test_years_since_filing_computed(self, synthetic_bmf, synthetic_revocations):
        df = build_labelled_dataset(synthetic_bmf, synthetic_revocations)
        assert "years_since_filing" in df.columns
        assert (df["years_since_filing"] >= 0).all()

    def test_ein_normalized_to_nine_digits(self, synthetic_bmf, synthetic_revocations):
        df = build_labelled_dataset(synthetic_bmf, synthetic_revocations)
        assert all(len(ein) == 9 for ein in df["ein"])

    def test_revenue_amount_is_numeric(self, synthetic_bmf, synthetic_revocations):
        df = build_labelled_dataset(synthetic_bmf, synthetic_revocations)
        assert pd.api.types.is_numeric_dtype(df["revenue_amount"])

    def test_preserves_all_rows(self, synthetic_bmf, synthetic_revocations):
        df = build_labelled_dataset(synthetic_bmf, synthetic_revocations)
        assert len(df) == len(synthetic_bmf)


# ── build_features_and_labels ─────────────────────────────────────────────────

class TestBuildFeaturesAndLabels:
    def test_returns_tuple_of_dataframe_and_series(
        self, synthetic_bmf, synthetic_revocations,
    ):
        df = build_labelled_dataset(synthetic_bmf, synthetic_revocations)
        X, y = build_features_and_labels(df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_y_is_binary(self, synthetic_bmf, synthetic_revocations):
        df = build_labelled_dataset(synthetic_bmf, synthetic_revocations)
        _, y = build_features_and_labels(df)
        assert set(y.unique()).issubset({0, 1})

    def test_feature_columns_are_expected(self, synthetic_bmf, synthetic_revocations):
        df = build_labelled_dataset(synthetic_bmf, synthetic_revocations)
        X, _ = build_features_and_labels(df)
        expected = {
            "asset_code_usd", "income_code_usd", "revenue_amount",
            "subsection_code", "foundation_code", "ntee_major",
            "years_since_ruling", "years_since_filing",
            "filing_req_code", "deductibility_code", "state",
        }
        assert set(X.columns) == expected

    def test_categorical_columns_have_category_dtype(
        self, synthetic_bmf, synthetic_revocations,
    ):
        df = build_labelled_dataset(synthetic_bmf, synthetic_revocations)
        X, _ = build_features_and_labels(df)
        assert X["ntee_major"].dtype.name == "category"
        assert X["state"].dtype.name == "category"

    def test_ordinal_codes_are_numeric(self, synthetic_bmf, synthetic_revocations):
        df = build_labelled_dataset(synthetic_bmf, synthetic_revocations)
        X, _ = build_features_and_labels(df)
        for col in ("subsection_code", "foundation_code", "filing_req_code"):
            assert pd.api.types.is_numeric_dtype(X[col]), f"{col} should be numeric"

    def test_x_and_y_same_length(self, synthetic_bmf, synthetic_revocations):
        df = build_labelled_dataset(synthetic_bmf, synthetic_revocations)
        X, y = build_features_and_labels(df)
        assert len(X) == len(y)
