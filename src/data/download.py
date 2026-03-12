"""
IRS data acquisition module.

Downloads two public IRS datasets:
  1. IRS Business Master File (BMF) — all active tax-exempt orgs (~1.8M records)
  2. IRS Auto-Revocation List — orgs that lost exempt status (used as risk labels)

Both are published by the IRS and freely available.
"""

import io
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

IRS_SOURCES = {
    "bmf_region1": "https://apps.irs.gov/pub/epostcard/data-download-pub78.zip",
    "revocations": "https://apps.irs.gov/pub/epostcard/data-download-revocation.zip",
}

BMF_COLS = [
    "ein",
    "name",
    "in_care_of_name",
    "street",
    "city",
    "state",
    "zip_code",
    "group_exemption",
    "subsection_code",
    "affiliation_code",
    "classification_code",
    "ruling_date",
    "deductibility_code",
    "foundation_code",
    "activity_code",
    "organization_code",
    "exempt_status_code",
    "tax_period",
    "asset_code",
    "income_code",
    "filing_req_code",
    "pf_filing_req_code",
    "acct_pd",
    "asset_amount",
    "income_amount",
    "revenue_amount",
    "ntee_code",
    "sort_name",
]

REVOCATION_COLS = [
    "ein",
    "name",
    "address1",
    "address2",
    "city",
    "state",
    "zip",
    "country",
    "exemption_type",
    "revocation_date",
    "revocation_posting_date",
    "exemption_reinstatement_date",
]


def _download_with_progress(url: str, dest: Path) -> Path:
    """Stream-download a file, showing a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=dest.name, total=total, unit="B", unit_scale=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    return dest


def _extract_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    """Extract the first CSV/txt file from a zip archive."""
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))]
        if not csv_names:
            raise ValueError(f"No CSV/txt found in {zip_path}")
        with zf.open(csv_names[0]) as f:
            raw = io.TextIOWrapper(f, encoding="latin-1")
            return pd.read_csv(raw, dtype=str, low_memory=False)


def download_bmf(force: bool = False) -> pd.DataFrame:
    """
    Download and parse the IRS Publication 78 (Pub78) data.
    Pub78 lists all organizations eligible to receive tax-deductible contributions.

    Returns a DataFrame with one row per tax-exempt organisation.
    """
    dest = RAW_DIR / "pub78.zip"
    if not dest.exists() or force:
        print("Downloading IRS Pub78 (eligible donee list)...")
        _download_with_progress(IRS_SOURCES["bmf_region1"], dest)

    print("Parsing Pub78...")
    with zipfile.ZipFile(dest) as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))][0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(
                io.TextIOWrapper(f, encoding="latin-1"),
                sep="|",
                header=None,
                names=["ein", "name", "city", "state", "country", "deductibility"],
                dtype=str,
            )

    df["ein"] = df["ein"].str.strip()
    print(f"  {len(df):,} organisations loaded from Pub78.")
    return df


def download_revocations(force: bool = False) -> pd.DataFrame:
    """
    Download and parse the IRS Auto-Revocation list.
    Orgs on this list had their exempt status revoked (our risk label).

    Returns a DataFrame of revoked organisations.
    """
    dest = RAW_DIR / "revocations.zip"
    if not dest.exists() or force:
        print("Downloading IRS Auto-Revocation list...")
        _download_with_progress(IRS_SOURCES["revocations"], dest)

    print("Parsing revocation list...")
    df = _extract_csv_from_zip(dest)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"taxpayer_name": "name", "eo_type": "exemption_type"})

    # Normalise EIN format
    if "ein" in df.columns:
        df["ein"] = df["ein"].astype(str).str.strip().str.zfill(9)

    print(f"  {len(df):,} revoked organisations loaded.")
    return df


def download_all(force: bool = False) -> dict[str, pd.DataFrame]:
    """Download all required IRS datasets. Returns a dict of DataFrames."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "bmf": download_bmf(force=force),
        "revocations": download_revocations(force=force),
    }


if __name__ == "__main__":
    data = download_all()
    for name, df in data.items():
        out = RAW_DIR / f"{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"Saved {out}")
