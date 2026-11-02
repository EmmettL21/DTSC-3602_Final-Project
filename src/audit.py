"""D2 data-quality audit: sentinel/outlier/duplicate flags, plus the issues we
document but do not attempt to fix.

Everything here is additive on top of `data_load.py`'s cleaned frames --
`postings_clean.parquet` and `resumes_clean.parquet` are read, never rewritten,
so D1's trained model (fit on the unflagged data) stays valid. New columns and
a new sibling parquet (`*_audited.parquet`) carry the findings instead.

    normalize_company_sentinels()  "0"/"-"/""/"OO" -> true NaN, companies.csv
    flag_salary_outliers()         salary_annual_capped + salary_outlier_flag
    flag_duplicate_postings()      is_likely_duplicate + duplicate_group_id
    flag_encoding_corruption()     count/flag literal U+FFFD -- documented, not fixed
    check_orphaned_fks()           job_skills/job_industries vs. postings.csv job_id
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C
from .data_load import load_postings, load_resumes

# Sentinel spellings observed in companies.csv standing in for a true null.
_SENTINELS = {"0", "-", "", "OO"}
_SENTINEL_COLS = ["state", "zip_code", "address", "country"]

_MOJIBAKE = "�"


def _is_sentinel(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().isin(_SENTINELS)


def normalize_company_sentinels(*, rebuild: bool = False) -> pd.DataFrame:
    """Raw `companies.csv`, with placeholder strings replaced by real NaN.

    `country` additionally treats "OO" as a sentinel -- not a real ISO-3166
    code, and the one non-country value that shows up at meaningful volume.
    """
    if C.COMPANIES_AUDITED_PARQUET.exists() and not rebuild:
        return pd.read_parquet(C.COMPANIES_AUDITED_PARQUET)

    C.ensure_artifacts()
    cols = ["company_id", "name", "company_size", *_SENTINEL_COLS]
    df = pd.read_csv(C.COMPANIES_CSV, usecols=cols)

    for col in _SENTINEL_COLS:
        was_sentinel = _is_sentinel(df[col])
        df[f"{col}_was_sentinel"] = was_sentinel
        df.loc[was_sentinel, col] = np.nan

    df.to_parquet(C.COMPANIES_AUDITED_PARQUET, index=False)
    return df


def flag_salary_outliers(postings: pd.DataFrame) -> pd.DataFrame:
    """Add `salary_annual_capped` + `salary_outlier_flag`.

    Fixed sanity bounds, not a percentile cut -- a percentile would clip
    legitimate high executive pay along with the clear data-entry errors
    (raw `normalized_salary` reaches $535.6M). `salary_annual` is left
    untouched so nothing downstream that already reads it changes behavior.
    """
    out = postings.copy()
    salary = out["salary_annual"]
    outlier = salary.notna() & (
        (salary < C.SALARY_SANITY_FLOOR) | (salary > C.SALARY_SANITY_CEILING)
    )
    out["salary_outlier_flag"] = outlier
    out["salary_annual_capped"] = salary.where(~outlier)
    return out


def flag_duplicate_postings(postings: pd.DataFrame) -> pd.DataFrame:
    """Add `is_likely_duplicate` + `duplicate_group_id`.

    Groups by (title, company_name, location) -- a shared `job_id` is already
    guaranteed unique by the source data, so a duplicate here means the same
    listing was scraped or re-posted under a different ID, not a key violation.
    First row in each group is kept as the canonical one; the rest are flagged
    rather than dropped, since D1's existing model already trained on the
    undeduped set and silently shrinking the corpus would move its metrics for
    reasons unrelated to the audit.
    """
    out = postings.copy()
    group_cols = C.DUPLICATE_KEY_COLS
    group_id = out.groupby(group_cols, dropna=False).ngroup()
    out["duplicate_group_id"] = group_id
    group_size = group_id.map(group_id.value_counts())
    is_dup_group = group_size > 1
    first_in_group_idx = out.drop_duplicates(group_cols, keep="first").index
    not_first = ~out.index.isin(first_in_group_idx)
    out["is_likely_duplicate"] = is_dup_group & not_first
    return out


def flag_encoding_corruption(series: pd.Series) -> pd.Series:
    """Boolean mask: literal U+FFFD replacement character present.

    Marks mojibake from a source-side charset mismatch. Documented as a known
    limitation rather than repaired -- guessing at which byte sequence a
    replacement character originally stood for risks inventing text that was
    never actually in the source document.
    """
    return series.fillna("").astype(str).str.contains(_MOJIBAKE, regex=False)


def mojibake_report() -> dict:
    """Corruption rate in each raw text source, read fresh (not the cleaned
    frames, since PII scrubbing does not touch U+FFFD either way)."""
    postings_desc = pd.read_csv(C.POSTINGS_CSV, usecols=["description"])["description"]
    resumes_text = pd.read_csv(C.RESUMES_CSV, usecols=["Resume_str"])["Resume_str"]
    post_bad = flag_encoding_corruption(postings_desc)
    resume_bad = flag_encoding_corruption(resumes_text)
    return {
        "postings_total": int(len(post_bad)),
        "postings_corrupted": int(post_bad.sum()),
        "resumes_total": int(len(resume_bad)),
        "resumes_corrupted": int(resume_bad.sum()),
    }


def check_orphaned_fks() -> dict:
    """`job_skills.csv` / `job_industries.csv` job_ids that `postings.csv` has
    no matching row for -- a referential-integrity gap in the raw data, not
    something this pipeline can fabricate a fix for."""
    posting_ids = set(pd.read_csv(C.POSTINGS_CSV, usecols=["job_id"])["job_id"])
    skills_ids = set(pd.read_csv(C.JOB_SKILLS_CSV, usecols=["job_id"])["job_id"])
    industries_ids = set(pd.read_csv(C.JOB_INDUSTRIES_CSV, usecols=["job_id"])["job_id"])
    return {
        "postings_job_ids": len(posting_ids),
        "job_skills_job_ids": len(skills_ids),
        "job_skills_orphaned": len(skills_ids - posting_ids),
        "job_industries_job_ids": len(industries_ids),
        "job_industries_orphaned": len(industries_ids - posting_ids),
    }


def salaries_outlier_stats() -> pd.DataFrame:
    """Raw `salaries.csv` (currently unread by `data_load.py`) with the same
    sanity-bound outlier flag, for the audit plot -- not merged into postings."""
    df = pd.read_csv(C.SALARIES_CSV)
    for col in ("min_salary", "max_salary", "med_salary"):
        df[f"{col}_outlier"] = df[col].notna() & (
            (df[col] < C.SALARY_SANITY_FLOOR) | (df[col] > C.SALARY_SANITY_CEILING)
        )
    return df


def audit_postings(*, rebuild: bool = False) -> pd.DataFrame:
    """Cleaned postings + the salary-outlier and duplicate flags above."""
    if C.POSTINGS_AUDITED_PARQUET.exists() and not rebuild:
        return pd.read_parquet(C.POSTINGS_AUDITED_PARQUET)

    C.ensure_artifacts()
    postings = load_postings()
    postings = flag_salary_outliers(postings)
    postings = flag_duplicate_postings(postings)
    postings.to_parquet(C.POSTINGS_AUDITED_PARQUET, index=False)
    return postings


def audit_companies(*, rebuild: bool = False) -> pd.DataFrame:
    return normalize_company_sentinels(rebuild=rebuild)


def _smoke() -> None:
    postings = audit_postings(rebuild=True)
    companies = audit_companies(rebuild=True)
    resumes = load_resumes()

    print(f"postings audited   {len(postings):>8,}")
    print(f"resumes             {len(resumes):>8,}")
    print()

    print("salary outliers (outside "
          f"${C.SALARY_SANITY_FLOOR:,.0f}-${C.SALARY_SANITY_CEILING:,.0f}/yr)")
    n_sal = postings["salary_annual"].notna().sum()
    n_out = int(postings["salary_outlier_flag"].sum())
    print(f"  {n_out:,} / {n_sal:,} salaried postings flagged ({100 * n_out / max(n_sal, 1):.2f}%)")
    print()

    print("likely-duplicate postings (title, company_name, location)")
    n_dup = int(postings["is_likely_duplicate"].sum())
    print(f"  {n_dup:,} / {len(postings):,} flagged ({100 * n_dup / len(postings):.1f}%)")
    print()

    print("company-field sentinel values normalized to NaN")
    for col in _SENTINEL_COLS:
        n = int(companies[f"{col}_was_sentinel"].sum())
        print(f"  {col:<10} {n:>6,} / {len(companies):,} ({100 * n / len(companies):.1f}%)")
    print()

    print("orphaned foreign keys (job_id in linking table, not in postings.csv)")
    for k, v in check_orphaned_fks().items():
        print(f"  {k:<24} {v:,}")
    print()

    print("encoding corruption (literal U+FFFD present)")
    for k, v in mojibake_report().items():
        print(f"  {k:<24} {v:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="(re)build the audited parquets")
    ap.add_argument("--smoke", action="store_true", help="print before/after counts")
    args = ap.parse_args()

    if args.build:
        audit_postings(rebuild=True)
        audit_companies(rebuild=True)
    if args.smoke or not args.build:
        _smoke()


if __name__ == "__main__":
    main()
