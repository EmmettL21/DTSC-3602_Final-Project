"""Generate `artifacts/data_dictionary.csv` by introspecting the actual data.

Deliberately generated, not hand-maintained: a hand-written dictionary drifts
the moment a column is added or a cleaning step changes, and nobody notices
until the write-up contradicts the code. This script is the single source
both the D2 data note and the Supabase upload read from.

Covers every table D2's audit touches, including the ones `data_load.py`
never loads (`company_industries.csv`, `company_specialities.csv`, partial
`benefits.csv` coverage) -- an unused column is still part of the dictionary.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import audit as A
from . import config as C
from .data_load import load_postings, load_resumes


def _row(table: str, col: str, series: pd.Series, note: str = "") -> dict:
    non_null = series.dropna()
    example = non_null.iloc[0] if len(non_null) else None
    is_list_col = len(non_null) and isinstance(non_null.iloc[0], (list, dict, np.ndarray))
    if is_list_col:
        example = str(example)[:80]
        n_unique = None
        pct_missing = round(
            100 * series.map(lambda v: not isinstance(v, (list, np.ndarray)) or len(v) == 0).mean(), 2
        )
    else:
        n_unique = int(series.nunique(dropna=True))
        pct_missing = round(100 * series.isna().mean(), 2)
    return {
        "table_name": table,
        "column_name": col,
        "dtype": str(series.dtype),
        "pct_missing": pct_missing,
        "n_unique": n_unique,
        "example_value": example,
        "note": (note + " (list column: pct_missing = share with an empty list)").strip()
        if is_list_col else note,
    }


def _describe(table: str, df: pd.DataFrame, notes: dict[str, str] | None = None) -> list[dict]:
    notes = notes or {}
    return [_row(table, col, df[col], notes.get(col, "")) for col in df.columns]


def build(*, rebuild: bool = False) -> pd.DataFrame:
    rows: list[dict] = []

    postings = load_postings(rebuild=rebuild)
    rows += _describe(
        "postings_clean",
        postings,
        {
            "salary_annual": "Fallback-filled from min/max when normalized_salary is absent.",
            "remote_allowed": "NaN in the raw file means unknown, filled to 0 here -- see "
            "postings_audited for an unmodified outlier view instead of trusting this blindly.",
        },
    )

    audited = A.audit_postings(rebuild=rebuild)
    rows += [
        _row("postings_audited", "salary_annual_capped",
             audited["salary_annual_capped"],
             "salary_annual with sanity-bound outliers (see audit.py) set to NaN."),
        _row("postings_audited", "salary_outlier_flag",
             audited["salary_outlier_flag"],
             f"Outside ${C.SALARY_SANITY_FLOOR:,.0f}-${C.SALARY_SANITY_CEILING:,.0f}/yr."),
        _row("postings_audited", "is_likely_duplicate",
             audited["is_likely_duplicate"],
             "Shares (title, company_name, location) with an earlier row."),
        _row("postings_audited", "duplicate_group_id",
             audited["duplicate_group_id"],
             "Group key for the duplicate check above."),
    ]

    companies = A.audit_companies(rebuild=rebuild)
    rows += _describe(
        "companies_audited",
        companies,
        {c: "Sentinel values (\"0\"/\"-\"/\"\"/\"OO\") normalized to NaN before this stat."
         for c in ("state", "zip_code", "address", "country")},
    )

    resumes = load_resumes(rebuild=rebuild)
    rows += _describe("resumes_clean", resumes)

    # Raw sources the pipeline reads for only one or two columns, or not at all.
    raw_sources = {
        "salaries.csv (raw)": (C.SALARIES_CSV, None),
        "benefits.csv (raw)": (C.BENEFITS_CSV,
            "Covers only ~24% of postings' job_ids -- not merged, sparse by nature."),
        "job_skills.csv (raw)": (C.JOB_SKILLS_CSV,
            "References job_ids absent from postings.csv -- see check_orphaned_fks()."),
        "job_industries.csv (raw)": (C.JOB_INDUSTRIES_CSV,
            "Same orphaned-FK issue as job_skills.csv."),
        "company_industries.csv (raw)": (C.COMPANY_INDUSTRIES_CSV,
            "Never joined into postings_clean -- documented, unused."),
        "company_specialities.csv (raw)": (C.COMPANY_SPECIALITIES_CSV,
            "Never joined into postings_clean -- documented, unused."),
    }
    for table, (path, note) in raw_sources.items():
        df = pd.read_csv(path)
        rows += _describe(table, df, {c: note for c in df.columns} if note else None)

    out = pd.DataFrame(rows)
    C.ensure_artifacts()
    out.to_csv(C.DATA_DICTIONARY_CSV, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="ignore cached parquets")
    args = ap.parse_args()

    dictionary = build(rebuild=args.rebuild)
    print(f"wrote {len(dictionary):,} rows to {C.DATA_DICTIONARY_CSV}")
    print(dictionary["table_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
