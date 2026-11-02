"""Push the D2 audit artifacts up to Supabase, for `app/dashboard.py` to read.

Reads only what's already on disk (the audited parquets, the generated data
dictionary, the reserved-split freeze JSON) -- nothing is computed here that
isn't computed elsewhere first. Upserts the two large tables on primary key;
delete-and-reload for the three small metadata tables, since they're cheap
to replace wholesale and have no meaningful row-level history to preserve.

    python scripts/upload_to_supabase.py --dry-run           # no network call
    python scripts/upload_to_supabase.py --tables postings   # just one table
    python scripts/upload_to_supabase.py                     # everything
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import audit as A
from src import config as C
from src.data_load import load_resumes

TABLES = ["postings", "resumes", "dictionary", "findings", "split-summary"]

POSTING_COLS = [
    "job_id", "title", "company_name", "location", "state",
    "formatted_work_type", "formatted_experience_level",
    "salary_annual", "salary_annual_capped", "salary_outlier_flag",
    "remote_allowed", "is_likely_duplicate", "duplicate_group_id",
    "skill_names", "job_posting_url", "application_url",
]


def _clean_records(df: pd.DataFrame) -> list[dict]:
    """NaN/NaT/pd.NA -> None, numpy/pandas scalars -> plain Python JSON types."""
    records = df.to_dict(orient="records")
    for row in records:
        for k, v in row.items():
            if isinstance(v, np.ndarray):
                row[k] = v.tolist()
            elif isinstance(v, list):
                pass
            elif isinstance(v, np.generic):
                row[k] = v.item()
            elif pd.isna(v):
                row[k] = None
    return records


def build_postings_payload() -> list[dict]:
    postings = A.audit_postings()[POSTING_COLS].copy()
    postings["remote_allowed"] = postings["remote_allowed"].astype(bool)
    return _clean_records(postings)


def build_resumes_payload() -> list[dict]:
    resumes = load_resumes()[["resume_id", "category", "resume_text"]]
    return _clean_records(resumes)


def build_dictionary_payload() -> list[dict]:
    if not C.DATA_DICTIONARY_CSV.exists():
        raise FileNotFoundError(
            f"{C.DATA_DICTIONARY_CSV} is missing. Run `python -m src.data_dictionary` first."
        )
    df = pd.read_csv(C.DATA_DICTIONARY_CSV)
    # n_unique is sometimes blank (list columns, see data_dictionary.py) so pandas
    # reads the whole column as float64 ("123781.0") -- a nullable int round-trips
    # to the bigint column cleanly instead.
    df["n_unique"] = df["n_unique"].astype("Int64")
    return _clean_records(df)


def build_findings_payload() -> list[dict]:
    """Hand-assembled from the same functions the notebook/data note cite --
    no number here is independent of what `src/audit.py` actually measured."""
    postings = A.audit_postings()
    companies = A.audit_companies()
    fk = A.check_orphaned_fks()
    moji = A.mojibake_report()

    n_salaried = int(postings["salary_annual"].notna().sum())
    n_outliers = int(postings["salary_outlier_flag"].sum())
    n_dup = int(postings["is_likely_duplicate"].sum())

    rows = [
        {
            "finding": "salary_outliers",
            "detail": f"salary_annual outside ${C.SALARY_SANITY_FLOOR:,.0f}-"
                      f"${C.SALARY_SANITY_CEILING:,.0f}/yr",
            "count": n_outliers,
            "pct_of_total": round(100 * n_outliers / max(n_salaried, 1), 2),
            "fixed": True,
        },
        {
            "finding": "likely_duplicate_postings",
            "detail": "shares (title, company_name, location) with another posting",
            "count": n_dup,
            "pct_of_total": round(100 * n_dup / len(postings), 2),
            "fixed": True,
        },
        {
            "finding": "company_sentinel_values",
            "detail": "state/zip_code/address/country held \"0\"/\"-\"/\"\"/\"OO\" placeholders",
            "count": int(companies[[c for c in companies.columns if c.endswith("_was_sentinel")]].any(axis=1).sum()),
            "pct_of_total": round(
                100 * companies[[c for c in companies.columns if c.endswith("_was_sentinel")]]
                .any(axis=1).mean(), 2,
            ),
            "fixed": True,
        },
        {
            "finding": "orphaned_fk_job_skills",
            "detail": "job_skills.csv job_ids with no matching postings.csv row",
            "count": fk["job_skills_orphaned"],
            "pct_of_total": round(100 * fk["job_skills_orphaned"] / fk["job_skills_job_ids"], 2),
            "fixed": False,
        },
        {
            "finding": "orphaned_fk_job_industries",
            "detail": "job_industries.csv job_ids with no matching postings.csv row",
            "count": fk["job_industries_orphaned"],
            "pct_of_total": round(100 * fk["job_industries_orphaned"] / fk["job_industries_job_ids"], 2),
            "fixed": False,
        },
        {
            "finding": "encoding_corruption_postings",
            "detail": "literal U+FFFD present in postings.csv description",
            "count": moji["postings_corrupted"],
            "pct_of_total": round(100 * moji["postings_corrupted"] / moji["postings_total"], 2),
            "fixed": False,
        },
        {
            "finding": "encoding_corruption_resumes",
            "detail": "literal U+FFFD present in Resume.csv Resume_str",
            "count": moji["resumes_corrupted"],
            "pct_of_total": round(100 * moji["resumes_corrupted"] / moji["resumes_total"], 2),
            "fixed": False,
        },
    ]
    return rows


def build_split_summary_payload() -> list[dict]:
    if not C.TEST_SPLIT_FREEZE_JSON.exists():
        raise FileNotFoundError(
            f"{C.TEST_SPLIT_FREEZE_JSON} is missing. Run "
            "`python -m src.pairs --freeze-check` first."
        )
    frozen = json.loads(C.TEST_SPLIT_FREEZE_JSON.read_text())
    return [{
        "id": 1,
        "n_pairs": frozen["n_pairs"],
        "n_resumes": frozen["n_resumes"],
        "n_jobs": frozen["n_jobs"],
        "resume_id_sha256": frozen["resume_id_sha256"],
        "job_id_sha256": frozen["job_id_sha256"],
        "frozen_on": frozen["frozen_on"],
    }]


BUILDERS = {
    "postings": ("postings", build_postings_payload, "job_id"),
    "resumes": ("resumes", build_resumes_payload, "resume_id"),
    "dictionary": ("data_dictionary", build_dictionary_payload, None),
    "findings": ("data_quality_findings", build_findings_payload, None),
    "split-summary": ("test_split_summary", build_split_summary_payload, None),
}


def push(table_key: str, *, dry_run: bool) -> None:
    table_name, builder, upsert_key = BUILDERS[table_key]
    rows = builder()

    if dry_run:
        print(f"[dry-run] {table_name}: {len(rows):,} rows ready "
              f"({'upsert on ' + upsert_key if upsert_key else 'delete-and-reload'})")
        return

    from src.supabase_client import get_client

    client = get_client()
    if upsert_key:
        chunk_size = 2000
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i:i + chunk_size]
            client.table(table_name).upsert(chunk, on_conflict=upsert_key).execute()
            print(f"  {table_name}: {min(i + chunk_size, len(rows)):,} / {len(rows):,}")
    else:
        first_key = list(rows[0].keys())[0]
        sentinel = -1 if isinstance(rows[0][first_key], int) else "__never_matches__"
        client.table(table_name).delete().neq(first_key, sentinel).execute()
        client.table(table_name).insert(rows).execute()
    print(f"{table_name}: pushed {len(rows):,} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", default="all",
                     help=f"comma-separated subset of {TABLES}, or 'all'")
    ap.add_argument("--dry-run", action="store_true",
                     help="build payloads and print row counts, no network call")
    args = ap.parse_args()

    selected = TABLES if args.tables == "all" else args.tables.split(",")
    for key in selected:
        if key not in BUILDERS:
            raise SystemExit(f"unknown table {key!r}, choose from {TABLES}")
        push(key, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
