"""Load and join the raw CSVs into two clean frames, cached as parquet.

    postings  one row per job_id, with skills/industries/company rolled in
    resumes   one row per resume ID, PII-scrubbed

Both are written to `artifacts/` so the rest of the pipeline never pays the
493MB `postings.csv` parse more than once.
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import config as C
from .text_clean import clean_series, truncate

# Only the columns the model or the UI actually uses. `postings.csv` is 493MB
# with newline-laden descriptions; naming columns explicitly keeps the parse
# from ballooning.
POSTING_COLS = [
    "job_id",
    "company_name",
    "title",
    "description",
    "location",
    "company_id",
    "views",
    "applies",
    "formatted_work_type",
    "formatted_experience_level",
    "remote_allowed",
    "normalized_salary",
    "pay_period",
    "min_salary",
    "max_salary",
    "job_posting_url",
    "application_url",
]

# US state abbreviations, for the location-match feature.
_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
}

_STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}

# Ordinal scale for the experience-level distance feature.
EXPERIENCE_ORDER = {
    "Internship": 0,
    "Entry level": 1,
    "Associate": 2,
    "Mid-Senior level": 3,
    "Director": 4,
    "Executive": 5,
}

# Annualization multipliers for `pay_period`.
PAY_PERIOD_TO_ANNUAL = {
    "HOURLY": 2080.0,
    "WEEKLY": 52.0,
    "BIWEEKLY": 26.0,
    "SEMIMONTHLY": 24.0,
    "MONTHLY": 12.0,
    "YEARLY": 1.0,
}


def parse_state(location: object) -> str | None:
    """Pull a two-letter state out of a free-text LinkedIn location string.

    Values look like `Dallas, TX`, `New York, New York`, `United States`, or
    `Greater Boston`. Anything we cannot resolve becomes None rather than a
    guess -- a wrong state would silently corrupt the location-match feature.
    """
    if location is None or (isinstance(location, float) and pd.isna(location)):
        return None
    text = str(location).strip()
    if not text:
        return None

    tail = text.rsplit(",", 1)[-1].strip()
    if tail.upper() in _STATES:
        return tail.upper()
    if tail.lower() in _STATE_NAMES:
        return _STATE_NAMES[tail.lower()]

    # Some rows are bare state names with no comma at all.
    low = text.lower()
    if low in _STATE_NAMES:
        return _STATE_NAMES[low]
    return None


def _annualize(salary: pd.Series, period: pd.Series) -> pd.Series:
    mult = period.map(PAY_PERIOD_TO_ANNUAL)
    return salary * mult


def load_postings(*, rebuild: bool = False, sample: int | None = None) -> pd.DataFrame:
    """One row per posting, with skills, industries and company joined in."""
    if C.POSTINGS_PARQUET.exists() and not rebuild:
        df = pd.read_parquet(C.POSTINGS_PARQUET)
        return df.head(sample) if sample else df

    C.ensure_artifacts()
    df = pd.read_csv(
        C.POSTINGS_CSV,
        usecols=POSTING_COLS,
        dtype={"job_id": "int64", "company_id": "float64"},
        low_memory=False,
    )

    df = df[df["description"].notna() & (df["description"].str.len() > 50)].copy()

    # --- skills: abbrevs + human-readable names ---
    skill_map = pd.read_csv(C.SKILL_MAP_CSV).set_index("skill_abr")["skill_name"]
    js = pd.read_csv(C.JOB_SKILLS_CSV)
    js = js[js["job_id"].isin(df["job_id"])]
    skills = js.groupby("job_id")["skill_abr"].apply(list)
    skill_names = skills.map(lambda abrs: [skill_map.get(a, a) for a in abrs])

    # --- industries: names via the id mapping ---
    ind_map = pd.read_csv(C.INDUSTRY_MAP_CSV).set_index("industry_id")["industry_name"]
    ji = pd.read_csv(C.JOB_INDUSTRIES_CSV)
    ji = ji[ji["job_id"].isin(df["job_id"])]
    ji["industry_name"] = ji["industry_id"].map(ind_map)
    industries = (
        ji.dropna(subset=["industry_name"]).groupby("job_id")["industry_name"].apply(list)
    )

    df["skill_abrs"] = df["job_id"].map(skills)
    df["skill_names"] = df["job_id"].map(skill_names)
    df["industries"] = df["job_id"].map(industries)
    for col in ("skill_abrs", "skill_names", "industries"):
        df[col] = df[col].map(lambda v: v if isinstance(v, list) else [])

    # --- company description, for the embedded card text ---
    companies = pd.read_csv(
        C.COMPANIES_CSV, usecols=["company_id", "name", "description", "company_size"]
    ).drop_duplicates("company_id")
    companies = companies.rename(
        columns={"description": "company_description", "name": "company_official_name"}
    )
    df = df.merge(companies, on="company_id", how="left")

    counts = (
        pd.read_csv(C.EMPLOYEE_COUNTS_CSV, usecols=["company_id", "employee_count", "follower_count"])
        .sort_values("company_id")
        .drop_duplicates("company_id", keep="last")
    )
    df = df.merge(counts, on="company_id", how="left")

    # --- derived fields ---
    df["state"] = df["location"].map(parse_state)
    df["experience_rank"] = df["formatted_experience_level"].map(EXPERIENCE_ORDER)
    df["remote_allowed"] = df["remote_allowed"].fillna(0).astype("int8")

    # `normalized_salary` is already annualized where present; fall back to the
    # min/max midpoint for rows that only carry a raw range.
    mid = df[["min_salary", "max_salary"]].mean(axis=1)
    fallback = _annualize(mid, df["pay_period"])
    df["salary_annual"] = df["normalized_salary"].fillna(fallback)

    # --- text for embedding ---
    title = df["title"].fillna("")
    company = df["company_name"].fillna("")
    skills_blob = df["skill_names"].map(lambda xs: ", ".join(xs))
    desc = clean_series(df["description"])
    combined = (
        title + ". " + company + ". " + skills_blob + ". " + desc
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    df["posting_text"] = combined.map(lambda t: truncate(t, C.MAX_TEXT_CHARS))

    keep = [
        "job_id", "title", "company_name", "company_official_name",
        "company_description", "company_size", "employee_count", "follower_count",
        "location", "state", "formatted_work_type", "formatted_experience_level",
        "experience_rank", "remote_allowed", "salary_annual", "pay_period",
        "views", "applies", "skill_abrs", "skill_names", "industries",
        "job_posting_url", "application_url", "posting_text",
    ]
    df = df[keep].reset_index(drop=True)
    df.to_parquet(C.POSTINGS_PARQUET, index=False)
    return df.head(sample) if sample else df


def load_resumes(*, rebuild: bool = False) -> pd.DataFrame:
    """One row per resume, PII-scrubbed, with its category label."""
    if C.RESUMES_PARQUET.exists() and not rebuild:
        return pd.read_parquet(C.RESUMES_PARQUET)

    C.ensure_artifacts()
    df = pd.read_csv(C.RESUMES_CSV, usecols=["ID", "Resume_str", "Category"])
    df = df.rename(columns={"ID": "resume_id", "Category": "category"})

    # `strip_resume_head` drops the all-caps name/headline block each record
    # opens with -- the most direct identifier in this dataset.
    df["resume_text_full"] = clean_series(df["Resume_str"], strip_resume_head=True)
    df["resume_text"] = df["resume_text_full"].map(lambda t: truncate(t, C.MAX_TEXT_CHARS))
    df = df.drop(columns=["Resume_str"])

    df = df[df["resume_text"].str.len() > 100].reset_index(drop=True)
    df.to_parquet(C.RESUMES_PARQUET, index=False)
    return df


def _smoke() -> None:
    postings = load_postings()
    resumes = load_resumes()

    print(f"postings           {len(postings):>8,}  unique {postings.job_id.nunique():,}")
    print(f"resumes            {len(resumes):>8,}  categories {resumes.category.nunique()}")
    print()
    print("join / field coverage (% of postings non-null)")
    for col, series in [
        ("skills", postings.skill_abrs.map(len) > 0),
        ("industries", postings.industries.map(len) > 0),
        ("company name", postings.company_official_name.notna()),
        ("state parsed", postings.state.notna()),
        ("experience lvl", postings.experience_rank.notna()),
        ("salary", postings.salary_annual.notna()),
        ("remote", postings.remote_allowed.astype(bool)),
    ]:
        print(f"  {col:<16} {100 * series.mean():>5.1f}%")

    print()
    print(f"posting_text  mean {int(postings.posting_text.str.len().mean()):,} chars")
    print(f"resume_text   mean {int(resumes.resume_text.str.len().mean()):,} chars")

    from .text_clean import has_pii

    leaked = resumes.resume_text.map(has_pii).mean()
    print(f"resumes with residual explicit PII: {100 * leaked:.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="print row counts and coverage")
    ap.add_argument("--rebuild", action="store_true", help="ignore the parquet cache")
    args = ap.parse_args()

    if args.rebuild:
        load_postings(rebuild=True)
        load_resumes(rebuild=True)
    _smoke()

if __name__ == "__main__":
    main()
