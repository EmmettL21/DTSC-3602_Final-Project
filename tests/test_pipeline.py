"""Checks for the parts of the pipeline that fail silently.

Run with:  python -m tests.test_pipeline

Deliberately not pytest -- the group runs this from a plain Python prompt, and
the only dependency worth adding is none. Each check prints PASS/FAIL and the
script exits non-zero if anything failed.

These target the failure modes that do not raise on their own: a leaky split
that inflates every metric, a stale embedding cache silently misaligned with
the postings frame, PII surviving the scrubber. A wrong answer from any of
these looks exactly like a right one in the output.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd

from src import config as C
from src import taxonomy as T
from src.features import Preferences, build_feature_matrix, resume_skills
from src.text_clean import clean_text, has_pii

_FAILURES: list[str] = []


def check(name: str):
    """Decorator that runs a check and records the outcome."""
    def wrap(fn):
        try:
            fn()
        except AssertionError as exc:
            _FAILURES.append(name)
            print(f"FAIL  {name}\n      {exc}")
        except Exception:
            _FAILURES.append(name)
            print(f"ERROR {name}")
            traceback.print_exc()
        else:
            print(f"PASS  {name}")
        return fn
    return wrap


# --- text cleaning ----------------------------------------------------------


@check("PII scrubber removes explicit identifiers")
def _():
    dirty = (
        "JANE DOE   Senior Analyst   Summary   Reach me at jane.doe@example.com "
        "or (704) 555-0137. Lives at 123 Main Street, Charlotte 28202. "
        "Portfolio: https://janedoe.example.com"
    )
    clean = clean_text(dirty, strip_resume_head=True)
    assert "jane.doe@example.com" not in clean, "email survived"
    assert "555-0137" not in clean, "phone survived"
    assert "28202" not in clean, "ZIP survived"
    assert "janedoe.example.com" not in clean, "URL survived"
    assert "123 Main Street" not in clean, "street address survived"
    assert not has_pii(clean), f"has_pii still flags: {clean!r}"
    assert "Analyst" in clean, "scrubbing destroyed the useful content"


@check("cleaner handles empty and null input")
def _():
    for value in ("", None, float("nan"), "   "):
        assert clean_text(value) == "", f"{value!r} should clean to empty string"


# --- taxonomy ---------------------------------------------------------------


@check("category adjacency is symmetric and complete")
def _():
    for name in T.CATEGORIES:
        for other in T.RELATED[name]:
            assert name in T.RELATED[other], f"{name} -> {other} is one-way"
        assert name not in T.RELATED[name], f"{name} is related to itself"
    assert len(T.CATEGORY_NAMES) == 24, f"expected 24 categories, got {len(T.CATEGORY_NAMES)}"


@check("assign_category routes obvious postings correctly")
def _():
    cases = [
        ("Senior Software Engineer", {"IT", "ENG"}, ["Software Development"],
         "INFORMATION-TECHNOLOGY"),
        ("Registered Nurse - ICU", {"HCPR"}, ["Hospitals and Health Care"], "HEALTHCARE"),
        ("Staff Accountant", {"ACCT"}, ["Accounting"], "ACCOUNTANT"),
        ("High School Math Teacher", {"EDU"}, ["Primary and Secondary Education"], "TEACHER"),
    ]
    for title, skills, industries, expected in cases:
        got, score = T.assign_category(title, skills, industries)
        assert got == expected, f"{title!r} -> {got} (score {score:.1f}), expected {expected}"


@check("vague postings get no category rather than a guess")
def _():
    got, score = T.assign_category("Team Member", set(), [])
    assert got is None, f"'Team Member' should be unlabelled, got {got} at {score}"


@check("pair_label implements the three tiers")
def _():
    assert T.pair_label("FINANCE", "FINANCE") == "Good Fit"
    assert T.pair_label("FINANCE", "ACCOUNTANT") == "Partial Fit"
    assert T.pair_label("FINANCE", "CHEF") == "No Fit"
    assert T.pair_label("FINANCE", None) == "No Fit"


# --- features ---------------------------------------------------------------


@check("resume_skills detects skills from text")
def _():
    found = resume_skills("Built Python and SQL pipelines; managed the data warehouse.")
    assert "IT" in found, f"expected IT in {found}"
    assert "HCPR" not in found, f"unexpected clinical skill in {found}"


@check("feature matrix has the right shape, columns and no NaNs")
def _():
    postings = pd.DataFrame({
        "job_id": [1, 2],
        "title": ["Staff Accountant", "Line Cook"],
        "skill_abrs": [["ACCT", "FIN"], ["PROD"]],
        "state": ["NC", None],
        "remote_allowed": [0, 1],
        "salary_annual": [80_000.0, np.nan],
        "experience_rank": [2.0, np.nan],
        "formatted_work_type": ["Full-time", "Part-time"],
        "company_size": [3.0, np.nan],
    })
    resumes = pd.DataFrame({
        "resume_id": [10],
        "resume_text": ["Accounting and audit work, payroll and ledger reconciliation."],
        "category": ["ACCOUNTANT"],
    })
    pairs = pd.DataFrame({"resume_id": [10, 10], "job_id": [1, 2]})

    X = build_feature_matrix(
        pairs, postings, resumes,
        cos_minilm=np.array([0.8, 0.1], dtype=np.float32),
        cos_tfidf=np.array([0.6, 0.05], dtype=np.float32),
        prefs={10: Preferences(target_salary=70_000, state="NC")},
    )

    from src.features import FEATURE_NAMES
    assert list(X.columns) == FEATURE_NAMES, "column order drifted from FEATURE_NAMES"
    assert len(X) == 2, f"expected 2 rows, got {len(X)}"
    assert not X.isna().to_numpy().any(), f"NaNs leaked into features:\n{X}"

    # The accounting posting should dominate on every content signal.
    assert X.loc[0, "skill_overlap_n"] > X.loc[1, "skill_overlap_n"]
    assert X.loc[0, "title_hits"] > 0, "accountant title should hit its own pattern"

    # Missingness is recorded, not imputed away.
    assert X.loc[0, "salary_missing"] == 0 and X.loc[1, "salary_missing"] == 1
    assert X.loc[0, "salary_meets_target"] == 1, "80k meets a 70k target"
    assert X.loc[0, "state_match"] == 1, "NC posting matches NC preference"
    assert X.loc[1, "state_match"] == 1, "remote posting satisfies any state"


@check("absent preferences do not crash or fabricate matches")
def _():
    postings = pd.DataFrame({
        "job_id": [1], "title": ["Line Cook"], "skill_abrs": [["PROD"]],
        "state": ["TX"], "remote_allowed": [0], "salary_annual": [50_000.0],
        "experience_rank": [1.0], "formatted_work_type": ["Full-time"],
        "company_size": [2.0],
    })
    resumes = pd.DataFrame({
        "resume_id": [10], "resume_text": ["Kitchen and culinary work."], "category": ["CHEF"],
    })
    pairs = pd.DataFrame({"resume_id": [10], "job_id": [1]})

    X = build_feature_matrix(
        pairs, postings, resumes,
        cos_minilm=np.array([0.5], np.float32),
        cos_tfidf=np.array([0.4], np.float32),
        prefs=None,
    )
    assert X.loc[0, "salary_missing"] == 1, "no target means the salary feature is missing"
    assert X.loc[0, "state_unknown"] == 1, "no state preference means unknown"
    assert X.loc[0, "work_type_match"] == 1, "no work-type filter accepts everything"


# --- split integrity (needs the cached artifacts) ---------------------------


@check("train/test split leaks no resume or job")
def _():
    if not C.PAIRS_PARQUET.exists():
        print("      skipped: run `python -m src.pairs --check` first")
        return
    pairs = pd.read_parquet(C.PAIRS_PARQUET)
    train = pairs[pairs.split == "train"]
    test = pairs[pairs.split == "test"]
    assert len(test) > 0, "test split is empty"

    shared_r = set(train.resume_id) & set(test.resume_id)
    shared_j = set(train.job_id) & set(test.job_id)
    assert not shared_r, f"{len(shared_r)} resume IDs appear on both sides"
    assert not shared_j, f"{len(shared_j)} job IDs appear on both sides"

    # All three classes must be present on both sides or the metrics are
    # undefined for whichever class is missing.
    for split, side in (("train", train), ("test", test)):
        missing = set(C.LABELS) - set(side.label.unique())
        assert not missing, f"{split} split is missing {missing}"


@check("embedding caches align with their indices")
def _():
    if not C.POST_EMB_NPY.exists():
        print("      skipped: run `python -m src.embed --build` first")
        return
    from src.embed import load_posting_vectors, load_resume_vectors

    posts, job_ids = load_posting_vectors("minilm")
    resumes_m, resume_ids = load_resume_vectors("minilm")

    assert posts.shape[1] == C.EMBED_DIM, f"expected {C.EMBED_DIM}-d, got {posts.shape[1]}"
    assert len(job_ids) == len(set(job_ids)), "duplicate job IDs in the embedding index"

    # Vectors must be L2-normalized or every cosine downstream is wrong.
    for name, matrix in (("postings", posts), ("resumes", resumes_m)):
        norms = np.linalg.norm(matrix[:1000], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), f"{name} vectors are not normalized"

    if C.TFIDF_POST_NPY.exists():
        tfidf, _ = load_posting_vectors("tfidf")
        assert len(tfidf) == len(posts), "TF-IDF and MiniLM caches disagree on row count"


@check("cached postings frame matches the embedding index order")
def _():
    if not (C.POST_EMB_INDEX.exists() and C.POSTINGS_PARQUET.exists()):
        print("      skipped: build the caches first")
        return
    postings = pd.read_parquet(C.POSTINGS_PARQUET, columns=["job_id"])
    index = pd.read_parquet(C.POST_EMB_INDEX)
    assert np.array_equal(postings.job_id.to_numpy(), index.job_id.to_numpy()), (
        "postings_clean.parquet and post_emb_index.parquet are out of order -- "
        "retrieval would return the wrong jobs"
    )


def main() -> int:
    print()
    if _FAILURES:
        print(f"\n{len(_FAILURES)} check(s) failed: {', '.join(_FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
