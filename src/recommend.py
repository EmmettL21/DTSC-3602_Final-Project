"""Serving path: resume text + preferences -> a ranked deck of job cards.

Two stages, which is what keeps it fast enough to feel instant in the UI:

    retrieve   cosine against all 123k cached posting vectors, keep the top 500
    rerank     build the full feature set for those 500 and sort by P(Good Fit)

Brute-force retrieval is a single 123k x 384 matrix-vector product (~190MB,
tens of milliseconds), so an approximate index like FAISS would add a
dependency and an accuracy loss to solve a problem we do not have.

The same function serves the employer direction -- `recommend_candidates`
swaps which side is the query and which is the corpus.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C
from . import embed
from .data_load import load_resumes
from .features import Preferences, build_feature_matrix
from .model import load_model
from .pairs import assign_posting_categories
from .data_load import load_postings

CARD_COLUMNS = [
    "job_id", "title", "company_name", "location", "state",
    "formatted_work_type", "formatted_experience_level", "salary_annual",
    "remote_allowed", "skill_names", "job_posting_url", "application_url",
    "posting_category",
]


@lru_cache(maxsize=1)
def _resources():
    """Load every artifact once per process. The UI reruns constantly."""
    postings = assign_posting_categories(load_postings())
    resumes = load_resumes()
    post_mini, post_ids = embed.load_posting_vectors("minilm")
    post_tfidf, _ = embed.load_posting_vectors("tfidf")
    res_mini, res_ids = embed.load_resume_vectors("minilm")
    bundle = load_model()

    # Row order of the embedding cache must match the postings frame, since we
    # index back into it by position after the cosine sort.
    if not np.array_equal(postings["job_id"].to_numpy(), post_ids):
        raise RuntimeError(
            "posting embeddings are out of sync with postings_clean.parquet -- "
            "rerun `python -m src.embed --rebuild`"
        )

    return {
        "postings": postings,
        "resumes": resumes,
        "post_mini": post_mini,
        "post_tfidf": post_tfidf,
        "post_ids": post_ids,
        "res_mini": res_mini,
        "res_ids": res_ids,
        "model": bundle["model"],
    }


def infer_category(vec: np.ndarray, *, k: int = 9) -> str:
    """Guess a pasted resume's category from its nearest labeled neighbours.

    The feature set needs a category to score title keywords against, and a
    pasted resume arrives without one. Reusing the labeled resume embeddings we
    already cache is cheaper and more honest than training a second classifier.

    Votes are weighted by cosine similarity rather than counted flat: several
    resume categories are deliberately adjacent in the taxonomy (HEALTHCARE and
    FITNESS, for one), so a flat k=5 mode vote can let two so-so neighbours
    outvote one much closer match. A nurse's resume whose single nearest
    neighbour is HEALTHCARE at 0.68 cosine, beaten 2-1 by two FITNESS resumes
    at 0.67 and 0.66, is exactly the failure this weighting fixes.
    """
    r = _resources()
    sims = r["res_mini"] @ vec
    top = np.argpartition(-sims, k)[:k]
    categories = r["resumes"].set_index("resume_id")["category"]

    weights: dict[str, float] = {}
    for i in top:
        cat = categories.get(int(r["res_ids"][i]))
        if cat:
            weights[cat] = weights.get(cat, 0.0) + max(float(sims[i]), 0.0)

    return max(weights, key=weights.get)


def recommend(
    resume_text: str,
    prefs: Preferences | None = None,
    *,
    top_n: int = 50,
    depth: int = C.RETRIEVAL_DEPTH,
    category: str | None = None,
) -> pd.DataFrame:
    """Ranked job cards for one resume."""
    r = _resources()
    prefs = prefs or Preferences()

    q_mini = embed.embed_query([resume_text], "minilm")[0]
    q_tfidf = embed.embed_query([resume_text], "tfidf")[0]
    category = category or infer_category(q_mini)

    # --- retrieve ---
    sims = r["post_mini"] @ q_mini
    depth = min(depth, len(sims))
    candidate_rows = np.argpartition(-sims, depth - 1)[:depth]
    candidates = r["postings"].iloc[candidate_rows]

    # --- rerank ---
    QUERY_ID = -1
    pairs = pd.DataFrame(
        {"resume_id": QUERY_ID, "job_id": candidates["job_id"].to_numpy()}
    )
    query_frame = pd.DataFrame(
        [{"resume_id": QUERY_ID, "resume_text": resume_text, "category": category}]
    )

    X = build_feature_matrix(
        pairs,
        candidates,
        query_frame,
        cos_minilm=sims[candidate_rows],
        cos_tfidf=r["post_tfidf"][candidate_rows] @ q_tfidf,
        prefs={QUERY_ID: prefs},
    )

    proba = r["model"].predict_proba(X)
    deck = candidates[CARD_COLUMNS].copy().reset_index(drop=True)
    deck["p_good"] = proba[:, C.LABEL_TO_ID["Good Fit"]]
    deck["p_partial"] = proba[:, C.LABEL_TO_ID["Partial Fit"]]
    deck["fit_tier"] = [C.LABELS[i] for i in proba.argmax(axis=1)]
    deck["cos_minilm"] = X["cos_minilm"].to_numpy()
    deck["inferred_category"] = category

    deck["why"] = [
        explain(deck.iloc[i], X.iloc[i], resume_text, prefs) for i in range(len(deck))
    ]

    return deck.sort_values("p_good", ascending=False).head(top_n).reset_index(drop=True)


def recommend_candidates(
    posting_text: str, *, top_n: int = 50, category: str | None = None
) -> pd.DataFrame:
    """The employer direction: a job description -> ranked resumes.

    Retrieval only. The reranker's preference features describe a job seeker's
    survey answers and have no employer-side equivalent, so ranking here is
    pure cosine rather than a classifier score that would be mostly padding.
    """
    r = _resources()
    q = embed.embed_query([posting_text], "minilm")[0]
    sims = r["res_mini"] @ q

    out = r["resumes"].copy()
    out["similarity"] = sims
    if category:
        out = out[out["category"] == category]
    return out.nlargest(top_n, "similarity")[
        ["resume_id", "category", "similarity", "resume_text"]
    ].reset_index(drop=True)


def explain(card: pd.Series, features: pd.Series, resume_text: str, prefs: Preferences) -> str:
    """A short, honest reason this card surfaced.

    Built from the raw interpretable features rather than model internals, so
    the wording stays the same whichever estimator was selected -- and so it
    never claims a reason the data does not support.
    """
    reasons: list[str] = []

    overlap_n = int(features.get("skill_overlap_n", 0))
    if overlap_n:
        reasons.append(f"{overlap_n} shared skill area{'s' if overlap_n > 1 else ''}")

    cos = float(features.get("cos_minilm", 0))
    if cos >= 0.5:
        reasons.append("strong wording overlap with your resume")
    elif cos >= 0.35:
        reasons.append("moderate wording overlap")

    if features.get("title_hits", 0):
        reasons.append(f"title matches your {card['inferred_category'].lower()} background")

    if prefs.state and features.get("state_match", 0):
        reasons.append("remote" if card.get("remote_allowed") else f"in {prefs.state}")
    if prefs.target_salary and features.get("salary_meets_target", 0):
        reasons.append("meets your salary target")

    return ", ".join(reasons).capitalize() if reasons else "Broad match on overall content"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume-id", type=int, help="score a resume from the dataset")
    ap.add_argument("--category", help="pick a random resume from this category instead")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--salary", type=float, default=None)
    ap.add_argument("--state", default=None)
    args = ap.parse_args()

    resumes = load_resumes()
    if args.resume_id:
        row = resumes.loc[resumes.resume_id == args.resume_id]
        if row.empty:
            raise SystemExit(f"no resume with id {args.resume_id}")
        row = row.iloc[0]
    elif args.category:
        pool = resumes[resumes.category == args.category.upper()]
        if pool.empty:
            raise SystemExit(f"no resumes in category {args.category!r}")
        row = pool.sample(1, random_state=C.SEED).iloc[0]
    else:
        row = resumes.sample(1, random_state=C.SEED).iloc[0]

    prefs = Preferences(target_salary=args.salary, state=args.state)
    print(f"resume {row.resume_id}  true category: {row.category}")
    print(f"preferences: {asdict(prefs)}\n")

    # The dataset already labels this resume -- pass that category through
    # rather than re-infer it. k-NN inference exists for pasted resumes that
    # arrive with no label; using it here would throw away known ground truth
    # and substitute a noisier guess for no reason.
    deck = recommend(row.resume_text, prefs, top_n=args.top, category=row.category)
    print(f"category used: {deck.inferred_category.iloc[0]}\n")

    for i, card in deck.iterrows():
        salary = f"${card.salary_annual:,.0f}" if pd.notna(card.salary_annual) else "—"
        print(f"{i + 1:>2}. [{card.fit_tier:<11} p={card.p_good:.2f}] {card.title}")
        print(f"    {card.company_name} · {card.location} · {salary} · {card.formatted_work_type}")
        print(f"    {card.why}")

    hit_rate = (deck["posting_category"] == row.category).mean()
    print(f"\ntop-{len(deck)} share matching the resume's true category: {hit_rate:.0%}")


if __name__ == "__main__":
    main()
