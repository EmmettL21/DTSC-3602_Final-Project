"""Weak-labeled (resume, posting) pairs and a leakage-free train/test split.

Assigns each posting a category via `taxonomy.assign_category`, then samples
pairs per resume: K same-category (Good Fit), K adjacent (Partial Fit) and 2K
unrelated (No Fit).

The split is **doubly disjoint**: resume IDs and job IDs are each partitioned
80/20 *before* any pair is drawn, and each side then samples only within its own
partition. No resume and no posting appears on both sides. A plain random split
over pairs would let the model memorize a specific resume from its training
pairs and then be scored on that same resume, inflating every metric.

Partitioning first rather than splitting finished pairs matters: drawing pairs
globally and keeping only those with both endpoints held out throws away ~32%
of the sample and leaves a test set too thin to rank against.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C
from . import taxonomy as T
from .data_load import load_postings, load_resumes


def assign_posting_categories(postings: pd.DataFrame) -> pd.DataFrame:
    """Add `posting_category` and `category_score` columns."""
    results = [
        T.assign_category(title, set(abrs), list(inds))
        for title, abrs, inds in zip(
            postings["title"].fillna(""),
            postings["skill_abrs"],
            postings["industries"],
            strict=True,
        )
    ]
    out = postings.copy()
    out["posting_category"] = [r[0] for r in results]
    out["category_score"] = [r[1] for r in results]
    return out


def build_pairs(
    postings: pd.DataFrame,
    resumes: pd.DataFrame,
    *,
    k: int = C.PAIRS_PER_RESUME_K,
    seed: int = C.SEED,
) -> pd.DataFrame:
    """Sample `4k` labeled pairs per resume: k Good, k Partial, 2k No Fit."""
    rng = np.random.default_rng(seed)

    labeled = postings[postings["posting_category"].notna()]
    if labeled.empty:
        raise RuntimeError("no postings were assigned a category -- check taxonomy.py")

    # job_id arrays bucketed by category, so sampling is an index draw rather
    # than a dataframe filter per resume.
    by_category: dict[str, np.ndarray] = {
        name: group["job_id"].to_numpy()
        for name, group in labeled.groupby("posting_category", observed=True)
    }
    all_categories = set(by_category)

    rows: list[tuple[int, int, str]] = []

    def take(pool: np.ndarray, n: int) -> np.ndarray:
        if len(pool) == 0:
            return np.empty(0, dtype=np.int64)
        return rng.choice(pool, size=min(n, len(pool)), replace=False)

    for resume_id, category in zip(
        resumes["resume_id"], resumes["category"], strict=True
    ):
        if category not in T.CATEGORIES:
            continue

        # Good: same category.
        for job_id in take(by_category.get(category, np.empty(0, np.int64)), k):
            rows.append((resume_id, int(job_id), "Good Fit"))

        # Partial: an adjacent category, spread across whichever are available.
        related = sorted(T.RELATED[category] & all_categories)
        if related:
            per = max(1, k // len(related))
            drawn = 0
            for other in related:
                if drawn >= k:
                    break
                for job_id in take(by_category[other], min(per, k - drawn)):
                    rows.append((resume_id, int(job_id), "Partial Fit"))
                    drawn += 1

        # No Fit: categories that are neither the resume's own nor adjacent.
        unrelated = sorted(all_categories - {category} - T.RELATED[category])
        if unrelated:
            picks = rng.choice(unrelated, size=min(2 * k, len(unrelated)), replace=False)
            for other in picks:
                for job_id in take(by_category[other], 1):
                    rows.append((resume_id, int(job_id), "No Fit"))

    pairs = pd.DataFrame(rows, columns=["resume_id", "job_id", "label"])
    pairs = pairs.drop_duplicates(subset=["resume_id", "job_id"]).reset_index(drop=True)
    pairs["label_id"] = pairs["label"].map(C.LABEL_TO_ID).astype("int8")
    return pairs


def group_split(
    ids: np.ndarray, *, test_frac: float, seed: int
) -> tuple[set[int], set[int]]:
    """Partition an ID array into `(train_ids, test_ids)`."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.sort(ids))
    n_test = int(len(shuffled) * test_frac)
    return set(shuffled[n_test:].tolist()), set(shuffled[:n_test].tolist())


def build(
    *,
    k: int = C.PAIRS_PER_RESUME_K,
    test_frac: float = C.TEST_GROUP_FRAC,
    rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return `(pairs, postings_with_categories, resumes)`.

    `postings` carries a `split` column too, so downstream ranking evaluation
    can draw its candidate pool from held-out postings only.
    """
    postings = assign_posting_categories(load_postings())
    resumes = load_resumes()

    # Partition first. Seeds are offset so the two partitions are independent.
    labeled_jobs = postings.loc[postings["posting_category"].notna(), "job_id"].to_numpy()
    train_jobs, test_jobs = group_split(labeled_jobs, test_frac=test_frac, seed=C.SEED)
    train_resumes, test_resumes = group_split(
        resumes["resume_id"].to_numpy(), test_frac=test_frac, seed=C.SEED + 1
    )

    postings["split"] = np.where(
        postings["job_id"].isin(test_jobs),
        "test",
        np.where(postings["job_id"].isin(train_jobs), "train", "unlabeled"),
    )
    resumes["split"] = np.where(
        resumes["resume_id"].isin(test_resumes), "test", "train"
    )

    if C.PAIRS_PARQUET.exists() and not rebuild:
        return pd.read_parquet(C.PAIRS_PARQUET), postings, resumes

    C.ensure_artifacts()
    frames = []
    for split, seed_offset in (("train", 10), ("test", 20)):
        side = build_pairs(
            postings[postings["split"] == split],
            resumes[resumes["split"] == split],
            k=k,
            seed=C.SEED + seed_offset,
        )
        side["split"] = split
        frames.append(side)

    pairs = pd.concat(frames, ignore_index=True)
    pairs.to_parquet(C.PAIRS_PARQUET, index=False)
    return pairs, postings, resumes


def _check() -> None:
    pairs, postings, resumes = build(rebuild=True)

    assigned = postings["posting_category"].notna()
    print(f"postings with a category: {assigned.sum():,} / {len(postings):,} "
          f"({100 * assigned.mean():.1f}%)")
    print()
    print("postings per category")
    counts = postings.loc[assigned, "posting_category"].value_counts()
    for name in T.CATEGORY_NAMES:
        print(f"  {name:<24} {counts.get(name, 0):>7,}")

    print()
    print(f"pairs: {len(pairs):,}")
    print(pairs["split"].value_counts().to_string())

    train = pairs[pairs.split == "train"]
    test = pairs[pairs.split == "test"]

    r_overlap = set(train.resume_id) & set(test.resume_id)
    j_overlap = set(train.job_id) & set(test.job_id)
    assert not r_overlap, f"{len(r_overlap)} resume IDs leak across the split"
    assert not j_overlap, f"{len(j_overlap)} job IDs leak across the split"
    print("\nleakage check: 0 shared resume IDs, 0 shared job IDs  OK")

    print("\nlabel balance (% within split)")
    balance = (
        pairs.groupby("split")["label"].value_counts(normalize=True).unstack() * 100
    )
    print(balance.round(1).to_string())

    print(
        f"\nresumes  train {train.resume_id.nunique():,}  test {test.resume_id.nunique():,}"
    )
    print(f"jobs     train {train.job_id.nunique():,}  test {test.job_id.nunique():,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="rebuild and verify the split")
    ap.add_argument("-k", type=int, default=C.PAIRS_PER_RESUME_K)
    args = ap.parse_args()

    if args.check:
        _check()
    else:
        pairs, _, _ = build(k=args.k, rebuild=True)
        print(f"wrote {len(pairs):,} pairs to {C.PAIRS_PARQUET}")


if __name__ == "__main__":
    main()
