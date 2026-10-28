"""Train and evaluate the three-class fit classifier.

Two candidates (multinomial logistic regression, histogram gradient boosting)
are scored against two baselines that use similarity alone. If neither trained
model beats the baselines, the extra machinery is not justified and the metrics
file says so -- that is a real result for the write-up, not a failure to hide.

Evaluation is deliberately two-sided:

    classification   macro F1, per-class report, confusion matrix
    ranking          Precision@10, NDCG@10, MRR over a held-out candidate pool

Ranking is what the product actually does -- a swipe deck is a ranked list, so
a model with good F1 but poor NDCG would still show users bad first cards.
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import embed
from .features import FEATURE_NAMES, build_feature_matrix, pair_cosines
from .pairs import build as build_pairs_and_data

# Relevance weights for NDCG: a Good Fit at the top of the deck is worth more
# than a Partial Fit, and a No Fit contributes nothing.
RELEVANCE = {"Good Fit": 2.0, "Partial Fit": 1.0, "No Fit": 0.0}


# --- data assembly ----------------------------------------------------------


def assemble(rebuild_pairs: bool = False):
    """Features and labels for every pair, plus the frames behind them."""
    pairs, postings, resumes = build_pairs_and_data(rebuild=rebuild_pairs)

    post_mini, post_ids = embed.load_posting_vectors("minilm")
    res_mini, res_ids = embed.load_resume_vectors("minilm")
    post_tfidf, _ = embed.load_posting_vectors("tfidf")
    res_tfidf, _ = embed.load_resume_vectors("tfidf")

    cos_mini = pair_cosines(
        pairs, post_matrix=post_mini, post_ids=post_ids,
        resume_matrix=res_mini, resume_ids=res_ids,
    )
    cos_tfidf = pair_cosines(
        pairs, post_matrix=post_tfidf, post_ids=post_ids,
        resume_matrix=res_tfidf, resume_ids=res_ids,
    )

    X = build_feature_matrix(
        pairs, postings, resumes, cos_minilm=cos_mini, cos_tfidf=cos_tfidf
    )
    y = pairs["label_id"].to_numpy()
    return pairs, postings, resumes, X, y


# --- ranking metrics --------------------------------------------------------


def _dcg(relevances: np.ndarray) -> float:
    return float(np.sum(relevances / np.log2(np.arange(2, len(relevances) + 2))))


def ranking_metrics(
    scores: np.ndarray, pairs: pd.DataFrame, *, k: int = 10
) -> dict[str, float]:
    """Mean Precision@k, NDCG@k and MRR, computed per resume.

    Each resume's candidates are ranked by `scores`; a Good Fit counts as a hit.
    Resumes with no Good Fit in their candidate set are skipped -- precision is
    undefined there and including them would just dilute the average.
    """
    frame = pairs[["resume_id", "label"]].copy()
    frame["score"] = scores
    frame["relevance"] = frame["label"].map(RELEVANCE)

    precisions, ndcgs, rrs = [], [], []
    for _, group in frame.groupby("resume_id", sort=False):
        if not (group["label"] == "Good Fit").any():
            continue
        ordered = group.sort_values("score", ascending=False)
        rel = ordered["relevance"].to_numpy()

        top = ordered.head(k)
        precisions.append(float((top["label"] == "Good Fit").mean()))

        ideal = np.sort(rel)[::-1]
        denominator = _dcg(ideal[:k])
        ndcgs.append(_dcg(rel[:k]) / denominator if denominator else 0.0)

        hits = np.flatnonzero(ordered["label"].to_numpy() == "Good Fit")
        rrs.append(1.0 / (hits[0] + 1))

    return {
        f"precision_at_{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"ndcg_at_{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "mrr": float(np.mean(rrs)) if rrs else 0.0,
        "n_resumes_scored": len(precisions),
    }


def evaluate(name: str, y_true, y_pred, scores, pairs) -> dict:
    """Classification + ranking metrics for one model."""
    result = {
        "name": name,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float((y_true == y_pred).mean()),
        "per_class": classification_report(
            y_true, y_pred, target_names=C.LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    result.update(ranking_metrics(scores, pairs))
    return result


# --- baselines --------------------------------------------------------------


def threshold_baseline(
    cos_train: np.ndarray, y_train: np.ndarray, cos_test: np.ndarray
) -> np.ndarray:
    """Bucket cosine into three tiers using train-set class proportions.

    Cut points come from the training label distribution, so the baseline gets
    the same class-prior information the classifier does. Picking thresholds on
    the test set instead would flatter the baseline with information it would
    never have in production.
    """
    shares = np.bincount(y_train, minlength=3) / len(y_train)
    # LABELS run worst -> best, so the lowest cosines become No Fit.
    q_no, q_partial = shares[0], shares[0] + shares[1]
    lo, hi = np.quantile(cos_train, [q_no, q_partial])
    return np.digitize(cos_test, [lo, hi])


# --- training ---------------------------------------------------------------


def train(*, rebuild_pairs: bool = False) -> dict:
    pairs, postings, resumes, X, y = assemble(rebuild_pairs=rebuild_pairs)

    is_train = (pairs["split"] == "train").to_numpy()
    X_train, X_test = X[is_train], X[~is_train]
    y_train, y_test = y[is_train], y[~is_train]
    test_pairs = pairs[~is_train]

    print(f"train {len(X_train):,} pairs   test {len(X_test):,} pairs")
    print(f"features: {len(FEATURE_NAMES)}\n")

    results = []

    # --- baselines: similarity alone ---
    for feature, label in [("cos_minilm", "baseline: MiniLM cosine"),
                           ("cos_tfidf", "baseline: TF-IDF cosine")]:
        cos_train = X_train[feature].to_numpy()
        cos_test = X_test[feature].to_numpy()
        pred = threshold_baseline(cos_train, y_train, cos_test)
        results.append(evaluate(label, y_test, pred, cos_test, test_pairs))

    # --- trained models ---
    candidates = {
        "logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=C.SEED
            ),
        ),
        "gradient boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=C.SEED,
        ),
    }

    fitted = {}
    for name, estimator in candidates.items():
        print(f"fitting {name} ...")
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        # Rank by P(Good Fit) -- the quantity the swipe deck is sorted on.
        good = estimator.predict_proba(X_test)[:, C.LABEL_TO_ID["Good Fit"]]
        results.append(evaluate(name, y_test, pred, good, test_pairs))
        fitted[name] = estimator

    # --- report ---
    print(f"\n{'model':<26} {'macro F1':>9} {'acc':>7} {'P@10':>7} {'NDCG@10':>8} {'MRR':>7}")
    print("-" * 68)
    for r in results:
        print(
            f"{r['name']:<26} {r['macro_f1']:>9.3f} {r['accuracy']:>7.3f} "
            f"{r['precision_at_10']:>7.3f} {r['ndcg_at_10']:>8.3f} {r['mrr']:>7.3f}"
        )

    trained = [r for r in results if not r["name"].startswith("baseline")]
    best = max(trained, key=lambda r: r["macro_f1"])
    best_baseline = max(
        (r for r in results if r["name"].startswith("baseline")),
        key=lambda r: r["macro_f1"],
    )
    beats = (
        best["macro_f1"] > best_baseline["macro_f1"]
        and best["ndcg_at_10"] >= best_baseline["ndcg_at_10"]
    )

    print(f"\nselected: {best['name']}")
    print(
        f"beats best baseline on both macro F1 and NDCG@10: {beats}"
        + ("" if beats else "  <-- the added complexity is NOT justified; report this")
    )

    print("\nper-class report (selected model)")
    for name in C.LABELS:
        row = best["per_class"][name]
        print(
            f"  {name:<14} precision {row['precision']:.3f}  "
            f"recall {row['recall']:.3f}  f1 {row['f1-score']:.3f}  n={int(row['support'])}"
        )

    print("\nconfusion matrix (rows = true, cols = predicted)")
    print(f"  {'':<14}" + "".join(f"{n:>14}" for n in C.LABELS))
    for name, row in zip(C.LABELS, best["confusion_matrix"], strict=True):
        print(f"  {name:<14}" + "".join(f"{v:>14,}" for v in row))

    model = fitted[best["name"]]
    fairness = fairness_report(model, X_test, test_pairs, resumes, postings)

    C.ensure_artifacts()
    joblib.dump(
        {"model": model, "feature_names": FEATURE_NAMES, "name": best["name"]},
        C.FIT_MODEL,
    )

    metrics = {
        "selected": best["name"],
        "beats_baselines": bool(beats),
        "n_train_pairs": int(len(X_train)),
        "n_test_pairs": int(len(X_test)),
        "results": results,
        "fairness": fairness,
        "coefficients": coefficient_report(model),
    }
    C.METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nwrote {C.FIT_MODEL.name} and {C.METRICS_JSON.name}")
    return metrics


# --- interpretation and fairness -------------------------------------------


def coefficient_report(model) -> dict | None:
    """Logistic coefficients per class, for the write-up. None for the tree."""
    if not hasattr(model, "named_steps"):
        return None
    lr = model.named_steps.get("logisticregression")
    if lr is None:
        return None
    return {
        label: dict(zip(FEATURE_NAMES, row.round(4).tolist(), strict=True))
        for label, row in zip(C.LABELS, lr.coef_, strict=True)
    }


def _rate_table(frame, group_col, *, min_n=15):
    """Good-Fit rate grouped by `group_col`, dropping groups too thin to read.

    A group with a handful of test pairs produces a rate that is mostly noise;
    `min_n` keeps those out of the spread statistic so one sparse category
    cannot masquerade as the widest disparity.
    """
    table = (
        frame.groupby(group_col)
        .agg(n=("p_good", "size"), good_rate=("predicted_good", "mean"),
             mean_p_good=("p_good", "mean"))
        .sort_values("good_rate", ascending=False)
    )
    return table[table["n"] >= min_n]


def fairness_report(model, X_test, test_pairs, resumes, postings) -> dict:
    """Good-Fit rate by resume category and by posting state.

    D1 names both explicitly as bias risks ("the model could be biased toward
    certain names, locations, writing styles"). This measures whether the
    model's predictions are even across each; it does not correct them -- a
    wide spread is a finding to report, not something to quietly rescale away.
    Groups with fewer than 15 test pairs are dropped from the spread stat
    (still visible in the full table) since a rate over a handful of pairs is
    noise, not evidence of bias.
    """
    good = model.predict_proba(X_test)[:, C.LABEL_TO_ID["Good Fit"]]
    frame = test_pairs[["resume_id", "job_id", "label"]].copy()
    frame["p_good"] = good
    frame["predicted_good"] = (
        model.predict(X_test) == C.LABEL_TO_ID["Good Fit"]
    ).astype(int)

    frame = frame.merge(
        resumes[["resume_id", "category"]], on="resume_id", how="left"
    ).merge(
        postings[["job_id", "state"]], on="job_id", how="left"
    )

    by_category = _rate_table(frame, "category")
    by_state = _rate_table(frame, "state")

    print("\nfairness: predicted Good-Fit rate by resume category")
    print(f"  spread {by_category.good_rate.min():.3f} .. {by_category.good_rate.max():.3f}"
          f"  ({len(by_category)} categories with >= 15 test pairs)")
    print(by_category.round(3).head(5).to_string())
    print("  ...")
    print(by_category.round(3).tail(5).to_string())

    print("\nfairness: predicted Good-Fit rate by posting state")
    if len(by_state) >= 2:
        print(f"  spread {by_state.good_rate.min():.3f} .. {by_state.good_rate.max():.3f}"
              f"  ({len(by_state)} states with >= 15 test pairs)")
        print(by_state.round(3).head(5).to_string())
        print("  ...")
        print(by_state.round(3).tail(5).to_string())
    else:
        print("  too few states with >= 15 test pairs to report a spread")

    return {
        "by_resume_category": by_category.round(4).to_dict(orient="index"),
        "good_rate_spread": float(by_category.good_rate.max() - by_category.good_rate.min()),
        "by_state": by_state.round(4).to_dict(orient="index"),
        "state_good_rate_spread": (
            float(by_state.good_rate.max() - by_state.good_rate.min())
            if len(by_state) >= 2 else None
        ),
    }


def load_model():
    """Load the persisted classifier bundle."""
    if not C.FIT_MODEL.exists():
        raise FileNotFoundError(
            f"{C.FIT_MODEL} is missing -- run `python -m src.model --train` first."
        )
    return joblib.load(C.FIT_MODEL)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--rebuild-pairs", action="store_true")
    args = ap.parse_args()

    if args.train:
        train(rebuild_pairs=args.rebuild_pairs)
    else:
        ap.error("pass --train")


if __name__ == "__main__":
    main()
