"""Two vector representations of the same text, cached to disk.

    MiniLM    all-MiniLM-L6-v2 sentence embeddings (384-d) -- what D1 promised
    TF-IDF    word 1-2 grams -> TruncatedSVD (300-d)       -- the baseline

Both are L2-normalized, so a cosine similarity is a plain dot product and the
two feed identical downstream code. Keeping the baseline is the point: it is
what tells us whether the neural embeddings actually earn their runtime cost,
rather than assuming they do.

Encoding all 123k postings on CPU takes roughly half an hour, so everything is
written to `artifacts/` and reloaded thereafter. Use `--sample N` while
iterating.
"""

from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from . import config as C
from .data_load import load_postings, load_resumes

_MODEL = None


def get_model():
    """Lazily load MiniLM, preferring the quantized ONNX build.

    The import is deferred because torch alone costs several seconds, and most
    callers (the pairs builder, the metrics reader) never need the model at all.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    from sentence_transformers import SentenceTransformer

    if C.EMBED_BACKEND == "onnx":
        try:
            _MODEL = SentenceTransformer(
                C.EMBED_MODEL_NAME,
                device="cpu",
                backend="onnx",
                model_kwargs={"file_name": C.EMBED_ONNX_FILE},
            )
        except Exception as exc:  # optimum/onnxruntime missing, or no such file
            print(f"ONNX backend unavailable ({exc.__class__.__name__}); using torch.")
            _MODEL = SentenceTransformer(C.EMBED_MODEL_NAME, device="cpu")
    else:
        _MODEL = SentenceTransformer(C.EMBED_MODEL_NAME, device="cpu")

    _MODEL.max_seq_length = C.EMBED_MAX_SEQ_LENGTH
    return _MODEL


def encode(texts, *, show_progress: bool = True) -> np.ndarray:
    """Encode to L2-normalized float32."""
    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=C.EMBED_BATCH_SIZE,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.astype(np.float32)


# --- MiniLM -----------------------------------------------------------------


def build_minilm(*, sample: int | None = None, rebuild: bool = False) -> None:
    C.ensure_artifacts()

    postings = load_postings(sample=sample)
    resumes = load_resumes()

    if rebuild or not C.RESUME_EMB_NPY.exists():
        print(f"encoding {len(resumes):,} resumes ...")
        start = time.perf_counter()
        np.save(C.RESUME_EMB_NPY, encode(resumes["resume_text"]))
        resumes[["resume_id"]].to_parquet(C.RESUME_EMB_INDEX, index=False)
        print(f"  {time.perf_counter() - start:.1f}s")

    if rebuild or not C.POST_EMB_NPY.exists():
        print(f"encoding {len(postings):,} postings ...")
        start = time.perf_counter()
        np.save(C.POST_EMB_NPY, encode(postings["posting_text"]))
        postings[["job_id"]].to_parquet(C.POST_EMB_INDEX, index=False)
        print(f"  {time.perf_counter() - start:.1f}s")


# --- TF-IDF baseline --------------------------------------------------------


def build_tfidf(*, sample: int | None = None, rebuild: bool = False) -> None:
    if not rebuild and C.TFIDF_POST_NPY.exists():
        return

    C.ensure_artifacts()
    postings = load_postings(sample=sample)
    resumes = load_resumes()

    print(f"fitting TF-IDF + SVD on {len(postings) + len(resumes):,} documents ...")
    start = time.perf_counter()

    # Fit on both corpora together so resumes and postings land in one shared
    # vocabulary -- fitting on postings alone would drop resume-only terms and
    # make every cosine artificially low.
    corpus = pd.concat(
        [postings["posting_text"], resumes["resume_text"]], ignore_index=True
    )

    pipeline = make_pipeline(
        TfidfVectorizer(
            max_features=C.TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=3,
            sublinear_tf=True,
        ),
        TruncatedSVD(n_components=C.SVD_COMPONENTS, random_state=C.SEED),
        Normalizer(copy=False),
    )
    reduced = pipeline.fit_transform(corpus).astype(np.float32)

    np.save(C.TFIDF_POST_NPY, reduced[: len(postings)])
    np.save(C.TFIDF_RESUME_NPY, reduced[len(postings) :])
    joblib.dump(pipeline, C.TFIDF_MODEL)

    explained = pipeline.named_steps["truncatedsvd"].explained_variance_ratio_.sum()
    print(f"  {time.perf_counter() - start:.1f}s, SVD explains {explained:.1%} of variance")


# --- loading ----------------------------------------------------------------


def load_posting_vectors(kind: str = "minilm") -> tuple[np.ndarray, np.ndarray]:
    """Return `(matrix, job_ids)` aligned row-for-row."""
    path = C.POST_EMB_NPY if kind == "minilm" else C.TFIDF_POST_NPY
    matrix = np.load(path)
    job_ids = pd.read_parquet(C.POST_EMB_INDEX)["job_id"].to_numpy()
    _check_aligned(matrix, job_ids, path.name)
    return matrix, job_ids


def load_resume_vectors(kind: str = "minilm") -> tuple[np.ndarray, np.ndarray]:
    """Return `(matrix, resume_ids)` aligned row-for-row."""
    path = C.RESUME_EMB_NPY if kind == "minilm" else C.TFIDF_RESUME_NPY
    matrix = np.load(path)
    resume_ids = pd.read_parquet(C.RESUME_EMB_INDEX)["resume_id"].to_numpy()
    _check_aligned(matrix, resume_ids, path.name)
    return matrix, resume_ids


def _check_aligned(matrix: np.ndarray, ids: np.ndarray, name: str) -> None:
    if len(matrix) != len(ids):
        raise RuntimeError(
            f"{name} has {len(matrix):,} rows but its index has {len(ids):,}. "
            "The cache is stale -- rerun `python -m src.embed --build --rebuild`."
        )


def row_lookup(ids: np.ndarray) -> dict[int, int]:
    """id -> row index, for gathering pair vectors."""
    return {int(v): i for i, v in enumerate(ids)}


def embed_query(texts, kind: str = "minilm") -> np.ndarray:
    """Embed free text at serving time, in whichever space is requested."""
    if kind == "minilm":
        return encode(texts, show_progress=False)
    pipeline = joblib.load(C.TFIDF_MODEL)
    return pipeline.transform(list(texts)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="build any missing caches")
    ap.add_argument("--rebuild", action="store_true", help="rebuild even if cached")
    ap.add_argument("--sample", type=int, default=None, help="limit postings, for dev")
    ap.add_argument("--skip-minilm", action="store_true")
    ap.add_argument("--skip-tfidf", action="store_true")
    args = ap.parse_args()

    if not (args.build or args.rebuild):
        ap.error("pass --build or --rebuild")

    if not args.skip_minilm:
        build_minilm(sample=args.sample, rebuild=args.rebuild)
    if not args.skip_tfidf:
        build_tfidf(sample=args.sample, rebuild=args.rebuild)

    for kind in ("minilm", "tfidf"):
        try:
            posts, job_ids = load_posting_vectors(kind)
            res, resume_ids = load_resume_vectors(kind)
        except FileNotFoundError:
            continue
        print(
            f"{kind:<8} postings {posts.shape} ids {job_ids.shape}   "
            f"resumes {res.shape} ids {resume_ids.shape}"
        )


if __name__ == "__main__":
    main()
