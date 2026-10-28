"""Shared paths and constants for the DTSC3602 Group 9 job-matching model."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- raw data ---------------------------------------------------------------
POSTINGS_DIR = ROOT / "postings"
POSTINGS_CSV = POSTINGS_DIR / "postings.csv"
JOB_SKILLS_CSV = POSTINGS_DIR / "jobs" / "job_skills.csv"
JOB_INDUSTRIES_CSV = POSTINGS_DIR / "jobs" / "job_industries.csv"
SALARIES_CSV = POSTINGS_DIR / "jobs" / "salaries.csv"
BENEFITS_CSV = POSTINGS_DIR / "jobs" / "benefits.csv"
SKILL_MAP_CSV = POSTINGS_DIR / "mappings" / "skills.csv"
INDUSTRY_MAP_CSV = POSTINGS_DIR / "mappings" / "industries.csv"
COMPANIES_CSV = POSTINGS_DIR / "companies" / "companies.csv"
EMPLOYEE_COUNTS_CSV = POSTINGS_DIR / "companies" / "employee_counts.csv"

RESUMES_CSV = ROOT / "resumes" / "Resume" / "Resume.csv"

# --- generated artifacts ----------------------------------------------------
ARTIFACTS = ROOT / "artifacts"
POSTINGS_PARQUET = ARTIFACTS / "postings_clean.parquet"
RESUMES_PARQUET = ARTIFACTS / "resumes_clean.parquet"
POST_EMB_NPY = ARTIFACTS / "post_emb.npy"
POST_EMB_INDEX = ARTIFACTS / "post_emb_index.parquet"
RESUME_EMB_NPY = ARTIFACTS / "resume_emb.npy"
RESUME_EMB_INDEX = ARTIFACTS / "resume_emb_index.parquet"
TFIDF_POST_NPY = ARTIFACTS / "post_tfidf_svd.npy"
TFIDF_RESUME_NPY = ARTIFACTS / "resume_tfidf_svd.npy"
TFIDF_MODEL = ARTIFACTS / "tfidf_svd.joblib"
PAIRS_PARQUET = ARTIFACTS / "pairs.parquet"
FIT_MODEL = ARTIFACTS / "fit_model.joblib"
METRICS_JSON = ARTIFACTS / "metrics.json"
SWIPE_LOG = ARTIFACTS / "swipe_log.csv"

# --- modelling constants ----------------------------------------------------
SEED = 3602

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# The int8 ONNX build of MiniLM runs ~3x faster on CPU than torch fp32 and, as
# benchmarked on this corpus, costs nothing measurable in retrieval quality.
# `embed.get_model` falls back to torch when optimum/onnxruntime is absent, so
# the pipeline still runs on a machine without it.
EMBED_BACKEND = "onnx"
EMBED_ONNX_FILE = "onnx/model_qint8_avx512_vnni.onnx"

# Tokens per document. MiniLM's own default is 256; see the table in the README
# for the measured speed/quality tradeoff behind this value.
EMBED_MAX_SEQ_LENGTH = 256
# 64 benchmarked slightly faster than 256 on this CPU -- larger batches pad more
# short documents up to the longest in the batch.
EMBED_BATCH_SIZE = 64
EMBED_DIM = 384

# MiniLM truncates at 256 word-pieces; ~2000 chars is comfortably past that, so
# the tail is dropped by the tokenizer anyway. Truncating in Python first keeps
# the cached frames small and the encode loop fast.
MAX_TEXT_CHARS = 2000

TFIDF_MAX_FEATURES = 50_000
SVD_COMPONENTS = 300

# Fit tiers. Ordered worst -> best so the integer code is also a ranking.
LABELS = ["No Fit", "Partial Fit", "Good Fit"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}

# Pair sampling: per resume, K good + K partial + 2K no-fit.
PAIRS_PER_RESUME_K = 3

# Fraction of resume IDs and of job IDs held out for the doubly-disjoint split.
TEST_GROUP_FRAC = 0.20

# Swipes faster than this are flagged as blind swipes (D1 data-quality risk).
BLIND_SWIPE_MS = 1500

# Candidates pulled by cosine before the classifier reranks them.
RETRIEVAL_DEPTH = 500


def ensure_artifacts() -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS
