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
# Not joined into `load_postings` -- documented as unused sources in the D2
# data dictionary rather than silently ignored.
COMPANY_INDUSTRIES_CSV = POSTINGS_DIR / "companies" / "company_industries.csv"
COMPANY_SPECIALITIES_CSV = POSTINGS_DIR / "companies" / "company_specialities.csv"

RESUMES_CSV = ROOT / "resumes" / "Resume" / "Resume.csv"

# --- generated artifacts ----------------------------------------------------
ARTIFACTS = ROOT / "artifacts"
POSTINGS_PARQUET = ARTIFACTS / "postings_clean.parquet"
RESUMES_PARQUET = ARTIFACTS / "resumes_clean.parquet"
# D2 audit outputs. Additive siblings of the two parquets above -- D1's
# pipeline (model.py, recommend.py, the swipe app) keeps reading the originals.
POSTINGS_AUDITED_PARQUET = ARTIFACTS / "postings_audited.parquet"
COMPANIES_AUDITED_PARQUET = ARTIFACTS / "companies_audited.parquet"
DATA_DICTIONARY_CSV = ARTIFACTS / "data_dictionary.csv"
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

# --- D2 data audit -----------------------------------------------------------
# Fixed sanity bounds for `salary_annual`, not a percentile cut -- a percentile
# would clip legitimate high executive salaries along with the clear data-entry
# errors (postings.csv has rows up to $535.6M). Anything outside this range is
# flagged, never dropped.
SALARY_SANITY_FLOOR = 1_000.0
SALARY_SANITY_CEILING = 1_000_000.0

# A posting sharing (title, company_name, location) with another is flagged as
# a likely duplicate -- re-posted listings, not necessarily bad job_ids.
DUPLICATE_KEY_COLS = ["title", "company_name", "location"]

# Tracked (not gitignored) D2 write-up assets -- `artifacts/` is gitignored, so
# anything that must survive in the repo (the reserved-split fingerprint, the
# chosen figures) lives under `write ups/` instead.
WRITEUPS_DIR = ROOT / "write ups"
D2_FIGURES_DIR = WRITEUPS_DIR / "d2_figures"
TEST_SPLIT_FREEZE_JSON = WRITEUPS_DIR / "d2_test_split_freeze.json"


def ensure_artifacts() -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS
