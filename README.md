# Tinder for Jobs — DTSC3602 Group 9

A recommendation model that matches resumes to job postings and labels each
match **Good Fit / Partial Fit / No Fit**, plus a swipe-card demo that logs the
interaction data our research question depends on.

Evan Smith · Andrew Wallace · Ikumi Uemura · Emmett Lashbrook · Brian Aguirre

## What this does

A resume goes in; a ranked deck of job cards comes out. Two stages:

1. **Retrieve** — cosine similarity between MiniLM sentence embeddings of the
   resume and all ~124k cached posting vectors; keep the top 500.
2. **Rerank** — build the full feature set for those 500 (similarity in two
   embedding spaces, skill overlap, title match, and the survey preferences:
   salary, state, remote, experience level, work type) and sort by
   P(Good Fit) from a trained three-class classifier.

## Where the fit labels come from

Neither dataset ships a resume↔posting fit label, so we manufacture one by
**weak supervision**. `src/taxonomy.py` describes each of the 24 resume
categories three ways — LinkedIn skill abbreviations, industry-name patterns,
and job-title patterns — and each posting is assigned the highest-scoring
category. A resume and posting in the same category is a Good Fit, adjacent
categories are a Partial Fit, everything else is No Fit.

**This is the single biggest caveat in the project.** The model learns our
taxonomy's notion of fit, not a hiring outcome. Every metric below is measured
against that proxy. Hand-labeling a gold test set is the honest upgrade.

## Setup

```bash
uv venv .venv
uv pip install -r requirements.txt
```

The raw data is not in the repo. Download and unpack into `postings/` and
`resumes/`:

- https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
- https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

## Running the pipeline

```bash
python -m src.data_load --rebuild --smoke   # parse + clean + cache  (~2 min)
python -m src.pairs --check                 # weak labels + split    (~1 min)
python -m src.embed --build                 # embeddings             (see note)
python -m src.model --train                 # train + evaluate       (~1 min)
python -m src.recommend --category FINANCE   # eyeball the output
streamlit run app/streamlit_app.py          # the swipe demo
```

Everything caches to `artifacts/`, so each step runs once.

### A note on embedding time

Encoding 124k postings is the one slow step. Measured on this machine (12-core
CPU, no GPU), per the config in `src/config.py`:

| backend | max tokens | full corpus |
|---|---|---|
| torch fp32 | 256 | ~291 min |
| torch fp32 | 128 | ~163 min |
| ONNX int8 | 256 | ~97 min |
| ONNX int8 | 128 | ~41 min |

We use the ONNX int8 backend. `src/embed.py` falls back to plain torch
automatically if `optimum[onnxruntime]` is not installed, so the pipeline still
runs on a teammate's machine without it — just slower.

## Evaluation

`python -m src.model --train` scores two trained models against two baselines
that use similarity alone, and writes everything to `artifacts/metrics.json`.

- **Classification**: macro F1, per-class precision/recall, confusion matrix.
- **Ranking**: Precision@10, NDCG@10, MRR — a swipe deck *is* a ranked list, so
  a model with good F1 but poor NDCG would still show bad first cards.
- **Fairness**: predicted Good-Fit rate by resume category, reported as a
  spread rather than asserted to be even.

The split is **doubly disjoint** — resume IDs and job IDs are each partitioned
80/20 *before* pairs are drawn, so no resume and no posting appears on both
sides. `python -m src.pairs --check` asserts this.

If neither trained model beats the cosine-only baselines, `metrics.json` says
`beats_baselines: false`. That is a real result to report, not a bug.

## Data audit (D2)

`src/audit.py` layers three fixes on top of the cleaned postings frame, additive only
— nothing D1 already reads is overwritten:

- **Salary outliers** (raw `normalized_salary` reaches $535.6M) flagged outside a fixed
  $1,000–$1,000,000/yr sanity band, in a new `salary_outlier_flag` column.
- **Sentinel values** in `companies.csv` (`"0"`, `"-"`, empty, the non-ISO `"OO"`
  country code) normalized to real `NaN`, with a `*_was_sentinel` flag per field.
- **Likely-duplicate postings** (same title/company/location, different `job_id`)
  flagged via `is_likely_duplicate`, not dropped.

```bash
python -m src.audit --build --smoke        # before/after counts
python -m src.data_dictionary              # writes artifacts/data_dictionary.csv
python -m src.pairs --freeze-check         # freeze/verify the reserved test split
```

Mojibake (`�`) and the orphaned foreign keys in `job_skills.csv`/`job_industries.csv`
are documented, not fixed — see `write ups/DTSC3602 Project Group 9- D2.md` for why.
Twelve candidate audit plots live in `notebooks/03_data_audit.ipynb`.

D2 reserves D1's existing doubly-disjoint test split (5,784 pairs) as the project's
final test set rather than drawing a new one. `write ups/d2_test_split_freeze.json`
holds a SHA256 fingerprint of it; `--freeze-check` asserts the fingerprint still
matches on every run.

## Dashboard

A second Streamlit app, `app/dashboard.py`, presents the data dictionary, live plots,
quality findings, and the reserved-split summary — reading from Supabase instead of
local parquet. It's separate from `app/streamlit_app.py` (D1's swipe demo), which is
untouched and still reads local files.

**Local setup:**

```bash
cp .env.example .env               # fill in SUPABASE_URL / SUPABASE_KEY
psql "$SUPABASE_DB_URL" -f supabase/schema.sql   # or paste into the SQL editor
python scripts/upload_to_supabase.py             # --dry-run first to sanity-check
streamlit run app/dashboard.py
```

**Deploying on Modal:**

```bash
modal secret create supabase-creds SUPABASE_URL=... SUPABASE_KEY=...
modal deploy modal_app.py
```

`modal_app.py` builds its image straight from `requirements.txt` and reads the same
two environment variable names locally and in production, via `src/supabase_client.py`.

## Privacy

`src/text_clean.py` strips emails, phone numbers, URLs, street addresses, ZIPs,
and the name/headline block that leads every resume, before anything is
embedded. It cannot remove writing style, school names, or regional phrasing,
which still carry demographic signal — hence the fairness table.

## The swipe log

`app/streamlit_app.py` writes every swipe to `artifacts/swipe_log.csv` with its
dwell time. This is the measurement instrument for D1's research question:

- engagement = cards swiped per session
- choice quality = like rate by predicted fit tier
- data quality = swipes under 1.5s flagged as `blind_swipe`

## Layout

```
src/config.py       paths, constants, seed
src/text_clean.py   PII scrubbing
src/data_load.py    CSV -> joined, cleaned parquet
src/taxonomy.py     24 categories -> skills / industries / titles + adjacency
src/pairs.py        weak labels + leakage-free split
src/embed.py        MiniLM and TF-IDF/SVD vectors, cached
src/features.py     per-pair feature matrix
src/model.py        train, evaluate, fairness, persist
src/recommend.py    serving: resume -> ranked deck
src/audit.py        D2: salary-outlier, sentinel, duplicate flags
src/data_dictionary.py  D2: generates artifacts/data_dictionary.csv
src/supabase_client.py  shared Supabase client (dashboard + upload script)
app/streamlit_app.py  D1 swipe demo + logging
app/dashboard.py     D2 data-audit dashboard, reads from Supabase
scripts/upload_to_supabase.py  pushes audit artifacts to Supabase
modal_app.py         Modal deployment entrypoint for app/dashboard.py
notebooks/          EDA (01) and D2 data-audit (03) plots for the write-up
```
