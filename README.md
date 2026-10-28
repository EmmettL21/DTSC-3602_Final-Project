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
app/streamlit_app.py  swipe demo + logging
notebooks/          EDA and results for the write-up
```
