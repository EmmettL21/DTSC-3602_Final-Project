# EDA Dashboard

**Live dashboard:** https://awalla63--job-tinder-dashboard-dashboard.us-east.modal.run/

This folder has the code for the project's exploratory data analysis (EDA) dashboard. The dashboard is a Streamlit app that reads cleaned data from Supabase and is hosted on Modal.

| File | Purpose |
| --- | --- |
| `prepare.py` | Audits and cleans the loaded data and writes the results to a separate `analytics` schema in Supabase. |
| `dashboard.py` | The Streamlit dashboard. It only runs read-only aggregate queries against the `analytics` schema. |
| `modal_dashboard.py` | Deploys the dashboard to Modal as a web server. |

---

## 1. Original data sources

| Dataset | Source | Raw rows | Rows used |
| --- | --- | ---: | ---: |
| Job postings | [LinkedIn Job Postings, `postings.csv` (Kaggle, arshkon)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings?select=postings.csv) | 123,849 | 44,138 |
| Resumes | [Resume Dataset, `Resume.csv` (Kaggle, snehaanbhawal)](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) | 2,484 | 2,483 |

---

## 2. How the data was cleaned

Cleaning happened in two stages. **Stage 1** happened when the raw CSVs were loaded into Supabase, which decided which rows exist at all. **Stage 2** (`prepare.py`) audits those loaded rows and records every problem it finds. In Stage 2, nothing is deleted.

### Stage 1: cleaning at load time

The raw postings file is 516 MB. The Supabase free tier caps the whole database at 500 MB, so the postings had to be cut down. Most of what was removed was low-quality or duplicate data rather than a random sample.

**Job postings: 123,849 down to about 44,000 rows**

| Step | Rows remaining |
| --- | ---: |
| All raw rows | 123,849 |
| Drop postings with no title or description, a description under 700 characters, a Volunteer/Other work type, or a closing date (already filled) | 118,268 |
| Remove duplicate company + title pairs, keeping the most-viewed copy | 92,656 |
| Require a known experience level | 70,326 |
| Keep at most 10 postings per company (the most-viewed ones) | 44,145 |

- **Duplicates were the biggest problem.** The dedupe step alone removed 26,291 rows (21%), because staffing agencies repost the same listing dozens of times.
- **Descriptions were cut off at 4,000 characters** to save space.
- **A US state code was parsed** out of the free-text location field, and the remote flag was converted to true/false.
- **The posting date (`listed_time`) was not loaded**, so the data supports no time-based analysis.

**Resumes: 2,484 down to 2,483 rows**

- Resumes under 200 characters were skipped (one row).
- Resumes were cut off at 8,000 characters.
- **Years of experience was estimated** as the span between the earliest and latest year mentioned in the resume. This is a rough heuristic, not a reported value.
- Each resume keeps the dataset's job category label (24 categories).

### Stage 2: audit and cleaning for analysis (`prepare.py`)

`prepare.py` never modifies the app's own tables. It writes everything to a separate `analytics` schema, keyed by the original row id, so every raw value can still be recovered. The core rule is that **problems are flagged, not deleted**.

- **Salary:** salaries outside $15,000 to $1,000,000 a year are set to null in the clean column. These are almost always mislabelled pay periods, such as an hourly or monthly figure stored as yearly, or a literal 0. The raw value is kept in `salary_raw`.
- **State:** 1,640 postings listed only a metro area, such as "Greater Boston" or "Dallas-Fort Worth Metroplex". These were mapped to their principal city's state. Postings for the whole country ("United States") are flagged `is_nationwide` and left out of per-state counts.
- **Company name:** blank names are set to null.
- **Truncated text:** descriptions that hit the 4,000-character cap and resumes that hit the 8,000-character cap are flagged.
- **Duplicates:** postings and resumes with byte-identical text are kept, and each row records how many copies share its text.
- **Privacy:** only the public Kaggle data is analysed. Real users' resumes and employer-posted jobs are excluded entirely.

### Audit results

These are all the checks `prepare.py` runs, with the number of rows each one affects:

| Check | Dataset | Rows affected | Share | Severity | Action |
| --- | --- | ---: | ---: | --- | --- |
| Posting date missing | Jobs | 44,138 | 100.0% | High | Column excluded; no time analysis possible |
| No salary posted | Jobs | 29,636 | 67.1% | High | Kept; salary analysis covers posted salaries only |
| Description cut at 4,000 chars | Jobs | 19,799 | 44.9% | Medium | Kept and flagged; length stats are capped |
| Nationwide location (no state) | Jobs | 3,544 | 8.0% | Low | Kept and flagged; left out of per-state counts |
| State recovered from metro area | Jobs | 1,640 | 3.7% | Low | Recovered using a metro-to-state lookup |
| Duplicate description | Jobs | 1,504 | 3.4% | Medium | Kept; duplicates placed in the same split |
| Salary out of range | Jobs | 193 | 0.4% | Medium | Clean salary set to null; raw value kept |
| State still missing | Jobs | 190 | 0.4% | Low | Kept; state is null |
| Blank company name | Jobs | 10 | <0.1% | Low | Set to null |
| Years of experience is estimated | Resumes | 2,468 | 99.4% | Medium | Kept; treat as approximate |
| Resume cut at 8,000 chars | Resumes | 415 | 16.7% | Low | Kept and flagged |
| No years-of-experience estimate | Resumes | 15 | 0.6% | Low | Kept; null |
| Duplicate resume | Resumes | 4 | 0.2% | Low | Kept; duplicates placed in the same split |

### Reserved test split

20% of jobs and resumes are held out as a test set and **never used in the dashboard's charts**. The split is:

- **Stratified:** by experience level for jobs and by category for resumes, so each group keeps the same 80/20 ratio.
- **Group-safe:** rows with identical text always land in the same split, so a duplicate can't appear in both train and test.
- **Deterministic:** assignment is a salted MD5 hash of each row, so rebuilding the schema never reshuffles it.

| Dataset | Train | Test | Test share |
| --- | ---: | ---: | ---: |
| Jobs | 35,278 | 8,860 | 20.1% |
| Resumes | 1,976 | 507 | 20.4% |

---

## 3. What we're working with

Here is the data the analysis rests on, with its limitations stated plainly:

- **Salary only covers a third of postings.** Only 32.6% of training jobs list a usable salary, so salary findings describe postings that disclose pay, not the job market as a whole.
- **There is no time dimension.** Posting dates were never loaded.
- **Description lengths are capped.** About 45% of descriptions hit the 4,000-character cap. The cut often lands in the qualifications section, which is the part that matters most for matching.
- **Resume experience is approximate.** Years of experience is a heuristic estimate for almost every resume.
- **Jobs and resumes differ in seniority.** Jobs skew junior (Entry level and Mid-Senior make up most postings), while the demo resumes skew senior.
- **Some resume categories are small.** BPO (17), Automobile (28) and Agriculture (50) have too few training resumes for reliable per-category metrics.

---

## 4. Dashboard contents

The dashboard shows four headline numbers: train/test counts for jobs and resumes, the share of jobs with a usable salary, and the share of remote-allowed jobs. Below those are five tabs:

1. **Data sources & cleaning:** a summary of the sources and both cleaning stages above.
2. **Plots**
   - The share of jobs affected by each data-quality check
   - Median posted salary by experience level (about $48k for internships up to $200k for executives)
   - Seniority mix of jobs compared with resumes
   - Number of resumes in each category
   - Description lengths, showing the spike at the 4,000-character cap
3. **Audit checks:** the full audit table.
4. **Data dictionary:** every column in the clean views, with its type, meaning, cleaning applied and measured null rate.
5. **Reserved test split:** train/test counts for each group.

---

## 5. Running it

The code needs a `DATABASE_URL` for the Supabase Postgres database in the repo's `.env`. Run these commands from the repo root:

```bash
pip install streamlit altair pandas "psycopg[binary]" python-dotenv

python "EDA dashboard Code/prepare.py" --dry-run   # print the audit, write nothing
python "EDA dashboard Code/prepare.py"             # build the analytics schema
streamlit run "EDA dashboard Code/dashboard.py"    # run the dashboard locally
modal deploy "EDA dashboard Code/modal_dashboard.py"  # deploy to Modal
```

For the Modal deployment, `DATABASE_URL` is supplied through the Modal secret `Supabase_Secret_Job_Tinder`.
