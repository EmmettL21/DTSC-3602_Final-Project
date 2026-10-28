"""Per-(resume, posting) feature matrix.

Three families:

    similarity   cosine in each embedding space
    content      skill overlap, title keyword hits
    preference   the survey answers D1 describes -- salary, state, remote,
                 experience level, work type

Missing values are carried as explicit `*_missing` indicator columns rather than
imputed. 71% of postings have no salary and 24% no experience level; imputing a
median there would invent preference agreement that the data never recorded,
and the indicator lets the model learn what an absent field is worth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import taxonomy as T
from .data_load import EXPERIENCE_ORDER

FEATURE_NAMES: list[str] = [
    "cos_minilm",
    "cos_tfidf",
    "skill_jaccard",
    "skill_overlap_n",
    "title_hits",
    "salary_ratio",
    "salary_meets_target",
    "salary_missing",
    "state_match",
    "state_unknown",
    "remote_ok",
    "experience_gap",
    "experience_missing",
    "work_type_match",
    "company_size",
    "company_size_missing",
]


@dataclass
class Preferences:
    """A job seeker's survey answers. Every field is optional."""

    target_salary: float | None = None
    state: str | None = None
    wants_remote: bool = False
    experience_level: str | None = None
    work_types: tuple[str, ...] = field(default_factory=tuple)

    @property
    def experience_rank(self) -> int | None:
        if self.experience_level is None:
            return None
        return EXPERIENCE_ORDER.get(self.experience_level)


# Resume text is matched against these to estimate which LinkedIn skill
# abbreviations a candidate demonstrates, so we can compare against the
# posting's own skill tags.
SKILL_CUES: dict[str, tuple[str, ...]] = {
    "ACCT": ("accounting", "bookkeeping", "audit", "payroll", "ledger", "tax"),
    "ADM": ("administrative", "scheduling", "clerical", "office manage", "data entry"),
    "ADVR": ("advertising", "campaign", "media buying", "ad "),
    "ANLS": ("analysis", "analytics", "reporting", "dashboard", "forecasting"),
    "ART": ("art", "creative", "illustration", "photography", "music"),
    "BD": ("business development", "partnership", "prospecting", "lead generation"),
    "CNSL": ("consulting", "advisory", "client engagement"),
    "CUST": ("customer service", "customer support", "client relations", "call center"),
    "DSGN": ("design", "figma", "photoshop", "illustrator", "ux", "ui", "cad"),
    "DIST": ("distribution", "logistics", "warehouse", "shipping"),
    "EDU": ("teaching", "curriculum", "classroom", "instruction", "tutoring"),
    "ENG": ("engineering", "mechanical", "electrical", "civil", "cad", "autocad"),
    "FIN": ("finance", "financial", "budget", "investment", "portfolio", "p&l"),
    "GENB": ("operations", "business", "management"),
    "HCPR": ("patient", "clinical", "nursing", "medical", "therapy", "diagnosis"),
    "HR": ("human resources", "recruiting", "onboarding", "benefits", "staffing"),
    "IT": ("software", "python", "java", "sql", "network", "cloud", "linux", "database"),
    "LGL": ("legal", "litigation", "contract", "compliance", "paralegal"),
    "MGMT": ("managed", "supervised", "led team", "director", "leadership"),
    "MNFC": ("manufacturing", "production line", "assembly", "fabrication", "machining"),
    "MRKT": ("marketing", "seo", "brand", "social media", "content strategy"),
    "PR": ("public relations", "press release", "communications", "media relations"),
    "PRCH": ("procurement", "purchasing", "vendor", "sourcing"),
    "PRDM": ("product management", "roadmap", "product owner"),
    "PRJM": ("project management", "pmp", "agile", "scrum", "milestone"),
    "PROD": ("production", "quality control", "operations", "inventory"),
    "QA": ("quality assurance", "testing", "qa", "inspection"),
    "RSCH": ("research", "study", "experiment", "publication"),
    "SALE": ("sales", "quota", "revenue", "closing", "account management", "crm"),
    "SCI": ("laboratory", "scientific", "chemistry", "biology", "experiment"),
    "STRA": ("strategy", "strategic planning", "roadmap", "market analysis"),
    "SUPL": ("supply chain", "procurement", "inventory", "logistics"),
    "TRNG": ("training", "mentoring", "coaching", "onboarding", "workshop"),
    "WRT": ("writing", "editing", "copywriting", "content", "editorial"),
}


def resume_skills(text: str) -> set[str]:
    """Skill abbreviations a resume demonstrates, by keyword cue."""
    low = text.lower()
    return {abr for abr, cues in SKILL_CUES.items() if any(c in low for c in cues)}


def title_keyword_hits(title: str, resume_category: str) -> int:
    """How many of the category's title cues appear in this posting's title."""
    pattern = T.title_regex(resume_category)
    if pattern is None:
        return 0
    return len(pattern.findall(title or ""))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_feature_matrix(
    pairs: pd.DataFrame,
    postings: pd.DataFrame,
    resumes: pd.DataFrame,
    *,
    cos_minilm: np.ndarray,
    cos_tfidf: np.ndarray,
    prefs: dict[int, Preferences] | None = None,
) -> pd.DataFrame:
    """Assemble features for every row of `pairs`.

    `cos_minilm` / `cos_tfidf` are precomputed per-pair similarities, aligned to
    `pairs` row order -- the caller gathers them from the cached matrices, which
    is far cheaper than a lookup per row here.
    """
    post = postings.set_index("job_id")
    res = resumes.set_index("resume_id")

    job = post.reindex(pairs["job_id"].to_numpy())
    cand = res.reindex(pairs["resume_id"].to_numpy())

    out = pd.DataFrame(index=pairs.index)
    out["cos_minilm"] = cos_minilm
    out["cos_tfidf"] = cos_tfidf

    # --- content overlap ---
    cand_skills = [resume_skills(t) for t in cand["resume_text"].to_numpy()]
    job_skills = [set(s) for s in job["skill_abrs"].to_numpy()]
    out["skill_jaccard"] = [_jaccard(a, b) for a, b in zip(cand_skills, job_skills, strict=True)]
    out["skill_overlap_n"] = [len(a & b) for a, b in zip(cand_skills, job_skills, strict=True)]
    out["title_hits"] = [
        title_keyword_hits(t, c)
        for t, c in zip(job["title"].to_numpy(), cand["category"].to_numpy(), strict=True)
    ]

    # --- preference features ---
    prefs = prefs or {}
    blank = Preferences()
    p = [prefs.get(int(rid), blank) for rid in pairs["resume_id"].to_numpy()]

    salary = job["salary_annual"].to_numpy(dtype="float64")
    target = np.array([x.target_salary if x.target_salary else np.nan for x in p])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = salary / target
    out["salary_ratio"] = np.clip(np.nan_to_num(ratio, nan=0.0), 0, 5)
    out["salary_meets_target"] = np.where(np.isnan(ratio), 0, (ratio >= 1).astype(int))
    out["salary_missing"] = np.isnan(ratio).astype(int)

    job_state = job["state"].to_numpy()
    want_state = np.array([x.state for x in p], dtype=object)
    remote = job["remote_allowed"].to_numpy().astype(bool)

    has_pref = pd.notna(want_state)
    state_known = pd.notna(job_state)
    # A remote posting satisfies any state preference on its own -- 15% of
    # postings never list a location, and treating those as a location failure
    # would bury exactly the remote roles a location-flexible user wants.
    satisfied = (state_known & (job_state == want_state)) | remote

    out["state_match"] = np.where(has_pref, satisfied, 0).astype(int)
    # Unknown means "could not be evaluated": no preference given, or the
    # posting has neither a parsed state nor a remote flag.
    out["state_unknown"] = (~has_pref | (~state_known & ~remote)).astype(int)
    out["remote_ok"] = np.where([x.wants_remote for x in p], remote.astype(int), 1)

    job_rank = job["experience_rank"].to_numpy(dtype="float64")
    want_rank = np.array([x.experience_rank if x.experience_rank is not None else np.nan for x in p])
    gap = np.abs(job_rank - want_rank)
    out["experience_gap"] = np.nan_to_num(gap, nan=0.0)
    out["experience_missing"] = np.isnan(gap).astype(int)

    job_wt = job["formatted_work_type"].to_numpy()
    out["work_type_match"] = [
        1 if (not x.work_types or wt in x.work_types) else 0
        for wt, x in zip(job_wt, p, strict=True)
    ]

    size = job["company_size"].to_numpy(dtype="float64")
    out["company_size"] = np.nan_to_num(size, nan=0.0)
    out["company_size_missing"] = np.isnan(size).astype(int)

    return out[FEATURE_NAMES].astype(np.float32)


def pair_cosines(
    pairs: pd.DataFrame,
    *,
    post_matrix: np.ndarray,
    post_ids: np.ndarray,
    resume_matrix: np.ndarray,
    resume_ids: np.ndarray,
) -> np.ndarray:
    """Cosine similarity for each pair, as a row-aligned vector.

    Both matrices are L2-normalized, so the row-wise dot product is the cosine.
    """
    post_row = {int(v): i for i, v in enumerate(post_ids)}
    res_row = {int(v): i for i, v in enumerate(resume_ids)}

    j = np.array([post_row[int(v)] for v in pairs["job_id"]])
    r = np.array([res_row[int(v)] for v in pairs["resume_id"]])
    return np.einsum("ij,ij->i", resume_matrix[r], post_matrix[j]).astype(np.float32)
