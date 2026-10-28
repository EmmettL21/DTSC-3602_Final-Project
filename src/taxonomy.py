"""Bridge between the 24 resume categories and the LinkedIn posting taxonomy.

Neither dataset carries a resume<->posting fit label, so we manufacture one.
Each resume category is described three ways -- by LinkedIn skill abbreviation,
by industry-name pattern, and by job-title pattern -- and a posting is assigned
the category that scores highest across those signals. A resume and a posting in
the same category become a `Good Fit` pair, adjacent categories become
`Partial Fit`, and everything else is `No Fit`.

This is weak supervision: the labels encode *our* notion of fit, not a hiring
outcome. Every metric downstream is measured against that proxy, which is the
single biggest caveat in the write-up.

Industries are matched by name regex rather than by the ~400 raw `industry_id`
values -- the IDs are opaque and hand-listing them invites silent typos.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Skill abbreviations from postings/mappings/skills.csv:
#   ACCT ADM ADVR ANLS ART BD CNSL CUST DSGN DIST EDU ENG FIN GENB HCPR HR IT
#   LGL MGMT MNFC MRKT OTHR PR PRCH PRDM PROD PRJM QA RSCH SALE SCI STRA SUPL
#   TRNG WRT


@dataclass(frozen=True)
class CategorySpec:
    """How one resume category shows up in the postings data."""

    skills: frozenset[str] = field(default_factory=frozenset)
    industry_pattern: str = ""
    title_pattern: str = ""
    related: frozenset[str] = field(default_factory=frozenset)


def _spec(skills, industry, title, related) -> CategorySpec:
    return CategorySpec(
        skills=frozenset(skills),
        industry_pattern=industry,
        title_pattern=title,
        related=frozenset(related),
    )


CATEGORIES: dict[str, CategorySpec] = {
    "ACCOUNTANT": _spec(
        ["ACCT", "FIN"],
        r"Accounting|Credit Intermediation|Claims Adjusting, Actuarial",
        r"\b(accountant|accounting|bookkeep|auditor|audit|payroll|accounts (payable|receivable)|controller|tax|cpa)\b",
        ["FINANCE", "BANKING"],
    ),
    "ADVOCATE": _spec(
        ["LGL"],
        r"Law Practice|Legal Services|Courts of Law|Alternative Dispute|Administration of Justice",
        r"\b(attorney|lawyer|legal|paralegal|counsel|litigation|advocate|compliance officer)\b",
        ["PUBLIC-RELATIONS", "HR"],
    ),
    "AGRICULTURE": _spec(
        ["PROD", "SCI", "OTHR"],
        r"Farming|Horticulture|Fisheries|Forestry|Animal Feed|Agricultural Chemical|Dairy|Wholesale Raw Farm|Veterinary",
        r"\b(agricultur|farm|agronom|crop|livestock|horticultur|greenhouse|irrigation|ranch|forestry)\b",
        ["CONSTRUCTION", "ENGINEERING"],
    ),
    "APPAREL": _spec(
        ["PROD", "DSGN", "MNFC"],
        r"Apparel|Textile Manufacturing|Footwear|Fashion Accessories|Luxury Goods|Wholesale Apparel",
        r"\b(apparel|fashion|garment|textile|merchandis|tailor|sewing|footwear|clothing)\b",
        ["DESIGNER", "SALES"],
    ),
    "ARTS": _spec(
        ["ART", "WRT", "DSGN"],
        r"Fine Art|Performing Arts|Musicians|Artists and Writers|Museums|Theater Companies|Photography|Entertainment",
        r"\b(artist|curator|musician|performer|dancer|theat(er|re)|sculpt|painter|photograph|gallery)\b",
        ["DESIGNER", "DIGITAL-MEDIA"],
    ),
    "AUTOMOBILE": _spec(
        ["MNFC", "ENG", "PROD"],
        r"Motor Vehicle|Automotive|Vehicle Repair|Retail Motor Vehicles|Wholesale Motor Vehicles|Alternative Fuel Vehicle",
        r"\b(automotive|automobile|vehicle|mechanic|technician.*(auto|vehicle)|collision|dealership|diesel)\b",
        ["ENGINEERING", "CONSTRUCTION"],
    ),
    "AVIATION": _spec(
        ["ENG", "OTHR", "PROD"],
        r"Airlines and Aviation|Aviation & Aerospace|Aviation and Aerospace|Defense and Space|Defense & Space|Space Research",
        r"\b(aviation|aircraft|aerospace|pilot|flight|airline|avionics|airport)\b",
        ["ENGINEERING", "AUTOMOBILE"],
    ),
    "BANKING": _spec(
        ["FIN", "ACCT"],
        r"Banking|Investment Banking|Capital Markets|Credit Intermediation|Loan Brokers|Securities and Commodity",
        r"\b(bank|teller|loan|mortgage|credit analyst|branch manager|underwrit|treasury)\b",
        ["FINANCE", "ACCOUNTANT"],
    ),
    "BPO": _spec(
        ["CUST", "ADM", "OTHR"],
        r"Outsourcing and Offshoring|Telephone Call Centers|Administrative and Support Services|Staffing and Recruiting",
        r"\b(call cent(er|re)|customer (service|support|care)|bpo|help ?desk|contact cent(er|re)|service representative)\b",
        ["HR", "SALES"],
    ),
    "BUSINESS-DEVELOPMENT": _spec(
        ["BD", "STRA", "SALE", "MGMT"],
        r"Business Consulting|Strategic Management|Venture Capital|Professional Services|Holding Companies",
        r"\b(business development|partnerships?|growth|strategy|account executive|territory manager|bd manager)\b",
        ["SALES", "CONSULTANT", "PUBLIC-RELATIONS"],
    ),
    "CHEF": _spec(
        ["PROD", "OTHR"],
        r"Restaurants|Food and Beverage Services|Caterers|Hospitality|Hotels and Motels|Accommodation and Food|Food Production|Baked Goods",
        r"\b(chef|cook|culinary|kitchen|sous|baker|pastry|line cook|food service|restaurant)\b",
        ["FITNESS", "APPAREL"],
    ),
    "CONSTRUCTION": _spec(
        ["ENG", "PRJM", "MNFC", "PROD"],
        r"Construction|Building Construction|Specialty Trade Contractors|Building Equipment|Building Materials|Utility System Construction|Civil Engineering",
        r"\b(construction|contractor|carpenter|electrician|plumb|welder|foreman|superintendent|mason|hvac|site manager)\b",
        ["ENGINEERING", "AUTOMOBILE", "AGRICULTURE"],
    ),
    "CONSULTANT": _spec(
        ["CNSL", "STRA", "ANLS", "MGMT"],
        r"Business Consulting|Operations Consulting|Strategic Management|Think Tanks|Professional Services|Research Services",
        r"\b(consultant|consulting|advisory|business analyst|strategy analyst|engagement manager)\b",
        ["BUSINESS-DEVELOPMENT", "INFORMATION-TECHNOLOGY", "FINANCE"],
    ),
    "DESIGNER": _spec(
        ["DSGN", "ART", "PRDM"],
        r"Design Services|Graphic Design|Interior Design|Architecture and Planning|Accessible Architecture",
        r"\b(designer|design|ux|ui|graphic|creative director|art director|interior design|architect)\b",
        ["ARTS", "DIGITAL-MEDIA", "APPAREL"],
    ),
    "DIGITAL-MEDIA": _spec(
        ["MRKT", "WRT", "ART", "PR"],
        r"Online Media|Broadcast Media|Media Production|Animation and Post-production|Movies, Videos|Internet Publishing|Social Networking Platforms|Technology, Information and Media",
        r"\b(digital media|social media|content creator|video (editor|produc)|multimedia|podcast|streaming|animator|motion graphic)\b",
        ["ARTS", "DESIGNER", "PUBLIC-RELATIONS"],
    ),
    "ENGINEERING": _spec(
        ["ENG", "QA", "PRJM", "SCI"],
        r"Engineering Services|Mechanical Or Industrial Engineering|Industrial Machinery|Machinery Manufacturing|Semiconductor|Automation Machinery|Robotics Engineering|Industrial Automation|Electrical Equipment",
        r"\b(engineer|engineering)\b(?!.*\b(sales|software|devops)\b)|\b(mechanical|electrical|civil|industrial|process|manufacturing) engineer\b",
        ["CONSTRUCTION", "INFORMATION-TECHNOLOGY", "AUTOMOBILE", "AVIATION"],
    ),
    "FINANCE": _spec(
        ["FIN", "ANLS", "ACCT"],
        r"Financial Services|Investment Management|Investment Advice|Insurance|Funds and Trusts|Pension Funds|Venture Capital",
        r"\b(financial analyst|finance|fp&a|investment|portfolio|actuar|risk analyst|financial advisor|cfo)\b",
        ["ACCOUNTANT", "BANKING", "CONSULTANT"],
    ),
    "FITNESS": _spec(
        ["HCPR", "EDU", "OTHR"],
        r"Wellness and Fitness|Health, Wellness & Fitness|Sports and Recreation Instruction|Recreational Facilities|Spectator Sports|Sports Teams|Golf Courses",
        r"\b(fitness|trainer|coach|gym|yoga|pilates|athletic|wellness|physical training|nutritionist)\b",
        ["HEALTHCARE", "TEACHER"],
    ),
    "HEALTHCARE": _spec(
        ["HCPR", "RSCH", "SCI"],
        r"Hospitals|Medical Practices|Health and Human Services|Nursing Homes|Home Health Care|Mental Health Care|Pharmaceutical|Biotechnology|Medical Device|Medical Equipment|Medical and Diagnostic|Dentists|Public Health|Alternative Medicine|Physical, Occupational",
        r"\b(nurse|rn\b|lpn\b|physician|doctor|medical|clinical|patient|healthcare|health care|therapist|pharmac|dental|caregiver|phlebotom)\b",
        ["FITNESS", "ENGINEERING"],
    ),
    "HR": _spec(
        ["HR", "TRNG", "ADM"],
        r"Human Resources|Staffing and Recruiting|Executive Search|Professional Training and Coaching|Insurance and Employee Benefit",
        r"\b(human resources|hr\b|recruit|talent acquisition|people operations|benefits|compensation|onboarding|staffing)\b",
        ["BPO", "TEACHER", "ADVOCATE"],
    ),
    "INFORMATION-TECHNOLOGY": _spec(
        ["IT", "ENG", "QA", "ANLS"],
        r"IT Services|IT System|Software Development|Computer Hardware|Computer Networking|Computer and Network Security|Data Infrastructure|Data Security|Technology, Information and Internet|Information Technology & Services|Desktop Computing|Business Intelligence|Blockchain|Computer Games|Telecommunications|Wireless Services|Information Services",
        r"\b(software|developer|programmer|it\b|information technology|data (engineer|analyst|scientist)|devops|sysadmin|system administrator|network|cloud|cyber ?security|qa engineer|full ?stack|web developer|database)\b",
        ["ENGINEERING", "CONSULTANT", "DIGITAL-MEDIA"],
    ),
    "PUBLIC-RELATIONS": _spec(
        ["PR", "MRKT", "WRT", "ADVR"],
        r"Public Relations and Communications|Advertising Services|Marketing Services|Government Relations|Non-profit|Political Organizations|Philanthropic Fundraising|Fundraising|Civic and Social",
        r"\b(public relations|pr\b|communications|marketing|media relations|brand|press|publicist|outreach|copywriter)\b",
        ["DIGITAL-MEDIA", "BUSINESS-DEVELOPMENT", "ARTS"],
    ),
    "SALES": _spec(
        ["SALE", "BD", "CUST", "MRKT"],
        r"Retail|Wholesale|Real Estate|Consumer Goods|Internet Marketplace|Online and Mail Order Retail|Food and Beverage Retail",
        r"\b(sales|account manager|retail|cashier|store manager|territory|inside sales|outside sales|sales representative|realtor|real estate agent)\b",
        ["BUSINESS-DEVELOPMENT", "BPO", "APPAREL"],
    ),
    "TEACHER": _spec(
        ["EDU", "TRNG", "RSCH"],
        r"Education|Higher Education|Primary and Secondary Education|E-Learning|E-learning|Education Management|Education Administration|Technical and Vocational Training|Libraries|Museums",
        r"\b(teacher|teaching|instructor|professor|educator|tutor|lecturer|curriculum|principal|faculty|academic)\b",
        ["HR", "FITNESS", "CONSULTANT"],
    ),
}

CATEGORY_NAMES: list[str] = sorted(CATEGORIES)

# Relative weight of each signal when assigning a category to a posting.
# Title is weighted highest: it is populated for every posting and is the most
# specific of the three, while skills and industries are coarse and sometimes
# missing entirely.
W_TITLE, W_SKILL, W_INDUSTRY = 3.0, 2.0, 1.0

# A posting must clear this to be assigned a category at all. At these weights
# it means "title matched" or "skill + industry both matched" -- one lone
# industry hit is not enough, since industries like `Retail` are near-universal.
MIN_CATEGORY_SCORE = 2.0

_TITLE_RES = {
    name: re.compile(spec.title_pattern, re.IGNORECASE)
    for name, spec in CATEGORIES.items()
    if spec.title_pattern
}
_INDUSTRY_RES = {
    name: re.compile(spec.industry_pattern, re.IGNORECASE)
    for name, spec in CATEGORIES.items()
    if spec.industry_pattern
}


def title_regex(category: str) -> re.Pattern[str] | None:
    """Compiled title pattern for a category, for callers scoring one posting."""
    return _TITLE_RES.get(category)


def _build_related() -> dict[str, frozenset[str]]:
    """Symmetric closure of the hand-declared `related` edges.

    Adjacency is undirected -- if an accountant partially fits finance roles,
    a finance resume partially fits accounting ones. Declaring each edge once
    and closing it here beats maintaining both directions by hand, which drifts
    the moment anyone edits the table.
    """
    edges: dict[str, set[str]] = {name: set() for name in CATEGORIES}
    for name, spec in CATEGORIES.items():
        for other in spec.related:
            if other not in CATEGORIES:
                raise ValueError(f"{name} relates to unknown category {other!r}")
            if other == name:
                raise ValueError(f"{name} is listed as related to itself")
            edges[name].add(other)
            edges[other].add(name)
    return {name: frozenset(others) for name, others in edges.items()}


RELATED: dict[str, frozenset[str]] = _build_related()


def validate() -> None:
    """Sanity-check the table at import time."""
    for name, spec in CATEGORIES.items():
        if not spec.title_pattern:
            raise ValueError(f"{name} has no title pattern")
        re.compile(spec.title_pattern)
        if spec.industry_pattern:
            re.compile(spec.industry_pattern)
        if not RELATED[name]:
            raise ValueError(f"{name} has no related categories")


def score_posting(title: str, skills: set[str], industries: list[str]) -> dict[str, float]:
    """Weighted per-category score for one posting."""
    scores: dict[str, float] = {}
    industry_blob = " | ".join(industries)

    for name, spec in CATEGORIES.items():
        score = 0.0
        if name in _TITLE_RES and _TITLE_RES[name].search(title or ""):
            score += W_TITLE
        if skills and spec.skills:
            score += W_SKILL * len(skills & spec.skills) / len(spec.skills)
        if industry_blob and name in _INDUSTRY_RES:
            if _INDUSTRY_RES[name].search(industry_blob):
                score += W_INDUSTRY
        if score:
            scores[name] = score
    return scores


def assign_category(
    title: str, skills: set[str], industries: list[str]
) -> tuple[str | None, float]:
    """Best category for a posting, or `(None, score)` if nothing is confident.

    Ties are broken alphabetically so the assignment is deterministic across
    runs -- important because the train/test split is derived from it.
    """
    scores = score_posting(title, skills, industries)
    if not scores:
        return None, 0.0
    best = min(
        (n for n in scores if scores[n] == max(scores.values()))
    )  # alphabetical tiebreak
    top = scores[best]
    return (best, top) if top >= MIN_CATEGORY_SCORE else (None, top)


def pair_label(resume_category: str, posting_category: str | None) -> str:
    """Weak fit label for a (resume, posting) pair."""
    if posting_category is None:
        return "No Fit"
    if resume_category == posting_category:
        return "Good Fit"
    if posting_category in RELATED[resume_category]:
        return "Partial Fit"
    return "No Fit"


validate()
