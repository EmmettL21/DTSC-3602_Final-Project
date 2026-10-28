"""Text normalization and PII scrubbing.

D1 commits to removing identifiable information before the model sees it, so
every resume and posting goes through `clean_text` on the way into the frames.
This removes the *explicit* identifiers (email, phone, URL, street address, the
name line at the top of a resume). It cannot remove writing style, school names
or regional phrasing, which still carry demographic signal -- see the fairness
table in `model.py` for how we check what leaked through anyway.
"""

from __future__ import annotations

import html
import re

import pandas as pd

# --- PII patterns -----------------------------------------------------------
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)

# Matches (123) 456-7890, 123-456-7890, 123.456.7890, +1 123 456 7890.
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"
)

STREET_RE = re.compile(
    r"\b\d{1,6}\s+(?:[NSEW]\.?\s+)?[A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){0,3}\s+"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|"
    r"Circle|Cir|Way|Place|Pl|Parkway|Pkwy|Terrace|Ter|Highway|Hwy|Suite|Ste|Apt)\b\.?",
    re.IGNORECASE,
)

ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")

# Resume text in this dataset starts with the person's name/headline in caps
# before the first section header. Drop that leading block.
RESUME_HEAD_RE = re.compile(
    r"^\s*(?:[A-Z][A-Z&/,.'\- ]{2,80}\s+){1,4}(?=[A-Z][a-z])"
)

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
# Long runs of punctuation used as visual rules in scraped postings.
RULE_RE = re.compile(r"([^\w\s])\1{3,}")


def clean_text(text: object, *, strip_resume_head: bool = False) -> str:
    """Normalize one document and strip explicit identifiers.

    `strip_resume_head=True` additionally drops the all-caps name/headline block
    that leads every record in the resume dataset.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    out = html.unescape(str(text))
    out = HTML_TAG_RE.sub(" ", out)

    # Order matters: pull emails and URLs before the phone pattern, otherwise
    # digit runs inside a URL get mangled into a phone placeholder.
    out = EMAIL_RE.sub(" ", out)
    out = URL_RE.sub(" ", out)
    out = PHONE_RE.sub(" ", out)
    out = STREET_RE.sub(" ", out)
    out = ZIP_RE.sub(" ", out)

    if strip_resume_head:
        out = RESUME_HEAD_RE.sub("", out, count=1)

    out = RULE_RE.sub(" ", out)
    out = WHITESPACE_RE.sub(" ", out)
    return out.strip()


def clean_series(s: pd.Series, *, strip_resume_head: bool = False) -> pd.Series:
    return s.map(lambda t: clean_text(t, strip_resume_head=strip_resume_head))


def truncate(text: str, max_chars: int) -> str:
    """Cut to `max_chars` on a word boundary when one is nearby."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    return cut[:space] if space > max_chars * 0.8 else cut


def has_pii(text: str) -> bool:
    """True if any explicit identifier survived cleaning. Used by tests."""
    return any(
        p.search(text) for p in (EMAIL_RE, URL_RE, PHONE_RE, STREET_RE, ZIP_RE)
    )
