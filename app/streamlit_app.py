"""Swipe-card demo for the Group 9 job matcher.

Beyond showing the model working, this app is the instrument for D1's research
question. Every swipe is logged with its dwell time, which gives us:

  * the engagement measure (how many cards a user gets through), and
  * the data-quality filter D1 calls for -- a swipe faster than anyone could
    read the card is flagged rather than silently trusted.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C
from src.data_load import EXPERIENCE_ORDER, load_resumes
from src.features import Preferences
from src.recommend import recommend

WORK_TYPES = ["Full-time", "Part-time", "Contract", "Internship", "Temporary", "Volunteer"]

TIER_STYLE = {
    "Good Fit": ("#1a7f5a", "#e7f6ef"),
    "Partial Fit": ("#8a6410", "#fdf4e0"),
    "No Fit": ("#8a2f2f", "#fbecec"),
}

st.set_page_config(page_title="Job Swipe", page_icon="💼", layout="centered")


# --- logging ----------------------------------------------------------------

LOG_FIELDS = [
    "timestamp", "session_id", "resume_id", "job_id", "rank", "fit_tier",
    "p_good", "direction", "dwell_ms", "blind_swipe",
]


def log_swipe(**row) -> None:
    """Append one swipe. Header is written only when the file is new."""
    C.ensure_artifacts()
    is_new = not C.SWIPE_LOG.exists()
    with C.SWIPE_LOG.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# --- cached loaders ---------------------------------------------------------


@st.cache_data(show_spinner=False)
def cached_resumes() -> pd.DataFrame:
    return load_resumes()


@st.cache_data(show_spinner="Scoring 123k postings...")
def cached_deck(
    resume_text: str, prefs_key: tuple, top_n: int, category: str | None = None
) -> pd.DataFrame:
    """`prefs_key` is a hashable form of Preferences, so the cache keys on it.

    `category` is passed through when the resume was picked from the dataset
    dropdown, where it's already known -- otherwise `recommend` falls back to
    inferring it by nearest-neighbour, which exists for pasted resumes that
    arrive with no label and is noticeably less reliable than a known label.
    """
    target, state, remote, level, work_types = prefs_key
    prefs = Preferences(
        target_salary=target,
        state=state,
        wants_remote=remote,
        experience_level=level,
        work_types=work_types,
    )
    return recommend(resume_text, prefs, top_n=top_n, category=category)


# --- session state ----------------------------------------------------------


def reset_deck() -> None:
    """Clear the current deck and counters.

    `index` and `liked` are reset rather than popped: this runs from the
    sidebar callback, which is *after* `init_state` in the same script run, so
    popping them would leave the rest of the run with no values to read.
    """
    st.session_state.pop("deck", None)
    st.session_state.pop("shown_at", None)
    st.session_state["index"] = 0
    st.session_state["liked"] = []


def init_state() -> None:
    st.session_state.setdefault("session_id", datetime.now().strftime("%Y%m%d-%H%M%S"))
    st.session_state.setdefault("index", 0)
    st.session_state.setdefault("liked", [])


init_state()


# --- sidebar: the preference survey ----------------------------------------

resumes = cached_resumes()

with st.sidebar:
    st.header("Your profile")

    source = st.radio("Resume source", ["Pick from dataset", "Paste my own"], index=0)

    if source == "Pick from dataset":
        category = st.selectbox("Category", sorted(resumes["category"].unique()))
        pool = resumes[resumes["category"] == category]
        resume_id = st.selectbox("Resume", pool["resume_id"].tolist())
        row = pool.loc[pool.resume_id == resume_id].iloc[0]
        resume_text, active_resume_id = row.resume_text, int(row.resume_id)
        with st.expander("Resume text"):
            st.caption(resume_text[:1500])
    else:
        resume_text = st.text_area("Paste your resume", height=220)
        active_resume_id = -1
        category = None  # unknown for a pasted resume -- recommend() infers it

    st.divider()
    st.header("What you're looking for")
    st.caption("D1's survey features. Leave blank to ignore a field.")

    use_salary = st.checkbox("I have a salary target")
    target_salary = (
        st.number_input("Target annual salary (USD)", 20_000, 500_000, 75_000, 5_000)
        if use_salary else None
    )

    use_state = st.checkbox("I have a location preference")
    state = st.text_input("State (2-letter)", "NC", max_chars=2).upper() if use_state else None

    wants_remote = st.checkbox("Remote only")
    use_level = st.checkbox("I have an experience level")
    experience_level = (
        st.selectbox("Experience level", list(EXPERIENCE_ORDER)) if use_level else None
    )
    work_types = tuple(st.multiselect("Work types", WORK_TYPES))

    deck_size = st.slider("Deck size", 10, 100, 30, 10)

    if st.button("Build deck", type="primary", width="stretch"):
        reset_deck()
        st.session_state["pending"] = True

prefs_key = (target_salary, state, wants_remote, experience_level, work_types)


# --- build the deck ---------------------------------------------------------

st.title("💼 Job Swipe")
st.caption("Swipe-card job matching · DTSC3602 Group 9")

if not resume_text or not resume_text.strip():
    st.info("Pick or paste a resume in the sidebar, then press **Build deck**.")
    st.stop()

if "deck" not in st.session_state:
    if not st.session_state.pop("pending", False):
        st.info("Press **Build deck** in the sidebar to start.")
        st.stop()
    try:
        st.session_state["deck"] = cached_deck(resume_text, prefs_key, deck_size, category)
    except FileNotFoundError as exc:
        st.error(
            f"A model artifact is missing: {exc}\n\n"
            "Run `python -m src.embed --build` then `python -m src.model --train`."
        )
        st.stop()
    st.session_state["shown_at"] = time.perf_counter()

deck: pd.DataFrame = st.session_state["deck"]
index: int = st.session_state["index"]

swipe_tab, matches_tab, stats_tab = st.tabs(["Swipe", "Matches", "Session stats"])


# --- swipe tab --------------------------------------------------------------

with swipe_tab:
    if index >= len(deck):
        st.success(f"Deck finished — {len(st.session_state['liked'])} likes.")
        st.caption("Adjust your preferences and press **Build deck** for a fresh deck.")
    else:
        card = deck.iloc[index]
        fg, bg = TIER_STYLE[card.fit_tier]
        salary = f"${card.salary_annual:,.0f}/yr" if pd.notna(card.salary_annual) else "Not listed"
        location = "Remote" if card.remote_allowed else (card.location or "—")

        st.progress((index + 1) / len(deck), text=f"Card {index + 1} of {len(deck)}")
        st.markdown(
            f"""
            <div style="border:1px solid #d8d8d8;border-radius:14px;padding:22px 24px;
                        box-shadow:0 2px 10px rgba(0,0,0,.06);">
              <span style="background:{bg};color:{fg};border-radius:999px;
                           padding:4px 12px;font-size:12px;font-weight:700;">
                {card.fit_tier} · {card.p_good:.0%} confidence
              </span>
              <h3 style="margin:14px 0 2px;">{card.title}</h3>
              <div style="color:#555;font-size:15px;">{card.company_name}</div>
              <div style="color:#777;font-size:14px;margin-top:10px;">
                📍 {location} &nbsp;·&nbsp; 💰 {salary}<br>
                🗂️ {card.formatted_work_type} &nbsp;·&nbsp;
                📈 {card.formatted_experience_level or "Not specified"}
              </div>
              <div style="margin-top:14px;padding-top:12px;border-top:1px solid #eee;
                          color:#444;font-size:14px;">
                <b>Why this matched:</b> {card.why}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if isinstance(card.skill_names, (list, tuple)) and len(card.skill_names):
            st.caption("Skills: " + " · ".join(card.skill_names))

        left, right = st.columns(2)

        def record(direction: str) -> None:
            shown_at = st.session_state.get("shown_at", time.perf_counter())
            dwell_ms = int((time.perf_counter() - shown_at) * 1000)
            log_swipe(
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                session_id=st.session_state["session_id"],
                resume_id=active_resume_id,
                job_id=int(card.job_id),
                rank=index + 1,
                fit_tier=card.fit_tier,
                p_good=round(float(card.p_good), 4),
                direction=direction,
                dwell_ms=dwell_ms,
                blind_swipe=int(dwell_ms < C.BLIND_SWIPE_MS),
            )
            if direction == "like":
                st.session_state["liked"].append(int(card.job_id))
            st.session_state["index"] = index + 1
            st.session_state["shown_at"] = time.perf_counter()

        if left.button("✕  Pass", width="stretch"):
            record("pass")
            st.rerun()
        if right.button("♥  Interested", type="primary", width="stretch"):
            record("like")
            st.rerun()


# --- matches tab ------------------------------------------------------------

with matches_tab:
    liked = st.session_state["liked"]
    if not liked:
        st.caption("Cards you mark **Interested** show up here, with a link to apply.")
    else:
        st.write(f"**{len(liked)}** saved posting{'s' if len(liked) > 1 else ''}")
        for job_id in reversed(liked):
            card = deck.loc[deck.job_id == job_id].iloc[0]
            with st.container(border=True):
                st.markdown(f"**{card.title}** — {card.company_name}")
                st.caption(f"{card.fit_tier} · {card.location or '—'}")
                url = card.application_url or card.job_posting_url
                if isinstance(url, str) and url.startswith("http"):
                    st.link_button("Apply", url)


# --- stats tab --------------------------------------------------------------

with stats_tab:
    st.subheader("This session")
    if not C.SWIPE_LOG.exists():
        st.caption("No swipes logged yet.")
    else:
        log = pd.read_csv(C.SWIPE_LOG)
        mine = log[log.session_id == st.session_state["session_id"]]
        if mine.empty:
            st.caption("No swipes logged yet.")
        else:
            a, b, c = st.columns(3)
            a.metric("Cards swiped", len(mine))
            b.metric("Like rate", f"{(mine.direction == 'like').mean():.0%}")
            c.metric("Median dwell", f"{mine.dwell_ms.median() / 1000:.1f}s")

            blind = int(mine.blind_swipe.sum())
            if blind:
                st.warning(
                    f"{blind} swipe{'s' if blind > 1 else ''} came in under "
                    f"{C.BLIND_SWIPE_MS / 1000:.1f}s — flagged as blind swipes and "
                    "excluded from engagement analysis."
                )

            st.caption("Like rate by predicted fit tier")
            st.dataframe(
                mine.groupby("fit_tier")
                .agg(cards=("direction", "size"),
                     like_rate=("direction", lambda s: (s == "like").mean()))
                .round(2),
                width="stretch",
            )
            st.caption(f"Full log: `{C.SWIPE_LOG}`")
