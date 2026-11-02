"""D2 data-audit dashboard: data dictionary, interpreted plots, data-quality
findings, and the reserved-test-data freeze -- all read live from Supabase.

This is a sibling to `app/streamlit_app.py` (D1's swipe demo), not a
replacement. The swipe app still reads local parquet/CSV; this one exists to
demonstrate the Supabase-backed, Modal-deployed pipeline D2 asks for.

Run locally:      streamlit run app/dashboard.py
Deploy on Modal:  modal deploy modal_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.supabase_client import SupabaseNotConfigured, get_client

st.set_page_config(page_title="D2 Data Audit", page_icon="\U0001F50E", layout="wide")

st.title("\U0001F50E D2 Data Audit Dashboard")
st.caption("DTSC3602 Group 9 · reads from Supabase, deployed on Modal")


@st.cache_data(show_spinner="Querying Supabase...", ttl=300)
def fetch(table: str, *, limit: int = 50_000) -> pd.DataFrame:
    client = get_client()
    resp = client.table(table).select("*").limit(limit).execute()
    return pd.DataFrame(resp.data)


try:
    get_client()
except SupabaseNotConfigured as exc:
    st.error(
        "**Supabase isn't configured yet.**\n\n"
        f"{exc}\n\n"
        "Once `.env` (or your deployment's secret) has `SUPABASE_URL` / "
        "`SUPABASE_KEY`, and `python scripts/upload_to_supabase.py` has run "
        "at least once, this page will populate."
    )
    st.stop()

dict_tab, plots_tab, findings_tab, split_tab = st.tabs(
    ["Data Dictionary", "Plots", "Data Quality Findings", "Reserved Test Data"]
)

with dict_tab:
    st.subheader("Generated from the actual data, not hand-maintained")
    try:
        dictionary = fetch("data_dictionary")
    except Exception as exc:  # noqa: BLE001 -- surface any Supabase-side error plainly
        st.error(f"Couldn't load the data dictionary: {exc}")
    else:
        tables = sorted(dictionary["table_name"].unique())
        choice = st.selectbox("Table", tables)
        st.dataframe(
            dictionary[dictionary.table_name == choice].drop(columns=["table_name"]),
            width="stretch",
            hide_index=True,
        )

with plots_tab:
    st.subheader("Regenerated live from the Supabase-backed postings/resumes tables")
    show_all = st.toggle("Show all candidates (not just the chosen 3-5)", value=False)

    try:
        postings = fetch("postings")
        resumes = fetch("resumes")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't load postings/resumes: {exc}")
    else:
        import matplotlib.pyplot as plt

        from src import viz

        viz.use_style()

        def chart(fig) -> None:
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("**Salary distribution: raw vs. outlier-flagged**")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        raw = postings["salary_annual"].dropna()
        capped = postings.loc[~postings["salary_outlier_flag"].fillna(False), "salary_annual"].dropna()
        axes[0].hist(raw.clip(upper=raw.quantile(0.999)) if len(raw) else raw, bins=40, color=viz.CATEGORICAL[0])
        axes[0].set_title("raw salary_annual")
        axes[1].hist(capped, bins=40, color=viz.CATEGORICAL[2])
        axes[1].set_title("outliers flagged, not plotted")
        for ax in axes:
            viz.despine(ax)
        plt.tight_layout()
        chart(fig)

        st.markdown("**Resumes per category**")
        counts = resumes["category"].value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(counts.index, counts.values, color=viz.CATEGORICAL[0])
        viz.despine(ax)
        plt.tight_layout()
        chart(fig)

        if show_all:
            st.info(
                "The full 12-candidate set (duplicate impact, sentinel-value "
                "missingness, orphaned FKs, encoding corruption, and more) lives "
                "in `notebooks/03_data_audit.ipynb` -- this live view regenerates "
                "the two plots that anchor the data note; the rest are static "
                "figures exported to `write ups/d2_figures/` rather than "
                "re-queried on every page load."
            )

with findings_tab:
    st.subheader("What we fixed vs. what we documented")
    try:
        findings = fetch("data_quality_findings")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't load findings: {exc}")
    else:
        findings = findings.sort_values("fixed", ascending=False)
        for _, row in findings.iterrows():
            icon = "✅ fixed" if row["fixed"] else "\U0001F4DD documented only"
            st.markdown(f"**{row['finding']}** — {icon}")
            st.caption(f"{row['detail']} — {row['count']:,} rows ({row['pct_of_total']:.2f}%)")

with split_tab:
    st.subheader("D2's reserved final test data")
    st.caption(
        "Not a new split -- D1's existing doubly-disjoint 80/20 partition, "
        "frozen and fingerprinted so it can't silently drift."
    )
    try:
        summary = fetch("test_split_summary")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't load the split summary: {exc}")
    else:
        if summary.empty:
            st.warning("No freeze record yet -- run `python -m src.pairs --freeze-check`.")
        else:
            row = summary.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Reserved pairs", f"{row['n_pairs']:,}")
            c2.metric("Resumes", f"{row['n_resumes']:,}")
            c3.metric("Jobs", f"{row['n_jobs']:,}")
            st.caption(f"Frozen on {row['frozen_on']}")
            st.code(
                f"resume_id_sha256  {row['resume_id_sha256']}\n"
                f"job_id_sha256     {row['job_id_sha256']}",
                language="text",
            )
