"""
Streamlit Dashboard for Campus Hiring Analytics.

Reads deduplicated drive data from SQLite and displays interactive
visualizations with filters and CSV export.

Run with: streamlit run src/dashboard/dashboard.py
"""

import sys
import os
import io

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from src.storage.db import DatabaseManager

# ── Page Config ──────────────────────────────────────────────

st.set_page_config(
    page_title="Campus Hiring Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .metric-card {
        border-radius: 16px;
        padding: 24px 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-value { font-size: 2rem; font-weight: 700; margin: 0; line-height: 1.2; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; margin: 4px 0 0 0; font-weight: 500; }
    .section-header { border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; margin-top: 32px; }
    div[data-testid="stDataFrame"] { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Gradient color presets ───────────────────────────────────

GRADIENTS = [
    ("#667eea", "#764ba2"),   # Purple
    ("#f093fb", "#f5576c"),   # Pink
    ("#4facfe", "#00f2fe"),   # Blue
    ("#43e97b", "#38f9d7"),   # Green
    ("#fa709a", "#fee140"),   # Orange
    ("#a18cd1", "#fbc2eb"),   # Lavender
    ("#ffecd2", "#fcb69f"),   # Peach
]


def metric_card(label: str, value, idx: int = 0):
    c1, c2 = GRADIENTS[idx % len(GRADIENTS)]
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, {c1} 0%, {c2} 100%);">
        <p class="metric-value">{value}</p>
        <p class="metric-label">{label}</p>
    </div>
    """, unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_drives() -> pd.DataFrame:
    """Load deduplicated drive data."""
    db = DatabaseManager()
    data = db.get_drives()
    return pd.DataFrame(data) if data else pd.DataFrame()


@st.cache_data(ttl=60)
def load_structured() -> pd.DataFrame:
    """Load per-email structured data (for deeper analysis)."""
    db = DatabaseManager()
    data = db.get_structured_data()
    return pd.DataFrame(data) if data else pd.DataFrame()


@st.cache_data(ttl=60)
def load_audit() -> pd.DataFrame:
    """Load audit log."""
    db = DatabaseManager()
    data = db.get_audit_log()
    return pd.DataFrame(data) if data else pd.DataFrame()


def main():
    # ── Header ───────────────────────────────────────────────

    st.title("🎓 Campus Hiring Analytics Dashboard")
    st.caption("LLM-powered semantic extraction • Deduplicated by drive • Filterable & exportable")

    # Refresh button
    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    df = load_drives()

    if df.empty:
        st.warning("No drive data found. Run the pipeline first:")
        st.code("python -m src.main --extract-only", language="bash")
        return

    # ── Sidebar Filters ──────────────────────────────────────

    st.sidebar.header("🔍 Filters")

    # Company filter
    companies = sorted(df["company_name"].dropna().unique())
    selected_companies = st.sidebar.multiselect("Companies", companies)

    # CTC range
    ctc_vals = df["ctc_lpa"].dropna()
    if not ctc_vals.empty:
        ctc_min, ctc_max = float(ctc_vals.min()), float(ctc_vals.max())
        if ctc_min < ctc_max:
            ctc_range = st.sidebar.slider("CTC Range (LPA)", ctc_min, ctc_max, (ctc_min, ctc_max))
        else:
            ctc_range = (ctc_min, ctc_max)
    else:
        ctc_range = None

    # CGPA filter
    cgpa_vals = df["cgpa_cutoff"].dropna()
    if not cgpa_vals.empty:
        cgpa_min, cgpa_max = float(cgpa_vals.min()), float(cgpa_vals.max())
        if cgpa_min < cgpa_max:
            cgpa_range = st.sidebar.slider("CGPA Cutoff Range", cgpa_min, cgpa_max, (cgpa_min, cgpa_max))
        else:
            cgpa_range = (cgpa_min, cgpa_max)
    else:
        cgpa_range = None

    # Apply filters
    fdf = df.copy()
    if selected_companies:
        fdf = fdf[fdf["company_name"].isin(selected_companies)]
    if ctc_range:
        fdf = fdf[(fdf["ctc_lpa"].isna()) | ((fdf["ctc_lpa"] >= ctc_range[0]) & (fdf["ctc_lpa"] <= ctc_range[1]))]
    if cgpa_range:
        fdf = fdf[(fdf["cgpa_cutoff"].isna()) | ((fdf["cgpa_cutoff"] >= cgpa_range[0]) & (fdf["cgpa_cutoff"] <= cgpa_range[1]))]

    # ── Section 1: Summary Metrics ───────────────────────────

    st.markdown('<h2 class="section-header">📈 Summary Metrics</h2>', unsafe_allow_html=True)

    valid_ctc = fdf["ctc_lpa"].dropna()
    valid_cgpa = fdf["cgpa_cutoff"].dropna()
    valid_sel = fdf["selection_count"].dropna()

    cols = st.columns(7)
    metrics = [
        ("Total Drives", len(fdf)),
        ("Unique Companies", fdf["company_name"].nunique()),
        ("Avg CTC (LPA)", f"{valid_ctc.mean():.1f}" if not valid_ctc.empty else "—"),
        ("Median CTC (LPA)", f"{valid_ctc.median():.1f}" if not valid_ctc.empty else "—"),
        ("Highest CTC (LPA)", f"{valid_ctc.max():.1f}" if not valid_ctc.empty else "—"),
        ("Avg CGPA Cutoff", f"{valid_cgpa.mean():.1f}" if not valid_cgpa.empty else "—"),
        ("Total Offers", int(valid_sel.sum()) if not valid_sel.empty else "—"),
    ]
    for i, (label, val) in enumerate(metrics):
        with cols[i]:
            metric_card(label, val, i)

    st.markdown("---")

    # ── Section 2: Drive Overview Table ──────────────────────

    st.markdown('<h2 class="section-header">🏢 Drive Overview</h2>', unsafe_allow_html=True)

    display_cols = [
        "company_name", "role", "ctc_lpa", "cgpa_cutoff",
        "eligibility_branches", "registration_deadline",
        "selection_count", "total_openings", "email_count"
    ]
    existing = [c for c in display_cols if c in fdf.columns]
    table_df = fdf[existing].copy()

    rename = {
        "company_name": "Company", "role": "Role", "ctc_lpa": "CTC (LPA)",
        "cgpa_cutoff": "CGPA Cutoff", "eligibility_branches": "Branches",
        "registration_deadline": "Deadline", "selection_count": "Selections",
        "total_openings": "Openings", "email_count": "Emails",
    }
    table_df = table_df.rename(columns={k: v for k, v in rename.items() if k in table_df.columns})

    search = st.text_input("🔎 Search companies, roles, or branches", "")
    if search:
        mask = table_df.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)
        table_df = table_df[mask]

    st.dataframe(table_df, use_container_width=True, height=420)

    # CSV export
    csv_buffer = io.StringIO()
    table_df.to_csv(csv_buffer, index=False)
    st.download_button(
        "📥 Export as CSV", csv_buffer.getvalue(),
        file_name="campus_hiring_drives.csv", mime="text/csv"
    )

    st.markdown("---")

    # ── Section 3: Visualizations ────────────────────────────

    st.markdown('<h2 class="section-header">📊 Visualizations</h2>', unsafe_allow_html=True)

    tabs = st.tabs([
        "📅 Hiring by Month", "💰 CTC Distribution", "📚 CGPA Distribution",
        "🏆 Top 10 CTC", "👔 Roles", "🎓 Branches", "🔥 Heatmap"
    ])

    # Tab 1: Hiring frequency by month
    with tabs[0]:
        if "first_seen" in fdf.columns:
            date_df = fdf.copy()
            date_df["month"] = pd.to_datetime(date_df["first_seen"], errors="coerce").dt.to_period("M").astype(str)
            mc = date_df.groupby("month").size().reset_index(name="Drives")
            fig = px.bar(mc, x="month", y="Drives", title="Hiring Frequency by Month",
                         color="Drives", color_continuous_scale="Viridis")
            fig.update_layout(xaxis_title="Month", yaxis_title="Number of Drives")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Date information not available.")

    # Tab 2: CTC distribution
    with tabs[1]:
        if not valid_ctc.empty:
            fig = px.histogram(fdf.dropna(subset=["ctc_lpa"]), x="ctc_lpa", nbins=20,
                               title="CTC Distribution (LPA)", color_discrete_sequence=["#667eea"])
            fig.update_layout(xaxis_title="CTC (LPA)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No CTC data available.")

    # Tab 3: CGPA cutoff distribution
    with tabs[2]:
        if not valid_cgpa.empty:
            fig = px.histogram(fdf.dropna(subset=["cgpa_cutoff"]), x="cgpa_cutoff", nbins=15,
                               title="CGPA Cutoff Distribution", color_discrete_sequence=["#43e97b"])
            fig.update_layout(xaxis_title="CGPA", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No CGPA data available.")

    # Tab 4: Top 10 CTC
    with tabs[3]:
        if not valid_ctc.empty:
            top10 = fdf.dropna(subset=["ctc_lpa"]).sort_values("ctc_lpa", ascending=False).head(10)
            fig = px.bar(top10, x="ctc_lpa", y="company_name", orientation="h",
                         title="Top 10 Highest Paying Companies",
                         color="ctc_lpa", color_continuous_scale="Plasma", text="ctc_lpa")
            fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="CTC (LPA)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No CTC data available.")

    # Tab 5: Role distribution
    with tabs[4]:
        roles = fdf["role"].dropna()
        if not roles.empty:
            rc = roles.value_counts().head(15).reset_index()
            rc.columns = ["Role", "Count"]
            fig = px.pie(rc, names="Role", values="Count", title="Role Distribution",
                         color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No role data available.")

    # Tab 6: Branch eligibility distribution
    with tabs[5]:
        branches_col = fdf["eligibility_branches"].dropna()
        if not branches_col.empty:
            all_branches = []
            for b_str in branches_col:
                all_branches.extend([b.strip() for b in str(b_str).split(",") if b.strip()])
            if all_branches:
                bc = pd.Series(all_branches).value_counts().head(15).reset_index()
                bc.columns = ["Branch", "Count"]
                fig = px.bar(bc, x="Branch", y="Count", title="Branch Eligibility Distribution",
                             color="Count", color_continuous_scale="Teal")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No branch data available.")
        else:
            st.info("No branch data available.")

    # Tab 7: CGPA vs CTC scatter
    with tabs[6]:
        scatter_df = fdf.dropna(subset=["ctc_lpa", "cgpa_cutoff"])
        if not scatter_df.empty:
            fig = px.scatter(scatter_df, x="cgpa_cutoff", y="ctc_lpa",
                             hover_data=["company_name", "role"],
                             title="CGPA Cutoff vs CTC",
                             color="ctc_lpa", color_continuous_scale="Inferno",
                             size="ctc_lpa", size_max=18)
            fig.update_layout(xaxis_title="CGPA Cutoff", yaxis_title="CTC (LPA)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for CGPA vs CTC scatter.")

    # ── Audit Log (expandable) ───────────────────────────────

    st.markdown("---")
    with st.expander("🔍 Data Audit Log"):
        audit_df = load_audit()
        if not audit_df.empty:
            st.dataframe(audit_df, use_container_width=True, height=300)
        else:
            st.info("No audit entries.")

    # ── Footer ───────────────────────────────────────────────

    st.markdown("---")
    st.caption(f"Showing {len(fdf)} drives from {len(df)} total • "
               f"Data source: placement_officer@kletech.ac.in")


if __name__ == "__main__":
    main()
