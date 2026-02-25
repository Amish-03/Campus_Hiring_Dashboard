"""
Streamlit Dashboard for Campus Hiring Analytics.

Reads structured data from SQLite and displays interactive visualizations.
Run with: streamlit run src/dashboard/dashboard.py
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.storage.db import DatabaseManager

# ── Page Config ──────────────────────────────────────────────

st.set_page_config(
    page_title="Campus Hiring Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .metric-label { font-size: 0.9rem; opacity: 0.85; margin: 0; }
    .stDataFrame { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


def metric_card(label: str, value, color_start: str = "#667eea", color_end: str = "#764ba2"):
    """Render a styled metric card."""
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, {color_start} 0%, {color_end} 100%);">
        <p class="metric-value">{value}</p>
        <p class="metric-label">{label}</p>
    </div>
    """, unsafe_allow_html=True)


def load_data() -> pd.DataFrame:
    """Load structured hiring data from SQLite into a DataFrame."""
    db = DatabaseManager()
    data = db.get_structured_data()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    return df


def main():
    st.title("🎓 Campus Hiring Analytics Dashboard")
    st.caption("Powered by LLM-based semantic extraction from placement emails")

    df = load_data()

    if df.empty:
        st.warning("No structured hiring data found. Run the LLM extraction pipeline first.")
        st.code("python -m src.main --extract-only", language="bash")
        return

    # ── Sidebar Filters ──────────────────────────────────────

    st.sidebar.header("🔍 Filters")

    # Company filter
    companies = sorted(df["company_name"].dropna().unique())
    selected_companies = st.sidebar.multiselect("Companies", companies, default=[])

    # CTC range
    ctc_values = df["ctc_lpa"].dropna()
    if not ctc_values.empty:
        ctc_min, ctc_max = float(ctc_values.min()), float(ctc_values.max())
        ctc_range = st.sidebar.slider(
            "CTC Range (LPA)", ctc_min, ctc_max, (ctc_min, ctc_max)
        )
    else:
        ctc_range = (0, 100)

    # Apply filters
    filtered_df = df.copy()
    if selected_companies:
        filtered_df = filtered_df[filtered_df["company_name"].isin(selected_companies)]
    if not ctc_values.empty:
        filtered_df = filtered_df[
            (filtered_df["ctc_lpa"].isna()) |
            ((filtered_df["ctc_lpa"] >= ctc_range[0]) & (filtered_df["ctc_lpa"] <= ctc_range[1]))
        ]

    # ── Section B: Summary Metrics ───────────────────────────

    st.header("📈 Summary Metrics")

    valid_ctc = filtered_df["ctc_lpa"].dropna()
    valid_cgpa = filtered_df["cgpa_cutoff"].dropna()
    valid_selections = filtered_df["selection_count"].dropna()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        metric_card("Total Companies", len(filtered_df["company_name"].dropna().unique()),
                     "#667eea", "#764ba2")
    with col2:
        avg_ctc = f"{valid_ctc.mean():.1f}" if not valid_ctc.empty else "N/A"
        metric_card("Avg CTC (LPA)", avg_ctc, "#f093fb", "#f5576c")
    with col3:
        max_ctc = f"{valid_ctc.max():.1f}" if not valid_ctc.empty else "N/A"
        metric_card("Highest CTC (LPA)", max_ctc, "#4facfe", "#00f2fe")
    with col4:
        avg_cgpa = f"{valid_cgpa.mean():.1f}" if not valid_cgpa.empty else "N/A"
        metric_card("Avg CGPA Cutoff", avg_cgpa, "#43e97b", "#38f9d7")
    with col5:
        total_offers = int(valid_selections.sum()) if not valid_selections.empty else "N/A"
        metric_card("Total Offers", total_offers, "#fa709a", "#fee140")

    st.markdown("---")

    # ── Section A: Company Overview Table ────────────────────

    st.header("🏢 Company Overview")

    display_cols = [
        "company_name", "role", "ctc_lpa", "cgpa_cutoff",
        "eligibility_branches", "registration_deadline",
        "selection_count", "subject"
    ]
    existing_cols = [c for c in display_cols if c in filtered_df.columns]
    table_df = filtered_df[existing_cols].copy()

    # Rename columns for display
    rename_map = {
        "company_name": "Company",
        "role": "Role",
        "ctc_lpa": "CTC (LPA)",
        "cgpa_cutoff": "CGPA Cutoff",
        "eligibility_branches": "Eligible Branches",
        "registration_deadline": "Reg. Deadline",
        "selection_count": "Selections",
        "subject": "Email Subject",
    }
    table_df = table_df.rename(columns={k: v for k, v in rename_map.items() if k in table_df.columns})

    # Search
    search = st.text_input("🔎 Search companies or roles", "")
    if search:
        mask = table_df.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)
        table_df = table_df[mask]

    st.dataframe(table_df, use_container_width=True, height=400)

    st.markdown("---")

    # ── Section C: Visualizations ────────────────────────────

    st.header("📊 Visualizations")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📅 Hiring by Month", "💰 CTC Distribution",
        "📚 CGPA Distribution", "🏆 Top 10 CTC", "👔 Role Distribution"
    ])

    # Tab 1: Hiring frequency by month
    with tab1:
        if "email_date" in filtered_df.columns:
            date_df = filtered_df.copy()
            date_df["month"] = pd.to_datetime(date_df["email_date"], errors="coerce").dt.to_period("M").astype(str)
            month_counts = date_df.groupby("month").size().reset_index(name="Count")
            fig = px.bar(
                month_counts, x="month", y="Count",
                title="Hiring Frequency by Month",
                color="Count", color_continuous_scale="Viridis",
            )
            fig.update_layout(xaxis_title="Month", yaxis_title="Number of Drives")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Date information not available.")

    # Tab 2: CTC distribution
    with tab2:
        if not valid_ctc.empty:
            fig = px.histogram(
                filtered_df.dropna(subset=["ctc_lpa"]),
                x="ctc_lpa", nbins=20,
                title="CTC Distribution (LPA)",
                color_discrete_sequence=["#667eea"],
            )
            fig.update_layout(xaxis_title="CTC (LPA)", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No CTC data available.")

    # Tab 3: CGPA cutoff distribution
    with tab3:
        if not valid_cgpa.empty:
            fig = px.histogram(
                filtered_df.dropna(subset=["cgpa_cutoff"]),
                x="cgpa_cutoff", nbins=15,
                title="CGPA Cutoff Distribution",
                color_discrete_sequence=["#43e97b"],
            )
            fig.update_layout(xaxis_title="CGPA Cutoff", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No CGPA data available.")

    # Tab 4: Top 10 highest paying companies
    with tab4:
        if not valid_ctc.empty:
            top10 = (
                filtered_df.dropna(subset=["ctc_lpa"])
                .sort_values("ctc_lpa", ascending=False)
                .drop_duplicates(subset=["company_name"])
                .head(10)
            )
            fig = px.bar(
                top10, x="ctc_lpa", y="company_name",
                orientation="h",
                title="Top 10 Highest Paying Companies",
                color="ctc_lpa", color_continuous_scale="Plasma",
                text="ctc_lpa",
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="CTC (LPA)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No CTC data available.")

    # Tab 5: Role distribution
    with tab5:
        valid_roles = filtered_df["role"].dropna()
        if not valid_roles.empty:
            role_counts = valid_roles.value_counts().head(15).reset_index()
            role_counts.columns = ["Role", "Count"]
            fig = px.pie(
                role_counts, names="Role", values="Count",
                title="Role Distribution (Top 15)",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No role data available.")

    # ── Footer ───────────────────────────────────────────────

    st.markdown("---")
    st.caption(f"Data: {len(filtered_df)} records from {len(df)} total | "
               f"Model: {df['model_used'].iloc[0] if 'model_used' in df.columns and not df.empty else 'N/A'}")


if __name__ == "__main__":
    main()
