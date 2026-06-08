from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.graph import run_deal_memo_graph
from src.render import SOURCE_LABELS, render_sources_used
from src.sample_data import SAMPLE_NOTES

try:
    import plotly.express as px
except Exception:  # pragma: no cover - Streamlit surface handles this.
    px = None


st.set_page_config(page_title="Deal Memo Drafter", page_icon="📈", layout="wide")

st.title("Deal Memo Drafter")
st.caption("LangGraph-powered analyst copilot for source-cited first-draft investment memos.")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Research provider", ["mock", "tavily", "exa"], index=0)
    prefer_langgraph = st.toggle("Use LangGraph runtime", value=True)
    st.info("Use `mock` for a deterministic demo. Use Tavily or Exa when the matching API key is configured.")

notes = st.text_area("Raw partner notes", value=SAMPLE_NOTES, height=260)
generate = st.button("Generate memo", type="primary")

if generate:
    if not notes.strip():
        st.error("Paste rough partner notes before generating a memo.")
        st.stop()

    with st.spinner("Researching, grounding, drafting, and validating memo..."):
        state = run_deal_memo_graph(notes, search_provider=provider, prefer_langgraph=prefer_langgraph)

    memo_tab, evidence_tab, charts_tab, questions_tab, trace_tab = st.tabs(
        ["Memo", "Research Evidence", "Charts", "Open Questions", "Trace"]
    )

    with memo_tab:
        if state["errors"]:
            for error in state["errors"]:
                st.warning(error)
        st.markdown(state["memo_markdown"])

    with evidence_tab:
        evidence = state["evidence"]
        if evidence:
            grouped = {}
            for item in evidence:
                grouped.setdefault(item["evidence_type"], []).append(item)

            st.metric("Sources Used", len(evidence))
            for evidence_type, items in grouped.items():
                label = SOURCE_LABELS.get(evidence_type, evidence_type.title())
                with st.expander(f"{label} ({len(items)})", expanded=False):
                    st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
        else:
            st.info("No public evidence was available. The memo is based on partner notes only.")

    with charts_tab:
        if px is None:
            st.error("Plotly is not installed. Run `pip install -r requirements.txt` to enable charts.")
        else:
            chart_data = state["chart_data"]
            visualizations = state["memo_json"].get("visualizations", {})
            evidence = state["evidence"]

            st.subheader("Evidence Coverage")
            coverage_rows = chart_data.get("evidence_coverage", {}).get("rows", [])
            if coverage_rows:
                coverage_df = pd.DataFrame(coverage_rows)
                fig = px.bar(
                    coverage_df,
                    x="coverage",
                    y="category",
                    orientation="h",
                    color="status",
                    range_x=[0, 100],
                    hover_data=["confidence", "missing_slots"],
                    title="Evidence coverage by investment dimension",
                    labels={"coverage": "Coverage %", "category": "Investment dimension"},
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
                focus = chart_data.get("evidence_coverage", {}).get("recommended_diligence_focus", [])
                if focus:
                    st.markdown("**Recommended diligence focus**")
                    st.dataframe(pd.DataFrame(focus), use_container_width=True, hide_index=True)

            st.subheader("Investment Scorecard")
            scorecard = visualizations.get("investment_scorecard", {})
            rating_rows = scorecard.get("ratings", [])
            if rating_rows:
                st.markdown(f"**Overall rating:** {scorecard.get('overall_rating', 'Unknown')}")
                st.caption(scorecard.get("rating_basis", ""))
                st.markdown(f"**Top strengths:** {', '.join(scorecard.get('top_strengths', [])) or 'Unknown'}")
                st.markdown(f"**Top concerns:** {', '.join(scorecard.get('top_concerns', [])) or 'Unknown'}")
                st.dataframe(pd.DataFrame(rating_rows), use_container_width=True, hide_index=True)
                factor_rows = []
                for row in scorecard.get("raw_scores", []):
                    for factor in row.get("factors", []):
                        factor_rows.append(
                            {
                                "category": row.get("dimension"),
                                "factor": factor.get("name"),
                                "evidence_signal": "Known" if factor.get("score") is not None else "Unknown",
                                "reason": factor.get("reason"),
                                "source_ids": factor.get("source_ids"),
                            }
                        )
                if factor_rows:
                    st.markdown("**Factor evidence notes**")
                    st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)
            unknown_scores = [row for row in rating_rows if row.get("rating") == "Unknown"]
            if unknown_scores:
                st.warning("Some categories are marked Unknown because reliable evidence was not found.")
                st.dataframe(pd.DataFrame(unknown_scores), use_container_width=True, hide_index=True)

            st.subheader("Partner Claim Verification")
            verification = visualizations.get("claim_verification_summary", {})
            verification_df = pd.DataFrame(
                [
                    {"status": "Verified", "count": verification.get("verified_count", 0)},
                    {"status": "Partially Verified", "count": verification.get("partially_verified_count", 0)},
                    {"status": "Not Verified", "count": verification.get("not_verified_count", 0)},
                ]
            )
            fig = px.bar(
                verification_df,
                x="count",
                y="status",
                orientation="h",
                title="Partner claims checked against public research",
                labels={"count": "Claim count", "status": "Verification status"},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pd.DataFrame(state["memo_json"].get("partner_note_verification", [])), use_container_width=True, hide_index=True)

            st.subheader("Risk Breakdown")
            risk_rows = visualizations.get("risk_breakdown", {}).get("risks", [])
            known_risks = [row for row in risk_rows if row.get("score") is not None]
            if known_risks:
                risk_df = pd.DataFrame(known_risks)
                fig = px.bar(
                    risk_df,
                    x="score",
                    y="risk",
                    orientation="h",
                    range_x=[0, 10],
                    hover_data=["risk_type", "confidence", "reason", "evidence", "source_ids"],
                    title="Risk score by diligence area; higher score = higher risk",
                    labels={"score": "Risk score", "risk": "Risk"},
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
            unknown_risks = [row for row in risk_rows if row.get("score") is None]
            if unknown_risks:
                st.warning("Some risks are marked Unknown because reliable evidence was not found.")
                st.dataframe(pd.DataFrame(unknown_risks), use_container_width=True, hide_index=True)

            st.subheader("Bull Case vs Bear Case")
            factors = visualizations.get("bull_bear_weights", {}).get("factors", [])
            if factors:
                factor_df = pd.DataFrame(factors)
                fig = px.bar(
                    factor_df,
                    x="value",
                    y="factor",
                    orientation="h",
                    color="value",
                    color_continuous_scale="RdYlGn",
                    range_x=[-10, 10],
                    hover_data=["reason", "source_ids"],
                    title="Weighted investment factors",
                    labels={"value": "Weight", "factor": "Factor"},
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Competitive Positioning Table")
            table_df = pd.DataFrame(chart_data.get("competitive_table", []))
            if table_df.empty:
                st.info("Insufficient competitor evidence for a positioning table.")
            else:
                st.dataframe(table_df, use_container_width=True, hide_index=True)

            st.subheader("Competitive Landscape Map")
            landscape_rows = chart_data.get("competitive_landscape", {}).get("rows", [])
            if landscape_rows:
                st.dataframe(pd.DataFrame(landscape_rows), use_container_width=True, hide_index=True)

            st.subheader("Traction Analysis")
            traction_rows = chart_data.get("traction_analysis", {}).get("rows", [])
            if traction_rows:
                st.caption(chart_data.get("traction_analysis", {}).get("summary", ""))
                st.dataframe(pd.DataFrame(traction_rows), use_container_width=True, hide_index=True)

            st.subheader("Business Model Analysis")
            business_rows = chart_data.get("business_model_analysis", {}).get("rows", [])
            if business_rows:
                st.caption(chart_data.get("business_model_analysis", {}).get("summary", ""))
                st.dataframe(pd.DataFrame(business_rows), use_container_width=True, hide_index=True)

            st.subheader("Risk Taxonomy")
            taxonomy_rows = chart_data.get("risk_taxonomy", {}).get("rows", [])
            if taxonomy_rows:
                st.dataframe(pd.DataFrame(taxonomy_rows), use_container_width=True, hide_index=True)

            used = sorted(
                {
                    sid
                    for row in rating_rows + risk_rows + factors
                    for sid in row.get("source_ids", [])
                }
            )
            st.markdown("**Sources used by visualizations**")
            st.markdown(render_sources_used(evidence, used))

    with questions_tab:
        profile_questions = state["company_profile"].get("open_questions", [])
        if profile_questions:
            for question in profile_questions:
                st.markdown(f"- {question}")
        else:
            st.success("No critical open questions detected from the notes.")

        if state["validation"].get("warnings"):
            st.subheader("Validation Warnings")
            for warning in state["validation"]["warnings"]:
                st.warning(warning)

    with trace_tab:
        st.json(state.get("trace", []))
        with st.expander("Full final state"):
            st.code(json.dumps(state, indent=2), language="json")
else:
    st.info("Paste partner notes or use the sample, then generate a memo.")
