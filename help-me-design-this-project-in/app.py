from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.graph import run_deal_memo_graph
from src.render import SOURCE_LABELS, render_sources_used
from src.sample_data import SAMPLE_NOTES
from src.search import get_secret

try:
    import plotly.express as px
except Exception:  # pragma: no cover - Streamlit surface handles this.
    px = None


PARTNER_SECTIONS = [
    "Company Overview",
    "Why Now",
    "Market Opportunity",
    "Competitive Landscape",
    "Key Risks",
    "Investment Thesis",
]


def _claim_line(claim: dict) -> str:
    text = claim.get("text", "")
    sources = " ".join(f"[{sid}]" for sid in claim.get("source_ids", []))
    if claim.get("note_reference"):
        sources = f"{sources} [Notes]".strip()
    if claim.get("analyst_inference"):
        sources = f"{sources} [Inference]".strip()
    return f"- {text} {sources}".strip()


def _section_claims(memo: dict, title: str) -> list[dict]:
    section = next((item for item in memo.get("sections", []) if item.get("title") == title), {})
    return section.get("claims", [])


def _top_gaps(memo: dict, limit: int = 3) -> list[str]:
    missing = memo.get("missing_data", {})
    gaps = missing.get("decision_blocking_unknowns", []) or missing.get("high_priority", [])
    return [gap.get("gap", "") for gap in gaps[:limit] if gap.get("gap")]


def _top_validation_warning(state: dict) -> str:
    validator_warnings = state["memo_json"].get("memo_validator", {}).get("warnings", [])
    if validator_warnings:
        return validator_warnings[0].get("warning", "")
    warnings = state["validation"].get("warnings", [])
    return warnings[0] if warnings else "No major validation warning."


def _company_description(memo: dict) -> str:
    understanding = memo.get("company_understanding", {})
    company = memo.get("company", "Company")
    product = understanding.get("primary_product", "product")
    buyer = understanding.get("target_buyer", "target buyers")
    if product in {"Unknown", "Clarification required"}:
        return f"{company}'s product and buyer require clarification from current notes."
    return f"{company} appears to provide {str(product).lower()} for {buyer}."


def build_deal_snapshot(state: dict) -> dict:
    memo = state["memo_json"]
    recommendation = memo.get("recommendation", {})
    thesis = memo.get("thesis_framework", {})
    missing = memo.get("missing_data", {})
    return {
        "company": memo.get("company", "Unknown"),
        "initial_view": memo.get("executive_summary", {}).get("view", "Needs diligence"),
        "recommendation": recommendation.get("decision", "Needs diligence"),
        "confidence": recommendation.get("confidence", memo.get("confidence", "Low")),
        "decision_readiness": missing.get("decision_readiness", recommendation.get("decision_readiness", 0)),
        "description": _company_description(memo),
        "why_now": thesis.get("why_now", {}).get("answer", "Why-now requires diligence."),
        "top_gaps": _top_gaps(memo),
        "top_warning": _top_validation_warning(state),
    }


def _research_trace(state: dict) -> list[dict]:
    return [
        item
        for item in state.get("trace", [])
        if item.get("node") in {"run_research", "run_context_research", "build_evidence_store", "build_final_evidence_store"}
    ]


def render_research_status(state: dict, provider: str) -> None:
    trace = _research_trace(state)
    provider_events = [
        item.get("output", {}).get("provider")
        for item in trace
        if item.get("output", {}).get("provider")
    ]
    provider_label = ", ".join(str(item) for item in provider_events) or provider
    st.caption(
        f"Research provider: `{provider}` | Provider events: `{provider_label}` | "
        f"Raw search results: `{len(state.get('search_results', []))}` | "
        f"Evidence after filtering: `{len(state.get('evidence', []))}`"
    )


def build_partner_memo_markdown(state: dict) -> str:
    memo = state["memo_json"]
    executive = memo.get("executive_summary", {})
    thesis = memo.get("thesis_framework", {})
    lines = [f"# {memo.get('company', 'Company')} Partner Memo", ""]

    lines.extend(["## Executive Summary", ""])
    lines.append(f"**Initial view:** {executive.get('view', 'Needs diligence')}")
    lines.append(f"**Conviction:** {executive.get('conviction', 'Low')}")
    lines.append("")
    for claim in executive.get("why_it_matters", [])[:3]:
        lines.append(_claim_line(claim))
    for risk in executive.get("key_open_risks", [])[:2]:
        if isinstance(risk, dict):
            lines.append(_claim_line(risk))
    lines.append("")

    lines.extend(["## Company Overview", ""])
    for claim in _section_claims(memo, "Company Overview")[:2]:
        lines.append(_claim_line(claim))
    lines.append("")

    lines.extend(["## Why Now", ""])
    lines.append(f"- {thesis.get('why_now', {}).get('answer', 'Timing requires diligence.')}")
    lines.append("")

    lines.extend(["## Why This Company", ""])
    lines.append(f"- {thesis.get('why_this_company', {}).get('answer', 'Right-to-win requires diligence.')}")
    lines.append("")

    lines.extend(["## Market / Customer Pain", ""])
    lines.append(f"- {thesis.get('why_this_market', {}).get('answer', 'Market pain requires diligence.')}")
    lines.append("")

    lines.extend(["## Traction Signals", ""])
    traction = memo.get("traction_analysis", {})
    lines.append(f"- {traction.get('summary', 'Traction requires diligence.')}")
    for row in [item for item in traction.get("rows", []) if item.get("status") != "Unknown"][:3]:
        lines.append(f"- {row.get('traction_type', '')}: {row.get('status', '')}; {row.get('interpretation', '')}")
    lines.append("")

    lines.extend(["## Competition Summary", ""])
    landscape = memo.get("competitive_landscape", {})
    lines.append(f"- {landscape.get('summary', 'Competition requires diligence.')}")
    for row in landscape.get("rows", [])[:4]:
        examples = ", ".join(row.get("examples", [])[:3]) or "examples require validation"
        lines.append(f"- {row.get('competitor_group', '')}: {examples}. {row.get('why_it_competes', '')}")
    lines.append("")

    lines.extend(["## Key Risks", ""])
    for risk in memo.get("visualizations", {}).get("risk_breakdown", {}).get("risks", [])[:5]:
        lines.append(f"- {risk.get('risk_name', risk.get('risk', 'Risk'))}: {risk.get('reason', '')}")
    lines.append("")

    lines.extend(["## What Needs To Be True", ""])
    for claim in memo.get("what_needs_to_be_true", [])[:6]:
        lines.append(_claim_line(claim))
    lines.append("")

    recommendation = memo.get("recommendation", {})
    lines.extend(["## Recommendation", ""])
    lines.append(f"**Recommendation:** {recommendation.get('decision', 'Needs diligence')}")
    lines.append(f"**Confidence:** {recommendation.get('confidence', 'Low')}")
    lines.append("")
    lines.append(recommendation.get("reason", "Recommendation requires diligence."))
    lines.append("")
    return "\n".join(lines)


def render_deal_snapshot(state: dict) -> None:
    snapshot = build_deal_snapshot(state)
    st.subheader("Deal Snapshot")
    cols = st.columns([1.1, 1.3, 0.8, 0.8])
    cols[0].metric("Company", snapshot["company"])
    cols[1].metric("Recommendation", snapshot["recommendation"])
    cols[2].metric("Confidence", str(snapshot["confidence"]).title())
    cols[3].metric("Readiness", f"{snapshot['decision_readiness']}%")
    st.markdown(f"**Description:** {snapshot['description']}")
    st.markdown(f"**Why now:** {snapshot['why_now']}")
    gaps = ", ".join(snapshot["top_gaps"]) or "No priority gaps detected."
    st.markdown(f"**Top diligence gaps:** {gaps}")
    if snapshot["top_warning"]:
        st.caption(f"Validation note: {snapshot['top_warning']}")


def _question_group(label: str) -> str:
    lower = label.lower()
    if "commercial" in lower or "traction" in lower:
        return "Commercial"
    if "product" in lower or "integration" in lower or "risk-specific" in lower:
        return "Product"
    if "market" in lower or "customer roi" in lower:
        return "Market"
    if "competition" in lower:
        return "Competition"
    if "financial" in lower or "coverage" in lower:
        return "Financial"
    if "team" in lower:
        return "Team"
    if "regulatory" in lower or "security" in lower or "legal" in lower:
        return "Legal/Regulatory"
    return "Commercial"


def grouped_diligence_questions(memo: dict) -> dict[str, list[str]]:
    grouped = {key: [] for key in ["Commercial", "Product", "Market", "Competition", "Financial", "Team", "Legal/Regulatory"]}
    for category, items in memo.get("next_diligence_priorities", {}).items():
        target = _question_group(category)
        for item in items:
            question = item.get("question", str(item)) if isinstance(item, dict) else str(item)
            grouped[target].append(question)
    for gap in memo.get("missing_data", {}).get("decision_blocking_unknowns", []):
        grouped["Commercial"].append(f"Resolve decision-blocking unknown: {gap.get('gap')} ({gap.get('category')}).")
    return {key: values for key, values in grouped.items() if values}


st.set_page_config(page_title="Deal Memo Drafter", page_icon="📈", layout="wide")

st.title("Deal Memo Drafter")
st.caption("LangGraph-powered analyst copilot for source-cited first-draft investment memos.")

with st.sidebar:
    st.header("Settings")
    providers = ["mock", "tavily", "exa"]
    default_provider = "tavily" if get_secret("TAVILY_API_KEY") else "mock"
    provider = st.selectbox("Research provider", providers, index=providers.index(default_provider))
    prefer_langgraph = st.toggle("Use LangGraph runtime", value=True)
    output_mode = st.radio("Output mode", ["Partner Memo", "Full Analyst Report"], index=0)
    st.info("Use `mock` for a deterministic demo. Use Tavily or Exa when the matching API key is configured.")

notes = st.text_area("Raw partner notes", value=SAMPLE_NOTES, height=260)
generate = st.button("Generate memo", type="primary")

if generate:
    if not notes.strip():
        st.error("Paste rough partner notes before generating a memo.")
        st.stop()

    with st.spinner("Researching, grounding, drafting, and validating memo..."):
        state = run_deal_memo_graph(notes, search_provider=provider, prefer_langgraph=prefer_langgraph)

    render_deal_snapshot(state)
    render_research_status(state, provider)
    st.divider()

    memo_tab, evidence_tab, diligence_tab, trace_tab = st.tabs(["Memo", "Evidence", "Diligence", "Trace"])

    with memo_tab:
        if state["errors"]:
            for error in state["errors"]:
                st.warning(error)
        partner_markdown = build_partner_memo_markdown(state)
        memo_markdown = partner_markdown if output_mode == "Partner Memo" else state["memo_markdown"]
        st.download_button("Download markdown", memo_markdown, file_name="deal_memo.md", mime="text/markdown")
        with st.expander("Copy memo", expanded=False):
            st.text_area("Memo markdown", value=memo_markdown, height=280)
        st.markdown(memo_markdown)

    with evidence_tab:
        evidence = state["evidence"]
        memo = state["memo_json"]
        chart_data = state["chart_data"]
        if evidence:
            grouped = {}
            for item in evidence:
                grouped.setdefault(item["evidence_type"], []).append(item)

            st.metric("Sources Used", len(evidence))
            if px is not None:
                quality_df = pd.DataFrame(
                    [
                        {
                            "source_id": item["source_id"],
                            "type": item.get("source_type", "unknown"),
                            "tier": item.get("source_tier_label", "unknown"),
                            "quality": item.get("source_quality", "medium"),
                        }
                        for item in evidence
                    ]
                )
                if not quality_df.empty:
                    fig = px.histogram(quality_df, x="quality", color="type", title="Source quality mix")
                    st.plotly_chart(fig, use_container_width=True)

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
                        title="Evidence coverage",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                verification = memo.get("visualizations", {}).get("claim_verification_summary", {})
                verification_df = pd.DataFrame(
                    [
                        {"status": "Verified", "count": verification.get("verified_count", 0)},
                        {"status": "Partially Verified", "count": verification.get("partially_verified_count", 0)},
                        {"status": "Not Verified", "count": verification.get("not_verified_count", 0)},
                    ]
                )
                fig = px.bar(verification_df, x="count", y="status", orientation="h", title="Claim verification")
                st.plotly_chart(fig, use_container_width=True)

            source_order = [
                ("website", "Product"),
                ("leadership", "Team"),
                ("funding", "Funding"),
                ("market", "Market"),
                ("competitor", "Competition"),
                ("news", "Traction / News"),
                ("business_model", "Business Model"),
            ]
            for evidence_type, label in source_order:
                items = grouped.get(evidence_type, [])
                if not items:
                    continue
                with st.expander(f"{label} sources ({len(items)})", expanded=False):
                    st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)

            with st.expander("Claim verification", expanded=False):
                st.dataframe(pd.DataFrame(memo.get("claim_verification", [])), use_container_width=True, hide_index=True)
            with st.expander("Evidence coverage", expanded=False):
                st.dataframe(pd.DataFrame(memo.get("evidence_coverage", {}).get("rows", [])), use_container_width=True, hide_index=True)
            conflicts = memo.get("company_understanding", {}).get("product_evidence_conflicts", [])
            conflict_rows = [row for row in memo.get("claim_verification", []) if row.get("verification_status") == "Conflicting evidence found"]
            with st.expander("Conflicting evidence", expanded=bool(conflicts or conflict_rows)):
                if conflicts:
                    for item in conflicts:
                        st.warning(item)
                if conflict_rows:
                    st.dataframe(pd.DataFrame(conflict_rows), use_container_width=True, hide_index=True)
                if not conflicts and not conflict_rows:
                    st.info("No explicit conflicts detected.")
            with st.expander("Source reliability warnings", expanded=False):
                for warning in memo.get("evidence_dashboard", {}).get("warnings", [])[:8]:
                    st.warning(warning)
        else:
            st.info("No public evidence was available. The memo is based on partner notes only.")
            render_research_status(state, provider)
            if state["errors"]:
                st.warning("Research errors were captured during generation:")
                for error in state["errors"]:
                    st.code(error)
            else:
                st.warning("The research provider returned no usable evidence after normalization/filtering.")
            if provider == "mock":
                st.write("Switch the sidebar Research provider to `tavily` for live web enrichment.")
            elif provider == "tavily":
                st.write("If you are running inside Codex, the Streamlit process may not have network access.")
                st.write("Run the app from a normal terminal, or start the Streamlit process with network approval.")
                st.write("Also confirm `.streamlit/secrets.toml` contains `TAVILY_API_KEY`, then fully restart Streamlit.")

    with diligence_tab:
        memo = state["memo_json"]
        if px is None:
            st.error("Plotly is not installed. Run `pip install -r requirements.txt` to enable charts.")
        else:
            chart_data = state["chart_data"]
            visualizations = state["memo_json"].get("visualizations", {})

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

            blocking = memo.get("missing_data", {}).get("decision_blocking_unknowns", [])
            st.subheader("Decision-Blocking Unknowns")
            if blocking:
                st.dataframe(pd.DataFrame(blocking), use_container_width=True, hide_index=True)
            else:
                st.success("No decision-blocking unknowns detected.")

            st.subheader("Prioritized Questions")
            grouped_questions = grouped_diligence_questions(memo)
            for group, questions in grouped_questions.items():
                with st.expander(f"{group} ({len(questions)})", expanded=group in {"Commercial", "Product"}):
                    for question in questions[:5]:
                        st.markdown(f"- {question}")
                    if len(questions) > 5:
                        with st.expander("Show full question list", expanded=False):
                            for question in questions[5:]:
                                st.markdown(f"- {question}")

            with st.expander("Risk-specific questions", expanded=False):
                risk_questions = [
                    row.get("diligence_question")
                    for row in memo.get("visualizations", {}).get("risk_breakdown", {}).get("risks", [])
                    if row.get("diligence_question")
                ]
                for question in risk_questions:
                    st.markdown(f"- {question}")

    with trace_tab:
        memo = state["memo_json"]
        visualizations = memo.get("visualizations", {})
        scorecard = visualizations.get("investment_scorecard", {})
        factor_rows = []
        for row in scorecard.get("raw_scores", []):
            for factor in row.get("factors", []):
                factor_rows.append({"category": row.get("dimension"), **factor})
        with st.expander("Identity resolution", expanded=False):
            st.json(memo.get("identity_resolution", {}))
        with st.expander("Confidence calibration", expanded=False):
            st.json(memo.get("confidence_calibration", {}))
        with st.expander("Raw extracted fields", expanded=False):
            st.json(memo.get("company_understanding", {}))
        with st.expander("Factor evidence notes", expanded=False):
            st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)
        with st.expander("Full risk taxonomy", expanded=False):
            st.dataframe(pd.DataFrame(memo.get("risk_taxonomy", {}).get("rows", [])), use_container_width=True, hide_index=True)
        with st.expander("Full defensibility framework", expanded=False):
            st.dataframe(pd.DataFrame(memo.get("defensibility_framework", {}).get("rows", [])), use_container_width=True, hide_index=True)
        with st.expander("Full traction analysis", expanded=False):
            st.dataframe(pd.DataFrame(memo.get("traction_analysis", {}).get("rows", [])), use_container_width=True, hide_index=True)
        with st.expander("Full business model analysis", expanded=False):
            st.dataframe(pd.DataFrame(memo.get("business_model_analysis", {}).get("rows", [])), use_container_width=True, hide_index=True)
        with st.expander("Validator internals", expanded=False):
            st.json(memo.get("memo_validator", {}))
        with st.expander("Full analyst report", expanded=False):
            st.markdown(state["memo_markdown"])
        st.subheader("Prompt / Model Trace")
        st.json(state.get("trace", []))
        with st.expander("Full final state"):
            st.code(json.dumps(state, indent=2), language="json")
else:
    st.info("Paste partner notes or use the sample, then generate a memo.")
