from __future__ import annotations

from functools import partial
from typing import Any

from .nodes import (
    build_evidence_store,
    build_final_evidence_store,
    extract_company_profile,
    generate_chart_data,
    plan_context_research,
    plan_research,
    refine_company_profile_from_evidence,
    run_context_research,
    run_research,
    validate_company_understanding,
    validate_grounding,
    write_structured_memo,
)
from .render import render_outputs
from .state import DealMemoState


def initial_state(raw_notes: str) -> DealMemoState:
    return {
        "raw_notes": raw_notes,
        "company_profile": {},
        "research_plan": [],
        "search_results": [],
        "evidence": [],
        "memo_json": {},
        "chart_data": {},
        "validation": {"is_valid": False, "unsupported_claims": [], "warnings": [], "source_coverage": {}},
        "memo_markdown": "",
        "errors": [],
        "trace": [],
    }


def _merge(state: DealMemoState, update: dict[str, Any]) -> DealMemoState:
    merged = dict(state)
    merged.update(update)
    return merged  # type: ignore[return-value]


def run_sequential_graph(raw_notes: str, search_provider: str = "mock") -> DealMemoState:
    state = initial_state(raw_notes)
    steps = [
        extract_company_profile,
        plan_research,
        validate_company_understanding,
        partial(run_research, provider=search_provider),
        build_evidence_store,
        refine_company_profile_from_evidence,
        plan_context_research,
        partial(run_context_research, provider=search_provider),
        build_final_evidence_store,
        write_structured_memo,
        generate_chart_data,
        validate_grounding,
        render_outputs,
    ]
    for step in steps:
        state = _merge(state, step(state))
    return state


def build_langgraph(search_provider: str = "mock"):
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(DealMemoState)
    builder.add_node("extract_company_profile", extract_company_profile)
    builder.add_node("plan_research", plan_research)
    builder.add_node("validate_company_understanding", validate_company_understanding)
    builder.add_node("run_research", partial(run_research, provider=search_provider))
    builder.add_node("build_evidence_store", build_evidence_store)
    builder.add_node("build_final_evidence_store", build_final_evidence_store)
    builder.add_node("refine_company_profile_from_evidence", refine_company_profile_from_evidence)
    builder.add_node("plan_context_research", plan_context_research)
    builder.add_node("run_context_research", partial(run_context_research, provider=search_provider))
    builder.add_node("write_structured_memo", write_structured_memo)
    builder.add_node("generate_chart_data", generate_chart_data)
    builder.add_node("validate_grounding", validate_grounding)
    builder.add_node("render_outputs", render_outputs)

    builder.add_edge(START, "extract_company_profile")
    builder.add_edge("extract_company_profile", "plan_research")
    builder.add_edge("plan_research", "validate_company_understanding")
    builder.add_edge("validate_company_understanding", "run_research")
    builder.add_edge("run_research", "build_evidence_store")
    builder.add_edge("build_evidence_store", "refine_company_profile_from_evidence")
    builder.add_edge("refine_company_profile_from_evidence", "plan_context_research")
    builder.add_edge("plan_context_research", "run_context_research")
    builder.add_edge("run_context_research", "build_final_evidence_store")
    builder.add_edge("build_final_evidence_store", "write_structured_memo")
    builder.add_edge("write_structured_memo", "generate_chart_data")
    builder.add_edge("generate_chart_data", "validate_grounding")
    builder.add_edge("validate_grounding", "render_outputs")
    builder.add_edge("render_outputs", END)
    return builder.compile()


def run_deal_memo_graph(raw_notes: str, search_provider: str = "mock", prefer_langgraph: bool = True) -> DealMemoState:
    if prefer_langgraph:
        try:
            graph = build_langgraph(search_provider)
            return graph.invoke(initial_state(raw_notes))
        except Exception as exc:
            state = run_sequential_graph(raw_notes, search_provider)
            state["errors"].append(f"LangGraph runtime unavailable or failed, used sequential fallback: {exc}")
            return state
    return run_sequential_graph(raw_notes, search_provider)
