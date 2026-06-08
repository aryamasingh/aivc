from __future__ import annotations

from .state import DealMemoState, EvidenceItem, MemoClaim


SOURCE_LABELS = {
    "website": "Product / Website",
    "funding": "Funding",
    "news": "News",
    "leadership": "Leadership",
    "market": "Market",
    "competitor": "Competition",
    "business_model": "Business Model",
    "notes": "Partner Notes",
}


def citation_label(claim: MemoClaim) -> str:
    labels = list(claim.get("source_ids", []))
    if claim.get("note_reference"):
        labels.append("Notes")
    if claim.get("analyst_inference"):
        labels.append("Inference")
    return " ".join(f"[{label}]" for label in labels)


def _source_ids_label(source_ids: list[str]) -> str:
    return " ".join(f"[{source_id}]" for source_id in source_ids) if source_ids else "[No public source]"


def _escape_table(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _score_bar(score: int | float | None, width: int = 10) -> str:
    if not isinstance(score, (int, float)):
        return "Unknown"
    filled = max(0, min(width, round(score)))
    return f"{'█' * filled}{'░' * (width - filled)} {score}/10"


def _signed_bar(value: int | float, width: int = 10) -> str:
    magnitude = max(0, min(width, abs(round(value))))
    bar = "█" * magnitude
    return f"+{bar} {value}" if value >= 0 else f"-{bar} {value}"


def render_investment_scorecard_graph(memo: dict) -> list[str]:
    scorecard = memo.get("visualizations", {}).get("investment_scorecard", {})
    ratings = scorecard.get("ratings", [])
    if not ratings:
        return []
    overall = scorecard.get("overall_rating", "Unknown")
    lines = ["## Investment Scorecard", ""]
    lines.append(f"**Overall Rating:** {overall}")
    if scorecard.get("rating_basis"):
        lines.append(f"**Rating basis:** {scorecard['rating_basis']}")
    lines.append(f"**Top strengths:** {', '.join(scorecard.get('top_strengths', [])) or 'Unknown'}")
    lines.append(f"**Top concerns:** {', '.join(scorecard.get('top_concerns', [])) or 'Unknown'}")
    lines.extend(["", "| Category | Rating | Evidence Strength | Reason | Key Diligence Gap | Sources |"])
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in ratings:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table(row.get("category", "")),
                    _escape_table(row.get("rating", "Unknown")),
                    _escape_table(row.get("evidence_strength", "Low")),
                    _escape_table(row.get("reason", "")),
                    _escape_table(row.get("key_diligence_gap", "")),
                    _escape_table(_source_ids_label(row.get("source_ids", []))),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_claim_verification_graph(memo: dict) -> list[str]:
    summary = memo.get("visualizations", {}).get("claim_verification_summary", {})
    if not summary:
        return []
    rows = [
        ("Verified", summary.get("verified_count", 0)),
        ("Partially Verified", summary.get("partially_verified_count", 0)),
        ("Not Verified", summary.get("not_verified_count", 0)),
    ]
    max_count = max([count for _, count in rows] or [1]) or 1
    lines = ["## Partner Claim Verification Graph", "", "| Status | Count | Bar |", "| --- | ---: | --- |"]
    for status, count in rows:
        filled = round((count / max_count) * 10) if count else 0
        lines.append(f"| {status} | {count} | {'█' * filled}{'░' * (10 - filled)} |")
    lines.append("")
    return lines


def render_risk_breakdown_graph(memo: dict) -> list[str]:
    risks = memo.get("visualizations", {}).get("risk_breakdown", {}).get("risks", [])
    if not risks:
        return []
    lines = ["## Risk Breakdown Graph", "", "Higher score = higher risk.", "", "| Risk | Type | Score | Confidence | Why It Matters | Evidence |"]
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in risks:
        score = row.get("score")
        reason = row.get("reason", "")
        if score is None:
            reason = f"{reason} Risk marked Unknown because reliable evidence is missing."
        evidence = row.get("evidence", row.get("source_ids", []))
        evidence_label = " ".join(f"[{item}]" for item in evidence) if evidence else _source_ids_label(row.get("source_ids", []))
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table(row.get("risk", "")),
                    _escape_table(row.get("risk_type", "workflow_inferred")),
                    _escape_table(_score_bar(score)),
                    _escape_table(row.get("confidence", "")),
                    _escape_table(reason),
                    _escape_table(evidence_label),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_bull_bear_chart(memo: dict) -> list[str]:
    factors = memo.get("visualizations", {}).get("bull_bear_weights", {}).get("factors", [])
    if not factors:
        return []
    lines = ["## Bull Case vs Bear Case Chart", "", "| Factor | Weight | Reason | Sources |"]
    lines.append("| --- | ---: | --- | --- |")
    for row in factors:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table(row.get("factor", "")),
                    _escape_table(_signed_bar(row.get("value", 0))),
                    _escape_table(row.get("reason", "")),
                    _escape_table(_source_ids_label(row.get("source_ids", []))),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_company_understanding(memo: dict) -> list[str]:
    understanding = memo.get("company_understanding", {})
    if not understanding:
        return []
    identity = memo.get("identity_resolution", {})
    field_sources = understanding.get("field_sources", {})
    fields = [
        ("Company", understanding.get("company_name", "")),
        ("Category", understanding.get("category", "")),
        ("Subindustry", understanding.get("subindustry", "")),
        ("Primary Product", understanding.get("primary_product", "")),
        ("Primary Product Scope", understanding.get("primary_product_scope", "")),
        ("Product Scope Confidence", understanding.get("product_scope_confidence", "")),
        ("Product Use Cases", ", ".join(understanding.get("product_use_cases", [])) or "None identified"),
        ("Product Evidence Conflicts", "; ".join(understanding.get("product_evidence_conflicts", [])) or "None"),
        ("Target Buyer", understanding.get("target_buyer", "")),
        ("Budget Owner", understanding.get("budget_owner", "")),
        ("Industry", understanding.get("industry", "")),
        ("Business Model Hypothesis", understanding.get("business_model_hypothesis", understanding.get("business_model", ""))),
        ("Critical Dependencies", ", ".join(understanding.get("critical_dependencies", []))),
        ("Revenue Drivers", ", ".join(understanding.get("revenue_drivers", []))),
        ("Core Workflow", understanding.get("core_workflow", "")),
        ("Competitors Mentioned In Notes", ", ".join(understanding.get("competitors_mentioned_in_notes", [])) or "None"),
        ("Relevant Competitors", ", ".join(understanding.get("competitors", []))),
        ("Diligence Topics", ", ".join(understanding.get("diligence_topics", []))),
    ]
    validation = understanding.get("validation", {})
    lines = []
    contradiction = validation.get("contradictions", {})
    if contradiction.get("severity") == "high":
        lines.extend(["## Conflicting Signals Detected", ""])
        for idx, item in enumerate(contradiction.get("interpretations", []), start=1):
            lines.append(f"**Interpretation {idx}:** {item.get('interpretation', '')} ({item.get('evidence', '')})")
            lines.append("")
        lines.append(f"**Recommendation:** {contradiction.get('recommendation', 'Clarify positioning before workflow-level analysis.')}")
        lines.append("")
    if identity:
        selected = identity.get("selected_entity", {})
        lines.extend(
            [
                "## Company Identity Resolution",
                "",
                f"**Input Name:** {identity.get('input_company_name', 'Unknown')}",
                "",
                f"**Selected Entity:** {selected.get('name', 'Unknown')}",
                "",
                f"**Identity Confidence:** {identity.get('confidence', 0)}",
                "",
                f"**Status:** {identity.get('message', '')}",
                "",
                "| Candidate | Product Match | Workflow Match | Buyer Match | Founder Match | Entity Score |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in identity.get("candidates", [])[:6]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_table(row.get("name", "")),
                        _escape_table(row.get("product_match", 0)),
                        _escape_table(row.get("workflow_match", 0)),
                        _escape_table(row.get("buyer_match", 0)),
                        _escape_table(row.get("founder_match", 0)),
                        _escape_table(row.get("entity_score", 0)),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend([
        "## Company Understanding",
        "",
        f"**Understanding Confidence:** {validation.get('confidence', 'Unknown')}",
        "",
        "| Field | Value | Source | Confidence |",
        "| --- | --- | --- | --- |",
    ])
    for label, value in fields:
        key = label.lower().replace(" ", "_")
        source_meta = field_sources.get(key, field_sources.get(label.lower(), {}))
        lines.append(
            f"| {_escape_table(label)} | {_escape_table(value)} | "
            f"{_escape_table(source_meta.get('source', 'Notes / Inference'))} | {_escape_table(source_meta.get('confidence', 'Medium'))} |"
        )
    failures = validation.get("failures", [])
    warnings = validation.get("warnings", [])
    if failures or warnings:
        lines.extend(["", "**Understanding Validation Warnings**", ""])
        for item in failures + warnings:
            lines.append(f"- {item}")
    lines.append("")
    return lines


def render_evidence_dashboard(memo: dict) -> list[str]:
    dashboard = memo.get("evidence_dashboard", {})
    if not dashboard:
        return []
    counts = dashboard.get("tier_counts", {})
    lines = [
        "## Evidence Quality",
        "",
        f"**Tier 1 Sources:** {counts.get('tier_1', 0)}",
        f"**Tier 2 Sources:** {counts.get('tier_2', 0)}",
        f"**Tier 3 Sources:** {counts.get('tier_3', 0)}",
        "",
        "| Topic | Evidence Quality | Tier 1 | Tier 2 | Tier 3 | Weighted Score |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in dashboard.get("heatmap", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table(row.get("topic", "")),
                    _escape_table(row.get("quality", "")),
                    _escape_table(row.get("tier_1_sources", 0)),
                    _escape_table(row.get("tier_2_sources", 0)),
                    _escape_table(row.get("tier_3_sources", 0)),
                    _escape_table(row.get("weighted_score", 0)),
                ]
            )
            + " |"
        )
    lines.extend(["", f"**Strong Evidence:** {', '.join(dashboard.get('strong_evidence', [])) or 'None'}"])
    lines.append(f"**Weak Evidence:** {', '.join(dashboard.get('weak_evidence', [])) or 'None'}")
    warnings = dashboard.get("warnings", [])
    if warnings:
        lines.extend(["", "### Evidence Quality Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    lines.append("")
    return lines


def render_evidence_coverage(memo: dict) -> list[str]:
    coverage = memo.get("evidence_coverage", {})
    if not coverage:
        return []
    lines = [
        "## Evidence Coverage",
        "",
        "| Category | Coverage | Status | Confidence | Coverage Bar |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for row in coverage.get("rows", []):
        pct = int(row.get("coverage", 0))
        filled = round(pct / 10)
        bar = "█" * filled + "░" * (10 - filled)
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table(row.get("category", "")),
                    f"{pct}%",
                    _escape_table(row.get("status", "")),
                    _escape_table(row.get("confidence", "")),
                    f"{bar} {pct}%",
                ]
            )
            + " |"
        )
    if coverage.get("high_priority_gaps"):
        lines.extend(["", "**High Priority Gaps**", ""])
        for gap in coverage["high_priority_gaps"]:
            lines.append(f"- {gap}")
    if coverage.get("medium_priority_gaps"):
        lines.extend(["", "**Medium Priority Gaps**", ""])
        for gap in coverage["medium_priority_gaps"]:
            lines.append(f"- {gap}")
    focus = coverage.get("recommended_diligence_focus", [])
    if focus:
        lines.extend(["", "**Recommended Diligence Focus**", ""])
        for idx, item in enumerate(focus, start=1):
            lines.append(f"{idx}. {item['category']} ({item['coverage']}%) - {item['status']}")
        lines.append("")
        lines.append("These are currently the largest unknowns.")
    lines.append("")
    return lines


def render_confidence_calibration(memo: dict) -> list[str]:
    calibration = memo.get("confidence_calibration", {})
    if not calibration:
        return []
    return [
        "## Confidence Calibration",
        "",
        "| Metric | Score |",
        "| --- | ---: |",
        f"| Identity Confidence | {calibration.get('identity_confidence', 0)} |",
        f"| Understanding Confidence | {calibration.get('understanding_confidence', 0)} |",
        f"| Evidence Quality Score | {calibration.get('evidence_quality_score', 0)} |",
        f"| Overall Confidence Score | {calibration.get('score', 0)} |",
        "",
        calibration.get("rule", ""),
        "",
    ]


def render_missing_data(memo: dict) -> list[str]:
    missing = memo.get("missing_data", {})
    if not missing:
        return []
    lines = [
        "## Can We Make A Decision Yet?",
        "",
        f"**Decision Readiness:** {missing.get('decision_readiness', 0)}%",
        "",
        f"**Recommendation:** {missing.get('recommendation_adjustment', 'Needs diligence.')}",
        "",
    ]
    blocking = missing.get("decision_blocking_unknowns", [])
    if blocking:
        lines.extend(["## Decision Blocking Unknowns", ""])
        for gap in blocking:
            lines.append(f"- {gap['gap']} ({gap['category']}; priority score {gap['priority_score']})")
        lines.append("")

    lines.extend(["## Critical Missing Information", ""])
    for label, key in [("High Priority", "high_priority"), ("Medium Priority", "medium_priority"), ("Low Priority", "low_priority")]:
        gaps = missing.get(key, [])
        if gaps:
            lines.extend([f"**{label}**", ""])
            for gap in gaps[:8]:
                lines.append(
                    f"- {gap['gap']} ({gap['category']}; impact {gap['business_impact']}, decision {gap['decision_impact']}, evidence gap {gap['evidence_gap']})"
                )
            lines.append("")

    return lines


def render_claim_verification_layer(memo: dict) -> list[str]:
    rows = memo.get("claim_verification", [])
    if not rows:
        return []
    lines = [
        "## Claim Verification",
        "",
        "| Claim Area | Status | Claim | Sources | Diligence Follow-up |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table(row.get("claim_area", "")),
                    _escape_table(row.get("verification_status", "")),
                    _escape_table(row.get("claim", "")),
                    _escape_table(_source_ids_label(row.get("source_ids", []))),
                    _escape_table(row.get("diligence_follow_up", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.extend(["**Verification Notes**", ""])
    for row in rows:
        rationale = row.get("rationale", "")
        if rationale:
            lines.append(f"- **{row.get('claim_area', '')}:** {rationale}")
    lines.append("")
    return lines


def render_thesis_framework(memo: dict) -> list[str]:
    framework = memo.get("thesis_framework", {})
    if not framework:
        return []
    labels = [
        ("why_now", "Why Now?"),
        ("why_this_company", "Why This Company?"),
        ("why_this_market", "Why This Market?"),
        ("what_needs_to_be_true", "What Needs To Be True?"),
        ("what_could_kill_the_deal", "What Could Kill The Deal?"),
        ("next_diligence_step", "Next Diligence Step"),
    ]
    lines = [
        "## Investment Thesis Framework",
        "",
        "| Investor Question | Answer | Evidence Strength | Sources | Key Diligence Gap |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, label in labels:
        row = framework.get(key, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table(label),
                    _escape_table(row.get("answer", "")),
                    _escape_table(row.get("evidence_strength", "Low")),
                    _escape_table(_source_ids_label(row.get("source_ids", []))),
                    _escape_table(row.get("diligence_gap", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_defensibility_framework(memo: dict) -> list[str]:
    framework = memo.get("defensibility_framework", {})
    rows = [
        row
        for row in framework.get("rows", [])
        if row.get("source_ids") and str(row.get("status", "")).lower() not in {"unknown", "not found"}
    ]
    if not rows:
        return []
    architecture = framework.get("architecture_question", {})
    lines = [
        "## Defensibility Framework",
        "",
        f"**Core question:** {architecture.get('question', '')}",
        "",
        f"**Current hypothesis:** {architecture.get('current_hypothesis', 'Unknown')}",
        "",
        f"**Rationale:** {architecture.get('rationale', '')}",
        "",
        framework.get("rule", "Do not claim a moat unless evidence supports it."),
        "",
        "| Potential Moat | Current Evidence | Diligence Needed | Risk | Sources |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        evidence_text = f"{row.get('status', 'Unknown')}: {row.get('current_evidence', '')}"
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table(row.get("potential_moat", "")),
                    _escape_table(evidence_text),
                    _escape_table(row.get("diligence_needed", "")),
                    _escape_table(row.get("risk", "")),
                    _escape_table(_source_ids_label(row.get("source_ids", []))),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_competitive_landscape(memo: dict) -> list[str]:
    landscape = memo.get("competitive_landscape", {})
    rows = landscape.get("rows", [])
    if not rows:
        return []
    lines = [
        "## Competitive Landscape",
        "",
        landscape.get("summary", ""),
        "",
        "| Competitor Group | Examples | Why It Competes | Buyer Overlap | Workflow Overlap | Budget Overlap | Threat Level | Evidence Quality | Sources |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table(row.get("competitor_group", "")),
                    _escape_table(", ".join(row.get("examples", [])) or "Unknown"),
                    _escape_table(row.get("why_it_competes", "")),
                    _escape_table(row.get("buyer_overlap", "")),
                    _escape_table(row.get("workflow_overlap", "")),
                    _escape_table(row.get("budget_overlap", "")),
                    _escape_table(row.get("likely_threat_level", "")),
                    _escape_table(row.get("evidence_quality", "")),
                    _escape_table(_source_ids_label(row.get("source_ids", []))),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_traction_analysis(memo: dict) -> list[str]:
    analysis = memo.get("traction_analysis", {})
    rows = [
        row
        for row in analysis.get("rows", [])
        if row.get("source_ids") and str(row.get("status", "")).lower() != "unknown"
    ]
    if not rows:
        return []
    lines = ["## Traction Analysis", "", analysis.get("summary", ""), ""]
    for rule in analysis.get("rules", []):
        lines.append(f"- {rule}")
    lines.extend(["", "| Traction Type | Status | Evidence Strength | Interpretation | Diligence Needed | Sources |", "| --- | --- | --- | --- | --- | --- |"])
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table(row.get("traction_type", "")),
                    _escape_table(row.get("status", "")),
                    _escape_table(row.get("evidence_strength", "")),
                    _escape_table(row.get("interpretation", "")),
                    _escape_table(row.get("diligence_needed", "")),
                    _escape_table(_source_ids_label(row.get("source_ids", []))),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_business_model_analysis(memo: dict) -> list[str]:
    analysis = memo.get("business_model_analysis", {})
    rows = analysis.get("rows", [])
    if not rows:
        return []
    lines = [
        "## Business Model Analysis",
        "",
        analysis.get("summary", ""),
        "",
        "| Field | Value | Confidence | Rationale | Diligence Needed | Sources |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table(row.get("field", "")),
                    _escape_table(row.get("value", "")),
                    _escape_table(row.get("confidence", "")),
                    _escape_table(row.get("rationale", "")),
                    _escape_table(row.get("diligence_needed", "")),
                    _escape_table(_source_ids_label(row.get("source_ids", []))),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_risk_taxonomy(memo: dict) -> list[str]:
    taxonomy = memo.get("risk_taxonomy", {})
    rows = [row for row in taxonomy.get("rows", []) if row.get("source_ids")]
    if not rows:
        return []
    lines = [
        "## Risk Taxonomy",
        "",
        taxonomy.get("summary", ""),
        "",
        "| Risk Category | Description | Source Of Risk | Severity | Evidence Confidence | Diligence Question | Mitigation Hypothesis | Sources |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table(row.get("risk_category", "")),
                    _escape_table(row.get("description", "")),
                    _escape_table(row.get("source_of_risk", "")),
                    _escape_table(row.get("severity", "")),
                    _escape_table(row.get("evidence_confidence", "")),
                    _escape_table(row.get("diligence_question", "")),
                    _escape_table(row.get("mitigation_hypothesis", "")),
                    _escape_table(_source_ids_label(row.get("source_ids", []))),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_memo_markdown(state: DealMemoState) -> str:
    memo = state["memo_json"]
    evidence_by_id = {item["source_id"]: item for item in state["evidence"]}
    lines = [
        f"# {memo.get('company', 'Company')} Investment Memo",
        "",
        f"**Overall Confidence:** {memo.get('confidence', 'medium').title()}",
        "",
    ]

    for warning in state["validation"].get("warnings", []):
        lines.append(f"> {warning}")
    if state["validation"].get("warnings"):
        lines.append("")

    executive = memo.get("executive_summary", {})
    if executive:
        lines.extend(
            [
                "## Executive Summary",
                "",
                f"**Initial View:** {executive.get('view', 'Needs diligence')}",
                "",
                f"**Conviction:** {executive.get('conviction', 'Medium')}",
                "",
                "**Why It Matters**",
                "",
            ]
        )
        for claim in executive.get("why_it_matters", []):
            lines.append(f"- {claim['text']} {citation_label(claim)}")
        lines.extend(["", "**Key Open Risks**", ""])
        for risk in executive.get("key_open_risks", []):
            if isinstance(risk, dict):
                lines.append(f"- {risk['text']} {citation_label(risk)}")
            else:
                lines.append(f"- {risk}")
        lines.append("")

    lines.extend(render_investment_scorecard_graph(memo))
    lines.extend(render_thesis_framework(memo))
    lines.extend(render_defensibility_framework(memo))
    lines.extend(render_traction_analysis(memo))
    lines.extend(render_risk_taxonomy(memo))
    lines.extend(render_missing_data(memo))

    recommendation = memo.get("recommendation", {})
    if recommendation:
        rec_claim = {
            "text": recommendation.get("reason", ""),
            "source_ids": recommendation.get("source_ids", []),
            "note_reference": recommendation.get("note_reference", False),
            "analyst_inference": recommendation.get("analyst_inference", False),
        }
        lines.extend(
            [
                "## Recommendation",
                "",
                f"**Recommendation:** {recommendation.get('decision', 'Needs diligence')}",
                "",
                f"**Confidence:** {recommendation.get('confidence', 'Medium')}",
                "",
                f"**Rationale:** {rec_claim['text']} {citation_label(rec_claim)}",
                "",
            ]
        )

    if memo.get("bull_case"):
        lines.extend(["## Bull Case", ""])
        for claim in memo["bull_case"]:
            lines.append(f"- {claim['text']} {citation_label(claim)}")
        lines.append("")

    if memo.get("bear_case"):
        lines.extend(["## Bear Case", ""])
        for claim in memo["bear_case"]:
            lines.append(f"- {claim['text']} {citation_label(claim)}")
        lines.append("")

    verification = memo.get("partner_note_verification", [])
    if verification:
        lines.extend(["## Partner Claim Verification Table", ""])
        lines.append("| Partner Claim | Public Evidence Found | Status | Sources | Diligence Follow-up |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in verification:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_table(row.get("partner_claim", "")),
                        _escape_table(row.get("public_evidence_found", "")),
                        _escape_table(row.get("verification_status", "")),
                        _escape_table(_source_ids_label(row.get("source_ids", []))),
                        _escape_table(row.get("diligence_follow_up", "")),
                    ]
                )
                + " |"
            )
        lines.append("")

    if memo.get("what_needs_to_be_true"):
        lines.extend(["## What Needs To Be True", ""])
        for index, claim in enumerate(memo["what_needs_to_be_true"], start=1):
            lines.append(f"{index}. {claim['text']} {citation_label(claim)}")
        lines.append("")

    priorities = memo.get("next_diligence_priorities", {})
    if priorities:
        lines.extend(["## Diligence Checklist", ""])
        for category, items in priorities.items():
            lines.append(f"**{category}**")
            for item in items:
                if isinstance(item, dict):
                    question = item.get("question", "")
                    note_reference = item.get("source_note_reference", "")
                    suffix = f" [Notes: {_escape_table(note_reference)}]" if note_reference else " [Diligence]"
                    lines.append(f"- {question}{suffix}")
                else:
                    lines.append(f"- {item} [Diligence]")
            lines.append("")

    signals = memo.get("signals", {})
    if signals:
        lines.extend(["## Key Signals", "", "### Positive Signals", ""])
        for claim in signals.get("positive", []):
            lines.append(f"- {claim['text']} {citation_label(claim)}")
        lines.extend(["", "### Negative Signals", ""])
        for claim in signals.get("negative", []):
            lines.append(f"- {claim['text']} {citation_label(claim)}")
        lines.append("")

    attribution = memo.get("source_attribution", {})
    if attribution:
        lines.extend(["## Source Attribution", ""])
        lines.append("**Partner Notes**")
        for item in attribution.get("partner_notes", []):
            if isinstance(item, dict):
                lines.append(f"- {item['text']} {citation_label(item)}")
            else:
                lines.append(f"- {item} [Notes]")
        lines.extend(["", "**Public Research**"])
        for item in attribution.get("public_research", []):
            if isinstance(item, dict):
                lines.append(f"- {item['text']} {citation_label(item)}")
            else:
                lines.append(f"- {item} [Sources]")
        lines.extend(["", "**Combined Insight**"])
        for item in attribution.get("combined_insight", []):
            if isinstance(item, dict):
                lines.append(f"- {item['text']} {citation_label(item)}")
            else:
                lines.append(f"- {item} [Inference]")
        lines.append("")

    lines.extend(render_evidence_dashboard(memo))
    lines.extend(render_confidence_calibration(memo))
    lines.extend(render_evidence_coverage(memo))
    lines.extend(render_claim_verification_layer(memo))

    for section in memo.get("sections", []):
        lines.extend([f"## {section['title']}", ""])
        if section["title"] == "Competitive Landscape" and state["chart_data"].get("competitive_table"):
            lines.append("| Company | Similarity | Buyer | Workflow | Replacement Target | Differentiation Factors | Funding / Scale | Revenue Evidence | Score | Sources |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- |")
            for row in state["chart_data"]["competitive_table"]:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _escape_table(row.get("company", "")),
                            _escape_table(row.get("similarity_tier", "")),
                            _escape_table(row.get("buyer", "")),
                            _escape_table(row.get("workflow", "")),
                            _escape_table(row.get("replacement_target", "")),
                            _escape_table(row.get("differentiation_factors", "")),
                            _escape_table(row.get("funding_info", "")),
                            _escape_table(row.get("revenue_available", "")),
                            _escape_table(row.get("competitor_score", "")),
                            _escape_table(row.get("source_ids", "")),
                        ]
                    )
                    + " |"
                )
            lines.append("")
            continue

        for claim in section.get("claims", []):
            lines.append(f"- {claim['text']} {citation_label(claim)}")
        lines.append("")

    lines.extend([f"## Sources Used ({len(state['evidence'])})", ""])
    if state["evidence"]:
        grouped: dict[str, list[EvidenceItem]] = {}
        for item in state["evidence"]:
            grouped.setdefault(item["evidence_type"], []).append(item)
        for evidence_type, items in grouped.items():
            lines.append(f"**{SOURCE_LABELS.get(evidence_type, evidence_type.title())} ({len(items)})**")
            for item in items[:5]:
                quality = item.get("source_quality", "medium")
                tier = item.get("source_tier_label", "Tier 3: Supplemental")
                flag = " - LOW CONFIDENCE" if item.get("source_tier", 3) == 3 or quality == "low" or item["confidence"] < 0.5 else ""
                source_type = str(item.get("source_type", "unknown")).replace("_", " ")
                independence = item.get("independence_level", "unknown")
                lines.append(
                    f"- [{item['source_id']}] [{item['title']}]({item['url']}) "
                    f"({tier}; {quality}; {source_type}; {independence}; confidence {item['confidence']:.2f}){flag}"
                )
            if len(items) > 5:
                lines.append(f"- {len(items) - 5} additional sources in Research Evidence tab.")
            lines.append("")
    else:
        lines.append("- No public sources available.")

    unsupported = state["validation"].get("unsupported_claims", [])
    if unsupported:
        lines.extend(["", "## Unsupported Claims Removed / Needs Review", ""])
        for claim in unsupported:
            lines.append(f"- {claim}")

    missing_sources = [
        source_id
        for section in memo.get("sections", [])
        for claim in section.get("claims", [])
        for source_id in claim.get("source_ids", [])
        if source_id not in evidence_by_id
    ]
    if missing_sources:
        lines.extend(["", "## Citation Integrity Warning", ""])
        lines.append(f"- Unknown source IDs referenced: {', '.join(sorted(set(missing_sources)))}")

    return "\n".join(lines)


def render_sources_used(evidence: list[EvidenceItem], source_ids: list[str]) -> str:
    by_id = {item["source_id"]: item for item in evidence}
    lines = []
    for source_id in source_ids:
        item = by_id.get(source_id)
        if item:
            lines.append(f"- [{source_id}] [{item['title']}]({item['url']})")
    return "\n".join(lines) if lines else "- No cited public source. Treat as directional."


def render_outputs(state: DealMemoState) -> dict[str, str]:
    memo_markdown = render_memo_markdown(state)
    trace = state.get("trace", [])
    return {"memo_markdown": memo_markdown, "trace": [*trace, {"node": "render_outputs", "output": {"memo_chars": len(memo_markdown)}}]}
