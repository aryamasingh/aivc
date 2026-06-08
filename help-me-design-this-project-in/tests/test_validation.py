from src.graph import run_sequential_graph
from src.nodes import (
    build_evidence_store,
    generate_chart_data,
    extract_company_profile,
    plan_context_research,
    plan_research,
    refine_company_profile_from_evidence,
    _filter_target_company_aliases,
    validate_company_understanding,
    write_structured_memo,
)
from src.render import citation_label
from src.sample_data import SAMPLE_NOTES
from src.search import normalize_evidence


def test_required_memo_sections_are_present():
    state = run_sequential_graph(SAMPLE_NOTES, search_provider="mock")
    sections = {section["title"] for section in state["memo_json"]["sections"]}
    assert "Company Overview" in sections
    assert "Investment Thesis" in sections
    assert "Open Questions" in sections
    assert "bull_case" in state["memo_json"]
    assert "bear_case" in state["memo_json"]
    assert "what_needs_to_be_true" in state["memo_json"]
    assert "partner_note_verification" in state["memo_json"]
    assert "next_diligence_priorities" in state["memo_json"]
    assert "opportunity_scorecard" in state["memo_json"]
    assert "sources_grouped_by_category" in state["memo_json"]
    assert "visualizations" in state["memo_json"]
    assert "evidence_dashboard" in state["memo_json"]
    assert "evidence_coverage" in state["memo_json"]
    assert "missing_data" in state["memo_json"]
    assert "claim_verification" in state["memo_json"]
    assert "thesis_framework" in state["memo_json"]
    assert "defensibility_framework" in state["memo_json"]
    assert "competitive_landscape" in state["memo_json"]
    assert "traction_analysis" in state["memo_json"]
    assert "business_model_analysis" in state["memo_json"]
    assert "risk_taxonomy" in state["memo_json"]


def test_all_claims_have_grounding():
    state = run_sequential_graph(SAMPLE_NOTES, search_provider="mock")
    for section in state["memo_json"]["sections"]:
        for claim in section["claims"]:
            assert claim["source_ids"] or claim["note_reference"] or claim["analyst_inference"]


def test_evidence_normalization_deduplicates_urls():
    evidence = normalize_evidence(
        [
            {"title": "A", "url": "https://example.com/a", "snippet": "one", "evidence_type": "news"},
            {"title": "A again", "url": "https://example.com/a", "snippet": "two", "evidence_type": "news"},
        ]
    )
    assert len(evidence) == 1
    assert evidence[0]["source_id"] == "S1"
    assert evidence[0]["source_quality"] in {"high", "medium", "low"}
    assert evidence[0]["source_tier"] in {1, 2, 3}
    assert evidence[0]["source_weight"] in {1.0, 0.6, 0.2}


def test_low_quality_sources_are_filtered_when_better_sources_exist():
    evidence = normalize_evidence(
        [
            {"title": "Reuters article", "url": "https://reuters.com/a", "snippet": "one", "evidence_type": "news", "confidence": 0.9},
            {"title": "Best healthcare AI alternatives", "url": "https://randomblog.example/a", "snippet": "two", "evidence_type": "news", "confidence": 0.4},
        ]
    )
    assert len(evidence) == 1
    assert evidence[0]["source_quality"] == "high"


def test_listicles_can_never_be_tier_one_even_when_tagged_website():
    evidence = normalize_evidence(
        [
            {
                "title": "9 Best Ambient AI Medical Scribes 2026",
                "url": "https://example.com/best-ambient-ai-medical-scribes",
                "snippet": "A review-style list of ambient AI medical scribe vendors.",
                "evidence_type": "website",
                "confidence": 0.95,
            }
        ]
    )
    assert evidence[0]["source_tier"] == 3
    assert evidence[0]["source_weight"] == 0.2
    assert evidence[0]["source_quality"] == "low"


def test_official_company_website_can_be_tier_one():
    evidence = normalize_evidence(
        [
            {
                "title": "Abridge official company website",
                "url": "https://www.abridge.com/about",
                "snippet": "Abridge describes its ambient clinical documentation product.",
                "evidence_type": "website",
                "confidence": 0.85,
            }
        ]
    )
    assert evidence[0]["source_tier"] == 1
    assert evidence[0]["source_weight"] == 1.0


def test_source_reliability_policy_fields_are_attached():
    evidence = normalize_evidence(
        [
            {
                "title": "Acme official company website",
                "url": "https://www.acme.com/about",
                "snippet": "Acme describes its product positioning and stated customer logos.",
                "evidence_type": "website",
                "confidence": 0.9,
            }
        ]
    )
    item = evidence[0]
    assert item["source_type"] == "company_owned"
    assert item["independence_level"] == "company-owned"
    assert "product positioning" in item["allowed_uses"]
    assert "independent traction verification" in item["disallowed_uses"]


def test_funding_evidence_does_not_verify_traction_without_operating_metrics():
    from src.nodes import _partner_claims

    evidence = normalize_evidence(
        [
            {
                "title": "Acme raises Series B",
                "url": "https://crunchbase.com/organization/acme",
                "snippet": "Acme raised a Series B from institutional investors.",
                "evidence_type": "funding",
                "confidence": 0.9,
            }
        ]
    )
    claims = _partner_claims({"company": "Acme", "product": "AI platform", "customers": []}, evidence)
    traction = next(row for row in claims if row["partner_claim"] == "The company has meaningful commercial traction.")
    assert traction["verification_status"] == "Not Verified"
    assert traction["source_ids"] == []
    assert "Funding alone does not verify" in traction["public_evidence_found"]


def test_funding_evidence_can_support_traction_only_with_explicit_metrics():
    from src.nodes import _partner_claims

    evidence = normalize_evidence(
        [
            {
                "title": "Acme raises Series B with customer momentum",
                "url": "https://crunchbase.com/organization/acme",
                "snippet": "Acme disclosed 120 customers and $20M ARR alongside the financing.",
                "evidence_type": "funding",
                "confidence": 0.9,
            }
        ]
    )
    claims = _partner_claims({"company": "Acme", "product": "AI platform", "customers": []}, evidence)
    traction = next(row for row in claims if row["partner_claim"] == "The company has meaningful commercial traction.")
    assert traction["verification_status"] == "Partially Verified"
    assert traction["source_ids"] == ["S1"]


def test_vendor_blogs_support_competitor_discovery_not_market_conviction():
    from src.nodes import _coverage_quality, _source_ids_for

    evidence = normalize_evidence(
        [
            {
                "title": "Acme vs BetaSoft competitor comparison",
                "url": "https://vendor.example/acme-vs-betasoft",
                "snippet": "A vendor comparison page names BetaSoft and Gamma Systems as competitors.",
                "evidence_type": "competitor",
                "confidence": 0.8,
            }
        ]
    )
    assert evidence[0]["source_type"] == "vendor_blog"
    assert _source_ids_for(evidence, "competitor", use="competitor discovery") == ["S1"]
    assert _coverage_quality(evidence, "competitor", use="market conviction") == "Unknown"


def test_company_website_supports_positioning_not_independent_validation():
    from src.nodes import _source_allows_use

    evidence = normalize_evidence(
        [
            {
                "title": "Acme official company website",
                "url": "https://www.acme.com",
                "snippet": "Acme says it is the market leader with strong customer traction.",
                "evidence_type": "website",
                "confidence": 0.9,
            }
        ]
    )
    assert _source_allows_use(evidence[0], "product positioning")
    assert not _source_allows_use(evidence[0], "independent traction verification")
    assert not _source_allows_use(evidence[0], "market leadership")


def test_low_confidence_sources_cannot_raise_section_to_high_confidence():
    from src.nodes import _section_confidence

    evidence = normalize_evidence(
        [
            {
                "title": "Random AI market blog",
                "url": "https://randomblog.example/ai-market",
                "snippet": "Generic context about AI market growth.",
                "evidence_type": "market",
                "confidence": 0.3,
            }
            for _ in range(5)
        ]
    )
    assert _section_confidence(evidence, "market", has_notes=False) == "Low"


def test_partner_verification_and_scorecard_render_as_tables():
    state = run_sequential_graph(SAMPLE_NOTES, search_provider="mock")
    markdown = state["memo_markdown"]
    assert "## Evidence Quality" in markdown
    assert "| Topic | Evidence Quality | Tier 1 | Tier 2 | Tier 3 | Weighted Score |" in markdown
    assert "## Evidence Coverage" in markdown
    assert "| Category | Coverage | Status | Confidence | Coverage Bar |" in markdown
    assert "## Can We Make A Decision Yet?" in markdown
    assert "## Critical Missing Information" in markdown
    assert "## Decision Blocking Unknowns" in markdown or "**Blocking Unknowns**" in markdown
    assert "## Known / Unknown By Section" in markdown
    assert "## Claim Verification" in markdown
    assert "| Claim Area | Status | Claim | Sources | Diligence Follow-up |" in markdown
    assert "## Investment Scorecard" in markdown
    assert "| Category | Rating | Evidence Strength | Reason | Key Diligence Gap | Sources |" in markdown
    assert "## Investment Thesis Framework" in markdown
    assert "| Investor Question | Answer | Evidence Strength | Sources | Key Diligence Gap |" in markdown
    assert "## Defensibility Framework" in markdown
    assert "| Potential Moat | Current Evidence | Diligence Needed | Risk | Sources |" in markdown
    assert "## Competitive Landscape" in markdown
    assert "| Competitor Group | Examples | Why It Competes | Buyer Overlap | Workflow Overlap | Budget Overlap | Threat Level | Evidence Quality | Sources |" in markdown
    assert "## Traction Analysis" in markdown
    assert "## Business Model Analysis" in markdown
    assert "## Risk Taxonomy" in markdown
    assert "## Partner Claim Verification Graph" in markdown
    assert "## Partner Claim Verification Table" in markdown
    assert "| Partner Claim | Public Evidence Found | Status | Sources | Diligence Follow-up |" in markdown
    assert "## Risk Breakdown Graph" in markdown
    assert "| Risk | Type | Score | Confidence | Why It Matters | Evidence |" in markdown
    assert "## Bull Case vs Bear Case Chart" in markdown
    assert "## What Needs To Be True" in markdown
    assert "| Company | Buyer | Workflow | Threat Level | Competitor Score | Why It Competes | Sources |" in markdown


def test_investment_thesis_framework_answers_core_questions():
    state = run_sequential_graph(SAMPLE_NOTES, search_provider="mock")
    framework = state["memo_json"]["thesis_framework"]
    expected = {
        "why_now",
        "why_this_company",
        "why_this_market",
        "what_needs_to_be_true",
        "what_could_kill_the_deal",
        "next_diligence_step",
    }
    assert expected == set(framework)
    for key in expected:
        row = framework[key]
        assert row["answer"]
        assert row["evidence_strength"] in {"High", "Medium", "Low"}
        assert row["diligence_gap"]
    assert "paid deployments" in framework["what_needs_to_be_true"]["answer"]
    assert "pilots" in framework["what_could_kill_the_deal"]["answer"]
    assert "request evidence" in framework["next_diligence_step"]["answer"].lower()


def test_why_now_uses_industry_agnostic_drivers():
    notes = """
    Met CarbonFlow.
    Climate tech platform for carbon accounting and emissions reporting sold to sustainability teams.
    Customer urgency appears driven by regulatory reporting, cost pressure, workflow digitization, and data availability.
    Need to understand ROI, data quality, competition, and buyer urgency.
    """
    state = run_sequential_graph(notes, search_provider="mock")
    why_now = state["memo_json"]["thesis_framework"]["why_now"]["answer"].lower()
    assert "regulatory change" in why_now
    assert "cost pressure" in why_now
    assert "workflow digitization" in why_now
    assert "data availability" in why_now


def test_defensibility_framework_includes_all_default_moats_and_architecture_question():
    state = run_sequential_graph(SAMPLE_NOTES, search_provider="mock")
    framework = state["memo_json"]["defensibility_framework"]
    rows = {row["potential_moat"]: row for row in framework["rows"]}
    expected = {
        "proprietary data",
        "feedback loops",
        "workflow lock-in",
        "integrations",
        "switching costs",
        "network effects",
        "regulatory/compliance advantage",
        "distribution advantage",
        "brand/category leadership",
        "cost advantage",
        "technical differentiation",
        "supply-side exclusivity",
        "demand-side aggregation",
        "ecosystem partnerships",
    }
    assert expected == set(rows)
    assert framework["architecture_question"]["question"].startswith("Is this company a feature")
    for row in rows.values():
        assert row["status"] in {"Unknown", "Plausible", "Proven"}
        assert row["diligence_needed"]
        assert row["risk"]


def test_defensibility_framework_marks_missing_evidence_unknown():
    notes = "Met Atlas. AI platform helping enterprises make better decisions. Need to understand product, buyer, competition, and defensibility."
    state = run_sequential_graph(notes, search_provider="mock")
    rows = state["memo_json"]["defensibility_framework"]["rows"]
    assert any(row["status"] == "Unknown" for row in rows)
    proprietary = next(row for row in rows if row["potential_moat"] == "proprietary data")
    assert proprietary["status"] == "Unknown"
    assert "does not prove this moat" in proprietary["current_evidence"]


def test_defensibility_framework_distinguishes_plausible_from_proven_moat():
    notes = "Met DataForge. AI platform for finance teams. Need to understand proprietary data, integrations, defensibility, and competition."
    state = {
        "raw_notes": notes,
        "company_profile": {},
        "research_plan": [],
        "search_results": [
            {
                "title": "DataForge official company website",
                "url": "https://dataforge.example/about",
                "snippet": "DataForge says it uses proprietary data and deep ERP integrations for finance workflows.",
                "query": "DataForge official website product proprietary data integrations",
                "evidence_type": "website",
                "confidence": 0.9,
            }
        ],
        "evidence": [],
        "memo_json": {},
        "chart_data": {},
        "validation": {"is_valid": False, "unsupported_claims": [], "warnings": [], "source_coverage": {}},
        "memo_markdown": "",
        "errors": [],
        "trace": [],
    }
    for step in [extract_company_profile, build_evidence_store, refine_company_profile_from_evidence, write_structured_memo]:
        state.update(step(state))
    rows = {row["potential_moat"]: row for row in state["memo_json"]["defensibility_framework"]["rows"]}
    assert rows["proprietary data"]["status"] == "Plausible"
    assert rows["integrations"]["status"] == "Plausible"
    assert rows["proprietary data"]["status"] != "Proven"


def test_competitive_landscape_explains_market_structure():
    notes = """
    Met Ramp team. Company provides corporate cards, expense management, bill pay, and procurement software
    for CFOs and finance teams. Need to understand competition from Brex, Airbase, Navan, Amex, and Coupa.
    """
    state = run_sequential_graph(notes, search_provider="mock")
    landscape = state["memo_json"]["competitive_landscape"]
    groups = {row["competitor_group"]: row for row in landscape["rows"]}
    assert "direct startups" in groups
    assert "incumbent vendors" in groups
    assert "internal build / manual process" in groups
    assert "system-of-record vendors" in groups
    assert "manual" in " ".join(groups["internal build / manual process"]["examples"]).lower()
    for row in groups.values():
        assert row["why_it_competes"]
        assert row["buyer_overlap"] in {"High", "Medium", "Low", "Unknown"}
        assert row["workflow_overlap"] in {"High", "Medium", "Low", "Unknown"}
        assert row["budget_overlap"] in {"High", "Medium", "Low", "Unknown"}
        assert row["likely_threat_level"]
        assert row["evidence_quality"]


def test_traction_analysis_distinguishes_traction_types_and_rules():
    notes = """
    Met Harvey.
    AI platform for legal professionals.
    Strong law firm adoption and several pilots.
    Need to understand paid usage, retention, expansion, and ARR.
    """
    state = run_sequential_graph(notes, search_provider="mock")
    analysis = state["memo_json"]["traction_analysis"]
    rows = {row["traction_type"]: row for row in analysis["rows"]}
    assert "Funding is not traction." in analysis["rules"]
    assert "Customer logos are not usage." in analysis["rules"]
    assert "Pilots are not retention." in analysis["rules"]
    assert rows["pilots"]["status"] in {"Partner notes only", "Partially verified"}
    assert rows["retention"]["status"] in {"Partner notes only", "Unknown", "Partially verified"}
    assert "investment-grade validation requires ARR" in analysis["summary"]


def test_business_model_analysis_marks_unknowns_without_evidence():
    notes = "Met Atlas. AI platform helping enterprises make better decisions. Need to understand revenue model, buyer, and why customers buy."
    state = run_sequential_graph(notes, search_provider="mock")
    rows = {row["field"]: row for row in state["memo_json"]["business_model_analysis"]["rows"]}
    assert rows["pricing model"]["value"] == "Unknown"
    assert rows["ACV range"]["value"] == "Unknown"
    assert rows["gross margin drivers"]["value"] == "Unknown"
    assert rows["customer concentration risk"]["value"] == "Unknown"
    assert rows["buyer"]["value"] in {"Enterprise customers", "Unknown"}


def test_risk_taxonomy_contains_default_categories():
    state = run_sequential_graph(SAMPLE_NOTES, search_provider="mock")
    rows = {row["risk_category"]: row for row in state["memo_json"]["risk_taxonomy"]["rows"]}
    expected = {
        "product risk",
        "technical risk",
        "adoption risk",
        "market timing risk",
        "competition risk",
        "pricing / willingness-to-pay risk",
        "gross margin risk",
        "implementation risk",
        "retention risk",
        "regulatory / compliance risk",
        "data/privacy/security risk",
        "concentration risk",
        "platform dependency risk",
        "financing risk",
        "team/execution risk",
    }
    assert expected == set(rows)
    for row in rows.values():
        assert row["description"]
        assert row["source_of_risk"]
        assert row["severity"] in {"High", "Medium", "Low", "Unknown"}
        assert row["evidence_confidence"] in {"High", "Medium", "Low"}
        assert row["diligence_question"]
        assert row["mitigation_hypothesis"]


def test_partner_memo_view_starts_with_executive_summary_before_diagnostics():
    state = run_sequential_graph(SAMPLE_NOTES, search_provider="mock")
    markdown = state["memo_markdown"]
    executive_idx = markdown.index("## Executive Summary")
    assert "## Company Understanding" not in markdown[:executive_idx]
    assert "## Evidence Quality" not in markdown[:executive_idx]
    assert "## Trace" not in markdown[:executive_idx]


def test_post_generation_validator_flags_flawed_memo_high_severity():
    from src.nodes import _post_generation_memo_validator

    flawed_memo = {
        "company": "Acme AI",
        "traceability": {"unsupported_count": 2},
        "claim_verification": [{"verification_status": "Verified by public evidence", "source_ids": ["S1"]}],
        "competitive_landscape": {"rows": [{"examples": ["Acme.ai"]}]},
        "traction_analysis": {"rules": []},
        "thesis_framework": {"why_now": {}, "why_this_company": {}, "what_could_kill_the_deal": {}},
        "next_diligence_priorities": {},
        "missing_data": {"decision_blocking_unknowns": []},
    }
    evidence = normalize_evidence(
        [
            {
                "title": "Random blog",
                "url": "https://randomblog.example/acme",
                "snippet": "Acme is the market leader with proven traction.",
                "evidence_type": "market",
                "confidence": 0.3,
            }
        ]
    )
    result = _post_generation_memo_validator(flawed_memo, evidence, "Met Acme.")
    assert not result["passed"]
    assert result["severity"] == "high"
    assert any("Unsupported claims" in item["warning"] for item in result["warnings"])
    assert any("Target company appears" in item["warning"] for item in result["warnings"])


def test_claim_verification_layer_covers_key_investment_claims():
    state = run_sequential_graph(SAMPLE_NOTES, search_provider="mock")
    rows = {row["claim_area"]: row for row in state["memo_json"]["claim_verification"]}
    expected = {
        "Company Identity",
        "Product Description",
        "Target Customer",
        "Market Size",
        "Traction",
        "Customers",
        "Funding",
        "Business Model",
        "Competition",
        "Team",
        "Defensibility",
        "Risks",
        "Investment Thesis",
    }
    assert expected.issubset(rows)
    allowed_statuses = {
        "Verified by public evidence",
        "Supported by partner notes only",
        "Partially verified",
        "Inferred",
        "Unsupported / requires diligence",
        "Conflicting evidence found",
    }
    assert {row["verification_status"] for row in rows.values()}.issubset(allowed_statuses)


def test_claim_verification_keeps_customer_adoption_cautious_without_operating_metrics():
    notes = """
    Met Harvey.
    AI platform for legal professionals.
    Strong law firm adoption.
    Need to understand accuracy, expansion beyond law firms, defensibility, and competition.
    """
    state = run_sequential_graph(notes, search_provider="mock")
    rows = {row["claim_area"]: row for row in state["memo_json"]["claim_verification"]}
    traction = rows["Traction"]
    assert traction["verification_status"] == "Unsupported / requires diligence"
    assert "paid usage" in traction["claim"]
    assert "retention" in traction["claim"]
    assert "Funding" not in traction["rationale"]
    assert "pilots, paid deployments, active usage, expansion, and renewal behavior" in state["memo_markdown"]


def test_claim_verification_flags_conflicting_product_evidence():
    notes = (
        "Met Lexora. AI platform for legal professionals. "
        "End-to-end solution for law firms. "
        "Need to understand accuracy, expansion, defensibility, and competition."
    )
    state = {
        "raw_notes": notes,
        "company_profile": {},
        "research_plan": [],
        "search_results": [
            {
                "title": "Lexora legal AI platform",
                "url": "https://lexora.example/product",
                "snippet": "Lexora supports contract review, legal research, and drafting workflows for legal teams.",
                "query": "Lexora official website product",
                "evidence_type": "website",
                "confidence": 0.85,
            }
        ],
        "evidence": [],
        "memo_json": {},
        "chart_data": {},
        "validation": {"is_valid": False, "unsupported_claims": [], "warnings": [], "source_coverage": {}},
        "memo_markdown": "",
        "errors": [],
        "trace": [],
    }
    for step in [extract_company_profile, build_evidence_store, refine_company_profile_from_evidence, write_structured_memo]:
        state.update(step(state))
    rows = {row["claim_area"]: row for row in state["memo_json"]["claim_verification"]}
    assert rows["Product Description"]["verification_status"] == "Conflicting evidence found"


def test_citation_label_includes_all_grounding_types():
    label = citation_label({"text": "claim", "source_ids": ["S1"], "note_reference": True, "analyst_inference": True})
    assert "[S1]" in label
    assert "[Notes]" in label
    assert "[Inference]" in label


def test_charts_have_directional_source_data_or_empty_state():
    state = run_sequential_graph(SAMPLE_NOTES, search_provider="mock")
    assert "investment_scorecard" in state["chart_data"]
    assert "claim_verification_summary" in state["chart_data"]
    assert "risk_breakdown" in state["chart_data"]
    assert "bull_bear_weights" in state["chart_data"]
    assert "source_coverage" not in state["chart_data"]
    assert "timeline" not in state["chart_data"]
    assert state["validation"]["is_valid"]


def test_visualization_ratings_include_unknown_guardrails():
    state = run_sequential_graph(SAMPLE_NOTES, search_provider="mock")
    scorecard = state["memo_json"]["visualizations"]["investment_scorecard"]
    assert scorecard["overall_rating"] in {"Strong positive", "Positive", "Mixed", "Weak", "Unknown", "High risk"}
    assert scorecard["top_strengths"]
    assert scorecard["top_concerns"]
    assert any(row["rating"] == "Unknown" for row in scorecard["ratings"])
    assert any(row["category"] == "Traction" for row in scorecard["ratings"])
    for row in scorecard["ratings"]:
        assert row["rating"] in {"Strong positive", "Positive", "Mixed", "Weak", "Unknown", "High risk"}
        assert row["evidence_strength"] in {"High", "Medium", "Low"}
        assert row["reason"]
        assert row["key_diligence_gap"]
    for row in scorecard["raw_scores"]:
        assert len(row["factors"]) >= 3
        known_factor_scores = [factor["score"] for factor in row["factors"] if factor["score"] is not None]
        if row["score"] is not None:
            assert row["score"] == round(sum(known_factor_scores) / len(known_factor_scores), 1)


def test_scorecard_factor_breakdown_is_rendered():
    state = run_sequential_graph(SAMPLE_NOTES, search_provider="mock")
    markdown = state["memo_markdown"]
    assert "**Factor Evidence Notes**" in markdown
    assert "| Category | Factor | Evidence Signal | Reason | Sources |" in markdown
    assert "Investment Score:" not in markdown
    assert "Factor Score" not in markdown


def test_confidence_never_exceeds_evidence_quality_for_inference_heavy_sections():
    state = run_sequential_graph(SAMPLE_NOTES, search_provider="mock")
    confidence = state["memo_json"]["section_confidence"]
    assert confidence["Business Model"] != "High"
    assert confidence["Competitive Landscape"] != "High"
    assert confidence["Investment Thesis"] != "High"
    assert "Business model remains a diligence gap" in state["memo_markdown"]


def test_tier_three_sources_do_not_drive_high_confidence():
    evidence = normalize_evidence(
        [
            {"title": f"LinkedIn post {idx}", "url": f"https://linkedin.com/posts/{idx}", "snippet": "social signal", "evidence_type": "business_model", "confidence": 0.8}
            for idx in range(5)
        ]
    )
    assert all(item["source_tier"] == 3 for item in evidence)
    notes = "Met Ramp team. Company provides corporate cards and spend management software for CFOs."
    state = run_sequential_graph(notes, search_provider="mock")
    state["evidence"] = evidence
    from src.nodes import _evidence_dashboard, _business_model_confidence

    dashboard = _evidence_dashboard(evidence)
    business_model_row = next(row for row in dashboard["heatmap"] if row["topic"] == "Business Model")
    assert business_model_row["quality"] in {"Weak", "Unknown"}
    assert _business_model_confidence(evidence) == "Low"


def test_evidence_coverage_is_separate_from_confidence_and_prioritizes_gaps():
    state = run_sequential_graph(SAMPLE_NOTES, search_provider="mock")
    coverage = state["memo_json"]["evidence_coverage"]
    rows = {row["category"]: row for row in coverage["rows"]}
    assert rows["Product"]["coverage"] >= 0
    assert rows["Product"]["confidence"] in {"Low", "Medium", "High"}
    assert "high_priority_gaps" in coverage
    assert "recommended_diligence_focus" in coverage
    assert coverage["recommended_diligence_focus"][0]["coverage"] <= coverage["recommended_diligence_focus"][-1]["coverage"]


def test_overall_confidence_is_bounded_by_identity_understanding_and_evidence_quality():
    notes = "Met Nimbus. Product and buyer are unclear. Funding unknown. Business model unknown."
    state = run_sequential_graph(notes, search_provider="mock")
    calibration = state["memo_json"]["confidence_calibration"]
    assert state["memo_json"]["confidence"] == "low"
    assert calibration["score"] == min(
        calibration["identity_confidence"],
        calibration["understanding_confidence"],
        calibration["evidence_quality_score"],
    )
    assert calibration["identity_confidence"] < 0.6
    assert state["memo_json"]["confidence"] != "high"
    assert "## Confidence Calibration" in state["memo_markdown"]


def test_generic_ai_platform_notes_have_low_understanding_confidence():
    notes = """
    Met Atlas.

    Everyone seems excited.

    AI platform helping enterprises make better decisions.

    Customers include several Fortune 500 companies.

    Founder previously sold a startup.

    Need to understand:
    - Product
    - Buyer
    - Competition
    - Revenue model
    - Why customers buy
    """
    state = run_sequential_graph(notes, search_provider="mock")
    understanding = state["memo_json"]["company_understanding"]
    assert understanding["validation"]["confidence"] <= 0.4
    assert understanding["category"] == "Unknown"
    assert understanding["primary_product"] == "Unknown"
    assert understanding["target_buyer"] == "Enterprise customers"
    assert understanding["budget_owner"] == "Unknown"
    assert understanding["business_model"] == "Unknown"
    assert "SMB" not in understanding["target_buyer"]
    assert "startup" not in understanding["target_buyer"].lower()
    extracted_fields = [
        understanding["primary_product"],
        understanding["target_buyer"],
        understanding["budget_owner"],
        understanding["business_model"],
        understanding["core_workflow"],
    ]
    assert not any("requires diligence" in str(field).lower() for field in extracted_fields)
    market_rating = next(
        row for row in state["memo_json"]["visualizations"]["investment_scorecard"]["ratings"] if row["category"] == "Market"
    )
    assert market_rating["rating"] == "Unknown"
    assert state["memo_json"]["confidence"] == "low"


def test_very_low_understanding_uses_insufficient_information_view():
    notes = """
    Met Atlas.
    Everyone seems excited.
    AI platform helping enterprises make better decisions.
    Need to understand product, buyer, competition, revenue model, and why customers buy.
    """
    state = run_sequential_graph(notes, search_provider="mock")
    executive = state["memo_json"]["executive_summary"]
    assert state["memo_json"]["company_understanding"]["validation"]["confidence"] < 0.2
    assert executive["view"] == "Insufficient Information"
    assert executive["conviction"] == "Low"
    assert "Positive" not in executive["view"]
    assert "Negative" not in executive["view"]
    assert "Bullish" not in executive["view"]
    assert "Bearish" not in executive["view"]


def test_diligence_questions_are_grounded_to_current_notes_without_template_leakage():
    contradiction_notes = """
    Met Orion.
    Founder describes company as cybersecurity.
    Website describes workflow automation.
    Customers describe it as compliance software.
    Need to understand actual product, buyer, and competitive landscape.
    """
    run_sequential_graph(contradiction_notes, search_provider="mock")

    nova_notes = """
    Met Nova.
    AI platform helping enterprises make better decisions.
    Need to understand:
    - Product
    - Buyer
    - Competition
    - Revenue model
    - Why customers buy
    """
    state = run_sequential_graph(nova_notes, search_provider="mock")
    markdown = state["memo_markdown"].lower()
    assert "cybersecurity, workflow automation, compliance software" not in markdown
    assert "separate cybersecurity, compliance, and workflow-automation" not in markdown
    for items in state["memo_json"]["next_diligence_priorities"].values():
        for item in items:
            assert isinstance(item, dict)
            assert item.get("source_note_reference")


def test_traceability_check_removes_unsupported_statements_and_keeps_note_supported_facts():
    notes = """
    Met Atlas.
    Customers include several Fortune 500 companies.
    Need to understand:
    - Product
    - Buyer
    - Competition
    - Revenue model
    """
    state = run_sequential_graph(notes, search_provider="mock")
    traceability = state["memo_json"]["traceability"]
    assert traceability["records"]
    assert any(
        record["statement"] == "Target buyer = Enterprise customers"
        and record["evidence"] == "Customers include several Fortune 500 companies."
        and record["status"] == "Supported"
        for record in traceability["records"]
    )
    assert not any("room to expand into adjacent" in claim["text"].lower() for claim in state["memo_json"]["bull_case"])
    assert state["validation"].get("removed_unsupported_claims")


def test_harvey_legal_platform_notes_generate_mock_sources():
    notes = """
    Met Harvey.

    AI platform for legal professionals.

    Strong law firm adoption.

    Interesting because legal workflows appear highly automatable.

    Need to understand:
    - Accuracy
    - Expansion beyond law firms
    - Defensibility
    - Competition
    """
    state = run_sequential_graph(notes, search_provider="mock")
    understanding = state["memo_json"]["company_understanding"]
    assert understanding["category"] == "LegalTech"
    assert understanding["primary_product"] == "AI legal workflow platform for legal professionals"
    assert "law firms and legal professionals" in understanding["target_buyer"]
    assert state["research_plan"]
    assert state["evidence"]
    assert any("Harvey" in item["title"] or "Legal AI" in item["title"] for item in state["evidence"])


def test_target_company_aliases_are_excluded_from_competitors():
    target_company = "Acme AI"
    competitor_candidates = ["Acme AI", "Acme.ai", "Acme Technologies", "BetaSoft", "Gamma Systems"]
    assert _filter_target_company_aliases(competitor_candidates, target_company) == ["BetaSoft", "Gamma Systems"]

    notes = """
    Met Harvey.
    AI platform for legal professionals.
    Strong law firm adoption.
    Need to understand accuracy, defensibility, and competition.
    """
    state = run_sequential_graph(notes, search_provider="mock")
    understanding = state["memo_json"]["company_understanding"]
    table_companies = [row["company"] for row in state["chart_data"]["competitive_table"]]
    diligence_text = " ".join(understanding["diligence_topics"])
    assert "Harvey" not in understanding["competitors"]
    assert "Harvey" not in table_companies
    assert "Harvey" not in diligence_text
    assert any(name in understanding["competitors"] for name in ["Legora", "Spellbook", "Ironclad"])


def test_broad_platform_notes_are_not_over_narrowed_by_public_use_cases():
    notes = (
        "Met Lexora. AI platform for legal professionals. "
        "End-to-end solution for law firms. "
        "Need to understand accuracy, expansion, defensibility, and competition."
    )
    state = {
        "raw_notes": notes,
        "company_profile": {},
        "research_plan": [],
        "search_results": [
            {
                "title": "Lexora legal AI platform",
                "url": "https://lexora.example/product",
                "snippet": "Lexora supports contract review, legal research, and drafting workflows for legal teams.",
                "query": "Lexora official website product",
                "evidence_type": "website",
                "confidence": 0.85,
            }
        ],
        "evidence": [],
        "memo_json": {},
        "chart_data": {},
        "validation": {"is_valid": False, "unsupported_claims": [], "warnings": [], "source_coverage": {}},
        "memo_markdown": "",
        "errors": [],
        "trace": [],
    }
    for step in [extract_company_profile, build_evidence_store, refine_company_profile_from_evidence, write_structured_memo]:
        state.update(step(state))

    understanding = state["memo_json"]["company_understanding"]
    assert understanding["primary_product"] == "AI legal workflow platform for legal professionals"
    assert understanding["primary_product_scope"] == "broad_platform"
    assert understanding["product_scope_confidence"] == "high"
    assert {"contract review", "legal research", "legal drafting"}.issubset(set(understanding["product_use_cases"]))
    assert understanding["product_evidence_conflicts"]
    assert "AI contract review, redlining, clause extraction, and risk detection" not in understanding["primary_product"]


def test_partner_notes_anchor_product_when_public_evidence_conflicts():
    notes = (
        "Met Orion. Founder describes company as cybersecurity software for security teams. "
        "Need to understand actual product, buyer, and competitive landscape."
    )
    state = {
        "raw_notes": notes,
        "company_profile": {},
        "research_plan": [],
        "search_results": [
            {
                "title": "Orion workflow automation blog",
                "url": "https://randomblog.example/orion-workflow",
                "snippet": "Orion is described as workflow automation for generic operations teams.",
                "query": "Orion product",
                "evidence_type": "website",
                "confidence": 0.5,
            }
        ],
        "evidence": [],
        "memo_json": {},
        "chart_data": {},
        "validation": {"is_valid": False, "unsupported_claims": [], "warnings": [], "source_coverage": {}},
        "memo_markdown": "",
        "errors": [],
        "trace": [],
    }
    for step in [extract_company_profile, build_evidence_store, refine_company_profile_from_evidence, write_structured_memo]:
        state.update(step(state))
    understanding = state["memo_json"]["company_understanding"]
    assert understanding["note_anchor"]["target_company"] == "Orion"
    assert "cybersecurity" in understanding["note_anchor"]["product_description"].lower()
    assert "cybersecurity" in understanding["primary_product"].lower()
    assert "workflow automation for generic operations" not in understanding["primary_product"].lower()
    assert state["company_profile"]["product_source_basis"].startswith("partner notes")


def test_non_healthcare_note_anchor_prevents_healthcare_template_leakage():
    notes = """
    Met Orion.
    Founder describes company as cybersecurity.
    Website describes workflow automation.
    Customers describe it as compliance software.
    Need to understand:
    - Actual product
    - Actual buyer
    - Competitive landscape
    """
    state = run_sequential_graph(notes, search_provider="mock")
    markdown = state["memo_markdown"].lower()
    forbidden = ["ambient clinical documentation", "ai scribing", "health systems and clinicians", "cmio", "epic", "hipaa", "phi", "baa", "clinical risk"]
    for term in forbidden:
        assert term not in markdown
    understanding = state["memo_json"]["company_understanding"]
    assert understanding["note_anchor"]["target_company"] == "Orion"
    assert understanding["validation"]["contradictions"]["severity"] == "high"


def test_broad_platform_memo_explains_use_cases_and_diligence_needed():
    notes = (
        "Met Lexora. AI platform for legal professionals. "
        "Platform used by law firms as an end-to-end solution. "
        "Need to understand accuracy, expansion, defensibility, and competition."
    )
    state = {
        "raw_notes": notes,
        "company_profile": {},
        "research_plan": [],
        "search_results": [
            {
                "title": "Lexora legal AI platform",
                "url": "https://lexora.example/product",
                "snippet": "Lexora supports contract review, legal research, and drafting workflows for legal teams.",
                "query": "Lexora official website product",
                "evidence_type": "website",
                "confidence": 0.85,
            }
        ],
        "evidence": [],
        "memo_json": {},
        "chart_data": {},
        "validation": {"is_valid": False, "unsupported_claims": [], "warnings": [], "source_coverage": {}},
        "memo_markdown": "",
        "errors": [],
        "trace": [],
    }
    for step in [
        extract_company_profile,
        build_evidence_store,
        refine_company_profile_from_evidence,
        write_structured_memo,
        generate_chart_data,
    ]:
        state.update(step(state))
    from src.nodes import validate_grounding
    from src.render import render_outputs

    state.update(validate_grounding(state))
    state.update(render_outputs(state))
    markdown = state["memo_markdown"].lower()
    assert "appears to provide a broad platform" in markdown
    assert "public evidence suggests use cases including" in markdown
    assert "exact initial wedge and usage mix require diligence" in markdown
    assert "| primary product scope | broad_platform |" in markdown


def test_new_industry_taxonomies_generate_specific_profiles_and_risks():
    cases = [
        (
            "ClimateTech",
            "Met CarbonFlow. Climate tech platform for carbon accounting and emissions reporting sold to sustainability teams. Need to understand data quality, regulatory reporting, competition, and ROI.",
            "Emissions data quality risk",
        ),
        (
            "HRTech",
            "Met PeoplePilot. HRTech software for talent acquisition and employee engagement sold to CHROs and people ops. Need to understand bias compliance, HRIS integration, retention, and competition.",
            "Employee data privacy risk",
        ),
        (
            "Supply Chain",
            "Met FlowChain. Supply chain software for logistics, inventory planning, warehouse operations, and suppliers. Need to understand data integration, forecast accuracy, implementation, and competition.",
            "Data integration risk",
        ),
        (
            "Robotics",
            "Met RoboLift. Robotics platform for warehouse automation and manufacturing operations. Need to understand hardware reliability, deployment complexity, unit economics, and competition.",
            "Hardware reliability risk",
        ),
        (
            "Biotech",
            "Met GeneForge. Biotech drug discovery platform for pharma R&D teams. Need to understand scientific validation, IP, clinical regulatory path, financing, and partnerships.",
            "Scientific validation risk",
        ),
        (
            "Defense Technology",
            "Met Sentinel. Defense technology platform using drones and sensors for military missions. Need to understand program concentration, procurement, field reliability, export controls, and prime competition.",
            "Defense procurement risk",
        ),
    ]
    for expected_category, notes, expected_risk in cases:
        state = run_sequential_graph(notes, search_provider="mock")
        understanding = state["memo_json"]["company_understanding"]
        assert understanding["category"] == expected_category
        assert understanding["primary_product"] != "Unknown"
        risk_names = {risk["risk_name"] for risk in state["memo_json"]["visualizations"]["risk_breakdown"]["risks"]}
        assert expected_risk in risk_names


def test_product_coverage_does_not_reach_full_when_product_evidence_is_unknown():
    notes = "Met Nimbus. Product appears to be ambient clinical documentation and AI scribing for health systems and clinicians."
    state = run_sequential_graph(notes, search_provider="mock")
    product_quality = next(row for row in state["memo_json"]["evidence_dashboard"]["heatmap"] if row["topic"] == "Product")
    product_coverage = next(row for row in state["memo_json"]["evidence_coverage"]["rows"] if row["category"] == "Product")
    assert product_quality["quality"] == "Unknown"
    assert product_coverage["coverage"] < 100


def test_missing_data_detection_prioritizes_decision_blocking_gaps():
    state = run_sequential_graph(SAMPLE_NOTES, search_provider="mock")
    missing = state["memo_json"]["missing_data"]
    assert isinstance(missing["decision_readiness"], int)
    assert 0 <= missing["decision_readiness"] <= 100
    assert missing["section_status"]
    assert missing["high_priority"]
    assert missing["decision_blocking_unknowns"]
    assert all(gap["priority_score"] == gap["business_impact"] + gap["decision_impact"] + gap["evidence_gap"] for gap in missing["high_priority"])
    assert "investment decision is premature" in missing["recommendation_adjustment"]
    assert "Conviction remains limited" in state["memo_json"]["recommendation"]["reason"]


def test_coverage_rewards_distinct_slots_not_duplicate_sources():
    from src.nodes import _evidence_coverage

    duplicate_funding_sources = normalize_evidence(
        [
            {
                "title": f"Company raises Series A duplicate {idx}",
                "url": f"https://businesswire.com/news/{idx}",
                "snippet": "Company raised a Series A funding round from investors.",
                "evidence_type": "funding",
                "confidence": 0.9,
            }
            for idx in range(5)
        ]
    )
    memo = {"section_confidence": {"Recent Funding": "High"}, "company_understanding": {}, "visualizations": {}, "opportunity_scorecard": []}
    coverage = _evidence_coverage(duplicate_funding_sources, memo)
    funding = next(row for row in coverage["rows"] if row["category"] == "Funding")
    assert funding["coverage"] < 100
    assert "latest round" in funding["found_slots"]
    assert "valuation" in funding["missing_slots"]


def test_competition_coverage_requires_all_specific_slots():
    from src.nodes import _evidence_coverage

    competitor_sources = normalize_evidence(
        [
            {
                "title": "Direct and adjacent competitor review",
                "url": "https://legaltechnews.example/competition",
                "snippet": "Direct competitors include Harvey and Spellbook. Adjacent competitors include CLM platforms. Incumbent competitors include legacy CLM suites.",
                "evidence_type": "competitor",
                "confidence": 0.8,
            }
        ]
    )
    memo = {"section_confidence": {"Competitive Landscape": "Medium"}, "company_understanding": {}, "visualizations": {}, "opportunity_scorecard": []}
    coverage = _evidence_coverage(competitor_sources, memo)
    competition = next(row for row in coverage["rows"] if row["category"] == "Competition")
    assert competition["coverage"] == 60
    assert "pricing overlap" in competition["missing_slots"]
    assert "win/loss data" in competition["missing_slots"]


def test_company_specific_evidence_defines_product_before_market_context():
    notes = "Met founder of Abridge. Healthcare AI company selling into health systems."
    state = {
        "raw_notes": notes,
        "company_profile": {},
        "research_plan": [],
        "search_results": [
            {
                "title": "Abridge ambient clinical documentation",
                "url": "https://example.com/abridge",
                "snippet": "Abridge uses ambient AI to generate clinical documentation from doctor-patient conversations for clinicians.",
                "query": "Abridge official website product",
                "evidence_type": "website",
                "confidence": 0.8,
            },
            {
                "title": "Prior authorization automation market",
                "url": "https://example.com/prior-auth",
                "snippet": "Prior authorization automation is a healthcare administrative workflow market.",
                "query": "healthcare AI market context",
                "evidence_type": "market",
                "confidence": 0.7,
            },
        ],
        "evidence": [],
        "memo_json": {},
        "chart_data": {},
        "validation": {"is_valid": False, "unsupported_claims": [], "warnings": [], "source_coverage": {}},
        "memo_markdown": "",
        "errors": [],
        "trace": [],
    }
    for step in [extract_company_profile, plan_research, build_evidence_store, refine_company_profile_from_evidence, write_structured_memo]:
        state.update(step(state))

    assert state["company_profile"]["product"] == "ambient clinical documentation and AI scribing"
    memo_text = str(state["memo_json"])
    assert "ambient clinical documentation" in memo_text
    assert "actual product wedge (ambient clinical documentation and ai scribing)" in memo_text


def test_market_research_runs_after_product_classification():
    notes = "Met founder of Abridge. Healthcare AI company selling into health systems."
    state = {
        "raw_notes": notes,
        "company_profile": {},
        "research_plan": [],
        "search_results": [
            {
                "title": "Abridge ambient clinical documentation",
                "url": "https://example.com/abridge",
                "snippet": "Abridge uses ambient AI to generate clinical documentation from doctor-patient conversations for clinicians.",
                "query": "Abridge official website product",
                "evidence_type": "website",
                "confidence": 0.8,
            }
        ],
        "evidence": [],
        "memo_json": {},
        "chart_data": {},
        "validation": {"is_valid": False, "unsupported_claims": [], "warnings": [], "source_coverage": {}},
        "memo_markdown": "",
        "errors": [],
        "trace": [],
    }
    state.update(extract_company_profile(state))
    state.update(plan_research(state))
    assert "market" not in {query["evidence_type"] for query in state["research_plan"]}

    state.update(build_evidence_store(state))
    state.update(refine_company_profile_from_evidence(state))
    state.update(plan_context_research(state))
    context_queries = " ".join(query["query"] for query in state["research_plan"]).lower()
    assert "ambient clinical documentation" in context_queries
    assert "prior authorization" not in context_queries


def test_contractpilot_company_understanding_uses_partner_notes_before_research():
    notes = (
        "ContractPilot is an AI contract review platform for legal teams. "
        "Automates redlining, clause extraction, and risk detection. "
        "Competition with Ironclad and Harvey."
    )
    state = {
        "raw_notes": notes,
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
    state.update(extract_company_profile(state))
    understanding = state["company_profile"]["company_understanding"]
    assert understanding["company_name"] == "ContractPilot"
    assert understanding["category"] == "LegalTech"
    assert understanding["primary_product"] == "AI contract review, redlining, clause extraction, and risk detection"
    assert understanding["target_buyer"] == "legal teams and legal operations"
    assert understanding["budget_owner"] == "General Counsel / Legal Operations"
    assert "contract review" in understanding["core_workflow"]
    assert understanding["competitors_mentioned_in_notes"] == ["Ironclad", "Harvey"]
    assert understanding["validation"]["confidence"] >= 0.8

    state.update(plan_research(state))
    state.update(validate_company_understanding(state))
    query_text = " ".join(query["query"] for query in state["research_plan"]).lower()
    assert "contract review" in query_text
    assert "legal teams" in query_text
    assert "ironclad" in query_text
    assert "harvey" in query_text
    assert "ai workflow automation competitors" not in query_text
    assert state["company_profile"]["company_understanding"]["validation"]["is_valid"]


def test_contractpilot_memo_uses_legaltech_competitors_and_risks():
    notes = (
        "ContractPilot is an AI contract review platform for legal teams. "
        "Automates redlining, clause extraction, and risk detection. "
        "Competition with Ironclad and Harvey."
    )
    state = {
        "raw_notes": notes,
        "company_profile": {},
        "research_plan": [],
        "search_results": [
            {
                "title": "ContractPilot AI contract review",
                "url": "https://contractpilot.example",
                "snippet": "ContractPilot provides AI contract review, redlining, clause extraction, and risk detection software for legal teams.",
                "query": "ContractPilot AI contract review legal teams",
                "evidence_type": "website",
                "confidence": 0.9,
            },
            {
                "title": "Legal AI contract review market",
                "url": "https://legaltechnews.example/contract-review",
                "snippet": "Legal teams use AI contract review software to reduce review time and standardize clause risk detection.",
                "query": "legal AI contract redlining software market",
                "evidence_type": "market",
                "confidence": 0.7,
            },
            {
                "title": "AI contract review competitors",
                "url": "https://legaltechnews.example/competitors",
                "snippet": "Ironclad, Harvey, Evisort, LinkSquares, Lexion, and Spellbook are relevant legal workflow and contract review vendors.",
                "query": "AI contract review software competitors Ironclad Harvey",
                "evidence_type": "competitor",
                "confidence": 0.75,
            },
        ],
        "evidence": [],
        "memo_json": {},
        "chart_data": {},
        "validation": {"is_valid": False, "unsupported_claims": [], "warnings": [], "source_coverage": {}},
        "memo_markdown": "",
        "errors": [],
        "trace": [],
    }
    for step in [extract_company_profile, plan_research, validate_company_understanding, build_evidence_store, refine_company_profile_from_evidence, write_structured_memo, generate_chart_data]:
        state.update(step(state))

    memo_text = str(state["memo_json"]).lower()
    risks = state["memo_json"]["visualizations"]["risk_breakdown"]["risks"]
    competitors = state["chart_data"]["competitive_table"]
    assert "legaltech" in memo_text
    assert "ai contract review" in memo_text
    assert any(row["company"] == "Ironclad" for row in competitors)
    assert any(row["company"] == "Harvey" for row in competitors)
    assert any(risk["risk_name"] == "Legal accuracy risk" for risk in risks)
    assert any(risk["risk_name"] == "Hallucinated clause interpretation risk" for risk in risks)
    assert "hipaa" not in str(risks).lower()
    assert "ehr" not in str(risks).lower()
    assert "healthcare procurement" not in memo_text


def test_ambiguous_nimbus_with_clear_notes_runs_workflow_level_diligence():
    notes = (
        "Met Nimbus. Product appears to be ambient clinical documentation and AI scribing "
        "for health systems and clinicians. Need to verify founder and customers."
    )
    state = {
        "raw_notes": notes,
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
    state.update(extract_company_profile(state))
    identity = state["company_profile"]["identity_resolution"]
    assert identity["input_company_name"] == "Nimbus"
    assert not identity["is_resolved"]
    assert identity["confidence"] < 0.8
    assert {candidate["name"] for candidate in identity["candidates"]} >= {"Nimbus Data", "Nimbus Therapeutics", "JobNimbus"}
    assert state["company_profile"]["company_understanding"]["validation"]["confidence"] >= 0.8

    state.update(plan_research(state))
    assert {query["evidence_type"] for query in state["research_plan"]} == {"market", "competitor"}
    query_text = " ".join(query["query"] for query in state["research_plan"]).lower()
    assert "ambient clinical documentation" in query_text
    assert "nimbus" not in query_text
    state.update(validate_company_understanding(state))
    assert "Unable to confidently identify company." not in state["errors"]

    for step in [build_evidence_store, refine_company_profile_from_evidence, plan_context_research, build_evidence_store, write_structured_memo, generate_chart_data]:
        state.update(step(state))
    assert not any(item["evidence_type"] == "funding" for item in state["evidence"])
    assert state["memo_json"]["recommendation"]["decision"] == "Proceed to Workflow-Level Diligence - Verify Company Identity"
    assert "workflow-level market, risk, and competitor diligence" in state["memo_json"]["recommendation"]["reason"]
    risk_names = [risk["risk_name"] for risk in state["memo_json"]["visualizations"]["risk_breakdown"]["risks"]]
    assert "Clinical adoption risk" in risk_names
    assert "## Company Identity Resolution" in state["memo_markdown"] or state["memo_json"]["identity_resolution"]["message"] == "Unable to confidently identify company."


def test_low_understanding_still_runs_identity_discovery_research():
    notes = "Met Atlas. AI platform helping enterprises make better decisions. Need to understand product, buyer, and revenue model."
    state = {
        "raw_notes": notes,
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
    state.update(extract_company_profile(state))
    assert not state["company_profile"]["identity_resolution"]["is_resolved"]
    assert state["company_profile"]["company_understanding"]["validation"]["confidence"] < 0.8

    state.update(plan_research(state))
    assert state["research_plan"]
    assert {query["evidence_type"] for query in state["research_plan"]} >= {"website", "news"}
    query_text = " ".join(query["query"] for query in state["research_plan"]).lower()
    assert "atlas" in query_text
    assert "official website" in query_text


def test_figma_notes_produce_product_design_understanding():
    notes = (
        "Met Figma.\n\n"
        "Collaborative design platform.\n\n"
        "Exceptional product love.\n\n"
        "Need to understand:\n"
        "- Enterprise expansion\n"
        "- AI roadmap\n"
        "- Competitive pressure\n"
    )
    state = {
        "raw_notes": notes,
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
    state.update(extract_company_profile(state))
    understanding = state["company_profile"]["company_understanding"]
    assert state["company_profile"]["identity_resolution"]["is_resolved"]
    assert understanding["company_name"] == "Figma"
    assert understanding["category"] == "Product Design Software"
    assert "collaborative product design" in understanding["primary_product"]
    assert "interface design" in understanding["core_workflow"]
    assert "design, product, engineering" in understanding["target_buyer"]
    assert "Adobe" in understanding["competitors"]

    state.update(plan_research(state))
    state.update(validate_company_understanding(state))
    assert "Unable to confidently identify company or understand workflow." not in state["errors"]
    query_text = " ".join(query["query"] for query in state["research_plan"]).lower()
    assert "figma" in query_text
    assert "collaborative" in query_text


def test_product_design_understanding_is_company_agnostic():
    notes = (
        "Met CanvasForge.\n\n"
        "Collaborative design platform for product teams.\n\n"
        "Exceptional product love.\n\n"
        "Need to understand:\n"
        "- Enterprise expansion\n"
        "- AI roadmap\n"
        "- Competitive pressure\n"
    )
    state = {
        "raw_notes": notes,
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
    state.update(extract_company_profile(state))
    understanding = state["company_profile"]["company_understanding"]
    assert state["company_profile"]["identity_resolution"]["is_resolved"]
    assert understanding["company_name"] == "CanvasForge"
    assert understanding["category"] == "Product Design Software"
    assert "collaborative product design" in understanding["primary_product"]
    assert "design, product, engineering" in understanding["target_buyer"]
    assert "Adobe" in understanding["competitors"]

    state.update(plan_research(state))
    state.update(validate_company_understanding(state))
    query_text = " ".join(query["query"] for query in state["research_plan"]).lower()
    assert "canvasforge" in query_text
    assert "collaborative" in query_text
    assert "Unable to confidently identify company or understand workflow." not in state["errors"]


def test_irrelevant_search_results_are_filtered_from_evidence():
    evidence = normalize_evidence(
        [
            {
                "title": "Oracle E-Business Suite Implementation Guide",
                "url": "https://example.com/oracle-erp",
                "snippet": "Oracle and SAP ERP migrations for finance operations.",
                "query": "Figma collaborative design platform market context product design",
                "evidence_type": "market",
                "confidence": 0.75,
            },
            {
                "title": "Product design software market context",
                "url": "https://example.com/product-design-market",
                "snippet": "Collaborative design platforms support prototyping, design systems, and developer handoff workflows.",
                "query": "CanvasForge collaborative design platform market context product design",
                "evidence_type": "market",
                "confidence": 0.75,
            },
        ]
    )
    assert len(evidence) == 1
    assert evidence[0]["title"] == "Product design software market context"


def test_unknown_category_does_not_emit_industry_specific_risks():
    notes = "Met Nimbus. Product and buyer are unclear. Need to identify the correct company."
    state = run_sequential_graph(notes, search_provider="mock")
    assert state["memo_json"]["recommendation"]["decision"] == "Do Not Advance Yet - Resolve Company Identity"
    risks = state["memo_json"]["visualizations"]["risk_breakdown"]["risks"]
    assert len(risks) == 1
    assert risks[0]["risk_name"] == "Industry-specific risks unavailable"
    assert risks[0]["risk_type"] == "workflow_inferred"
    assert risks[0]["score"] is None
    assert risks[0]["confidence"] == "Low"
    assert "Industry-specific risks unavailable until company category is verified." in risks[0]["reason"]
    risk_text = str(risks).lower()
    assert "hipaa" not in risk_text
    assert "ehr" not in risk_text
    assert "interchange" not in risk_text
    assert "contract review" not in risk_text


def test_risks_use_note_workflow_even_when_formal_category_is_weak():
    from src.nodes import _risk_breakdown

    profile = {
        "raw_notes": "Product improves documentation workflow for health systems. Buyer is CMIO. Need diligence on HIPAA, PHI, integration burden, physician trust, and security.",
        "sector": "Unknown",
        "product": "Not specified",
        "identity_resolution": {"is_resolved": False, "confidence": 0.4},
        "company_understanding": {
            "industry": "Unknown",
            "subindustry": "Unknown",
            "primary_product": "documentation workflow for health systems",
            "target_buyer": "CMIO / CIO / clinical operations",
            "core_workflow": "clinical documentation workflow",
            "critical_dependencies": [],
            "validation": {"confidence": 0.5},
        },
    }
    risks = _risk_breakdown([], "Nimbus", profile)["risks"]
    risk_names = {risk["risk_name"] for risk in risks}
    assert "Industry-specific risks unavailable" not in risk_names
    assert "Clinical adoption risk" in risk_names
    assert "HIPAA / compliance risk" in risk_names
    assert any(risk["risk_type"] == "note_explicit" for risk in risks)


def test_conflicting_positioning_escalates_and_blocks_fallback_templates():
    notes = """
    Met Orion.

    Founder describes company as cybersecurity.
    Website describes workflow automation.
    Customers describe it as compliance software.

    Interesting because category seems large.

    Need to understand:
    - Actual product
    - Actual buyer
    - Competitive landscape
    """
    state = run_sequential_graph(notes, search_provider="mock")
    understanding = state["memo_json"]["company_understanding"]
    contradiction = understanding["validation"]["contradictions"]
    assert contradiction["severity"] == "high"
    assert understanding["validation"]["confidence"] <= 0.3
    assert state["memo_json"]["recommendation"]["decision"] == "Clarify Positioning Before Workflow-Level Analysis"
    assert "Conflicting signals detected" in state["memo_markdown"]
    assert "Interpretation 1" in state["memo_markdown"]
    assert state["evidence"] == []

    memo_text = state["memo_markdown"].lower()
    forbidden = ["ambient clinical documentation", "ai scribing", "health systems and clinicians", "cmio", "epic ecosystem", "hipaa", "clinical workflows"]
    for term in forbidden:
        assert term not in memo_text

    risks = state["memo_json"]["visualizations"]["risk_breakdown"]["risks"]
    assert len(risks) == 1
    assert risks[0]["risk_name"] == "Conflicting positioning signals"


def test_note_explicit_radiology_risks_are_generated_without_resolved_identity():
    notes = (
        "Met Nimbus. Product is radiology report generation AI for radiologists and imaging teams. "
        "Partner flagged FDA regulatory risk, diagnostic accuracy, clinical liability, physician trust, PACS/RIS integration, and privacy/security."
    )
    state = run_sequential_graph(notes, search_provider="mock")
    identity = state["memo_json"]["identity_resolution"]
    understanding = state["memo_json"]["company_understanding"]
    assert not identity["is_resolved"]
    assert understanding["validation"]["confidence"] >= 0.8
    assert state["memo_json"]["recommendation"]["decision"] == "Proceed to Workflow-Level Diligence - Verify Company Identity"

    risks = state["memo_json"]["visualizations"]["risk_breakdown"]["risks"]
    by_name = {risk["risk_name"]: risk for risk in risks}
    assert by_name["Diagnostic accuracy risk"]["risk_type"] == "note_explicit"
    assert by_name["Diagnostic accuracy risk"]["confidence"] == "High"
    assert by_name["FDA / regulatory risk"]["risk_type"] == "note_explicit"
    assert by_name["Clinical liability risk"]["risk_type"] == "note_explicit"
    assert by_name["PACS/RIS/EHR integration risk"]["risk_type"] == "note_explicit"
    assert by_name["Data privacy and security risk"]["risk_type"] == "note_explicit"
    assert any(risk["risk_type"] == "workflow_inferred" for risk in risks)
    assert "## Risk Breakdown Graph" in state["memo_markdown"]
    assert "| Risk | Type | Score | Confidence | Why It Matters | Evidence |" in state["memo_markdown"]


def test_abridge_competitors_prioritize_same_workflow_buyer_and_category():
    notes = "Met founder of Abridge. Healthcare AI company selling into health systems."
    state = {
        "raw_notes": notes,
        "company_profile": {},
        "research_plan": [],
        "search_results": [
            {
                "title": "Abridge ambient clinical documentation",
                "url": "https://example.com/abridge",
                "snippet": "Abridge uses ambient AI to generate clinical documentation from doctor-patient conversations for clinicians.",
                "query": "Abridge official website product",
                "evidence_type": "website",
                "confidence": 0.8,
            },
            {
                "title": "Ambient clinical documentation competitors",
                "url": "https://example.com/ambient-competitors",
                "snippet": "Ambient clinical documentation competitors include Nuance DAX, Suki, Nabla, and DeepScribe.",
                "query": "Abridge ambient clinical documentation competitors",
                "evidence_type": "competitor",
                "confidence": 0.8,
            },
        ],
        "evidence": [],
        "memo_json": {},
        "chart_data": {},
        "validation": {"is_valid": False, "unsupported_claims": [], "warnings": [], "source_coverage": {}},
        "memo_markdown": "",
        "errors": [],
        "trace": [],
    }
    for step in [extract_company_profile, plan_research, build_evidence_store, refine_company_profile_from_evidence, plan_context_research]:
        state.update(step(state))
    query_text = " ".join(query["query"] for query in state["research_plan"])
    assert "Nuance DAX" in query_text
    assert "Suki" in query_text
    assert "Nabla" in query_text
    assert "DeepScribe" in query_text
    assert "prior authorization" not in query_text.lower()

    state.update(generate_chart_data(state))
    table_text = str(state["chart_data"]["competitive_table"])
    assert "Nuance DAX" in table_text
    assert "Suki" in table_text
    assert "Nabla" in table_text
    assert "DeepScribe" in table_text
    assert "ambient clinical documentation" in table_text
    assert "buyer_overlap" in table_text
    assert "workflow_overlap" in table_text


def test_ramp_company_understanding_drives_domain_specific_memo():
    notes = """
    Met Ramp team. Company provides corporate cards, expense management, bill pay, and procurement software
    for CFOs and finance teams. Selling to startups, SMBs, mid-market, and enterprise customers.
    Need to understand interchange dependence, software attach, retention, competition from Brex, Airbase,
    Navan, and legacy banks, plus path to durable margins.
    """
    state = run_sequential_graph(notes, search_provider="mock")
    understanding = state["memo_json"]["company_understanding"]
    assert understanding["company_name"] == "Ramp"
    assert understanding["category"] == "Fintech / Spend Management"
    assert understanding["subindustry"] == "Spend Management"
    assert understanding["primary_product"] == "corporate cards, spend management, bill pay, and procurement software"
    assert "CFOs and finance teams" in understanding["target_buyer"]
    assert understanding["core_workflow"] == "corporate spend management, expense control, bill pay, and procurement"
    assert "Banking partners" in understanding["critical_dependencies"]
    assert "Card spend" in understanding["revenue_drivers"]

    markdown = state["memo_markdown"]
    assert "## Company Understanding" in markdown
    assert "CFOs and finance teams" in markdown
    assert "HIPAA" not in markdown
    assert "clinical workflow" not in markdown.lower()
    assert "healthcare procurement" not in markdown.lower()

    risks = state["memo_json"]["visualizations"]["risk_breakdown"]["risks"]
    risk_names = {risk["risk_name"] for risk in risks}
    assert "Interchange concentration risk" in risk_names
    assert "Banking partner risk" in risk_names
    assert "Fraud risk" in risk_names
    assert "Credit risk" in risk_names
    assert "Clinical adoption risk" not in risk_names
    assert all("diligence_question" in risk for risk in risks)


def test_abridge_healthcare_ai_risks_are_industry_specific():
    notes = """
    Met Abridge team. Company uses ambient AI to listen to doctor-patient conversations and generate
    clinical documentation for physicians. Selling into large health systems. Need to understand Epic
    integration, clinician adoption, ROI, and competition from Nuance DAX, Suki, Nabla, and DeepScribe.
    """
    state = run_sequential_graph(notes, search_provider="mock")
    understanding = state["memo_json"]["company_understanding"]
    assert understanding["industry"] == "Healthcare AI"
    assert understanding["subindustry"] == "Clinical Documentation"
    assert "Epic ecosystem" in understanding["critical_dependencies"]
    assert "Provider adoption" in understanding["revenue_drivers"]

    risks = state["memo_json"]["visualizations"]["risk_breakdown"]["risks"]
    risk_names = {risk["risk_name"] for risk in risks}
    assert "Clinical adoption risk" in risk_names
    assert "Documentation accuracy risk" in risk_names
    assert "EHR dependency risk" in risk_names
    assert "HIPAA / compliance risk" in risk_names
    assert "Interchange concentration risk" not in risk_names
    assert any("Epic" in risk["diligence_question"] for risk in risks)


def test_ramp_competitor_discovery_uses_economic_overlap_not_keywords():
    notes = """
    Met Ramp team. Company provides corporate cards, expense management, bill pay, and procurement software
    for CFOs and finance teams. Need to understand competition from Brex, Airbase, Navan, Amex, and Coupa.
    """
    state = run_sequential_graph(notes, search_provider="mock")
    table = state["chart_data"]["competitive_table"]
    table_text = str(table)
    assert "Brex" in table_text
    assert "Airbase" in table_text
    assert "Navan" in table_text
    assert "American Express" in table_text
    assert "SAP Concur" in table_text
    assert "Coupa" in table_text
    assert "Mastra" not in table_text
    assert "Gumloop" not in table_text
    assert all(row["competitor_score"] >= 6 for row in table if row["competitor_score"] is not None)
    assert any(row["category"] == "Direct Competitors" and row["threat_level"].startswith("High") for row in table)
    assert all("budget" in row["why_it_competes"].lower() for row in table)
