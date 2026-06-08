from __future__ import annotations

import re
from typing import Any

from .search import normalize_evidence, run_search
from .state import DealMemoState, EvidenceItem, MemoClaim, ResearchQuery


REQUIRED_SECTIONS = [
    "Executive Summary",
    "Recommendation",
    "Bull Case",
    "Bear Case",
    "What Needs To Be True",
    "Partner Note Verification",
    "Next Diligence Priorities",
    "Opportunity Scorecard",
    "Company Overview",
    "Founding Team",
    "Product",
    "Why Now",
    "Market Opportunity",
    "Competitive Landscape",
    "Business Model",
    "Recent Funding",
    "Key Risks",
    "Investment Thesis",
    "Open Questions",
]


def _trace(state: DealMemoState, node: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [*state.get("trace", []), {"node": node, "output": payload}]


def _guess_company(raw_notes: str) -> str:
    direct_match = re.search(
        r"(?i:\bmet\s+(?:the\s+)?)([A-Z][A-Za-z0-9 .&-]{2,40}?)(?:\s+team|\s+founder|\s+exec|\s+executive|[.,\n]|$)",
        raw_notes,
    )
    if direct_match:
        return direct_match.group(1).strip().rstrip(".")
    match = re.search(
        r"(?:founder of|met founder of|company is|intro from)\s+([A-Z][A-Za-z0-9 .-]{2,40}?)(?:\s+after|\s+with|\s+that|\s+who|[.,\n]|$)",
        raw_notes,
        re.I,
    )
    if match:
        return match.group(1).strip().rstrip(".")
    proper = re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\b", raw_notes)
    return proper[0] if proper else "Unknown company"


AMBIGUOUS_COMPANY_NAMES = {"Nimbus", "Mercury", "Pilot", "Cedar", "Coda", "Bench", "Arc", "Pulse"}


def _extract_founder(raw_notes: str) -> str:
    patterns = [
        r"(?:founder|ceo|co-founder)\s+(?:is|=|:)?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})",
        r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\s+(?:is|was)\s+(?:the\s+)?(?:founder|ceo|co-founder)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_notes)
        if match:
            name = match.group(1).strip()
            if name.lower().split()[0] not in {"and", "or", "the", "team", "founder", "customers"}:
                return name
    return ""


def _identity_attributes(raw_notes: str, product: str, sector: str, customers: list[str]) -> dict[str, Any]:
    framework = _competitor_framework(product, customers or ["not specified"])
    return {
        "product": product,
        "buyer": framework["primary_buyer"],
        "workflow": framework["primary_workflow"],
        "founder": _extract_founder(raw_notes),
        "customers": customers,
        "sector": sector,
    }


def _candidate_entities(company: str, attrs: dict[str, Any]) -> list[dict[str, Any]]:
    product = attrs.get("product", "")
    workflow = attrs.get("workflow", "")
    buyer = attrs.get("buyer", "")
    founder = attrs.get("founder", "")
    sector = attrs.get("sector", "Unknown")
    base = {
        "name": company,
        "product": product,
        "workflow": workflow,
        "buyer": buyer,
        "founder": founder,
        "source": "partner_notes",
    }
    if company == "Unknown company":
        return [base]
    if company in AMBIGUOUS_COMPANY_NAMES:
        if company == "Nimbus":
            return [
                base,
                {"name": "Nimbus Data", "product": "enterprise flash storage arrays", "workflow": "data storage infrastructure", "buyer": "IT infrastructure teams", "founder": "Thomas Isakovich", "source": "known_ambiguous_entity"},
                {"name": "Nimbus Therapeutics", "product": "small molecule therapeutics", "workflow": "drug discovery and clinical development", "buyer": "pharma partners and healthcare investors", "founder": "", "source": "known_ambiguous_entity"},
                {"name": "JobNimbus", "product": "CRM and project management for contractors", "workflow": "contractor sales and project management", "buyer": "roofing and home service contractors", "founder": "", "source": "known_ambiguous_entity"},
                {"name": "Nimbus Portal Solutions", "product": "customer portal software", "workflow": "portal and workflow management", "buyer": "operations teams", "founder": "", "source": "known_ambiguous_entity"},
            ]
        return [base, {"name": f"{company} Data", "product": "data infrastructure", "workflow": "data management", "buyer": "IT teams", "founder": "", "source": "synthetic_ambiguity_check"}]
    if sector == "Unknown" or product == "Not specified":
        return [base]
    return [base]


def _token_overlap_score(left: str, right: str) -> int:
    left_tokens = {token for token in re.split(r"[^a-z0-9]+", left.lower()) if len(token) > 3}
    right_tokens = {token for token in re.split(r"[^a-z0-9]+", right.lower()) if len(token) > 3}
    if not left_tokens or not right_tokens:
        return 0
    overlap = len(left_tokens & right_tokens) / len(left_tokens)
    return round(overlap * 10)


def _score_entity_candidate(candidate: dict[str, Any], attrs: dict[str, Any]) -> dict[str, Any]:
    product_match = _token_overlap_score(attrs.get("product", ""), candidate.get("product", ""))
    workflow_match = _token_overlap_score(attrs.get("workflow", ""), candidate.get("workflow", ""))
    buyer_match = _token_overlap_score(attrs.get("buyer", ""), candidate.get("buyer", ""))
    founder = attrs.get("founder", "")
    founder_match = 10 if founder and founder.lower() == str(candidate.get("founder", "")).lower() else 0
    entity_score = round(
        (0.4 * product_match + 0.3 * workflow_match + 0.2 * buyer_match + 0.1 * founder_match) / 10,
        2,
    )
    return {
        **candidate,
        "product_match": product_match,
        "workflow_match": workflow_match,
        "buyer_match": buyer_match,
        "founder_match": founder_match,
        "entity_score": entity_score,
    }


def _resolve_company_identity(company: str, raw_notes: str, product: str, sector: str, customers: list[str]) -> dict[str, Any]:
    attrs = _identity_attributes(raw_notes, product, sector, customers)
    scored = [_score_entity_candidate(candidate, attrs) for candidate in _candidate_entities(company, attrs)]
    scored = sorted(scored, key=lambda candidate: candidate["entity_score"], reverse=True)
    selected = scored[0] if scored else {"name": company, "entity_score": 0}
    confidence = float(selected.get("entity_score", 0))
    if company in AMBIGUOUS_COMPANY_NAMES and selected.get("source") == "partner_notes" and not attrs.get("founder"):
        confidence = min(confidence, 0.75)
        selected = {**selected, "entity_score": confidence}
        scored = [{**candidate, "entity_score": confidence} if candidate.get("name") == selected.get("name") else candidate for candidate in scored]
    if company == "Unknown company" or product == "Not specified" or sector == "Unknown":
        confidence = min(confidence, 0.5)
        selected = {**selected, "entity_score": confidence}
        scored = [{**candidate, "entity_score": confidence} if candidate.get("name") == selected.get("name") else candidate for candidate in scored]
    return {
        "input_company_name": company,
        "extracted_attributes": attrs,
        "candidates": scored,
        "selected_entity": selected,
        "confidence": confidence,
        "is_resolved": confidence >= 0.8,
        "message": "Resolved company identity." if confidence >= 0.8 else "Unable to confidently identify company.",
    }


def _contains_any(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def _customer_context_contains(raw_notes: str, terms: list[str]) -> bool:
    customer_context = re.compile(
        r"\b(customers?|buyers?|selling|sells?|sold\s+to|serving|serves?|used\s+by|users?|clients?)\b",
        re.I,
    )
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", raw_notes):
        lower = sentence.lower()
        if customer_context.search(sentence) and any(term in lower for term in terms):
            return True
    return False


def _infer_product_from_text(text: str) -> str:
    lower = text.lower()
    if _contains_any(lower, ["contract review", "redlining", "clause extraction", "contract redlining", "legal teams", "legal ops"]):
        return "AI contract review, redlining, clause extraction, and risk detection"
    if _contains_any(lower, ["legal professionals", "law firm", "law firms", "legal workflow", "legal workflows"]):
        return "AI legal workflow platform for legal professionals"
    if _contains_any(lower, ["radiology", "radiologist", "imaging", "pacs", "diagnostic report", "radiology report"]):
        return "radiology report generation and imaging workflow AI"
    if _contains_any(lower, ["defense", "autonomous systems", "drones", "sensors", "military", "border security"]):
        return "defense technology, autonomous systems, sensors, and mission software"
    if _contains_any(lower, ["robotics", "robot", "robots", "warehouse automation", "industrial automation", "autonomous mobile robot", "amr", "cobots"]):
        return "robotics and automation systems for physical operations"
    if _contains_any(lower, ["corporate card", "expense management", "spend management", "bill pay", "finance teams"]):
        return "corporate cards, spend management, bill pay, and procurement software"
    if _contains_any(lower, ["carbon accounting", "carbon management", "emissions", "climate risk", "energy management", "grid", "renewable", "decarbonization", "climatetech", "climate tech"]):
        return "climate software for emissions, energy, and decarbonization workflows"
    if _contains_any(lower, ["hr", "human resources", "talent", "recruiting", "recruitment", "payroll", "benefits", "employee engagement", "workforce"]):
        return "HR software for talent, workforce, payroll, or employee workflows"
    if _contains_any(lower, ["supply chain", "logistics", "freight", "inventory", "warehouse", "warehousing", "procurement", "supplier", "demand planning"]):
        return "supply chain software for logistics, inventory, procurement, and planning workflows"
    if _contains_any(lower, ["biotech", "drug discovery", "therapeutics", "clinical trial", "clinical trials", "genomics", "bioinformatics", "life sciences", "wet lab"]):
        return "biotech platform for drug discovery, clinical development, or life sciences workflows"
    if _contains_any(lower, ["ambient", "medical scribe", "ai scribe", "clinical documentation", "clinical note", "doctor-patient", "physician documentation"]):
        return "ambient clinical documentation and AI scribing"
    if _contains_any(lower, ["prior authorization", "utilization management"]):
        return "prior authorization and utilization management automation"
    if _contains_any(lower, ["revenue cycle", "rcm"]):
        return "revenue cycle automation"
    return "Not specified"


BROAD_PRODUCT_TERMS = [
    "platform",
    "suite",
    "operating system",
    "infrastructure",
    "marketplace",
    "end-to-end solution",
    "end to end solution",
]


def _product_scope_from_text(raw_notes: str, product: str) -> tuple[str, str]:
    combined = f"{raw_notes} {product}".lower()
    product_known = not _is_placeholder_entity(product) and product != "Not specified"
    if _contains_any(combined, ["marketplace", "exchange", "network connecting", "two-sided"]):
        return "marketplace", "high" if "marketplace" in raw_notes.lower() else "medium"
    if _contains_any(combined, ["infrastructure", "api platform", "developer platform", "data infrastructure", "cloud infrastructure"]):
        return "infrastructure", "high" if "infrastructure" in raw_notes.lower() else "medium"
    if _contains_any(combined, ["services-enabled", "service-enabled", "managed service", "human-in-the-loop", "tech-enabled service"]):
        return "services_enabled", "high"
    if _contains_any(raw_notes, BROAD_PRODUCT_TERMS):
        return "broad_platform", "high"
    if product_known and _contains_any(product, ["platform", "suite", "infrastructure"]):
        return "broad_platform", "medium"
    if product_known:
        return "specific_workflow", "medium"
    return "unclear", "low"


def _extract_use_cases_from_text(text: str) -> list[str]:
    lower = text.lower()
    use_case_map = [
        ("contract review", ["contract review", "redlining", "clause extraction"]),
        ("legal research", ["legal research", "case law research"]),
        ("legal drafting", ["legal drafting", "drafting legal", "brief drafting", "drafting workflows"]),
        ("carbon accounting", ["carbon accounting", "emissions reporting"]),
        ("energy management", ["energy management", "energy optimization"]),
        ("decarbonization planning", ["decarbonization planning", "climate risk"]),
        ("talent acquisition", ["talent acquisition", "recruiting", "recruitment"]),
        ("payroll and benefits", ["payroll", "benefits"]),
        ("employee engagement", ["employee engagement", "workforce engagement"]),
        ("inventory planning", ["inventory planning", "demand planning"]),
        ("logistics operations", ["logistics", "freight", "shipment"]),
        ("warehouse operations", ["warehouse", "warehousing", "warehouse operations"]),
        ("supplier management", ["supplier management", "supplier"]),
        ("robot fleet operations", ["robot fleet", "fleet operations", "autonomous mobile robot"]),
        ("manufacturing automation", ["manufacturing automation", "industrial automation"]),
        ("drug discovery", ["drug discovery"]),
        ("clinical development", ["clinical development", "clinical trial"]),
        ("genomics and bioinformatics", ["genomics", "bioinformatics"]),
        ("clinical documentation", ["clinical documentation", "ambient documentation", "medical scribe", "ai scribe"]),
        ("radiology reporting", ["radiology reporting", "radiology report", "diagnostic report"]),
        ("prior authorization", ["prior authorization", "utilization management"]),
        ("revenue cycle", ["revenue cycle", "rcm"]),
        ("expense management", ["expense management", "spend management"]),
        ("corporate cards", ["corporate card", "corporate cards"]),
        ("bill pay", ["bill pay"]),
        ("procurement", ["procurement"]),
        ("compliance workflows", ["compliance software", "compliance workflow"]),
        ("cybersecurity workflows", ["cybersecurity", "security operations"]),
        ("workflow automation", ["workflow automation"]),
    ]
    use_cases = []
    for label, terms in use_case_map:
        if _contains_any(lower, terms) and label not in use_cases:
            use_cases.append(label)
    return use_cases


def _product_evidence_conflicts(note_product: str, evidence_product: str, note_scope: str) -> list[str]:
    if evidence_product == "Not specified":
        return []
    note_tokens = {token for token in re.split(r"[^a-z0-9]+", note_product.lower()) if len(token) > 3}
    evidence_tokens = {token for token in re.split(r"[^a-z0-9]+", evidence_product.lower()) if len(token) > 3}
    overlap = len(note_tokens & evidence_tokens)
    conflicts = []
    if note_scope == "broad_platform" and evidence_product != note_product:
        conflicts.append(
            "Public product evidence appears narrower than broad partner-note framing; treat it as use-case evidence until the initial wedge is verified."
        )
    elif note_tokens and evidence_tokens and overlap == 0:
        conflicts.append("Public product evidence does not clearly match the product described in partner notes.")
    return conflicts


def _infer_sector_from_text(text: str) -> str:
    lower = text.lower()
    if _contains_any(lower, ["legaltech", "legal tech", "contract review", "redlining", "clause extraction", "general counsel", "legal teams", "legal ops", "legal professionals", "law firm", "law firms", "legal workflow", "legal workflows"]):
        return "LegalTech"
    if _contains_any(lower, ["defense", "military", "autonomous systems", "drones", "sensors", "government", "border security"]):
        return "Defense Technology"
    if _contains_any(lower, ["robotics", "robot", "robots", "warehouse automation", "industrial automation", "autonomous mobile robot", "cobot"]):
        return "Robotics"
    if _contains_any(lower, ["corporate card", "expense management", "spend management", "bill pay", "cfo", "finance team", "fintech"]):
        return "Fintech / Spend Management"
    if _contains_any(lower, ["climatetech", "climate tech", "carbon accounting", "carbon management", "emissions", "renewable", "decarbonization", "energy management", "grid"]):
        return "ClimateTech"
    if _contains_any(lower, ["hrtech", "hr tech", "human resources", "talent", "recruiting", "recruitment", "payroll", "benefits", "employee engagement", "workforce"]):
        return "HRTech"
    if _contains_any(lower, ["supply chain", "logistics", "freight", "inventory", "warehouse", "warehousing", "supplier", "demand planning"]):
        return "Supply Chain"
    if _contains_any(lower, ["biotech", "drug discovery", "therapeutics", "clinical trial", "clinical trials", "genomics", "bioinformatics", "life sciences", "wet lab"]):
        return "Biotech"
    if _contains_any(lower, ["healthcare", "hospital", "health system", "health systems", "clinical", "physician", "doctor", "patient", "ehr", "epic", "cmio", "hipaa", "phi", "prior authorization", "radiology", "radiologist", "imaging"]):
        return "Healthcare AI"
    return "Unknown"


def _infer_market_keywords(product: str, sector: str) -> list[str]:
    if "radiology report generation" in product:
        return ["radiology AI reporting", "AI radiology workflow", "imaging report generation"]
    if sector == "LegalTech" or "contract review" in product:
        return ["AI contract review software", "legal AI contract redlining", "CLM contract analytics"]
    if "corporate cards" in product:
        return ["corporate spend management", "expense management software", "procurement automation"]
    if sector == "ClimateTech":
        return ["climate software carbon accounting", "decarbonization software", "energy management software"]
    if sector == "HRTech":
        return ["HR software talent management", "recruiting automation", "workforce management software"]
    if sector == "Supply Chain":
        return ["supply chain software", "logistics automation", "inventory planning software"]
    if sector == "Robotics":
        return ["robotics automation", "warehouse robotics", "industrial automation robotics"]
    if sector == "Biotech":
        return ["biotech drug discovery platform", "life sciences AI platform", "clinical development biotech"]
    if "defense technology" in product:
        return ["defense technology autonomous systems", "defense software procurement", "military drones sensors"]
    if "ambient clinical documentation" in product:
        return ["ambient clinical documentation", "AI medical scribe", "clinical documentation automation"]
    if "prior authorization" in product:
        return ["prior authorization automation", "utilization management automation"]
    if "revenue cycle" in product:
        return ["healthcare revenue cycle automation", "RCM AI automation"]
    if sector == "Healthcare AI":
        return ["healthcare AI workflow automation"]
    return [sector.lower()]


def _infer_business_model(product: str, sector: str) -> str:
    if sector == "LegalTech":
        return "Enterprise legal software subscription sold to legal teams, legal operations, and General Counsel"
    if sector == "Fintech / Spend Management":
        return "B2B financial software with potential card interchange, SaaS, bill pay, and procurement monetization"
    if sector == "ClimateTech":
        return "Enterprise SaaS, project finance, usage-based, or services revenue tied to emissions, energy, and decarbonization outcomes"
    if sector == "HRTech":
        return "B2B HR software subscription sold per employee, seat, hiring workflow, or payroll/benefits module"
    if sector == "Supply Chain":
        return "Enterprise supply-chain software subscription, transaction-based logistics revenue, or usage-based planning/visibility platform"
    if sector == "Robotics":
        return "Hardware sales, robotics-as-a-service, software subscriptions, maintenance, and deployment services"
    if sector == "Biotech":
        return "Biotech R&D model with partnerships, milestones, licensing, grants, or therapeutic pipeline economics"
    if sector == "Defense Technology":
        return "Government contracting, defense programs, hardware/software systems, and mission software revenue"
    if sector == "Healthcare AI":
        return "Enterprise healthcare software or workflow automation sold to health systems, providers, payers, or life sciences buyers"
    return "Unknown"


def _subindustry_for(product: str, sector: str) -> str:
    if "radiology report generation" in product:
        return "Radiology Workflow"
    if sector == "LegalTech":
        return "AI Contract Review"
    if sector == "Fintech / Spend Management":
        return "Spend Management"
    if sector == "ClimateTech":
        return "Climate Software"
    if sector == "HRTech":
        return "Workforce / Talent Software"
    if sector == "Supply Chain":
        return "Supply Chain Software"
    if sector == "Robotics":
        return "Physical Automation"
    if sector == "Biotech":
        return "Life Sciences Platform"
    if sector == "Defense Technology":
        return "Autonomous Defense Systems"
    if "ambient clinical documentation" in product:
        return "Clinical Documentation"
    if "prior authorization" in product:
        return "Prior Authorization"
    if "revenue cycle" in product:
        return "Revenue Cycle Automation"
    if sector == "Healthcare AI":
        return "Healthcare Workflow Automation"
    return "Unknown"


def _critical_dependencies(product: str, sector: str) -> list[str]:
    if "radiology report generation" in product:
        return ["PACS integrations", "RIS integrations", "EHR integrations", "Radiologist workflows", "Clinical governance", "HIPAA/security compliance"]
    if sector == "LegalTech":
        return ["Contract repositories", "CLM integrations", "Document management systems", "Legal review workflows", "Enterprise security review"]
    if sector == "Fintech / Spend Management":
        return ["Banking partners", "Card networks", "Issuer processors", "ERP/accounting integrations", "Fraud and underwriting systems"]
    if sector == "ClimateTech":
        return ["Emissions data sources", "ERP/utility integrations", "Regulatory frameworks", "Carbon accounting standards", "Customer sustainability workflows"]
    if sector == "HRTech":
        return ["HRIS integrations", "Payroll systems", "ATS integrations", "Employee data security", "Change management"]
    if sector == "Supply Chain":
        return ["ERP integrations", "WMS/TMS integrations", "Supplier data", "Carrier networks", "Forecasting data quality"]
    if sector == "Robotics":
        return ["Hardware supply chain", "Manufacturing capacity", "Fleet management software", "Site deployment", "Maintenance operations"]
    if sector == "Biotech":
        return ["Scientific validation", "Wet-lab capacity", "Clinical/regulatory pathway", "IP portfolio", "Pharma partnerships"]
    if sector == "Defense Technology":
        return ["Government procurement channels", "Defense program sponsors", "Hardware supply chain", "Security clearances", "Export-control compliance"]
    if "ambient clinical documentation" in product:
        return ["EHR integrations", "Epic ecosystem", "Clinical workflows", "Provider adoption", "HIPAA/security compliance"]
    if "prior authorization" in product:
        return ["Payer/provider integrations", "Clinical policy rules", "Utilization management workflows", "Regulatory compliance"]
    if sector == "Healthcare AI":
        return ["Healthcare workflow integrations", "Security/compliance", "Customer change management"]
    return ["Customer integrations", "Distribution channels", "Implementation capacity"]


def _revenue_drivers(product: str, sector: str) -> list[str]:
    if "radiology report generation" in product:
        return ["Radiologist adoption", "Imaging volume", "Health-system deployments", "PACS/RIS integration depth", "Renewals"]
    if sector == "LegalTech":
        return ["Legal seat adoption", "Matter or contract volume", "Enterprise expansion", "Workflow integrations", "Renewals"]
    if sector == "Fintech / Spend Management":
        return ["Card spend", "Software attach", "Bill pay volume", "Procurement module adoption", "Customer retention"]
    if sector == "ClimateTech":
        return ["Enterprise deployments", "Emissions data volume", "Energy savings", "Regulatory reporting demand", "Expansion across sites/business units"]
    if sector == "HRTech":
        return ["Employee seats", "Hiring volume", "Payroll/benefits attach", "Expansion across HR modules", "Retention"]
    if sector == "Supply Chain":
        return ["Shipment volume", "Planner seats", "Supplier/carrier network scale", "Inventory under management", "Expansion across facilities"]
    if sector == "Robotics":
        return ["Robots deployed", "Utilization", "RaaS subscriptions", "Maintenance attach", "Site expansions"]
    if sector == "Biotech":
        return ["Partnership milestones", "Pipeline progression", "Platform licensing", "Grant/non-dilutive funding", "Clinical trial progress"]
    if sector == "Defense Technology":
        return ["Awarded contracts", "Program expansion", "Hardware deployments", "Software subscriptions", "Maintenance/support"]
    if "ambient clinical documentation" in product:
        return ["Provider adoption", "Health-system deployments", "Seat/provider expansion", "EHR integration depth", "Renewals"]
    if "prior authorization" in product:
        return ["Authorization volume", "Payer/provider contracts", "Automation rate", "Clinical review savings"]
    if sector == "Healthcare AI":
        return ["Enterprise contracts", "Usage volume", "Workflow expansion", "Renewals"]
    return ["ARR", "Expansion revenue", "Retention"]


def _budget_owner_for(product: str, sector: str) -> str:
    if "radiology report generation" in product:
        return "Radiology Chair / CMIO / CIO"
    if sector == "LegalTech":
        return "General Counsel / Legal Operations"
    if sector == "Fintech / Spend Management":
        return "CFO"
    if sector == "ClimateTech":
        return "CSO / CFO / Operations / Energy Procurement"
    if sector == "HRTech":
        return "CHRO / People Operations / Talent Acquisition"
    if sector == "Supply Chain":
        return "Chief Supply Chain Officer / COO / Procurement Leader"
    if sector == "Robotics":
        return "COO / Operations / Manufacturing or Warehouse Leader"
    if sector == "Biotech":
        return "Pharma BD / R&D Leader / Biotech Executive Team"
    if sector == "Defense Technology":
        return "Defense program sponsor / procurement officer"
    if "ambient clinical documentation" in product:
        return "CMIO / CIO / Clinical Operations"
    if "prior authorization" in product:
        return "Chief Medical Officer / Utilization Management / Revenue Cycle"
    if "revenue cycle" in product:
        return "CFO / Revenue Cycle Leader"
    if sector == "Healthcare AI":
        return "CIO / Operations Leader"
    return "Unknown"


def _competitor_framework(product: str, customers: list[str] | str) -> dict[str, Any]:
    customer_text = ", ".join(customers) if isinstance(customers, list) else str(customers)
    if "radiology report generation" in product:
        return {
            "primary_workflow": "radiology report generation, imaging interpretation workflow, and diagnostic documentation",
            "primary_buyer": customer_text if customer_text != "not specified" else "radiology groups, imaging centers, and health systems",
            "product_category": "radiology workflow AI",
            "known_competitors": ["Rad AI", "Aidoc", "Viz.ai", "DeepHealth", "Qure.ai", "Gleamer"],
            "competitor_groups": {
                "Direct Competitors": ["Rad AI", "DeepHealth", "Qure.ai", "Gleamer"],
                "Adjacent Competitors": ["Aidoc", "Viz.ai"],
                "Incumbents": ["PACS vendors", "RIS vendors", "Nuance radiology reporting"],
            },
            "exclusions": ["generic AI workflow tools", "non-clinical documentation vendors"],
        }
    if "contract review" in product:
        return {
            "primary_workflow": "contract review, redlining, clause extraction, and legal risk detection",
            "primary_buyer": customer_text if customer_text != "not specified" else "legal teams and legal operations",
            "product_category": "AI contract review software",
            "known_competitors": ["Ironclad", "Harvey", "Lexion", "LinkSquares", "Evisort", "Spellbook"],
            "competitor_groups": {
                "Direct Competitors": ["Ironclad", "Harvey", "Lexion", "LinkSquares", "Evisort", "Spellbook"],
                "Adjacent Competitors": ["DocuSign CLM", "Sirion"],
                "Incumbents": ["Microsoft Word / Office legal workflows", "legacy CLM suites"],
            },
            "exclusions": ["generic workflow tools", "healthcare workflow vendors", "IT automation platforms"],
        }
    if "legal workflow platform" in product:
        return {
            "primary_workflow": "legal workflow automation, legal research, drafting, and professional services work",
            "primary_buyer": customer_text if customer_text != "not specified" else "law firms and legal professionals",
            "product_category": "AI legal workflow platform",
            "known_competitors": ["Thomson Reuters CoCounsel", "Legora", "Spellbook", "Eve", "Lexion", "Ironclad"],
            "competitor_groups": {
                "Direct Competitors": ["Thomson Reuters CoCounsel", "Legora", "Spellbook", "Eve"],
                "Adjacent Competitors": ["Lexion", "Ironclad"],
                "Incumbents": ["Thomson Reuters", "LexisNexis", "Microsoft legal workflows"],
            },
            "exclusions": ["healthcare workflow vendors", "generic AI workflow tools"],
        }
    if "corporate cards" in product:
        return {
            "primary_workflow": "corporate spend management, expense control, bill pay, and procurement",
            "primary_buyer": customer_text if customer_text != "not specified" else "CFOs and finance teams",
            "product_category": "corporate spend management software",
            "known_competitors": ["Brex", "Airbase", "Navan", "BILL", "Mercury", "American Express", "SAP Concur", "Coupa"],
            "competitor_groups": {
                "Direct Competitors": ["Brex", "Airbase", "Navan"],
                "Adjacent Competitors": ["BILL", "Mercury"],
                "Incumbents": ["American Express", "SAP Concur", "Coupa"],
            },
            "exclusions": ["healthcare workflow automation vendors"],
        }
    if "climate software" in product:
        return {
            "primary_workflow": "carbon accounting, emissions reporting, energy management, and decarbonization planning",
            "primary_buyer": customer_text if customer_text != "not specified" else "sustainability, finance, operations, and energy teams",
            "product_category": "climate software",
            "known_competitors": ["Watershed", "Persefoni", "Sweep", "Plan A", "CarbonChain", "SINAI"],
            "competitor_groups": {
                "Direct Competitors": ["Watershed", "Persefoni", "Sweep", "Plan A"],
                "Adjacent Competitors": ["CarbonChain", "SINAI"],
                "Incumbents": ["Sphera", "Workiva", "Microsoft Cloud for Sustainability", "SAP Sustainability"],
            },
            "exclusions": ["generic ESG content sites", "unrelated renewable developers"],
        }
    if "HR software" in product:
        return {
            "primary_workflow": "talent acquisition, workforce management, payroll, benefits, and employee engagement",
            "primary_buyer": customer_text if customer_text != "not specified" else "HR, people operations, talent acquisition, and payroll teams",
            "product_category": "HR technology",
            "known_competitors": ["Workday", "Rippling", "Gusto", "Greenhouse", "Lever", "BambooHR"],
            "competitor_groups": {
                "Direct Competitors": ["Rippling", "Gusto", "BambooHR"],
                "Adjacent Competitors": ["Greenhouse", "Lever"],
                "Incumbents": ["Workday", "ADP", "SAP SuccessFactors", "Oracle HCM"],
            },
            "exclusions": ["generic productivity tools"],
        }
    if "supply chain software" in product:
        return {
            "primary_workflow": "logistics, inventory planning, procurement, supplier management, and supply-chain visibility",
            "primary_buyer": customer_text if customer_text != "not specified" else "supply chain, logistics, procurement, and operations leaders",
            "product_category": "supply chain software",
            "known_competitors": ["Flexport", "Project44", "FourKites", "Kinaxis", "Anaplan", "Blue Yonder"],
            "competitor_groups": {
                "Direct Competitors": ["Project44", "FourKites", "Kinaxis", "Blue Yonder"],
                "Adjacent Competitors": ["Flexport", "Anaplan"],
                "Incumbents": ["SAP IBP", "Oracle SCM", "Manhattan Associates"],
            },
            "exclusions": ["generic workflow tools"],
        }
    if "robotics" in product:
        return {
            "primary_workflow": "physical operations automation, robotics deployment, fleet operations, and site productivity",
            "primary_buyer": customer_text if customer_text != "not specified" else "operations, manufacturing, warehouse, and facilities leaders",
            "product_category": "robotics and physical automation",
            "known_competitors": ["Locus Robotics", "AutoStore", "Symbotic", "Figure AI", "Agility Robotics", "Boston Dynamics"],
            "competitor_groups": {
                "Direct Competitors": ["Locus Robotics", "AutoStore", "Symbotic"],
                "Adjacent Competitors": ["Figure AI", "Agility Robotics"],
                "Incumbents": ["ABB Robotics", "Fanuc", "KUKA", "Boston Dynamics"],
            },
            "exclusions": ["pure software workflow vendors"],
        }
    if "biotech platform" in product:
        return {
            "primary_workflow": "drug discovery, life sciences R&D, clinical development, and biotech platform validation",
            "primary_buyer": customer_text if customer_text != "not specified" else "pharma BD, R&D leaders, biotech teams, and life sciences investors",
            "product_category": "biotech platform",
            "known_competitors": ["Recursion", "Schrodinger", "Exscientia", "Insilico Medicine", "Generate Biomedicines", "Benchling"],
            "competitor_groups": {
                "Direct Competitors": ["Recursion", "Schrodinger", "Exscientia", "Insilico Medicine"],
                "Adjacent Competitors": ["Generate Biomedicines", "Benchling"],
                "Incumbents": ["Large pharma internal R&D", "CROs", "Academic labs"],
            },
            "exclusions": ["generic AI tools without scientific validation"],
        }
    if "defense technology" in product:
        return {
            "primary_workflow": "autonomous defense systems, sensors, and mission software",
            "primary_buyer": customer_text if customer_text != "not specified" else "defense agencies and government buyers",
            "product_category": "defense technology platforms",
            "known_competitors": ["Palantir", "Shield AI", "General Atomics", "Lockheed Martin", "Northrop Grumman"],
            "competitor_groups": {
                "Direct Competitors": ["Shield AI", "General Atomics"],
                "Adjacent Competitors": ["Palantir"],
                "Incumbents": ["Lockheed Martin", "Northrop Grumman"],
            },
            "exclusions": ["commercial fintech or healthcare workflow vendors"],
        }
    if "ambient clinical documentation" in product:
        return {
            "primary_workflow": "ambient clinical documentation and AI medical scribing",
            "primary_buyer": customer_text if customer_text != "not specified" else "health systems and clinicians",
            "product_category": "ambient clinical AI documentation",
            "known_competitors": ["Nuance DAX", "Suki", "Nabla", "DeepScribe", "Augmedix", "Ambience", "Epic documentation tools", "Oracle Cerner solutions"],
            "competitor_groups": {
                "Direct Competitors": ["Nuance DAX", "Suki", "Nabla", "DeepScribe"],
                "Adjacent Competitors": ["Augmedix", "Ambience"],
                "Incumbents": ["Epic documentation tools", "Oracle Cerner solutions"],
            },
            "exclusions": ["prior authorization software", "utilization management vendors"],
        }
    if "prior authorization" in product:
        return {
            "primary_workflow": "prior authorization and utilization management automation",
            "primary_buyer": customer_text if customer_text != "not specified" else "health plans, providers, and revenue cycle teams",
            "product_category": "prior authorization automation",
            "known_competitors": ["Cohere Health", "Rhyme", "Availity", "Waystar"],
            "competitor_groups": {
                "Direct Competitors": ["Cohere Health", "Rhyme"],
                "Adjacent Competitors": ["Availity"],
                "Incumbents": ["Waystar"],
            },
            "exclusions": ["ambient clinical documentation vendors"],
        }
    if "revenue cycle" in product:
        return {
            "primary_workflow": "revenue cycle automation",
            "primary_buyer": customer_text if customer_text != "not specified" else "provider revenue cycle teams",
            "product_category": "healthcare revenue cycle automation",
            "known_competitors": ["AKASA", "Waystar", "R1", "Notable"],
            "competitor_groups": {
                "Direct Competitors": ["AKASA", "Notable"],
                "Adjacent Competitors": ["Waystar"],
                "Incumbents": ["R1"],
            },
            "exclusions": ["ambient clinical documentation vendors"],
        }
    return {
        "primary_workflow": "Unknown" if _is_placeholder_entity(product) else product,
        "primary_buyer": customer_text if customer_text != "not specified" else "Unknown",
        "product_category": "Unknown" if _is_placeholder_entity(product) else product,
        "known_competitors": [],
        "competitor_groups": {},
        "exclusions": [],
    }


KNOWN_COMPETITOR_NAMES = [
    "Ironclad",
    "Harvey",
    "Lexion",
    "LinkSquares",
    "Evisort",
    "Spellbook",
    "Brex",
    "Airbase",
    "Navan",
    "BILL",
    "Mercury",
    "American Express",
    "SAP Concur",
    "Coupa",
    "Nuance DAX",
    "Suki",
    "Nabla",
    "DeepScribe",
    "Augmedix",
    "Ambience",
]


COMPANY_SUFFIX_TOKENS = {
    "ai",
    "technologies",
    "technology",
    "tech",
    "labs",
    "lab",
    "systems",
    "system",
    "software",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "llc",
    "plc",
    "com",
    "io",
}


def _normalized_company_name(name: str) -> str:
    lower = str(name or "").lower().strip()
    lower = re.sub(r"https?://", "", lower)
    lower = re.sub(r"^www\.", "", lower)
    lower = re.sub(r"\.[a-z]{2,4}\b", " ", lower)
    tokens = [token for token in re.split(r"[^a-z0-9]+", lower) if token]
    meaningful = [token for token in tokens if token not in COMPANY_SUFFIX_TOKENS]
    return " ".join(meaningful or tokens)


def _is_target_company_alias(candidate: str, target_company: str) -> bool:
    candidate_key = _normalized_company_name(candidate)
    target_key = _normalized_company_name(target_company)
    return bool(candidate_key and target_key and candidate_key == target_key)


def _filter_target_company_aliases(candidates: list[str], target_company: str) -> list[str]:
    filtered = []
    seen = set()
    for candidate in candidates:
        if _is_target_company_alias(candidate, target_company):
            continue
        key = _normalized_company_name(candidate)
        if key in seen:
            continue
        seen.add(key)
        filtered.append(candidate)
    return filtered


def _filter_competitor_framework(framework: dict[str, Any], target_company: str) -> dict[str, Any]:
    if not target_company:
        return framework
    groups = {
        group_name: _filter_target_company_aliases(list(names), target_company)
        for group_name, names in framework.get("competitor_groups", {}).items()
    }
    groups = {group_name: names for group_name, names in groups.items() if names}
    return {
        **framework,
        "known_competitors": _filter_target_company_aliases(list(framework.get("known_competitors", [])), target_company),
        "competitor_groups": groups,
    }


def _competitors_mentioned(raw_notes: str, target_company: str = "") -> list[str]:
    lower = raw_notes.lower()
    mentioned = [name for name in KNOWN_COMPETITOR_NAMES if name.lower() in lower]
    return _filter_target_company_aliases(mentioned, target_company) if target_company else mentioned


POSITIONING_SIGNAL_TERMS = {
    "Cybersecurity": ["cybersecurity", "cyber security", "security software", "security platform", "threat", "soc", "vulnerability"],
    "Workflow automation": ["workflow automation", "automation platform", "process automation"],
    "Compliance software": ["compliance software", "compliance", "grc", "audit management", "controls"],
    "Healthcare AI": ["healthcare", "clinical", "health system", "health systems", "hipaa", "phi", "cmio", "ehr"],
    "LegalTech": ["contract review", "legal team", "legal teams", "general counsel", "clm"],
    "Fintech / Spend Management": ["corporate card", "spend management", "expense management", "interchange"],
    "ClimateTech": ["climate tech", "climatetech", "carbon accounting", "emissions", "decarbonization"],
    "HRTech": ["hrtech", "human resources", "talent acquisition", "payroll", "employee engagement"],
    "Supply Chain": ["supply chain", "logistics", "freight", "inventory", "warehouse"],
    "Robotics": ["robotics", "robots", "warehouse automation", "industrial automation"],
    "Biotech": ["biotech", "drug discovery", "therapeutics", "clinical trial", "genomics"],
    "Defense Technology": ["defense", "military", "drones", "sensors", "border security"],
}


def _positioning_signals(raw_notes: str) -> list[dict[str, str]]:
    lower = raw_notes.lower()
    signals = []
    for label, terms in POSITIONING_SIGNAL_TERMS.items():
        matched = [term for term in terms if term in lower]
        if matched:
            signals.append({"interpretation": label, "evidence": ", ".join(matched[:3])})
    return signals


def _positioning_contradictions(raw_notes: str) -> dict[str, Any]:
    signals = _positioning_signals(raw_notes)
    signal_names = {signal["interpretation"] for signal in signals}
    if {"Cybersecurity", "Workflow automation", "Compliance software"}.issubset(signal_names):
        severity = "high"
    elif len(signals) == 2:
        severity = "medium"
    elif len(signals) >= 3:
        severity = "medium"
    else:
        severity = "low"
    return {
        "has_contradiction": severity in {"medium", "high"},
        "severity": severity,
        "interpretations": signals,
        "recommendation": "Clarify positioning before workflow-level analysis." if severity == "high" else "Resolve conflicting positioning signals before high-conviction analysis.",
    }


def _understanding_confidence(understanding: dict[str, Any]) -> float:
    product = str(understanding.get("primary_product", ""))
    workflow = str(understanding.get("core_workflow", ""))
    buyer = str(understanding.get("target_buyer", ""))
    category = str(understanding.get("category", ""))
    risks = understanding.get("diligence_topics", [])
    weights = {
        "product_clarity": 0.25,
        "workflow_clarity": 0.25,
        "buyer_clarity": 0.20,
        "category_clarity": 0.20,
        "risk_clarity": 0.10,
    }
    scores = {
        "product_clarity": _specificity_score(product, generic_terms=["ai platform", "ai workflow automation", "platform", "software"]),
        "workflow_clarity": _specificity_score(workflow, generic_terms=["ai workflow automation", "workflow automation", "workflow", "platform"]),
        "buyer_clarity": _specificity_score(buyer, generic_terms=["enterprises", "enterprise customers", "companies", "customers", "fortune 500 companies"]),
        "category_clarity": 0.0 if category in {"", "Unknown", "Conflicting signals detected"} else 1.0,
        "risk_clarity": _risk_specificity_score(risks),
    }
    return round(sum(scores[key] * weights[key] for key in weights), 2)


def _is_placeholder_entity(value: str) -> bool:
    lower = str(value).lower().strip()
    if not lower:
        return True
    exact_placeholders = {
        "not specified",
        "unknown",
        "clarification required",
        "business model requires diligence",
        "economic buyer requires diligence",
    }
    if lower in exact_placeholders:
        return True
    return "requires diligence" in lower


def _specificity_score(value: str, generic_terms: list[str]) -> float:
    lower = value.lower().strip()
    if _is_placeholder_entity(lower):
        return 0.0
    if any(term == lower for term in generic_terms):
        return 0.15
    if any(term in lower for term in ["better decisions", "ai platform", "workflow automation"]) and not any(
        marker in lower
        for marker in [
            "contract review",
            "redlining",
            "radiology",
            "clinical documentation",
            "prior authorization",
            "expense management",
            "spend management",
            "corporate cards",
            "defense",
        ]
    ):
        return 0.2
    if len([token for token in re.split(r"[^a-z0-9]+", lower) if token]) <= 2:
        return 0.35
    return 1.0


def _risk_specificity_score(risks: list[str]) -> float:
    if not risks:
        return 0.0
    generic = {"ARR", "retention", "gross margin", "customer ROI", "competitive differentiation", "implementation complexity"}
    specific = [risk for risk in risks if risk not in generic]
    if not specific:
        return 0.2
    return 1.0


def _validate_company_understanding(understanding: dict[str, Any], raw_notes: str) -> dict[str, Any]:
    lower = raw_notes.lower()
    failures = []
    warnings = []
    contradictions = _positioning_contradictions(raw_notes)
    if understanding.get("category") in {"", "Unknown"}:
        failures.append("Category is unknown.")
    if _is_placeholder_entity(str(understanding.get("target_buyer", ""))):
        failures.append("Target buyer is unknown.")
    if _is_placeholder_entity(str(understanding.get("core_workflow", ""))):
        failures.append("Primary workflow is unknown.")
    if "legal team" in lower and "legal" not in str(understanding.get("target_buyer", "")).lower():
        failures.append("Target buyer parsing failed: notes mention legal teams.")
    if "contract review" in lower and "contract review" not in str(understanding.get("core_workflow", "")).lower():
        failures.append("Workflow parsing failed: notes mention contract review.")
    if "redlining" in lower and "redlining" not in str(understanding.get("core_workflow", "")).lower():
        failures.append("Workflow parsing failed: notes mention redlining.")
    if "hipaa" in str(understanding).lower() and understanding.get("category") != "Healthcare AI":
        warnings.append("Healthcare-specific terms appeared in a non-healthcare company understanding.")
    confidence = _understanding_confidence(understanding)
    if contradictions["severity"] == "high":
        failures.append("Conflicting positioning signals detected.")
        confidence = min(confidence, 0.3)
    elif contradictions["severity"] == "medium":
        warnings.append("Multiple positioning signals detected; treat category and workflow as provisional.")
    return {
        "confidence": confidence,
        "is_valid": not failures and confidence >= 0.8,
        "failures": failures,
        "warnings": warnings,
        "contradictions": contradictions,
    }


def _company_understanding(profile: dict[str, Any]) -> dict[str, Any]:
    target_company = profile.get("raw_company_name") or profile.get("company", "")
    framework = _filter_competitor_framework(
        _competitor_framework(profile.get("product", ""), profile.get("customers", [])),
        target_company,
    )
    product = profile.get("product", "Not specified")
    product_label = "Unknown" if _is_placeholder_entity(product) else product
    sector = profile.get("sector", "Unknown")
    raw_notes = profile.get("raw_notes", "")
    product_scope, product_scope_confidence = _product_scope_from_text(raw_notes, product_label)
    product_use_cases = list(
        dict.fromkeys(_extract_use_cases_from_text(raw_notes) + profile.get("public_product_use_cases", []))
    )
    mentioned = _filter_target_company_aliases(profile.get("competitors_mentioned_in_notes", []), target_company)
    diligence_topics = _diligence_topics_for_profile(profile)
    understanding = {
        "company_name": profile.get("company", "Unknown company"),
        "category": sector,
        "industry": sector,
        "subindustry": _subindustry_for(product, sector),
        "primary_product": product_label,
        "primary_product_scope": product_scope,
        "product_scope_confidence": product_scope_confidence,
        "product_use_cases": product_use_cases,
        "product_evidence_conflicts": profile.get("product_evidence_conflicts", []),
        "target_buyer": framework["primary_buyer"],
        "budget_owner": _budget_owner_for(product, sector),
        "business_model": _infer_business_model(product, sector),
        "business_model_hypothesis": _infer_business_model(product, sector),
        "critical_dependencies": _critical_dependencies(product, sector),
        "revenue_drivers": _revenue_drivers(product, sector),
        "competitors": framework["known_competitors"],
        "competitors_mentioned_in_notes": mentioned,
        "core_workflow": framework["primary_workflow"],
        "product_category": framework["product_category"],
        "diligence_topics": diligence_topics,
        "key_diligence_questions": diligence_topics[:5],
        "field_sources": {
            "category": {"source": "Notes", "confidence": "High" if sector != "Unknown" else "Low"},
            "target_buyer": {"source": "Notes", "confidence": "High" if not _is_placeholder_entity(framework["primary_buyer"]) else "Low"},
            "core_workflow": {"source": "Notes", "confidence": "High" if not _is_placeholder_entity(framework["primary_workflow"]) else "Low"},
            "competitors": {"source": "Notes" if mentioned else "Profile-derived", "confidence": "High" if mentioned else "Medium"},
            "primary_product_scope": {"source": "Notes / Public Evidence", "confidence": product_scope_confidence.capitalize()},
            "product_scope_confidence": {"source": "Rules", "confidence": product_scope_confidence.capitalize()},
            "product_use_cases": {"source": "Notes / Public Evidence", "confidence": "Medium" if product_use_cases else "Low"},
            "product_evidence_conflicts": {"source": "Validation", "confidence": "High" if profile.get("product_evidence_conflicts") else "Low"},
        },
    }
    understanding["validation"] = _validate_company_understanding(understanding, profile.get("raw_notes", ""))
    if understanding["validation"].get("contradictions", {}).get("severity") == "high":
        understanding.update(
            {
                "category": "Conflicting signals detected",
                "industry": "Conflicting signals detected",
                "subindustry": "Clarification required",
                "primary_product": "Clarification required",
                "primary_product_scope": "unclear",
                "product_scope_confidence": "low",
                "product_use_cases": [],
                "product_evidence_conflicts": ["Partner notes contain conflicting positioning signals; do not narrow product until clarified."],
                "target_buyer": "Clarification required",
                "budget_owner": "Clarification required",
                "business_model": "Clarification required",
                "business_model_hypothesis": "Clarification required",
                "critical_dependencies": [],
                "revenue_drivers": [],
                "competitors": [],
                "core_workflow": "Clarification required",
                "product_category": "Clarification required",
                "diligence_topics": ["Clarify whether positioning is cybersecurity, workflow automation, compliance software, or another category."],
                "key_diligence_questions": ["Clarify actual product, buyer, workflow, and competitive category before diligence analysis."],
            }
        )
    return understanding


def _diligence_topics_for_profile(profile: dict[str, Any]) -> list[str]:
    sector = profile.get("sector", "Unknown")
    target_company = profile.get("raw_company_name") or profile.get("company", "")
    if sector == "LegalTech":
        legal_competitors = _filter_target_company_aliases(
            ["Ironclad", "Harvey", "Evisort", "LinkSquares", "Lexion", "Spellbook"],
            target_company,
        )
        return [
            "ARR, growth rate, and legal seat adoption",
            "contract volume and expansion by legal team",
            "accuracy and hallucination rates in clause interpretation",
            "enterprise security and confidentiality posture",
            "CLM/document repository integrations",
            f"win/loss versus {', '.join(legal_competitors)}" if legal_competitors else "win/loss versus relevant legal workflow competitors",
            "gross margin after model and services costs",
        ]
    if sector == "Fintech / Spend Management":
        return [
            "ARR and software attach rate",
            "interchange dependence",
            "net revenue retention",
            "gross margin by product line",
            "fraud and credit losses",
            "ERP/procurement integrations",
            "customer payback and finance-team ROI",
        ]
    if sector == "ClimateTech":
        return [
            "enterprise deployments and site expansion",
            "emissions data quality and auditability",
            "regulatory reporting requirements",
            "customer ROI from energy savings or compliance",
            "competition versus ESG and sustainability platforms",
            "gross margin after implementation/services",
        ]
    if sector == "HRTech":
        return [
            "ARR, retention, and employee-seat expansion",
            "HRIS/ATS/payroll integration depth",
            "candidate or employee data privacy",
            "measurable recruiting, payroll, or engagement ROI",
            "competition versus HR suites and point solutions",
            "gross margin and services burden",
        ]
    if sector == "Supply Chain":
        return [
            "ARR, shipment volume, and facilities deployed",
            "ERP/WMS/TMS integration depth",
            "forecast accuracy and inventory impact",
            "supplier or carrier network quality",
            "competition versus SCM incumbents",
            "implementation timeline and services burden",
        ]
    if sector == "Robotics":
        return [
            "robots deployed and utilization",
            "site-level ROI and labor savings",
            "hardware reliability and maintenance burden",
            "manufacturing scale and supply chain",
            "deployment timeline and customer success load",
            "competition versus robotics incumbents",
        ]
    if sector == "Biotech":
        return [
            "scientific validation and reproducibility",
            "pipeline stage and clinical/regulatory path",
            "IP position and data moat",
            "pharma partnerships and milestone economics",
            "cash runway and R&D burn",
            "competition versus platform biotech and pharma internal R&D",
        ]
    if sector == "Defense Technology":
        return [
            "program concentration",
            "contract backlog and quality",
            "gross margin by hardware/software mix",
            "manufacturing scale",
            "government procurement timeline",
            "security clearance and export-control exposure",
            "competitive positioning versus primes",
        ]
    if sector == "Healthcare AI":
        return [
            "ARR and growth rate",
            "live customers versus pilots",
            "customer ROI",
            "workflow integration depth",
            "gross margin after services and model costs",
            "security and compliance posture",
            "win/loss versus workflow competitors",
        ]
    return ["ARR", "retention", "gross margin", "customer ROI", "competitive differentiation", "implementation complexity"]


def extract_company_profile(state: DealMemoState) -> dict[str, Any]:
    raw = state["raw_notes"]
    company = _guess_company(raw)
    sector = _infer_sector_from_text(raw)
    product = _infer_product_from_text(raw)
    customers = []
    if _contains_any(raw, ["hospital", "health system", "health systems"]):
        customers.append("hospital systems")
    if _contains_any(raw, ["specialty"]):
        customers.append("specialty groups")
    if _contains_any(raw, ["physician", "doctor", "provider", "clinician"]):
        customers.append("clinicians/providers")
    if _contains_any(raw, ["cmio", "cio", "clinical operations"]):
        customers.append("CMIO / CIO / clinical operations")
    if _contains_any(raw, ["radiologist", "radiologists", "radiology group", "imaging center", "imaging centers"]):
        customers.append("radiologists and imaging teams")
    if _contains_any(raw, ["cfo", "finance team", "finance teams", "controller", "accounting", "procurement"]):
        customers.append("CFOs and finance teams")
    if _customer_context_contains(raw, ["fortune 500", "large enterprise", "large enterprises", "enterprise customer", "enterprise customers"]):
        customers.append("Enterprise customers")
    if _customer_context_contains(raw, ["mid-market", "mid market"]):
        customers.append("mid-market customers")
    if _customer_context_contains(raw, ["smb", "small business", "small businesses"]):
        customers.append("SMB customers")
    if _customer_context_contains(raw, ["startup", "startups"]):
        customers.append("startup customers")
    if _contains_any(raw, ["government", "defense agencies", "military", "dod", "public sector"]):
        customers.append("government defense agencies")
    if _contains_any(raw, ["legal team", "legal teams", "legal ops", "legal operations", "general counsel"]):
        customers.append("legal teams and legal operations")
    if _contains_any(raw, ["legal professionals", "law firm", "law firms"]):
        customers.append("law firms and legal professionals")
    if _contains_any(raw, ["sustainability", "carbon", "emissions", "energy procurement", "facilities"]):
        customers.append("sustainability, finance, operations, and energy teams")
    if _contains_any(raw, ["chro", "people ops", "people operations", "talent acquisition", "recruiters", "hr teams", "employees"]):
        customers.append("HR, people operations, talent acquisition, and payroll teams")
    if _contains_any(raw, ["supply chain", "logistics", "procurement", "warehouse", "warehouses", "carriers", "suppliers"]):
        customers.append("supply chain, logistics, procurement, and operations leaders")
    if _contains_any(raw, ["manufacturing", "factory", "factories", "robotics", "robots"]):
        customers.append("operations, manufacturing, warehouse, and facilities leaders")
    if _contains_any(raw, ["pharma", "biotech", "r&d", "research teams", "clinical development", "life sciences"]):
        customers.append("pharma BD, R&D leaders, biotech teams, and life sciences investors")

    risks = []
    for label, terms in {
        "integration burden": ["integration"],
        "competitive intensity": ["competitive", "competitor"],
        "self-reported traction": ["self-reported", "pilot"],
        "unclear buyer urgency": ["buyer urgency"],
    }.items():
        if _contains_any(raw, terms):
            risks.append(label)

    open_questions = []
    if company == "Unknown company":
        open_questions.append("What is the company name?")
    if not customers:
        open_questions.append("Who is the buyer and end customer?")
    if "fund" not in raw.lower():
        open_questions.append("What is the latest funding history and investor base?")
    if not risks:
        open_questions.append("What are the main diligence risks?")

    identity_resolution = _resolve_company_identity(company, raw, product, sector, customers)
    resolved_company = identity_resolution["selected_entity"].get("name", company) if identity_resolution["is_resolved"] else company

    profile = {
        "raw_notes": raw,
        "company": resolved_company,
        "raw_company_name": company,
        "identity_resolution": identity_resolution,
        "sector": sector,
        "product": product,
        "customers": customers or ["not specified"],
        "team": "Partner notes mention founder/team background." if _contains_any(raw, ["founder", "team", "executive"]) else "not specified",
        "market_keywords": _infer_market_keywords(product, sector),
        "risks": risks,
        "open_questions": open_questions,
        "competitors_mentioned_in_notes": _competitors_mentioned(raw, company),
        "confidence": "low" if len(raw.split()) < 35 or company == "Unknown company" or not identity_resolution["is_resolved"] else "medium",
    }
    profile["company_understanding"] = _company_understanding(profile)
    validation = profile["company_understanding"]["validation"]
    if validation["confidence"] >= 0.8:
        profile["confidence"] = "high"
    elif validation["confidence"] >= 0.5:
        profile["confidence"] = "medium"
    else:
        profile["confidence"] = "low"
    return {"company_profile": profile, "trace": _trace(state, "extract_company_profile", profile)}


def plan_research(state: DealMemoState) -> dict[str, Any]:
    profile = state["company_profile"]
    identity = profile.get("identity_resolution", {})
    understanding = profile.get("company_understanding", {})
    contradiction = understanding.get("validation", {}).get("contradictions", {})
    if contradiction.get("severity") == "high":
        return {
            "research_plan": [],
            "trace": _trace(
                state,
                "plan_company_research",
                {"queries": [], "blocked": True, "reason": "Conflicting positioning signals detected."},
            ),
        }
    company = profile.get("company", "company")
    product = understanding.get("primary_product", profile.get("product", ""))
    workflow = understanding.get("core_workflow", product)
    buyer = understanding.get("target_buyer", "")
    competitors = " ".join(understanding.get("competitors_mentioned_in_notes", []) or understanding.get("competitors", [])[:4])
    understanding_confidence = float(understanding.get("validation", {}).get("confidence", 0))
    if identity and not identity.get("is_resolved", False):
        if understanding_confidence >= 0.8:
            plan: list[ResearchQuery] = [
                {
                    "query": f"{product} market context {workflow} {buyer}",
                    "evidence_type": "market",
                    "rationale": "Identity is unresolved, so research only workflow-level market context from note-derived understanding.",
                },
                {
                    "query": f"{workflow} competitors {buyer} {competitors}",
                    "evidence_type": "competitor",
                    "rationale": "Identity is unresolved, so identify competitors by same workflow, buyer, and budget rather than company name.",
                },
            ]
            return {
                "research_plan": plan,
                "trace": _trace(
                    state,
                    "plan_company_research",
                    {"queries": plan, "mode": "workflow_level_only", "reason": "Company identity unresolved but note-derived understanding is strong."},
                ),
            }
        return {
            "research_plan": [],
            "trace": _trace(
                state,
                "plan_company_research",
                {"queries": [], "blocked": True, "reason": "Unable to confidently identify company or understand workflow."},
            ),
        }
    plan: list[ResearchQuery] = [
        {"query": f"{company} {product} {buyer}", "evidence_type": "website", "rationale": "Use company-specific sources to define the product, buyer, and workflow."},
        {"query": f"{company} funding investors Crunchbase", "evidence_type": "funding", "rationale": "Find funding history and investor context."},
        {"query": f"{company} news partnerships customers {product}", "evidence_type": "news", "rationale": "Find company-specific commercial momentum."},
        {"query": f"{company} founder leadership team", "evidence_type": "leadership", "rationale": "Verify founder and team background."},
        {"query": f"{company} pricing business model customers", "evidence_type": "business_model", "rationale": "Infer business model only if evidence supports it."},
        {"query": f"{workflow} competitors {competitors}", "evidence_type": "competitor", "rationale": "Validate competitors mentioned in notes against same buyer, workflow, and budget owner."},
    ]
    return {"research_plan": plan, "trace": _trace(state, "plan_company_research", {"queries": plan})}


def validate_company_understanding(state: DealMemoState) -> dict[str, Any]:
    profile = state["company_profile"]
    identity = profile.get("identity_resolution", {})
    understanding = profile.get("company_understanding", {})
    validation = dict(understanding.get("validation", {}))
    query_text = " ".join(query["query"] for query in state.get("research_plan", [])).lower()
    product = str(understanding.get("primary_product", "")).lower()
    workflow = str(understanding.get("core_workflow", "")).lower()
    buyer = str(understanding.get("target_buyer", "")).lower()
    plan_failures = []
    if product and product != "not specified" and not any(term in query_text for term in product.split()[:3]):
        plan_failures.append("Research plan does not appear to include the primary product.")
    workflow_terms = [term for term in re.split(r"[^a-z0-9]+", workflow) if len(term) > 4]
    if workflow_terms and not any(term in query_text for term in workflow_terms[:5]):
        plan_failures.append("Research plan does not appear to include the primary workflow.")
    buyer_terms = [term for term in re.split(r"[^a-z0-9]+", buyer) if len(term) > 4]
    if buyer_terms and not any(term in query_text for term in buyer_terms[:4]):
        plan_failures.append("Research plan does not appear to include the target buyer.")

    identity_resolved = bool(identity.get("is_resolved", True))
    understanding_confidence = float(validation.get("confidence", 0))
    validation["identity_resolution"] = identity
    validation["identity_confidence"] = float(identity.get("confidence", 1.0))
    validation["understanding_confidence"] = understanding_confidence
    validation["research_plan_failures"] = plan_failures
    validation["is_valid"] = bool(validation.get("is_valid")) and not plan_failures
    understanding["validation"] = validation
    profile["company_understanding"] = understanding
    errors = state["errors"]
    if identity and not identity_resolved and understanding_confidence < 0.8:
        errors = [*errors, "Unable to confidently identify company or understand workflow."]
    elif understanding_confidence < 0.8:
        errors = [*errors, "Insufficient company understanding. Need clarification before generating investment thesis."]
    return {
        "company_profile": profile,
        "errors": errors,
        "trace": _trace(state, "validate_company_understanding", validation),
    }


def plan_context_research(state: DealMemoState) -> dict[str, Any]:
    profile = state["company_profile"]
    identity = profile.get("identity_resolution", {})
    understanding = profile.get("company_understanding", {})
    contradiction = understanding.get("validation", {}).get("contradictions", {})
    if contradiction.get("severity") == "high":
        return {
            "research_plan": [],
            "trace": _trace(
                state,
                "plan_context_research",
                {"queries": [], "blocked": True, "reason": "Conflicting positioning signals detected."},
            ),
        }
    company = profile.get("company", "company")
    product = understanding.get("primary_product", profile.get("product", ""))
    framework = _filter_competitor_framework(
        _competitor_framework(product, understanding.get("target_buyer", profile.get("customers", []))),
        profile.get("raw_company_name") or profile.get("company", ""),
    )
    workflow = understanding.get("core_workflow") or " ".join(profile.get("market_keywords") or _infer_market_keywords(product, profile.get("sector", "Unknown")))
    competitor_targets = " ".join(framework["known_competitors"])
    understanding_confidence = float(understanding.get("validation", {}).get("confidence", 0))
    if identity and not identity.get("is_resolved", False):
        if understanding_confidence < 0.8:
            return {
                "research_plan": [],
                "trace": _trace(
                    state,
                    "plan_context_research",
                    {"queries": [], "blocked": True, "reason": "Unable to confidently identify company or understand workflow."},
                ),
            }
        plan: list[ResearchQuery] = [
            {
                "query": f"{product} market context {workflow} {understanding.get('target_buyer', '')}",
                "evidence_type": "market",
                "rationale": "Use note-derived workflow context without company-specific claims because identity is unresolved.",
            },
            {
                "query": f"{framework['primary_workflow']} competitors {framework['product_category']} {competitor_targets}",
                "evidence_type": "competitor",
                "rationale": "Map workflow competitors without relying on unresolved company identity.",
            },
        ]
        return {
            "research_plan": plan,
            "trace": _trace(
                state,
                "plan_context_research",
                {
                    "mode": "workflow_level_only",
                    "primary_product": product,
                    "primary_workflow": framework["primary_workflow"],
                    "primary_buyer": framework["primary_buyer"],
                    "product_category": framework["product_category"],
                    "known_competitors": framework["known_competitors"],
                    "queries": plan,
                },
            ),
        }
    plan: list[ResearchQuery] = [
        {
            "query": f"{company} {product} market context {workflow}",
            "evidence_type": "market",
            "rationale": "Contextualize the already-classified company workflow.",
        },
        {
            "query": f"{company} {framework['primary_workflow']} competitors {framework['product_category']} {competitor_targets}",
            "evidence_type": "competitor",
            "rationale": "Map competitors by same workflow, same buyer, and same product category rather than loose keyword overlap.",
        },
    ]
    return {
        "research_plan": plan,
        "trace": _trace(
            state,
            "plan_context_research",
            {
                "primary_product": product,
                "primary_workflow": framework["primary_workflow"],
                "primary_buyer": framework["primary_buyer"],
                "product_category": framework["product_category"],
                "known_competitors": framework["known_competitors"],
                "excluded_competitor_types": framework["exclusions"],
                "queries": plan,
            },
        ),
    }


def run_research(state: DealMemoState, provider: str = "mock") -> dict[str, Any]:
    try:
        results = run_search(state["research_plan"], provider)
        return {"search_results": results, "errors": state["errors"], "trace": _trace(state, "run_research", {"provider": provider, "result_count": len(results)})}
    except Exception as exc:
        fallback = run_search(state["research_plan"], "mock")
        errors = [*state["errors"], f"{provider} search failed: {exc}. Used mock evidence fallback."]
        return {"search_results": fallback, "errors": errors, "trace": _trace(state, "run_research", {"provider": "mock_fallback", "error": str(exc)})}


def run_context_research(state: DealMemoState, provider: str = "mock") -> dict[str, Any]:
    try:
        results = run_search(state["research_plan"], provider)
        combined = [*state["search_results"], *results]
        return {
            "search_results": combined,
            "errors": state["errors"],
            "trace": _trace(state, "run_context_research", {"provider": provider, "result_count": len(results), "combined_count": len(combined)}),
        }
    except Exception as exc:
        fallback = run_search(state["research_plan"], "mock")
        combined = [*state["search_results"], *fallback]
        errors = [*state["errors"], f"{provider} context search failed: {exc}. Used mock evidence fallback."]
        return {
            "search_results": combined,
            "errors": errors,
            "trace": _trace(state, "run_context_research", {"provider": "mock_fallback", "error": str(exc), "combined_count": len(combined)}),
        }


def build_evidence_store(state: DealMemoState) -> dict[str, Any]:
    evidence = normalize_evidence(state["search_results"])
    return {"evidence": evidence, "trace": _trace(state, "build_evidence_store", {"source_count": len(evidence)})}


def build_final_evidence_store(state: DealMemoState) -> dict[str, Any]:
    evidence = normalize_evidence(state["search_results"])
    return {"evidence": evidence, "trace": _trace(state, "build_final_evidence_store", {"source_count": len(evidence)})}


def refine_company_profile_from_evidence(state: DealMemoState) -> dict[str, Any]:
    profile = dict(state["company_profile"])
    company = profile.get("company", "")
    company_text = _company_specific_evidence_text(state["evidence"], company)
    evidence_product = _infer_product_from_text(company_text)
    note_product = profile.get("product", "Not specified")
    note_scope, _ = _product_scope_from_text(profile.get("raw_notes", ""), note_product)
    public_use_cases = _extract_use_cases_from_text(company_text)
    product_conflicts = _product_evidence_conflicts(note_product, evidence_product, note_scope)

    if evidence_product != "Not specified" and note_scope == "broad_platform" and not _is_placeholder_entity(note_product):
        profile["public_product_use_cases"] = public_use_cases
        profile["product_evidence_conflicts"] = product_conflicts
        profile["market_keywords"] = _infer_market_keywords(note_product, profile.get("sector", "Unknown"))
        profile["product_source_ids"] = _source_ids_for(state["evidence"], "website") + _source_ids_for(state["evidence"], "news", limit=1)
        profile["product_source_basis"] = "partner notes preserve broad platform framing; public evidence treated as use cases"
    elif evidence_product != "Not specified":
        profile["product"] = evidence_product
        profile["public_product_use_cases"] = public_use_cases
        profile["product_evidence_conflicts"] = product_conflicts
        profile["market_keywords"] = _infer_market_keywords(evidence_product, profile.get("sector", "Unknown"))
        profile["product_source_ids"] = _source_ids_for(state["evidence"], "website") + _source_ids_for(state["evidence"], "news", limit=1)
        profile["product_source_basis"] = "company-specific public evidence"
    else:
        profile["public_product_use_cases"] = public_use_cases
        profile["product_evidence_conflicts"] = product_conflicts
        profile["product_source_ids"] = []
        profile["product_source_basis"] = "partner notes only"

    profile["company_understanding"] = _company_understanding(profile)
    trace_payload = {
        "product": profile.get("product"),
        "product_source_basis": profile.get("product_source_basis"),
        "product_source_ids": profile.get("product_source_ids", []),
        "public_product_use_cases": profile.get("public_product_use_cases", []),
        "product_evidence_conflicts": profile.get("product_evidence_conflicts", []),
        "company_understanding": profile["company_understanding"],
    }
    return {"company_profile": profile, "trace": _trace(state, "refine_company_profile_from_evidence", trace_payload)}


DEFAULT_SOURCE_USE_BY_EVIDENCE_TYPE = {
    "website": "product positioning",
    "funding": "funding history",
    "news": "market context",
    "leadership": "preliminary team mapping",
    "market": "market context",
    "competitor": "competitor discovery",
    "business_model": "revenue quality",
}


HIGH_CONVICTION_USE_BY_EVIDENCE_TYPE = {
    "website": "product positioning",
    "funding": "funding history",
    "news": "market context",
    "leadership": "definitive team verification",
    "market": "market conviction",
    "competitor": "market conviction",
    "business_model": "revenue quality",
}


TRACTION_KEYWORDS = [
    "arr",
    "revenue",
    "net revenue retention",
    "nrr",
    "customers",
    "customer count",
    "usage",
    "active users",
    "deployments",
    "live customers",
    "gross margin",
    "retention",
]


def _source_allows_use(item: EvidenceItem, use: str) -> bool:
    source_type = item.get("source_type", "")
    disallowed = {entry.lower() for entry in item.get("disallowed_uses", [])}
    allowed = {entry.lower() for entry in item.get("allowed_uses", [])}
    normalized_use = use.lower()
    if item.get("confidence", 0) < 0.45:
        return normalized_use in {"weak context", "hypothesis generation"}
    if source_type == "generic_blog" and normalized_use not in {"weak context", "hypothesis generation"}:
        return False
    if source_type == "vendor_blog" and normalized_use in {"market conviction", "investment scoring", "investment attractiveness"}:
        return False
    if source_type == "company_owned" and normalized_use in {
        "independent traction verification",
        "traction verification",
        "market leadership",
        "product superiority",
        "market conviction",
        "investment scoring",
    }:
        return False
    if source_type == "funding_database" and normalized_use in {
        "traction verification",
        "customer traction",
        "revenue quality",
        "retention",
        "investment scoring",
    }:
        return _source_mentions_traction_metrics(item)
    if normalized_use in disallowed:
        return False
    if any(normalized_use == denied or normalized_use in denied for denied in disallowed):
        return False
    if normalized_use in allowed:
        return True
    if any(normalized_use in allowed_use or allowed_use in normalized_use for allowed_use in allowed):
        return True
    return source_type in {"independent_news", "funding_database"} and item.get("source_tier", 3) <= 2


def _source_mentions_traction_metrics(item: EvidenceItem) -> bool:
    text = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
    return any(keyword in text for keyword in TRACTION_KEYWORDS)


def _source_ids_for(evidence: list[EvidenceItem], evidence_type: str, limit: int = 2, use: str | None = None) -> list[str]:
    requested_use = use or DEFAULT_SOURCE_USE_BY_EVIDENCE_TYPE.get(evidence_type, "weak context")
    return [
        item["source_id"]
        for item in evidence
        if item["evidence_type"] == evidence_type and _source_allows_use(item, requested_use)
    ][:limit]


def _reliable_source_ids_for(evidence: list[EvidenceItem], evidence_type: str, limit: int = 2, use: str | None = None) -> list[str]:
    requested_use = use or HIGH_CONVICTION_USE_BY_EVIDENCE_TYPE.get(evidence_type, DEFAULT_SOURCE_USE_BY_EVIDENCE_TYPE.get(evidence_type, "weak context"))
    return [
        item["source_id"]
        for item in evidence
        if item["evidence_type"] == evidence_type
        and item.get("source_tier", 3) <= 2
        and item.get("confidence", 0) >= 0.55
        and _source_allows_use(item, requested_use)
    ][:limit]


def _reliable_count(evidence: list[EvidenceItem], evidence_type: str, use: str | None = None) -> int:
    return len(_reliable_source_ids_for(evidence, evidence_type, limit=20, use=use))


def _total_count(evidence: list[EvidenceItem], evidence_type: str, use: str | None = None) -> int:
    return len(_source_ids_for(evidence, evidence_type, limit=20, use=use))


def _tier_counts(evidence: list[EvidenceItem], evidence_type: str | None = None, use: str | None = None) -> dict[int, int]:
    counts = {1: 0, 2: 0, 3: 0}
    for item in evidence:
        if evidence_type and item["evidence_type"] != evidence_type:
            continue
        if use and not _source_allows_use(item, use):
            continue
        tier = int(item.get("source_tier", 3))
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def _weighted_evidence_score(evidence: list[EvidenceItem], evidence_type: str, use: str | None = None) -> float:
    return round(
        sum(
            float(item.get("source_weight", 0.2))
            for item in evidence
            if item["evidence_type"] == evidence_type and (not use or _source_allows_use(item, use))
        ),
        2,
    )


def _coverage_quality(evidence: list[EvidenceItem], evidence_type: str, use: str | None = None) -> str:
    requested_use = use or HIGH_CONVICTION_USE_BY_EVIDENCE_TYPE.get(evidence_type)
    counts = _tier_counts(evidence, evidence_type, requested_use)
    weighted = _weighted_evidence_score(evidence, evidence_type, requested_use)
    if counts[1] >= 2 or (counts[1] >= 1 and counts[2] >= 1 and weighted >= 1.6):
        return "Strong"
    if counts[1] >= 1 or counts[2] >= 2 or weighted >= 1.0:
        if counts[1] == 0 and counts[2] == 0:
            return "Weak"
        return "Medium"
    if counts[2] >= 1 or counts[3] >= 1:
        return "Weak"
    return "Unknown"


def _cap_confidence(confidence: str, cap: str) -> str:
    rank = {"Low": 0, "Medium": 1, "High": 2}
    reverse = {0: "Low", 1: "Medium", 2: "High"}
    return reverse[min(rank[confidence], rank[cap])]


def _evidence_line(evidence: list[EvidenceItem], evidence_type: str, fallback: str) -> str:
    item = next((source for source in evidence if source["evidence_type"] == evidence_type), None)
    if not item:
        return fallback
    snippet = item["snippet"].strip()
    if len(snippet) > 220:
        snippet = snippet[:217].rstrip() + "..."
    return snippet or item["title"]


def _company_specific_evidence_text(evidence: list[EvidenceItem], company: str) -> str:
    company_lower = company.lower()
    chunks = []
    for item in evidence:
        if item["evidence_type"] not in {"website", "news", "business_model"}:
            continue
        haystack = f"{item['title']} {item['snippet']}".lower()
        if company_lower and company_lower in haystack:
            chunks.append(f"{item['title']} {item['snippet']}")
    return "\n".join(chunks)


def _claim(text: str, source_ids: list[str] | None = None, note: bool = False, inference: bool = False) -> MemoClaim:
    return {"text": text, "source_ids": source_ids or [], "note_reference": note, "analyst_inference": inference}


def _section_confidence(evidence: list[EvidenceItem], evidence_type: str, has_notes: bool = True, important_unknowns: bool = False) -> str:
    coverage = _coverage_quality(evidence, evidence_type)
    if coverage == "Strong" and not important_unknowns:
        return "High"
    if coverage in {"Strong", "Medium"} or has_notes:
        return "Medium"
    if coverage == "Weak":
        return "Low"
    return "Low"


def _cross_section_confidence(
    evidence: list[EvidenceItem],
    evidence_types: list[str],
    has_notes: bool = True,
    important_unknowns: bool = False,
) -> str:
    reliable_count = sum(_reliable_count(evidence, evidence_type) for evidence_type in evidence_types)
    total_count = sum(_total_count(evidence, evidence_type) for evidence_type in evidence_types)
    covered_types = sum(1 for evidence_type in evidence_types if _reliable_count(evidence, evidence_type) > 0)
    if reliable_count >= 3 and covered_types >= 2 and not important_unknowns:
        return "High"
    if total_count >= 2 or has_notes:
        return "Medium"
    return "Low"


def _business_model_confidence(evidence: list[EvidenceItem]) -> str:
    return _section_confidence(evidence, "business_model", has_notes=False, important_unknowns=True)


def _competition_confidence(evidence: list[EvidenceItem]) -> str:
    direct = _section_confidence(evidence, "competitor", has_notes=False)
    return _cap_confidence(direct, "Medium")


def _investment_thesis_confidence(evidence: list[EvidenceItem]) -> str:
    base = _cross_section_confidence(evidence, ["website", "market", "news", "competitor", "business_model"], has_notes=True, important_unknowns=True)
    if _reliable_count(evidence, "business_model") == 0 or _reliable_count(evidence, "funding") == 0:
        return _cap_confidence(base, "Medium")
    return base


def _default_open_questions(profile: dict[str, Any]) -> list[str]:
    questions = list(profile.get("open_questions", []))
    diligence_questions = [
        "What is current ARR and growth rate?",
        "What is net revenue retention?",
        "How many customers are live versus in pilot?",
        "What is the customer payback period and measurable ROI?",
        "How long are sales cycles and implementation timelines?",
        "What is gross margin after services and model-inference costs?",
        "Which competitors are most frequently encountered in deals?",
    ]
    for question in diligence_questions:
        if question not in questions:
            questions.append(question)
    return questions


def _partner_claims(profile: dict[str, Any], evidence: list[EvidenceItem]) -> list[dict[str, Any]]:
    company = profile.get("company", "Company")
    customers = ", ".join(profile.get("customers", []))
    traction_sources = [
        item["source_id"]
        for item in evidence
        if item["evidence_type"] in {"funding", "business_model", "news"}
        and _source_allows_use(item, "traction verification")
        and _source_mentions_traction_metrics(item)
    ][:2]
    traction_line = next(
        (
            item["snippet"] or item["title"]
            for item in evidence
            if item["source_id"] in traction_sources
        ),
        "No public operating traction evidence found. Funding alone does not verify ARR, customer count, retention, revenue quality, or usage.",
    )
    claims = [
        {
            "partner_claim": f"{company}'s product is {profile.get('product', 'not specified')}.",
            "public_evidence_found": _evidence_line(evidence, "website", "No direct public website evidence found."),
            "verification_status": "Verified" if _source_ids_for(evidence, "website") else "Not Verified",
            "source_ids": _source_ids_for(evidence, "website"),
            "diligence_follow_up": "Confirm exact product scope, launch status, and customer deployment depth.",
        },
        {
            "partner_claim": f"Target customers include {customers}.",
            "public_evidence_found": _evidence_line(evidence, "news", "No public customer evidence found."),
            "verification_status": "Partially Verified" if _source_ids_for(evidence, "news") else "Not Verified",
            "source_ids": _source_ids_for(evidence, "news"),
            "diligence_follow_up": "Request customer list, live deployments, pilots, and reference calls.",
        },
        {
            "partner_claim": "Founder/team background is relevant to the workflow.",
            "public_evidence_found": _evidence_line(evidence, "leadership", "No public leadership evidence found."),
            "verification_status": "Partially Verified" if _source_ids_for(evidence, "leadership") else "Not Verified",
            "source_ids": _source_ids_for(evidence, "leadership"),
            "diligence_follow_up": f"Verify founder biography, prior roles, and depth of {profile.get('sector', 'domain')} operating experience.",
        },
        {
            "partner_claim": "The company has meaningful commercial traction.",
            "public_evidence_found": traction_line,
            "verification_status": "Partially Verified" if traction_sources else "Not Verified",
            "source_ids": traction_sources,
            "diligence_follow_up": "Ask for ARR, growth, NRR, pipeline, ACV, implementation backlog, and churn.",
        },
    ]
    return claims


def _source_ids_with_traction_metrics(evidence: list[EvidenceItem], limit: int = 2) -> list[str]:
    return [
        item["source_id"]
        for item in evidence
        if item["evidence_type"] in {"funding", "business_model", "news", "website"}
        and _source_mentions_traction_metrics(item)
        and _source_allows_use(item, "traction verification")
    ][:limit]


def _has_company_owned_only(evidence: list[EvidenceItem], source_ids: list[str]) -> bool:
    by_id = {item["source_id"]: item for item in evidence}
    selected = [by_id[source_id] for source_id in source_ids if source_id in by_id]
    return bool(selected) and all(item.get("source_type") == "company_owned" for item in selected)


def _verification_row(
    claim_area: str,
    claim: str,
    status: str,
    source_ids: list[str] | None = None,
    rationale: str = "",
    diligence_follow_up: str = "",
) -> dict[str, Any]:
    return {
        "claim_area": claim_area,
        "claim": claim,
        "verification_status": status,
        "source_ids": source_ids or [],
        "rationale": rationale,
        "diligence_follow_up": diligence_follow_up,
    }


def _claim_verification_layer(profile: dict[str, Any], evidence: list[EvidenceItem], memo: dict[str, Any]) -> list[dict[str, Any]]:
    understanding = memo.get("company_understanding", {})
    identity = memo.get("identity_resolution", {})
    company = memo.get("company", profile.get("company", "Company"))
    product = understanding.get("primary_product", profile.get("product", "Unknown"))
    buyer = understanding.get("target_buyer", "Unknown")
    validation = understanding.get("validation", {})
    contradictions = validation.get("contradictions", {})
    product_conflicts = understanding.get("product_evidence_conflicts", [])

    website_sources = _source_ids_for(evidence, "website", use="product positioning")
    news_sources = _source_ids_for(evidence, "news", use="market context")
    market_sources = _reliable_source_ids_for(evidence, "market", use="market conviction")
    competitor_discovery_sources = _source_ids_for(evidence, "competitor", use="competitor discovery")
    competitor_conviction_sources = _reliable_source_ids_for(evidence, "competitor", use="market conviction")
    funding_sources = _source_ids_for(evidence, "funding", use="funding history")
    traction_sources = _source_ids_with_traction_metrics(evidence)
    business_sources = _reliable_source_ids_for(evidence, "business_model", use="revenue quality")
    leadership_sources = _source_ids_for(evidence, "leadership", use="preliminary team mapping")

    rows = []
    if contradictions.get("severity") == "high":
        identity_status = "Conflicting evidence found"
        identity_reason = "Partner notes contain conflicting positioning signals; resolve identity and category before high-conviction analysis."
    elif identity.get("is_resolved"):
        identity_status = "Verified by public evidence" if website_sources else "Supported by partner notes only"
        identity_reason = "Identity resolution is high confidence." if identity.get("is_resolved") else "Identity is based on partner notes."
    else:
        identity_status = "Unsupported / requires diligence"
        identity_reason = "Unable to confidently identify the exact company entity."
    rows.append(
        _verification_row(
            "Company Identity",
            f"Company identity is {company}.",
            identity_status,
            website_sources,
            identity_reason,
            "Confirm exact legal entity, website, and funding/profile records before relying on company-specific public data.",
        )
    )

    product_status = "Conflicting evidence found" if product_conflicts else "Verified by public evidence" if website_sources else "Supported by partner notes only"
    rows.append(
        _verification_row(
            "Product Description",
            f"Primary product is {product}.",
            product_status,
            website_sources,
            "Company-owned sources can support positioning but do not independently validate superiority or traction." if website_sources else "Product description is currently note-derived.",
            "Confirm initial wedge, current production workflows, and whether public use cases reflect actual usage mix.",
        )
    )

    customer_sources = news_sources + _source_ids_for(evidence, "website", use="stated customers")
    customer_status = "Partially verified" if customer_sources else "Supported by partner notes only" if buyer != "Unknown" else "Unsupported / requires diligence"
    customer_reason = (
        "Customer logos or announcements indicate adoption signals, but they are not the same as active usage, retention, or expansion."
        if customer_sources
        else "Target customer is inferred from notes and workflow."
    )
    rows.append(_verification_row("Target Customer", f"Target customer is {buyer}.", customer_status, customer_sources, customer_reason, "Separate buyer, end user, budget owner, paid customer, pilot, active usage, and renewal behavior."))

    rows.append(
        _verification_row(
            "Market Size",
            f"Market context exists for {understanding.get('product_category', product)}.",
            "Verified by public evidence" if market_sources else "Inferred" if news_sources else "Unsupported / requires diligence",
            market_sources or news_sources,
            "Only independent/reputable market sources can support market conviction; vendor blogs and SEO pages are weak context only.",
            "Build bottom-up market sizing around buyer, workflow, budget owner, and willingness to pay.",
        )
    )

    traction_status = "Partially verified" if traction_sources else "Unsupported / requires diligence"
    rows.append(
        _verification_row(
            "Traction",
            "Partner notes indicate meaningful customer adoption, but public evidence does not yet verify deployment depth, paid usage, retention, or expansion.",
            traction_status,
            traction_sources,
            "Public funding is not commercial traction. Customer logos are not active usage or retention.",
            "Diligence should separate pilots, paid deployments, active usage, expansion, and renewal behavior.",
        )
    )

    rows.append(
        _verification_row(
            "Customers",
            "Customer adoption signals require separation from active usage and retention.",
            "Partially verified" if customer_sources else "Unsupported / requires diligence",
            customer_sources,
            "Company-owned case studies and logos are company-influenced; they can show stated adoption but not independently prove ROI or retention.",
            "Request customer list, reference calls, go-live dates, usage data, renewal behavior, and expansion cohorts.",
        )
    )

    rows.append(
        _verification_row(
            "Funding",
            "Funding history is publicly checkable only through funding databases, filings, or reputable financing coverage.",
            "Verified by public evidence" if funding_sources else "Unsupported / requires diligence",
            funding_sources,
            "Funding databases support round timing, investors, and valuation if available; they do not verify revenue quality.",
            "Confirm latest round, valuation, cash balance, burn, runway, and investor rights.",
        )
    )

    rows.append(
        _verification_row(
            "Business Model",
            f"Business model is {understanding.get('business_model', 'Unknown')}.",
            "Verified by public evidence" if business_sources else "Inferred" if understanding.get("business_model") not in {"Unknown", "Clarification required"} else "Unsupported / requires diligence",
            business_sources,
            "Business model should not be high confidence without pricing, ACV, margin, retention, or revenue mix evidence.",
            "Request pricing, ACV, gross margin, services burden, cohort retention, and revenue mix.",
        )
    )

    rows.append(
        _verification_row(
            "Competition",
            "Competitors are identified by buyer, budget, workflow, and category overlap.",
            "Partially verified" if competitor_discovery_sources and not competitor_conviction_sources else "Verified by public evidence" if competitor_conviction_sources else "Inferred",
            competitor_conviction_sources or competitor_discovery_sources,
            "Vendor comparison pages may identify names and vocabulary but cannot support market conviction or leadership claims.",
            "Run customer win/loss calls and compare pricing, buyer ownership, replacement target, and product depth.",
        )
    )

    rows.append(
        _verification_row(
            "Team",
            "Team background requires corroborated public evidence.",
            "Partially verified" if leadership_sources else "Supported by partner notes only" if profile.get("team") and profile.get("team") != "not specified" else "Unsupported / requires diligence",
            leadership_sources,
            "Professional profiles are preliminary mapping, not definitive verification without corroboration.",
            "Verify founder background, references, domain depth, prior exits, and key executive gaps.",
        )
    )

    rows.append(
        _verification_row(
            "Defensibility",
            "Defensibility depends on data advantage, workflow lock-in, distribution, integrations, and switching costs.",
            "Inferred",
            website_sources + competitor_conviction_sources,
            "Defensibility is typically an analyst inference until proven through product depth, customer behavior, and win/loss data.",
            "Test proprietary data, integration depth, switching costs, renewal rates, and competitive displacement.",
        )
    )

    risk_sources = list({source_id for risk in memo.get("visualizations", {}).get("risk_breakdown", {}).get("risks", []) for source_id in risk.get("source_ids", [])})
    rows.append(
        _verification_row(
            "Risks",
            "Risks are derived from partner-note signals, workflow/category inference, and public evidence where available.",
            "Partially verified" if risk_sources else "Inferred",
            risk_sources,
            "Workflow risks can be useful even when entity-specific risks remain unverified.",
            "Validate risk severity with customers, technical diligence, compliance review, and competitor references.",
        )
    )

    thesis_sources = list(dict.fromkeys((market_sources or news_sources) + website_sources + competitor_conviction_sources + business_sources))
    rows.append(
        _verification_row(
            "Investment Thesis",
            "Investment thesis requires repeatable adoption, measurable ROI, and defensible workflow integration.",
            "Inferred" if thesis_sources else "Unsupported / requires diligence",
            thesis_sources,
            "The thesis is an analyst inference and should remain cautious unless supported by high-quality operating evidence.",
            "Do not make an investment decision until traction, retention, ROI, economics, and competitive win/loss are verified.",
        )
    )
    return rows


def _factor(name: str, score: int | None, reason: str, source_ids: list[str]) -> dict[str, Any]:
    return {"name": name, "score": score, "reason": reason, "source_ids": source_ids}


def _score_or_unknown(
    label: str,
    rationale: str,
    source_ids: list[str],
    factors: list[dict[str, Any]],
    inference: bool = True,
    confidence: str = "Medium",
) -> dict[str, Any]:
    known_scores = [factor["score"] for factor in factors if isinstance(factor.get("score"), (int, float))]
    score = round(sum(known_scores) / len(known_scores), 1) if known_scores else None
    if not source_ids and label in {"Team", "Business Model", "Exit Potential", "Defensibility"}:
        confidence = "Low"
        rationale = f"Unknown: {rationale}"
    return {
        "dimension": label,
        "score": score,
        "rationale": rationale,
        "source_ids": source_ids,
        "factors": factors,
        "analyst_inference": inference,
        "confidence": confidence,
    }


def _rating_from_score(score: int | float | None, confidence: str, dimension: str) -> str:
    if score is None:
        return "Unknown"
    if dimension in {"Competition", "Defensibility", "Business Model", "Traction"} and score <= 4:
        return "High risk"
    if score >= 8:
        return "Strong positive"
    if score >= 6.5:
        return "Positive"
    if score >= 4.5:
        return "Mixed"
    if confidence == "Low":
        return "Unknown"
    return "Weak"


def _diligence_gap_for_dimension(dimension: str) -> str:
    gaps = {
        "Market": "Bottom-up TAM, budget owner urgency, willingness to pay, and timing proof.",
        "Product": "Live product demo, customer workflow depth, use-case mix, and differentiated outcomes.",
        "Team": "Founder references, execution track record, domain depth, and key leadership gaps.",
        "Business Model": "Pricing, ACV, gross margin, services burden, retention, and revenue mix.",
        "Competition": "Direct win/loss, pricing overlap, incumbent replacement behavior, and differentiation proof.",
        "Defensibility": "Proprietary data, integration depth, switching costs, renewal behavior, and distribution advantage.",
        "Exit Potential": "Scale evidence, strategic buyer interest, category durability, and financing path.",
        "Traction": "ARR, customer count, usage frequency, gross retention, NRR, expansion, and renewal behavior.",
    }
    return gaps.get(dimension, "Evidence quality, operating metrics, and customer references.")


def _qualitative_scorecard(scorecard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in scorecard:
        dimension = row.get("dimension", "")
        score = row.get("score")
        confidence = row.get("confidence", "Low")
        rating = _rating_from_score(score, confidence, dimension)
        reason = row.get("rationale", "")
        if rating == "Unknown":
            reason = reason.replace("score", "rating").replace("Score", "Rating")
            if "Unknown" not in reason:
                reason = f"Unknown because reliable evidence is missing. {reason}"
        rows.append(
            {
                "category": dimension,
                "rating": rating,
                "evidence_strength": confidence if confidence in {"High", "Medium", "Low"} else "Low",
                "reason": reason,
                "key_diligence_gap": _diligence_gap_for_dimension(dimension),
                "source_ids": row.get("source_ids", []),
            }
        )
    if not any(row["category"] == "Traction" for row in rows):
        rows.append(
            {
                "category": "Traction",
                "rating": "Unknown",
                "evidence_strength": "Low",
                "reason": "Partner notes may suggest customer adoption, but public evidence does not verify paid usage, retention, deployment depth, or expansion.",
                "key_diligence_gap": _diligence_gap_for_dimension("Traction"),
                "source_ids": [],
            }
        )
    return rows


def _overall_qualitative_rating(ratings: list[dict[str, Any]]) -> str:
    rank = {
        "Strong positive": 5,
        "Positive": 4,
        "Mixed": 3,
        "Weak": 2,
        "High risk": 1,
        "Unknown": 0,
    }
    known = [rank[row["rating"]] for row in ratings if row["rating"] != "Unknown"]
    if not known:
        return "Unknown"
    average = sum(known) / len(known)
    if any(row["rating"] == "High risk" for row in ratings):
        return "High risk"
    if average >= 4.5:
        return "Strong positive"
    if average >= 3.5:
        return "Positive"
    if average >= 2.5:
        return "Mixed"
    return "Weak"


def _investment_visualization(scorecard: list[dict[str, Any]]) -> dict[str, Any]:
    known_scores = [row["score"] for row in scorecard if isinstance(row.get("score"), (int, float))]
    overall = round(sum(known_scores) / len(known_scores), 1) if known_scores else None
    ranked = sorted(
        [row for row in scorecard if isinstance(row.get("score"), (int, float))],
        key=lambda row: row["score"],
        reverse=True,
    )
    unknown = [row for row in scorecard if row.get("score") is None]
    concerns = [row["dimension"] for row in unknown[:2]]
    if len(concerns) < 2:
        concerns.extend([row["dimension"] for row in ranked[: -3 : -1]][: 2 - len(concerns)])
    ratings = _qualitative_scorecard(scorecard)
    positive_ratings = [row for row in ratings if row["rating"] in {"Strong positive", "Positive"}]
    concern_ratings = [row for row in ratings if row["rating"] in {"Unknown", "Weak", "High risk"}]
    return {
        "ratings": ratings,
        "overall_rating": _overall_qualitative_rating(ratings),
        "rating_basis": "Qualitative assessment based on evidence strength, source reliability, and unresolved diligence gaps. Numeric working scores are retained only in trace/debug state.",
        "top_strengths": [row["category"] for row in positive_ratings[:2]] or [row["dimension"] for row in ranked[:2]],
        "top_concerns": [row["category"] for row in concern_ratings[:2]] or concerns,
        "raw_scores": scorecard,
        "overall_score": overall,
    }


def _claim_verification_visualization(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "verified_count": sum(1 for row in rows if row["verification_status"] == "Verified"),
        "partially_verified_count": sum(1 for row in rows if row["verification_status"] == "Partially Verified"),
        "not_verified_count": sum(1 for row in rows if row["verification_status"] == "Not Verified"),
    }


def _domain_terms(profile: dict[str, Any]) -> dict[str, str]:
    sector = profile.get("sector", "Unknown")
    understanding = profile.get("company_understanding", {})
    product = understanding.get("core_workflow", profile.get("product", "product"))
    if sector in {"Unknown", "Conflicting signals detected"} or understanding.get("category") == "Conflicting signals detected":
        return {
            "sales_cycle": "Sales cycle and procurement motion require clarification because category and buyer are unresolved.",
            "integration": "Integration requirements require clarification after product workflow is confirmed.",
            "regulatory": "Regulatory and compliance exposure require clarification after category is confirmed.",
            "technical": "Technical risk cannot be specified until actual product workflow is clarified.",
            "competition": "Competitive risk cannot be specified until the product category is clarified.",
            "value_metric": "buyer ROI and workflow value require clarification",
            "procurement": "the confirmed buyer and procurement process",
            "security": "security, privacy, compliance, and audit requirements appropriate to the confirmed category",
            "workflow": product,
        }
    if sector == "LegalTech":
        return {
            "sales_cycle": "Enterprise legal procurement, security review, and General Counsel budget approval can slow conversion.",
            "integration": "Integration with CLM systems, document repositories, Microsoft Word, e-signature, and matter-management workflows is central to adoption.",
            "regulatory": "Legal confidentiality, data privacy, privilege protection, and enterprise security require diligence.",
            "technical": "Technical risk centers on legal accuracy, hallucinated clause interpretation, workflow fit, and document handling.",
            "competition": "Public competitor evidence should be evaluated against AI contract review and CLM-adjacent legal workflow competitors.",
            "value_metric": "legal review cycle time, outside counsel savings, clause risk reduction, and contract throughput",
            "procurement": "legal software procurement, security review, and CLM/document integration cycles",
            "security": "SOC 2/security posture, data confidentiality, privilege protection, model data retention, and document access controls",
            "workflow": product,
        }
    if sector == "Fintech / Spend Management":
        return {
            "sales_cycle": "Finance software sales cycles, CFO budget scrutiny, and ERP/procurement integration can slow conversion.",
            "integration": "Integration with ERP, accounting, HRIS, payroll, card issuing, and procurement systems is central to adoption.",
            "regulatory": "Payments, card issuing, data security, fraud, and financial controls require diligence.",
            "technical": "Technical risk centers on workflow automation quality, payment reliability, controls, and integrations.",
            "competition": "Public competitor evidence indicates crowded spend management and corporate card competition.",
            "value_metric": "spend visibility, policy compliance, finance-team productivity, savings, and procurement cycle time",
            "procurement": "finance software procurement and integration cycles",
            "security": "SOC 2/security posture, payments compliance, fraud controls, data retention, and financial auditability",
            "workflow": product,
        }
    if sector == "ClimateTech":
        return {
            "sales_cycle": "Enterprise sustainability, finance, and operations sales cycles can slow adoption.",
            "integration": "ERP, utility, facilities, supplier, and emissions-data integrations are central to adoption.",
            "regulatory": "Disclosure rules, carbon accounting standards, and auditability require diligence.",
            "technical": "Technical risk centers on emissions-data quality, methodology, verification, and workflow fit.",
            "competition": "Competition includes climate software platforms, ESG incumbents, and sustainability modules from enterprise suites.",
            "value_metric": "audit-ready reporting, energy savings, emissions reduction, compliance readiness, and site-level operational impact",
            "procurement": "sustainability, finance, operations, and energy-procurement buying cycles",
            "security": "enterprise security, data lineage, audit controls, and regulatory reporting readiness",
            "workflow": product,
        }
    if sector == "HRTech":
        return {
            "sales_cycle": "HR budget cycles, employee-data review, and change management can slow adoption.",
            "integration": "HRIS, ATS, payroll, benefits, and identity integrations are central to adoption.",
            "regulatory": "Employment law, bias/fairness, privacy, and payroll/benefits compliance require diligence.",
            "technical": "Technical risk centers on matching quality, workflow adoption, data privacy, and fairness.",
            "competition": "Competition includes HR suites, payroll platforms, ATS vendors, and point solutions.",
            "value_metric": "time-to-hire, retention, payroll accuracy, engagement, employee productivity, and HR team efficiency",
            "procurement": "HR, finance, security, and legal procurement cycles",
            "security": "employee data privacy, SOC 2, access controls, retention, and compliance auditability",
            "workflow": product,
        }
    if sector == "Supply Chain":
        return {
            "sales_cycle": "Supply-chain software sales depend on operational urgency, integration scope, and facilities rollout.",
            "integration": "ERP, WMS, TMS, supplier, carrier, and forecasting-data integrations are central to adoption.",
            "regulatory": "Trade, customs, supplier compliance, and operational controls may require diligence.",
            "technical": "Technical risk centers on forecast accuracy, data completeness, exception handling, and integration reliability.",
            "competition": "Competition includes supply-chain planning, visibility, logistics, ERP, and procurement incumbents.",
            "value_metric": "inventory reduction, service-level improvement, freight savings, supplier reliability, and planner productivity",
            "procurement": "operations, supply-chain, procurement, and IT buying cycles",
            "security": "supplier data security, integration controls, operational resilience, and auditability",
            "workflow": product,
        }
    if sector == "Robotics":
        return {
            "sales_cycle": "Robotics sales depend on site ROI, safety review, deployment complexity, and operations budget.",
            "integration": "Fleet software, facilities workflows, hardware supply chain, and maintenance operations are central to adoption.",
            "regulatory": "Workplace safety, robotics certification, labor rules, and site-level compliance require diligence.",
            "technical": "Technical risk centers on uptime, reliability, autonomy performance, safety, and maintainability.",
            "competition": "Competition includes robotics startups, automation integrators, and industrial incumbents.",
            "value_metric": "labor savings, throughput, uptime, safety, utilization, and payback period",
            "procurement": "operations, manufacturing, warehouse, safety, and facilities buying cycles",
            "security": "fleet security, operational safety, remote access controls, and site compliance",
            "workflow": product,
        }
    if sector == "Biotech":
        return {
            "sales_cycle": "Biotech progress depends on scientific validation, pharma BD cycles, financing, and clinical/regulatory milestones.",
            "integration": "Wet-lab workflows, data pipelines, assay systems, and pharma collaboration processes are central.",
            "regulatory": "FDA/clinical pathway, quality systems, IP, and bioethics require diligence.",
            "technical": "Technical risk centers on reproducibility, translational validity, data quality, and platform/pipeline fit.",
            "competition": "Competition includes platform biotechs, pharma internal R&D, CROs, and academic approaches.",
            "value_metric": "validated hits/leads, pipeline progression, partnership value, clinical milestones, and IP strength",
            "procurement": "pharma BD, R&D partnership, and biotech financing cycles",
            "security": "IP controls, data governance, lab quality systems, and clinical/regulatory documentation",
            "workflow": product,
        }
    if sector == "Defense Technology":
        return {
            "sales_cycle": "Defense procurement cycles, program concentration, and government budget timing can slow conversion.",
            "integration": "Integration with defense systems, field operations, hardware supply chains, and mission workflows is central to adoption.",
            "regulatory": "Export controls, classified work, security clearances, contracting rules, and geopolitical exposure require diligence.",
            "technical": "Technical risk centers on autonomy performance, hardware reliability, sensor fusion, field deployment, and mission safety.",
            "competition": "Public competitor evidence indicates pressure from defense primes and specialized defense technology companies.",
            "value_metric": "mission effectiveness, cost per capability, deployment speed, reliability, and program expansion",
            "procurement": "government defense procurement and program approval cycles",
            "security": "export controls, classified data handling, cybersecurity, supply-chain assurance, and government compliance",
            "workflow": product,
        }
    return {
        "sales_cycle": "Enterprise healthcare sales cycles, procurement, and implementation approvals can slow conversion.",
        "integration": "Integration with clinical or administrative systems is likely central to adoption and must be verified.",
        "regulatory": "Healthcare data privacy, security review, and workflow governance require diligence.",
        "technical": "Technical risk centers on AI quality, workflow fit, data availability, and model governance.",
        "competition": "Public competitor evidence indicates crowded healthcare automation / AI workflow activity.",
        "value_metric": "time saved, cost reduction, revenue capture, or reduced administrative backlog",
        "procurement": "healthcare procurement, security review, and integration cycles",
        "security": "HIPAA/security posture, SOC 2 status, data retention, auditability, and model governance",
        "workflow": product,
    }


def _risk_breakdown(evidence: list[EvidenceItem], company: str, profile: dict[str, Any]) -> dict[str, Any]:
    understanding = profile.get("company_understanding", {})
    contradiction = understanding.get("validation", {}).get("contradictions", {})
    if contradiction.get("severity") == "high":
        reason = "Conflicting signals detected. Clarify positioning before workflow-level risk analysis."
        return {
            "risks": [
                {
                    "risk": "Conflicting positioning signals",
                    "risk_name": "Conflicting positioning signals",
                    "risk_type": "note_explicit",
                    "score": None,
                    "reason": reason,
                    "rationale": reason,
                    "evidence": ["Notes"],
                    "source_ids": [],
                    "confidence": "High",
                    "diligence_question": "Is the company primarily cybersecurity, workflow automation, compliance software, or another category?",
                }
            ]
        }
    industry = understanding.get("industry", profile.get("sector", "Unknown"))
    subindustry = understanding.get("subindustry", "Unknown")
    identity = profile.get("identity_resolution", {})
    understanding_confidence = float(understanding.get("validation", {}).get("confidence", 0))
    risk_context = _note_derived_risk_context(profile, understanding, industry, subindustry)
    industry = risk_context["industry"]
    subindustry = risk_context["subindustry"]
    if industry in {"", "Unknown"} or (understanding_confidence < 0.8 and not risk_context["has_workflow_context"]):
        reason = "Industry-specific risks unavailable until company category is verified."
        return {
            "risks": [
                {
                    "risk": "Industry-specific risks unavailable",
                    "risk_name": "Industry-specific risks unavailable",
                    "risk_type": "workflow_inferred",
                    "score": None,
                    "reason": reason,
                    "rationale": reason,
                    "evidence": [],
                    "source_ids": [],
                    "confidence": "Low",
                    "diligence_question": "Confirm the correct company entity, category, buyer, workflow, and product before generating an industry-specific risk model.",
                }
            ]
        }
    dependencies = understanding.get("critical_dependencies", [])
    revenue_drivers = understanding.get("revenue_drivers", [])
    product = understanding.get("primary_product", profile.get("product", "product"))
    buyer = understanding.get("target_buyer", "target buyer")

    website_sources = _source_ids_for(evidence, "website")
    market_sources = _source_ids_for(evidence, "market")
    competitor_sources = _source_ids_for(evidence, "competitor")
    news_sources = _source_ids_for(evidence, "news")
    business_sources = _source_ids_for(evidence, "business_model")
    funding_sources = _source_ids_for(evidence, "funding")
    note_risk_signals = _note_risk_signals(profile.get("raw_notes", ""))

    if industry == "LegalTech":
        risk_specs = [
            ("Legal accuracy risk", 8, "Contract review AI must accurately identify clause issues, legal risk, and negotiation-relevant language.", website_sources, "Medium" if website_sources else "Low"),
            ("Hallucinated clause interpretation risk", 8, "Incorrect or invented legal interpretations could erode attorney trust and create customer liability concerns.", website_sources, "Medium" if website_sources else "Low"),
            ("Attorney trust / adoption risk", 7, f"{product} only creates value if legal teams trust it inside daily review and redlining workflows.", website_sources, "Medium" if website_sources else "Low"),
            ("Data privacy and confidentiality risk", 7, "Contracts can contain sensitive commercial, employment, and customer data, making confidentiality and access controls central diligence items.", news_sources or website_sources, "Medium" if news_sources or website_sources else "Low"),
            ("Enterprise legal procurement risk", 6, f"{buyer} may require security, legal operations, and procurement approval before rollout.", news_sources, "Medium" if news_sources else "Low"),
            ("Incumbent CLM competition risk", 7, "CLM suites and legal AI vendors can compete for the same contract review budget and workflow.", competitor_sources, "Medium" if competitor_sources else "Low"),
            ("AI commoditization risk", 7, "AI contract review features may commoditize unless the company has superior workflow integration, proprietary playbooks, or distribution.", competitor_sources or website_sources, "Medium" if competitor_sources or website_sources else "Low"),
            ("Workflow integration risk", 6, f"Dependencies include {', '.join(dependencies[:3])}; weak integration can limit adoption.", website_sources, "Medium" if website_sources else "Low"),
        ]
    elif industry == "Fintech / Spend Management":
        risk_specs = [
            ("Interchange concentration risk", 7, f"Revenue may depend on card spend and interchange economics; revenue drivers include {', '.join(revenue_drivers)}.", business_sources or website_sources, "Medium"),
            ("Banking partner risk", 7, f"{company} depends on {', '.join(dependencies[:3])}, creating partner and operational dependency.", website_sources, "Medium"),
            ("Fraud risk", 6, "Spend management and card products require strong fraud controls, underwriting, and transaction monitoring.", news_sources or website_sources, "Medium"),
            ("Credit risk", 6, "Corporate card exposure can create credit losses if underwriting or customer mix deteriorates.", business_sources, "Low" if not business_sources else "Medium"),
            ("SMB macro exposure risk", 6, f"If {buyer} includes startups/SMBs, spend volume and retention may be macro-sensitive.", market_sources, "Medium" if market_sources else "Low"),
            ("Competitive pressure from incumbents", 7, "Competition from Brex, Amex, banks, ERP/procurement suites, and spend-management platforms can pressure margins and CAC.", competitor_sources, "Medium"),
        ]
    elif industry == "ClimateTech":
        risk_specs = [
            ("Emissions data quality risk", 8, "Climate software depends on complete, accurate, and auditable emissions and energy data.", website_sources, "Medium" if website_sources else "Low"),
            ("Regulatory reporting risk", 7, "Changing disclosure rules and carbon accounting standards can affect product requirements and buyer urgency.", news_sources or market_sources, "Medium" if news_sources or market_sources else "Low"),
            ("Customer ROI proof risk", 7, "Customers need measurable compliance, savings, or decarbonization outcomes to justify expansion.", market_sources, "Medium" if market_sources else "Low"),
            ("Integration burden risk", 6, f"Dependencies include {', '.join(dependencies[:3])}.", website_sources, "Medium" if website_sources else "Low"),
            ("ESG platform competition risk", 6, "ESG, sustainability, and enterprise software incumbents can compete for the same reporting budget.", competitor_sources, "Medium" if competitor_sources else "Low"),
            ("Services margin risk", 6, "Implementation and data-cleaning services can pressure gross margin if not productized.", business_sources, "Low" if not business_sources else "Medium"),
        ]
    elif industry == "HRTech":
        risk_specs = [
            ("Employee data privacy risk", 8, "HR systems process sensitive employee and candidate data, making privacy and access controls critical.", website_sources or news_sources, "Medium" if website_sources or news_sources else "Low"),
            ("Bias / compliance risk", 8, "Talent and workforce AI can create employment-law, fairness, and auditability concerns.", news_sources, "Medium" if news_sources else "Low"),
            ("HRIS integration risk", 7, f"Dependencies include {', '.join(dependencies[:3])}.", website_sources, "Medium" if website_sources else "Low"),
            ("Change management risk", 6, "Managers, recruiters, employees, and HR teams must adopt the workflow for value to materialize.", market_sources or website_sources, "Medium" if market_sources or website_sources else "Low"),
            ("Suite competition risk", 7, "Workday, ADP, Oracle, SAP, and Rippling can bundle competing HR workflows.", competitor_sources, "Medium" if competitor_sources else "Low"),
            ("Retention / seat contraction risk", 6, "Revenue can be exposed to hiring cycles, employee counts, and module expansion.", business_sources, "Low" if not business_sources else "Medium"),
        ]
    elif industry == "Supply Chain":
        risk_specs = [
            ("Data integration risk", 8, "Supply-chain value depends on reliable ERP/WMS/TMS, supplier, carrier, and inventory data.", website_sources, "Medium" if website_sources else "Low"),
            ("Forecast accuracy risk", 7, "Planning products must prove forecast accuracy and operational impact across volatile demand patterns.", market_sources or website_sources, "Medium" if market_sources or website_sources else "Low"),
            ("Implementation complexity risk", 7, "Multi-site, multi-system deployments can create services burden and slow time to value.", website_sources, "Medium" if website_sources else "Low"),
            ("Network density risk", 6, "Visibility and logistics platforms may depend on supplier/carrier network coverage.", market_sources, "Medium" if market_sources else "Low"),
            ("Incumbent SCM competition risk", 7, "ERP, SCM, logistics, and planning incumbents can compete through existing enterprise relationships.", competitor_sources, "Medium" if competitor_sources else "Low"),
            ("Cyclicality risk", 5, "Shipping volumes and supply-chain budgets can fluctuate with macro and inventory cycles.", market_sources, "Medium" if market_sources else "Low"),
        ]
    elif industry == "Robotics":
        risk_specs = [
            ("Hardware reliability risk", 8, "Robotics adoption depends on uptime, safety, durability, and field reliability.", website_sources, "Medium" if website_sources else "Low"),
            ("Deployment complexity risk", 7, "Site mapping, workflow redesign, safety review, and support can slow rollouts.", website_sources, "Medium" if website_sources else "Low"),
            ("Manufacturing / supply-chain risk", 7, f"Dependencies include {', '.join(dependencies[:3])}.", website_sources, "Medium" if website_sources else "Low"),
            ("Unit economics risk", 7, "Robotics-as-a-service and hardware deployments require attractive payback, utilization, and maintenance economics.", business_sources, "Low" if not business_sources else "Medium"),
            ("Safety / regulatory risk", 6, "Physical automation must satisfy workplace safety, site compliance, and operational risk requirements.", news_sources, "Medium" if news_sources else "Low"),
            ("Robotics incumbent competition risk", 6, "Automation incumbents and specialized robotics vendors can compete on reliability, service, and procurement trust.", competitor_sources, "Medium" if competitor_sources else "Low"),
        ]
    elif industry == "Biotech":
        risk_specs = [
            ("Scientific validation risk", 9, "Biotech platforms must prove reproducible biology, assay validity, and translational relevance.", website_sources, "Medium" if website_sources else "Low"),
            ("Clinical / regulatory risk", 9, "Therapeutic or clinical-development claims face regulatory, trial-design, and patient-safety risk.", news_sources or website_sources, "Medium" if news_sources or website_sources else "Low"),
            ("IP / data moat risk", 8, "Platform value depends on defensible IP, proprietary datasets, and know-how.", website_sources, "Medium" if website_sources else "Low"),
            ("Financing / runway risk", 8, "Biotech companies often require substantial capital before commercial revenue.", funding_sources if 'funding_sources' in locals() else [], "Low"),
            ("Pharma partnership risk", 7, "Milestone and licensing economics depend on pharma BD validation and partnership quality.", business_sources, "Low" if not business_sources else "Medium"),
            ("Platform-to-pipeline risk", 8, "A platform must translate into validated assets or partnerships, not just discovery activity.", market_sources or website_sources, "Medium" if market_sources or website_sources else "Low"),
        ]
    elif industry == "Healthcare AI" and subindustry == "Radiology Workflow":
        risk_specs = [
            ("Diagnostic accuracy risk", 9, "Radiology AI that affects interpretation or report generation must be accurate enough for clinical use and radiologist trust.", website_sources, "Medium" if website_sources else "Low"),
            ("FDA / regulatory risk", 8, "Radiology AI may face FDA, clinical governance, validation, or quality-system diligence depending on product claims and deployment model.", news_sources or website_sources, "Medium" if news_sources or website_sources else "Low"),
            ("Clinical liability risk", 8, "Errors in imaging workflow or drafted reports could create patient-safety, malpractice, and adoption barriers.", website_sources, "Medium" if website_sources else "Low"),
            ("Physician trust / adoption risk", 7, "Radiologists must trust outputs before relying on drafted findings or workflow recommendations.", website_sources, "Medium" if website_sources else "Low"),
            ("PACS/RIS/EHR integration risk", 7, f"Adoption may depend on {', '.join([dep for dep in dependencies if dep])}.", website_sources, "Medium" if website_sources else "Low"),
            ("Data privacy and security risk", 7, "Imaging workflows involve PHI and enterprise clinical systems, requiring privacy, security, and access-control diligence.", news_sources or website_sources, "Medium" if news_sources or website_sources else "Low"),
            ("Incumbent competition risk", 6, "Radiology reporting, PACS/RIS vendors, and imaging AI companies can compete for the same workflow budget.", competitor_sources, "Medium" if competitor_sources else "Low"),
        ]
    elif industry == "Healthcare AI" and subindustry == "Clinical Documentation":
        risk_specs = [
            ("Clinical adoption risk", 7, f"{product} only works if clinicians adopt it in daily workflow.", website_sources, "Medium"),
            ("Documentation accuracy risk", 8, "Clinical documentation quality must be accurate enough for provider trust, coding, and downstream care.", website_sources, "Medium"),
            ("Provider workflow disruption risk", 6, "Even useful AI can fail if it disrupts visit flow, note review, or clinician preferences.", website_sources, "Medium"),
            ("EHR dependency risk", 7, f"Adoption may depend on {', '.join([dep for dep in dependencies if 'EHR' in dep or 'Epic' in dep] or ['EHR integrations'])}.", website_sources, "Medium"),
            ("Healthcare procurement risk", 7, "Health-system procurement, security review, and implementation cycles can slow sales.", news_sources, "Medium"),
            ("HIPAA / compliance risk", 6, "Handling clinical conversations and documentation requires healthcare privacy, security, and governance diligence.", news_sources or website_sources, "Medium"),
            ("AI commoditization risk", 7, "Ambient AI documentation is increasingly competitive, so differentiation and workflow depth must be proven.", competitor_sources, "Medium"),
        ]
    elif industry == "Healthcare AI":
        risk_specs = [
            ("Clinical adoption risk", 6, f"{product} needs adoption inside healthcare workflows.", website_sources, "Medium"),
            ("Workflow disruption risk", 6, "Healthcare workflow changes can slow adoption even when ROI is attractive.", website_sources, "Medium"),
            ("Integration dependency risk", 7, f"Dependencies include {', '.join(dependencies[:3])}.", website_sources, "Medium"),
            ("Healthcare procurement risk", 7, "Healthcare procurement and security review can slow enterprise sales.", news_sources, "Medium"),
            ("HIPAA / compliance risk", 6, "Healthcare data and workflow automation require privacy and compliance diligence.", news_sources or website_sources, "Medium"),
            ("Reimbursement / policy risk", 5, "Healthcare workflows may be affected by policy, reimbursement, or payer behavior.", market_sources, "Low" if not market_sources else "Medium"),
        ]
    elif industry == "Defense Technology":
        risk_specs = [
            ("Program concentration risk", 7, "Revenue may be concentrated in a small number of government programs or awards.", business_sources, "Low" if not business_sources else "Medium"),
            ("Defense procurement risk", 8, "Government procurement cycles, budget timing, and program approvals can delay conversion.", news_sources, "Medium"),
            ("Manufacturing / supply-chain risk", 7, f"Dependencies include {', '.join(dependencies[:3])}.", website_sources, "Medium"),
            ("Technical field-performance risk", 7, "Autonomous systems must perform reliably in field conditions and mission workflows.", website_sources, "Medium"),
            ("Regulatory / export-control risk", 7, "Defense products may face export-control, classified-work, and contracting compliance constraints.", news_sources, "Medium"),
            ("Prime contractor competition risk", 6, "Defense primes and specialized vendors can compete on procurement access and bundled capabilities.", competitor_sources, "Medium"),
        ]
    else:
        risk_specs = [
            ("Churn risk", 6, "Retention must be proven through cohort and customer-reference diligence.", business_sources, "Low" if not business_sources else "Medium"),
            ("CAC payback risk", 6, "Sales efficiency and payback period are not yet verified.", business_sources, "Low" if not business_sources else "Medium"),
            ("Market saturation risk", 5, "Market saturation depends on category maturity and competitive density.", market_sources, "Medium" if market_sources else "Low"),
            ("Product differentiation risk", 6, "Differentiation must be proven through win/loss and product-depth diligence.", website_sources + competitor_sources, "Medium"),
            ("Platform dependency risk", 6, f"Dependencies include {', '.join(dependencies) if dependencies else 'unverified third-party platforms'}.", website_sources, "Low" if not website_sources else "Medium"),
        ]

    selected = []
    for name, score, rationale, source_ids, confidence in risk_specs[:8]:
        selected.append(_risk_row(name, score, rationale, source_ids, confidence, industry, note_risk_signals))
    return {"risks": selected}


def _note_derived_risk_context(
    profile: dict[str, Any],
    understanding: dict[str, Any],
    industry: str,
    subindustry: str,
) -> dict[str, Any]:
    text = " ".join(
        [
            str(profile.get("raw_notes", "")),
            str(profile.get("product", "")),
            str(profile.get("sector", "")),
            str(understanding.get("primary_product", "")),
            str(understanding.get("core_workflow", "")),
            str(understanding.get("target_buyer", "")),
            " ".join(understanding.get("critical_dependencies", [])),
        ]
    ).lower()
    has_workflow_context = False
    if industry in {"", "Unknown"}:
        if _contains_any(text, ["health system", "health systems", "hospital", "clinical", "physician", "doctor", "patient", "ehr", "epic", "cmio", "hipaa", "phi", "radiology", "radiologist", "imaging", "prior authorization"]):
            industry = "Healthcare AI"
            has_workflow_context = True
        elif _contains_any(text, ["contract review", "redlining", "clause extraction", "legal team", "general counsel", "clm"]):
            industry = "LegalTech"
            has_workflow_context = True
        elif _contains_any(text, ["corporate card", "spend management", "expense management", "interchange", "banking partner", "cfo"]):
            industry = "Fintech / Spend Management"
            has_workflow_context = True
        elif _contains_any(text, ["carbon accounting", "emissions", "climate", "decarbonization", "energy management", "renewable", "grid"]):
            industry = "ClimateTech"
            has_workflow_context = True
        elif _contains_any(text, ["hr", "human resources", "talent", "recruiting", "payroll", "benefits", "employee", "workforce"]):
            industry = "HRTech"
            has_workflow_context = True
        elif _contains_any(text, ["supply chain", "logistics", "freight", "inventory", "warehouse", "supplier", "demand planning"]):
            industry = "Supply Chain"
            has_workflow_context = True
        elif _contains_any(text, ["robotics", "robot", "robots", "warehouse automation", "industrial automation", "autonomous mobile robot"]):
            industry = "Robotics"
            has_workflow_context = True
        elif _contains_any(text, ["biotech", "drug discovery", "therapeutics", "clinical trial", "genomics", "bioinformatics", "life sciences"]):
            industry = "Biotech"
            has_workflow_context = True
        elif _contains_any(text, ["developer", "api", "sdk", "open source", "cloud platform"]):
            industry = "Developer Tools"
            has_workflow_context = True
    else:
        has_workflow_context = True

    if industry == "Healthcare AI":
        if _contains_any(text, ["radiology", "radiologist", "imaging", "pacs", "diagnostic report"]):
            subindustry = "Radiology Workflow"
        elif _contains_any(text, ["ambient", "scribe", "clinical documentation", "clinical note", "physician documentation"]):
            subindustry = "Clinical Documentation"
        elif _contains_any(text, ["prior authorization", "utilization management"]):
            subindustry = "Prior Authorization"
        elif subindustry in {"", "Unknown"}:
            subindustry = "Healthcare Workflow Automation"
    return {"industry": industry, "subindustry": subindustry, "has_workflow_context": has_workflow_context}


NOTE_RISK_SIGNAL_TERMS = {
    "regulatory": ["fda", "regulatory", "compliance"],
    "liability": ["liability", "malpractice", "clinical liability"],
    "accuracy": ["accuracy", "accurate", "diagnostic", "hallucination", "hallucinated"],
    "trust": ["trust", "adoption", "physician trust", "attorney trust"],
    "integration": ["integration", "pacs", "ris", "ehr", "clm"],
    "competition": ["competition", "competitive", "competitor", "commoditization", "commoditize"],
    "retention": ["retention", "churn", "nrr"],
    "privacy": ["privacy", "security", "hipaa", "confidentiality"],
    "data moat": ["data moat", "proprietary data", "data advantage"],
    "data quality": ["data quality", "auditability", "forecast accuracy", "emissions data"],
    "hardware": ["hardware", "uptime", "reliability", "maintenance"],
    "scientific validation": ["scientific validation", "reproducibility", "clinical trial", "regulatory pathway", "ip"],
}


RISK_SIGNAL_MAP = {
    "Diagnostic accuracy risk": ["accuracy"],
    "FDA / regulatory risk": ["regulatory"],
    "Clinical liability risk": ["liability"],
    "Physician trust / adoption risk": ["trust"],
    "PACS/RIS/EHR integration risk": ["integration"],
    "Data privacy and security risk": ["privacy"],
    "Legal accuracy risk": ["accuracy"],
    "Hallucinated clause interpretation risk": ["accuracy"],
    "Attorney trust / adoption risk": ["trust"],
    "Data privacy and confidentiality risk": ["privacy"],
    "Enterprise legal procurement risk": ["regulatory"],
    "Incumbent CLM competition risk": ["competition"],
    "AI commoditization risk": ["competition", "data moat"],
    "Workflow integration risk": ["integration"],
    "Interchange concentration risk": ["retention"],
    "Fraud risk": ["security"],
    "Credit risk": ["regulatory"],
    "Banking partner risk": ["integration"],
    "Regulatory / export-control risk": ["regulatory"],
    "HIPAA / compliance risk": ["privacy", "regulatory"],
    "EHR dependency risk": ["integration"],
    "Documentation accuracy risk": ["accuracy"],
    "Clinical adoption risk": ["trust"],
    "Provider workflow disruption risk": ["trust"],
    "Competitive pressure from incumbents": ["competition"],
    "Incumbent competition risk": ["competition"],
    "Emissions data quality risk": ["data quality"],
    "Regulatory reporting risk": ["regulatory"],
    "Employee data privacy risk": ["privacy"],
    "Bias / compliance risk": ["regulatory"],
    "Data integration risk": ["integration", "data quality"],
    "Forecast accuracy risk": ["data quality"],
    "Hardware reliability risk": ["hardware"],
    "Unit economics risk": ["retention"],
    "Scientific validation risk": ["scientific validation"],
    "Clinical / regulatory risk": ["scientific validation", "regulatory"],
    "IP / data moat risk": ["data moat", "scientific validation"],
}


def _note_risk_signals(raw_notes: str) -> set[str]:
    lower = raw_notes.lower()
    return {signal for signal, terms in NOTE_RISK_SIGNAL_TERMS.items() if any(term in lower for term in terms)}


def _risk_row(
    name: str,
    score: int,
    rationale: str,
    source_ids: list[str],
    confidence: str,
    industry: str,
    note_risk_signals: set[str],
) -> dict[str, Any]:
    matched_signals = [signal for signal in RISK_SIGNAL_MAP.get(name, []) if signal in note_risk_signals]
    if matched_signals:
        risk_type = "note_explicit"
        confidence = "High"
        evidence: list[str] = ["Notes"]
        final_score: int | None = score
        source_ids = source_ids
        rationale = f"Partner notes explicitly flag {', '.join(matched_signals)} risk. {rationale}"
    else:
        risk_type = "workflow_inferred"
        confidence = "Medium" if confidence != "Low" or not source_ids else confidence
        evidence = source_ids or ["Notes"]
        final_score = score
    return {
        "risk": name,
        "risk_name": name,
        "risk_type": risk_type,
        "score": final_score,
        "reason": rationale,
        "rationale": rationale,
        "evidence": evidence,
        "source_ids": source_ids,
        "confidence": confidence,
        "diligence_question": _risk_diligence_question(name, industry),
    }


def _risk_diligence_question(risk_name: str, industry: str) -> str:
    questions = {
        "Interchange concentration risk": "What percentage of revenue comes from interchange versus software and other monetization streams?",
        "Banking partner risk": "Which banking, issuing, processing, and card-network partners are critical, and what happens if one relationship changes?",
        "Fraud risk": "What are fraud loss rates, approval/decline rules, and monitoring controls by customer segment?",
        "Credit risk": "What credit exposure does the company retain, and how have losses trended by cohort?",
        "SMB macro exposure risk": "How sensitive are spend volume, churn, and card utilization to SMB macro conditions?",
        "Competitive pressure from incumbents": "How often does the company compete with Brex, Amex, banks, ERP suites, or procurement platforms, and why does it win?",
        "Clinical adoption risk": "What percentage of target clinicians use the product weekly after deployment?",
        "Documentation accuracy risk": "What quality metrics prove documentation accuracy, coding reliability, and clinician trust?",
        "Provider workflow disruption risk": "Where does the product add review burden or alter clinician workflow?",
        "EHR dependency risk": "How dependent is customer acquisition and usage on Epic or other EHR integrations?",
        "Healthcare procurement risk": "What is the median sales cycle from first meeting to go-live for health-system customers?",
        "HIPAA / compliance risk": "What security, privacy, audit, and data-retention controls are required by enterprise customers?",
        "AI commoditization risk": "What proprietary data, workflow embedding, or distribution advantage prevents feature commoditization?",
        "Program concentration risk": "What percentage of revenue and backlog depends on the top three government programs?",
        "Defense procurement risk": "Which awards are funded production programs versus pilots, prototypes, or IDIQ ceiling values?",
        "Manufacturing / supply-chain risk": "Which components or suppliers constrain production scale?",
        "Technical field-performance risk": "What field reliability, autonomy, and mission-success metrics have been demonstrated?",
        "Regulatory / export-control risk": "What export-control, classified-work, and cybersecurity requirements constrain sales?",
        "Prime contractor competition risk": "Where do primes compete directly, and when does the company partner versus compete?",
        "Legal accuracy risk": "What accuracy benchmarks prove the product correctly identifies clause risk and redlining issues across contract types?",
        "Hallucinated clause interpretation risk": "How does the product prevent, detect, and disclose hallucinated or unsupported legal interpretations?",
        "Attorney trust / adoption risk": "What percentage of attorneys use the product weekly after deployment, and where do they override its suggestions?",
        "Data privacy and confidentiality risk": "How are contracts stored, retained, encrypted, permissioned, and excluded from model training if required?",
        "Enterprise legal procurement risk": "What is the median cycle from legal-team pilot to enterprise rollout and budget approval?",
        "Incumbent CLM competition risk": "How often does the company compete with CLM suites or legal AI tools, and why does it win?",
        "Workflow integration risk": "Which CLM, document repository, Word, e-signature, and matter systems are required for adoption?",
        "Emissions data quality risk": "What source systems feed emissions data, how is it audited, and how much customer cleanup is required?",
        "Regulatory reporting risk": "Which disclosure regimes or carbon accounting standards drive urgency, and how often do requirements change?",
        "Customer ROI proof risk": "What measurable savings, compliance benefit, or emissions reduction has been proven with customers?",
        "ESG platform competition risk": "How often does the company compete with ESG, sustainability, ERP, or reporting incumbents?",
        "Services margin risk": "How much implementation or data-cleaning service work is required per deployment?",
        "Employee data privacy risk": "What employee or candidate data is processed, retained, permissioned, and excluded from model training?",
        "Bias / compliance risk": "How does the product monitor fairness, bias, employment-law compliance, and auditability?",
        "HRIS integration risk": "Which HRIS, ATS, payroll, benefits, and identity systems are required for deployment?",
        "Change management risk": "Which users must change behavior, and what adoption metrics prove workflow pull?",
        "Suite competition risk": "How often do HR suites or payroll incumbents bundle similar functionality?",
        "Retention / seat contraction risk": "How exposed is revenue to hiring cycles, employee count, or module attach?",
        "Data integration risk": "Which ERP, WMS, TMS, supplier, carrier, and inventory systems are required for value?",
        "Forecast accuracy risk": "What forecast accuracy, service-level, inventory, or freight-savings impact has been proven?",
        "Implementation complexity risk": "What is the median deployment time by customer size and system complexity?",
        "Network density risk": "How much supplier, carrier, or facility network coverage is required for defensibility?",
        "Incumbent SCM competition risk": "How often do ERP, SCM, planning, or logistics incumbents compete in deals?",
        "Cyclicality risk": "How sensitive are usage, transaction volume, and budgets to freight, inventory, or macro cycles?",
        "Hardware reliability risk": "What uptime, failure-rate, safety, and maintenance metrics have been proven in production?",
        "Deployment complexity risk": "What site work, workflow redesign, training, and support are required before go-live?",
        "Unit economics risk": "What are robot-level gross margin, payback, utilization, maintenance cost, and RaaS churn?",
        "Safety / regulatory risk": "What safety certifications, site approvals, and incident data are required for deployment?",
        "Robotics incumbent competition risk": "How often do robotics incumbents or automation integrators compete, and why does the company win?",
        "Scientific validation risk": "What reproducible assays, benchmarks, or peer-reviewed evidence validate the platform?",
        "Clinical / regulatory risk": "What is the clinical or regulatory path, and what milestones reduce development risk?",
        "IP / data moat risk": "Which patents, proprietary datasets, assays, or know-how create defensibility?",
        "Financing / runway risk": "How much capital is needed to reach the next scientific, clinical, or partnership milestone?",
        "Pharma partnership risk": "What pharma partnerships, milestones, or licensing economics validate platform value?",
        "Platform-to-pipeline risk": "How does the platform translate into validated assets, partnerships, or clinical candidates?",
    }
    return questions.get(risk_name, f"What evidence would reduce {risk_name.lower()} for this {industry} company?")


def _bull_bear_weights(memo: dict[str, Any]) -> dict[str, Any]:
    positives = [
        {"factor": "Market pain / buyer urgency", "value": 8, "reason": memo["bull_case"][0]["text"], "source_ids": memo["bull_case"][0]["source_ids"]},
        {"factor": "Product-thesis alignment", "value": 7, "reason": memo["bull_case"][1]["text"], "source_ids": memo["bull_case"][1]["source_ids"]},
        {"factor": "Expansion potential", "value": 6, "reason": memo["bull_case"][2]["text"], "source_ids": memo["bull_case"][2]["source_ids"]},
    ]
    negatives = [
        {"factor": "Unverified commercial traction", "value": -8, "reason": memo["bear_case"][0]["text"], "source_ids": memo["bear_case"][0]["source_ids"]},
        {"factor": "Competitive intensity", "value": -7, "reason": memo["bear_case"][1]["text"], "source_ids": memo["bear_case"][1]["source_ids"]},
        {"factor": "AI commoditization risk", "value": -6, "reason": memo["bear_case"][2]["text"], "source_ids": memo["bear_case"][2]["source_ids"]},
    ]
    return {"factors": positives + negatives}


def _sources_grouped_by_category(evidence: list[EvidenceItem]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in evidence:
        grouped.setdefault(item["evidence_type"], []).append(
            {
                "source_id": item["source_id"],
                "title": item["title"],
                "url": item["url"],
                "confidence": item["confidence"],
                "source_quality": item.get("source_quality", "medium"),
                "quality_notes": item.get("quality_notes", ""),
                "source_tier": item.get("source_tier", 3),
                "source_weight": item.get("source_weight", 0.2),
                "source_tier_label": item.get("source_tier_label", "Tier 3: Supplemental"),
                "source_type": item.get("source_type", "generic_blog"),
                "independence_level": item.get("independence_level", "unknown"),
                "allowed_uses": item.get("allowed_uses", []),
                "disallowed_uses": item.get("disallowed_uses", []),
            }
        )
    return grouped


WHY_NOW_DRIVERS = {
    "technology inflection": ["ai", "automation", "machine learning", "model", "llm", "robotics", "api"],
    "regulatory change": ["regulatory", "compliance", "fda", "hipaa", "sec", "reporting", "audit"],
    "labor shortage": ["labor", "staffing", "clinician", "attorney", "worker", "shortage", "productivity"],
    "cost pressure": ["cost", "margin", "roi", "savings", "efficiency", "payback"],
    "budget shift": ["budget", "cfo", "procurement", "spend", "enterprise"],
    "new distribution channel": ["channel", "partner", "marketplace", "ecosystem", "platform"],
    "workflow digitization": ["workflow", "digitization", "automation", "system of record", "integration"],
    "data availability": ["data", "analytics", "visibility", "measurement", "benchmark"],
    "customer behavior change": ["customer", "adoption", "usage", "self-serve", "remote"],
    "incumbent weakness": ["incumbent", "legacy", "manual", "fragmented", "slow"],
    "macroeconomic pressure": ["macro", "interest", "inflation", "budget scrutiny", "runway"],
}


def _why_now_drivers(raw_notes: str, evidence: list[EvidenceItem], domain_terms: dict[str, str]) -> list[str]:
    note_lower = raw_notes.lower()
    context = " ".join([domain_terms.get("workflow", ""), domain_terms.get("value_metric", "")])
    evidence_text = " ".join(f"{item.get('title', '')} {item.get('snippet', '')}" for item in evidence[:8])
    context_lower = f"{context} {evidence_text}".lower()
    explicit = [driver for driver, keywords in WHY_NOW_DRIVERS.items() if any(keyword in note_lower for keyword in keywords)]
    contextual = [
        driver
        for driver, keywords in WHY_NOW_DRIVERS.items()
        if driver not in explicit and any(keyword in context_lower for keyword in keywords)
    ]
    drivers = explicit + contextual
    return drivers[:6] or ["workflow digitization", "cost pressure"]


def _source_ids_for_any(evidence: list[EvidenceItem], evidence_types: list[str], limit: int = 3) -> list[str]:
    source_ids = []
    for evidence_type in evidence_types:
        source_ids.extend(_source_ids_for(evidence, evidence_type, limit=limit))
    return list(dict.fromkeys(source_ids))[:limit]


def _thesis_framework(
    profile: dict[str, Any],
    evidence: list[EvidenceItem],
    memo: dict[str, Any],
    domain_terms: dict[str, str],
    market_is_known: bool,
    buyer_is_known: bool,
) -> dict[str, dict[str, Any]]:
    understanding = memo.get("company_understanding", {})
    company = memo.get("company", profile.get("company", "Company"))
    product = understanding.get("primary_product", profile.get("product", "product"))
    buyer = understanding.get("target_buyer", "Unknown buyer")
    workflow = understanding.get("core_workflow", product)
    drivers = _why_now_drivers(profile.get("raw_notes", ""), evidence, domain_terms)
    website_sources = _source_ids_for(evidence, "website", use="product positioning")
    market_sources = _reliable_source_ids_for(evidence, "market", use="market conviction")
    news_sources = _source_ids_for(evidence, "news", use="market context")
    competitor_sources = _source_ids_for(evidence, "competitor", use="competitor discovery")
    traction_sources = _source_ids_with_traction_metrics(evidence)
    business_sources = _reliable_source_ids_for(evidence, "business_model", use="revenue quality")
    missing = memo.get("missing_data", {})
    blocking = [gap.get("gap", "") for gap in missing.get("decision_blocking_unknowns", [])[:5]]
    next_gap = blocking[0] if blocking else "customer references and operating metrics"
    market_reason = (
        f"The pain appears tied to {domain_terms['value_metric']} in {workflow}, but severity, frequency, willingness to pay, and budget urgency still need proof."
        if market_is_known
        else "Market attractiveness cannot be assessed until the product category, workflow, and buyer are clear."
    )
    why_this_status = "Partially verified" if website_sources else "Supported by partner notes only"
    if not buyer_is_known or _is_placeholder_entity(workflow):
        why_this_status = "Unsupported / requires diligence"
    return {
        "why_now": {
            "answer": f"Potential timing drivers include {', '.join(drivers)}. These could make {workflow} more urgent for {buyer}, but timing remains a diligence hypothesis until buyer pull and budget urgency are verified.",
            "evidence_strength": "Medium" if market_sources or news_sources else "Low",
            "source_ids": market_sources or news_sources,
            "diligence_gap": "Verify which timing driver is actually creating budget now: technology, regulation, labor, costs, distribution, workflow digitization, data availability, customer behavior, incumbent weakness, or macro pressure.",
        },
        "why_this_company": {
            "answer": f"{company} may be interesting if its product scope, workflow depth, customer proof, and distribution give it an edge in {workflow}. Current evidence mainly supports positioning, not durable right-to-win.",
            "evidence_strength": "Medium" if website_sources else "Low",
            "source_ids": website_sources,
            "verification_status": why_this_status,
            "diligence_gap": "Confirm product differentiation, customer outcomes, win/loss evidence, implementation depth, and whether the company has proprietary data or distribution.",
        },
        "why_this_market": {
            "answer": market_reason,
            "evidence_strength": "Medium" if market_sources else "Low",
            "source_ids": market_sources,
            "diligence_gap": "Quantify pain frequency, economic severity, budget owner, buying urgency, market growth, and replacement target.",
        },
        "what_needs_to_be_true": {
            "answer": f"Diligence must confirm paid deployments, measurable ROI, repeatable sales, attractive gross margin, defensible differentiation, and credible expansion beyond the initial wedge.",
            "evidence_strength": "Low" if not traction_sources or not business_sources else "Medium",
            "source_ids": list(dict.fromkeys(traction_sources + business_sources + website_sources)),
            "diligence_gap": ", ".join(blocking) if blocking else "ARR, customer count, usage frequency, gross retention, NRR, gross margin, and competitive win/loss.",
        },
        "what_could_kill_the_deal": {
            "answer": "The deal becomes unattractive if adoption is mostly pilots, ROI is weak, retention or expansion is unproven, implementation requires heavy services, incumbents can bundle the workflow, or differentiation is not visible in customer win/loss.",
            "evidence_strength": "Low",
            "source_ids": competitor_sources,
            "diligence_gap": "Look for evidence of churn, failed deployments, long payback, poor gross margin, weak usage frequency, commoditized features, or consistent losses to incumbents.",
        },
        "next_diligence_step": {
            "answer": f"Next step: request evidence for {next_gap}, then run customer references and win/loss calls before upgrading conviction.",
            "evidence_strength": "High" if missing else "Medium",
            "source_ids": _source_ids_for_any(evidence, ["website", "business_model", "competitor", "news"]),
            "diligence_gap": "Ask for ARR bridge, customer list, cohort retention, usage logs, implementation timeline, gross margin, pricing, pipeline, and competitive win/loss.",
        },
    }


DEFENSIBILITY_CATEGORIES = [
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
]


DEFENSIBILITY_KEYWORDS = {
    "proprietary data": ["proprietary data", "unique data", "data moat", "dataset", "training data"],
    "feedback loops": ["feedback loop", "improves with usage", "learning loop", "usage data"],
    "workflow lock-in": ["workflow", "embedded", "daily use", "system of record", "lock-in"],
    "integrations": ["integration", "integrates", "api", "ehr", "erp", "clm", "hris", "pacs", "wms", "tms"],
    "switching costs": ["switching cost", "migration", "renewal", "retention", "implementation"],
    "network effects": ["network effect", "marketplace", "network density", "participants"],
    "regulatory/compliance advantage": ["compliance", "regulatory", "certification", "audit", "hipaa", "soc 2", "fda"],
    "distribution advantage": ["distribution", "channel", "partner", "go-to-market", "sales motion"],
    "brand/category leadership": ["leader", "category leadership", "brand", "recognized", "standard"],
    "cost advantage": ["cost advantage", "lower cost", "gross margin", "unit economics"],
    "technical differentiation": ["technical differentiation", "patent", "accuracy", "model", "benchmark", "performance"],
    "supply-side exclusivity": ["exclusive supply", "supplier exclusivity", "exclusive provider"],
    "demand-side aggregation": ["demand aggregation", "aggregates demand", "buyer network"],
    "ecosystem partnerships": ["ecosystem", "partnership", "partner", "platform partner", "marketplace partner"],
}


def _evidence_for_moat(evidence: list[EvidenceItem], category: str) -> list[EvidenceItem]:
    keywords = DEFENSIBILITY_KEYWORDS.get(category, [])
    matches = []
    for item in evidence:
        if item["evidence_type"] not in {"website", "news", "business_model", "competitor", "market"}:
            continue
        text = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
        if any(keyword in text for keyword in keywords):
            matches.append(item)
    return matches[:3]


def _moat_status(matches: list[EvidenceItem]) -> str:
    if not matches:
        return "Unknown"
    independent = [item for item in matches if item.get("source_type") == "independent_news" and item.get("source_tier", 3) <= 2]
    if len(independent) >= 2:
        return "Proven"
    return "Plausible"


def _company_architecture_question(understanding: dict[str, Any]) -> dict[str, str]:
    scope = understanding.get("primary_product_scope", "unclear")
    product_category = understanding.get("product_category", "Unknown")
    if scope == "broad_platform":
        hypothesis = "platform"
    elif scope == "marketplace":
        hypothesis = "marketplace"
    elif "system of record" in str(understanding).lower():
        hypothesis = "system of record/intelligence"
    elif scope == "specific_workflow":
        hypothesis = "workflow system or point solution"
    else:
        hypothesis = "Unknown"
    return {
        "question": "Is this company a feature, a point solution, a workflow system, a platform, or a system of record/intelligence?",
        "current_hypothesis": hypothesis,
        "rationale": f"Current product category is {product_category}; product scope is {scope}. Confirm through customer usage depth, system replacement behavior, integrations, and renewal data.",
    }


def _defensibility_framework(profile: dict[str, Any], evidence: list[EvidenceItem], memo: dict[str, Any]) -> dict[str, Any]:
    understanding = memo.get("company_understanding", {})
    rows = []
    for category in DEFENSIBILITY_CATEGORIES:
        matches = _evidence_for_moat(evidence, category)
        source_ids = [item["source_id"] for item in matches]
        status = _moat_status(matches)
        if status == "Unknown":
            current_evidence = "Unknown: current evidence does not prove this moat."
            risk = f"{category.title()} may be weak or irrelevant; avoid treating it as a moat until proven."
        elif status == "Plausible":
            current_evidence = "Plausible moat signal found, but evidence is not yet strong enough to call it proven."
            risk = f"{category.title()} may not translate into durable pricing power, retention, or win/loss advantage."
        else:
            current_evidence = "Multiple independent sources support this moat signal."
            risk = f"Even a proven {category} moat requires testing against customer behavior and competitor response."
        rows.append(
            {
                "potential_moat": category,
                "status": status,
                "current_evidence": current_evidence,
                "source_ids": source_ids,
                "diligence_needed": _diligence_for_moat(category),
                "risk": risk,
            }
        )
    return {
        "architecture_question": _company_architecture_question(understanding),
        "rows": rows,
        "rule": "Do not claim a moat unless evidence supports it; distinguish plausible moat from proven moat.",
    }


def _diligence_for_moat(category: str) -> str:
    questions = {
        "proprietary data": "What data is proprietary, how is it collected, and does it improve outcomes versus competitors?",
        "feedback loops": "Does usage generate learning that improves product quality, retention, or conversion?",
        "workflow lock-in": "Is the product embedded in daily workflow, or is it an easily replaceable feature?",
        "integrations": "Which systems are integrated, how deep are integrations, and do they create switching friction?",
        "switching costs": "What breaks if the customer leaves, and what migration effort would be required?",
        "network effects": "Does each new participant improve value for other participants?",
        "regulatory/compliance advantage": "Does compliance capability create a real buying advantage or just table stakes?",
        "distribution advantage": "Does the company have privileged access to buyers or lower CAC than peers?",
        "brand/category leadership": "Do customers independently name the company as a category leader?",
        "cost advantage": "Can the company deliver materially better unit economics than competitors?",
        "technical differentiation": "What benchmarks, patents, accuracy, reliability, or performance prove technical edge?",
        "supply-side exclusivity": "Are any suppliers, data sources, or partners exclusive and contractually durable?",
        "demand-side aggregation": "Does aggregated demand create pricing power, selection, or lower acquisition costs?",
        "ecosystem partnerships": "Are partnerships strategic, exclusive, revenue-producing, and hard for competitors to replicate?",
    }
    return questions.get(category, "What evidence would prove this moat is durable?")


TRACTION_TYPES = [
    "anecdotal interest",
    "pilots",
    "unpaid pilots",
    "paid pilots",
    "paid customers",
    "active usage",
    "expansion",
    "retention",
    "NRR / upsell",
    "logo quality",
    "revenue scale",
    "repeatable sales motion",
    "implementation success",
    "customer concentration",
]


TRACTION_KEYWORD_MAP = {
    "anecdotal interest": ["interest", "excited", "customer interest", "adoption"],
    "pilots": ["pilot", "pilots", "design partner"],
    "unpaid pilots": ["unpaid pilot"],
    "paid pilots": ["paid pilot"],
    "paid customers": ["paid customer", "customers", "customer count"],
    "active usage": ["active usage", "usage", "weekly active", "daily active"],
    "expansion": ["expansion", "expand", "land and expand"],
    "retention": ["retention", "gross retention", "renewal"],
    "NRR / upsell": ["nrr", "net revenue retention", "upsell"],
    "logo quality": ["fortune 500", "enterprise customer", "logo"],
    "revenue scale": ["arr", "revenue", "$"],
    "repeatable sales motion": ["sales motion", "repeatable sales", "pipeline"],
    "implementation success": ["implementation", "deployment", "go-live", "live deployment"],
    "customer concentration": ["concentration", "top customer", "largest customer"],
}


def _traction_analysis(profile: dict[str, Any], evidence: list[EvidenceItem]) -> dict[str, Any]:
    raw_notes = profile.get("raw_notes", "")
    note_lower = raw_notes.lower()
    evidence_by_id = {
        item["source_id"]: item
        for item in evidence
        if item["evidence_type"] in {"business_model", "news", "website", "funding"}
    }
    rows = []
    for traction_type in TRACTION_TYPES:
        keywords = TRACTION_KEYWORD_MAP[traction_type]
        note_supported = any(keyword in note_lower for keyword in keywords)
        public_ids = [
            item["source_id"]
            for item in evidence_by_id.values()
            if any(keyword in f"{item.get('title', '')} {item.get('snippet', '')}".lower() for keyword in keywords)
            and _source_allows_use(item, "traction verification")
        ][:2]
        if public_ids:
            status = "Partially verified"
            evidence_strength = "Medium"
            interpretation = "Public evidence contains an operating traction signal, but diligence must verify quality and durability."
        elif note_supported:
            status = "Partner notes only"
            evidence_strength = "Low"
            interpretation = "Partner notes suggest this signal, but public evidence does not independently verify it."
        else:
            status = "Unknown"
            evidence_strength = "Low"
            interpretation = "No reliable evidence found."
        rows.append(
            {
                "traction_type": traction_type,
                "status": status,
                "evidence_strength": evidence_strength,
                "interpretation": interpretation,
                "source_ids": public_ids,
                "diligence_needed": _traction_diligence(traction_type),
            }
        )
    return {
        "summary": "Traction appears promising based on partner notes, but investment-grade validation requires ARR, paid customer count, deployment depth, usage frequency, retention, expansion, implementation timeline, and sales efficiency.",
        "rules": [
            "Funding is not traction.",
            "Press coverage is not traction.",
            "Customer logos are not usage.",
            "Pilots are not retention.",
            "ARR without retention is incomplete.",
            "Growth without gross margin can be misleading.",
        ],
        "rows": rows,
    }


def _traction_diligence(traction_type: str) -> str:
    mapping = {
        "anecdotal interest": "Request pipeline conversion, buyer urgency, and reason for urgency.",
        "pilots": "Separate design partners, unpaid pilots, paid pilots, and production deployments.",
        "unpaid pilots": "Confirm conversion rate from unpaid pilot to paid deployment.",
        "paid pilots": "Confirm contract value, scope, success criteria, and conversion to rollout.",
        "paid customers": "Request paid customer count, ACV, live date, and referenceability.",
        "active usage": "Request usage logs, frequency, active seats/users, and workflow completion data.",
        "expansion": "Request account expansion cohorts and module/location/seat growth.",
        "retention": "Request gross retention, logo retention, renewal cohorts, and churn reasons.",
        "NRR / upsell": "Request NRR, upsell motion, expansion driver, and cohort consistency.",
        "logo quality": "Confirm whether logos are paid, live, active, referenceable, and renewing.",
        "revenue scale": "Request ARR, growth bridge, revenue recognition, and contracted backlog.",
        "repeatable sales motion": "Request pipeline, win rate, sales cycle, CAC payback, and quota productivity.",
        "implementation success": "Request deployment timeline, services hours, go-live rate, and support burden.",
        "customer concentration": "Request revenue concentration by top customers and contract renewal dates.",
    }
    return mapping.get(traction_type, "Request direct operating evidence.")


BUSINESS_MODEL_FIELDS = [
    "pricing model",
    "buyer",
    "budget owner",
    "sales motion",
    "ACV range",
    "implementation burden",
    "gross margin drivers",
    "services component",
    "expansion motion",
    "renewal motion",
    "usage-based vs seat-based vs transaction-based pricing",
    "customer concentration risk",
]


def _business_model_analysis(profile: dict[str, Any], evidence: list[EvidenceItem], memo: dict[str, Any]) -> dict[str, Any]:
    understanding = memo.get("company_understanding", {})
    source_ids = _reliable_source_ids_for(evidence, "business_model", use="revenue quality")
    rows = []
    for field in BUSINESS_MODEL_FIELDS:
        value, confidence, rationale = _business_model_field_value(field, understanding, evidence, source_ids)
        rows.append(
            {
                "field": field,
                "value": value,
                "confidence": confidence,
                "rationale": rationale,
                "source_ids": source_ids if confidence != "Low" else [],
                "diligence_needed": _business_model_diligence(field),
            }
        )
    return {
        "summary": "Business model conclusions remain provisional unless pricing, ACV, gross margin, services burden, retention, and expansion are supported by evidence.",
        "rows": rows,
    }


def _business_model_field_value(field: str, understanding: dict[str, Any], evidence: list[EvidenceItem], source_ids: list[str]) -> tuple[str, str, str]:
    if field == "buyer":
        buyer = understanding.get("target_buyer", "Unknown")
        return (buyer if not _is_placeholder_entity(buyer) else "Unknown", "Medium" if buyer != "Unknown" else "Low", "Derived from company understanding / notes.")
    if field == "budget owner":
        owner = understanding.get("budget_owner", "Unknown")
        return (owner if not _is_placeholder_entity(owner) else "Unknown", "Medium" if owner != "Unknown" else "Low", "Derived from product category and buyer.")
    if field == "pricing model":
        model = understanding.get("business_model", "Unknown")
        return (model if source_ids else "Unknown", "Medium" if source_ids else "Low", "Requires public business-model evidence; do not infer attractive economics from category alone.")
    text = " ".join(f"{item.get('title', '')} {item.get('snippet', '')}" for item in evidence if item["source_id"] in source_ids).lower()
    keyword_map = {
        "sales motion": ["sales", "enterprise", "self-serve", "channel"],
        "ACV range": ["acv", "annual contract", "$"],
        "implementation burden": ["implementation", "deployment", "services"],
        "gross margin drivers": ["gross margin", "margin", "inference cost", "services"],
        "services component": ["services", "implementation", "managed service"],
        "expansion motion": ["expansion", "upsell", "land and expand"],
        "renewal motion": ["renewal", "retention"],
        "usage-based vs seat-based vs transaction-based pricing": ["usage", "seat", "transaction", "subscription"],
        "customer concentration risk": ["concentration", "top customer"],
    }
    if any(keyword in text for keyword in keyword_map.get(field, [])):
        return "Evidence found; requires quantification", "Medium", "Public business-model evidence contains a directional signal."
    return "Unknown", "Low", "Not available from current evidence."


def _business_model_diligence(field: str) -> str:
    mapping = {
        "pricing model": "Request pricing metric, packaging, discounting, and contract terms.",
        "buyer": "Confirm economic buyer, user, champion, and procurement owner.",
        "budget owner": "Confirm budget line item and replacement target.",
        "sales motion": "Request sales cycle, channel mix, win rate, and quota productivity.",
        "ACV range": "Request ACV distribution, expansion by cohort, and discounting.",
        "implementation burden": "Request implementation timeline, services hours, and go-live rate.",
        "gross margin drivers": "Request gross margin bridge, model/infrastructure costs, support, and services margin.",
        "services component": "Quantify implementation and ongoing services as a percentage of revenue.",
        "expansion motion": "Request expansion cohorts and attach-rate by product/module/location.",
        "renewal motion": "Request gross retention, renewal timing, churn reasons, and NRR.",
        "usage-based vs seat-based vs transaction-based pricing": "Confirm pricing metric and alignment with customer value.",
        "customer concentration risk": "Request revenue by customer and renewal exposure.",
    }
    return mapping.get(field, "Request supporting business model evidence.")


DEFAULT_RISK_TAXONOMY = [
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
]


def _risk_taxonomy(profile: dict[str, Any], evidence: list[EvidenceItem], memo: dict[str, Any]) -> dict[str, Any]:
    understanding = memo.get("company_understanding", {})
    risks = memo.get("visualizations", {}).get("risk_breakdown", {}).get("risks", [])
    risk_text = " ".join(risk.get("risk_name", "") + " " + risk.get("rationale", "") for risk in risks).lower()
    rows = []
    for category in DEFAULT_RISK_TAXONOMY:
        related = [risk for risk in risks if _risk_category_match(category, risk, risk_text)]
        severity = _taxonomy_severity(category, related)
        source_ids = list(dict.fromkeys(source_id for risk in related for source_id in risk.get("source_ids", [])))[:3]
        confidence = "Medium" if source_ids or related else "Low"
        rows.append(
            {
                "risk_category": category,
                "description": _risk_category_description(category, understanding),
                "source_of_risk": _risk_source(category, related),
                "severity": severity,
                "evidence_confidence": confidence,
                "diligence_question": _risk_category_question(category),
                "mitigation_hypothesis": _risk_mitigation(category),
                "source_ids": source_ids,
            }
        )
    return {
        "summary": "Risks are organized by an industry-agnostic taxonomy so the memo captures both company-specific and generic investment risks.",
        "rows": rows,
    }


def _risk_category_match(category: str, risk: dict[str, Any], risk_text: str) -> bool:
    text = f"{risk.get('risk_name', '')} {risk.get('rationale', '')}".lower()
    keywords = {
        "product risk": ["product", "differentiation", "accuracy", "workflow"],
        "technical risk": ["technical", "accuracy", "reliability", "model", "hardware"],
        "adoption risk": ["adoption", "trust", "change management", "workflow"],
        "market timing risk": ["market", "timing", "cyclicality", "procurement"],
        "competition risk": ["competition", "competitor", "incumbent", "commoditization"],
        "pricing / willingness-to-pay risk": ["pricing", "roi", "willingness", "budget"],
        "gross margin risk": ["margin", "services", "unit economics", "inference"],
        "implementation risk": ["implementation", "integration", "deployment"],
        "retention risk": ["retention", "churn", "renewal"],
        "regulatory / compliance risk": ["regulatory", "compliance", "fda", "hipaa", "export"],
        "data/privacy/security risk": ["data", "privacy", "security", "confidentiality"],
        "concentration risk": ["concentration", "program", "top customer"],
        "platform dependency risk": ["dependency", "platform", "ehr", "erp", "banking partner"],
        "financing risk": ["financing", "runway", "capital"],
        "team/execution risk": ["team", "execution", "hiring"],
    }
    return any(keyword in text for keyword in keywords.get(category, []))


def _taxonomy_severity(category: str, related: list[dict[str, Any]]) -> str:
    scores = [risk.get("score") for risk in related if isinstance(risk.get("score"), (int, float))]
    if not scores:
        return "Unknown"
    avg = sum(scores) / len(scores)
    if avg >= 8:
        return "High"
    if avg >= 6:
        return "Medium"
    return "Low"


def _risk_source(category: str, related: list[dict[str, Any]]) -> str:
    if related:
        names = ", ".join(risk.get("risk_name", category) for risk in related[:2])
        return f"Mapped from generated risk(s): {names}."
    return "Default taxonomy risk; no company-specific evidence yet."


def _risk_category_description(category: str, understanding: dict[str, Any]) -> str:
    workflow = understanding.get("core_workflow", "the workflow")
    return f"Risk that {category.replace(' risk', '')} issues reduce adoption, economics, or defensibility in {workflow}."


def _risk_category_question(category: str) -> str:
    questions = {
        "product risk": "What evidence proves the product solves a high-priority workflow better than alternatives?",
        "technical risk": "What benchmarks, reliability data, accuracy data, or technical reviews reduce this risk?",
        "adoption risk": "What usage and change-management data proves repeated adoption?",
        "market timing risk": "Why is now the right buying window?",
        "competition risk": "What win/loss evidence proves differentiation versus startups and incumbents?",
        "pricing / willingness-to-pay risk": "What budget, ROI, and pricing evidence proves willingness to pay?",
        "gross margin risk": "What gross margin bridge proves economics after services and infrastructure costs?",
        "implementation risk": "How long does deployment take and what services are required?",
        "retention risk": "What renewal, churn, NRR, and usage cohorts prove durability?",
        "regulatory / compliance risk": "What compliance requirements could delay or block adoption?",
        "data/privacy/security risk": "What data, security, privacy, and audit controls are required?",
        "concentration risk": "How concentrated are revenue, pipeline, suppliers, or partners?",
        "platform dependency risk": "Which third-party platforms can block, bundle, or tax the product?",
        "financing risk": "How much runway is needed to reach the next proof point?",
        "team/execution risk": "What team gaps could prevent execution?",
    }
    return questions.get(category, "What evidence would reduce this risk?")


def _risk_mitigation(category: str) -> str:
    return {
        "product risk": "Run product demo, customer calls, and workflow-depth review.",
        "technical risk": "Conduct technical diligence and benchmark review.",
        "adoption risk": "Verify active usage, training burden, and user pull.",
        "market timing risk": "Validate budget urgency with buyers and market data.",
        "competition risk": "Run win/loss calls and pricing comparisons.",
        "pricing / willingness-to-pay risk": "Request ROI cases, pricing history, and discounting.",
        "gross margin risk": "Review gross margin bridge and services mix.",
        "implementation risk": "Audit deployment timeline and customer success workload.",
        "retention risk": "Review renewal cohorts, churn reasons, and NRR.",
        "regulatory / compliance risk": "Run compliance/legal review.",
        "data/privacy/security risk": "Review security posture, data handling, and customer requirements.",
        "concentration risk": "Review customer, supplier, and partner concentration.",
        "platform dependency risk": "Assess platform APIs, partner agreements, and bundling risk.",
        "financing risk": "Review burn, runway, milestones, and financing options.",
        "team/execution risk": "Conduct references and org gap review.",
    }.get(category, "Define mitigation through diligence.")


def _evidence_dashboard(evidence: list[EvidenceItem]) -> dict[str, Any]:
    topic_map = {
        "Product": "website",
        "Team": "leadership",
        "Funding": "funding",
        "Competition": "competitor",
        "Business Model": "business_model",
        "Traction": "news",
        "Retention": "business_model",
        "Market": "market",
    }
    heatmap = []
    strong = []
    weak = []
    for topic, evidence_type in topic_map.items():
        counts = _tier_counts(evidence, evidence_type)
        quality = _coverage_quality(evidence, evidence_type)
        heatmap.append(
            {
                "topic": topic,
                "evidence_type": evidence_type,
                "tier_1_sources": counts[1],
                "tier_2_sources": counts[2],
                "tier_3_sources": counts[3],
                "weighted_score": _weighted_evidence_score(evidence, evidence_type),
                "quality": quality,
            }
        )
        if quality == "Strong":
            strong.append(topic)
        if quality in {"Weak", "Unknown"}:
            weak.append(topic)
    total_counts = _tier_counts(evidence)
    warnings = []
    for row in heatmap:
        if row["quality"] in {"Weak", "Unknown"}:
            warnings.append(f"{row['topic']} evidence is {row['quality'].lower()}; avoid strong conclusions without further diligence.")
        elif row["tier_3_sources"] > row["tier_1_sources"] + row["tier_2_sources"]:
            warnings.append(f"{row['topic']} relies heavily on Tier 3 supplemental sources.")
    if not any(item["evidence_type"] == "business_model" and item.get("source_tier", 3) <= 2 for item in evidence):
        warnings.append("ARR, retention, pricing, and business-model quality are unavailable from reliable public evidence.")
    return {
        "tier_counts": {
            "tier_1": total_counts[1],
            "tier_2": total_counts[2],
            "tier_3": total_counts[3],
        },
        "heatmap": heatmap,
        "strong_evidence": strong,
        "weak_evidence": weak,
        "warnings": warnings,
    }


def _evidence_quality_score(evidence: list[EvidenceItem]) -> float:
    if not evidence:
        return 0.0
    investment_usable = [item for item in evidence if _source_allows_use(item, "investment scoring")]
    weighted = sum(float(item.get("source_weight", 0.2)) for item in investment_usable)
    reliable_types = {item["evidence_type"] for item in investment_usable if item.get("source_tier", 3) <= 2}
    diversity_bonus = min(0.25, len(reliable_types) * 0.05)
    return round(min(1.0, weighted / 5.0 + diversity_bonus), 2)


def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def _calibrated_overall_confidence(profile: dict[str, Any], memo: dict[str, Any], evidence: list[EvidenceItem]) -> dict[str, Any]:
    identity = profile.get("identity_resolution", {})
    understanding = memo.get("company_understanding", {})
    identity_confidence = float(identity.get("confidence", 1.0))
    understanding_confidence = float(understanding.get("validation", {}).get("confidence", 0.0))
    evidence_quality = _evidence_quality_score(evidence)
    bounded_score = min(identity_confidence, understanding_confidence, evidence_quality)
    return {
        "label": _confidence_label(bounded_score),
        "score": round(bounded_score, 2),
        "identity_confidence": round(identity_confidence, 2),
        "understanding_confidence": round(understanding_confidence, 2),
        "evidence_quality_score": evidence_quality,
        "rule": "Overall confidence = min(identity_confidence, understanding_confidence, evidence_quality_score).",
    }


COVERAGE_REQUIREMENTS = {
    "Product": ["product description", "customer use case", "workflow", "differentiation", "product expansion"],
    "Team": ["founder background", "domain expertise", "execution track record", "leadership depth", "hiring quality"],
    "Market": ["TAM", "growth", "budget urgency", "market timing", "buyer pain"],
    "Competition": ["direct competitors", "adjacent competitors", "incumbent competitors", "pricing overlap", "win/loss data"],
    "Business Model": ["pricing", "revenue model", "gross margin", "ACV", "retention"],
    "Funding": ["latest round", "investors", "valuation", "cash runway", "financing history"],
    "Traction": ["ARR", "growth rate", "customer count", "NRR", "win rate"],
    "Customers": ["target buyer", "live customers", "customer segments", "references", "deployment depth"],
    "Risks": ["industry risks", "technical risks", "regulatory risks", "dependency risks", "competitive risks"],
    "Investment Thesis": ["why now", "why this company", "defensibility", "upside case", "key diligence gates"],
}


COVERAGE_SLOT_KEYWORDS = {
    "product description": ["product", "platform", "software", "automates", "provides", "uses"],
    "customer use case": ["customer", "buyer", "use case", "users", "teams", "clinicians", "finance", "legal teams", "general counsel"],
    "workflow": ["workflow", "expense", "documentation", "procurement", "clinical", "spend", "authorization", "contract review", "redlining", "clause extraction"],
    "differentiation": ["differentiation", "different", "advantage", "unique", "moat", "integrated"],
    "product expansion": ["expansion", "module", "suite", "platform", "adjacent"],
    "founder background": ["founder", "ceo", "founded", "leadership"],
    "domain expertise": ["experience", "background", "domain", "expertise"],
    "execution track record": ["track record", "previous", "former", "built", "scaled"],
    "leadership depth": ["team", "executives", "leadership"],
    "hiring quality": ["hiring", "employees", "talent"],
    "TAM": ["tam", "market size", "market"],
    "growth": ["growth", "growing", "cagr"],
    "budget urgency": ["budget", "urgency", "pain", "priority", "cost"],
    "market timing": ["why now", "timing", "adoption", "tailwind"],
    "buyer pain": ["pain", "burden", "manual", "costly", "friction"],
    "direct competitors": ["direct competitor", "direct competitors", "same workflow competitor", "same buyer competitor", "versus"],
    "adjacent competitors": ["adjacent competitor", "adjacent competitors", "adjacent alternative", "adjacent alternatives"],
    "incumbent competitors": ["incumbent competitor", "incumbent competitors", "incumbent alternative", "legacy competitor", "legacy alternative"],
    "pricing overlap": ["pricing overlap", "price overlap", "same budget", "budget overlap", "pricing comparison", "price comparison"],
    "win/loss data": ["win/loss", "win loss", "competitive win rate", "loss reason", "loss reasons", "deal overlap"],
    "pricing": ["pricing", "price", "subscription", "fee"],
    "revenue model": ["revenue", "business model", "interchange", "saas", "contract"],
    "gross margin": ["gross margin", "margin"],
    "ACV": ["acv", "contract value"],
    "retention": ["retention", "nrr", "renewal", "churn"],
    "latest round": ["series", "raised", "funding", "round"],
    "investors": ["investor", "backed by"],
    "valuation": ["valuation", "valued"],
    "cash runway": ["runway", "cash"],
    "financing history": ["funding history", "financing"],
    "ARR": ["arr", "annual recurring"],
    "growth rate": ["growth rate"],
    "customer count": ["customers", "clients", "health systems", "companies"],
    "NRR": ["nrr", "net revenue retention"],
    "win rate": ["win rate"],
    "target buyer": ["buyer", "cfo", "health system", "clinician", "finance team", "legal teams", "general counsel"],
    "live customers": ["live", "deployed", "production"],
    "customer segments": ["segment", "smb", "enterprise", "mid-market"],
    "references": ["reference", "case study", "testimonial"],
    "deployment depth": ["deployment", "rollout", "adoption"],
    "industry risks": ["risk", "regulatory", "industry"],
    "technical risks": ["technical", "integration", "accuracy", "hallucinated", "clause interpretation"],
    "regulatory risks": ["regulatory", "compliance", "hipaa", "fraud", "confidentiality", "privacy"],
    "dependency risks": ["dependency", "partner", "epic", "banking", "clm", "document repository"],
    "competitive risks": ["competitive", "competition"],
    "why now": ["why now", "timing", "tailwind"],
    "why this company": ["advantage", "differentiation", "positioned"],
    "defensibility": ["defensible", "moat", "lock-in"],
    "upside case": ["upside", "opportunity", "expansion"],
    "key diligence gates": ["diligence", "verify", "unknown"],
}


def _evidence_text_for_coverage(evidence: list[EvidenceItem], memo: dict[str, Any]) -> str:
    return " ".join(f"{item['title']} {item['snippet']}" for item in evidence).lower()


def _coverage_status(coverage: int) -> str:
    if coverage >= 90:
        return "Strong"
    if coverage >= 70:
        return "Good"
    if coverage >= 40:
        return "Partial"
    if coverage >= 10:
        return "Weak"
    return "Unknown"


def _evidence_coverage(evidence: list[EvidenceItem], memo: dict[str, Any]) -> dict[str, Any]:
    text = _evidence_text_for_coverage(evidence, memo)
    section_confidence = memo.get("section_confidence", {})
    rows = []
    high_priority = []
    medium_priority = []
    for category, slots in COVERAGE_REQUIREMENTS.items():
        found = []
        missing = []
        for slot in slots:
            keywords = COVERAGE_SLOT_KEYWORDS.get(slot, [slot])
            if any(keyword.lower() in text for keyword in keywords):
                found.append(slot)
            else:
                missing.append(slot)
        coverage = round((len(found) / len(slots)) * 100) if slots else 0
        status = _coverage_status(coverage)
        confidence = section_confidence.get(category)
        if confidence is None:
            confidence = {
                "Product": section_confidence.get("Product", "Low"),
                "Team": section_confidence.get("Founding Team", "Low"),
                "Market": section_confidence.get("Market Opportunity", "Low"),
                "Competition": section_confidence.get("Competitive Landscape", "Low"),
                "Business Model": section_confidence.get("Business Model", "Low"),
                "Funding": section_confidence.get("Recent Funding", "Low"),
                "Investment Thesis": section_confidence.get("Investment Thesis", "Low"),
            }.get(category, "Low")
        row = {
            "category": category,
            "coverage": coverage,
            "status": status,
            "confidence": confidence,
            "found_slots": found,
            "missing_slots": missing,
        }
        rows.append(row)
        if coverage < 40:
            high_priority.extend(missing[:2])
        elif coverage < 70:
            medium_priority.extend(missing[:2])
    focus = sorted(rows, key=lambda row: row["coverage"])[:3]
    return {
        "rows": rows,
        "high_priority_gaps": list(dict.fromkeys(high_priority))[:8],
        "medium_priority_gaps": list(dict.fromkeys(medium_priority))[:8],
        "recommended_diligence_focus": [
            {
                "category": row["category"],
                "coverage": row["coverage"],
                "status": row["status"],
                "reason": f"{row['category']} has {row['coverage']}% coverage; missing {', '.join(row['missing_slots'][:3]) or 'no major slots'}.",
            }
            for row in focus
        ],
    }


GAP_CATEGORY_MAP = {
    "ARR": "Commercial",
    "growth rate": "Commercial",
    "NRR": "Commercial",
    "customer count": "Commercial",
    "win rate": "Commercial",
    "pricing": "Financial",
    "ACV": "Financial",
    "gross margin": "Financial",
    "retention": "Commercial",
    "cash runway": "Financial",
    "deployment depth": "Product",
    "workflow": "Product",
    "product expansion": "Product",
    "win/loss data": "Competition",
    "pricing overlap": "Competition",
    "incumbent competitors": "Competition",
    "differentiation": "Competition",
    "founder background": "Team",
    "leadership depth": "Team",
}


BLOCKING_GAPS = {"ARR", "growth rate", "NRR", "customer count", "gross margin", "retention", "win/loss data"}


def _gap_category(slot: str) -> str:
    for key, category in GAP_CATEGORY_MAP.items():
        if key.lower() in slot.lower():
            return category
    return "Commercial"


def _score_gap(slot: str, coverage: int) -> dict[str, Any]:
    business_impact = 5 if slot in BLOCKING_GAPS else 4 if slot in {"pricing", "ACV", "deployment depth", "differentiation"} else 3
    decision_impact = 5 if slot in BLOCKING_GAPS else 4 if coverage < 40 else 3
    evidence_gap = 5 if coverage < 20 else 4 if coverage < 40 else 3 if coverage < 70 else 1
    priority_score = business_impact + decision_impact + evidence_gap
    if priority_score >= 13:
        priority = "High"
    elif priority_score >= 10:
        priority = "Medium"
    else:
        priority = "Low"
    return {
        "gap": slot,
        "category": _gap_category(slot),
        "business_impact": business_impact,
        "decision_impact": decision_impact,
        "evidence_gap": evidence_gap,
        "priority_score": priority_score,
        "priority": priority,
    }


def _missing_data_detection(memo: dict[str, Any]) -> dict[str, Any]:
    coverage = memo.get("evidence_coverage", {})
    rows = coverage.get("rows", [])
    section_status = []
    all_gaps = []
    for row in rows:
        known = row.get("found_slots", [])
        unknown = row.get("missing_slots", [])
        if row.get("coverage", 0) >= 70:
            status = "Known"
        elif row.get("coverage", 0) >= 20:
            status = "Partially Known"
        else:
            status = "Unknown"
        section_status.append({"section": row["category"], "status": status, "known": known, "unknown": unknown})
        for slot in unknown:
            all_gaps.append(_score_gap(slot, row.get("coverage", 0)))

    deduped: dict[str, dict[str, Any]] = {}
    for gap in sorted(all_gaps, key=lambda item: item["priority_score"], reverse=True):
        deduped.setdefault(gap["gap"], gap)
    gaps = list(deduped.values())
    high = [gap for gap in gaps if gap["priority"] == "High"]
    medium = [gap for gap in gaps if gap["priority"] == "Medium"]
    low = [gap for gap in gaps if gap["priority"] == "Low"]
    blocking = [gap for gap in gaps if gap["gap"] in BLOCKING_GAPS][:8]
    avg_coverage = round(sum(row.get("coverage", 0) for row in rows) / len(rows)) if rows else 0
    blocking_penalty = min(25, len(blocking) * 4)
    decision_readiness = max(0, min(100, avg_coverage - blocking_penalty))
    if blocking:
        recommendation_adjustment = "Proceed to diligence, but investment decision is premature until blocking unknowns are resolved."
    else:
        recommendation_adjustment = "Available evidence is sufficient for a preliminary view, subject to confirmatory diligence."
    return {
        "decision_readiness": decision_readiness,
        "section_status": section_status,
        "high_priority": high,
        "medium_priority": medium,
        "low_priority": low,
        "decision_blocking_unknowns": blocking,
        "recommendation_adjustment": recommendation_adjustment,
    }


def _note_lines(raw_notes: str) -> list[str]:
    return [line.strip(" -\t") for line in raw_notes.splitlines() if line.strip(" -\t")]


def _source_note_reference_for_question(question: str, raw_notes: str) -> str:
    lines = _note_lines(raw_notes)
    lower_question = question.lower()
    keyword_groups = {
        "product": ["product", "platform", "workflow"],
        "buyer": ["buyer", "customer", "customers", "end user", "budget"],
        "competition": ["competition", "competitor", "competitors", "competitive", "vendor", "vendors"],
        "revenue": ["revenue", "pricing", "business model", "arr", "acv", "gross margin", "retention"],
        "roi": ["roi", "why customers buy", "value", "payback"],
        "risk": ["risk", "security", "compliance", "regulatory", "privacy"],
        "traction": ["traction", "customers", "fortune 500", "usage", "growth"],
        "team": ["founder", "team", "sold a startup"],
        "identity": ["company", "website", "category", "actual product"],
        "market": ["market", "tam"],
    }
    wanted = {
        key
        for key, terms in keyword_groups.items()
        if any(term in lower_question for term in terms)
    }
    if any(term in lower_question for term in ["competition", "competitor", "competitors", "vendor", "vendors"]):
        wanted = {"competition"}
    best_line = ""
    best_score = 0
    for line in lines:
        lower_line = line.lower()
        if not any(term in lower_line for terms in keyword_groups.values() for term in terms):
            continue
        matched_groups = {
            key
            for key, terms in keyword_groups.items()
            if any(term in lower_line for term in terms)
        }
        if wanted and not (matched_groups & wanted):
            continue
        exact_bonus = 0
        score = len(matched_groups & wanted) * 3 + len(matched_groups) + exact_bonus
        if score > best_score:
            best_score = score
            best_line = line
    return best_line


def _ground_diligence_priorities(raw_notes: str, priorities: dict[str, list[str]]) -> dict[str, list[dict[str, str]]]:
    grounded: dict[str, list[dict[str, str]]] = {}
    for category, items in priorities.items():
        grounded_items = []
        for item in items:
            question = item.get("question", "") if isinstance(item, dict) else str(item)
            reference = item.get("source_note_reference", "") if isinstance(item, dict) else ""
            reference = reference or _source_note_reference_for_question(question, raw_notes)
            if reference:
                grounded_items.append({"question": question, "source_note_reference": reference})
        if grounded_items:
            grounded[category] = grounded_items
    return grounded


def _unknown_category_diligence_priorities(company: str, profile: dict[str, Any]) -> dict[str, list[str]]:
    raw_notes = profile.get("raw_notes", "")
    questions: dict[str, list[str]] = {
        "Commercial traction": [],
        "Customer ROI": [],
        "Product / integration": [],
        "Competition": [],
        "Financials": [],
        "Regulatory / security": [],
    }
    if _source_note_reference_for_question("Clarify actual product and workflow.", raw_notes):
        questions["Product / integration"].append(f"Clarify {company}'s actual product and workflow before workflow-level analysis.")
    if _source_note_reference_for_question("Clarify buyer and customer segment.", raw_notes):
        questions["Customer ROI"].append("Clarify the end user, customer segment, economic buyer, and why customers buy.")
        questions["Commercial traction"].append("Validate which customer claims are current, live, paid, and referenceable.")
    if _source_note_reference_for_question("Clarify competition.", raw_notes):
        questions["Competition"].append("Identify competitors only after confirming the product workflow, buyer, and budget owner.")
    if _source_note_reference_for_question("Clarify revenue model pricing ARR ACV.", raw_notes):
        questions["Financials"].append("Clarify revenue model, pricing, ARR/ACV, gross margin, retention, and sales motion.")
    if _source_note_reference_for_question("Clarify risk security compliance regulatory.", raw_notes):
        questions["Regulatory / security"].append("Clarify which security, privacy, compliance, or regulatory requirements apply to the confirmed category.")
    if not any(questions.values()) and raw_notes.strip():
        questions["Product / integration"].append("Clarify the actual product, buyer, workflow, and category using the current partner notes.")
    return {category: items for category, items in questions.items() if items}


def _next_diligence_priorities(company: str, profile: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    sector = profile.get("sector", "Unknown")
    terms = _domain_terms(profile)
    if sector == "LegalTech":
        return _ground_diligence_priorities(profile.get("raw_notes", ""), {
            "Commercial traction": [
                f"Validate {company}'s ARR, growth rate, ACV, legal seat adoption, contract volume, and sales-cycle length.",
                "Separate pilots, paid departmental rollouts, and enterprise-wide legal deployments.",
            ],
            "Customer ROI": [
                "Quantify contract review cycle-time reduction, outside counsel savings, clause-risk reduction, and attorney productivity.",
                "Run reference calls with General Counsel, legal operations, and frontline attorneys on trust, usage, and payback.",
            ],
            "Product / integration": [
                "Review redlining quality, clause extraction accuracy, risk scoring, playbook configurability, and attorney override workflows.",
                "Assess integrations with CLM, document repositories, Microsoft Word, e-signature, and matter-management systems.",
            ],
            "Competition": [
                "Map deal overlap with Ironclad, Harvey, Lexion, LinkSquares, Evisort, Spellbook, DocuSign CLM, and legal AI incumbents.",
                "Identify whether differentiation comes from accuracy, workflow depth, legal playbooks, integrations, or distribution.",
            ],
            "Financials": [
                "Request pricing, ACV, gross margin after model/services costs, CAC, burn, runway, and expansion by legal department.",
                "Stress-test whether growth depends on high implementation or legal-services support cost.",
            ],
            "Regulatory / security": [
                f"Verify {terms['security']}.",
                "Review confidentiality, privilege, data retention, customer data use for model training, and enterprise audit requirements.",
            ],
        })
    if sector == "Fintech / Spend Management":
        return _ground_diligence_priorities(profile.get("raw_notes", ""), {
            "Commercial traction": [
                f"Validate {company}'s ARR, software attach rate, ACV, pipeline quality, win rate, and sales-cycle length.",
                "Separate software revenue from interchange, float, bill pay, and procurement monetization.",
            ],
            "Customer ROI": [
                "Quantify finance-team time savings, spend reduction, policy compliance, and procurement-cycle improvements.",
                "Run reference calls with CFOs/controllers on adoption, controls, reporting quality, and payback.",
            ],
            "Product / integration": [
                "Review ERP, accounting, payroll, HRIS, card issuing, bill pay, and procurement integrations.",
                "Assess workflow depth across card spend, expense management, AP, procurement, approvals, and reporting.",
            ],
            "Competition": [
                "Map deal overlap with Brex, Airbase, Navan, Amex, Coupa, banks, and ERP/procurement suites.",
                "Identify whether differentiation comes from software depth, savings, underwriting, distribution, or bundling.",
            ],
            "Financials": [
                "Request revenue mix, gross margin by product, interchange economics, credit/fraud losses, CAC, burn, and runway.",
                "Stress-test margin durability if interchange compresses or software attach lags.",
            ],
            "Regulatory / security": [
                f"Verify {terms['security']}.",
                "Review card program partners, money movement controls, underwriting exposure, and financial audit requirements.",
            ],
        })
    if sector == "Defense Technology":
        return _ground_diligence_priorities(profile.get("raw_notes", ""), {
            "Commercial traction": [
                f"Validate {company}'s awarded contracts, backlog, recompete risk, program concentration, and pipeline.",
                "Separate prototypes, pilots, IDIQ ceilings, and funded production programs.",
            ],
            "Customer ROI": [
                "Quantify mission effectiveness, cost per capability, deployment speed, and operational reliability.",
                "Run customer/reference calls focused on procurement sponsor, field adoption, and expansion path.",
            ],
            "Product / integration": [
                "Review hardware/software maturity, field reliability, supply chain, autonomy stack, and mission-system integration.",
                "Assess production scalability and support burden.",
            ],
            "Competition": [
                "Map deal overlap with defense primes, Palantir, Shield AI, General Atomics, and specialized defense tech vendors.",
                "Understand whether the company wins on speed, cost, capability, integration, or procurement path.",
            ],
            "Financials": [
                "Request revenue by program, backlog conversion, gross margin by hardware/software mix, working capital, burn, and runway.",
                "Stress-test cash needs under delayed government procurement cycles.",
            ],
            "Regulatory / security": [
                f"Verify {terms['security']}.",
                "Review export controls, classified work, clearances, cyber requirements, and government contracting compliance.",
            ],
        })
    if sector in {"Unknown", "Conflicting signals detected"} or profile.get("company_understanding", {}).get("category") == "Conflicting signals detected":
        return _ground_diligence_priorities(
            profile.get("raw_notes", ""),
            _unknown_category_diligence_priorities(company, profile),
        )
    return _ground_diligence_priorities(profile.get("raw_notes", ""), {
        "Commercial traction": [
            f"Validate {company}'s ARR, growth rate, ACV, pipeline quality, win rate, and sales-cycle length.",
            "Separate live production customers from pilots, design partners, and unpaid evaluations.",
        ],
        "Customer ROI": [
            "Collect before/after workflow metrics and customer payback analysis.",
            "Run reference calls focused on adoption, time saved, quality lift, and budget owner.",
        ],
        "Product / integration": [
            "Review implementation timeline, EHR/system integrations, data requirements, and services load.",
            "Assess how much workflow is automated versus routed to humans or customer operations teams.",
        ],
        "Competition": [
            "Map top competitors by deal overlap, buyer perception, pricing, and win/loss reasons.",
            "Identify whether incumbents can bundle similar functionality into existing contracts.",
        ],
        "Financials": [
            "Request revenue cohort data, gross margin, inference/model costs, services margin, burn, and runway.",
            "Stress-test whether growth depends on unusually high implementation or support cost.",
        ],
        "Regulatory / security": [
            f"Verify {terms['security']}.",
            "Confirm customer contracting constraints around PHI, BAAs, and clinical risk allocation.",
        ],
    })


def _competitive_table(profile: dict[str, Any], evidence: list[EvidenceItem]) -> list[dict[str, Any]]:
    understanding = profile.get("company_understanding", {})
    competitors = _discover_economic_competitors(profile, evidence)
    if not competitors:
        return [
            {
                "company": "Relevant competitors could not be confidently identified from current evidence.",
                "buyer": understanding.get("target_buyer", "Unknown"),
                "workflow": understanding.get("core_workflow", "Unknown"),
                "threat_level": "Low confidence",
                "why_it_competes": "Company profile is insufficient to identify economic competitors without fabricating.",
                "competitor_score": None,
                "category": "Unknown",
                "source_ids": "",
            }
        ]
    return competitors


COMPETITOR_GROUPS = [
    "direct startups",
    "incumbent vendors",
    "horizontal platforms",
    "internal build / manual process",
    "services firms",
    "open-source / low-cost alternatives",
    "adjacent workflow tools",
    "marketplace alternatives",
    "system-of-record vendors",
]


def _competitive_landscape(profile: dict[str, Any], evidence: list[EvidenceItem]) -> dict[str, Any]:
    understanding = profile.get("company_understanding", {})
    target_company = understanding.get("company_name") or profile.get("raw_company_name") or profile.get("company", "")
    buyer = understanding.get("target_buyer", "Unknown")
    budget_owner = understanding.get("budget_owner", "Unknown")
    workflow = understanding.get("core_workflow", "Unknown")
    category = understanding.get("product_category", "Unknown")
    competitor_sources = _source_ids_for(evidence, "competitor", use="competitor discovery")
    framework = _filter_competitor_framework(
        _competitor_framework(understanding.get("primary_product", profile.get("product", "")), buyer),
        target_company,
    )
    group_names = framework.get("competitor_groups", {})
    rows = []
    direct_names = _filter_target_company_aliases(group_names.get("Direct Competitors", [])[:5], target_company)
    incumbent_names = _filter_target_company_aliases(group_names.get("Incumbents", [])[:5], target_company)
    adjacent_names = _filter_target_company_aliases(group_names.get("Adjacent Competitors", [])[:5], target_company)
    group_to_names = {
        "direct startups": direct_names,
        "incumbent vendors": incumbent_names,
        "horizontal platforms": ["Microsoft", "Google", "Salesforce"] if buyer != "Unknown" else [],
        "internal build / manual process": ["Internal build", "Manual spreadsheet / email process"],
        "services firms": ["Consultants / agencies", "BPO or managed-services providers"],
        "open-source / low-cost alternatives": ["Open-source tools", "Low-cost point tools"],
        "adjacent workflow tools": adjacent_names,
        "marketplace alternatives": ["Marketplace alternatives"] if "marketplace" in str(understanding).lower() else [],
        "system-of-record vendors": _system_of_record_vendors(profile),
    }
    for group in COMPETITOR_GROUPS:
        names = _filter_target_company_aliases(group_to_names.get(group, []), target_company)
        if not names and group not in {"internal build / manual process", "incumbent vendors"}:
            names = []
        threat = _group_threat_level(group, bool(competitor_sources), bool(names))
        rows.append(
            {
                "competitor_group": group,
                "examples": names,
                "why_it_competes": _group_competition_reason(group, buyer, budget_owner, workflow, category),
                "buyer_overlap": _group_overlap(group, "buyer"),
                "workflow_overlap": _group_overlap(group, "workflow"),
                "budget_overlap": _group_overlap(group, "budget"),
                "likely_threat_level": threat,
                "evidence_quality": _coverage_quality(evidence, "competitor", use="competitor discovery") if competitor_sources else "Profile-derived / requires validation",
                "source_ids": competitor_sources,
            }
        )
    return {
        "summary": f"Competition is mapped by economic overlap with {buyer}, {budget_owner} budget, and the {workflow} workflow; this avoids treating keyword-adjacent companies as true competitors.",
        "rows": rows,
    }


def _system_of_record_vendors(profile: dict[str, Any]) -> list[str]:
    sector = profile.get("sector", "")
    if sector == "LegalTech":
        return ["Ironclad", "DocuSign CLM", "iManage"]
    if sector == "Fintech / Spend Management":
        return ["SAP Concur", "Coupa", "Oracle NetSuite"]
    if sector == "Healthcare AI":
        return ["Epic", "Oracle Cerner"]
    if sector == "HRTech":
        return ["Workday", "ADP", "SAP SuccessFactors"]
    if sector == "Supply Chain":
        return ["SAP SCM", "Oracle SCM", "Manhattan Associates"]
    return []


def _group_overlap(group: str, dimension: str) -> str:
    scores = {
        "direct startups": {"buyer": "High", "workflow": "High", "budget": "High"},
        "incumbent vendors": {"buyer": "High", "workflow": "Medium", "budget": "High"},
        "horizontal platforms": {"buyer": "Medium", "workflow": "Medium", "budget": "Medium"},
        "internal build / manual process": {"buyer": "High", "workflow": "High", "budget": "Medium"},
        "services firms": {"buyer": "Medium", "workflow": "High", "budget": "Medium"},
        "open-source / low-cost alternatives": {"buyer": "Low", "workflow": "Medium", "budget": "Low"},
        "adjacent workflow tools": {"buyer": "Medium", "workflow": "Medium", "budget": "Medium"},
        "marketplace alternatives": {"buyer": "Medium", "workflow": "Medium", "budget": "Medium"},
        "system-of-record vendors": {"buyer": "High", "workflow": "Medium", "budget": "High"},
    }
    return scores.get(group, {}).get(dimension, "Unknown")


def _group_threat_level(group: str, has_sources: bool, has_names: bool) -> str:
    if group in {"direct startups", "incumbent vendors", "system-of-record vendors", "internal build / manual process"}:
        return "High" if has_sources or group == "internal build / manual process" else "Medium"
    if has_names:
        return "Medium"
    return "Low / requires validation"


def _group_competition_reason(group: str, buyer: str, budget_owner: str, workflow: str, category: str) -> str:
    reasons = {
        "direct startups": f"Compete for the same buyer ({buyer}), budget owner ({budget_owner}), and workflow ({workflow}).",
        "incumbent vendors": f"Can bundle adjacent functionality into existing {budget_owner} or enterprise vendor relationships.",
        "horizontal platforms": "Broad AI/productivity platforms can absorb lightweight workflows if the product is only a feature.",
        "internal build / manual process": f"Enterprise buyers may keep the existing manual workflow or build internally if ROI is unclear.",
        "services firms": f"Services providers can solve the same workflow with people-heavy delivery instead of software.",
        "open-source / low-cost alternatives": "Low-cost tools can cap pricing if the product is not deeply embedded.",
        "adjacent workflow tools": f"Adjacent tools can expand into {workflow} and compete for the same workflow budget.",
        "marketplace alternatives": "Marketplaces can aggregate supply/demand and compete for the same transaction or discovery budget.",
        "system-of-record vendors": f"System-of-record vendors near {category} can bundle, block integration, or own procurement.",
    }
    return reasons.get(group, "Requires validation.")


def _discover_economic_competitors(profile: dict[str, Any], evidence: list[EvidenceItem]) -> list[dict[str, Any]]:
    understanding = profile.get("company_understanding", {})
    target_company = understanding.get("company_name") or profile.get("raw_company_name") or profile.get("company", "")
    framework = _filter_competitor_framework(
        _competitor_framework(
            understanding.get("primary_product", profile.get("product", "")),
            understanding.get("target_buyer", profile.get("customers", [])),
        ),
        target_company,
    )
    groups = framework.get("competitor_groups", {})
    if not groups:
        return []

    buyer = understanding.get("target_buyer", framework["primary_buyer"])
    budget_owner = understanding.get("budget_owner", "Unknown")
    workflow = understanding.get("core_workflow", framework["primary_workflow"])
    category = understanding.get("product_category", framework["product_category"])
    competitor_sources = _source_ids_for(evidence, "competitor")

    rows: list[dict[str, Any]] = []
    for group_name, names in groups.items():
        for name in names:
            if _is_target_company_alias(name, target_company):
                continue
            overlaps = _competitor_overlap_scores(group_name)
            score = round(
                0.35 * overlaps["workflow_overlap"]
                + 0.30 * overlaps["buyer_overlap"]
                + 0.20 * overlaps["budget_overlap"]
                + 0.15 * overlaps["category_overlap"],
                1,
            )
            if score < 6:
                continue
            confidence = "High" if group_name == "Direct Competitors" and competitor_sources else "Medium"
            if confidence not in {"Medium", "High"}:
                continue
            rows.append(
                {
                    "company": name,
                    "buyer": buyer,
                    "workflow": workflow,
                    "budget_owner": budget_owner,
                    "category": group_name,
                    "threat_level": _threat_level(score),
                    "why_it_competes": _competitor_reason(group_name, buyer, budget_owner, workflow, category),
                    "buyer_overlap": overlaps["buyer_overlap"],
                    "workflow_overlap": overlaps["workflow_overlap"],
                    "budget_overlap": overlaps["budget_overlap"],
                    "category_overlap": overlaps["category_overlap"],
                    "competitor_score": score,
                    "confidence": confidence,
                    "source_ids": ", ".join(competitor_sources) if competitor_sources else "Profile-derived; verify with competitor research",
                }
            )
    return sorted(rows, key=lambda row: row["competitor_score"], reverse=True)


def _competitor_overlap_scores(group_name: str) -> dict[str, int]:
    if group_name == "Direct Competitors":
        return {"workflow_overlap": 9, "buyer_overlap": 9, "budget_overlap": 9, "category_overlap": 9}
    if group_name == "Adjacent Competitors":
        return {"workflow_overlap": 7, "buyer_overlap": 8, "budget_overlap": 7, "category_overlap": 6}
    if group_name == "Incumbents":
        return {"workflow_overlap": 6, "buyer_overlap": 8, "budget_overlap": 8, "category_overlap": 5}
    return {"workflow_overlap": 4, "buyer_overlap": 4, "budget_overlap": 4, "category_overlap": 4}


def _threat_level(score: float) -> str:
    if score >= 8:
        return "High - Direct competitor"
    if score >= 6:
        return "Medium - Strong adjacent / incumbent competitor"
    if score >= 4:
        return "Low - Indirect competitor"
    return "Not meaningful"


def _competitor_reason(group_name: str, buyer: str, budget_owner: str, workflow: str, category: str) -> str:
    if group_name == "Direct Competitors":
        return f"Competes for the same {budget_owner} budget, same buyer ({buyer}), and same workflow: {workflow}."
    if group_name == "Adjacent Competitors":
        return f"Adjacent product that can compete for related {budget_owner} budget and workflow expansion around {workflow}."
    if group_name == "Incumbents":
        return f"Incumbent alternative already near the {buyer} purchasing decision and {budget_owner} budget."
    return f"Potential overlap with {category}; requires verification."


def write_structured_memo(state: DealMemoState) -> dict[str, Any]:
    profile = state["company_profile"]
    understanding = profile.get("company_understanding", {})
    evidence = state["evidence"]
    company = understanding.get("company_name", profile.get("company", "Company"))
    sector = understanding.get("industry", profile.get("sector", "the target sector"))
    product = understanding.get("primary_product", profile.get("product", "the product"))
    target_buyer = understanding.get("target_buyer", ", ".join(profile.get("customers", [])))
    core_workflow = understanding.get("core_workflow", product)
    business_model_description = understanding.get("business_model", _infer_business_model(product, sector))
    product_source_ids = profile.get("product_source_ids", [])
    product_basis = profile.get("product_source_basis", "partner notes only")
    product_scope = understanding.get("primary_product_scope", "unclear")
    product_use_cases = understanding.get("product_use_cases", [])
    product_conflicts = understanding.get("product_evidence_conflicts", [])
    if product_scope == "broad_platform":
        use_case_text = ", ".join(product_use_cases[:4]) if product_use_cases else "no specific public use cases yet"
        product_scope_summary = (
            f"{company} appears to provide a broad platform for {target_buyer}. "
            f"Public evidence suggests use cases including {use_case_text}. "
            "The exact initial wedge and usage mix require diligence."
        )
    elif product_scope == "unclear":
        product_scope_summary = (
            f"{company}'s primary product remains unclear from current notes and public evidence; product, buyer, and workflow require diligence."
        )
    else:
        product_scope_summary = f"{company}'s primary product appears to be {product.lower()} for {target_buyer}."
    if product_conflicts:
        product_scope_summary += " Product evidence conflict: " + " ".join(product_conflicts)
    customers = target_buyer
    website_sources = _source_ids_for(evidence, "website")
    market_sources = _source_ids_for(evidence, "market")
    competitor_sources = _source_ids_for(evidence, "competitor")
    news_sources = _source_ids_for(evidence, "news")
    funding_sources = _source_ids_for(evidence, "funding")
    business_sources = _source_ids_for(evidence, "business_model")
    leadership_sources = _source_ids_for(evidence, "leadership")
    open_questions = _default_open_questions(profile)
    website_fact = _evidence_line(evidence, "website", f"No public website evidence found for {company}.")
    market_fact = _evidence_line(evidence, "market", "No market evidence found.")
    news_fact = _evidence_line(evidence, "news", "No recent news evidence found.")
    competitor_fact = _evidence_line(evidence, "competitor", "No competitor evidence found.")
    funding_fact = _evidence_line(evidence, "funding", "No funding evidence found.")
    investment_confidence = _investment_thesis_confidence(evidence)
    business_confidence = _business_model_confidence(evidence)
    competition_confidence = _competition_confidence(evidence)
    domain_terms = _domain_terms(profile)
    evidence_dashboard = _evidence_dashboard(evidence)
    identity_resolution = profile.get("identity_resolution", {})
    identity_resolved = bool(identity_resolution.get("is_resolved", True))
    understanding_validation = understanding.get("validation", {})
    contradiction = understanding_validation.get("contradictions", {})
    understanding_confidence = float(understanding_validation.get("confidence", 0))
    product_is_known = not _is_placeholder_entity(product)
    workflow_is_known = not _is_placeholder_entity(core_workflow)
    market_is_known = sector not in {"", "Unknown", "Conflicting signals detected"} and product_is_known and workflow_is_known
    buyer_is_known = not _is_placeholder_entity(target_buyer)
    weak_source_quality = not any(item.get("source_tier", 3) <= 2 and _source_allows_use(item, "investment scoring") for item in evidence)
    traction_unknown = _reliable_count(evidence, "business_model") == 0 and _reliable_count(evidence, "news") == 0
    if contradiction.get("severity") == "high":
        recommendation_decision = "Clarify Positioning Before Workflow-Level Analysis"
        recommendation_reason = "Conflicting signals detected across partner notes. Interpretations include: " + "; ".join(
            f"{item['interpretation']} ({item['evidence']})" for item in contradiction.get("interpretations", [])
        ) + ". Clarify positioning before workflow-level analysis."
        recommendation_confidence = "Low"
    elif not identity_resolved and understanding_confidence >= 0.8:
        recommendation_decision = "Proceed to Workflow-Level Diligence - Verify Company Identity"
        recommendation_reason = f"Partner notes provide a usable understanding of the product, buyer, and workflow ({product.lower()} for {customers}), but company identity is unresolved. Continue workflow-level market, risk, and competitor diligence, and verify the exact company entity before any investment decision."
        recommendation_confidence = "Low"
    elif not identity_resolved:
        recommendation_decision = "Do Not Advance Yet - Resolve Company Identity"
        recommendation_reason = "Unable to confidently identify company or understand the workflow. Need clarification before collecting sources or generating an investment thesis."
        recommendation_confidence = "Low"
    elif understanding_confidence < 0.8:
        recommendation_decision = "Do Not Advance Yet - Clarify Company Identity"
        recommendation_reason = "Insufficient company understanding. Need clarification before generating an investment thesis."
        recommendation_confidence = "Low"
    elif weak_source_quality and traction_unknown:
        recommendation_decision = "Proceed to Preliminary Diligence Only"
        recommendation_reason = f"{company}'s product, buyer, and workflow are understood from partner notes, but reliable evidence was not found for traction or source-backed investment conclusions."
        recommendation_confidence = "Low"
    elif understanding_confidence >= 0.8 and _coverage_quality(evidence, "website") in {"Strong", "Medium"} and competition_confidence in {"Medium", "High"}:
        recommendation_decision = "Advance to Partner Diligence"
        recommendation_reason = f"{company} merits partner diligence based on its company-specific product wedge: {product.lower()} for {customers}. Market research is used only as context, not as the product definition."
        recommendation_confidence = investment_confidence if investment_confidence != "High" else "Medium"
    else:
        recommendation_decision = "Proceed to Preliminary Diligence Only"
        recommendation_reason = f"{company}'s company understanding is usable, but source quality and traction evidence are not yet strong enough for a high-conviction advance recommendation."
        recommendation_confidence = "Medium" if investment_confidence != "Low" else "Low"
    executive_view = "Insufficient Information" if understanding_confidence < 0.2 else "Positive, pending diligence"
    executive_conviction = "Low" if understanding_confidence < 0.2 else investment_confidence if investment_confidence != "High" else "Medium"

    memo = {
        "company": company,
        "confidence": profile.get("confidence", "medium"),
        "identity_resolution": identity_resolution,
        "recommendation": {
            "decision": recommendation_decision,
            "confidence": recommendation_confidence,
            "reason": recommendation_reason,
            "source_ids": website_sources + market_sources + news_sources,
            "note_reference": True,
            "analyst_inference": True,
        },
        "executive_summary": {
            "view": executive_view,
            "conviction": executive_conviction,
            "why_it_matters": [
                _claim(f"{product_scope_summary} Product basis: {product_basis}.", product_source_ids or website_sources, note=True, inference=True),
                _claim(f"Public market context for that product wedge: {market_fact}", market_sources, inference=True),
                _claim(f"Recent public signal found: {news_fact}", news_sources, inference=True),
            ],
            "key_open_risks": [
                _claim("Partner notes indicate meaningful customer adoption, but public evidence does not yet verify deployment depth, paid usage, retention, or expansion. Diligence should separate pilots, paid deployments, active usage, expansion, and renewal behavior.", note=True, inference=True),
                _claim(f"Competitive pressure remains a diligence item; public competitor evidence includes: {competitor_fact}", competitor_sources, inference=True),
                _claim("Implementation burden and integration dependency need direct customer validation.", website_sources, note=True, inference=True),
            ],
        },
        "bull_case": [
            _claim(f"{company} is attacking a high-friction workflow where buyers may have budgeted urgency if ROI is measurable.", market_sources, note=True, inference=True),
            _claim(f"Company-specific product evidence supports the thesis for {company}: {website_fact}", product_source_ids or website_sources, note=True),
            _claim(f"The company may have room to expand into adjacent {sector.lower()} workflows if the initial wedge proves sticky.", news_sources, inference=True),
        ],
        "bear_case": [
            _claim(f"{company}'s traction is not investment-grade until ARR, NRR, live customer count, and implementation economics are verified.", note=True, inference=True),
            _claim(f"Competitive differentiation is uncertain given public competitor context: {competitor_fact}", competitor_sources, inference=True),
            _claim("AI automation features may commoditize unless the company has proprietary workflow data, distribution, or deep system integration.", competitor_sources + website_sources, inference=True),
        ],
        "what_needs_to_be_true": [
            _claim(f"{company} must show repeatable live deployments beyond pilots or design partners.", note=True, inference=True),
            _claim("Customer ROI must be measurable in time saved, cost reduction, revenue capture, or reduced administrative backlog.", market_sources, inference=True),
            _claim(f"Implementation must be lightweight enough for {target_buyer} to adopt without heavy services drag.", website_sources, inference=True),
            _claim("The product must remain differentiated against incumbent and startup competitors.", competitor_sources, inference=True),
            _claim("Gross margin must remain attractive after services, support, and model-inference costs.", note=True, inference=True),
            _claim(f"Security, privacy, and compliance posture must satisfy {domain_terms['procurement']}.", news_sources, inference=True),
        ],
        "partner_note_verification": _partner_claims(profile, evidence),
        "company_understanding": understanding,
        "evidence_dashboard": evidence_dashboard,
        "next_diligence_priorities": _next_diligence_priorities(company, profile),
        "sources_grouped_by_category": _sources_grouped_by_category(evidence),
        "opportunity_scorecard": [
            _score_or_unknown(
                "Market",
                f"Market score is the average of TAM/pain, growth, and budget urgency factors for {product.lower()}." if market_is_known else "Market score is Unknown because the notes do not establish a specific market, category, or workflow.",
                market_sources if market_is_known else [],
                [
                    _factor("TAM / pain severity", 8 if market_is_known and market_sources else None, f"Public market context supports a painful workflow: {market_fact}" if market_is_known else "Cannot assess TAM/pain until the market is identified.", market_sources if market_is_known else []),
                    _factor("Growth / timing", 7 if market_is_known and (market_sources or news_sources) else None, f"Timing is supported by market/news context: {market_fact if market_sources else news_fact}" if market_is_known else "Cannot assess timing until the market is identified.", (market_sources + news_sources) if market_is_known else []),
                    _factor("Budget urgency", 7 if market_is_known and buyer_is_known else None, f"Partner notes indicate buyer relevance for {customers}." if buyer_is_known else "Budget owner and buyer urgency are not specified.", []),
                ],
                inference=True,
            ),
            _score_or_unknown(
                "Product",
                f"Product score is the average of differentiation, customer value, and workflow depth for {company}'s company-specific product wedge." if product_is_known else "Product score is Unknown because the notes do not identify a specific product or workflow.",
                website_sources if product_is_known else [],
                [
                    _factor("Differentiation", 6 if product_is_known and website_sources else None, f"Company-specific product evidence: {website_fact}" if product_is_known else "Cannot assess differentiation until product is clarified.", website_sources if product_is_known else []),
                    _factor("Customer value", 7 if product_is_known and website_sources else None, "Potential value depends on reducing work, cost, or friction in the target workflow." if product_is_known else "Cannot assess customer value until workflow is clarified.", website_sources if product_is_known else []),
                    _factor("Workflow depth", 6 if product_is_known and website_sources else None, "Workflow depth must be verified through product demo and customer references." if product_is_known else "Cannot assess workflow depth until workflow is clarified.", website_sources if product_is_known else []),
                ],
                inference=True,
            ),
            _score_or_unknown(
                "Team",
                f"Team score is the average of founder-market fit, execution track record, and {sector.lower()} depth.",
                leadership_sources,
                [
                    _factor("Founder-market fit", 6 if leadership_sources else None, "Founder-market fit requires independent leadership evidence.", leadership_sources),
                    _factor("Execution track record", None, "Execution history is not verified from current public evidence.", []),
                    _factor(f"{sector} depth", 6 if leadership_sources else None, "Domain depth requires founder/team biography verification.", leadership_sources),
                ],
                inference=True,
            ),
            _score_or_unknown(
                "Business Model",
                "Business model score is the average of pricing clarity, margin quality, and retention visibility.",
                business_sources,
                [
                    _factor("Pricing clarity", 5 if business_sources else None, "Pricing and ACV are not verified unless business-model evidence exists.", business_sources),
                    _factor("Margin quality", None, "Gross margin after services and model costs is not yet known.", []),
                    _factor("Retention visibility", None, "NRR, churn, and cohort retention are not yet known.", []),
                ],
                inference=True,
            ),
            _score_or_unknown(
                "Competition",
                "Competition score is the average of competitive intensity, incumbent pressure, and differentiation evidence.",
                competitor_sources if market_is_known else [],
                [
                    _factor("Competitive intensity", 4 if market_is_known and competitor_sources else None, f"Competitor context suggests deal pressure: {competitor_fact}" if market_is_known else "Cannot assess competitors until the product category is identified.", competitor_sources if market_is_known else []),
                    _factor("Incumbent pressure", 5 if market_is_known and competitor_sources else None, "Potential incumbent pressure must be tested in win/loss diligence." if market_is_known else "Cannot assess incumbent pressure until the market is identified.", competitor_sources if market_is_known else []),
                    _factor("Differentiation evidence", 5 if market_is_known and website_sources and competitor_sources else None, "Differentiation is not fully proven from current public evidence." if market_is_known else "Cannot assess differentiation versus competitors until product category is clarified.", (website_sources + competitor_sources) if market_is_known else []),
                ],
                inference=True,
            ),
            _score_or_unknown(
                "Defensibility",
                "Defensibility score is the average of data advantage, workflow lock-in, and distribution advantage.",
                website_sources + competitor_sources,
                [
                    _factor("Data advantage", None, "No direct evidence of proprietary data advantage.", []),
                    _factor("Workflow lock-in", 5 if website_sources else None, "Workflow embedding is plausible but not verified.", website_sources),
                    _factor("Distribution advantage", None, "No direct evidence of durable distribution advantage.", []),
                ],
                inference=True,
            ),
            _score_or_unknown(
                "Exit Potential",
                "Exit potential score is the average of strategic buyer fit, category momentum, and scale evidence.",
                funding_sources + news_sources,
                [
                    _factor("Strategic buyer fit", 6 if news_sources else None, "Strategic fit is directional from healthcare AI/news context.", news_sources),
                    _factor("Category momentum", 6 if funding_sources or news_sources else None, f"Momentum signal found: {funding_fact if funding_sources else news_fact}", funding_sources + news_sources),
                    _factor("Scale evidence", None, "Scale evidence is not verified without ARR, customer count, or funding details.", funding_sources),
                ],
                inference=True,
            ),
        ],
        "signals": {
            "positive": [
                _claim(f"{company} targets {customers}, a buyer segment called out in the partner notes.", note=True),
                _claim(f"Public product evidence: {website_fact}", website_sources),
                _claim(f"Public market/news context: {market_fact if market_sources else news_fact}", market_sources + news_sources, inference=True),
            ],
            "negative": [
                _claim(f"Competitive market evidence: {competitor_fact}", competitor_sources, inference=True),
                _claim(f"{company}'s ROI and deployment scale are not yet independently verified.", note=True, inference=True),
                _claim(f"{domain_terms['procurement'].capitalize()} may slow growth.", news_sources, inference=True),
            ],
        },
        "source_attribution": {
            "partner_notes": [
                _claim("Partner notes describe the product, target customer hints, founder/team background, and initial thesis.", note=True),
            ],
            "public_research": [
                _claim("Public research is used to corroborate product category, market context, competitive set, and recent signals.", website_sources + market_sources + competitor_sources + news_sources),
            ],
            "combined_insight": [
                _claim("The investment question is whether a painful workflow plus credible automation translates into repeatable enterprise adoption and defensible distribution.", market_sources + website_sources, note=True, inference=True),
            ],
        },
        "section_confidence": {
            "Company Overview": _section_confidence(evidence, "website"),
            "Founding Team": _section_confidence(evidence, "leadership"),
            "Product": _section_confidence(evidence, "website"),
            "Market Opportunity": _section_confidence(evidence, "market"),
            "Competitive Landscape": competition_confidence,
            "Business Model": business_confidence,
            "Recent Funding": _section_confidence(evidence, "funding", has_notes=False),
            "Investment Thesis": investment_confidence,
            "Recommendation": investment_confidence if investment_confidence != "High" else "Medium",
        },
        "sections": [
            {"title": "Company Overview", "claims": [_claim(product_scope_summary, website_sources, note=True, inference=True)]},
            {"title": "Founding Team", "claims": [_claim(profile.get("team", "Team background requires diligence."), leadership_sources, note=True)]},
            {"title": "Product", "claims": [_claim(f"{product_scope_summary} Company-specific evidence: {website_fact}", product_source_ids or website_sources, note=True, inference=True)]},
            {"title": "Why Now", "claims": [
                _claim(f"Partner notes frame urgency around {product.lower()} and {domain_terms['value_metric']}.", note=True),
                _claim(f"Public market/news support only contextualizes timing for this product wedge: {market_fact if market_sources else news_fact}", market_sources + news_sources, inference=True),
            ]},
            {"title": "Market Opportunity", "claims": [_claim(f"Market opportunity is contextualized around {company}'s actual product wedge ({product.lower()}): {market_fact}", market_sources, note=True, inference=True)]},
            {"title": "Competitive Landscape", "claims": [_claim(f"Competitors are prioritized by same workflow, same buyer, and same product category around {company}'s product wedge ({product.lower()}); public context found: {competitor_fact}", competitor_sources, inference=True)]},
            {"title": "Business Model", "claims": [_claim(f"Expected business model: {business_model_description}. Business model remains a diligence gap for {company}; public evidence found: {_evidence_line(evidence, 'business_model', 'No pricing or business model evidence found.')}", business_sources, inference=True)]},
            {"title": "Recent Funding", "claims": [_claim(f"Funding evidence found: {funding_fact}", funding_sources, inference=True)]},
            {"title": "Key Risks", "claims": [_claim(f"Key risks for {company}: unverified traction, buyer ROI, competitive differentiation, integration burden, security review, and business-model quality.", news_sources + competitor_sources, note=True, inference=True)]},
            {"title": "Investment Thesis", "claims": [_claim(f"Advance {company} only if diligence confirms repeatable adoption, measurable ROI, and defensible workflow integration.", market_sources + website_sources, note=True, inference=True)]},
            {"title": "Open Questions", "claims": [_claim(question, note=True) for question in open_questions]},
        ],
    }
    memo["claim_verification"] = _claim_verification_layer(profile, evidence, memo)
    memo["visualizations"] = {
        "investment_scorecard": _investment_visualization(memo["opportunity_scorecard"]),
        "claim_verification_summary": _claim_verification_visualization(memo["partner_note_verification"]),
        "risk_breakdown": _risk_breakdown(evidence, company, profile),
        "bull_bear_weights": _bull_bear_weights(memo),
    }
    memo["evidence_coverage"] = _evidence_coverage(evidence, memo)
    memo["confidence_calibration"] = _calibrated_overall_confidence(profile, memo, evidence)
    memo["confidence"] = memo["confidence_calibration"]["label"]
    memo["missing_data"] = _missing_data_detection(memo)
    memo["thesis_framework"] = _thesis_framework(profile, evidence, memo, domain_terms, market_is_known, buyer_is_known)
    memo["defensibility_framework"] = _defensibility_framework(profile, evidence, memo)
    memo["competitive_landscape"] = _competitive_landscape(profile, evidence)
    memo["traction_analysis"] = _traction_analysis(profile, evidence)
    memo["business_model_analysis"] = _business_model_analysis(profile, evidence, memo)
    memo["risk_taxonomy"] = _risk_taxonomy(profile, evidence, memo)
    blocking_names = [gap["gap"] for gap in memo["missing_data"]["decision_blocking_unknowns"][:4]]
    if blocking_names:
        memo["recommendation"]["reason"] += f" Conviction remains limited because the following decision-blocking data is missing: {', '.join(blocking_names)}."
        memo["recommendation"]["decision_readiness"] = memo["missing_data"]["decision_readiness"]
        memo["recommendation"]["recommendation_adjustment"] = memo["missing_data"]["recommendation_adjustment"]
    raw_notes = profile.get("raw_notes", "")
    risk_questions = [
        {"question": risk["diligence_question"], "source_note_reference": _source_note_reference_for_question(risk["diligence_question"], raw_notes)}
        for risk in memo["visualizations"]["risk_breakdown"]["risks"]
        if risk.get("risk_type") == "note_explicit"
    ]
    risk_questions = [item for item in risk_questions if item["source_note_reference"]]
    coverage_focus = [
        {
            "question": f"{item['category']} ({item['coverage']}%): {item['reason']}",
            "source_note_reference": _source_note_reference_for_question(item["category"], raw_notes),
        }
        for item in memo["evidence_coverage"]["recommended_diligence_focus"]
    ]
    coverage_focus = [item for item in coverage_focus if item["source_note_reference"]]
    if risk_questions:
        memo["next_diligence_priorities"]["Risk-specific questions"] = risk_questions
    if coverage_focus:
        memo["next_diligence_priorities"]["Coverage-driven focus"] = coverage_focus
    return {"memo_json": memo, "trace": _trace(state, "write_structured_memo", {"sections": len(memo["sections"])})}


def generate_chart_data(state: DealMemoState) -> dict[str, Any]:
    visualizations = state["memo_json"].get("visualizations", {})
    chart_data = {
        **visualizations,
        "evidence_coverage": state["memo_json"].get("evidence_coverage", {}),
        "competitive_table": _competitive_table(state["company_profile"], state["evidence"]),
        "competitive_landscape": state["memo_json"].get("competitive_landscape", {}),
        "traction_analysis": state["memo_json"].get("traction_analysis", {}),
        "business_model_analysis": state["memo_json"].get("business_model_analysis", {}),
        "risk_taxonomy": state["memo_json"].get("risk_taxonomy", {}),
    }
    return {"chart_data": chart_data, "trace": _trace(state, "generate_chart_data", {"charts": list(chart_data)})}


def _section_to_evidence_type(section: str) -> str:
    mapping = {
        "Executive Summary": "market",
        "Recommendation": "market",
        "Bull Case": "market",
        "Bear Case": "competitor",
        "What Needs To Be True": "market",
        "Partner Note Verification": "website",
        "Next Diligence Priorities": "notes",
        "Opportunity Scorecard": "market",
        "Company Overview": "website",
        "Founding Team": "leadership",
        "Product": "website",
        "Why Now": "news",
        "Market Opportunity": "market",
        "Competitive Landscape": "competitor",
        "Business Model": "business_model",
        "Recent Funding": "funding",
        "Key Risks": "news",
        "Investment Thesis": "market",
    }
    return mapping.get(section, "notes")


def _post_generation_memo_validator(memo: dict[str, Any], evidence: list[EvidenceItem], raw_notes: str) -> dict[str, Any]:
    findings: list[dict[str, str]] = []

    def add(severity: str, warning: str, suggested_fix: str) -> None:
        findings.append({"severity": severity, "warning": warning, "suggested_fix": suggested_fix})

    if memo.get("_unsupported_claims_for_validator") or (
        memo.get("traceability", {}).get("unsupported_count", 0) > 0 and not memo.get("_removed_unsupported_for_validator")
    ):
        add("high", "Unsupported claims remain after grounding.", "Remove unsupported claims or move them to diligence questions.")
    strong_words = ["market leader", "best-in-class", "dominant", "proven traction", "highly defensible"]
    memo_text = str(memo).lower()
    if any(word in memo_text for word in strong_words):
        add("medium", "Overconfident language appears in memo.", "Replace with evidence-calibrated language and cite support.")
    weak_sources = {item["source_id"] for item in evidence if item.get("source_tier", 3) == 3 or item.get("source_quality") == "low"}
    strong_claim_sources = [
        source_id
        for row in memo.get("claim_verification", [])
        if row.get("verification_status") == "Verified by public evidence"
        for source_id in row.get("source_ids", [])
    ]
    if weak_sources & set(strong_claim_sources):
        add("high", "Weak sources are used for strong verified claims.", "Downgrade claim status or replace with independent reliable evidence.")
    target = memo.get("company", "")
    for row in memo.get("competitive_landscape", {}).get("rows", []):
        if any(_is_target_company_alias(name, target) for name in row.get("examples", [])):
            add("high", "Target company appears in competitor list.", "Filter target company aliases from all competitor groups.")
    understanding = memo.get("company_understanding", {})
    if understanding.get("product_evidence_conflicts"):
        add("medium", "Product over-narrowing or product evidence conflict detected.", "Preserve note-derived scope and show public use cases separately.")
    traction = memo.get("traction_analysis", {})
    if "funding is not traction" not in " ".join(traction.get("rules", [])).lower():
        add("high", "Funding may be treated as traction.", "Add explicit funding-not-traction rule and require operating traction metrics.")
    traction_text = str(traction).lower()
    if "logo quality" in traction_text and "retention" not in traction_text:
        add("medium", "Customer logos may be treated as retention.", "Separate logo quality, active usage, retention, and expansion.")
    thesis = memo.get("thesis_framework", {})
    for key, label in [
        ("why_now", "why now"),
        ("why_this_company", "why this company"),
        ("what_could_kill_the_deal", "what could kill the deal"),
    ]:
        if not thesis.get(key, {}).get("answer"):
            add("high", f"Missing {label}.", f"Populate thesis framework field: {key}.")
    if not memo.get("next_diligence_priorities") and not memo.get("missing_data", {}).get("decision_blocking_unknowns"):
        add("medium", "Missing diligence questions.", "Generate diligence questions from coverage gaps and risks.")
    if any(header in memo_text[:1000] for header in ["company identity resolution", "evidence quality", "traceability"]):
        add("medium", "Debug or diagnostic content appears before memo substance.", "Start Partner Memo View with Executive Summary and keep diagnostics later or in Trace.")
    low_count = sum(1 for item in evidence if item.get("source_tier", 3) == 3 or item.get("source_quality") == "low")
    if evidence and low_count / len(evidence) > 0.5:
        add("medium", "Too many low-confidence sources.", "Use low-quality sources only for hypotheses and prioritize reliable independent sources.")
    if "partner notes" not in memo_text or "public evidence" not in memo_text:
        add("medium", "Memo does not clearly distinguish notes from public evidence.", "Label note-derived, public-source, and inferred claims.")

    severity_rank = {"low": 0, "medium": 1, "high": 2}
    max_severity = max((severity_rank[item["severity"]] for item in findings), default=0)
    reverse = {0: "low", 1: "medium", 2: "high"}
    return {
        "passed": not any(item["severity"] == "high" for item in findings),
        "severity": reverse[max_severity],
        "warnings": findings,
        "suggested_fixes": [item["suggested_fix"] for item in findings],
    }


def validate_grounding(state: DealMemoState) -> dict[str, Any]:
    memo = dict(state["memo_json"])
    evidence_by_id = {item["source_id"]: item for item in state["evidence"]}
    raw_notes = state["raw_notes"]
    unsupported: list[str] = []
    removed_unsupported: list[str] = []
    coverage: dict[str, int] = {}
    warnings: list[str] = []
    traceability: list[dict[str, Any]] = []

    def support_record(
        statement: str,
        source_ids: list[str] | None = None,
        note_reference: bool = False,
        inference: bool = False,
        bucket: str = "",
        count_unsupported: bool = True,
    ) -> dict[str, Any]:
        source_ids = source_ids or []
        note_evidence = _source_note_reference_for_question(statement, raw_notes) if note_reference else ""
        public_evidence = [
            {
                "source_id": source_id,
                "title": evidence_by_id[source_id]["title"],
                "snippet": evidence_by_id[source_id]["snippet"],
                "url": evidence_by_id[source_id]["url"],
            }
            for source_id in source_ids
            if source_id in evidence_by_id
        ]
        valid_source_ids = [item["source_id"] for item in public_evidence]
        source_text = " ".join([item["title"] + " " + item["snippet"] for item in public_evidence])
        note_supported = bool(note_evidence)
        public_supported = bool(valid_source_ids) and (_token_overlap_score(statement, source_text) > 0 or inference)
        absence_supported = bool(
            inference
            and any(
                marker in statement.lower()
                for marker in [
                    "no public",
                    "no pricing",
                    "no funding",
                    "no recent",
                    "no competitor",
                    "not verified",
                    "diligence gap",
                    "requires diligence",
                    "missing",
                ]
            )
        )
        inference_supported = bool(inference and (valid_source_ids or note_supported))
        supported = note_supported or public_supported or inference_supported or absence_supported
        if not supported and count_unsupported:
            unsupported.append(statement or f"Untitled claim in {bucket}")
        record = {
            "statement": statement,
            "bucket": bucket,
            "evidence": note_evidence or "; ".join(f"{item['source_id']}: {item['title']}" for item in public_evidence),
            "source_ids": valid_source_ids,
            "status": "Supported" if supported else "Unsupported",
            "reason": "Has current-note evidence, public evidence, or grounded inference." if supported else "No current-note or valid public evidence supports this statement.",
        }
        traceability.append(record)
        return record

    def claim_supported(claim: dict[str, Any], bucket: str) -> bool:
        record = support_record(
            claim.get("text", f"Untitled claim in {bucket}"),
            claim.get("source_ids", []),
            claim.get("note_reference", False),
            claim.get("analyst_inference", False),
            bucket,
            True,
        )
        return record["status"] == "Supported"

    def validate_claim(claim: dict[str, Any], bucket: str) -> int:
        claim_supported(claim, bucket)
        return len([source_id for source_id in claim.get("source_ids", []) if source_id in evidence_by_id])

    def prune_claims(claims: list[Any], bucket: str) -> list[Any]:
        kept = []
        for claim in claims:
            if not isinstance(claim, dict):
                kept.append(claim)
                continue
            record = support_record(
                claim.get("text", f"Untitled claim in {bucket}"),
                claim.get("source_ids", []),
                claim.get("note_reference", False),
                claim.get("analyst_inference", False),
                bucket,
                False,
            )
            if record["status"] == "Supported":
                kept.append(claim)
            else:
                removed_unsupported.append(record["statement"])
        return kept

    def prune_scorecard_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pruned_rows = []
        for row in rows:
            if row.get("score") is None:
                pruned_rows.append(row)
                continue
            record = support_record(
                row.get("rationale", row.get("dimension", "")),
                row.get("source_ids", []),
                False,
                row.get("analyst_inference", True),
                "Opportunity Scorecard",
                False,
            )
            if record["status"] == "Supported":
                pruned_rows.append(row)
            else:
                removed_unsupported.append(record["statement"])
                pruned_rows.append(
                    {
                        **row,
                        "score": None,
                        "rationale": f"Unknown: removed numeric score because traceability check found no supporting note or public evidence for {row.get('dimension', 'this category')}.",
                        "source_ids": [],
                        "confidence": "Low",
                        "factors": [
                            {**factor, "score": None, "reason": f"Unsupported: {factor.get('reason', '')}", "source_ids": []}
                            for factor in row.get("factors", [])
                        ],
                    }
                )
        return pruned_rows

    memo["bull_case"] = prune_claims(memo.get("bull_case", []), "Bull Case")
    memo["bear_case"] = prune_claims(memo.get("bear_case", []), "Bear Case")
    memo["what_needs_to_be_true"] = prune_claims(memo.get("what_needs_to_be_true", []), "What Needs To Be True")
    for signal_bucket, claims in list(memo.get("signals", {}).items()):
        memo["signals"][signal_bucket] = prune_claims(claims, f"Signals: {signal_bucket}")
    if memo.get("executive_summary"):
        memo["executive_summary"]["why_it_matters"] = prune_claims(
            memo["executive_summary"].get("why_it_matters", []),
            "Executive Summary",
        )
        memo["executive_summary"]["key_open_risks"] = prune_claims(
            memo["executive_summary"].get("key_open_risks", []),
            "Executive Summary",
        )
    for section in memo.get("sections", []):
        section["claims"] = prune_claims(section.get("claims", []), section.get("title", "Untitled"))
    memo["opportunity_scorecard"] = prune_scorecard_rows(memo.get("opportunity_scorecard", []))
    if "visualizations" in memo:
        memo["visualizations"]["investment_scorecard"] = _investment_visualization(memo["opportunity_scorecard"])
    understanding_fields = {
        "Target buyer": memo.get("company_understanding", {}).get("target_buyer", ""),
        "Primary product": memo.get("company_understanding", {}).get("primary_product", ""),
        "Core workflow": memo.get("company_understanding", {}).get("core_workflow", ""),
        "Business model": memo.get("company_understanding", {}).get("business_model", ""),
    }
    for label, value in understanding_fields.items():
        if value and not _is_placeholder_entity(str(value)):
            support_record(f"{label} = {value}", [], True, False, "Company Understanding", False)

    for section in memo.get("sections", []):
        section_count = 0
        for claim in section.get("claims", []):
            section_count += validate_claim(claim, section.get("title", "Untitled"))
        coverage[section.get("title", "Untitled")] = section_count

    executive = memo.get("executive_summary", {})
    for claim in executive.get("why_it_matters", []):
        coverage["Executive Summary"] = coverage.get("Executive Summary", 0) + validate_claim(claim, "Executive Summary")
    for claim in executive.get("key_open_risks", []):
        if isinstance(claim, dict):
            coverage["Executive Summary"] = coverage.get("Executive Summary", 0) + validate_claim(claim, "Executive Summary")

    recommendation = memo.get("recommendation", {})
    if recommendation:
        coverage["Recommendation"] = coverage.get("Recommendation", 0) + validate_claim(
            {
                "text": recommendation.get("reason", ""),
                "source_ids": recommendation.get("source_ids", []),
                "note_reference": recommendation.get("note_reference", False),
                "analyst_inference": recommendation.get("analyst_inference", False),
            },
            "Recommendation",
        )

    for bucket, claims in memo.get("signals", {}).items():
        for claim in claims:
            coverage[f"Signals: {bucket}"] = coverage.get(f"Signals: {bucket}", 0) + validate_claim(claim, f"Signals: {bucket}")

    for bucket in ("bull_case", "bear_case", "what_needs_to_be_true"):
        for claim in memo.get(bucket, []):
            coverage[bucket] = coverage.get(bucket, 0) + validate_claim(claim, bucket)

    for row in memo.get("opportunity_scorecard", []):
        if row.get("score") is None:
            continue
        coverage["Opportunity Scorecard"] = coverage.get("Opportunity Scorecard", 0) + validate_claim(
            {
                "text": row.get("rationale", ""),
                "source_ids": row.get("source_ids", []),
                "note_reference": False,
                "analyst_inference": row.get("analyst_inference", True),
            },
            "Opportunity Scorecard",
        )

    for row in memo.get("partner_note_verification", []):
        if row.get("verification_status") == "Verified" and not row.get("source_ids"):
            unsupported.append(row.get("partner_claim", "Partner claim marked verified without source"))
        coverage["Partner Note Verification"] = coverage.get("Partner Note Verification", 0) + len(row.get("source_ids", []))

    for bucket, claims in memo.get("source_attribution", {}).items():
        for claim in claims:
            if isinstance(claim, dict):
                coverage[f"Source Attribution: {bucket}"] = coverage.get(f"Source Attribution: {bucket}", 0) + validate_claim(
                    claim, f"Source Attribution: {bucket}"
                )

    if state["company_profile"].get("confidence") == "low":
        warnings.append("Sparse notes or ambiguous company name. Treat memo as low confidence.")
    identity = state["company_profile"].get("identity_resolution", {})
    if identity and not identity.get("is_resolved", False):
        warnings.append("Unable to confidently identify company.")
    if identity and float(identity.get("confidence", 0)) < 0.8 and "Advance" in memo.get("recommendation", {}).get("decision", ""):
        warnings.append("Consistency issue: recommendation advances despite unresolved company identity.")
    understanding = state["company_profile"].get("company_understanding", {})
    understanding_validation = understanding.get("validation", {})
    if float(understanding_validation.get("confidence", 0)) < 0.8:
        warnings.append("Company understanding confidence is below 0.8; do not rely on a full investment thesis until clarified.")
    warnings.extend(understanding_validation.get("failures", []))
    warnings.extend(understanding_validation.get("research_plan_failures", []))
    recommendation = memo.get("recommendation", {})
    if float(understanding_validation.get("confidence", 0)) < 0.8 and "Advance" in recommendation.get("decision", ""):
        warnings.append("Consistency issue: recommendation advances despite low company understanding.")
    coverage_rows = {row["category"]: row for row in memo.get("evidence_coverage", {}).get("rows", [])}
    business_model_row = coverage_rows.get("Business Model", {})
    if business_model_row.get("coverage", 0) >= 80 and _business_model_confidence(state["evidence"]) == "Low":
        warnings.append("Consistency issue: Business Model coverage is high but source confidence is weak.")
    raw_lower = state["raw_notes"].lower()
    if "legal team" in raw_lower and "legal" not in str(understanding.get("target_buyer", "")).lower():
        warnings.append("Consistency issue: notes mention legal teams but target buyer is not legal.")
    if "contract review" in raw_lower and "contract review" not in str(understanding.get("core_workflow", "")).lower():
        warnings.append("Consistency issue: notes mention contract review but workflow does not match.")
    if not state["evidence"]:
        warnings.append("No public evidence available. Charts should be disabled or marked insufficient.")
    warnings.extend(memo.get("evidence_dashboard", {}).get("warnings", []))
    memo["traceability"] = {
        "records": traceability,
        "supported_count": sum(1 for record in traceability if record["status"] == "Supported"),
        "unsupported_count": sum(1 for record in traceability if record["status"] == "Unsupported"),
    }
    memo["_unsupported_claims_for_validator"] = unsupported
    memo["_removed_unsupported_for_validator"] = removed_unsupported
    memo["memo_validator"] = _post_generation_memo_validator(memo, state["evidence"], state["raw_notes"])
    memo.pop("_unsupported_claims_for_validator", None)
    memo.pop("_removed_unsupported_for_validator", None)

    validation = {
        "is_valid": not unsupported and memo["memo_validator"].get("passed", False),
        "unsupported_claims": unsupported,
        "removed_unsupported_claims": removed_unsupported,
        "warnings": warnings,
        "source_coverage": coverage,
        "traceability": memo["traceability"],
        "memo_validator": memo["memo_validator"],
    }
    return {"memo_json": memo, "validation": validation, "trace": _trace(state, "validate_grounding", validation)}
