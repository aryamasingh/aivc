from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


EvidenceType = Literal[
    "website",
    "funding",
    "news",
    "leadership",
    "market",
    "competitor",
    "business_model",
    "notes",
]


class ResearchQuery(TypedDict):
    query: str
    evidence_type: EvidenceType
    rationale: str


class EvidenceItem(TypedDict):
    source_id: str
    title: str
    url: str
    snippet: str
    query: str
    evidence_type: EvidenceType
    confidence: float
    retrieved_at: str
    source_quality: NotRequired[str]
    quality_notes: NotRequired[str]
    source_tier: NotRequired[int]
    source_weight: NotRequired[float]
    source_tier_label: NotRequired[str]
    source_type: NotRequired[str]
    independence_level: NotRequired[str]
    allowed_uses: NotRequired[list[str]]
    disallowed_uses: NotRequired[list[str]]


class MemoClaim(TypedDict):
    text: str
    source_ids: list[str]
    note_reference: bool
    analyst_inference: bool


class MemoSection(TypedDict):
    title: str
    claims: list[MemoClaim]


class ValidationResult(TypedDict):
    is_valid: bool
    unsupported_claims: list[str]
    warnings: list[str]
    source_coverage: dict[str, int]


class DealMemoState(TypedDict):
    raw_notes: str
    company_profile: dict[str, Any]
    research_plan: list[ResearchQuery]
    search_results: list[dict[str, Any]]
    evidence: list[EvidenceItem]
    memo_json: dict[str, Any]
    chart_data: dict[str, Any]
    validation: ValidationResult
    memo_markdown: str
    errors: list[str]
    trace: NotRequired[list[dict[str, Any]]]
