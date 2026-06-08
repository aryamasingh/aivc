from __future__ import annotations

from difflib import SequenceMatcher
import os
import re
from urllib.parse import urlparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .sample_data import MOCK_EVIDENCE
from .state import EvidenceItem, ResearchQuery


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def get_secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st

        secret_value = st.secrets.get(name)
        return str(secret_value) if secret_value else None
    except Exception:
        pass

    secrets_path = Path(".streamlit/secrets.toml")
    if not secrets_path.exists():
        return None
    try:
        import tomllib

        data = tomllib.loads(secrets_path.read_text())
        file_value = data.get(name)
        return str(file_value) if file_value else None
    except Exception:
        return None


LOW_QUALITY_DOMAINS = (
    "medium.com",
    "substack.com",
    "blogspot.",
    "wordpress.",
    "wixsite.",
    "instagram.com",
    "linkedin.com",
    "reddit.com",
    "quora.com",
    "x.com",
    "twitter.com",
    "g2.com",
    "capterra.com",
    "softwareadvice.com",
    "getapp.com",
    "saasworthy.com",
    "sourceforge.net",
    "slashdot.org",
)

TIER_1_ALLOWLIST_DOMAINS = (
    "sec.gov",
    ".gov",
    ".mil",
    "edgar.sec.gov",
    "crunchbase.com",
    "pitchbook.com",
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "businesswire.com",
    "prnewswire.com",
    "globenewswire.com",
)

TIER_2_ALLOWLIST_DOMAINS = (
    "cnbc.com",
    "forbes.com",
    "techcrunch.com",
    "theinformation.com",
    "gartner.com",
    "forrester.com",
    "cbinsights.com",
    "dealroom.co",
    "tracxn.com",
    "fiercehealthcare.com",
    "healthcaredive.com",
    "healthcareitnews.com",
    "modernhealthcare.com",
    "beckershospitalreview.com",
    "statnews.com",
    "law.com",
    "legaltechnology.com",
    "sifted.eu",
    "eu-startups.com",
)

LISTICLE_TERMS = ("top ", "best ", "alternatives", "listicle", "review sites", "reviews", "comparison", "top 10", "top ten")
FUNDING_DATABASE_DOMAINS = ("crunchbase.com", "pitchbook.com", "tracxn.com", "cbinsights.com", "dealroom.co")
REPUTABLE_NEWS_DOMAINS = (
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "cnbc.com",
    "techcrunch.com",
    "theinformation.com",
    "statnews.com",
    "fiercehealthcare.com",
    "healthcaredive.com",
    "healthcareitnews.com",
    "modernhealthcare.com",
)
WIRE_DOMAINS = ("businesswire.com", "prnewswire.com", "globenewswire.com")
PROFESSIONAL_PROFILE_DOMAINS = ("linkedin.com", "theorg.com", "apollo.io", "zoominfo.com", "crunchbase.com/people")
VENDOR_COMPARISON_TERMS = ("alternatives", "compare", "comparison", "vs ", " versus ", "competitors")
SEO_TERMS = ("best ", "top ", "top 10", "top ten", "review", "reviews", "listicle", "buyers guide", "buyer guide", "ultimate guide")
CONTENT_FARM_TERMS = ("content farm", "ai-generated", "scraped", "directory", "software directory")

SOURCE_POLICIES = {
    "company_owned": {
        "independence_level": "company-owned",
        "allowed_uses": ["product positioning", "stated customers", "stated partnerships", "launch descriptions"],
        "disallowed_uses": ["independent traction verification", "market leadership", "product superiority", "investment attractiveness"],
    },
    "funding_database": {
        "independence_level": "database / institutional",
        "allowed_uses": ["funding history", "investors", "round timing", "valuation if available"],
        "disallowed_uses": ["customer traction", "revenue quality", "retention", "product superiority"],
    },
    "independent_news": {
        "independence_level": "independent",
        "allowed_uses": ["launches", "partnerships", "financing", "customer announcements", "market context"],
        "disallowed_uses": ["unqualified company claims", "uncorroborated superiority"],
    },
    "customer_case_study": {
        "independence_level": "company-influenced",
        "allowed_uses": ["customer adoption", "reported ROI", "deployment examples"],
        "disallowed_uses": ["independent ROI verification", "market leadership", "broad traction"],
    },
    "vendor_blog": {
        "independence_level": "vendor-influenced",
        "allowed_uses": ["competitor discovery", "market vocabulary", "feature vocabulary"],
        "disallowed_uses": ["market conviction", "market leadership", "product superiority", "investment attractiveness"],
    },
    "generic_blog": {
        "independence_level": "low-independence",
        "allowed_uses": ["weak context", "hypothesis generation"],
        "disallowed_uses": ["investment scoring", "high-confidence conclusions", "market conviction", "traction verification"],
    },
    "professional_profile": {
        "independence_level": "profile / database",
        "allowed_uses": ["preliminary team mapping", "role history hypotheses"],
        "disallowed_uses": ["definitive team verification", "investment conviction without corroboration"],
    },
}


RELEVANCE_STOPWORDS = {
    "official",
    "website",
    "company",
    "funding",
    "investors",
    "news",
    "market",
    "context",
    "customers",
    "customer",
    "product",
    "products",
    "team",
    "teams",
    "enterprise",
    "enterprises",
    "collaboration",
    "collaborative",
    "pricing",
    "business",
    "model",
    "competitors",
    "alternatives",
    "platform",
    "software",
}


def _meaningful_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", text.lower())
        if len(token) > 3 and token not in RELEVANCE_STOPWORDS
    }


def _fuzzy_query_entity_match(query: str, result_text: str) -> bool:
    query_tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", query.lower())
        if len(token) > 2 and token not in RELEVANCE_STOPWORDS
    ]
    result_tokens = [token for token in re.split(r"[^a-z0-9]+", result_text.lower()) if len(token) > 2]
    if not query_tokens or not result_tokens:
        return False
    candidate_query = "".join(query_tokens[:2])
    for window_size in {1, 2}:
        for idx in range(0, max(0, len(result_tokens) - window_size + 1)):
            candidate_result = "".join(result_tokens[idx : idx + window_size])
            if SequenceMatcher(None, candidate_query, candidate_result).ratio() >= 0.86:
                return True
    return False


def _result_is_relevant(item: dict[str, Any]) -> bool:
    query = str(item.get("query") or "")
    title = str(item.get("title") or "")
    url = str(item.get("url") or "")
    snippet = str(item.get("snippet") or item.get("content") or "")
    evidence_type = str(item.get("evidence_type") or "")
    query_tokens = _meaningful_tokens(query)
    result_tokens = _meaningful_tokens(f"{title} {url} {snippet}")
    if not query_tokens:
        return True

    overlap = query_tokens & result_tokens
    first_query_token = next(iter(_meaningful_tokens(query.split()[0] if query.split() else "")), "")
    if first_query_token and first_query_token in result_tokens:
        return True
    if evidence_type in {"website", "news", "business_model", "leadership", "funding"} and _fuzzy_query_entity_match(query, f"{title} {url} {snippet}"):
        return True
    if evidence_type in {"market", "competitor"} and len(overlap) >= 2:
        return True
    if evidence_type in {"website", "news", "business_model", "leadership"} and len(overlap) >= 2:
        return True
    if evidence_type == "funding" and len(overlap) >= 1 and any(hint in url.lower() for hint in FUNDING_DATABASE_DOMAINS):
        return True
    return False


def _domain_matches(domain: str, patterns: tuple[str, ...]) -> bool:
    return any(domain == pattern or domain.endswith(f".{pattern}") or pattern in domain for pattern in patterns if not pattern.startswith("."))


def _domain_has_suffix(domain: str, suffix: str) -> bool:
    return domain == suffix.removeprefix(".") or domain.endswith(suffix)


def _is_tier_1_allowlisted(domain: str) -> bool:
    return _domain_matches(domain, tuple(pattern for pattern in TIER_1_ALLOWLIST_DOMAINS if not pattern.startswith("."))) or any(
        _domain_has_suffix(domain, suffix) for suffix in TIER_1_ALLOWLIST_DOMAINS if suffix.startswith(".")
    )


def _is_tier_2_allowlisted(domain: str) -> bool:
    return _domain_matches(domain, TIER_2_ALLOWLIST_DOMAINS)


def _is_low_quality_domain(domain: str, text: str) -> bool:
    return any(marker in domain or marker in text for marker in LOW_QUALITY_DOMAINS) or any(term in text for term in CONTENT_FARM_TERMS)


def _is_seo_or_listicle(text: str) -> bool:
    return any(term in text for term in SEO_TERMS) or any(term in text for term in LISTICLE_TERMS)


def _is_named_customer_case_study(title: str, url: str, evidence_type: str) -> bool:
    text = f"{title} {url}".lower()
    return evidence_type in {"news", "website", "business_model"} and any(term in text for term in ["case study", "customer story", "customers/"])


def _source_quality_sort_key(item: EvidenceItem) -> tuple[int, int, float]:
    quality_rank = {"high": 0, "medium": 1, "low": 2}
    return (
        int(item.get("source_tier", 3)),
        quality_rank.get(str(item.get("source_quality", "medium")), 1),
        -float(item.get("confidence", 0.5)),
    )


def is_company_source(evidence_type: str, title: str, url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    text = f"{title} {domain}".lower()
    if any(term in text for term in LISTICLE_TERMS):
        return False
    return evidence_type == "website" and any(term in text for term in ["official", "about", "company", "homepage"])


def is_tier_1_company_website(title: str, url: str, evidence_type: str) -> bool:
    text = f"{title} {url}".lower()
    if evidence_type != "website":
        return False
    if any(term in text for term in LISTICLE_TERMS):
        return False
    return any(term in text for term in ["official", "about", "company", "homepage"])


def source_quality(title: str, url: str, evidence_type: str, confidence: float) -> tuple[str, str]:
    domain = urlparse(url).netloc.lower()
    text = f"{title} {url}".lower()
    if _is_seo_or_listicle(text):
        return "low", "SEO/listicle source; use only as weak context or competitor discovery."
    if _is_low_quality_domain(domain, text):
        return "low", "Low-quality, social, review, forum, or content-farm source; use only as weak context."
    if is_tier_1_company_website(title, url, evidence_type) or _is_tier_1_allowlisted(domain) or _is_named_customer_case_study(title, url, evidence_type):
        return "high", "Tier 1 allowlisted source."
    if _is_tier_2_allowlisted(domain):
        return "medium", "Tier 2 allowlisted source."
    if confidence < 0.45:
        return "low", "Search confidence below threshold."
    return "medium", "Unlisted source; usable only with diligence verification."


def source_tier(title: str, url: str, evidence_type: str, confidence: float) -> tuple[int, float, str]:
    domain = urlparse(url).netloc.lower()
    text = f"{title} {url}".lower()
    if _is_seo_or_listicle(text) or _is_low_quality_domain(domain, text) or confidence < 0.45:
        return 3, 0.2, "Tier 3: Low trust / context only"
    if (
        is_tier_1_company_website(title, url, evidence_type)
        or _is_tier_1_allowlisted(domain)
        or _is_named_customer_case_study(title, url, evidence_type)
    ):
        return 1, 1.0, "Tier 1: Highest trust"
    if _is_tier_2_allowlisted(domain) or any(term in text for term in ["analyst", "interview", "podcast", "industry report", "trade publication"]):
        return 2, 0.6, "Tier 2: Medium trust"
    return 2, 0.6, "Tier 2: Secondary Expert"


def source_reliability_policy(title: str, url: str, evidence_type: str, confidence: float) -> dict[str, Any]:
    domain = urlparse(url).netloc.lower()
    path = urlparse(url).path.lower()
    text = f"{title} {url}".lower()
    source_type = "independent_news"
    if is_company_source(evidence_type, title, url):
        source_type = "company_owned"
    elif any(domain_hint in domain for domain_hint in FUNDING_DATABASE_DOMAINS) or evidence_type == "funding":
        source_type = "funding_database"
    elif evidence_type == "leadership" or any(domain_hint in domain or domain_hint in text for domain_hint in PROFESSIONAL_PROFILE_DOMAINS):
        source_type = "professional_profile"
    elif "case stud" in text or "customer story" in text or "customers/" in path:
        source_type = "customer_case_study"
    elif any(domain_hint in domain for domain_hint in WIRE_DOMAINS):
        source_type = "company_owned"
    elif _is_seo_or_listicle(text) and evidence_type == "competitor":
        source_type = "vendor_blog"
    elif any(term in text for term in VENDOR_COMPARISON_TERMS) and evidence_type == "competitor":
        source_type = "vendor_blog"
    elif confidence < 0.45 or _is_low_quality_domain(domain, text) or _is_seo_or_listicle(text):
        source_type = "generic_blog"
    elif any(domain_hint in domain for domain_hint in REPUTABLE_NEWS_DOMAINS):
        source_type = "independent_news"

    policy = SOURCE_POLICIES[source_type]
    return {
        "source_type": source_type,
        "independence_level": policy["independence_level"],
        "allowed_uses": list(policy["allowed_uses"]),
        "disallowed_uses": list(policy["disallowed_uses"]),
    }


def normalize_evidence(items: list[dict[str, Any]]) -> list[EvidenceItem]:
    seen: set[str] = set()
    evidence: list[EvidenceItem] = []

    ordered_items = sorted(
        items,
        key=lambda item: (
            0 if is_company_source(str(item.get("evidence_type", "")), str(item.get("title", "")), str(item.get("url", ""))) else 1,
            -float(item.get("confidence", 0.5)),
        ),
    )

    for item in ordered_items:
        if not _result_is_relevant(item):
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "Untitled source").strip()
        key = url or title.lower()
        if key in seen:
            continue
        seen.add(key)

        confidence = float(item.get("confidence", 0.5))
        evidence_type = str(item.get("evidence_type", "news"))
        quality, quality_notes = source_quality(title, url, evidence_type, confidence)
        tier, weight, tier_label = source_tier(title, url, evidence_type, confidence)
        reliability = source_reliability_policy(title, url, evidence_type, confidence)
        if quality == "low" and any(existing["evidence_type"] == item.get("evidence_type", "news") and existing.get("source_quality") != "low" for existing in evidence):
            continue

        evidence.append(
            {
                "source_id": "",
                "title": title,
                "url": url,
                "snippet": str(item.get("snippet") or item.get("content") or "").strip(),
                "query": str(item.get("query") or "").strip(),
                "evidence_type": item.get("evidence_type", "news"),
                "confidence": confidence,
                "retrieved_at": str(item.get("retrieved_at") or utc_now()),
                "source_quality": quality,
                "quality_notes": quality_notes,
                "source_tier": tier,
                "source_weight": weight,
                "source_tier_label": tier_label,
                **reliability,
            }
        )

    sorted_evidence = sorted(evidence, key=_source_quality_sort_key)
    return [
        {
            **item,
            "source_id": f"S{index + 1}",
        }
        for index, item in enumerate(sorted_evidence)
    ]


def mock_search(research_plan: list[ResearchQuery]) -> list[dict[str, Any]]:
    if not research_plan:
        return []
    results = []
    def tokens(text: str) -> set[str]:
        import re

        stop = {"company", "market", "context", "competitors", "news", "customers", "platform", "software"}
        return {token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) > 3 and token not in stop}

    for item in MOCK_EVIDENCE:
        item_text = f"{item.get('title', '')} {item.get('snippet', '')} {item.get('query', '')}"
        item_tokens = tokens(item_text)
        for planned in research_plan:
            if item["evidence_type"] != planned["evidence_type"]:
                continue
            query_tokens = tokens(planned["query"])
            if len(item_tokens & query_tokens) >= 2:
                enriched = dict(item)
                enriched["query"] = planned["query"]
                results.append(enriched)
                break
    return results


def tavily_search(research_plan: list[ResearchQuery]) -> list[dict[str, Any]]:
    api_key = get_secret("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured.")

    import requests

    output: list[dict[str, Any]] = []
    for planned in research_plan:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": planned["query"],
                "search_depth": "basic",
                "max_results": 4,
                "include_answer": False,
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        for result in data.get("results", []):
            output.append(
                {
                    "title": result.get("title", "Untitled source"),
                    "url": result.get("url", ""),
                    "snippet": result.get("content", ""),
                    "query": planned["query"],
                    "evidence_type": planned["evidence_type"],
                    "confidence": float(result.get("score") or 0.65),
                }
            )
    return output


def exa_search(research_plan: list[ResearchQuery]) -> list[dict[str, Any]]:
    api_key = get_secret("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("EXA_API_KEY is not configured.")

    import requests

    output: list[dict[str, Any]] = []
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    for planned in research_plan:
        response = requests.post(
            "https://api.exa.ai/search",
            headers=headers,
            json={"query": planned["query"], "numResults": 4, "contents": {"text": True}},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        for result in data.get("results", []):
            text = result.get("text") or ""
            output.append(
                {
                    "title": result.get("title", "Untitled source"),
                    "url": result.get("url", ""),
                    "snippet": text[:600],
                    "query": planned["query"],
                    "evidence_type": planned["evidence_type"],
                    "confidence": 0.7,
                }
            )
    return output


def run_search(research_plan: list[ResearchQuery], provider: str = "mock") -> list[dict[str, Any]]:
    if provider == "tavily":
        return tavily_search(research_plan)
    if provider == "exa":
        return exa_search(research_plan)
    return mock_search(research_plan)
