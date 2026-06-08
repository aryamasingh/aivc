# Deal Memo Drafter Architecture Notes

## Prototype

This project is a functioning Python + Streamlit prototype for a LangGraph-powered Deal Memo Drafter. It takes rough, unstructured partner notes and produces a structured first-draft investment memo with public-data enrichment, citations, evidence quality checks, diligence gaps, and analyst traceability. The app can run in mock mode for reliable demos or use Tavily/Exa for live enrichment from company websites, funding-style databases, news, market sources, and competitor references.

## How It Works

The workflow is organized as a graph of analyst-style steps:

1. **Note-first company understanding** extracts the target company, product description, target customer, workflow, market thesis, team signal, traction claim, and diligence questions directly from partner notes. These notes are the initial anchor.
2. **Identity resolution and company profile refinement** separates company identity confidence from business understanding confidence. Public evidence can verify, enrich, or challenge the notes, but weak or ambiguous sources cannot overwrite note-derived framing.
3. **Research planning and enrichment** creates targeted searches for website, funding, news, leadership, market context, competitors, and business model. The system supports live Tavily/Exa research plus deterministic mock evidence.
4. **Evidence store construction** normalizes, deduplicates, classifies, and ranks sources. Every source receives a stable `source_id`, source type, credibility tier, independence level, confidence, allowed uses, and disallowed uses.
5. **Memo generation** creates a partner-facing memo with executive summary, company overview, why now, why this company, market/customer pain, traction signals, competition summary, risks, what needs to be true, and recommendation.
6. **Validation and diagnostics** check unsupported claims, weak-source misuse, target-company self-inclusion in competitors, over-narrowed product framing, funding treated as traction, missing thesis elements, and template leakage.
7. **Streamlit rendering** shows a concise Deal Snapshot first, then four tabs: Memo, Evidence, Diligence, and Trace. Partner Memo mode stays concise; Full Analyst Report exposes deeper diagnostics.

## What Makes It Investment-Oriented

The prototype preserves provenance across the memo. Claims are labeled as public-source verified, partner-note-only, inferred, partially verified, conflicting, or unsupported. Coverage and confidence are tracked separately so the memo can answer both "how much do we know?" and "how trustworthy is it?" The system also aggregates missing data into decision-blocking diligence gaps, such as ARR, NRR, customer count, gross margin, deployment depth, retention, or win/loss data.

## Production Improvements

For production, I would replace the current heuristic-heavy parsing with structured LLM outputs plus schema validation, add a stronger entity-resolution layer using company domains and institutional databases, cache and persist evidence in a database, introduce source allowlists/denylists by industry, and add human review checkpoints before final memo export. I would also add observability for every graph node, regression evals for hallucination and leakage, role-based access control, encrypted secret management, and CRM/data-room integrations.

## Edge Cases

- **Sparse notes:** produce a low-confidence, note-grounded memo, limit charts, and prioritize clarification questions.
- **Conflicting public data:** preserve both interpretations, show the conflict, and avoid silently choosing one.
- **Ambiguous company identity:** keep workflow-level analysis when notes are clear, but restrict company-specific claims such as funding, investors, customers, valuation, and partnerships until identity is verified.
- **No web presence:** use partner notes only, mark public enrichment unavailable, and avoid source-backed claims or charts that lack evidence.
- **Weak sources:** vendor blogs, SEO pages, and social posts can provide context but cannot drive high-confidence conclusions or investment recommendations.
- **Template leakage:** validator checks prevent unrelated industry language, stale diligence prompts, and target-company aliases from appearing in competitor outputs.
