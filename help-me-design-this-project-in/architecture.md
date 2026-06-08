# Deal Memo Drafter Architecture

## 1. What The Prototype Does

This prototype is a Python + Streamlit deal memo drafter powered by a LangGraph-style pipeline. It takes rough, unstructured VC/PE partner notes and turns them into a structured first-draft investment memo with public-data enrichment, citations, evidence quality scoring, diligence gaps, and a technical trace.

The goal is to behave like an analyst copilot: understand the company from notes first, research public sources, verify key claims, identify what is still unknown, and produce a concise partner-facing memo.

## 2. Input / Output Flow

**Input**

- Raw partner notes pasted into the Streamlit app.
- Optional live enrichment through Tavily or Exa.
- Mock evidence fallback for reliable demos and tests.

**Output**

- Deal Snapshot with recommendation, confidence, decision readiness, and key gaps.
- Partner-facing memo.
- Evidence table with source quality and citations.
- Diligence questions and decision-blocking unknowns.
- Trace/debug view showing intermediate pipeline outputs.

## 3. Main Pipeline Stages

1. **Note ingestion**  
   Accept raw partner notes and initialize shared workflow state.

2. **Fact extraction**  
   Extract note-derived company understanding: target company, product, buyer, workflow, market thesis, team signal, traction claims, and diligence questions.

3. **Identity resolution**  
   Resolve the likely company entity and separate `identity_confidence` from `understanding_confidence`. If identity is uncertain, company-specific claims are restricted, but workflow-level analysis can continue when notes are clear.

4. **Public research**  
   Generate targeted searches for company website, funding-style data, news, leadership, market context, business model, and competitors. Results come from Tavily, Exa, or mock data.

5. **Evidence scoring**  
   Normalize, deduplicate, and classify sources. Each source receives a stable `source_id`, source type, credibility tier, confidence level, allowed uses, and disallowed uses.

6. **Memo generation**  
   Generate a structured investment memo with citations and provenance labels: public-source verified, partner-note-only, inferred, partially verified, conflicting, or unsupported.

7. **Validation**  
   Check for unsupported claims, weak sources used for strong conclusions, product over-narrowing, target-company self-inclusion in competitor lists, funding treated as traction, missing thesis elements, and template leakage.

8. **UI rendering**  
   Render the memo-first Streamlit interface: `Memo`, `Evidence`, `Diligence`, and `Trace`. Partner Memo mode stays concise; Full Analyst Report exposes deeper diagnostics.

## 4. What Works Well

- Notes anchor the analysis, so public research enriches or challenges the partner's framing instead of overwriting it.
- Evidence quality is separated from evidence quantity, so weak sources cannot create high-confidence conclusions.
- Claim verification makes the memo more diligence-oriented than a simple research summary.
- The UI is partner-friendly by default while preserving detailed diagnostics in secondary tabs.
- The validator catches common investment memo failures such as unsupported claims, template leakage, and treating funding as traction.

## 5. What Would Change For Production

- Replace more heuristic extraction with structured LLM outputs validated against strict schemas.
- Add stronger entity resolution using company domains, Crunchbase/PitchBook-style APIs, regulatory records, and manual confirmation when confidence is low.
- Persist companies, evidence, memo versions, source metadata, and analyst feedback in a database.
- Add asynchronous research, caching, source allowlists/denylists, and better recency handling.
- Add formal evals across industries and edge cases to measure hallucination rate, citation accuracy, and diligence usefulness.
- Add authentication, secret management, audit logs, observability, export workflows, and CRM/data-room integrations.

## 6. Edge Cases

- **Sparse notes:** Produce a low-confidence, note-grounded memo and prioritize clarification questions instead of inventing details.
- **Conflicting public data:** Preserve both interpretations, flag the conflict, and avoid silently choosing one source.
- **Companies with limited web presence:** Use partner notes only, mark public enrichment unavailable, and avoid unsupported public-source claims.
- **Ambiguous company names:** Separate company identity from business understanding. If identity confidence is low, restrict company-specific claims such as funding, investors, customers, valuation, and partnerships.
- **Low-quality sources:** Vendor blogs, SEO pages, social posts, and generic listicles can provide context but cannot drive high-confidence conclusions or investment recommendations.

## 7. Known Limitations

- The prototype still relies on heuristic post-processing in several places; production should use stronger structured extraction and validation.
- Live research quality depends on search API results and may miss paywalled or private data.
- Crunchbase-style funding data is approximated through public search unless a dedicated data provider is integrated.
- Some investment judgments are directional because private metrics such as ARR, NRR, gross margin, CAC, and retention are usually unavailable publicly.
- The app is designed for first-draft diligence support, not final investment decisioning.
