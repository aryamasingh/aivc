# Deal Memo Drafter Architecture

## What It Does

The prototype is a Python + Streamlit analyst copilot that turns rough partner notes into a first-draft investment memo. It uses a LangGraph-style workflow to extract company understanding, enrich it with public research, score evidence quality, verify claims, identify diligence gaps, and render a concise partner-facing memo with supporting evidence available in secondary tabs.

The core design principle is **note-first reasoning**: partner notes establish the initial company framing, while public data is used to verify, enrich, challenge, or mark claims as unknown. Public search results should not overwrite clear note-derived context unless the source is credible and entity-matched.

## How It Works

1. **Note ingestion**  
   The user pastes unstructured notes into Streamlit. The app initializes workflow state with raw notes, errors, trace data, and optional search-provider settings.

2. **Fact extraction**  
   The pipeline extracts note-derived fields: company name, product description, buyer, workflow, market thesis, traction signal, team signal, risks, and diligence questions. Broad descriptions such as “platform,” “suite,” or “infrastructure” are preserved instead of forced into a narrow workflow.

3. **Identity resolution**  
   The system separates `identity_confidence` from `understanding_confidence`. If the exact public entity is ambiguous, company-specific claims such as funding, investors, customers, and partnerships are restricted, but workflow-level reasoning can still continue when the notes are clear.

4. **Public research**  
   Targeted searches collect company website, funding-style data, news, founder/team information, market context, business model signals, and competitor evidence. The app supports Tavily/Exa-style live enrichment plus mock data for demos and tests.

5. **Evidence scoring**  
   Search results are normalized, deduplicated, and assigned stable source IDs. Sources are tiered by credibility. Tier 1 sources include official company pages, SEC/government filings, reputable funding databases, press releases, and major business/industry publications. Lower-quality SEO, vendor, social, or listicle sources can appear as weak context but cannot drive high-conviction conclusions.

6. **Memo generation**  
   The memo generator creates a structured investment memo with provenance for key claims: public verified, partner-note-only, inferred, partially verified, conflicting, or unsupported. It includes company overview, why now, why this company, market/customer pain, traction signals, competition, risks, what needs to be true, recommendation, and diligence priorities.

7. **Validation**  
   A post-generation validator flags unsupported claims, overconfident language, weak sources used for strong conclusions, target-company self-inclusion in competitors, product over-narrowing, funding treated as traction, and missing thesis elements.

8. **UI rendering**  
   The app is memo-first. The `Memo` tab shows a concise Deal Snapshot and partner memo. `Evidence` contains sources, source quality, claim verification, and coverage. `Diligence` aggregates decision-blocking unknowns and questions. `Trace` keeps verbose diagnostics and intermediate node outputs.

## What Works Well

- Notes anchor the analysis, reducing noisy-search failures.
- Evidence quality is separated from evidence quantity, so weak sources cannot create high confidence.
- The memo distinguishes facts, inferences, partner-note claims, and public verification.
- Competitor analysis is based on economic overlap: buyer, workflow, budget, replacement target, and job-to-be-done.
- The system aggregates missing data into diligence priorities instead of scattering “unknowns” throughout the memo.
- The UI keeps the partner memo concise while preserving technical traceability for review.

## What I Would Change For Production

- Use stricter structured LLM schemas with automatic retries and stronger validation.
- Integrate reliable data providers such as Crunchbase, PitchBook, LinkedIn/company graph data, SEC/regulatory sources, and news APIs.
- Add deeper page retrieval rather than relying mainly on search snippets.
- Persist evidence, memo versions, analyst edits, and source metadata in a database.
- Add caching, async research, observability, audit logs, authentication, and production secret management.
- Build a formal evaluation set across industries to measure entity-resolution accuracy, citation quality, hallucination rate, and diligence usefulness.
- Add a human confirmation step when company identity or source quality falls below threshold.

## Edge Cases

- **Sparse notes:** Generate a low-confidence note-only memo, avoid unsupported claims, and prioritize clarification questions.
- **Conflicting public data:** Show both interpretations, mark the conflict, and avoid silently choosing one source.
- **Limited web presence:** Use partner notes as the main evidence, disable unsupported public-source claims, and clearly state that public enrichment was unavailable.
- **Ambiguous company names:** Separate identity from understanding. If identity confidence is low, restrict company-specific facts but still allow workflow-level risk, market, and diligence analysis when notes support it.
- **Low-quality sources:** Downgrade or exclude SEO blogs, vendor comparison pages, social posts, and generic listicles from confidence and recommendation logic.

## Known Limitations

- **Search can be noisy or incomplete.** The system now anchors on partner notes first, but Tavily/Exa results can still return irrelevant pages, weak snippets, or miss important paywalled/private sources. This is why the app separates public enrichment quality from note-derived understanding.
- **Entity resolution is still a hard problem.** Ambiguous company names can map to multiple real companies. The prototype scores identity confidence and restricts company-specific claims when uncertain, but production should use stronger domain matching, company databases, and analyst confirmation.
- **Product categorization can be imperfect.** Earlier failures over-narrowed broad platforms into specific workflows or misclassified companies based on unrelated search results. The current version preserves broad platform framing and flags conflicts, but highly novel categories may still require analyst review.
- **Source credibility is heuristic.** The system downgrades SEO pages, vendor blogs, social posts, and listicles, but production would need a maintained source-quality registry, better domain reputation checks, and deeper page parsing.
- **Competitor mapping is directional.** The prototype now uses buyer, workflow, budget, replacement target, and job-to-be-done instead of keyword overlap, but competitive intensity, win/loss data, and pricing overlap usually require customer calls or proprietary data.
- **Founder/team extraction is conservative.** It avoids hallucinating names and returns `Unknown` when evidence is weak, but search snippets may miss valid founder information that appears only inside full pages or LinkedIn-style profiles.
- **Private metrics remain unavailable.** ARR, NRR, gross margin, CAC, retention, customer count, expansion, and win/loss data are usually not public. The app should surface these as diligence gaps rather than inventing them.
- **Some validation is deterministic.** Guardrails catch common failures such as unsupported claims, template leakage, target-company self-inclusion, funding treated as traction, and overconfident conclusions. In production, I would add broader eval datasets and human feedback loops.
- **This is a first-draft diligence tool.** It supports analyst workflow and memo drafting, but it is not a final investment decision system.
