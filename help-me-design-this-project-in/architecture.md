# Deal Memo Drafter Architecture

## How It Works

The prototype converts rough partner notes into a structured, source-cited deal memo through a LangGraph workflow:

1. Extract a company profile from notes.
2. Plan targeted research queries for company website, funding, news, leadership, market, competitors, and business model.
3. Run Tavily or Exa search, with a mock fallback for demo reliability.
4. Normalize public results into an evidence store with stable source IDs.
5. Draft a structured memo where every claim is grounded in public sources, partner notes, or labeled analyst inference.
6. Validate citation coverage and produce analyst charts.
7. Render memo, evidence, charts, open questions, and workflow trace in Streamlit.

## Production Improvements

- Replace heuristic drafting with OpenAI structured outputs for extraction, research planning, memo writing, and citation validation.
- Persist companies, meetings, memos, evidence, and partner feedback in a database.
- Add a human review checkpoint before memo finalization.
- Use source-quality ranking and domain allow/deny lists for stronger evidence hygiene.
- Add a knowledge graph with Company, Founder, Investor, Market, and Competitor nodes.

## Edge Cases

- Sparse notes produce a low-confidence memo and open diligence questions.
- Missing web presence keeps the memo note-grounded and disables unsupported charts.
- Conflicting data should preserve both sources and flag the conflict.
- Ambiguous company names should ask the analyst to confirm the target before treating public data as factual.

## Evaluation

- Memo quality: accuracy, completeness, readability, and investment usefulness.
- Grounding: percentage of factual claims with source IDs, note references, or labeled inference.
- Hallucination rate: unsupported factual claims found by human review.
- Analyst leverage: time from raw notes to first draft versus the current 2-3 hour manual workflow.
