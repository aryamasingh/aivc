# Deal Memo Drafter

Python + Streamlit prototype for the AIVC Deal Memo Drafter challenge. It takes rough partner notes and produces a structured, citation-backed first-draft investment memo with research evidence, open questions, and analyst charts.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For live enrichment, configure one provider:

```bash
export TAVILY_API_KEY="..."
# or
export EXA_API_KEY="..."
```

## Run

```bash
streamlit run app.py
```

Use `mock` provider for a deterministic demo. Use `tavily` or `exa` for live public enrichment.

## What To Show In The Interview

- The memo is not a pure summarizer: it extracts entities, plans research, builds an evidence store, writes cited claims, validates grounding, and generates charts.
- Every factual memo claim is tied to public `source_ids`, partner `[Notes]`, or `[Inference]`.
- The memo leads with Executive Summary, Recommendation, Key Signals, Source Attribution, and Section Confidence so it reads like an investment document.
- Charts are directional and cite the sources used.
- The Trace tab makes the LangGraph workflow easy to discuss.

## Project Structure

```text
app.py
src/
  graph.py
  nodes.py
  render.py
  sample_data.py
  search.py
  state.py
tests/
  test_validation.py
architecture.md
requirements.txt
```
