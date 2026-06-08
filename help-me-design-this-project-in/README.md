# Deal Memo Drafter

Python + Streamlit prototype for the AIVC Deal Memo Drafter challenge. It turns rough partner notes into a concise, citation-backed first-draft investment memo with public enrichment, evidence quality scoring, claim verification, diligence gaps, founder/team checks, and competitor mapping.

The main design principle is **note-first reasoning**: partner notes anchor the initial company understanding; public research verifies, enriches, challenges, or marks claims as unknown.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional live enrichment:

```bash
export TAVILY_API_KEY="..."
# or
export EXA_API_KEY="..."
```

You can also set keys in `.streamlit/secrets.toml`:

```toml
TAVILY_API_KEY = "..."
EXA_API_KEY = "..."
```

## Run

```bash
streamlit run app.py
```

Use `mock` for deterministic demos/tests. Use `tavily` or `exa` for live public research. If live search returns no evidence inside a sandbox, run Streamlit from a normal terminal with network access and restart after setting secrets.

## App Structure

- `Memo`: concise partner-facing memo and Deal Snapshot.
- `Evidence`: sources, source quality, evidence coverage, claim verification, and conflicts.
- `Diligence`: prioritized questions and decision-blocking unknowns.
- `Trace`: technical pipeline outputs and debug details.

Output modes:

- `Partner Memo`: default concise memo.
- `Full Analyst Report`: expanded version with supporting tables, including Claim Verification.

## Pipeline

1. Ingest raw notes.
2. Extract note-derived company profile: company, product, buyer, workflow, thesis, traction, team signal, risks, and diligence questions.
3. Resolve company identity separately from business understanding.
4. Plan and run public research through mock, Tavily, or Exa.
5. Normalize and score evidence with stable source IDs.
6. Generate memo JSON with provenance for facts, notes, and inferences.
7. Build chart/table data for scorecards, evidence coverage, risks, and competitors.
8. Validate grounding and remove or flag unsupported claims.
9. Render Memo, Evidence, Diligence, and Trace views.

## Key Features

- Separates partner-note claims from public verification.
- Keeps company understanding anchored to notes when public research is noisy.
- Handles ambiguous company names by separating `identity_confidence` from `understanding_confidence`.
- Extracts founders/team conservatively; returns `Unknown` if names are unsupported.
- Maps competitors by buyer, workflow, budget, replacement target, and job-to-be-done.
- Aggregates missing data into diligence priorities.
- Uses validation checks for unsupported claims, weak-source overreach, product over-narrowing, target-company self-inclusion, and template leakage.
