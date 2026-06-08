# Deal Memo Drafter

Python + Streamlit prototype for the AIVC Deal Memo Drafter challenge. It turns rough partner notes into a concise, citation-backed first-draft investment memo with public enrichment, evidence quality scoring, diligence gaps, founder/team verification, and economic competitor mapping.

The prototype is designed to feel like an analyst copilot, not a generic summarizer. Partner notes anchor the initial company understanding; public research verifies, enriches, challenges, or marks claims as unknown.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For live enrichment, configure at least one provider:

```bash
export TAVILY_API_KEY="..."
# or
export EXA_API_KEY="..."
```

You can also put Streamlit secrets in `.streamlit/secrets.toml`:

```toml
TAVILY_API_KEY = "..."
EXA_API_KEY = "..."
```

## Run

```bash
streamlit run app.py
```

Use `mock` provider for deterministic demos and tests. Use `tavily` or `exa` for live public enrichment. If live search returns no evidence inside a sandboxed environment, run Streamlit from a normal terminal with network access and restart the app after setting secrets.

## App Experience

The UI is memo-first and diagnostics-second.

- `Memo`: concise partner-facing memo with Deal Snapshot, executive summary, founder/team, thesis, competition, risks, assumptions, and recommendation.
- `Evidence`: sources used, source quality, evidence coverage, claim verification, conflicts, and reliability warnings.
- `Diligence`: prioritized open questions, decision-blocking unknowns, diligence checklist, and risk-specific questions.
- `Trace`: technical pipeline outputs, raw extracted fields, validation details, and debug internals.

The app has two output modes:

- `Partner Memo`: concise default view for IC-style reading.
- `Full Analyst Report`: expanded memo with selected supporting tables, including Claim Verification, while keeping heavier diagnostics in Evidence or Trace.

## Core Pipeline

1. `extract_company_profile`  
   Extracts company name, product, buyer, workflow, market thesis, team signal, traction claims, risks, and diligence questions from notes.

2. `plan_research`  
   Builds targeted searches for company website, funding, news, leadership/founders, business model, market context, and competitors.

3. `run_research`  
   Runs Tavily, Exa, or mock search.

4. `build_evidence_store`  
   Normalizes, deduplicates, ranks, and classifies evidence with stable `source_id`s.

5. `refine_company_profile_from_evidence`  
   Uses public evidence to enrich or challenge the note-derived company understanding without blindly overwriting it.

6. `write_structured_memo`  
   Generates structured memo JSON with provenance for factual claims.

7. `generate_chart_data`  
   Produces chart/table-ready data for scorecards, coverage, risks, and competitors.

8. `validate_grounding`  
   Flags unsupported claims, weak-source overreach, product over-narrowing, target-company self-inclusion in competitors, funding-as-traction errors, noisy enrichment, and template leakage.

9. `render_outputs`  
   Renders the memo, evidence, diligence, and trace views.

## Evidence And Source Quality

Sources are classified by credibility and allowed use.

- Tier 1: company websites, official blogs, SEC/government filings, reputable funding databases, official press releases, public company filings, and major reputable publications.
- Tier 2: reputable industry publications, analyst-style reports, credible trade publications, partner/customer announcements, and verified company profiles.
- Tier 3: SEO blogs, vendor comparison pages, generic listicles, scraped databases, forums, social posts, and weak context sources.

Weak sources can appear in the Evidence tab but cannot drive high-conviction investment conclusions. Funding evidence is not treated as traction unless it explicitly includes operating metrics such as revenue, customers, usage, ARR, or retention.

## Claim Verification

The system classifies important memo claims as:

- verified by public evidence,
- supported by partner notes only,
- partially verified,
- inferred,
- unsupported / requires diligence,
- conflicting evidence found.

Claim Verification appears in the `Evidence` tab and in `Full Analyst Report`. It is intentionally omitted from the default `Partner Memo` mode so the first view stays concise. Partner notes are not treated as public verification, and public funding is not treated as commercial traction unless the source includes operating evidence.

## Founder And Founding Team Verification

The system is intentionally conservative:

- Names are extracted only from partner notes or public sources that explicitly mention founder/CEO/leadership context.
- If no supported name is found, the memo returns `Unknown`.
- A single public source is labeled `Single public source - requires corroboration`.
- Multiple public sources, or partner notes plus public evidence, improve verification status.
- LinkedIn/profile-style evidence is useful for preliminary team mapping but is not treated as definitive by itself.

## Competitive Landscape

Competitors are identified by economic competition, not keyword overlap alone.

The system evaluates:

- same buyer,
- same workflow,
- same budget owner,
- same replacement target,
- same job-to-be-done,
- product/category adjacency.

The Full Analyst Report keeps one detailed competitive table with:

```text
Company | Similarity | Buyer | Workflow | Replacement Target | Differentiation Factors | Funding / Scale | Revenue Evidence | Score | Sources
```

Rows are tailored by category where possible, including workspace/productivity, product design, spend management, healthcare AI, legaltech, climate tech, HRTech, supply chain, robotics, biotech, and defense technology.

If the company does not fit a known taxonomy, the app falls back to note-derived product, buyer, workflow, and generic software/infrastructure risk reasoning rather than forcing `Unknown` throughout the memo.

## Testing

Run the lightweight direct test harness:

```bash
python - <<'PY'
import inspect
import tests.test_validation as tv

passed = 0
failed = []
for name, fn in sorted(vars(tv).items()):
    if name.startswith("test_") and callable(fn):
        try:
            if inspect.signature(fn).parameters:
                failed.append((name, "requires fixtures/parameters"))
                continue
            fn()
            passed += 1
        except Exception as exc:
            failed.append((name, repr(exc)))

print(f"{passed} direct tests passed")
if failed:
    print("Failures/skipped:")
    for item in failed:
        print("-", item[0], item[1])
    raise SystemExit(1)
PY
```

Compile check:

```bash
python -m compileall app.py src tests
```

## What To Show In The Interview

- The system follows an agentic workflow: notes -> company understanding -> research planning -> evidence store -> claim verification -> memo -> validation.
- Partner notes anchor the analysis; public evidence enriches or challenges the note-derived view.
- The memo distinguishes public facts, partner-note claims, and analyst inference.
- Full Analyst Report includes Claim Verification so an interviewer can see how the system separates verified facts from partner-note-only and inferred claims.
- The source credibility framework prevents weak sources from driving strong conclusions.
- Founder/team extraction avoids hallucination and labels single-source names as preliminary.
- Competitive landscape uses buyer/workflow/budget/replacement overlap rather than keyword similarity.
- Missing information is aggregated into diligence priorities and decision-blocking unknowns.
- The Trace tab makes the LangGraph-style workflow easy to explain technically.

## Known Limitations

- Live research quality depends on search provider snippets and may miss paywalled/private sources.
- Founder/team extraction does not fully crawl every web page; it relies on returned titles/snippets and available public evidence.
- Crunchbase/PitchBook-style funding data is approximated through public search unless a dedicated provider is integrated.
- Private metrics such as ARR, NRR, gross margin, CAC, win/loss, and retention usually require diligence requests.
- Unknown-category companies are handled with generic fallbacks, but highly novel categories may still need analyst review to improve competitor and risk specificity.
- The prototype supports first-draft diligence; it is not intended to make final investment decisions.
