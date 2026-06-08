# Repository Instructions

This project is a Deal Memo Drafter: a LangGraph-powered analyst copilot that turns rough partner notes into a structured first-draft investment memo with public-data enrichment.

## Expectations

- Preserve provenance across the system: distinguish partner-note facts, public-source facts, and analyst/model inferences.
- Do not allow weak public sources to support strong investment claims.
- Keep the IC-facing memo concise; put diagnostics, intermediate state, validation details, and graph traces in a separate trace/debug view.
- Run existing tests before finishing.
- If tests do not exist, add lightweight unit tests for any new parsing, scoring, or validation logic.
- Do not introduce new production dependencies unless necessary.
- Prefer deterministic post-processing rules for validation bugs, especially grounding, confidence, coverage, source-quality, and traceability issues.

## Product Judgment

The original assignment values investment judgment and trade-offs, not boilerplate generation. When changing the system, prioritize:

- Company-specific understanding before broad market research.
- Evidence quality over evidence quantity.
- Clear uncertainty over invented precision.
- Diligence questions that are traceable to current notes or cited evidence.
- Industry-specific reasoning when the notes support it, with graceful degradation to `Unknown` when they do not.
