# Semantic adjudication fallback update

## Files supplied

Replace the existing files in the project root with the matching complete files
in this package:

- `app.py`
- `src/verification_engine.py`
- `src/semantic_adjudicator.py`
- `.env.example`

Add these new files:

- `src/verification_pipeline.py`
- `tests/test_semantic_adjudication_fallback.py`
- `scripts/test_semantic_adjudication_live.py`

No other project files need to be replaced.

## What changed

1. `verify_claims_batch()` now centralizes independent evidence retrieval,
   deterministic verification, and optional semantic adjudication. The Streamlit
   app uses this reusable batch pipeline.
2. The deterministic verifier marks only selected low-risk wording mismatches as
   `semantic_review_candidate`. An unfamiliar frequency phrase such as
   `one time in a day` can be escalated when compared with `once daily`.
3. Clear, explicit schedule conflicts such as `once daily` versus `twice daily`
   remain deterministic contradictions and are not sent for LLM override.
4. Gemini can return `SUPPORTED`, `CONTRADICTED`, or `INSUFFICIENT_EVIDENCE`.
   Decisive outcomes require citations to fact IDs included in that claim's
   retrieved-evidence prompt. Invented IDs are rejected.
5. A Gemini failure leaves the original deterministic batch unchanged.
6. Semantic verification remains disabled by default.

The LLM is not given mutation labels, expected verdicts, target fact IDs, or
known incorrect values. It is told which single field is unresolved because
that is the field it must adjudicate; the claim and retrieved source facts remain
its evidence.

## Offline tests

From the project root, run:

```powershell
uv run python -m compileall -q app.py src tests
uv run python -m pytest -q
```

The package was checked with the existing 57-test regression suite plus the
new semantic-adjudication tests: **71 passed, 13 subtests passed**. The test
suite uses local fixtures and mocked model responses; it does not call Gemini.

## Live Gemini smoke test

Do not overwrite an existing `.env` file. Add or confirm these values in the
local `.env` file:

```dotenv
GEMINI_API_KEY=your_actual_key
GEMINI_MODEL=your_configured_model
```

Then run:

```powershell
uv run python scripts/test_semantic_adjudication_live.py
```

This makes one live Gemini request using a synthetic source fact from
`SYN-DS-0001`, retrieved independently by the project's evidence retriever. It
checks whether `one time in a day` is interpreted as equivalent to `once daily`.
The script enables the fallback only for its own process and does not edit `.env`.
This is a smoke test of the LLM integration, not a performance/generalization
measurement.

## Run the application

With semantic adjudication still disabled (the safe default):

```powershell
uv run streamlit run app.py
```

To enable the fallback during application verification, set these values in
`.env`:

```dotenv
SEMANTIC_VERIFICATION_ENABLED=true
SEMANTIC_VERIFICATION_MAX_CLAIMS=60
```

Restart Streamlit after changing `.env`. Only eligible claims are sent to
Gemini. If the API call fails, deterministic results are retained.

## Scope and remaining evaluation

This update improves the adjudication path and makes it reusable by application
and evaluation code. The existing held-out/end-to-end evaluation scripts have
not yet been changed to run an LLM-on versus LLM-off comparison. That should be
done separately so the fallback's impact on false positives, false negatives,
precision, recall, F1, abstentions, and claim-type performance can be measured
rather than assumed.
