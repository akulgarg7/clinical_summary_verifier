# Semantic adjudication UI update

This update makes the Streamlit UI show the final claim-level verdict only.

## UI behaviour

- A claim receives the `by Gemini` marker only when the final claim-level verdict differs from the verdict produced before semantic adjudication.
- Claims whose final verdict remains `CONTRADICTED` stay in the existing discrepancy table. The table shows the final verdict and `by Gemini` only when Gemini changed that overall verdict.
- Claims whose final verdict changed to `SUPPORTED`, `PARTIALLY_SUPPORTED`, or `INSUFFICIENT_EVIDENCE` appear in a separate `Gemini-reviewed claims` section. This section shows only the final verdict, the `by Gemini` marker, the reviewed field, rationale, and supporting evidence where available.
- If Gemini confirms a verdict without changing the overall claim-level verdict, the UI does not show `by Gemini`.
- The UI does not display the original verdict or a before/after comparison. The raw verification object retains prior-verdict metadata for machine-readable research auditing.

## Batch status counters

The previous implementation counted `applied` as SUPPORTED + CONTRADICTED, while individual records marked an accepted INSUFFICIENT_EVIDENCE abstention as applied. This update makes the counters consistent:

- `applied` and `accepted`: number of accepted adjudication responses, including abstentions.
- `decisive`: number of accepted SUPPORTED or CONTRADICTED adjudications.
- `supported`, `contradicted`, `abstained`: outcome counts.

## Files in this update

Replace these complete files:

- `app.py`
- `src/semantic_adjudicator.py`
- `tests/test_semantic_adjudication_fallback.py`

No changes are required to `src/verification_engine.py` or `src/verification_pipeline.py` for this UI update.

## Testing

From the project root:

```powershell
uv run python -m compileall -q app.py src tests
uv run python -m pytest -q
```

The update was tested against the complete project snapshot available for this work: **72 passed, 13 subtests passed**. The additional test checks that accepted abstentions are counted consistently.

Then launch the app:

```powershell
uv run streamlit run app.py
```

For a visual check, use an actual result where Gemini changes the claim-level verdict, such as the live smoke-test case. Confirm that:

1. A changed-to-SUPPORTED result appears in `Gemini-reviewed claims` with `SUPPORTED` and `by Gemini` only.
2. A changed-to-CONTRADICTED result appears in the discrepancy table with `CONTRADICTED` and `by Gemini`.
3. A Gemini response that leaves the overall verdict unchanged does not receive the marker.
4. The prior-verdict value is not displayed in the Streamlit UI.

Semantic verification remains controlled by `SEMANTIC_VERIFICATION_ENABLED` in `.env`; this update does not change the setting in the user's local `.env`.
