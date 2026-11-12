# Implementation notes for the updated package

## Changes made

### Evidence retrieval

- Replaced unsafe raw-substring entity matching with token-boundary matching and a small, explicitly extensible alias normalizer for common laboratory analytes and measurements. This prevents `K⁺` from matching the `k` in `peak` and retrieving a creatinine result as potassium evidence.
- Strengthened medication-section ranking for pre-admission, inpatient, and discharge contexts.
- Added generic event/observation matching for facts stored under sections such as physical examination, vital signs, investigation, and laboratory observations.
- Prevented unrelated subjects such as `Primary specialty` and `Primary diagnosis` from matching only because they share a generic heading word.

### Deterministic verification

- Normalized common route aliases and equivalent frequency expressions, including `q6h PRN`, `every 6 hours as needed`, `every 6 hours (regular)`, `every 48 hours (renally adjusted)`, and `once daily before breakfast`.
- Separated delivery modes such as bolus/infusion from schedules, normalized duration wording, and mapped `resumed` to `continued` for status comparison.
- Compared compound admission date/time assertions against separate date and time facts when both are available.
- Fixed medication claims where an explicit temporal verb such as `started` coexists with a numeric dose value, while avoiding treating a narrative indication like `hypertension` as a dose merely because the extraction assigned a dose unit.
- Made contradiction decisions more conservative for narrative text. A phrase mismatch alone no longer establishes that an action, instruction, or narrative result is false. Explicit numeric, route, frequency, date, status, and similar structured mismatches remain eligible for deterministic contradiction findings.
- Preserved the field-by-field comparisons and candidate evidence for traceability.

### Optional semantic fallback

- Added `src/semantic_adjudicator.py` and an opt-in app integration.
- Disabled by default via `SEMANTIC_VERIFICATION_ENABLED=false`.
- Uses retrieved fact IDs as citation constraints. It only considers uncertain claims with one unresolved field, rejects unsupported/made-up citations, and never overrides a deterministic contradiction.
- The API fallback was not invoked during offline testing.

### Project readiness

- Added `pyproject.toml` for `uv sync` and a dev dependency group for pytest.
- Added `markdown-it-py` to the runtime dependencies required by `app.py`.
- Made Gemini client creation lazy in `claim_extractor.py`, so importing the app to view saved records/claims does not require an API key immediately. The key is required when a Gemini extraction call is requested.
- Added an offline `unittest` regression suite. Existing summaries, claims, injected summaries, and previous verification reports were not overwritten.
- The deliverable ZIP intentionally excludes `.env`, `.git`, `.venv`, bytecode caches, and secrets. Create your own `.env` from `.env.example` on your machine.

## Validation performed on the included synthetic data

`python -m unittest discover -s tests -v` passes all 57 offline regression tests (plus 13 pytest subtests) after the generalization changes.

The tests verify these known injected-error cases independently through retrieved evidence:

- `SYN-DS-0003_E003`: the admission creatinine claim `2.9 mg/dL` remains `CONTRADICTED` against the source value `1.9 mg/dL`; it is the only contradiction among the 81 extracted claims in that file after the false-positive fixes.
- `SYN-DS-0006_E001`: the injected rivaroxaban dose `20 mg` remains `CONTRADICTED` against the source dose `10 mg`; it is the only contradiction among the 118 claims in that file.
- `SYN-DS-0006_E002`: the injected rivaroxaban frequency `twice daily` remains `CONTRADICTED` against the source frequency `once daily`; it is the only contradiction among the 128 claims in that file.

In the latest end-to-end run, no other saved claims were labelled `CONTRADICTED`; this does not mean all other claims are correct. Most were `INSUFFICIENT_EVIDENCE`, and the remaining claims have not been exhaustively labelled by independent adjudicators. The `INSUFFICIENT_EVIDENCE` and `PARTIALLY_SUPPORTED` verdicts are uncertainty states, not necessarily errors in the summary.

## Running the current tests

```powershell
uv run python -m unittest discover -s tests -v
uv run streamlit run app.py
```


## Generalization work and held-out evaluation (2026-10-25)

- Added generic date conflict detection across relations such as `discontinued on`, scalar numeric comparison for structured measurements, opposite categorical/binary result handling, relation aliases for medication status, and a numeric guard before phrase-subset paraphrase acceptance. These changes address reusable comparison failure modes rather than hard-coded case IDs or wrong values.
- Fixed the mutation generator so a broad `daily` match cannot turn an existing `twice daily` assertion into a no-op. The evaluator now counts one exact control per eligible source fact and may generate multiple distinct contradiction mutations from that fact.
- Latest post-hoc controlled-mutation validation: 290 exact controls and 333 mutations across `SYN-DS-0001`, `SYN-DS-0002`, `SYN-DS-0004`, and `SYN-DS-0005`. 325/333 mutations were detected (97.60% recall; 100% mutation precision; F1=98.78%), with 0/290 controls falsely contradicted after a general lab-versus-medication relation-selection fix. Eight mutations were not detected as contradictions (five dates, three laboratory numeric values). Because one control was inspected to diagnose a bug before rerunning, this is post-hoc validation, not a blind holdout. Full output is `results/final_holdout_evaluation_report.json`.
- The eight missed mutations in the current post-hoc controlled-mutation validation consist of five date mutations and three numeric changes to venous pH, serum bicarbonate, and blood beta-hydroxybutyrate in `SYN-DS-0004`; they were returned as a non-contradicted verdict.
- Records `SYN-DS-0007` and `SYN-DS-0008` were used for iterative development and should not be presented as an independent test. Their validation report is `results/development_evaluation_report.json`.
- This experiment uses claim-like assertions constructed from synthetic facts and programmatically generated mutations. It is not a clinician-labeled natural-language evaluation and cannot establish real-world clinical validity. Records used for final evaluation were outside the targeted regression tests, but existing claims from them appeared in an earlier broad audit; record-level blindness is therefore incomplete.


## End-to-end saved-claim verification audit (2026-10-25)

- Reused the 1,157 persisted claims that had previously been extracted from 11 summary texts; verified `source_text` traceability for all 1,157 claims.
- Retrieval and verification ran before the evaluator loaded the error manifest. The verifier was not passed target IDs, expected verdicts, mutation labels, or injected values.
- Latest verdict distribution: 550 `SUPPORTED`, 3 `CONTRADICTED`, 56 `PARTIALLY_SUPPORTED`, and 548 `INSUFFICIENT_EVIDENCE`.
- All three manifest-labelled injected errors were detected with the expected error types and matched target facts: `SYN-DS-0003_E003` admission creatinine, `SYN-DS-0006_E001` inpatient rivaroxaban dose, and `SYN-DS-0006_E002` rivaroxaban discharge frequency.
- Precision/recall/F1 over all 1,157 claims are not calculated because only the three intentional injected targets have authoritative labels. Claim `source_text` alignment does not assess completeness or field extraction correctness.
- Fresh Gemini extraction was not run because no credentials were configured and the OpenAI client was absent in the execution environment. The saved-claim pipeline was tested, not an entirely fresh LLM extraction run.
- Full evaluation details: `results/end_to_end_evaluation_report.json`; explanatory summary: `results/END_TO_END_EVALUATION_SUMMARY.md`.


## Latest corrections and additional injection test (2026-10-26)

- Added regression coverage for generic attribute-slot matching, written-date day disambiguation, equivalent scalar/narrative lab-result evidence, and prevention of a laboratory concentration from being reinterpreted as a medication dose.
- The current suite passes 57 unit tests, plus 13 pytest subtests.
- Current saved-claim verification: 1,157 claims; 550 supported, 3 contradicted, 56 partially supported, 548 insufficient evidence; 3/3 registered injected errors detected. The initial run had 732 partial/insufficient verdicts; the current run has 604. This reduction is not independent adjudication of all changed verdicts.
- Current controlled-mutation validation: 325/333 mutations detected, 8 misses (five dates, three lab numeric values), zero false contradictions among 290 exact controls. This is post-hoc validation because a control was inspected to fix a false positive.
- Additional stress test: 90/90 controlled field mutations detected; 90/90 paired controls remained supported. Controls are selected from baseline-supported claims and mutations change saved claim objects and summary snippets without rerunning LLM extraction.
- See `results/FINAL_EVALUATION_SUMMARY.md` for current paper-safe metrics and limitations.
