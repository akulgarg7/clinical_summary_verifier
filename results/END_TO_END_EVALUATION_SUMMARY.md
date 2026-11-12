# End-to-end verification audit on saved extracted claims

**Run date:** 26 October 2026  
**Dataset:** synthetic only; eight underlying source records and 11 saved original/error-injected summary files.

## Scope

This run loads the atomic claims already saved under `data/claims/`, checks their `source_text` against each summary, independently retrieves evidence from the corresponding structured ground-truth record, and verifies each claim. The error-injection manifest is used by the evaluator only after predictions. The verifier is not given the target fact ID, expected verdict, mutation label, or deliberately wrong value.

The current environment did not freshly invoke the Gemini extractor; saved claims were reused. This is therefore an evaluation of the saved-claim verification path, not a newly run extraction-to-verification experiment.

## Latest verdict distribution

- Saved summary files evaluated: **11**
- Claims verified: **1,157**
- Claim source spans that matched their documents: **1,157/1,157**

| Verdict | Count | Share |
|---|---:|---:|
| `SUPPORTED` | 550 | 47.5% |
| `CONTRADICTED` | 3 | 0.3% |
| `PARTIALLY_SUPPORTED` | 56 | 4.8% |
| `INSUFFICIENT_EVIDENCE` | 548 | 47.4% |

All three registered injected errors were detected with the expected error type and matching target fact:

| Case | Claim | Expected discrepancy | Result |
|---|---|---|---|
| `SYN-DS-0003_E003` | `C041` | Admission creatinine 2.9 mg/dL vs ground truth 1.9 mg/dL | `CONTRADICTED`, `result_mismatch` |
| `SYN-DS-0006_E001` | `C065` | Inpatient rivaroxaban 20 mg vs ground truth 10 mg | `CONTRADICTED`, `dose_mismatch` |
| `SYN-DS-0006_E002` | `C107` | Rivaroxaban twice daily vs once daily | `CONTRADICTED`, `frequency_mismatch` |

## Change from initial saved-claim run

The initial run reported 422 `SUPPORTED`, 3 `CONTRADICTED`, 55 `PARTIALLY_SUPPORTED`, and 677 `INSUFFICIENT_EVIDENCE`. The latest code reports 550, 3, 56, and 548 respectively. Thus 128 fewer claims remain in the two uncertainty categories combined (732 down to 604). This is a change in verdict distribution, **not proof that all 128 reclassified claims are correct**. The audit records transitions in `uncertain_claims_after_fixes.csv` and `UNCERTAIN_VERDICT_AUDIT_AFTER_FIXES.md`.

Several broad fixes include safer entity matching, slot-aware relation resolution, explicit date/day disambiguation, timepoint-aware laboratory retrieval, treating equivalent structured and narrative results as one quantity where appropriate, and preventing a laboratory concentration unit such as mmol/L from causing a laboratory result to be resolved as a medication dose.

## What can and cannot be scored

The three registered mutation targets are the only authoritative end-to-end error labels in these saved outputs. Therefore, do **not** calculate global precision, recall, F1, or accuracy over all 1,157 claims as if the remaining 1,154 had verified gold labels. A 100% source-span match rate checks traceability only; it does not establish extraction completeness or correct field assignment.

There remain 548 `INSUFFICIENT_EVIDENCE` and 56 `PARTIALLY_SUPPORTED` claims. Some are justified abstentions; others may indicate retrieval/schema alignment defects. The current audit flags evidence for human review but does not independently adjudicate every uncertain claim. For example, a “normal glucose” assertion should remain uncertain if the source has a numeric value but no valid range or explicit interpretation.

## Reproduction

```powershell
uv run python -m unittest discover -s tests -v
uv run python scripts/evaluate_end_to_end.py --output results/end_to_end_after_fixes.json
uv run python scripts/audit_uncertain_claims.py
```

For a fresh LLM extraction evaluation, configure Gemini credentials in `.env`, back up `data/claims`, then run `uv run python -m src.claim_extraction_runner --force` followed by the evaluation command. `--force` regenerates and overwrites saved claims.
