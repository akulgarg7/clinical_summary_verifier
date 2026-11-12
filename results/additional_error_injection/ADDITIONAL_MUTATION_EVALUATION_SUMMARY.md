# Additional controlled error-injection evaluation

Run time: 2026-10-26T05:35:03+00:00

## Scope and fairness

This test starts with atomic claims previously extracted from the original synthetic discharge summaries. It creates paired mutated summaries and mutated claim objects by changing one field at a time. The verifier receives only the mutated claim and the corresponding ground-truth record; mutation type, target fact ID, expected verdict, and changed value are kept in the scoring manifest and attached after predictions.

This is an additional **verification-stage controlled perturbation test**, not a fresh LLM extraction test. The baseline claims were selected because this verifier marked them SUPPORTED, so the original controls are a stability check and must not be interpreted as an unbiased specificity estimate. Cases come from the existing eight records, so the test is not an independent record-level holdout.

## Overall result

- Mutations generated: 90
- Detected as CONTRADICTED: 90/90 (100.0%)
- Not detected as CONTRADICTED: 0
- Controls rerun: 90; still SUPPORTED: 90; other verdict: 0; falsely CONTRADICTED: 0

## By error type

| Mutation type | N | Detected | Missed | Detection rate | Verdicts |
|---|---:|---:|---:|---:|---|
| allergy_polarity | 1 | 1 | 0 | 100.0% | CONTRADICTED: 1 |
| duration | 1 | 1 | 0 | 100.0% | CONTRADICTED: 1 |
| event_date | 6 | 6 | 0 | 100.0% | CONTRADICTED: 6 |
| frequency | 12 | 12 | 0 | 100.0% | CONTRADICTED: 12 |
| laboratory_numeric | 12 | 12 | 0 | 100.0% | CONTRADICTED: 12 |
| medication_dose | 12 | 12 | 0 | 100.0% | CONTRADICTED: 12 |
| multi_field_medication | 12 | 12 | 0 | 100.0% | CONTRADICTED: 12 |
| narrative_polarity | 3 | 3 | 0 | 100.0% | CONTRADICTED: 3 |
| route | 12 | 12 | 0 | 100.0% | CONTRADICTED: 12 |
| temporal_value_swap | 7 | 7 | 0 | 100.0% | CONTRADICTED: 7 |
| unit | 12 | 12 | 0 | 100.0% | CONTRADICTED: 12 |

## Missed mutations

No mutated claims were missed in this generated set.

## Interpretation and limitations

A mutation detection rate is reported instead of a global F1 score. The mutations are selected from verifier-supported baseline claims, which creates selection bias. The source-claim fields are deterministically updated to reflect the injected sentence; the LLM extractor is not rerun. This test evaluates whether the retrieval/verification stage catches additional controlled discrepancies; it does not establish end-to-end performance on independently generated summaries or real-world clinical notes.
