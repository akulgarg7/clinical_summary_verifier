# Controlled-mutation evaluation summary (post-hoc validation run)

**Run date:** 26 October 2026  
**Data:** synthetic only  
**Scope:** verification-stage controlled mutations generated from structured synthetic facts; not a fresh LLM extraction benchmark.

## Results

The evaluator used 290 exact controls and generated 333 contradiction mutations across records `SYN-DS-0001`, `SYN-DS-0002`, `SYN-DS-0004`, and `SYN-DS-0005`.

| Metric | Result |
|---|---:|
| Mutations detected as `CONTRADICTED` | 325/333 (97.60% recall) |
| Mutation precision | 100.00% |
| Mutation F1 | 98.78% |
| Missed mutations | 8/333 |
| Exact controls falsely marked `CONTRADICTED` | 0/290 |
| Exact-control specificity | 100.00% |
| Overall accuracy over controls + mutations | 98.72% |

The 8 missed mutations comprise five date changes and three numeric laboratory-value changes (venous pH, serum bicarbonate, and blood beta-hydroxybutyrate). A mutation that receives `SUPPORTED`, `PARTIALLY_SUPPORTED`, or `INSUFFICIENT_EVIDENCE` is counted as missed because the strict detection endpoint is `CONTRADICTED`.

## Important qualification: post-hoc, not blind holdout

The selected records were outside the targeted regression-case IDs, but they had already appeared in earlier audits. During this debugging round, one exact potassium-result control was inspected after it was wrongly classified as a medication dose mismatch. A general category-level rule was then added to prevent laboratory claims from being re-resolved as medication doses, and this benchmark was rerun. Consequently, these latest numbers are **post-hoc validation results, not an untouched blinded holdout**. Do not describe them as a final independent estimate of generalization.

The evaluator itself does not pass the target fact ID, mutation type, expected verdict, or changed value into retrieval or verification. Labels are used after predictions for scoring. Nevertheless, this is a small synthetic fact-level benchmark built from the same eight-record corpus, with claims constructed directly from source facts. It is easier than naturally authored summaries and does not demonstrate real-world clinical accuracy.

See `final_holdout_evaluation_report.json` for row-level predictions and evidence IDs.
