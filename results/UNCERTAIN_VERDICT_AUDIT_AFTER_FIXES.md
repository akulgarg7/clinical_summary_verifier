# Uncertain-verdict audit after generalized fixes

This report was generated from the latest saved-claim verification output. It is an audit aid, not independent adjudication.

## Current verdicts

- Claims verified: **1157**
- Supported: **550**
- Contradicted: **3**
- Partially supported: **56**
- Insufficient evidence: **548**
- Uncertain claims listed in CSV: **604**

## Change from initial saved-claim report

| Previous verdict | Current verdict | Count |
|---|---|---:|
| CONTRADICTED | CONTRADICTED | 3 |
| INSUFFICIENT_EVIDENCE | INSUFFICIENT_EVIDENCE | 538 |
| INSUFFICIENT_EVIDENCE | PARTIALLY_SUPPORTED | 8 |
| INSUFFICIENT_EVIDENCE | SUPPORTED | 131 |
| PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | 47 |
| PARTIALLY_SUPPORTED | SUPPORTED | 8 |
| SUPPORTED | INSUFFICIENT_EVIDENCE | 10 |
| SUPPORTED | PARTIALLY_SUPPORTED | 1 |
| SUPPORTED | SUPPORTED | 411 |

## Review-priority signals in uncertain claims

- No candidates retrieved: **81**
- At least one exact value among top retrieved candidates: **78**
- Exact value in a candidate with subject-compatible wording: **62**

These are automatic review flags only. Exact numeric/text overlap does not prove support if the entity, unit, relation, timepoint, polarity, or context differs.

## Interpretation and limitations

The number of `INSUFFICIENT_EVIDENCE` and `PARTIALLY_SUPPORTED` verdicts is not itself an error count. Some are justified abstentions (for example, a normality claim when the source gives a number but no normal range or interpretation); others indicate retrieval or schema-alignment failures. The CSV includes source text, field-level results, candidate facts, and previous verdicts to support manual review.

Do not calculate overall precision/recall/F1 using unlabelled claims as if the system's own verdicts were gold labels. The three registered mutation targets are the only authoritative end-to-end injected-error labels in this dataset.

## Reproducibility

Run `python scripts/evaluate_end_to_end.py --output results/end_to_end_after_fixes.json` first, then run `python scripts/audit_uncertain_claims.py`. The script uses only the saved ground-truth-derived evaluation report and performs no API calls.
