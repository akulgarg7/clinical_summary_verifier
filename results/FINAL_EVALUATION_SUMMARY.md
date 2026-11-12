# Clinical Summary Verifier — Final evaluation snapshot

**Snapshot date:** 26 October 2026  
**Dataset:** synthetic-only, eight underlying records  
**Status:** updated deterministic verifier and offline tests pass; end-to-end gold-label coverage remains limited.

## What was changed

The latest changes address general classes of errors rather than case-specific IDs: subject/attribute-slot retrieval, month/day date compatibility, timepoint selection for serial laboratory results, duplicate structured/narrative representations of the same numeric result, and a cross-category bug that could resolve a laboratory concentration claim as a medication dose. These fixes were added with regression tests. The verifier continues to return `INSUFFICIENT_EVIDENCE` when source evidence is absent or insufficient; the goal was not to eliminate abstention at any cost.

## Offline validation

- `python -m unittest discover -s tests -v`: **57 tests passed**.
- `pytest -q`: **57 passed, 13 subtests passed**.
- Python compile check passed for `src`, `scripts`, and `tests`.

## A. Saved summary-derived claims

The verification engine reran over 11 saved original/error-injected summary files and 1,157 existing atomic claims. Source text alignment was valid for 1,157/1,157 claims. Gemini extraction was not freshly invoked during this run; previously extracted claims were reused.

| Verdict | Count | Share |
|---|---:|---:|
| `SUPPORTED` | 550 | 47.5% |
| `CONTRADICTED` | 3 | 0.3% |
| `PARTIALLY_SUPPORTED` | 56 | 4.8% |
| `INSUFFICIENT_EVIDENCE` | 548 | 47.4% |

All three pre-registered injected errors remained detected with the expected error type: admission creatinine result mismatch, rivaroxaban inpatient dose mismatch, and rivaroxaban discharge frequency mismatch. Only these three claims have authoritative manifest labels in the saved-claim evaluation; global precision/recall/F1 is therefore not calculated.

Compared with the initial saved-claim run (422 supported, 3 contradicted, 55 partial, 677 insufficient), the current run has 128 fewer claims in the two uncertainty categories combined (732 to 604). This redistribution is a useful audit signal, not proof that every revised verdict is correct. The audit CSV/Markdown retains individual claims, retrieved evidence, and automatic review flags. Exact value overlap is not treated as sufficient proof when entity, timepoint, units, polarity, or relation may differ.

## B. Controlled mutation benchmark (post-hoc validation)

A fact-level mutation benchmark on records `SYN-DS-0001`, `SYN-DS-0002`, `SYN-DS-0004`, and `SYN-DS-0005` used 290 exact controls and 333 generated contradictions. Current results:

| Metric | Result |
|---|---:|
| True positives | 325 |
| False negatives | 8 |
| False positives on exact controls | 0/290 |
| True negatives | 290 |
| Mutation precision | 100.00% |
| Mutation recall | 97.60% |
| Mutation F1 | 98.78% |
| Specificity | 100.00% |
| Overall accuracy | 98.72% |

The eight mutations not classified as `CONTRADICTED` were five date mutations and three numeric laboratory-value mutations (venous pH, serum bicarbonate, blood beta-hydroxybutyrate). This is **post-hoc validation, not a blind final holdout**: a false-positive control in this benchmark was inspected while debugging and prompted a category-level fix to stop laboratory-result claims being resolved as medication doses. The benchmark was rerun after that fix. Do not present these metrics as an untouched independent estimate.

## C. Additional controlled mutation stress test

A separate test generated **90 new mutations** spanning allergy polarity, duration, dates, frequency, laboratory numeric values, medication dose, multi-field medication instructions, narrative polarity, route, temporal value swaps, and units. All 90 were returned as `CONTRADICTED`. All 90 paired source controls remained `SUPPORTED` when rerun.

This is a verification-stage controlled stress test, not end-to-end extraction: it mutates saved claim objects and matching summary snippets, then passes the mutated claim to retrieval/verification. The LLM extractor is not rerun. The controls were selected from claims that the baseline verifier already marked `SUPPORTED`, so their 90/90 stability is not an unbiased specificity estimate. This 100% mutation-detection result must not be generalized beyond the test construction.

## Remaining limitations

1. The corpus contains only eight underlying synthetic records.
2. Fresh Gemini claim extraction was not run in this environment; the 1,157 saved extracted claims were reused.
3. 548 claims remain `INSUFFICIENT_EVIDENCE` and 56 remain `PARTIALLY_SUPPORTED`. Some abstentions are appropriate (for example, an interpretation such as “normal” without a documented reference range/interpretation), while others may still be retrieval/schema-alignment problems. The remaining 604 are not all confirmed errors, and the audit is not independent adjudication.
4. The additional 90 mutations are selected controlled perturbations; their success does not establish natural-summary error-detection performance.
5. No clinician-annotated or real-world clinical dataset was used. This is a research prototype, not a clinical decision support tool.

## Recommended paper wording

> In an evaluation of previously extracted atomic claims from 11 synthetic summary files, the verifier detected all three registered injected discrepancies among 1,157 claims. Following generalized retrieval and comparison corrections, 550 claims were labelled supported, 3 contradicted, 56 partially supported, and 548 insufficient evidence. In a separate post-hoc synthetic fact-level mutation benchmark, 325 of 333 generated contradictions were detected (97.60% recall, 100% mutation precision, 98.78% mutation F1), with no false contradiction labels among 290 exact controls after a category-level false-positive fix. A further targeted stress test detected 90/90 programmatically generated field mutations, but used saved claim objects rather than rerunning the LLM extractor and used controls selected from previously supported claims. The results support feasibility on the tested synthetic cases; the small dataset, lack of independently adjudicated labels for most claims, and absence of fresh extraction and real-world evaluation mean broader generalizability is not established.

## Reproduction

```powershell
uv sync
uv run python -m unittest discover -s tests -v
uv run pytest -q
uv run python scripts/evaluate_end_to_end.py --output results/end_to_end_after_fixes.json
uv run python scripts/audit_uncertain_claims.py
uv run python scripts/evaluate_heldout.py
uv run python scripts/evaluate_additional_mutations.py
```

To evaluate a fresh Gemini extraction run, configure credentials in `.env`, back up `data/claims`, then run `uv run python -m src.claim_extraction_runner --force` and repeat the saved-claim evaluation. `--force` overwrites the saved claim files.
