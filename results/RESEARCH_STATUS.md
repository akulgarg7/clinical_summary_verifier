# Research status and defensible conclusions

**Project:** Clinical Summary Verifier  
**Evaluation date:** 26 October 2026  
**Data:** synthetic-only dataset with eight underlying records. No real clinical records or clinician-annotated gold-standard set was used.

## Current evaluation results

### 1. Saved-claim verification run

- Saved summary files: 11
- Previously extracted atomic claims: 1,157
- Claim source spans matching source summaries: 1,157/1,157
- Registered injected errors detected with expected error type: 3/3

| Verdict | Count | Percentage |
|---|---:|---:|
| `SUPPORTED` | 550 | 47.5% |
| `CONTRADICTED` | 3 | 0.3% |
| `PARTIALLY_SUPPORTED` | 56 | 4.8% |
| `INSUFFICIENT_EVIDENCE` | 548 | 47.4% |

The initial saved-claim audit had 422 supported, 3 contradicted, 55 partially supported, and 677 insufficient-evidence claims. The updated engine moved many claims out of uncertainty, but that change does not prove every revised verdict is correct. The current audit lists 604 partial/insufficient cases for review; automated evidence-overlap flags are not gold labels.

The three registered errors are:
- `SYN-DS-0003_E003`, claim `C041`: admission creatinine 2.9 mg/dL versus source 1.9 mg/dL (`result_mismatch`).
- `SYN-DS-0006_E001`, claim `C065`: rivaroxaban inpatient dose 20 mg versus source 10 mg (`dose_mismatch`).
- `SYN-DS-0006_E002`, claim `C107`: rivaroxaban discharge frequency twice daily versus source once daily (`frequency_mismatch`).

Global precision/recall/F1 over the 1,157 claims is not calculated because only three injected errors have authoritative labels. The saved claims were previously extracted from summary text; Gemini extraction was not freshly invoked in this evaluation environment.

### 2. Controlled mutation evaluation over four records

The current post-hoc validation report evaluates 290 exact controls and 333 generated contradiction mutations on `SYN-DS-0001`, `SYN-DS-0002`, `SYN-DS-0004`, and `SYN-DS-0005`:

| Measure | Result |
|---|---:|
| Mutations detected | 325/333 (97.60% recall) |
| Mutation precision | 100.00% |
| Mutation F1 | 98.78% |
| Mutations not labelled contradicted | 8 (5 date, 3 numeric lab) |
| Exact controls falsely contradicted | 0/290 |
| Specificity on exact controls | 100.00% |

This is **post-hoc validation, not a blind final holdout**. Although these records were outside the targeted regression-test records, their saved claims were inspected in previous audits. During this round, one potassium-result control exposed a cross-category bug and was used to guide a general rule preventing laboratory claims from being treated as medication doses. The benchmark was rerun after that change. Do not treat these numbers as an untouched independent estimate.

### 3. Additional controlled mutation stress test

A separate script generated 90 additional field-level mutations across allergy polarity, duration, dates, frequency, laboratory numeric values, medication dose, multi-field medication changes, narrative polarity, route, temporal value swaps, and units. The verifier marked 90/90 as `CONTRADICTED`. The paired exact controls (90/90 `SUPPORTED`) were selected from claims the baseline verifier already supported, so this is a biased stability check, not an unbiased specificity estimate. The mutation script updates saved atomic-claim objects and summary snippets; it does not rerun the LLM extractor.

## Code and regression status

The current offline test suite passes **57 tests**, plus 13 pytest subtests. Added/fixed mechanisms include concept-aware entity matching, generic relation/attribute-slot normalization, precise date/day handling, lab timepoint selection, equivalent numeric structured/narrative result signatures, and lab-versus-medication dose disambiguation. The pipeline retains abstention when available evidence does not establish a claim; uncertainty must not be forcibly converted to `SUPPORTED`.

## Important limitations

1. Only eight underlying synthetic records are available.
2. The LLM extraction stage was not freshly rerun during this evaluation because the execution environment did not have the required Gemini configuration/client available.
3. The 90 additional mutations and the 333-mutation benchmark are controlled fact-level perturbations, not natural end-to-end summary errors.
4. Most of the 1,157 extracted claims do not have independent adjudicated labels. Overall end-to-end precision/recall/F1 therefore cannot be responsibly reported.
5. No clinician-annotated or real-world EHR evaluation was conducted. The prototype is not validated for clinical use.

## Suggested wording for the research paper

> In an evaluation on saved atomic claims extracted previously from synthetic discharge summaries, the evidence-grounded verification pipeline identified all three registered injected discrepancies. The updated run assigned 550 claims `SUPPORTED`, 3 `CONTRADICTED`, 56 `PARTIALLY_SUPPORTED`, and 548 `INSUFFICIENT_EVIDENCE` verdicts among 1,157 claims. In a separate post-hoc controlled-mutation validation, 325 of 333 generated contradiction mutations were detected (97.60% recall, 100% mutation precision, and 98.78% mutation F1); no exact control was labelled contradicted after a category-level false-positive fix. A further targeted stress test detected 90 of 90 generated field-level mutations, but its control set was selected from previously supported claims. Because the corpus is small, labels are synthetic/programmatically generated, most natural claims lack independently adjudicated verdicts, and extraction was not freshly rerun, these results demonstrate feasibility on tested synthetic cases rather than general clinical accuracy or external validity.

## Reproduction commands

```powershell
uv sync
uv run python -m unittest discover -s tests -v
uv run pytest -q
uv run python scripts/evaluate_end_to_end.py --output results/end_to_end_after_fixes.json
uv run python scripts/audit_uncertain_claims.py
uv run python scripts/evaluate_heldout.py
uv run python scripts/evaluate_additional_mutations.py
```

To test freshly extracted claims, save a backup of `data/claims`, configure Gemini credentials in `.env`, then run `uv run python -m src.claim_extraction_runner --force` followed by the end-to-end evaluation. The `--force` flag overwrites saved claim files.
