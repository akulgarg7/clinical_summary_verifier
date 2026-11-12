# Clinical Summary Verifier

An experimental Streamlit research prototype for checking factual claims in AI-generated clinical discharge summaries against structured **synthetic** ground-truth patient records. It is not a clinical decision-support system and must not be used to guide real patient care.

## What the application does

1. Reads a synthetic patient record and an existing discharge summary.
2. Reuses saved claim files when they exist; existing summaries and claims are **not regenerated automatically**.
3. Retrieves source facts independently for each extracted claim.
4. Compares values and populated medication fields using deterministic normalization and field-specific checks.
5. Displays claim verdicts, field-level comparisons, matched fact IDs, and retrieved candidate evidence.
6. Lets you download a JSON verification report from the app.

The extraction prompt now explicitly asks for claims to be split across medication regimen transitions and to use the smallest continuous source span that supports each atomic assertion. This prompt revision has not yet been validated with a fresh Gemini extraction run in this environment.

The code does not use an injected-error ID, configured target fact ID, or known wrong value to decide a verdict.

## Requirements

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/) (recommended), or `pip`
- A Gemini API key only when generating new summaries/claims or when the optional semantic fallback is enabled

## Setup with `uv` (recommended)

Open PowerShell in the project directory:

```powershell
uv sync
Copy-Item .env.example .env
```

Edit `.env` and set your own credentials/configuration:

```dotenv
GEMINI_API_KEY=your_real_gemini_api_key
GEMINI_MODEL=your_configured_gemini_model
SEMANTIC_VERIFICATION_ENABLED=false
```

Do not share or commit `.env`. Keep API keys out of screenshots, reports, and source files.

Start the app:

```powershell
uv run streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Setup with `pip`

If you are not using `uv`, create and activate a virtual environment, then install the runtime dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, configure Gemini as above, and run `streamlit run app.py`.

## Run the offline regression tests

No Gemini request is made by this test command:

```powershell
uv run python -m unittest discover -s tests -v
```

The regression tests exercise entity matching, the known `K⁺`/creatinine retrieval issue, temporal date handling, frequency/route/unit normalization, the previously reported false positives, the known injected creatinine mismatch, and the existing rivaroxaban dose/frequency error cases.

If you prefer pytest, it is included in the `dev` dependency group:

```powershell
uv run pytest -q
```

## Optional Gemini semantic fallback

Deterministic verification is the default. To try the experimental semantic fallback, edit `.env` and set:

```dotenv
SEMANTIC_VERIFICATION_ENABLED=true
```

This fallback makes one batch request after deterministic verification, and only for eligible uncertain claims that have retrieved source evidence and exactly one unresolved field. It accepts a model-supported result only when the response cites at least one fact ID supplied in that claim's retrieved evidence. It never overrides a deterministic `CONTRADICTED` verdict. If the request fails or the setting is false, deterministic results are retained. This safeguard does not make model adjudication infallible; inspect the displayed evidence and downloaded report.

The default cap is 60 eligible uncertain claims per request. To change it, optionally set `SEMANTIC_VERIFICATION_MAX_CLAIMS` in `.env`.

## Important project folders

- `data/patient_records/`: synthetic structured ground truth
- `data/discharge_summaries/`: saved original synthetic discharge summaries
- `data/error_injected/`: synthetic summaries with injected errors and prior reports
- `data/claims/`: saved extracted claim files
- `src/`: extraction, evidence retrieval, deterministic verification, and optional semantic fallback
- `tests/`: offline regression tests
- `scripts/`: evaluation/debug utilities

## Safety and evaluation notes

- All included patient data is synthetic.
- A missing source fact means **insufficient evidence**, not automatically a contradiction. Natural-language paraphrases can remain uncertain if the supplied facts do not settle them.
- The verifier is still a research prototype. The regression tests establish specific behaviors on included synthetic cases, not general clinical accuracy across hospitals, specialties, medications, or real patient records.
- Existing saved verification JSON files are preserved for comparison; rerun verification in the app to generate a report using the updated code.

## Research status and honest interpretation

See `results/RESEARCH_STATUS.md` for the combined results, known limitations, recommended paper wording, and reproduction protocol. The document distinguishes the controlled fact-level mutation benchmark from verification of claims previously extracted from saved summary text; their scores must not be merged.

## Current evaluation results (26 October 2026)

The current saved-claim verification run rechecked 1,157 persisted claims across 11 synthetic summary files. All claim source spans matched their summaries. Verdicts: 550 `SUPPORTED`, 3 `CONTRADICTED`, 56 `PARTIALLY_SUPPORTED`, and 548 `INSUFFICIENT_EVIDENCE`. All three registered injected errors remain detected. Global precision/recall/F1 is not calculated because only three claims have authoritative end-to-end labels; the other claims have not been independently adjudicated.

A separate post-hoc controlled-mutation validation on four synthetic records detected 325/333 generated contradictions (97.60% recall, 100% mutation precision, 98.78% mutation F1), with 0/290 exact controls falsely contradicted after correcting one lab-versus-medication relation-selection bug. Eight mutations were missed (five date changes and three numeric lab changes). Because the benchmark was examined during debugging, these results are post-hoc validation and must not be described as an untouched blind holdout.

A further targeted stress test detected 90/90 additional controlled field mutations, with 90/90 paired original controls remaining supported. Those controls were selected because the baseline verifier already supported them; this is not an unbiased specificity estimate, and the LLM extractor was not rerun for those mutations.

Read [`results/RESEARCH_STATUS.md`](results/RESEARCH_STATUS.md), [`results/END_TO_END_EVALUATION_SUMMARY.md`](results/END_TO_END_EVALUATION_SUMMARY.md), [`results/HELDOUT_EVALUATION_SUMMARY.md`](results/HELDOUT_EVALUATION_SUMMARY.md), and [`results/UNCERTAIN_VERDICT_AUDIT_AFTER_FIXES.md`](results/UNCERTAIN_VERDICT_AUDIT_AFTER_FIXES.md) before using any numbers in the paper. The audit flags 604 claims still labelled `PARTIALLY_SUPPORTED` or `INSUFFICIENT_EVIDENCE`; those require interpretation and must not all be forced to `SUPPORTED`.

### Reproduce the results

```powershell
uv run python -m unittest discover -s tests -v
uv run pytest -q
uv run python scripts/evaluate_end_to_end.py --output results/end_to_end_after_fixes.json
uv run python scripts/audit_uncertain_claims.py
uv run python scripts/evaluate_heldout.py
uv run python scripts/evaluate_additional_mutations.py
```

The saved claim files were previously extracted from summary text. Fresh Gemini extraction was not run in this execution environment. To rerun extraction, configure credentials in `.env`, back up `data/claims`, and use `uv run python -m src.claim_extraction_runner --force`. This overwrites saved claims.
