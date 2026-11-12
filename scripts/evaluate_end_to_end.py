"""Run saved summary-extracted claims through retrieval and verification.

This script evaluates the full *available* document-to-verdict path using
persisted claims that were previously extracted from actual summary text. It
validates each claim's source_text against the summary, performs independent
retrieval, verifies claims, and only then reads the injected-error manifest to
score the three pre-registered injected targets.

No target fact ID, injected value, mutation label, or expected verdict is
passed into retrieval or verification. The claim-extraction LLM is not invoked
by default because existing claims are reused. To refresh extraction, first run
`uv run python -m src.claim_extraction_runner --force` with Gemini credentials
configured, then rerun this evaluation.

Usage:
    uv run python scripts/evaluate_end_to_end.py
    uv run python scripts/evaluate_end_to_end.py --output results/end_to_end_report.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path, PureWindowsPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_builder import GROUND_TRUTH_PATH, retrieve_candidate_facts
from src.verification_engine import verify_claim_against_facts

CLAIMS_DIR = ROOT / "data" / "claims"
SUMMARIES_DIR = ROOT / "data" / "discharge_summaries"
ERROR_DIR = ROOT / "data" / "error_injected"
MANIFEST_PATH = ERROR_DIR / "error_injection_manifest.json"


def _normalise_source_text(value: Any) -> str:
    """Use the same whitespace/dash normalisation as claim extraction."""
    text = str(value or "").lower()
    text = text.replace("–", "-").replace("—", "-").replace("*", "")
    return " ".join(text.split())


def _case_id_from_filename(filename: str) -> str:
    match = re.match(r"(SYN-DS-\d+)", filename)
    if not match:
        raise ValueError(f"Cannot derive synthetic case ID from filename: {filename}")
    return match.group(1)


def _summary_path(payload: dict[str, Any], claims_path: Path) -> Path:
    raw = str(payload.get("summary_file", "") or "").replace("\\", "/")
    if raw:
        path = (ROOT / raw).resolve()
        if path.is_file():
            return path
    stem = claims_path.stem.removesuffix("_claims")
    case_id = _case_id_from_filename(stem)
    if "_E" in stem:
        return ERROR_DIR / f"{stem}.txt"
    return SUMMARIES_DIR / f"{case_id}.txt"


def _read_claims(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claims = payload.get("claims") if isinstance(payload, dict) else payload
    if not isinstance(claims, list):
        raise ValueError(f"Expected a list of claims in {path}")
    return payload if isinstance(payload, dict) else {}, claims


def _prediction_for_claim(claim: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    candidates = retrieve_candidate_facts(
        claim,
        record.get("ground_truth_facts", []),
        top_k=20,
        record=record,
    )
    result = verify_claim_against_facts(claim, candidates, record=record)
    return {
        "claim_id": claim.get("claim_id"),
        "source_text": claim.get("source_text", ""),
        "category": claim.get("category", ""),
        "subject": claim.get("subject", ""),
        "relation": claim.get("relation", ""),
        "value": claim.get("value", ""),
        "unit": claim.get("unit", ""),
        "route": claim.get("route", ""),
        "frequency": claim.get("frequency", ""),
        "duration": claim.get("duration", ""),
        "date_time": claim.get("date_time", ""),
        "candidate_count": len(candidates),
        "verdict": result.get("verdict", "ERROR"),
        "error_type": result.get("error_type"),
        "field_results": result.get("field_results", {}),
        "matched_fact_ids": result.get("matched_fact_ids", []),
        "top_candidates": [
            {
                "fact_id": item.get("fact", {}).get("fact_id"),
                "subject": item.get("fact", {}).get("subject"),
                "relation": item.get("fact", {}).get("relation"),
                "value": item.get("fact", {}).get("value"),
                "unit": item.get("fact", {}).get("unit"),
                "source_section": item.get("fact", {}).get("source_section"),
                "score": item.get("score"),
            }
            for item in candidates[:5]
        ],
    }


def _contains_incorrect_assertion(claim_row: dict[str, Any], wrong_value: str, target_subject: str) -> bool:
    row_text = " ".join(
        str(claim_row.get(key, "") or "")
        for key in ("source_text", "value", "unit", "route", "frequency", "duration")
    ).casefold()
    claim_subject = str(claim_row.get("subject", "") or "").casefold()
    wrong_value = str(wrong_value or "").casefold().strip()
    subject_match = bool(target_subject and (target_subject in claim_subject or claim_subject in target_subject))
    wrong_match = bool(wrong_value and wrong_value in row_text)
    return subject_match and wrong_match


def run(output_path: Path) -> dict[str, Any]:
    records = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    records_by_id = {item["case_metadata"]["case_id"]: item for item in records}
    claim_files = sorted(CLAIMS_DIR.glob("*_claims.json"))
    if not claim_files:
        raise FileNotFoundError(f"No saved claim files found in {CLAIMS_DIR}")

    predictions: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    exceptions: list[dict[str, str]] = []

    # Phase 1: run predictions. The error manifest is not loaded during this phase.
    for claims_path in claim_files:
        try:
            payload, claims = _read_claims(claims_path)
            case_id = _case_id_from_filename(claims_path.name)
            record = records_by_id[case_id]
            summary_path = _summary_path(payload, claims_path)
            if not summary_path.is_file():
                raise FileNotFoundError(f"Corresponding summary not found: {summary_path}")
            summary_text = _normalise_source_text(summary_path.read_text(encoding="utf-8"))
            source_checks = [
                bool(_normalise_source_text(claim.get("source_text", ""))
                     and _normalise_source_text(claim.get("source_text", "")) in summary_text)
                for claim in claims
            ]

            file_predictions = []
            for index, claim in enumerate(claims):
                try:
                    row = _prediction_for_claim(claim, record)
                except Exception as exc:  # preserve run completeness for diagnostics
                    row = {
                        "claim_id": claim.get("claim_id", f"C{index + 1:03d}"),
                        "source_text": claim.get("source_text", ""),
                        "category": claim.get("category", ""),
                        "subject": claim.get("subject", ""),
                        "relation": claim.get("relation", ""),
                        "value": claim.get("value", ""),
                        "unit": claim.get("unit", ""),
                        "candidate_count": 0,
                        "verdict": "ERROR",
                        "error_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "matched_fact_ids": [],
                        "top_candidates": [],
                    }
                    exceptions.append({"file": claims_path.name, "claim_id": str(row["claim_id"]), "error": str(exc)})
                row["source_text_valid"] = source_checks[index]
                row["summary_file"] = str(summary_path.relative_to(ROOT))
                row["case_id"] = case_id
                row["claim_file"] = claims_path.name
                file_predictions.append(row)
                predictions.append(row)

            summary_rows.append({
                "claim_file": claims_path.name,
                "summary_file": str(summary_path.relative_to(ROOT)),
                "case_id": case_id,
                "is_error_injected_summary": "_E" in summary_path.stem,
                "claim_count": len(claims),
                "source_text_valid_count": sum(source_checks),
                "source_text_invalid_count": len(source_checks) - sum(source_checks),
                "verdict_counts": dict(Counter(row["verdict"] for row in file_predictions)),
                "contradicted_claim_ids": [row["claim_id"] for row in file_predictions if row["verdict"] == "CONTRADICTED"],
                "error_claim_ids": [row["claim_id"] for row in file_predictions if row["verdict"] == "ERROR"],
            })
        except Exception as exc:
            exceptions.append({"file": claims_path.name, "error": f"{type(exc).__name__}: {exc}"})

    # Phase 2: only after all predictions are recorded do we load labels and score
    # the specifically documented, deliberately injected errors.
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else []
    scored_targets = []
    for entry in manifest:
        claims_filename = PureWindowsPath(str(entry.get("output_file", ""))).stem + "_claims.json"
        fact_record = records_by_id.get(entry.get("case_id"), {})
        target_fact = next((fact for fact in fact_record.get("ground_truth_facts", []) if fact.get("fact_id") == entry.get("fact_id")), {})
        target_subject = str(target_fact.get("subject", "")).split("(", 1)[0].strip().casefold()
        candidates = [row for row in predictions if row.get("claim_file") == claims_filename]
        matching = [row for row in candidates if _contains_incorrect_assertion(row, entry.get("incorrect_value", ""), target_subject)]
        expected_relation = str(target_fact.get("relation", "")).casefold()
        expected_error_type = {
            "dose": "dose_mismatch",
            "frequency": "frequency_mismatch",
            "result": "result_mismatch",
        }.get(expected_relation)
        scored_targets.append({
            "error_id": entry.get("error_id"),
            "case_id": entry.get("case_id"),
            "summary_file": str(entry.get("output_file", "")).replace("\\", "/"),
            "claim_file": claims_filename,
            "target_claims_found": len(matching),
            "target_claims": [
                {
                    "claim_id": row.get("claim_id"),
                    "predicted_verdict": row.get("verdict"),
                    "predicted_error_type": row.get("error_type"),
                    "expected_verdict": entry.get("expected_verdict", "CONTRADICTED"),
                    "expected_error_type_from_fact_relation": expected_error_type,
                    "matched_fact_ids": row.get("matched_fact_ids", []),
                    "matched_target_fact_id": entry.get("fact_id") in row.get("matched_fact_ids", []),
                    "source_text": row.get("source_text", ""),
                }
                for row in matching
            ],
        })

    total = len(predictions)
    verdict_counts = dict(Counter(row.get("verdict", "ERROR") for row in predictions))
    source_valid_count = sum(bool(row.get("source_text_valid")) for row in predictions)
    target_claim_count = sum(item["target_claims_found"] for item in scored_targets)
    detected_targets = sum(
        any(
            row.get("predicted_verdict") == item.get("target_claims", [{}])[0].get("expected_verdict", "CONTRADICTED")
            and row.get("predicted_error_type") == row.get("expected_error_type_from_fact_relation")
            for row in item.get("target_claims", [])
        )
        for item in scored_targets
    )
    all_targets_have_one = bool(scored_targets) and all(item["target_claims_found"] >= 1 for item in scored_targets)
    all_targets_detected = bool(scored_targets) and all(
        any(row.get("predicted_verdict") == "CONTRADICTED" and row.get("predicted_error_type") == row.get("expected_error_type_from_fact_relation") for row in item.get("target_claims", []))
        for item in scored_targets
    )

    report = {
        "title": "End-to-end verification run on saved summary-extracted claims",
        "generated_at_local": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": {
            "synthetic_only": True,
            "saved_summary_files": len(summary_rows),
            "saved_claim_count": total,
            "claim_source_text_validation": "Each persisted claim's source_text was checked against the corresponding summary text using the same whitespace/dash normalization used by claim extraction.",
            "claim_extraction_was_rerun": False,
            "claim_extraction_note": "Persisted claims were reused. Fresh Gemini extraction was not run in this environment because GEMINI_API_KEY/GEMINI_MODEL were not configured and the OpenAI client package was absent.",
            "verifier_inputs_do_not_include_error_manifest": True,
            "evaluator_manifest_loaded_after_predictions": True,
            "complete_natural_language_claim_extraction_quality_established": False,
        },
        "aggregate": {
            "claim_count": total,
            "source_text_valid_count": source_valid_count,
            "source_text_invalid_count": total - source_valid_count,
            "source_text_valid_rate": source_valid_count / total if total else None,
            "verdict_counts": verdict_counts,
            "contradictions_in_error_injected_summaries": {
                row["claim_file"]: row["contradicted_claim_ids"]
                for row in summary_rows if row["is_error_injected_summary"]
            },
            "known_injected_targets": len(scored_targets),
            "known_injected_target_claims_found": target_claim_count,
            "known_injected_targets_found": sum(item["target_claims_found"] >= 1 for item in scored_targets),
            "known_injected_targets_detected_with_expected_error_type": sum(
                any(row.get("predicted_verdict") == "CONTRADICTED" and row.get("predicted_error_type") == row.get("expected_error_type_from_fact_relation") for row in item.get("target_claims", []))
                for item in scored_targets
            ),
            "known_injected_target_recall": (
                sum(any(row.get("predicted_verdict") == "CONTRADICTED" for row in item.get("target_claims", [])) for item in scored_targets) / len(scored_targets)
                if scored_targets else None
            ),
            "end_to_end_precision_recall_f1": None,
            "why_precision_recall_f1_are_null": "Only three injected errors have authoritative target labels in the manifest. The other extracted claims are not fully clinician- or adjudicator-labelled, so computing global precision/recall/F1 would be misleading.",
        },
        "known_injected_error_results": scored_targets,
        "per_summary": summary_rows,
        "claim_predictions": predictions,
        "execution_errors": exceptions,
        "interpretation": [
            "This run evaluates retrieval and verification on claims previously extracted from actual summary texts; the claim extractor was not called again in this environment.",
            "All source_text strings must pass validation before a claim can count as faithful extraction output; this validates source traceability, not extraction completeness or semantic correctness.",
            "The three injected targets are scored after predictions. Their results are a small targeted sensitivity check, not a reliable estimate of generalization.",
            "Verdict counts on non-target claims are descriptive only because those claims do not have complete independent gold labels.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "end_to_end_evaluation_report.json")
    args = parser.parse_args()
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    report = run(output_path)
    aggregate = report["aggregate"]
    print("=== END-TO-END EVALUATION (SAVED CLAIMS) ===")
    print(f"Summaries: {len(report['per_summary'])}")
    print(f"Claims verified: {aggregate['claim_count']}")
    print(f"Claim source text validated: {aggregate['source_text_valid_count']}/{aggregate['claim_count']}")
    print(f"Verdicts: {aggregate['verdict_counts']}")
    print(f"Known injected errors detected: {aggregate['known_injected_targets_detected_with_expected_error_type']}/{aggregate['known_injected_targets']}")
    print("Global precision/recall/F1: NOT CALCULATED (insufficient claim-level gold labels)")
    print(f"Report: {output_path}")
    if report["execution_errors"]:
        print(f"Execution errors: {len(report['execution_errors'])}")
        raise SystemExit(1)
    if aggregate["source_text_invalid_count"]:
        raise SystemExit("One or more claims failed source_text validation.")


if __name__ == "__main__":
    main()
