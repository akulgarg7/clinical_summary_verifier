"""Targeted live tests for the Gemini semantic-adjudication fallback.

This script supplements, rather than replaces, the existing smoke test and
mocked unit tests. It makes two live Gemini requests:
  1. A semantic frequency conflict that should be escalated and contradicted.
  2. An insufficient-evidence case that should make Gemini abstain.

It also runs a clear deterministic contradiction that should NOT be sent to
Gemini. All inputs are synthetic test fixtures or the project's synthetic data.

Required .env values:
    GEMINI_API_KEY=...
    GEMINI_MODEL=...

Run from the project root:
    uv run python scripts/test_semantic_adjudication_scenarios.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.evidence_builder import GROUND_TRUTH_PATH
from src.semantic_adjudicator import maybe_apply_semantic_adjudications
from src.verification_pipeline import verify_claims_batch


def _print_result(title: str, result: dict[str, Any], status: dict[str, Any]) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    print("Semantic status:")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    print("Final result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _load_test_record() -> tuple[dict[str, Any], dict[str, Any]]:
    records = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    record = next(
        item for item in records
        if item.get("case_metadata", {}).get("case_id") == "SYN-DS-0001"
    )
    fact = next(
        item for item in record.get("ground_truth_facts", [])
        if item.get("fact_id") == "SYN-DS-0001-F023"
    )
    return record, fact


def _make_frequency_claim(source_fact: dict[str, Any], claim_value: str, claim_id: str) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "category": "medication",
        "subject": source_fact["subject"],
        "relation": "frequency",
        "value": claim_value,
        "unit": "",
        "route": "",
        "frequency": "",
        "duration": "",
        "date_time": "",
        "time_of_day": "",
        "negated": False,
        "source_text": f"{source_fact['subject']} frequency {claim_value}",
    }


def _run_pipeline_case(
    *,
    name: str,
    claim: dict[str, Any],
    record: dict[str, Any],
    expected_verdict: str,
    should_call_gemini: bool,
) -> bool:
    results, status = verify_claims_batch(
        [claim],
        record.get("ground_truth_facts", []),
        record=record,
        top_k=10,
    )
    result = results[0].get("result") if results else None
    result = result or {}
    _print_result(name, result, status)

    actual_verdict = str(result.get("verdict", "")).upper()
    adjudication = result.get("semantic_adjudication") or {}
    applied = bool(adjudication.get("applied"))
    frequency_result = (result.get("field_results") or {}).get("frequency", {})
    marked_candidate = bool(frequency_result.get("semantic_review_candidate"))

    checks = [actual_verdict == expected_verdict]
    if should_call_gemini:
        checks.extend([
            marked_candidate or applied,
            applied,
            str(adjudication.get("verdict", "")).upper() == expected_verdict,
            bool(adjudication.get("evidence_fact_ids")),
        ])
    else:
        checks.extend([
            not applied,
            int(status.get("considered", 0)) == 0,
            int(status.get("applied", 0)) == 0,
        ])

    passed = all(checks)
    print("TEST RESULT:", "PASS" if passed else "FAIL")
    if not passed:
        print(
            "Expected verdict:", expected_verdict,
            "| semantic candidate:", marked_candidate,
            "| adjudication applied:", applied,
        )
    return passed


def _run_insufficient_evidence_case() -> bool:
    """Call Gemini with a synthetic, deliberately non-specific source fact.

    This isolates the adjudicator's abstention behaviour. It does not test
    retrieval because the batch result is a controlled fixture representing an
    uncertain result that retrieval/verification has already produced.
    """
    claim_id = "C-LIVE-SEMANTIC-INSUFFICIENT"
    fact_id = "LIVE-SYNTHETIC-INSUFFICIENT-FREQ-001"
    claim_value = "once daily"
    source_value = "frequency is not documented in the available source record"

    claim = {
        "claim_id": claim_id,
        "category": "medication",
        "subject": "Synthetic test medication",
        "relation": "frequency",
        "value": claim_value,
        "unit": "",
        "route": "",
        "frequency": "",
        "duration": "",
        "date_time": "",
        "time_of_day": "",
        "negated": False,
        "source_text": "Synthetic test medication schedule is not stated.",
    }
    fact = {
        "fact_id": fact_id,
        "subject": "Synthetic test medication",
        "category": "medication",
        "relation": "frequency",
        "value": source_value,
        "unit": "",
        "date_time": "",
        "source_section": "synthetic_test_fixture",
    }
    batch_item = {
        "claim": claim,
        "candidates": [{"fact": fact, "score": 1.0, "match_reasons": ["synthetic test fixture"]}],
        "result": {
            "claim_id": claim_id,
            "verdict": "INSUFFICIENT_EVIDENCE",
            "error_type": None,
            "field_results": {
                "frequency": {
                    "claim_value": claim_value,
                    "source_value": source_value,
                    "matched_fact_id": fact_id,
                    "verdict": "INSUFFICIENT_EVIDENCE",
                    "ambiguous_evidence": True,
                    "matched_relation": "frequency",
                    "matched_fact_ids": [fact_id],
                    "semantic_review_candidate": True,
                }
            },
            "matched_fact_ids": [fact_id],
        },
        "error": None,
    }

    results, status = maybe_apply_semantic_adjudications([batch_item])
    result = results[0].get("result") if results else {}
    _print_result("TEST 3 — GEMINI ABSTENTION WITH INSUFFICIENT EVIDENCE", result or {}, status)

    adjudication = (result or {}).get("semantic_adjudication") or {}
    passed = (
        (result or {}).get("verdict") == "INSUFFICIENT_EVIDENCE"
        and adjudication.get("applied") is True
        and adjudication.get("verdict") == "INSUFFICIENT_EVIDENCE"
        and int(status.get("abstained", 0)) == 1
    )
    print("TEST RESULT:", "PASS" if passed else "FAIL")
    if not passed:
        print("Expected Gemini to return INSUFFICIENT_EVIDENCE and abstain.")
    return passed


def main() -> int:
    load_dotenv(ROOT / ".env")
    if not os.getenv("GEMINI_API_KEY") or not os.getenv("GEMINI_MODEL"):
        print("Missing GEMINI_API_KEY or GEMINI_MODEL in the project .env file.")
        return 2

    # Enable the fallback only for this Python process. The .env file is not edited.
    os.environ["SEMANTIC_VERIFICATION_ENABLED"] = "true"
    os.environ["SEMANTIC_VERIFICATION_MAX_CLAIMS"] = "1"

    record, source_fact = _load_test_record()
    checks: list[bool] = []

    # Test 1: an explicit, clear contradiction should remain deterministic and
    # should not be sent to Gemini.
    checks.append(_run_pipeline_case(
        name="TEST 1 — CLEAR CONTRADICTION MUST STAY DETERMINISTIC",
        claim=_make_frequency_claim(source_fact, "twice daily", "C-LIVE-SEMANTIC-CLEAR-CONFLICT"),
        record=record,
        expected_verdict="CONTRADICTED",
        should_call_gemini=False,
    ))

    # Test 2: this non-canonical schedule is deliberately eligible for semantic
    # review. Against once-daily evidence it means once every two days, so the
    # expected semantic verdict is CONTRADICTED.
    checks.append(_run_pipeline_case(
        name="TEST 2 — SEMANTICALLY DIFFERENT FREQUENCY SHOULD BE CONTRADICTED",
        claim=_make_frequency_claim(source_fact, "once every other day", "C-LIVE-SEMANTIC-FREQ-CONFLICT"),
        record=record,
        expected_verdict="CONTRADICTED",
        should_call_gemini=True,
    ))

    # Test 3: a constructed evidence fixture contains no actual schedule.
    # This tests Gemini's abstention response and citation-free abstention path.
    checks.append(_run_insufficient_evidence_case())

    print("\n" + "=" * 72)
    print(f"SUMMARY: {sum(checks)}/{len(checks)} scenarios passed")
    print("Note: this is a small integration check, not a model accuracy benchmark.")
    print("=" * 72)
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
