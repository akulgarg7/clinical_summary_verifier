"""Live Gemini smoke test for the semantic-adjudication fallback.

This is a focused integration check, not a research evaluation. It uses a real
synthetic source fact, retrieves evidence independently, and asks Gemini to
adjudicate an equivalent but deliberately unfamiliar frequency expression.

Required local .env values:
    GEMINI_API_KEY=...
    GEMINI_MODEL=...

Run:
    uv run python scripts/test_semantic_adjudication_live.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.evidence_builder import GROUND_TRUTH_PATH
from src.verification_pipeline import verify_claims_batch


def main() -> int:
    load_dotenv(ROOT / ".env")
    if not os.getenv("GEMINI_API_KEY") or not os.getenv("GEMINI_MODEL"):
        print(
            "Missing GEMINI_API_KEY or GEMINI_MODEL. Configure them in the local .env file."
        )
        return 2

    # This script explicitly enables the live fallback for this process only;
    # it does not edit the user's .env file.
    os.environ["SEMANTIC_VERIFICATION_ENABLED"] = "true"
    os.environ["SEMANTIC_VERIFICATION_MAX_CLAIMS"] = "1"

    records = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    record = next(
        item for item in records
        if item.get("case_metadata", {}).get("case_id") == "SYN-DS-0001"
    )
    source_fact = next(
        fact for fact in record.get("ground_truth_facts", [])
        if fact.get("fact_id") == "SYN-DS-0001-F023"
    )

    claim = {
        "claim_id": "C-LIVE-SEMANTIC-SMOKE",
        "category": "medication",
        "subject": source_fact["subject"],
        "relation": "frequency",
        "value": "one time in a day",
        "unit": "",
        "route": "",
        "frequency": "",
        "duration": "",
        "date_time": "",
        "time_of_day": "",
        "negated": False,
        "source_text": f"{source_fact['subject']} frequency one time in a day",
    }

    results, semantic_status = verify_claims_batch(
        [claim],
        record.get("ground_truth_facts", []),
        record=record,
        top_k=10,
    )
    result = results[0].get("result") if results else None

    print("Semantic adjudication status:")
    print(json.dumps(semantic_status, indent=2, ensure_ascii=False))
    print("Final verification result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if semantic_status.get("error"):
        print("FAIL: the semantic fallback could not complete; deterministic result was retained.")
        return 1
    if not result or result.get("verdict") != "SUPPORTED":
        print(
            "REVIEW: Gemini did not resolve the equivalent wording as SUPPORTED. "
            "Inspect the returned rationale; this smoke test is not a benchmark."
        )
        return 1
    if not result.get("semantic_adjudication", {}).get("applied"):
        print("FAIL: the result is SUPPORTED, but no semantic adjudication was recorded.")
        return 1

    print("PASS: Gemini resolved the unfamiliar wording using retrieved source evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
