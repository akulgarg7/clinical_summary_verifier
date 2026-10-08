import json
from pathlib import Path

from claim_extractor import extract_claims
from evidence_builder import load_ground_truth, build_evidence
from verification_engine import verify_claim


TESTS = [
    {
        "case_id": "SYN-DS-0006",
        "error_id": "E001",
        "summary_file": "data/error_injected/SYN-DS-0006_E001.txt",
        "fact_id": "SYN-DS-0006-F139",
        "wrong_value": "20 mg",
        "expected_verdict": "CONTRADICTED",
    },
    {
        "case_id": "SYN-DS-0006",
        "error_id": "E002",
        "summary_file": "data/error_injected/SYN-DS-0006_E002.txt",
        "fact_id": "SYN-DS-0006-F141",
        "wrong_value": "twice daily",
        "expected_verdict": "CONTRADICTED",
    },
]


def load_summary(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def run_test(test, evidence):

    print("\n" + "=" * 70)
    print(f"Testing {test['case_id']} - {test['error_id']}")
    print("=" * 70)

    # Load discharge summary
    summary_path = Path(test["summary_file"])

    if not summary_path.exists():
        print(f"\nERROR: File not found:")
        print(summary_path)
        return

    summary = load_summary(summary_path)

    # Extract claims
    claims = extract_claims(summary)

    print(f"\nExtracted {len(claims)} claims.")

    # Get target ground-truth fact
    target_fact = evidence[
        test["case_id"]
    ]["fact_index"][
        test["fact_id"]
    ]

    print("\nTARGET GROUND-TRUTH FACT:")
    print(target_fact)

    # Find injected claim
    target_claim = None

    wrong_value = test["wrong_value"].lower()

    for claim in claims:

        fields_to_check = [
            claim.get("value", ""),
            claim.get("unit", ""),
            claim.get("frequency", ""),
            claim.get("route", ""),
            claim.get("duration", ""),
            claim.get("source_text", ""),
        ]

        combined_text = " ".join(
            str(field).lower()
            for field in fields_to_check
            if field
        )

        if wrong_value in combined_text:
            target_claim = claim
            break

    if target_claim is None:
        print(
            f"\nERROR: Could not find injected value "
            f"'{test['wrong_value']}'."
        )
        return

    # Verify
    result = verify_claim(
        target_claim,
        target_fact
    )

    print("\n" + "=" * 70)
    print("TARGET CLAIM")
    print("=" * 70)

    print(target_claim)

    print("\n" + "=" * 70)
    print("GROUND TRUTH")
    print("=" * 70)

    print(target_fact)

    print("\n" + "=" * 70)
    print("VERIFICATION RESULT")
    print("=" * 70)

    print(result)

    print("\n" + "=" * 70)
    print("EXPECTED")
    print("=" * 70)

    print(
        f"Verdict: {test['expected_verdict']}"
    )

    # Check result
    if result["verdict"] == test["expected_verdict"]:
        print("\nPASS")
    else:
        print("\nFAIL")


def main():

    records = load_ground_truth()
    evidence = build_evidence(records)

    for test in TESTS:
        run_test(test, evidence)


if __name__ == "__main__":
    main()