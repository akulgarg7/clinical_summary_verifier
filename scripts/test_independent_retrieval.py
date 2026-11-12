
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_builder import retrieve_candidate_facts
from src.verification_engine import verify_claim_against_facts

GROUND_TRUTH = (
    ROOT
    / "data"
    / "patient_records"
    / "synthetic_discharge_ground_truth.json"
)

CLAIM_FILES = [
    ROOT / "data" / "claims" / "SYN-DS-0006_E001_claims.json",
    ROOT / "data" / "claims" / "SYN-DS-0006_E002_claims.json",
]


def is_medication_claim(claim):
    """Identify medication claims across differently named categories."""
    category = str(claim.get("category", "")).strip().casefold()
    relation = str(claim.get("relation", "")).strip().casefold()

    # Include claims from explicitly medication-related categories.
    if "medication" in category or "drug" in category:
        return True

    # Include medication dose/route/frequency claims that occur
    # in categories such as "Hospital Course".
    medication_relations = {
        "dose",
        "dosage",
        "route",
        "frequency",
        "dosing frequency",
        "administration frequency",
    }

    return relation in medication_relations


def main():
    records = json.loads(
        GROUND_TRUTH.read_text(encoding="utf-8")
    )

    record = next(
        item
        for item in records
        if item["case_metadata"]["case_id"] == "SYN-DS-0006"
    )

    facts = record["ground_truth_facts"]

    for claims_path in CLAIM_FILES:
        if not claims_path.exists():
            print(f"\nMISSING FILE: {claims_path}")
            continue

        payload = json.loads(
            claims_path.read_text(encoding="utf-8")
        )
        claims = payload["claims"]

        medication_claims = [
            claim
            for claim in claims
            if is_medication_claim(claim)
        ]

        print(f"\n{'=' * 70}")
        print(f"FILE: {claims_path.name}")
        print(f"TOTAL CLAIMS: {len(claims)}")
        print(f"MEDICATION CLAIMS: {len(medication_claims)}")

        for claim in medication_claims:
            candidates = retrieve_candidate_facts(
                claim,
                facts,
                top_k=20,
                record=record,
            )

            result = verify_claim_against_facts(
                claim,
                candidates,
                record=record,
            )

            print("\n" + "-" * 60)
            print("CLAIM ID:", claim.get("claim_id"))
            print("SUBJECT:", claim.get("subject"))
            print("CATEGORY:", claim.get("category"))
            print("RELATION:", claim.get("relation"))
            print("VALUE:", claim.get("value"), claim.get("unit", ""))
            print("FREQUENCY:", claim.get("frequency"))
            print("SOURCE TEXT:", claim.get("source_text"))
            print("VERDICT:", result.get("verdict"))
            print("ERROR TYPE:", result.get("error_type"))
            print("CANDIDATES:", len(candidates))
            print("FIELD RESULTS:")
            print(
                json.dumps(
                    result.get("field_results", {}),
                    indent=2,
                    ensure_ascii=False,
                )
            )
            print("MATCHED FACT IDS:", result.get("matched_fact_ids"))

            print("TOP 5 CANDIDATES:")
            for candidate in candidates[:5]:
                fact = candidate["fact"]
                print(
                    f"  score={candidate['score']} "
                    f"id={fact.get('fact_id')} "
                    f"subject={fact.get('subject')} "
                    f"relation={fact.get('relation')} "
                    f"value={fact.get('value')} "
                    f"section={fact.get('source_section')}"
                )


if __name__ == "__main__":
    main()
