import json
from pathlib import Path


GROUND_TRUTH_PATH = Path(
    "data\patient_records\synthetic_discharge_ground_truth.json"
)


def load_ground_truth(file_path=GROUND_TRUTH_PATH):
    """Load the synthetic patient records from the JSON file."""

    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("Ground-truth JSON must contain a list of records.")

    return records


def build_evidence(records):
    """
    Convert ground-truth records into a case-based evidence structure.

    Returns:
        {
            "CASE_ID": {
                "facts": [...],
                "fact_index": {
                    "FACT_ID": {...}
                }
            }
        }
    """

    evidence = {}

    for record in records:
        case_id = record["case_metadata"]["case_id"]
        facts = record.get("ground_truth_facts", [])

        evidence[case_id] = {
            "facts": facts,
            "fact_index": {
                fact["fact_id"]: fact
                for fact in facts
                if "fact_id" in fact
            }
        }

    return evidence


def main():
    records = load_ground_truth()
    evidence = build_evidence(records)

    print(f"Loaded {len(records)} patient records.\n")

    for case_id, case_data in evidence.items():
        print(
            f"{case_id}: "
            f"{len(case_data['facts'])} ground-truth facts"
        )


if __name__ == "__main__":
    main()