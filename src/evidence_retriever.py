
import re
from datetime import date
import argparse
import json
from pathlib import Path

def normalize_text(value):
    """Normalize text for basic matching."""

    if value is None:
        return ""

    value = str(value).lower()
    value = value.replace("–", "-").replace("—", "-")
    return " ".join(value.split()).strip()


def tokenize(value):
    """Extract simple word tokens."""

    return set(re.findall(r"[a-z0-9]+", normalize_text(value)))


def category_group(value):
    """Map common claim and fact categories to broad groups."""

    category = normalize_text(value)

    groups = {
        "medication": {
            "medication",
            "discharge medication",
            "medication order",
        },
        "laboratory": {
            "laboratory",
            "laboratory result",
            "lab",
            "investigation",
        },
        "diagnosis": {
            "diagnosis",
            "condition",
            "clinical diagnosis",
        },
        "allergy": {
            "allergy",
            "allergies",
        },
        "procedure": {
            "procedure",
            "intervention",
        },
    }

    for group, names in groups.items():
        if category in names or any(
            name in category for name in names
        ):
            return group

    return category


def retrieve_candidates(claim, facts, top_k=10):
    """
    Rank possible source facts for a claim.

    IMPORTANT:
    This function does not receive an error ID or target fact ID.
    It deliberately does not use claim values to retrieve evidence.
    """

    claim_subject = tokenize(claim.get("subject", ""))
    claim_category = category_group(claim.get("category", ""))
    claim_date = normalize_text(claim.get("date_time", ""))

    if not claim_subject:
        return []

    candidates = []

    for fact in facts:
        fact_subject = tokenize(fact.get("subject", ""))

        # Require at least one shared subject term.
        shared_terms = claim_subject & fact_subject

        if not shared_terms:
            continue

        # Subject similarity is the main retrieval signal.
        subject_score = len(shared_terms) / len(claim_subject)

        score = subject_score

        # Category is a supporting signal, not a strict requirement.
        fact_category = category_group(fact.get("category", ""))

        if claim_category == fact_category:
            score += 0.25

        # Use an exact date match only when the claim provides a date.
        fact_date = normalize_text(fact.get("date_time", ""))

        if claim_date and claim_date == fact_date:
            score += 0.50

        candidates.append({
            "score": round(score, 4),
            "shared_subject_terms": sorted(shared_terms),
            "fact": fact,
        })

    # Stable sorting makes ties reproducible.
    candidates.sort(
        key=lambda item: (
            -item["score"],
            item["fact"].get("fact_id", ""),
        )
    )

    return candidates[:top_k]


def main():
    parser = argparse.ArgumentParser(
        description="Inspect retrieved evidence candidates."
    )
    parser.add_argument(
        "claims_file",
        help="Path to a saved claims JSON file.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Only inspect claims matching this subject.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    claims_path = project_root / args.claims_file

    if not claims_path.is_file():
        raise FileNotFoundError(
            f"Claims file not found: {claims_path}"
        )

    saved_data = json.loads(
        claims_path.read_text(encoding="utf-8")
    )

    summary_path = Path(saved_data["summary_file"])
    case_id = summary_path.stem.split("_E")[0]

    ground_truth_path = (
        project_root
        / "data"
        / "patient_records"
        / "synthetic_discharge_ground_truth.json"
    )

    records = json.loads(
        ground_truth_path.read_text(encoding="utf-8")
    )

    record = next(
        (
            item for item in records
            if item["case_metadata"]["case_id"] == case_id
        ),
        None,
    )

    if record is None:
        raise ValueError(f"Patient record not found: {case_id}")

    facts = record.get("ground_truth_facts", [])
    claims = saved_data["claims"]

    if args.subject:
        requested_subject = normalize_text(args.subject)
        claims = [
            claim for claim in claims
            if requested_subject in normalize_text(
                claim.get("subject", "")
            )
        ]

    print(f"Patient: {case_id}")
    print(f"Ground-truth facts searched: {len(facts)}")
    print(f"Claims inspected: {len(claims)}")

    for claim in claims:
        candidates = retrieve_candidates(claim, facts)

        print("\n" + "=" * 60)
        print(
            f"Claim {claim.get('claim_id')}: "
            f"{claim.get('subject')} | "
            f"{claim.get('relation')} | "
            f"{claim.get('value')} {claim.get('unit', '')}"
        )

        if not candidates:
            print("No candidates retrieved.")
            continue

        for candidate in candidates:
            fact = candidate["fact"]
            print(
                f"Score: {candidate['score']:.4f} | "
                f"{fact.get('fact_id')} | "
                f"{fact.get('subject')} | "
                f"{fact.get('relation')} | "
                f"{fact.get('value')} {fact.get('unit', '')}"
            )


if __name__ == "__main__":
    main()
