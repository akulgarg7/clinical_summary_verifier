import json
from pathlib import Path
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

ERROR_INJECTION_TYPES = [
    "medication_dose",
    "medication_route",
    "medication_frequency",
    "medication_status",
    "diagnosis",
    "allergy",
    "laboratory",
    "imaging",
    "procedure",
    "temporal",
    "treatment_response",
    "severity",
    "follow_up",
    "unsupported_fact",
    "omission",
    "unit",
]

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODEL = os.getenv("OPENROUTER_MODEL")

SUMMARY_DIR = Path(
    "data/discharge_summaries"
)

OUTPUT_DIR = Path(
    "data/error_injected"
)

MANIFEST_PATH = Path(
    "data/error_injected/error_injection_manifest.json"
)


def load_summary(case_id):
    """Load the original discharge summary for a case."""

    summary_path = SUMMARY_DIR / f"{case_id}.txt"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Discharge summary not found: {summary_path}"
        )

    with open(summary_path, "r", encoding="utf-8") as f:
        return f.read()


def save_injected_summary(case_id, error_id, summary):
    """Save an error-injected discharge summary."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / (
        f"{case_id}_{error_id}.txt"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)

    return output_path


def save_manifest(new_entries):
    """Append new error entries to the existing manifest."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    existing_entries = []

    if MANIFEST_PATH.exists():
        with open(
            MANIFEST_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            existing_entries = json.load(f)

    existing_ids = {
        entry["error_id"]
        for entry in existing_entries
    }

    for entry in new_entries:
        if entry["error_id"] not in existing_ids:
            existing_entries.append(entry)

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            existing_entries,
            f,
            indent=2,
            ensure_ascii=False
        )


def inject_error(
    case_id,
    original_summary,
    error_type,
    target_fact,
    incorrect_value
):
    """
    Ask the LLM to modify only the part of the discharge summary
    corresponding to the specified ground-truth fact.
    """

    prompt = f"""
You are modifying a synthetic medical discharge summary for a
controlled research experiment.

Your task is to introduce EXACTLY ONE factual error.

ERROR TYPE:
{error_type}

TARGET FACT:
{json.dumps(target_fact, indent=2, ensure_ascii=False)}

INCORRECT VALUE TO USE:
{incorrect_value}

RULES:
- Locate the statement in the summary corresponding to the target fact.
- Change only that factual information.
- Preserve the rest of the summary.
- Keep the writing natural and clinically realistic.
- Do not introduce any other errors.
- Do not add explanations or comments.
- Return ONLY the modified discharge summary.

ORIGINAL DISCHARGE SUMMARY:
{original_summary}
"""

    max_retries = 3

    for attempt in range(1, max_retries + 1):

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        if response.choices:
            break

        error = getattr(response, "error", None)

        print(
            f"Attempt {attempt}/{max_retries} failed: "
            f"{error}"
        )

        if attempt < max_retries:
            print("Retrying in 5 seconds...\n")
            time.sleep(5)

    else:
        raise ValueError(
            "Error injection model failed after all retry attempts."
        )

    if not response.choices:
        error = getattr(response, "error", None)
        raise ValueError(
            f"Error injection model failed: {error}"
        )

    injected_summary = response.choices[0].message.content

    if not injected_summary:
        raise ValueError(
            "The model returned an empty modified summary."
        )

    return injected_summary.strip()

ERROR_SPECIFICATIONS = [

    {
        "error_id": "E001",
        "case_id": "SYN-DS-0006",
        "error_type": "medication_dose",
        "fact_id": "SYN-DS-0006-F139",
        "incorrect_value": "20 mg",
        "expected_verdict": "CONTRADICTED",
        "expected_severity": "HIGH"
    },

    {
        "error_id": "E002",
        "case_id": "SYN-DS-0006",
        "error_type": "medication_frequency",
        "fact_id": "SYN-DS-0006-F141",
        "incorrect_value": "twice daily",
        "expected_verdict": "CONTRADICTED",
        "expected_severity": "HIGH"
    },

    {
    "error_id": "E003",
    "case_id": "SYN-DS-0003",
    "error_type": "laboratory",
    "fact_id": "SYN-DS-0003-F055",
    "incorrect_value": "2.9 mg/dL",
    "expected_verdict": "CONTRADICTED",
    "expected_severity": "HIGH"
    }

]

def find_fact(records, case_id, fact_id):
    """Find a specific ground-truth fact."""

    for record in records:

        record_case_id = record["case_metadata"]["case_id"]

        if record_case_id != case_id:
            continue

        for fact in record.get("ground_truth_facts", []):

            if fact.get("fact_id") == fact_id:
                return fact

    raise ValueError(
        f"Fact {fact_id} not found for {case_id}"
    )


GROUND_TRUTH_PATH = Path(
    "data/patient_records/synthetic_discharge_ground_truth.json"
)


def load_ground_truth():
    """Load synthetic ground-truth patient records."""

    with open(
        GROUND_TRUTH_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(
            "Ground-truth JSON must contain a list of records."
        )

    return records




if __name__ == "__main__":

    records = load_ground_truth()

    manifest = []

    for specification in ERROR_SPECIFICATIONS:

        case_id = specification["case_id"]
        error_id = specification["error_id"]

        output_path = OUTPUT_DIR / (
            f"{case_id}_{error_id}.txt"
        )

        print(
            f"Processing {error_id} "
            f"({specification['error_type']}) "
            f"for {case_id}..."
        )

        # Find the ground-truth fact
        target_fact = find_fact(
            records,
            case_id,
            specification["fact_id"]
        )

        # ------------------------------------------------
        # If the injected summary already exists, do not
        # call the LLM again.
        # ------------------------------------------------

        if output_path.exists():

            print(
                f"Already exists: {output_path} "
                f"— skipping generation."
            )

        else:

            print("Generating error-injected summary...")

            original_summary = load_summary(case_id)

            injected_summary = inject_error(
                case_id=case_id,
                original_summary=original_summary,
                error_type=specification["error_type"],
                target_fact=target_fact,
                incorrect_value=specification["incorrect_value"]
            )

            output_path = save_injected_summary(
                case_id,
                error_id,
                injected_summary
            )

            print(
                f"Saved: {output_path}"
            )

        # ------------------------------------------------
        # Always create/preserve the manifest entry
        # ------------------------------------------------

        manifest_entry = {
            "error_id": error_id,
            "case_id": case_id,
            "error_type": specification["error_type"],
            "fact_id": specification["fact_id"],
            "original_value": target_fact.get("value"),
            "incorrect_value": specification["incorrect_value"],
            "expected_verdict": specification["expected_verdict"],
            "expected_severity": specification["expected_severity"],
            "output_file": str(output_path)
        }

        manifest.append(manifest_entry)

        print()

    save_manifest(manifest)

    print(
        f"Error injection processing complete."
    )
    print(
        f"Manifest: {MANIFEST_PATH}"
    )