import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from openai import OpenAI


# Load variables from .env
load_dotenv()

# OpenRouter configuration
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODEL = os.getenv("OPENROUTER_MODEL")

GROUND_TRUTH_PATH = Path(
    "data/patient_records/synthetic_discharge_ground_truth.json"
)

OUTPUT_DIR = Path(
    "data/discharge_summaries"
)


def load_prompt():
    prompt_path = (
        Path(__file__).parent.parent
        / "prompts"
        / "discharge_summary_generation.txt"
    )

    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read()


def load_records():
    with open(
        GROUND_TRUTH_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "Ground-truth JSON must contain a list of records."
        )

    return records


def generate_summary(record):
    prompt = load_prompt()

    # Ground-truth facts are verification metadata.
    # They are not needed for generating the discharge summary.
    generation_record = {
        key: value
        for key, value in record.items()
        if key != "ground_truth_facts"
    }

    record_json = json.dumps(
        generation_record,
        indent=2,
        ensure_ascii=False
    )

    prompt = prompt.replace(
        "[INSERT PATIENT RECORD HERE]",
        record_json
    )

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

        # Handle temporary provider failures
        if not response.choices:

            error = getattr(response, "error", None)

            print(
                f"Attempt {attempt}/{max_retries} failed: "
                f"{error}"
            )

            if attempt < max_retries:
                print("Retrying in 5 seconds...\n")
                time.sleep(5)
                continue

            raise ValueError(
                "The model failed after all retry attempts."
            )

        summary = response.choices[0].message.content

        if not summary:
            raise ValueError(
                "The model returned an empty discharge summary."
            )

        return summary.strip()


def save_summary(case_id, summary):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / f"{case_id}.txt"

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(summary)

    return output_path


def main():
    records = load_records()

    print(
        f"Loaded {len(records)} patient records.\n"
    )

    for record in records:

        case_id = record[
            "case_metadata"
        ]["case_id"]

        output_path = OUTPUT_DIR / f"{case_id}.txt"

        if output_path.exists():
            print(
                f"Already exists: {output_path} — skipping.\n"
            )
            continue

        print(
            f"Generating summary for {case_id}..."
        )

        summary = generate_summary(record)

        output_path = save_summary(
            case_id,
            summary
        )

        print(
            f"Saved: {output_path}\n"
        )


if __name__ == "__main__":
    main()