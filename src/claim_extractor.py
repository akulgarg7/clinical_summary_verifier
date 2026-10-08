import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# Load variables from .env
load_dotenv()

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY"),
    timeout=120.0
)

MODEL = os.getenv("GEMINI_MODEL")


def normalize_text(text):
    text = text.lower()
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("*", "")
    text = " ".join(text.split())
    return text



def load_prompt():
    prompt_path = Path(__file__).parent.parent / "prompts" / "claim_extraction.txt"

    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read()


def validate_claims(claims, discharge_summary):
    required_fields = {
        "claim_id",
        "category",
        "subject",
        "relation",
        "value",
        "unit",
        "route",
        "frequency",
        "duration",
        "date_time",
        "negated",
        "source_text"
    }

    optional_defaults = {
        "unit": "",
        "route": "",
        "frequency": "",
        "duration": "",
        "date_time": "",
        "negated": False,
    }

    normalized_summary = normalize_text(discharge_summary)

    for claim in claims:

        # --------------------------------------------
        # Add missing optional fields
        # --------------------------------------------

        for field, default_value in optional_defaults.items():

            if field not in claim:
                claim[field] = default_value

        # --------------------------------------------
        # Check truly required fields
        # --------------------------------------------

        missing_fields = required_fields - claim.keys()

        if missing_fields:
            raise ValueError(
                f"Claim {claim.get('claim_id', 'UNKNOWN')} "
                f"is missing required fields: {missing_fields}"
            )

        # --------------------------------------------
        # Validate source_text
        # --------------------------------------------

        source_text = claim["source_text"]

        normalized_source = normalize_text(source_text)

        print(
            "SOURCE TEXT FROM MODEL:",
            repr(source_text),
            "IN SUMMARY:",
            normalized_source in normalized_summary
        )

        if normalized_source not in normalized_summary:
            raise ValueError(
                f"Claim {claim['claim_id']} contains source_text "
                f"that does not match the discharge summary:\n"
                f"{source_text}"
            )

    return claims


def extract_claims(discharge_summary):
    prompt = load_prompt()

    prompt = prompt.replace(
        "[INSERT DISCHARGE SUMMARY HERE]",
        discharge_summary
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )


    result = response.choices[0].message.content

    if not result:
        raise ValueError(
            "Model returned no message content. "
            "Check the FULL MODEL RESPONSE above."
        )

    data = json.loads(result)

    claims = data.get("claims", [])

    validate_claims(claims, discharge_summary)

    return claims