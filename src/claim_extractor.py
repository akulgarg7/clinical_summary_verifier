import json
import os
from pathlib import Path

from dotenv import load_dotenv


def _get_gemini_client_and_model():
    """Create the API client only when extraction is actually requested.

    The app can therefore open and reuse saved claims without requiring an API
    key at import time. Missing or placeholder configuration produces a clear
    message only when a Gemini-backed operation is invoked.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip()
    if not api_key or api_key == "gemini_api_key":
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Copy .env.example to .env and "
            "add your Gemini API key before extracting new claims."
        )
    if not model or model == "model_name":
        raise RuntimeError(
            "GEMINI_MODEL is not configured. Set it to the Gemini model you use."
        )
    # Imported lazily as well, so the app can read persisted data even before
    # runtime API dependencies/configuration are installed.
    from openai import OpenAI

    return OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
        timeout=120.0,
    ), model


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
        "time_of_day",
        "duration",
        "date_time",
        "negated",
        "source_text"
    }

    optional_defaults = {
        "unit": "",
        "route": "",
        "frequency": "",
        "time_of_day": "",
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

        if normalized_source not in normalized_summary:
            raise ValueError(
                f"Claim {claim['claim_id']} contains source_text "
                f"that does not match the discharge summary:\n"
                f"{source_text}"
            )

    return claims


def extract_claims(discharge_summary, max_attempts=2):
    client, model = _get_gemini_client_and_model()
    base_prompt = load_prompt()

    base_prompt = base_prompt.replace(
        "[INSERT DISCHARGE SUMMARY HERE]",
        discharge_summary
    )

    last_error = None

    for attempt in range(1, max_attempts + 1):
        prompt = base_prompt

        # If validation failed previously, ask the model to correct
        # its extraction while preserving strict source validation.
        if last_error is not None:
            prompt += (
                "\n\nIMPORTANT CORRECTION:\n"
                f"The previous response failed validation: {last_error}\n"
                "Generate a corrected, complete JSON response. "
                "Every source_text must be copied verbatim from the "
                "discharge summary, including numbers, units, dates, "
                "and wording. Do not correct or paraphrase the source "
                "text. Check every source_text against the summary "
                "before returning the JSON."
            )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        result = response.choices[0].message.content

        try:
            if not result:
                raise ValueError(
                    "Model returned no message content."
                )

            data = json.loads(result)
            claims = data.get("claims", [])

            # Keep strict validation. Invalid claims must not pass.
            validate_claims(claims, discharge_summary)

            return claims

        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc

            if attempt < max_attempts:
                print(
                    f"Claim extraction validation failed "
                    f"on attempt {attempt}/{max_attempts}. "
                    "Retrying extraction..."
                )

    raise ValueError(
        f"Claim extraction failed after {max_attempts} attempts. "
        f"Last error: {last_error}"
    ) from last_error