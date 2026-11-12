import re
from datetime import datetime
import json
from pathlib import Path


def normalize(value):
    """Normalize values for basic, case-insensitive comparison."""
    if value is None:
        return ""
    return " ".join(str(value).lower().split()).strip()

def normalize_route(value):
    """Normalize common equivalent medication routes."""
    value = normalize(value)

    equivalents = {
        "sc": "subcutaneous",
        "s.c.": "subcutaneous",
        "subcut": "subcutaneous",
        "iv": "intravenous",
        "i.v.": "intravenous",
        "po": "oral",
        "p.o.": "oral",
        "by mouth": "oral",
        "im": "intramuscular",
        "i.m.": "intramuscular",
    }

    return equivalents.get(value, value)


def find_source_medication(claim, record):
    """Select the medication source section using claim context."""
    subject = normalize(claim.get("subject", ""))
    if not subject:
        return None, None

    category = normalize(claim.get("category", "")).replace("_", " ")
    relation = normalize(claim.get("relation", "")).replace("_", " ")
    source_text = normalize(claim.get("source_text", ""))

    # Explicitly discontinued or withheld medication claims.
    is_discontinued_claim = (
        "discontinued medication" in category
        or "discontinued medication" in relation
        or "medication withheld" in category
    )

    # Medication taken before admission or a home regimen.
    is_pre_admission_claim = (
        "pre admission" in category
        or "before admission" in category
        or "prior to admission" in relation
        or "pre admission" in relation
        or "home medication" in category
        or "home regimen" in source_text
    )

    # Explicit discharge prescription claims.
    is_discharge_claim = (
        "discharge" in category
        or "discharge" in relation
        or "at discharge" in source_text
        or "discharged on" in source_text
    )

    if is_discontinued_claim:
        section = "discontinued_medications"
    elif is_pre_admission_claim:
        section = "medications_before_admission"
    elif is_discharge_claim:
        section = "discharge_medications"
    else:
        section = "medications_during_hospitalization"

    medications = record.get(section, [])

    matches = [
        medication
        for medication in medications
        if normalize(medication.get("name", "")) == subject
    ]

    # Never choose an arbitrary entry when multiple matches exist.
    if len(matches) != 1:
        return None, section

    return matches[0], section





def normalize_frequency(value):
    """Normalize equivalent dosing-frequency descriptions."""
    value = normalize(value)

    equivalents = {
    "daily": "once daily",
    "once a day": "once daily",
    "once per day": "once daily",
    "every day": "once daily",
    "nightly": "once daily",
    "every night": "once daily",
    "at bedtime": "once daily",
    "twice a day": "twice daily",
    "two times daily": "twice daily",
    "two times a day": "twice daily",
    "bid": "twice daily",
    "three times a day": "three times daily",
    "tid": "three times daily",
    "qid": "four times daily",
}

    return equivalents.get(value, value)

def classify_time_of_day(value):
    """Classify medication timing into a broad time-of-day category."""
    value = normalize(value)

    if not value:
        return "UNSPECIFIED"

    # Explicit time-of-day wording
    periods = {
        "MORNING": (
            "morning", "in the morning", "every morning"
        ),
        "AFTERNOON": (
            "afternoon", "in the afternoon"
        ),
        "EVENING": (
            "evening", "in the evening"
        ),
        "NIGHT": (
            "night", "nightly", "at night", "every night",
            "bedtime", "at bedtime"
        ),
    }

    for period, phrases in periods.items():
        if any(phrase in value for phrase in phrases):
            return period

    return "UNSPECIFIED"


def parse_time(value):
    """Extract a numeric time from common medication instructions."""
    value = normalize(value)

    patterns = [
        r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b",
        r"\b(\d{1,2})\s*(am|pm)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, value)

        if not match:
            continue

        if len(match.groups()) == 3:
            hour = int(match.group(1))
            minute = int(match.group(2))
            meridiem = match.group(3)
        else:
            hour = int(match.group(1))
            minute = 0
            meridiem = match.group(2)

        if minute > 59:
            return None

        if meridiem:
            if hour < 1 or hour > 12:
                return None

            if meridiem == "am":
                hour = 0 if hour == 12 else hour
            else:
                hour = 12 if hour == 12 else hour + 12
        elif hour > 23:
            return None

        if hour > 23:
            return None

        return hour * 60 + minute

    return None




def extract_frequency(value):
    """Extract dosing frequency separately from administration timing."""
    original = normalize(value)

    # A standalone nightly instruction means once daily.
    if original in {"nightly", "every night", "at bedtime"}:
        return "once daily"

    value = original

    # Remove numeric times, such as 21:00, 9 PM, or 9:30 AM.
    value = re.sub(
        r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b",
        " ",
        value,
    )
    value = re.sub(
        r"\b\d{1,2}\s*(?:am|pm)\b",
        " ",
        value,
    )

    # Remove time-of-day wording and connecting words.
    value = re.sub(
        r"\b(in the morning|in the afternoon|in the evening|"
        r"at night|at bedtime|every morning|every night|"
        r"morning|afternoon|evening|nightly|night|bedtime|at)\b",
        " ",
        value,
    )

    value = normalize(value)

    equivalents = {
        "once": "once daily",
        "daily": "once daily",
        "once daily": "once daily",
        "once a day": "once daily",
        "once per day": "once daily",
        "every day": "once daily",
        "twice": "twice daily",
        "twice daily": "twice daily",
        "twice a day": "twice daily",
        "two times daily": "twice daily",
        "two times a day": "twice daily",
        "bid": "twice daily",
        "three times daily": "three times daily",
        "three times a day": "three times daily",
        "tid": "three times daily",
        "qid": "four times daily",
    }

    return equivalents.get(value, normalize_frequency(value))



def compare_timing(claim_value, source_value):
    """Compare medication frequency and time-of-day meaning."""
    claim_text = normalize(claim_value)
    source_text = normalize(source_value)

    if not claim_text or not source_text:
        return "INSUFFICIENT_EVIDENCE"

    claim_frequency = extract_frequency(claim_text)
    source_frequency = extract_frequency(source_text)

    if claim_frequency != source_frequency:
        return "CONTRADICTED"

    claim_period = classify_time_of_day(claim_text)
    source_period = classify_time_of_day(source_text)


    # Only contradict when both specify a time period and
    # those periods differ.
    if (
        claim_period != "UNSPECIFIED"
        and source_period != "UNSPECIFIED"
        and claim_period != source_period
    ):
        return "CONTRADICTED"

    return "SUPPORTED"



def compare_field(claim_value, source_value, field=""):
    """Compare one claim field with its source value."""
    if claim_value is None or str(claim_value).strip() == "":
        return "INSUFFICIENT_EVIDENCE"

    if source_value is None or str(source_value).strip() == "":
        return "INSUFFICIENT_EVIDENCE"

    if field == "frequency":
        claim_normalized = normalize_frequency(claim_value)
        source_normalized = normalize_frequency(source_value)
    elif field == "route":
        claim_normalized = normalize_route(claim_value)
        source_normalized = normalize_route(source_value)
    else:
        claim_normalized = normalize(claim_value)
        source_normalized = normalize(source_value)

    if claim_normalized == source_normalized:
        return "SUPPORTED"

    return "CONTRADICTED"


def verify_medication_claim(claim, record):
    """
    Verify the core medication fields against the discharge list.

    Initial scope: medicine, dose, route, frequency, and status.
    """
    medication, source_section = find_source_medication(claim, record)

    if medication is None:
        return {
            "claim_id": claim.get("claim_id"),
            "source_section": source_section,
            "verdict": "INSUFFICIENT_EVIDENCE",
            "reason": (
                "Could not identify exactly one matching "
                "discharge medication."
            ),
            "field_results": {},
        }

    # The claim extractor may represent the medicine in its
    # subject field, while other attributes may appear in relation
    # or separate fields. Do not infer missing attributes.
    
    claim_value = claim.get("value")
    claim_unit = claim.get("unit")

    # Only interpret a value as a dose when its unit indicates
    # a measurable quantity. For now, support common mass units.
    dose_units = {
        "mg", "g", "mcg", "μg", "µg",
        "unit", "units", "iu",
        "international unit", "international units",
    }

    is_dose = (
        claim_value is not None
        and normalize(claim_unit) in dose_units
    )

    claim_dose = (
        f"{claim_value} {claim_unit}".strip()
        if is_dose
        else None
    )

    claim_fields = {
        "medicine": claim.get("subject"),
        "dose": claim_dose,
        "route": claim.get("route"),
        "frequency": claim.get("frequency"),
        "time_of_day": claim.get("time_of_day"),
    }


    source_fields = {
        "medicine": medication.get("name"),
        "dose": (
            f"{medication.get('dose', '')} "
            f"{medication.get('unit', '')}"
        ).strip(),
        "route": medication.get("route"),
        "frequency": medication.get("frequency"),
        "time_of_day": classify_time_of_day(
            medication.get("frequency", "")
        ),
    }


    field_results = {}

    for field, claim_value in claim_fields.items():
        source_value = source_fields[field]

        if field == "frequency":
            verdict = compare_timing(
                f"{claim_value or ''} "
                f"{claim.get('time_of_day') or ''}",
                source_value,
            )

        elif field == "time_of_day":
            # This field is optional. Missing information should
            # not downgrade an otherwise supported medication claim.
            if not claim_value or not str(claim_value).strip():
                verdict = "SUPPORTED"
            else:
                claim_period = classify_time_of_day(claim_value)

                if claim_period == "UNSPECIFIED":
                    verdict = "INSUFFICIENT_EVIDENCE"
                elif source_value == "UNSPECIFIED":
                    verdict = "INSUFFICIENT_EVIDENCE"
                elif claim_period == source_value:
                    verdict = "SUPPORTED"
                else:
                    verdict = "CONTRADICTED"

        else:
            verdict = compare_field(
                claim_value,
                source_value,
                field=field,
            )

        field_results[field] = {
            "claim_value": claim_value,
            "source_value": source_value,
            "verdict": verdict,
        }


    
    assessed = [
        result["verdict"]
        for result in field_results.values()
        if result["verdict"] != "INSUFFICIENT_EVIDENCE"
    ]

    all_verdicts = [
        result["verdict"]
        for result in field_results.values()
    ]

    if "CONTRADICTED" in assessed:
        overall = "CONTRADICTED"
    elif not assessed:
        overall = "INSUFFICIENT_EVIDENCE"
    elif "INSUFFICIENT_EVIDENCE" in all_verdicts:
        overall = "PARTIALLY_SUPPORTED"
    else:
        overall = "SUPPORTED"


    return {
        "claim_id": claim.get("claim_id"),
        "source_section": source_section,
        "verdict": overall,
        "source_medication": medication.get("name"),
        "field_results": field_results,
    }


def load_record(case_id, project_root):
    path = (
        project_root
        / "data"
        / "patient_records"
        / "synthetic_discharge_ground_truth.json"
    )

    records = json.loads(path.read_text(encoding="utf-8"))

    for record in records:
        if record.get("case_metadata", {}).get("case_id") == case_id:
            return record

    raise ValueError(f"Patient record not found: {case_id}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test discharge medication field verification."
    )
    parser.add_argument("claims_file")
    parser.add_argument("--subject", default="Rivaroxaban")
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
    record = load_record(case_id, project_root)

    requested_subject = normalize(args.subject)
    claims = [
        claim for claim in saved_data["claims"]
        if requested_subject in normalize(claim.get("subject", ""))
    ]

    print(f"Patient: {case_id}")
    print(f"Claims inspected: {len(claims)}")

    for claim in claims:
        result = verify_medication_claim(claim, record)
        print("\n" + "=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
