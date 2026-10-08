def normalize(value):
    """Simple normalization for comparison."""

    if value is None:
        return ""

    value = str(value).lower()
    value = value.replace("–", "-")
    value = value.replace("—", "-")
    value = value.replace("×", "x")

    return " ".join(value.split()).strip()


def values_match(claim_value, fact_value):
    """Compare two values."""

    claim = normalize(claim_value)
    fact = normalize(fact_value)

    if claim == fact:
        return True

    try:
        return float(claim) == float(fact)
    except (ValueError, TypeError):
        return False


def verify_claim(claim, fact):
    """
    Compare one extracted claim directly
    against one ground-truth fact.
    """

    claim_value = claim.get("value", "")
    fact_value = fact.get("value", "")

    if values_match(claim_value, fact_value):

        return {
            "claim_id": claim.get("claim_id"),
            "verdict": "SUPPORTED",
            "error_type": None,
            "confidence": 0.95,
            "severity": "LOW",
            "matched_fact_id": fact.get("fact_id")
        }

    return {
        "claim_id": claim.get("claim_id"),
        "verdict": "CONTRADICTED",
        "error_type": "value_mismatch",
        "confidence": 0.95,
        "severity": fact.get(
            "severity_if_incorrect",
            "MEDIUM"
        ),
        "matched_fact_id": fact.get("fact_id")
    }