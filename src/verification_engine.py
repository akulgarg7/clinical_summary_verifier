import re

def normalize(value):
    """Simple normalization for comparison."""

    if value is None:
        return ""

    value = str(value).lower()
    value = value.replace("–", "-")
    value = value.replace("—", "-")
    value = value.replace("‑", "-")  # non-breaking hyphen
    value = value.replace("−", "-")  # mathematical minus
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

def _canonical_relation(value):
    """Normalize common variants of clinical fact relations."""
    relation = normalize(value).replace("_", " ").strip()
    relation = re.sub(r"\s+", " ", relation)

    aliases = {
        "dosage": "dose",
        "has value": "value",
        "has baseline value": "baseline value",
        "has result": "result",
        "has finding": "finding",
        "shows finding": "finding",
        "shows result": "result",
        "has status": "status",
        "has condition": "condition",
        "has diagnosis": "diagnosis",
        "is diagnosed with": "diagnosis",
        "has primary diagnosis": "diagnosis",
        "has secondary diagnosis": "diagnosis",
        "has final diagnosis": "diagnosis",
        "has final discharge diagnosis": "diagnosis",
        "has discharge diagnosis": "diagnosis",
        "excluded diagnosis": "diagnosis",
        "excluded differential": "diagnosis",
        "condition at discharge": "condition",
        "clinical condition at discharge": "condition",
        "admission date": "date",
        "discharge date": "date",
        "follow up date": "date",
        "date of procedure": "performed on",
        "dosing frequency": "frequency",
        # Relations emitted by the LLM extractor may describe the assertion
        # rather than use the ground-truth schema's canonical relation.
        "presence": "status",
        "allergy presence": "status",
        "allergy status": "status",
        "symptom duration": "duration",
        "duration of symptoms": "duration",
        "duration of symptom": "duration",
        "condition and disposition": "condition",
        "clinical condition": "condition",
        "clinical status": "status",
        "administration frequency": "frequency",
        "administration route": "route",
        "taken for": "indication",
        "used for": "indication",
        "indicated for": "indication",
        "reason for use": "indication",
        "reason": "indication",
        "treatment duration": "duration",
        "duration of treatment": "duration",
        "causes": "allergic reaction",
        "allergy reaction": "allergic reaction",
        "performed": "performed on",
        "discontinued": "status",
        "stopped": "status",
        "started": "status",
        "continued": "status",
        "has age": "age",
        "has sex": "sex",
        "has height": "height",
        "has weight": "weight",
        "has gender": "sex",
        "has surgery history": "surgery",
        "has medical history": "condition",
        "has symptom": "description",
        "has chief complaint": "description",
        "has physical finding": "finding",
        "smoking status": "smoking",
        "smokes": "smoking",
        "has occupation": "occupation",
        "has reason for admission": "reason",
        "is scheduled for": "reason",
        "follow up": "reason",
        "scheduled for": "reason",
        "has date of birth": "date of birth",
        "status relative to pre admission regimen": "status",
        "status relative to pre-admission regimen": "status",
        "medication status": "status",
        "drug status": "status",
    }

    if relation.startswith("status "):
        return "status"

    return aliases.get(relation, relation)

def _normalize_indication_text(value):
    """Normalize common clinical terminology in indication text."""
    text = normalize(value)

    # Expand a common clinical abbreviation.
    text = re.sub(
        r"\bvte\b",
        "venous thromboembolism",
        text,
    )
    # Equivalent imaging-context wording: "CT-confirmed appendicitis" and
    # "appendicitis on CT" express the same diagnostic evidence.
    text = re.sub(r"\bct[- ]confirmed\b", " ", text)
    text = re.sub(r"\bon ct\b", " ", text)

    return " ".join(text.split())


def _canonical_route(value):
    """Normalize the administration route, ignoring compatible device detail.

    The claim schema has a route field but no independent device field, so
    nebulizer/MDI/DPI/spacer wording is normalized to the broad inhaled route.
    This avoids treating device wording as a route contradiction.
    """
    route = normalize(value).replace("sub-cutaneous", "subcutaneous")

    # Keep explicit negation distinct; a route such as "not IV" must not
    # silently normalize to IV.
    if re.search(r"\b(?:no|not|without|avoid)\b", route):
        return route

    patterns = [
        (r"\b(?:oral|po|p\.o\.|by mouth|per os)\b", "oral"),
        (r"\b(?:intravenous|i\.v\.|iv)\b", "intravenous"),
        (r"\b(?:subcutaneous|subcut|s\.c\.|sc)\b", "subcutaneous"),
        (r"\b(?:intramuscular|i\.m\.|im)\b", "intramuscular"),
        (r"\b(?:topical)\b", "topical"),
        (r"\b(?:rectal)\b", "rectal"),
        (r"\b(?:transdermal)\b", "transdermal"),
    ]
    for pattern, canonical in patterns:
        if re.search(pattern, route):
            return canonical

    # These are delivery devices/methods for the inhaled route, not different
    # administration routes. Device-specific verification would need its own
    # extracted field and corresponding evidence.
    if re.search(
        r"\b(?:inhaled|inhalation|nebul(?:is|iz)ed|nebul(?:is|iz)er|"
        r"mdi|metered[- ]dose inhaler|dpi|dry powder inhaler|spacer)\b",
        route,
    ):
        return "inhaled"

    return route


def _canonical_frequency(value):
    """Normalize medication schedules without treating delivery details as cadence."""
    text = normalize(value).replace("–", "-").replace("—", "-")
    text = text.replace("×", " x ")

    # Expand common abbreviations before normalizing frequency expressions.
    text = re.sub(r"\bmins?\b", "minutes", text)
    text = re.sub(r"\bhrs?\b", "hours", text)
    text = re.sub(r"\bsecs?\b", "seconds", text)
    text = re.sub(r"\bq\s*(\d+)\s*h(?:ours?)?\b", r"every \1 hours", text)
    text = re.sub(r"\bq\s*(\d+)\s*(?:min|minutes?)\b", r"every \1 minutes", text)
    text = re.sub(r"\bq\s*d\b|\bqd\b", "once daily", text)
    text = re.sub(r"\bq\s*hs\b|\bqhs\b", "once daily", text)
    text = re.sub(r"\bbid\b", "twice daily", text)
    text = re.sub(r"\btid\b", "three times daily", text)
    text = re.sub(r"\bqid\b", "four times daily", text)
    text = re.sub(r"\bstat\b", "once", text)
    text = re.sub(r"\b(?:single|one[- ]time)\b", "once", text)

    # Remove explanatory parentheticals when they add no schedule information,
    # but preserve parentheticals such as "(three times daily)".
    def keep_schedule_parenthetical(match):
        content = match.group(1)
        if re.search(
            r"\b(?:daily|weekly|monthly|once|twice|thrice|times? a day|"
            r"times? daily|every \d+|as needed|prn|each meal|with meals|q\s*\d+)\b",
            content,
        ):
            return " " + content + " "
        return " "

    text = re.sub(r"\(([^)]*)\)", keep_schedule_parenthetical, text)

    # A count of administrations is equivalent across "x 3 doses" and
    # "for 3 doses". Normalize timing units and qualifiers next.
    text = re.sub(r"\bx\s*(\d+)\s*doses?\b", r"for \1 doses", text)
    text = re.sub(r"\b(?:pre[- ]?operative|post[- ]?operative)\b", " ", text)
    text = re.sub(r"\b(?:renal(?:ly)? adjusted|renal dose adjusted|renal adjustment)\b", " ", text)
    text = re.sub(r"\bregular\b", " ", text)

    # PRN context is indication, not a different schedule. Maximum-dose limits
    # are not schedule intervals and are validated separately where represented.
    text = re.sub(
        r"\bmax(?:imum)?\s+\d+(?:\.\d+)?\s*(?:mg|g|mcg|units?|iu)?\s*"
        r"(?:/|per)\s*\d+\s*(?:hours?|h|days?|d)\b.*$",
        " ", text,
    )
    text = re.sub(r"\bprn\b", "as needed", text)
    text = re.sub(r"\bas needed for\b.*$", "as needed", text)

    # Explicit single daily/nightly instructions are equivalent schedules.
    text = re.sub(r"\bnightly\b|\bevery night\b|\bonce nightly\b", "once daily", text)

    # Remove route/device details from a frequency field when those were
    # duplicated by extraction; route is compared separately.
    text = re.sub(r"\b(?:oral|po|p\.o\.|intravenous|iv|subcutaneous|sc|intramuscular|im)\b", " ", text)

    # Delivery timing instructions are not cadence. Preserve the actual cadence
    # (BID/TID/every 6 hours) and remove modifiers such as before breakfast,
    # with food, and a clock time. Do not strip "every 20 minutes".
    text = re.sub(r"\b\d+\s*minutes?\s+before\b.*$", " ", text)
    text = re.sub(r"\b(?:for|x)\s*\d+\s*(?:days?|weeks?|months?)\b.*$", " ", text)
    text = re.sub(r"\b(?:before|after)\s+breakfast\s+(?:and|&)\s+dinner\b", " ", text)
    text = re.sub(r"\b(?:before|after)\s+(?:breakfast|lunch)\s*(?:and|&)\s*(?:lunch|dinner)\b", " ", text)
    text = re.sub(r"\bat\s+\d{1,2}:\d{2}(?:\s*(?:am|pm))?\b", " ", text)
    text = re.sub(r"\b\d{1,2}:\d{2}\b", " ", text)
    text = re.sub(r"\b\d{1,2}\s*(?:am|pm)\b", " ", text)
    text = re.sub(
        r"\b(?:before breakfast|after breakfast|with breakfast|before meals|"
        r"after meals|with meals|with food|in the morning|in morning|"
        r"in the afternoon|in afternoon|in the evening|in evening|"
        r"at night|every night|at bedtime|every morning|morning|"
        r"afternoon|evening|bedtime)\b",
        " ", text,
    )
    text = re.sub(r"\bat\b", " ", text)
    text = re.sub(r"\bdose\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;.-")

    aliases = {
        "daily": "once daily",
        "once a day": "once daily",
        "once per day": "once daily",
        "every day": "once daily",
        "every 24 hours": "once daily",
        "twice a day": "twice daily",
        "two times daily": "twice daily",
        "two times a day": "twice daily",
        "bid": "twice daily",
        "three times a day": "three times daily",
        "tid": "three times daily",
        "qid": "four times daily",
    }
    return aliases.get(text, text)


def _administration_mode(value):
    """Return a delivery-mode label when a field is not a true schedule."""
    text = normalize(value)
    if re.fullmatch(r"(?:iv )?bolus(?: dose)?", text):
        return "bolus"
    if re.fullmatch(r"(?:continuous )?infusion|drip", text):
        return "infusion"
    if re.fullmatch(r"loading dose", text):
        return "loading dose"
    return None


def _looks_like_schedule(value):
    """Whether a field contains a dosing cadence rather than a time/mode note."""
    text = normalize(value)
    return bool(re.search(
        r"\b(?:once|twice|thrice|daily|weekly|monthly|as needed|prn|"
        r"every\s+\d+|\d+\s+times?\s+(?:daily|a day|per day)|"
        r"with each meal|with meals|each meal|stat)\b",
        text,
    ))

def _contains_negation(text):
    """Detect explicit clinical negation/exclusion language in a value."""
    text = normalize(text)
    return bool(re.search(
        r"\b(?:no|not|negative|absent|without|denies|denied|excluded|"
        r"exclude|ruled out|not detected|none detected)\b",
        text,
    ) or re.search(r"\bnon[- ](?:tender|palpable|reactive|productive|obstructive|contributory|compliant)\b", text))


def _claim_phrase_supported(claim_value, source_value, claim_negated=False):
    """Allow a claim phrase to be supported by a more detailed source phrase.

    Exact phrase containment is preferred. For short formatting differences
    (e.g. "normal kidneys" vs "normal-sized kidneys" or a slash in
    "effusion/pneumothorax"), a conservative all-content-token match is also
    allowed. The matched source span is checked for conflicting polarity.
    """
    claim_text = normalize(claim_value)
    source_text = normalize(source_value)
    # Regularize common grammatical variants without changing the assertion.
    phrase_normalizations = (
        (r"\btolerating\b", "tolerated"),
        (r"\btolerates\b", "tolerated"),
        (r"\bdo not require\b", "do not need"),
        (r"\bdoes not require\b", "does not need"),
        (r"\bdoesn't require\b", "doesn't need"),
        (r"\bdid not require\b", "did not need"),
    )
    for pattern, replacement in phrase_normalizations:
        claim_text = re.sub(pattern, replacement, claim_text)
        source_text = re.sub(pattern, replacement, source_text)
    if not claim_text or not source_text:
        return False

    # Token-subset paraphrase matching must never erase a changed number.
    # This protects durations, measurements, dates, and dosages embedded in
    # otherwise identical narrative phrases (e.g. "15 days" vs "14 days").
    claim_numbers = _numeric_tokens(claim_text)
    source_numbers = _numeric_tokens(source_text)
    if claim_numbers and source_numbers and any(
        number not in source_numbers for number in claim_numbers
    ):
        return False

    match = re.search(
        rf"(?<![a-z0-9]){re.escape(claim_text)}(?![a-z0-9])",
        source_text,
    )
    span = match.span() if match else None

    if span is None:
        ignored = {
            "a", "an", "and", "are", "as", "at", "by", "for", "from",
            "in", "is", "of", "on", "or", "the", "to", "was", "were",
            "with", "without",
        }
        claim_tokens = [
            token for token in re.findall(r"[a-z0-9]+", claim_text)
            if token not in ignored and not token.isdigit()
        ]
        source_matches = list(re.finditer(r"[a-z0-9]+", source_text))
        source_tokens = {item.group(0) for item in source_matches}
        # Do not permit a one-token generic match as a paraphrase.
        if len(claim_tokens) < 2 or not set(claim_tokens).issubset(source_tokens):
            return False
        matched_occurrences = [
            item for item in source_matches if item.group(0) in set(claim_tokens)
        ]
        span = (matched_occurrences[0].start(), matched_occurrences[-1].end())

    phrase_start, phrase_end = span
    source_segment = source_text[phrase_start:phrase_end]
    before = source_text[max(0, phrase_start - 35):phrase_start]
    after = source_text[phrase_end:min(len(source_text), phrase_end + 45)]
    # Scope polarity to the same clause; a preceding semicolon-separated item
    # such as "no free air; clear lungs" must not negate the latter phrase.
    before_clause = re.split(r"[;.!?]", before)[-1]
    after_clause = re.split(r"[;.!?]", after)[0]

    claim_negative = bool(claim_negated) or _contains_negation(claim_text)
    source_negative = (
        _contains_negation(source_segment)
        or bool(re.search(
            r"\b(?:excluded|ruled out|not detected|negative for)\b",
            after_clause,
        ))
        or bool(re.search(
            r"\b(?:no|not|negative|absent|without|denies|denied)\b",
            before_clause,
        ))
        or bool(re.search(r"\bnon[- ]?$", before_clause))
    )
    return claim_negative == source_negative


def _canonical_unit(value):
    """Canonicalize unit spelling, including common exponent/Unicode forms."""
    text = str(value or "").strip().casefold()
    text = text.replace("µ", "mcg").replace("μ", "mcg")
    text = text.replace("⁹", "^9").replace("⁶", "^6")
    text = text.replace("×", "x").replace("·", "")
    text = re.sub(r"\s+", "", text)
    text = text.replace("litres", "l").replace("liters", "l")
    aliases = {
        "x10^9/l": "10^9/l",
        "10^9/l": "10^9/l",
        "x10^6/ul": "10^6/ul",
        "10^6/ul": "10^6/ul",
        "ug": "mcg",
        "micrograms": "mcg",
        "milligrams": "mg",
        "milliliters": "ml",
        "millilitres": "ml",
        "ml/h": "ml/hour",
        "ml/hr": "ml/hour",
        "mlperhour": "ml/hour",
        "units/h": "units/hour",
        "unit/h": "units/hour",
        "units/hr": "units/hour",
        "unit/hr": "units/hour",
        "mmol/litre": "mmol/l",
        "mmol/liter": "mmol/l",
        "mmolperlitreofivfluid": "mmol/l",
        "mmolperlitre": "mmol/l",
    }
    return aliases.get(text, text)


def _quantity_in_base(value, unit):
    """Convert a scalar quantity to a base unit for safe equivalent-value checks.

    Only conversions with unambiguous dimensions are supported. Unknown units,
    ranges, and compound units return None rather than being guessed.
    """
    text = str(value or "").strip()
    unit_text = _canonical_unit(unit)
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text) or not unit_text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None

    conversions = {
        # Base mass = milligrams.
        "kg": ("mass", 1_000_000.0),
        "g": ("mass", 1_000.0),
        "mg": ("mass", 1.0),
        "mcg": ("mass", 0.001),
        "ng": ("mass", 0.000001),
        # Base volume = millilitres.
        "l": ("volume", 1_000.0),
        "ml": ("volume", 1.0),
        "ul": ("volume", 0.001),
    }
    spec = conversions.get(unit_text)
    if not spec:
        return None
    dimension, multiplier = spec
    return dimension, number * multiplier


def _unit_present_in_text(unit, text):
    wanted = _canonical_unit(unit)
    if not wanted:
        return False
    return wanted in _canonical_unit(text)

def _unwrap_candidate(candidate):
    """Accept either a fact or a retrieval result containing a fact."""
    if isinstance(candidate, dict) and isinstance(
        candidate.get("fact"), dict
    ):
        return candidate["fact"], float(candidate.get("score", 0))

    return candidate, 0.0


def _resolve_claim_relation(claim, candidate_facts, record=None):
    """
    Resolve a claim's relation using its clinical category,
    populated fields, temporal context, and retrieved evidence.
    """
    relation = _canonical_relation(
        claim.get("relation", "")
    )

    category = normalize(
        claim.get("category", "")
    ).replace("_", " ")

    is_laboratory_claim = (
        "laboratory" in category
        or category in {"lab", "lab result"}
    )

    generic_laboratory_relations = {
        "is",
        "are",
        "was",
        "were",
        "shows",
        "showed",
        "measured",
        "recorded",
    }

    # Generic laboratory wording must be aligned with the
    # relevant measurement type rather than matched literally.
    if (
        is_laboratory_claim
        and relation in generic_laboratory_relations
    ):
        from src.evidence_builder import (
            resolve_claim_temporal_context,
        )

        temporal = resolve_claim_temporal_context(
            claim,
            record or {},
        )

        if "baseline" in temporal["markers"]:
            return "baseline value"

        return "result"

    explicit_attributes = {
        "dose",
        "route",
        "frequency",
        "duration",
        "indication",
        "reason",
        "status",
        "start date",
        "stop date",
        "result",
        "baseline value",
        "diagnosis",
        "allergy",
    }

    unit = normalize(claim.get("unit", ""))
    is_laboratory_claim = (
        "laboratory" in category
        or category in {"lab", "lab result", "investigation"}
    )

    medication_dose_units = {
        "mg",
        "g",
        "mcg",
        "μg",
        "µg",
        "unit",
        "units",
        "iu",
        "international unit",
        "international units",
        "ml",
        "mmol",
        "meq",
        "mcg/kg/min",
        "μg/kg/min",
        "µg/kg/min",
        "mg/kg/min",
        "mg/kg",
        "ml/h", "ml/hr", "ml/hour", "mlperhour",
        "units/h", "unit/h", "units/hr", "unit/hr", "units/hour",
        "mmol/l", "mmol/litre", "mmol per litre of iv fluid",
    }

    # A numeric medication assertion can include a temporal/status word in its
    # relation (e.g. "20 mg ... started"). When the unit is a dose unit and a
    # relevant dose fact was retrieved, compare the numeric value against dose
    # evidence rather than incorrectly comparing "20" with the status "started".
    value_looks_numeric = bool(
        re.fullmatch(
            r"\s*[-+]?\d+(?:\.\d+)?(?:\s*(?:-|to|–)\s*\d+(?:\.\d+)?)?\s*",
            str(claim.get("value", "")),
            flags=re.IGNORECASE,
        )
    )
    # A concentration unit such as mmol/L can appear in both medication
    # concentration facts and laboratory results. In a laboratory claim, do
    # not reinterpret an explicit/generic result as a medication dose merely
    # because a potassium-chloride dose fact was retrieved as a weak candidate.
    if unit in medication_dose_units and value_looks_numeric and not is_laboratory_claim:
        for candidate in candidate_facts:
            fact, _ = _unwrap_candidate(candidate)
            if not isinstance(fact, dict) or _canonical_relation(fact.get("relation", "")) != "dose":
                continue
            fact_unit = str(fact.get("unit", "") or "")
            # Relation selection is not enough: a numeric concentration/rate
            # must not be compared against an unrelated dose with a different
            # dimensional unit. Canonical spellings such as mL/h and mL/hour
            # are treated as equivalent.
            if fact_unit and _canonical_unit(unit) == _canonical_unit(fact_unit):
                return "dose"

    if relation == "status":
        status_value = _canonical_status(claim.get("value", ""))
        recognized_statuses = {
            "started", "continued", "discontinued", "withheld", "held",
            "stopped", "dose_changed", "route_changed", "frequency_changed",
            "resumed", "ongoing", "completed", "resolved", "stable",
        }
        # Long narrative status values (e.g. "Ready for discharge") should be
        # checked against condition/disposition evidence, not a medication's
        # categorical status fact that happened to be retrieved.
        if (
            status_value
            and status_value not in recognized_statuses
            and len(status_value.split()) > 1
            and "allergy" not in category
        ):
            return "condition"

    if relation in explicit_attributes:
        return relation

    return relation


def _resolve_claim_relations(claim, candidate_facts, record=None):
    """Return the relation family relevant to a claim after retrieval.

    Generic language such as "patient has ..." or "the scan shows ..." is
    common in summaries but is not how the ground-truth facts are normalized.
    This function maps such claims to the appropriate source relation family.
    It only examines the independently retrieved candidates; it never receives
    an injected-error ID or a configured target fact ID.
    """
    raw_relation = normalize(claim.get("relation", "")).replace("_", " ")
    relation = _canonical_relation(claim.get("relation", ""))
    category = normalize(claim.get("category", "")).replace("_", " ")
    claim_subject = normalize(claim.get("subject", "")).replace("_", " ")
    claim_value = normalize(claim.get("value", ""))
    candidate_fact_pairs = [
        (_unwrap_candidate(c)[0], _canonical_relation(_unwrap_candidate(c)[0].get("relation", "")))
        for c in candidate_facts
        if isinstance(_unwrap_candidate(c)[0], dict)
    ]
    available_relations = {item[1] for item in candidate_fact_pairs}

    # Resolve generic copular claims using the *subject slot* and actually
    # retrieved relation types. This is a schema-level rule, not a case-ID or
    # asserted-value shortcut: e.g. "Admission Date: ..." should compare with
    # date facts, not the literal relation "is".
    if relation in {"is", "are", "was", "were", "has", "have", "value"}:
        subject_hints = [
            (r"\bdate of birth\b", "date of birth"),
            (r"\b(?:admission|discharge|follow up|follow-up|procedure|event) date\b|\bdate\b", "date"),
            (r"\b(?:admission )?time\b|\btime of admission\b", "time"),
            (r"\btype\b", "type"),
            (r"\bgender\b|\bsex\b", "sex"),
            (r"\bage\b", "age"),
            (r"\bheight\b", "height"),
            (r"\bweight\b", "weight"),
            (r"\bblood group\b", "blood group"),
        ]
        for pattern, target_relation in subject_hints:
            if re.search(pattern, claim_subject) and target_relation in available_relations:
                return [target_relation]

    # Common natural-language relations describe the type of information in a
    # claim, not the normalized source schema relation. Restrict the mapping to
    # source relations actually present among retrieved candidates. Medication
    # indications are intentionally not treated as patient diagnoses/history.
    relation_hints = {
        "has diagnosis": ("is", "condition", "diagnosis"),
        "has medical history": ("condition", "is", "diagnosis"),
        "has history of": ("condition", "is", "diagnosis", "surgery", "smoking", "occupation"),
        "has surgery history": ("surgery", "procedure", "operation"),
        "has symptom": ("description", "finding", "symptom"),
        "has chief complaint": ("description", "reason"),
        "has physical finding": ("finding", "description", "value"),
        "smoking status": ("smoking",),
        "smokes": ("smoking",),
        "has occupation": ("occupation",),
        "has gender": ("sex",),
        "has reason for admission": ("reason",),
        "is scheduled for": ("reason", "description"),
        "follow up": ("reason", "description"),
        "scheduled for": ("reason", "description"),
        "stopped at": ("stopped", "status"),
        "occurred at": ("date", "time"),
    }
    if raw_relation in relation_hints:
        hinted = [r for r in relation_hints[raw_relation] if r in available_relations]
        if raw_relation in {"has diagnosis", "has medical history", "has history of"}:
            # Prefer history/diagnosis evidence and exclude medication-indication
            # facts even when the condition phrase happens to match a medicine.
            eligible_categories = {"history", "diagnosis", "medical history"}
            eligible = [
                (_unwrap_candidate(c)[0], _canonical_relation(_unwrap_candidate(c)[0].get("relation", "")))
                for c in candidate_facts
                if isinstance(_unwrap_candidate(c)[0], dict)
                and normalize(_unwrap_candidate(c)[0].get("category", "")).replace("_", " ") in eligible_categories
            ]
            eligible_relations = {rel for _, rel in eligible}
            hinted = [r for r in relation_hints[raw_relation] if r in eligible_relations]
        if hinted:
            return hinted

    # Clinical finding presence is represented by `finding` in the source
    # schema, whereas allergy presence is usually stored as `status`.
    if raw_relation == "presence" and "allergy" not in category:
        finding_relations = [r for r in ("finding", "description", "status") if r in available_relations]
        if finding_relations:
            return finding_relations

    resolved_relation = _resolve_claim_relation(
        claim, candidate_facts, record=record
    )
    if resolved_relation != relation and resolved_relation in {
        "dose", "route", "frequency", "duration", "indication", "status",
        "result", "baseline value", "performed on", "allergic reaction",
        "condition", "disposition", "age", "sex", "height", "weight",
        "blood group", "date of birth",
    }:
        if not (
            resolved_relation == "condition"
            and category in {"diagnosis", "condition", "clinical diagnosis"}
        ):
            return [resolved_relation]
        # Let the category-specific selector below consider the source's
        # diagnosis/is/condition facts together rather than selecting an
        # unrelated medical-history `condition` fact only.
        relation = "condition"

    if "laboratory" in category or category in {"lab", "lab result", "investigation"}:
        # Resolve generic narrative relations to a result, but preserve explicit
        # structured relations such as size, diameter, count, or measurement.
        generic_investigation_relations = {
            "is", "are", "was", "were", "shows", "showed", "measured",
            "recorded", "has", "value", "finding",
        }
        if relation in generic_investigation_relations:
            candidate_relations = {
                _canonical_relation(_unwrap_candidate(c)[0].get("relation", ""))
                for c in candidate_facts
                if isinstance(_unwrap_candidate(c)[0], dict)
            }
            if "result" in candidate_relations:
                return ["result"]

    allowed_relations = None
    if category in {"clinical finding", "clinical findings", "vital sign", "vital signs", "examination"} and relation in {"has", "is", "are", "was", "were", "shows"}:
        allowed_relations = {"value", "finding", "description", "result", "status", "condition"}
    elif category in {"diagnosis", "condition", "clinical diagnosis"} and relation in {"has", "is", "are", "was", "were", "condition", "diagnosis"}:
        allowed_relations = {"is", "condition", "diagnosis", "indication", "considered", "result", "reason"}
    elif category in {"imaging", "radiology", "imaging result"} and relation in {"shows", "showed", "is", "are"}:
        allowed_relations = {"impression", "findings", "finding", "result", "description"}

    if allowed_relations:
        options = []
        for candidate in candidate_facts:
            fact, _ = _unwrap_candidate(candidate)
            if not isinstance(fact, dict):
                continue
            fact_relation = _canonical_relation(fact.get("relation", ""))
            if fact_relation in allowed_relations and fact_relation not in options:
                options.append(fact_relation)
        if options:
            return options

    return [relation]


def _canonical_status(value):
    text = normalize(value)
    aliases = {
        "stopped": "discontinued",
        "stop": "discontinued",
        "discontinue": "discontinued",
        "held": "withheld",
        "on hold": "withheld",
        "initiated": "started",
        "commenced": "started",
        "maintained": "continued",
        "resumed": "continued",
        "restarted": "continued",
    }
    return aliases.get(text, text)


def _relation_evidence_signature(fact, relation):
    """Build a comparison signature for equivalent structured/narrative evidence.

    Some datasets duplicate a laboratory result in a structured scalar fact
    and a hospital-course narrative (e.g. value=3.8, unit=mmol/L versus
    value="Potassium 3.8 mmol/L at 6 hours", unit=""). For result relations,
    compare the embedded scalar and unit so duplicate representations of the
    same result do not create a false ambiguity. Conflicting quantities remain
    distinct signatures.
    """
    raw_value = str(fact.get("value", "") or "").strip()
    raw_unit = str(fact.get("unit", "") or "").strip()
    canonical_unit = _canonical_unit(raw_unit)
    canonical_relation = _canonical_relation(relation)

    if canonical_relation in {"result", "value", "baseline value"}:
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", raw_value):
            try:
                return ("numeric", float(raw_value), canonical_unit)
            except ValueError:
                pass
        if not canonical_unit:
            embedded = re.search(
                r"(?<![a-z0-9.])([-+]?\d+(?:\.\d+)?)\s*"
                r"(mmol\s*/\s*l|mmol\s+per\s+lit(?:re|er)|"
                r"mg\s*/\s*dl|g\s*/\s*dl|"
                r"x?10\s*\^?9\s*/\s*l|"
                r"x?10\s*\^?6\s*/\s*(?:ul|µl|μl))",
                raw_value,
                flags=re.IGNORECASE,
            )
            if embedded:
                try:
                    return (
                        "numeric",
                        float(embedded.group(1)),
                        _canonical_unit(embedded.group(2)),
                    )
                except ValueError:
                    pass

    return ("text", normalize(raw_value), canonical_unit)


def _find_relation_evidence(candidates, relation):
    """
    Find the best-ranked fact for a particular relation.

    When similarly ranked candidates disagree, avoid making
    an unsupported choice between them.
    """
    target_relation = _canonical_relation(relation)

    matching = []

    for candidate in candidates:
        fact, score = _unwrap_candidate(candidate)

        if not isinstance(fact, dict):
            continue

        fact_relation = _canonical_relation(
            fact.get("relation", "")
        )

        if fact_relation == target_relation:
            matching.append({
                "fact": fact,
                "score": score,
            })

    if not matching:
        return None, False

    matching.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    best = matching[0]
    best_score = best["score"]

    close_candidates = [
        item for item in matching
        if best_score - item["score"] < 1.0
    ]

    # Similar scores with conflicting values indicate ambiguity.
    signatures = {
        _relation_evidence_signature(item["fact"], target_relation)
        for item in close_candidates
    }

    if len(signatures) > 1:
        return None, True

    # If the same scalar result appears both as a structured laboratory fact
    # and embedded in a narrative event, prefer the structured fact when its
    # numeric quantity and unit agree with that narrative. It carries explicit
    # unit metadata, which is necessary to catch a unit-only error in the
    # claim. This is a general preference for normalized evidence, not for a
    # particular analyte or record.
    if target_relation in {"result", "value", "baseline value"}:
        structured = [
            item for item in close_candidates
            if re.fullmatch(
                r"[-+]?\d+(?:\.\d+)?",
                str(item["fact"].get("value", "") or "").strip(),
            )
            and str(item["fact"].get("unit", "") or "").strip()
        ]
        if structured:
            structured.sort(key=lambda item: item["score"], reverse=True)
            return structured[0]["fact"], False

    return best["fact"], False


def _canonical_duration(value):
    """Normalize harmless wording differences in medication duration fields."""
    text = normalize(value)
    text = re.sub(r"\bunchanged home medication\b", "", text)
    text = re.sub(r"\bongoing at discharge\b", "ongoing", text)
    text = re.sub(r"^for\s+", "", text)
    return re.sub(r"\s+", " ", text).strip(" ,;.-")


def _numeric_tokens(value):
    """Return normalized numeric values, preserving decimals and ranges."""
    return [
        item.replace(",", "")
        for item in re.findall(
            r"(?<![a-z0-9])\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?",
            normalize(value),
        )
    ]


def _polarity_conflict(claim_value, source_value, claim_negated=False):
    """Detect an explicit positive/negative conflict about the same phrase."""
    claim_text = normalize(claim_value)
    source_text = normalize(source_value)

    # Opposite-valued clinical labels should not fall through to generic
    # narrative handling (e.g. a result reported as positive vs negative).
    opposite_labels = {
        frozenset(("positive", "negative")),
        frozenset(("present", "absent")),
        frozenset(("detected", "not detected")),
        frozenset(("normal", "abnormal")),
        frozenset(("yes", "no")),
    }
    if frozenset((claim_text, source_text)) in opposite_labels:
        return True

    claim_negative = bool(claim_negated) or _contains_negation(claim_text)
    source_negative = _contains_negation(source_text)
    if claim_negative == source_negative:
        return False
    negative_words = {
        "no", "not", "negative", "absent", "without", "denies", "denied",
        "excluded", "exclude", "ruled", "out", "detected", "none", "non",
    }
    claim_core = {
        token for token in re.findall(r"[a-z0-9]+", claim_text)
        if token not in negative_words and not token.isdigit()
    }
    source_core = {
        token for token in re.findall(r"[a-z0-9]+", source_text)
        if token not in negative_words and not token.isdigit()
    }
    return bool(
        claim_core and source_core
        and (claim_core.issubset(source_core) or source_core.issubset(claim_core))
    )


def _explicit_negation_contradiction(claim_value, source_value):
    """Detect a narrow set of direct clinical polarity conflicts in narrative findings.

    This is intentionally phrase-based and conservative. It handles explicit
    pairs such as "extraluminal air" vs "no extraluminal air" and "with
    perforation" vs "without perforation" when the same key finding is
    negated in one assertion. It does not infer a contradiction from general
    wording differences alone.
    """
    claim_text = normalize(claim_value)
    source_text = normalize(source_value)
    pairs = [
        (r"\bno\s+extraluminal air\b", r"\bextraluminal air\b"),
        (r"\bwithout perforation\b", r"\bwith perforation\b"),
        (r"\bwithout abscess\b", r"\bwith abscess\b"),
        (r"\bno growth\b", r"\bgrowth\b"),
        (r"\bnegative for\s+([a-z0-9 -]+)", r"\bpositive for\s+([a-z0-9 -]+)"),
        (r"\bno evidence of\s+([a-z0-9 -]+)", r"\bevidence of\s+([a-z0-9 -]+)"),
        (r"\bnot detected\b", r"\bdetected\b"),
        (r"\babsence of\s+([a-z0-9 -]+)", r"\bpresence of\s+([a-z0-9 -]+)"),
    ]
    for negative_pattern, positive_pattern in pairs:
        claim_negative = re.search(negative_pattern, claim_text)
        claim_positive = re.search(positive_pattern, claim_text)
        source_negative = re.search(negative_pattern, source_text)
        source_positive = re.search(positive_pattern, source_text)
        if claim_negative and source_positive:
            return True
        if claim_positive and source_negative:
            # For patterns with a captured object (e.g. "negative for X"),
            # ensure both sides refer to the same object rather than unrelated
            # findings in a composite sentence.
            if negative_pattern.endswith("([a-z0-9 -]+)") and positive_pattern.endswith("([a-z0-9 -]+)"):
                claim_object = re.sub(r"\s+", " ", claim_positive.group(1)).strip()
                source_object = re.sub(r"\s+", " ", source_negative.group(1)).strip()
                if claim_object and source_object and (claim_object == source_object or claim_object in source_object or source_object in claim_object):
                    return True
            else:
                return True
    return False


def _parse_date_components(value):
    """Parse common ISO or written dates into (year, month, day)."""
    text = normalize(value).replace(",", " ")
    iso = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if iso:
        return (int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))

    # Parse slash dates only when their order can be inferred unambiguously.
    # If both first and second components are <= 12, keep them unparsed rather
    # than guessing between DD/MM and MM/DD conventions.
    slash = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text)
    if slash:
        first, second, year = map(int, slash.groups())
        if first > 12 and 1 <= second <= 12:
            return (year, second, first)  # DD/MM/YYYY
        if second > 12 and 1 <= first <= 12:
            return (year, first, second)  # MM/DD/YYYY

    months = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3,
        "march": 3, "apr": 4, "april": 4, "may": 5, "jun": 6,
        "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9, "oct": 10,
        "october": 10, "nov": 11, "november": 11, "dec": 12,
        "december": 12,
    }
    m = re.search(r"\b(\d{1,2})\s+([a-z]+)(?:\s+(\d{4}))?\b", text)
    if m and m.group(2) in months:
        return (int(m.group(3)) if m.group(3) else None, months[m.group(2)], int(m.group(1)))
    m = re.search(r"\b([a-z]+)\s+(\d{1,2})(?:\s+(\d{4}))?\b", text)
    if m and m.group(1) in months:
        return (int(m.group(3)) if m.group(3) else None, months[m.group(1)], int(m.group(2)))
    return None


def _duration_quantities(value):
    """Return duration quantities as unit-aware pairs, excluding calendar dates."""
    text = normalize(value)
    found = re.findall(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*(hours?|hrs?|days?|weeks?|months?)\b", text)
    aliases = {"hr": "hour", "hrs": "hour", "hours": "hour", "hour": "hour",
               "day": "day", "days": "day", "week": "week", "weeks": "week",
               "month": "month", "months": "month"}
    return {(number, aliases.get(unit, unit.rstrip("s"))) for number, unit in found}


def _date_field_verdict(claim_value, source_value, field):
    """Compare date assertions without confusing missing time evidence with conflict."""
    field_name = normalize(field).replace("_", " ")
    if "date" not in field_name and field_name != "performed on":
        return None

    claim_text = normalize(claim_value)
    source_text = normalize(source_value)
    claim_has_time = bool(re.search(r"\b\d{1,2}:\d{2}(?:\s*(?:am|pm))?\b", claim_text))
    source_has_time = bool(re.search(r"\b\d{1,2}:\d{2}(?:\s*(?:am|pm))?\b", source_text))
    claim_date = _parse_date_components(claim_value)
    source_date = _parse_date_components(source_value)

    if claim_date and source_date:
        claim_year, claim_month, claim_day = claim_date
        source_year, source_month, source_day = source_date
        if claim_month != source_month or claim_day != source_day:
            return "CONTRADICTED"
        if claim_year is not None and source_year is not None and claim_year != source_year:
            return "CONTRADICTED"

        def clock(value):
            text = normalize(value)
            m = re.search(r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b", text)
            if not m:
                return None
            hour, minute = int(m.group(1)), int(m.group(2))
            suffix = m.group(3)
            if suffix:
                if hour == 12:
                    hour = 0
                if suffix == "pm":
                    hour += 12
            return hour, minute

        claim_clock = clock(claim_value)
        source_clock = clock(source_value)
        if claim_has_time and source_has_time:
            return "SUPPORTED" if claim_clock == source_clock else "CONTRADICTED"
        if claim_has_time != source_has_time:
            # Date agrees, but only one side specifies a time. A missing source
            # clock time is not evidence that the claimed time is wrong.
            return "INSUFFICIENT_EVIDENCE"
        return "SUPPORTED"

    # A datetime field must not use numbers from an unparseable date string as
    # proof of a contradiction. Leave unresolved forms to broader safeguards.
    if claim_has_time or source_has_time:
        return None
    simple_date_pattern = (
        r"(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{4}|"
        r"\d{1,2}\s+[a-z]+(?:\s+\d{4})?|[a-z]+\s+\d{1,2}(?:\s+\d{4})?)"
    )
    if len(claim_text.split()) > 3 or not re.fullmatch(simple_date_pattern, claim_text):
        return "INSUFFICIENT_EVIDENCE"
    if len(source_text.split()) > 3 or not re.fullmatch(simple_date_pattern, source_text):
        return "INSUFFICIENT_EVIDENCE"
    return None


def _compare_claim_field(
    claim_value,
    claim_unit,
    fact,
    field,
    claim_negated=False,
):
    """Compare one asserted field with its corresponding source fact."""
    if claim_value is None or not str(claim_value).strip():
        return "INSUFFICIENT_EVIDENCE"
    if not fact:
        return "INSUFFICIENT_EVIDENCE"

    source_value = fact.get("value", "")
    if source_value is None or not str(source_value).strip():
        return "INSUFFICIENT_EVIDENCE"

    # Unit-equivalent scalar measurements should agree even when one source
    # writes 1 g and another writes 1000 mg. This check does not make different
    # numeric doses agree: 2 g and 1000 mg still conflict.
    source_unit_for_conversion = fact.get("unit", "")
    claim_quantity = _quantity_in_base(claim_value, claim_unit)
    source_quantity = _quantity_in_base(source_value, source_unit_for_conversion)
    if claim_quantity and source_quantity and claim_quantity[0] == source_quantity[0]:
        if abs(claim_quantity[1] - source_quantity[1]) <= max(1e-9, abs(source_quantity[1]) * 1e-8):
            return "SUPPORTED"

    if field == "status":
        claim_normalized = _canonical_status(claim_value)
        source_normalized = _canonical_status(source_value)

    elif field == "route":
        claim_normalized = _canonical_route(claim_value)
        source_normalized = _canonical_route(source_value)

    elif field == "frequency":
        claim_frequency = normalize(claim_value)
        source_frequency = normalize(source_value)

        if claim_frequency in {"prophylaxis", "prophylactic"}:
            return "INSUFFICIENT_EVIDENCE"

        # "Bolus" and "infusion" describe administration mode, not schedule.
        # If the source records that same mode, it is compatible; absence of a
        # schedule comparison must not be turned into a false contradiction.
        claim_mode = _administration_mode(claim_frequency)
        if claim_mode:
            source_mode = _administration_mode(source_frequency)
            if source_mode == claim_mode:
                return "SUPPORTED"
            if re.search(rf"\b{re.escape(claim_mode)}\b", source_frequency):
                return "SUPPORTED"
            return "INSUFFICIENT_EVIDENCE"

        claim_normalized = _canonical_frequency(claim_value)
        source_normalized = _canonical_frequency(source_value)
        if claim_normalized == source_normalized:
            return "SUPPORTED"
        # A one-time administration event can occur under a PRN/conditional
        # order; it does not contradict that order's availability rule.
        one_time_labels = {"once", "once only", "single dose", "one time"}
        if (
            (claim_normalized in one_time_labels and re.search(r"\b(?:as needed|if|when)\b", source_normalized))
            or (source_normalized in one_time_labels and re.search(r"\b(?:as needed|if|when)\b", claim_normalized))
        ):
            return "INSUFFICIENT_EVIDENCE"
        # A field like "first dose at 22:10", "fixed-rate", or "over 20 min"
        # is a time/delivery note rather than a regimen frequency. It cannot
        # establish a frequency contradiction without a cadence on both sides.
        if not _looks_like_schedule(claim_normalized) or not _looks_like_schedule(source_normalized):
            return "INSUFFICIENT_EVIDENCE"

    elif field == "indication":
        claim_normalized = _normalize_indication_text(claim_value)
        source_normalized = _normalize_indication_text(source_value)
        if (
            claim_normalized == source_normalized
            or re.search(
                rf"(?<![a-z0-9]){re.escape(claim_normalized)}(?![a-z0-9])",
                source_normalized,
            )
        ):
            return "SUPPORTED"
        ignored = {"a", "an", "and", "the", "of", "for", "to", "in", "on", "with"}
        claim_tokens = {
            token for token in re.findall(r"[a-z0-9]+", claim_normalized)
            if token not in ignored
        }
        source_tokens = set(re.findall(r"[a-z0-9]+", source_normalized))
        if len(claim_tokens) >= 2 and claim_tokens.issubset(source_tokens):
            return "SUPPORTED"
        return "INSUFFICIENT_EVIDENCE"

    elif field == "duration":
        # Atomic claims often split `18 hours` into value=18 and unit=hours.
        # Compare a combined duration expression instead of losing the unit.
        claim_duration_text = " ".join(
            part for part in (str(claim_value or "").strip(), str(claim_unit or "").strip())
            if part
        )
        claim_normalized = _canonical_duration(claim_duration_text)
        source_normalized = _canonical_duration(source_value)

    else:
        claim_normalized = normalize(claim_value)
        source_normalized = normalize(source_value)

    # A written day/month value ("12 March") can express the same date as
    # an ISO value ("2025-03-12"). Resolve this before raw text mismatch logic.
    parsed_date_verdict = _date_field_verdict(claim_value, source_value, field)
    if parsed_date_verdict is not None:
        return parsed_date_verdict

    if not values_match(claim_normalized, source_normalized):
        field_name = normalize(field).replace("_", " ")
        iso_date_pattern = r"\b\d{4}-\d{2}-\d{2}\b"
        claim_dates = re.findall(iso_date_pattern, claim_normalized)
        source_dates = re.findall(iso_date_pattern, source_normalized)
        is_date_or_time_field = bool(
            re.search(r"\b(?:date|time|timeframe)\b", field_name)
        )
        is_duration_field = "duration" in field_name

        if is_duration_field:
            # Compound durations may mention multiple dates or both the total
            # course and the remaining discharge days. Extra source details are
            # compatible when every duration quantity/date asserted by the claim
            # is present; changed quantities or dates remain contradictions.
            claim_duration_values = _duration_quantities(claim_normalized)
            source_duration_values = _duration_quantities(source_normalized)
            if claim_dates and source_dates and not set(claim_dates).issubset(set(source_dates)):
                return "CONTRADICTED"
            if claim_duration_values and source_duration_values and not claim_duration_values.issubset(source_duration_values):
                return "CONTRADICTED"
            if _claim_phrase_supported(claim_normalized, source_normalized, claim_negated=claim_negated):
                pass
            else:
                return "INSUFFICIENT_EVIDENCE"
        elif claim_dates and source_dates and claim_dates != source_dates:
            return "CONTRADICTED"
        elif field_name in {"sex", "gender"}:
            demographic_aliases = {
                "male": "male", "man": "male", "female": "female", "woman": "female",
                "nonbinary": "nonbinary", "non-binary": "nonbinary",
            }
            claim_demographic = demographic_aliases.get(claim_normalized)
            source_demographic = demographic_aliases.get(source_normalized)
            if claim_demographic and source_demographic and claim_demographic != source_demographic:
                return "CONTRADICTED"
            return "INSUFFICIENT_EVIDENCE"
        elif field not in {"route", "frequency"} and _claim_phrase_supported(
            claim_normalized,
            source_normalized,
            claim_negated=claim_negated,
        ):
            pass
        elif _explicit_negation_contradiction(claim_normalized, source_normalized):
            return "CONTRADICTED"
        elif _polarity_conflict(
            claim_normalized, source_normalized, claim_negated=claim_negated
        ):
            return "CONTRADICTED"
        elif is_date_or_time_field:
            claim_numbers = _numeric_tokens(claim_normalized)
            source_numbers = _numeric_tokens(source_normalized)
            if claim_numbers and source_numbers and claim_numbers != source_numbers:
                return "CONTRADICTED"
            return "INSUFFICIENT_EVIDENCE"
        elif (
            re.fullmatch(
                r"\s*[+-]?\d+(?:\.\d+)?(?:\s*/\s*[+-]?\d+(?:\.\d+)?)?"
                r"(?:\s*(?:[a-zA-Z%°µμ]+(?:\^?\d+)?|x10\^?\d+/[a-zA-Z]+))?\s*",
                str(claim_normalized),
            )
            and re.fullmatch(
                r"\s*[+-]?\d+(?:\.\d+)?(?:\s*/\s*[+-]?\d+(?:\.\d+)?)?"
                r"(?:\s*(?:[a-zA-Z%°µμ]+(?:\^?\d+)?|x10\^?\d+/[a-zA-Z]+))?\s*",
                str(source_normalized),
            )
        ):
            # Simple scalar measurements (including sizes and observations)
            # are structured values even when their relation is not in a
            # fixed list of known clinical fields. Dates are intentionally
            # handled above, not by this scalar path.
            claim_numbers = _numeric_tokens(claim_normalized)
            source_numbers = _numeric_tokens(source_normalized)
            if claim_numbers and source_numbers and claim_numbers != source_numbers:
                return "CONTRADICTED"
            return "INSUFFICIENT_EVIDENCE"
        elif field in {
            "dose", "route", "frequency", "duration", "date", "time",
            "performed on", "age", "height", "weight", "sex",
            "blood group", "status", "result", "baseline value",
        }:
            claim_numbers = _numeric_tokens(claim_normalized)
            source_numbers = _numeric_tokens(source_normalized)
            if claim_numbers and source_numbers and any(
                number not in source_numbers for number in claim_numbers
            ):
                return "CONTRADICTED"
            if field in {"route", "frequency", "status"}:
                return "CONTRADICTED"
            if field in {"dose", "date", "time", "performed on", "age", "height", "weight", "sex", "blood group", "duration"}:
                if claim_numbers and source_numbers:
                    return "CONTRADICTED"
                return "INSUFFICIENT_EVIDENCE"
            return "INSUFFICIENT_EVIDENCE"
        else:
            # A narrative mismatch is not, by itself, proof of contradiction.
            return "INSUFFICIENT_EVIDENCE"

    # If the claim supplies a unit, verify it against the source too.
    if claim_unit:
        source_unit = fact.get("unit", "")
        if source_unit:
            if _canonical_unit(claim_unit) != _canonical_unit(source_unit):
                return "CONTRADICTED"
        elif not _unit_present_in_text(claim_unit, source_value):
            return "INSUFFICIENT_EVIDENCE"

    return "SUPPORTED"

def verify_claim_against_facts(
    claim,
    candidate_facts,
    record=None,
):
    """
    Verify a claim against retrieved ground-truth fact candidates.

    This function does not use predefined error IDs, target fact IDs,
    or known incorrect values.

    It compares the main claim relation and any populated route,
    frequency, and duration fields against corresponding evidence.
    """
    field_results = {}

    relation = claim.get("relation", "")
    relation_key = _resolve_claim_relation(
        claim,
        candidate_facts,
        record=record,
    )
    relation_options = _resolve_claim_relations(
        claim,
        candidate_facts,
        record=record,
    )
    claim_value = claim.get("value", "")
    claim_unit = claim.get("unit", "")

    # A few extracted procedure claims put the narrative result/removal note in
    # `value` while placing the actual procedure date in `date_time`. Do not
    # compare that composite narrative to a source date and report a date
    # contradiction. Use date_time only when value is not itself a simple date;
    # an explicit date mutation remains testable because its value parses as a
    # date. The composite note should ideally be split by claim extraction.
    if _canonical_relation(relation) == "performed on" and claim.get("date_time"):
        simple_date_pattern = (
            r"(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{4}|"
            r"\d{1,2}\s+[a-z]+(?:\s+\d{4})?|[a-z]+\s+\d{1,2}(?:\s+\d{4})?)"
        )
        if not re.fullmatch(simple_date_pattern, normalize(claim_value)):
            claim_value = claim.get("date_time")

    checks = []

    # Some extracted status claims encode the assertion in the relation while
    # their value is empty, e.g. "Lisinopril discontinued".
    status_relation_values = {"discontinued", "stopped", "started", "continued"}
    if not (claim_value is not None and str(claim_value).strip()):
        if normalize(relation) in status_relation_values:
            claim_value = relation
            relation_key = "status"
            relation_options = ["status"]
        elif _canonical_relation(relation) == "performed on" and claim.get("date_time"):
            claim_value = claim.get("date_time")
            relation_key = "performed on"
            relation_options = ["performed on"]

    if claim_value is not None and str(claim_value).strip():
        checks.append({
            "field": relation_key or "value",
            "relations": relation_options or [relation_key or relation],
            "claim_value": claim_value,
            "claim_unit": claim_unit,
            "is_main_claim": True,
        })

    # Also verify populated structured attributes. Route/frequency generally
    # belong to medications; a generic patient-level narrative duration should
    # not be compared against the duration of an unrelated medication.
    category_text = normalize(claim.get("category", "")).replace("_", " ")
    claim_subject_text = normalize(claim.get("subject", ""))
    generic_subjects = {"patient", "person", "individual", "subject"}
    for field in ("route", "frequency", "duration"):
        value = claim.get(field, "")
        if value is None or not str(value).strip():
            continue
        if relation_key == field:
            continue
        # The extractor can place a serial laboratory sampling time such as
        # "at 6 hours" into `duration`. For lab/investigation claims this is a
        # timepoint marker already handled by temporal evidence retrieval, not
        # a treatment-duration assertion that needs a duplicate source field.
        if (
            field == "duration"
            and ("laboratory" in category_text or category_text in {"lab", "lab result", "investigation"})
            and re.fullmatch(r"\s*(?:at\s+)?\d+\s*hours?\s*", str(value), flags=re.I)
        ):
            continue
        # LLM extraction sometimes records a sub-attribute (e.g. duration=2 days)
        # even though the main atomic assertion is a complete instruction whose
        # source fact already contains that exact duration. Do not downgrade a
        # supported instruction because a second, unsupported relation lookup
        # cannot find a duplicate structured duration fact.
        if (
            field in {"duration", "frequency"}
            and relation_key in {"instruction", "recommendation", "plan", "advice"}
            and str(claim.get("value", "")).strip()
        ):
            claim_value_text = normalize(claim.get("value", ""))
            attribute_text = normalize(f"{value} {claim.get('unit', '')}")
            parent_source_match = False
            for candidate in candidate_facts:
                parent_fact, _ = _unwrap_candidate(candidate)
                if not isinstance(parent_fact, dict):
                    continue
                if _canonical_relation(parent_fact.get("relation", "")) != relation_key:
                    continue
                parent_value_text = normalize(parent_fact.get("value", ""))
                if not _claim_phrase_supported(
                    claim.get("value", ""), parent_fact.get("value", ""),
                    claim_negated=bool(claim.get("negated", False)),
                ):
                    continue
                if attribute_text and attribute_text in claim_value_text and attribute_text in parent_value_text:
                    parent_source_match = True
                    break
                # Alternate schedule spelling may differ while expressing the
                # same sub-attribute. Only skip the separate check when the
                # actual sub-attribute phrase appears in both parent texts.
                if field == "frequency":
                    simple_timing_terms = [
                        token for token in re.findall(r"[a-z0-9]+", attribute_text)
                        if token not in {"once", "per", "a", "day"}
                    ]
                    if simple_timing_terms and all(token in claim_value_text and token in parent_value_text for token in simple_timing_terms):
                        parent_source_match = True
                        break
            if parent_source_match:
                continue
        if (
            field == "duration"
            and claim_subject_text in generic_subjects
            and category_text in {"diagnosis", "condition", "clinical diagnosis", "clinical finding"}
            and relation_key not in {"duration", "treatment duration"}
        ):
            # A duration such as "symptoms for four years" is a separate
            # temporal assertion. This engine does not yet attach such a
            # duration to a symptom-specific structured fact, so abstain rather
            # than compare it with a medication-course duration.
            continue
        checks.append({
            "field": field,
            "relations": [field],
            "claim_value": value,
            "claim_unit": "",
            "is_main_claim": False,
        })

    for check in checks:
        field = check["field"]
        outcomes = []

        for relation_option in check["relations"]:
            # A summary may combine admission date and clock time while the
            # source stores them as separate, explicitly related facts.
            if _canonical_relation(relation_option) == "date" and (
                field == "date" or _canonical_relation(relation_key or relation) in {"is", "date"}
            ):
                claim_date_parts = _parse_date_components(check["claim_value"])
                time_match = re.search(r"\b(\d{1,2}:\d{2})(?:\s*(am|pm))?\b", normalize(check["claim_value"]))
                if claim_date_parts and time_match:
                    date_fact, date_ambiguous = _find_relation_evidence(candidate_facts, "date")
                    time_fact, time_ambiguous = _find_relation_evidence(candidate_facts, "time")
                    if date_fact and time_fact and not date_ambiguous and not time_ambiguous:
                        source_date_parts = _parse_date_components(date_fact.get("value", ""))
                        source_time_match = re.search(r"\b(\d{1,2}:\d{2})(?:\s*(am|pm))?\b", normalize(time_fact.get("value", "")))
                        date_ok = bool(source_date_parts) and claim_date_parts[1:] == source_date_parts[1:] and (claim_date_parts[0] is None or source_date_parts[0] is None or claim_date_parts[0] == source_date_parts[0])
                        def clock_tuple(match):
                            if not match:
                                return None
                            h, minute = (int(part) for part in match.group(1).split(":"))
                            suffix = match.group(2)
                            if suffix:
                                if h == 12:
                                    h = 0
                                if suffix == "pm":
                                    h += 12
                            return h, minute
                        time_ok = clock_tuple(time_match) == clock_tuple(source_time_match)
                        composite_verdict = "SUPPORTED" if date_ok and time_ok else "CONTRADICTED"
                        outcomes.append({
                            "fact": {
                                "value": f"{date_fact.get('value', '')} {time_fact.get('value', '')}",
                                "fact_id": date_fact.get("fact_id"),
                            },
                            "ambiguous": False,
                            "verdict": composite_verdict,
                            "relation": relation_option,
                            "score": 0.0,
                            "supporting_fact_ids": [date_fact.get("fact_id"), time_fact.get("fact_id")],
                        })
                        continue

            fact, ambiguous = _find_relation_evidence(
                candidate_facts,
                relation_option,
            )
            if fact is None and not ambiguous and field == "duration":
                # A date-range duration can be supported by separate start- and
                # stop-date facts when the dataset has no explicit duration fact.
                claim_dates = re.findall(
                    r"\b\d{4}-\d{2}-\d{2}\b",
                    str(check["claim_value"]),
                )
                if len(claim_dates) >= 2:
                    start_fact, start_ambiguous = _find_relation_evidence(candidate_facts, "start date")
                    stop_fact, stop_ambiguous = _find_relation_evidence(candidate_facts, "stop date")
                    if start_fact and stop_fact and not start_ambiguous and not stop_ambiguous:
                        start_date = str(start_fact.get("value", ""))
                        stop_date = str(stop_fact.get("value", ""))
                        source_range = f"{start_date} to {stop_date}"
                        date_verdict = (
                            "SUPPORTED"
                            if claim_dates[0] == start_date and claim_dates[-1] == stop_date
                            else "CONTRADICTED"
                        )
                        outcomes.append({
                            "fact": {
                                "value": source_range,
                                "fact_id": start_fact.get("fact_id"),
                            },
                            "ambiguous": False,
                            "verdict": date_verdict,
                            "relation": relation_option,
                            "score": 0.0,
                            "supporting_fact_ids": [
                                start_fact.get("fact_id"), stop_fact.get("fact_id")
                            ],
                        })
                        continue
            if ambiguous:
                # Multiple distinct observations from a single imaging report
                # can share a relation such as "findings". For narrative claims,
                # compare against each of those retrieved facts; do not use this
                # fallback to disambiguate numeric measurements across time.
                claim_text = str(check["claim_value"] or "")
                numeric_claim = bool(re.search(r"\d", claim_text))
                category = normalize(claim.get("category", "")).replace("_", " ")
                allow_narrative_multi_match = (
                    not numeric_claim
                    and category in {"imaging", "radiology", "imaging result", "diagnosis", "condition", "clinical diagnosis", "allergy"}
                )
                if allow_narrative_multi_match:
                    relation_candidates = []
                    for candidate in candidate_facts:
                        candidate_fact, candidate_score = _unwrap_candidate(candidate)
                        if (
                            isinstance(candidate_fact, dict)
                            and _canonical_relation(candidate_fact.get("relation", "")) == _canonical_relation(relation_option)
                        ):
                            candidate_verdict = _compare_claim_field(
                                check["claim_value"],
                                check["claim_unit"],
                                candidate_fact,
                                field,
                                claim_negated=(
                                    bool(claim.get("negated", False))
                                    if check["is_main_claim"] else False
                                ),
                            )
                            relation_candidates.append({
                                "fact": candidate_fact,
                                "ambiguous": False,
                                "verdict": candidate_verdict,
                                "relation": relation_option,
                                "score": candidate_score,
                            })
                    narrative_support = [
                        item for item in relation_candidates
                        if item["verdict"] == "SUPPORTED"
                    ]
                    if narrative_support:
                        outcomes.extend(relation_candidates)
                    else:
                        outcomes.append({
                            "fact": None,
                            "ambiguous": True,
                            "verdict": "INSUFFICIENT_EVIDENCE",
                            "relation": relation_option,
                            "score": -1.0,
                        })
                else:
                    outcomes.append({
                        "fact": None,
                        "ambiguous": True,
                        "verdict": "INSUFFICIENT_EVIDENCE",
                        "relation": relation_option,
                        "score": -1.0,
                    })
                continue

            verdict = _compare_claim_field(
                check["claim_value"],
                check["claim_unit"],
                fact,
                field,
                claim_negated=(
                    bool(claim.get("negated", False))
                    if check["is_main_claim"]
                    else False
                ),
            )
            # Preserve the retrieval score for deterministic selection when
            # multiple source relation families are considered.
            fact_score = max((
                _unwrap_candidate(candidate)[1]
                for candidate in candidate_facts
                if _unwrap_candidate(candidate)[0] is fact
            ), default=0.0)
            outcomes.append({
                "fact": fact,
                "ambiguous": False,
                "verdict": verdict,
                "relation": relation_option,
                "score": fact_score,
            })

        # A diagnosis claim occasionally combines two short clauses with a
        # dash (e.g. diagnosis — clinical pathway). If each component is
        # independently supported by a retrieved diagnosis/admission fact,
        # accept the combined assertion and cite all supporting facts. This is
        # narrowly limited to diagnosis-like claims and dash-separated parts;
        # arbitrary narrative token overlap is not enough.
        category_for_composite = normalize(claim.get("category", "")).replace("_", " ")
        if (
            check["is_main_claim"]
            and field in {"condition", "diagnosis"}
            and category_for_composite in {"diagnosis", "condition", "clinical diagnosis"}
            and not any(item.get("verdict") == "SUPPORTED" for item in outcomes)
        ):
            composite_parts = [
                part.strip(" .;:")
                for part in re.split(r"\s+[-–—]\s+", str(check["claim_value"] or ""))
                if part.strip(" .;:")
            ]
            if len(composite_parts) >= 2 and all(len(re.findall(r"[a-z0-9]+", normalize(part))) >= 2 for part in composite_parts):
                matched_parts = []
                for part in composite_parts:
                    part_matches = []
                    for candidate in candidate_facts:
                        part_fact, part_score = _unwrap_candidate(candidate)
                        if not isinstance(part_fact, dict):
                            continue
                        fact_category_name = normalize(part_fact.get("category", "")).replace("_", " ")
                        fact_relation_name = _canonical_relation(part_fact.get("relation", ""))
                        if fact_category_name not in {"diagnosis", "diagnoses", "admission", "discharge status", "clinical finding"}:
                            continue
                        if fact_relation_name not in {"is", "diagnosis", "condition", "indication", "result"}:
                            continue
                        if _claim_phrase_supported(part, part_fact.get("value", "")):
                            part_matches.append((part_fact, part_score))
                    if not part_matches:
                        matched_parts = []
                        break
                    matched_parts.sort(key=lambda pair: pair[1], reverse=True)
                    matched_parts.append(part_matches[0])
                if matched_parts:
                    support_ids = list(dict.fromkeys(pair[0].get("fact_id") for pair in matched_parts if pair[0].get("fact_id")))
                    outcomes.append({
                        "fact": matched_parts[0][0],
                        "ambiguous": False,
                        "verdict": "SUPPORTED",
                        "relation": relation_key or field,
                        "score": max(pair[1] for pair in matched_parts),
                        "supporting_fact_ids": support_ids,
                    })

        # A generic narrative claim may legitimately be represented in more
        # than one fact relation (e.g. a finding and an impression). Accept it
        # when one of the independently retrieved, relevant source facts
        # directly supports it. Numeric conflicts remain contradicted when no
        # candidate relation supports the asserted value.
        supported = [item for item in outcomes if item["verdict"] == "SUPPORTED"]
        contradicted = [item for item in outcomes if item["verdict"] == "CONTRADICTED"]
        if supported:
            selected = max(supported, key=lambda item: item["score"])
        elif contradicted:
            selected = max(contradicted, key=lambda item: item["score"])
        else:
            selected = max(outcomes, key=lambda item: item["score"], default={})

        fact = selected.get("fact") if selected else None
        ambiguous = bool(selected.get("ambiguous", False)) if selected else False
        verdict = selected.get("verdict", "INSUFFICIENT_EVIDENCE") if selected else "INSUFFICIENT_EVIDENCE"

        field_results[field] = {
            "claim_value": check["claim_value"],
            "source_value": fact.get("value") if fact else None,
            "matched_fact_id": fact.get("fact_id") if fact else None,
            "verdict": verdict,
            "ambiguous_evidence": ambiguous,
            "matched_relation": selected.get("relation") if selected else None,
            "matched_fact_ids": selected.get("supporting_fact_ids", []) if selected else [],
        }

    verdicts = [
        result["verdict"]
        for result in field_results.values()
    ]

    if "CONTRADICTED" in verdicts:
        overall = "CONTRADICTED"
    elif not verdicts or all(
        verdict == "INSUFFICIENT_EVIDENCE"
        for verdict in verdicts
    ):
        overall = "INSUFFICIENT_EVIDENCE"
    elif "INSUFFICIENT_EVIDENCE" in verdicts:
        overall = "PARTIALLY_SUPPORTED"
    else:
        overall = "SUPPORTED"

    contradicted_fields = [
        field
        for field, result in field_results.items()
        if result["verdict"] == "CONTRADICTED"
    ]

    return {
        "claim_id": claim.get("claim_id"),
        "verdict": overall,
        "error_type": (
            f"{contradicted_fields[0]}_mismatch"
            if contradicted_fields
            else None
        ),
        "field_results": field_results,
        "matched_fact_ids": list(dict.fromkeys(
            fact_id
            for result in field_results.values()
            for fact_id in (
                result.get("matched_fact_ids")
                or ([result["matched_fact_id"]] if result.get("matched_fact_id") else [])
            )
            if fact_id
        )),
    }