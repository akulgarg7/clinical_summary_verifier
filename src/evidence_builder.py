import json
import re
from functools import lru_cache
from pathlib import Path


GROUND_TRUTH_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "patient_records"
    / "synthetic_discharge_ground_truth.json"
)


# Generic subjects do not identify a specific clinical entity.
_GENERIC_SUBJECTS = {
    "patient",
    "person",
    "individual",
    "subject",
}

_RETRIEVAL_STOPWORDS = {
    "a", "an", "and", "the", "of", "for", "to", "in",
    "on", "at", "by", "with", "is", "was", "were",
    "be", "been", "being", "patient", "blood", "cell", "count",
    # Common headings/qualifiers are not meaningful clinical entities.
    "primary", "secondary", "details", "specialty",
    # Route, dose, and regimen metadata are not drug identity tokens.
    "iv", "intravenous", "oral", "po", "subcutaneous", "sc",
    "intramuscular", "im", "topical", "inhaled", "inhalation",
    "fluid", "fluids", "infusion", "inpatient", "outpatient",
    "medication", "medications", "drug", "drugs", "dose", "dosing",
    "unit", "units", "mg", "mcg", "g", "ml", "mmol", "meq",
    "litre", "litres", "liter", "liters", "per", "hour", "hours",
    "day", "days", "from", "until", "started", "stopped", "ongoing",
}


def normalize_text(value):
    """Normalize text for comparison and retrieval."""
    return " ".join(
        re.findall(r"[a-z0-9]+", str(value or "").lower())
    )


@lru_cache(maxsize=50000)
def normalize_clinical_entity(value):
    """Normalize common clinical entity aliases for retrieval only.

    This is deliberately separate from normalize_text(): punctuation such as
    the plus sign in K+ is clinically meaningful for identifying an analyte,
    but should not be used as an unbounded substring in ordinary text.
    """
    raw = str(value or "").strip().casefold()
    raw = raw.replace("⁺", "+").replace("⁻", "-").replace("−", "-")
    raw = raw.replace("₂", "2").replace("₃", "3")
    raw = raw.replace("α", "alpha").replace("β", "beta")

    aliases = [
        (r"^(?:k\s*\+|potassium|serum potassium)$", "potassium"),
        (r"^(?:na\s*\+|sodium|serum sodium)$", "sodium"),
        (r"^(?:cl\s*-|chloride|serum chloride)$", "chloride"),
        (r"^(?:ca\s*2\s*\+|calcium|serum calcium)$", "calcium"),
        (r"^(?:mg\s*2\s*\+|magnesium|serum magnesium)$", "magnesium"),
        (r"^(?:hb|haemoglobin|hemoglobin)$", "hemoglobin"),
        (r"^(?:serum\s+)?creatinine(?:\s*\([^)]*\))?$", "creatinine"),
        (r"^(?:serum\s+)?urea(?:\s+nitrogen)?$", "urea"),
        (r"^(?:wbc|white blood cells?|white blood cell count)$", "white blood cell count"),
        (r"^(?:rbc|red blood cells?|red blood cell count)$", "red blood cell count"),
        (r"^(?:platelets?|platelet count)$", "platelet count"),
        (r"^(?:crp|c reactive protein|c reactive protein level)$", "c reactive protein"),
        (r"^(?:tsh|thyroid stimulating hormone|thyrotropin)$", "thyroid stimulating hormone"),
        (r"^(?:hr|heart rate)$", "heart rate"),
        (r"^(?:rr|respiratory rate)$", "respiratory rate"),
        (r"^(?:bp|blood pressure)$", "blood pressure"),
        (r"^(?:map|mean arterial pressure)$", "mean arterial pressure"),
        (r"^(?:spo2|oxygen saturation|o2 saturation)$", "oxygen saturation"),
        (r"^(?:cxr|chest x ray|chest xray)$", "chest xray"),
        (r"^(?:renal us|renal ultrasound)$", "renal ultrasound"),
        (r"^(?:temp|temperature)$", "temperature"),
        (r"^(?:ecg|ekg|(?:12[- ]lead )?electrocardiogram)$", "electrocardiogram"),
    ]

    collapsed = " ".join(raw.split())
    for pattern, canonical in aliases:
        if re.fullmatch(pattern, collapsed):
            return canonical

    # Replace common aliases within labels while retaining useful context such
    # as "peak" or "pre-discharge" for temporal ranking.
    replacements = [
        (r"\bnacl\b", "sodium chloride"),
        (r"\bnormal saline\b", "sodium chloride 0.9%"),
        (r"(?<![a-z0-9])k\s*\+(?![a-z0-9])", "potassium"),
        (r"(?<![a-z0-9])na\s*\+(?![a-z0-9])", "sodium"),
        (r"(?<![a-z0-9])cl\s*-(?![a-z0-9])", "chloride"),
        (r"(?<![a-z0-9])ca\s*2\s*\+(?![a-z0-9])", "calcium"),
        (r"(?<![a-z0-9])mg\s*2\s*\+(?![a-z0-9])", "magnesium"),
        (r"\bwbc\b|\bwhite blood cells?\b", "white blood cell count"),
        (r"\brbc\b|\bred blood cells?\b", "red blood cell count"),
        (r"\bplatelets?\b", "platelet count"),
        (r"\bcrp\b|\bc-reactive protein\b", "c reactive protein"),
        (r"\btsh\b|\bthyrotropin\b", "thyroid stimulating hormone"),
        (r"\becg\b|\bekg\b", "electrocardiogram"),
        (r"\bhr\b", "heart rate"),
        (r"\brr\b", "respiratory rate"),
        (r"\bbp\b", "blood pressure"),
        (r"\bmap\b", "mean arterial pressure"),
        (r"\bspo2\b|\bo2 saturation\b", "oxygen saturation"),
        (r"\bcxr\b|\bchest[- ]x[- ]?ray\b", "chest xray"),
        (r"\brenal us\b", "renal ultrasound"),
        (r"\btemp\b", "temperature"),
        (r"\bhb\b|\bhaemoglobin\b", "hemoglobin"),
        (r"\bserum\s+creatinine\b", "creatinine"),
    ]
    for pattern, replacement in replacements:
        collapsed = re.sub(pattern, replacement, collapsed)

    return normalize_text(collapsed)



def _canonical_retrieval_relation(value):
    """Normalize equivalent extractor/source relation labels for retrieval."""
    relation = normalize_text(value)
    aliases = {
        "symptom duration": "duration",
        "duration of symptoms": "duration",
        "duration of symptom": "duration",
        "presence": "status",
        "allergy presence": "status",
        "allergy status": "status",
        "condition and disposition": "condition",
        "clinical condition": "condition",
        "clinical status": "status",
        "status relative to pre admission regimen": "status",
    }
    return aliases.get(relation, relation)


def _relation_slot_fallback(claim, fact):
    """Match relation-compatible schema slots with non-entity heading subjects.

    Some facts use a field label such as 'Chief complaint' as the subject while
    the claim's subject names the symptom. This fallback is deliberately narrow:
    it requires an alias-compatible relation, category compatibility, and a
    known schema-heading subject. It never uses the asserted value to retrieve
    evidence.
    """
    claim_relation = _canonical_retrieval_relation(claim.get("relation", ""))
    fact_relation = _canonical_retrieval_relation(fact.get("relation", ""))
    claim_category = normalize_text(claim.get("category", ""))
    fact_category = normalize_text(fact.get("category", ""))
    fact_subject = normalize_text(fact.get("subject", ""))
    broad_subjects = {
        "chief complaint", "presenting complaint", "clinical status",
        "discharge status", "discharge condition", "allergies", "drug allergies",
        "admission status", "initial condition",
    }
    if fact_subject not in broad_subjects:
        return False, 0.0, ""
    if claim_relation == fact_relation == "duration":
        claim_is_symptom = any(token in claim_category for token in ("clinical finding", "symptom", "diagnosis", "condition"))
        fact_is_history = fact_category in {"admission", "history", "clinical finding", "symptom"}
        if claim_is_symptom and fact_is_history:
            return True, 2.5, "compatible symptom-duration slot"
    if claim_relation == fact_relation == "status" and "allergy" in claim_category and fact_category == "allergy":
        return True, 2.5, "compatible allergy-status slot"
    if claim_relation == fact_relation == "condition" and claim_category in {"clinical finding", "clinical findings", "condition", "diagnosis"}:
        return True, 2.5, "compatible clinical-condition slot"
    return False, 0.0, ""

def _extract_temporal_dates(*values):
    """Extract normalized dates and month-year references from text."""
    dates = set()

    month_numbers = {
        "january": "01", "jan": "01",
        "february": "02", "feb": "02",
        "march": "03", "mar": "03",
        "april": "04", "apr": "04",
        "may": "05",
        "june": "06", "jun": "06",
        "july": "07", "jul": "07",
        "august": "08", "aug": "08",
        "september": "09", "sep": "09", "sept": "09",
        "october": "10", "oct": "10",
        "november": "11", "nov": "11",
        "december": "12", "dec": "12",
    }

    for value in values:
        text = str(value or "")

        dates.update(
            re.findall(r"\b\d{4}-\d{2}(?:-\d{2})?\b", text)
        )

        # Normalize unambiguous slash dates. Ambiguous forms such as 04/05/2025
        # are deliberately skipped because their date convention is unknown.
        for first, second, year in re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text):
            first, second = int(first), int(second)
            if first > 12 and 1 <= second <= 12:
                dates.add(f"{year}-{second:02d}-{first:02d}")
            elif second > 12 and 1 <= first <= 12:
                dates.add(f"{year}-{first:02d}-{second:02d}")
        # Yearless numeric dates are only normalized when DD/MM vs MM/DD is
        # unambiguous (e.g. 14/02); ambiguous 04/05 is not guessed.
        for first, second in re.findall(r"\b(\d{1,2})/(\d{1,2})\b", text):
            first, second = int(first), int(second)
            if first > 12 and 1 <= second <= 12:
                dates.add(f"--{second:02d}-{first:02d}")
            elif second > 12 and 1 <= first <= 12:
                dates.add(f"--{first:02d}-{second:02d}")

        for month, year in re.findall(
            r"\b([a-zA-Z]+)\s+(\d{4})\b",
            text,
        ):
            month_number = month_numbers.get(month.lower())

            if month_number:
                dates.add(f"{year}-{month_number}")

        # Day/month references without a year are kept as partial dates so they
        # can match an ISO date only on the same month and day.
        for day, month in re.findall(r"\b(\d{1,2})\s+([a-zA-Z]+)\b", text):
            month_number = month_numbers.get(month.lower())
            if month_number and 1 <= int(day) <= 31:
                dates.add(f"--{month_number}-{int(day):02d}")
        for month, day in re.findall(r"\b([a-zA-Z]+)\s+(\d{1,2})\b", text):
            month_number = month_numbers.get(month.lower())
            if month_number and 1 <= int(day) <= 31:
                dates.add(f"--{month_number}-{int(day):02d}")

    return sorted(dates)


def resolve_claim_temporal_context(claim, record):
    """Resolve only the temporal context local to the assertion being checked.

    A source sentence can contain multiple measurements at different times
    (e.g. admission creatinine 1.0 followed by post-contrast creatinine 1.1).
    Looking at the entire sentence for a marker incorrectly assigns the first
    measurement's context to the second. This function isolates the clause
    around the asserted value before interpreting baseline/admission/peak/etc.
    """
    date_time = str(claim.get("date_time", "") or "")
    source_text = str(claim.get("source_text", "") or "")
    category = str(claim.get("category", "") or "")
    relation = str(claim.get("relation", "") or "")
    claim_value = str(claim.get("value", "") or "").strip()

    # Retain punctuation here so clauses separated by commas/semicolons/arrows
    # remain distinct while identifying the context surrounding the value.
    raw_source = source_text.replace("–", "-").replace("—", "-")
    target_segment = raw_source
    value_match = None
    if claim_value:
        try:
            value_pattern = re.escape(claim_value)
            value_match = re.search(
                rf"(?<![a-zA-Z0-9]){value_pattern}(?![a-zA-Z0-9])",
                raw_source,
                flags=re.IGNORECASE,
            )
        except re.error:
            value_match = None
    if value_match:
        start, end = value_match.span()
        delimiters = [",", ";", "→", "->", "\n"]
        left_positions = [raw_source.rfind(token, 0, start) for token in delimiters]
        left = max(left_positions, default=-1) + 1
        right_candidates = []
        for token in delimiters:
            idx = raw_source.find(token, end)
            if idx >= 0:
                right_candidates.append(idx)
        right = min(right_candidates) if right_candidates else len(raw_source)
        segment = raw_source[left:right].strip()
        if segment:
            target_segment = segment

    normalized_date = normalize_text(date_time)
    normalized_source = normalize_text(source_text)
    normalized_context = normalize_text(f"{date_time} {source_text} {category} {relation}")
    target_context = normalize_text(f"{date_time} {target_segment} {category} {relation}")
    dates = set(_extract_temporal_dates(date_time, target_segment))

    markers = []
    mentions = []

    pre_admission_reference = bool(re.search(
        r"\b(pre admission|before admission|prior to admission)\b",
        target_context,
    ))
    admission_reference = bool(re.search(
        r"\b(admission|on presentation|at presentation)\b",
        target_context,
    ))

    if pre_admission_reference:
        markers.append("pre_admission")
    elif admission_reference:
        markers.append("admission")
        admission_date = record.get("admission", {}).get("admission_date")
        if admission_date:
            dates.add(str(admission_date))

    if re.search(r"\bbaseline\b", target_context):
        markers.append("baseline")
    if re.search(r"\bpeak(?:ed)?\b", target_context):
        markers.append("peak")

    pre_discharge_reference = bool(re.search(r"\bpre discharge\b", target_context))
    discharge_reference = bool(re.search(r"\b(discharge|discharged)\b", target_context))
    if pre_discharge_reference:
        markers.append("pre_discharge")
    elif discharge_reference:
        markers.append("discharge")
        timeline = record.get("timeline", [])
        discharge_dates = [
            str(event.get("date")) for event in timeline
            if event.get("event_type") == "discharge" and event.get("date")
        ]
        if len(discharge_dates) == 1:
            dates.add(discharge_dates[0])

    for marker in ("baseline", "peak", "admission", "discharge"):
        if re.search(rf"\b{re.escape(marker)}\b", normalized_context):
            mentions.append(marker)

    return {
        "dates": sorted(dates),
        "markers": markers,
        "mentions": mentions,
        "target_text": target_segment,
    }


def retrieval_tokens(value):
    """Return informative tokens for entity matching."""
    return {
        token
        for token in re.findall(
            r"[a-z0-9]+",
            str(value or "").lower(),
        )
        if token not in _RETRIEVAL_STOPWORDS and not token.isdigit()
    }


def load_ground_truth(file_path=GROUND_TRUTH_PATH):
    """Load the synthetic patient records."""

    with open(file_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "Ground-truth JSON must contain a list of records."
        )

    return records


def build_evidence(records):
    """
    Index ground-truth facts by patient case and fact ID.
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
            },
        }

    return evidence


def match_subject(claim_subject, fact_subject, fact_value):
    """Match a clinical entity without unsafe substring matches.

    Common analyte aliases are canonicalized first. Other subjects use
    token-boundary phrase matching and token overlap, so "K" cannot match the
    final character of "peak" or a different entity mentioned in a value.
    """
    subject = normalize_clinical_entity(claim_subject)
    fact_subject_text = normalize_clinical_entity(fact_subject)
    # Normalize clinical aliases in values as well as subjects, so e.g. a
    # claim for blood pressure can retrieve a hospital-course fact written as
    # "BP 112/70" without relying on its broad event subject.
    fact_value_text = normalize_text(normalize_clinical_entity(fact_value))

    if not subject or subject in _GENERIC_SUBJECTS:
        return True, 0.0, "generic or missing subject"

    # Do not allow overlap on generic words to conflate distinct structured
    # clinical concepts (e.g. BP vs white blood cell count, or WBC vs RBC).
    specific_concepts = {
        "potassium", "sodium", "chloride", "calcium", "magnesium",
        "hemoglobin", "creatinine", "urea", "white blood cell count",
        "red blood cell count", "platelet count", "c reactive protein",
        "thyroid stimulating hormone", "heart rate", "respiratory rate",
        "blood pressure", "mean arterial pressure", "oxygen saturation",
        "chest xray", "renal ultrasound", "temperature", "electrocardiogram",
    }
    if subject in specific_concepts and fact_subject_text in specific_concepts and subject != fact_subject_text:
        return False, 0.0, "different specific clinical concepts"

    # Token-boundary phrase matches only; never use `subject in text` without
    # boundaries (e.g., potassium token K must not match creatinine (peak)).
    if subject == fact_subject_text or re.search(
        rf"(?<![a-z0-9]){re.escape(subject)}(?![a-z0-9])",
        fact_subject_text,
    ):
        return True, 5.0, "subject matches fact subject"

    if fact_subject_text and re.search(
        rf"(?<![a-z0-9]){re.escape(fact_subject_text)}(?![a-z0-9])",
        subject,
    ):
        return True, 4.0, "fact subject is contained in claim subject"

    subject_tokens = retrieval_tokens(subject)
    fact_subject_tokens = retrieval_tokens(fact_subject_text)

    if subject_tokens and fact_subject_tokens:
        overlap = subject_tokens & fact_subject_tokens
        coverage = len(overlap) / len(subject_tokens)

        # Require at least half of the informative subject tokens, but use
        # complete-token equality rather than raw character substring checks.
        if overlap and coverage >= 0.5:
            return (
                True,
                3.0 * coverage,
                "partial subject-token overlap",
            )

    is_general_event = (
        not fact_subject_tokens
        or fact_subject_text in _GENERIC_SUBJECTS
        or "event" in fact_subject_tokens
    )
    is_general_observation = any(
        marker in fact_subject_text
        for marker in (
            "physical examination", "vital sign", "vitals",
            "initial condition", "clinical status", "hospital course event",
            "laboratory", "investigation",
        )
    )

    if is_general_event or is_general_observation:
        if re.search(
            rf"(?<![a-z0-9]){re.escape(subject)}(?![a-z0-9])",
            fact_value_text,
        ):
            return True, 4.5, "subject phrase mentioned in clinical event"

        if subject_tokens:
            fact_value_tokens = retrieval_tokens(fact_value_text)
            overlap = subject_tokens & fact_value_tokens
            coverage = len(overlap) / len(subject_tokens)
            if overlap and coverage >= 0.6:
                return (
                    True,
                    3.0 * coverage,
                    "subject mentioned in clinical event",
                )

    return False, 0.0, "subject does not match"

def score_assertion_content(claim, fact):
    """
    Score lexical relevance between a claim's assertion and a fact.

    Used as a retrieval fallback when the claim has a generic subject.
    Numeric values and common measurement units are excluded so that
    incorrect numerical values do not determine which facts are found.
    """
    claim_subject = normalize_text(claim.get("subject", ""))

    if claim_subject not in _GENERIC_SUBJECTS:
        return 0.0

    assertion = " ".join([
        str(claim.get("value", "") or ""),
        str(claim.get("source_text", "") or ""),
    ])

    ignored_tokens = {
        "mg", "g", "mcg", "units", "unit", "iu",
        "ml", "dl", "mmhg", "bpm", "kg", "cm",
        "day", "days", "daily", "hour", "hours",
    }

    claim_tokens = {
        token
        for token in retrieval_tokens(assertion)
        if not token.isdigit()
        and not any(char.isdigit() for char in token)
        and token not in ignored_tokens
    }

    fact_text = " ".join([
        str(fact.get("subject", "") or ""),
        str(fact.get("relation", "") or ""),
        str(fact.get("value", "") or ""),
    ])

    fact_tokens = {
        token
        for token in retrieval_tokens(fact_text)
        if not token.isdigit()
        and not any(char.isdigit() for char in token)
        and token not in ignored_tokens
    }

    if not claim_tokens:
        return 0.0

    overlap = claim_tokens & fact_tokens

    if not overlap:
        return 0.0

    # Reward strong coverage while still allowing a claim to be
    # supported by a fact that expresses only part of the assertion.
    coverage = len(overlap) / len(claim_tokens)

    return round(6.0 * coverage, 3)


def retrieve_candidate_facts(
    claim,
    facts,
    top_k=10,
    record=None,
):
    """
    Retrieve and rank relevant ground-truth facts.

    Retrieval uses the claim's entity, category, relation,
    clinical context, and date where available.

    IMPORTANT:
    The claim's asserted value, unit, dose, and frequency
    are not used to select matching evidence. They should
    be compared after evidence retrieval.

    Returns:
        [
            {
                "fact": original_ground_truth_fact,
                "score": relevance_score,
                "match_reasons": [...]
            }
        ]
    """

    if top_k < 1:
        return []

    claim_subject = normalize_clinical_entity(claim.get("subject", ""))
    claim_category = normalize_text(claim.get("category", ""))
    claim_relation = normalize_text(claim.get("relation", ""))

    # Use summary wording and category for contextual hints.
    # Do not use the asserted clinical value.
    claim_context = normalize_text(
        f"{claim.get('category', '')} "
        f"{claim.get('relation', '')} "
        f"{claim.get('date_time', '')} "
        f"{claim.get('source_text', '')}"
    )

    wants_discharge = "discharge" in claim_context

    wants_pre_admission = any(
        phrase in claim_context
        for phrase in (
            "pre admission",
            "before admission",
            "home medication",
            "medications on admission",
            "medication on admission",
            "meds on admission",
            "taken on admission",
            "home regimen",
        )
    )

    wants_hospital_context = any(
        phrase in claim_context
        for phrase in (
            "hospital course",
            "during hospitalization",
            "during hospitalisation",
            "inpatient",
            "hospitalisation",
            "hospitalization",
        )
    )

    temporal_context = (
        resolve_claim_temporal_context(claim, record)
        if record is not None
        else {
            "dates": sorted(_extract_temporal_dates(claim.get("date_time", ""), claim.get("source_text", ""))),
            "markers": [],
            "mentions": [],
            "target_text": claim.get("source_text", ""),
        }
    )
    claim_dates = set(_extract_temporal_dates(
        claim.get("date_time", ""), temporal_context.get("target_text", "")
    ))
    # Only dates explicitly present in the claim/date_time participate in
    # date relevance scoring. `temporal_context["dates"]` may also contain a
    # date inferred from the record (for example, the admission date merely
    # because the subject says "Admission Type"). Using that inferred date as
    # an asserted date incorrectly boosts unrelated admission facts.
    # Avoid letting a redundant month-year token (e.g. 2025-03) make a
    # different day in the same month look temporally compatible. Keep the
    # month-only token only when no full date for that same month is available.
    full_claim_dates = {
        item for item in claim_dates
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", item)
    }
    if full_claim_dates:
        claim_dates = {
            item for item in claim_dates
            if not (
                re.fullmatch(r"\d{4}-\d{2}", item)
                and any(full.startswith(item + "-") for full in full_claim_dates)
            )
        }
    # A written full date may be parsed both as --MM-DD and YYYY-MM. When
    # an exact month/day exists, the month/year token is redundant and would
    # otherwise make a different day in that month seem compatible.
    partial_claim_months = {
        item[2:4] for item in claim_dates
        if re.fullmatch(r"--\d{2}-\d{2}", item)
    }
    if partial_claim_months:
        claim_dates = {
            item for item in claim_dates
            if not (
                re.fullmatch(r"\d{4}-\d{2}", item)
                and item[-2:] in partial_claim_months
            )
        }
    claim_markers = set(temporal_context["markers"])

    def lab_timepoint_markers(*values):
        """Extract explicit temporal sampling labels from a local lab clause."""
        text = normalize_text(" ".join(str(value or "") for value in values))
        markers = set()
        if re.search(r"\b(?:baseline|initial|at presentation|on presentation)\b", text):
            markers.add("initial")
        if re.search(r"\b(?:resolution|resolved|after treatment|post treatment|post-treatment|after correction|normalised|normalized)\b", text):
            markers.add("resolution")
        if re.search(r"\bpeak(?:ed)?\b", text):
            markers.add("peak")
        if re.search(r"\bpre discharge\b", text):
            markers.add("pre_discharge")
        elif re.search(r"\bdischarge(?:d)?\b", text):
            markers.add("discharge")
        # Hour-based labels are sampling timepoints when they appear in local
        # context such as (0 hour), (3 hours), or at 6 hours. Do not infer a
        # timepoint from numeric values or units alone.
        for match in re.finditer(r"\b(\d+)\s*hours?\b", text):
            start = max(0, match.start() - 18)
            prefix = text[start:match.start()]
            if re.search(r"(?:\(|\bat\s*$|\bafter\s*$|\btime\s*$|\bpost\s*$)", prefix) or match.group(1) in {"0", "1", "3", "6", "12", "24"}:
                markers.add(f"hour:{int(match.group(1))}")
        pod = re.search(r"\b(?:post operative|postoperative|pod)\s*(?:day\s*)?(\d+)\b", text)
        if pod:
            markers.add(f"postop_day:{int(pod.group(1))}")
        return markers

    is_lab_claim = (
        "laboratory" in claim_category
        or claim_category in {"lab", "lab result", "investigation", "laboratory result"}
    )
    claim_timepoint_markers = lab_timepoint_markers(
        temporal_context.get("target_text", ""), claim.get("date_time", "")
    ) if is_lab_claim else set()

    # Ground-truth facts representing one medication regimen share an exact
    # subject label. Use their associated start/stop/duration facts to date the
    # regimen as a group, so two sequential regimens of the same medication do
    # not compete solely on lexical similarity.
    facts_by_subject = {}
    for item in facts:
        key = str(item.get("subject", "") or "")
        facts_by_subject.setdefault(key, []).append(item)

    claim_relation_context = normalize_text(claim.get("relation", ""))
    start_action = bool(re.search(r"\b(started|start|initiated|commenced|resumed|restarted)\b", claim_relation_context))
    stop_action = bool(re.search(r"\b(stopped|stop|completed|complete|discontinued|weaned|ended|finished)\b", claim_relation_context))

    results = []

    for fact in facts:
        fact_subject = fact.get("subject", "")
        fact_value = fact.get("value", "")
        fact_relation = normalize_text(fact.get("relation", ""))
        fact_category = normalize_text(fact.get("category", ""))
        fact_section = normalize_text(fact.get("source_section", ""))

        fact_context = normalize_text(f"{fact_subject} {fact_section}")

        # First establish entity relevance. Unrelated specific entities
        # should not enter the ranking simply because category or
        # relation happens to match.
        subject_matched, subject_score, subject_reason = match_subject(
            claim_subject,
            fact_subject,
            fact_value,
        )
        if not subject_matched:
            slot_match, slot_score, slot_reason = _relation_slot_fallback(claim, fact)
            if slot_match:
                subject_matched, subject_score, subject_reason = True, slot_score, slot_reason

        if not subject_matched:
            # A historical procedure may be represented in a history
            # fact's value rather than its subject.
            claim_category_text = normalize_text(
                claim.get("category", "")
            )
            fact_category_text = normalize_text(
                fact.get("category", "")
            )
            fact_relation_text = normalize_text(
                fact.get("relation", "")
            )

            claim_entity = normalize_text(
                claim.get("subject", "")
            )
            fact_value_text = normalize_text(
                fact.get("value", "")
            )

            is_related_surgical_history = (
                "procedure" in claim_category_text
                and fact_category_text == "history"
                and fact_relation_text in {
                    "surgery",
                    "procedure",
                    "operation",
                }
            )

            if (
                is_related_surgical_history
                and claim_entity
                and claim_entity in fact_value_text
            ):
                subject_matched = True
                subject_score = 4.0
                subject_reason = (
                    "procedure matches historical surgery fact"
                )
            else:
                continue

        score = subject_score
        reasons = []

        # Attribute-slot matching: when a claim subject names an attribute
        # (e.g. "admission type"), rank facts whose normalized relation is that
        # attribute and whose entity phrase overlaps the claim. This ranking
        # signal does not inspect the asserted value.
        claim_subject_norm = normalize_text(claim.get("subject", ""))
        claim_subject_tokens = set(retrieval_tokens(claim_subject_norm))
        fact_subject_tokens = set(retrieval_tokens(fact_subject))
        fact_relation_slot = _canonical_retrieval_relation(fact_relation)
        slot_terms = {"type", "date", "time", "status", "age", "sex", "gender", "weight", "height", "route", "frequency", "duration"}
        matched_slots = (claim_subject_tokens & slot_terms) & ({fact_relation_slot} if fact_relation_slot else set())
        entity_overlap = (claim_subject_tokens - slot_terms) & fact_subject_tokens
        same_category_context = (
            "admission" in claim_subject_tokens
            and "admission" in fact_subject_tokens
        )
        if matched_slots and (entity_overlap or same_category_context):
            score += 7.0
            reasons.append("claim attribute slot matches source relation")

        if subject_reason != "generic or missing subject":
            reasons.append(subject_reason)

        if normalize_text(claim.get("subject", "")) in _GENERIC_SUBJECTS:
            content_score = score_assertion_content(claim, fact)

            if content_score > 0:
                score += content_score
                reasons.append(
                    "claim assertion overlaps fact content"
                )

        is_discharge = "discharge" in fact_context

        is_pre_admission = (
            "pre admission" in fact_context
            or "before admission" in fact_context
        )

        is_hospital_context = any(
            phrase in fact_context
            for phrase in (
                "hospital course",
                "during hospitalization",
                "during hospitalisation",
                "inpatient",
            )
        )

        # Context changes the ranking, but does not alone determine
        # whether a fact is valid evidence.
        if wants_discharge:
            if is_discharge:
                score += 8.0
                reasons.append("discharge context matches")
            elif is_pre_admission or is_hospital_context:
                score -= 4.0
                reasons.append("different medication context")

        elif wants_pre_admission:
            if is_pre_admission:
                score += 8.0
                reasons.append("pre-admission context matches")
            elif is_discharge or is_hospital_context:
                score -= 4.0
                reasons.append("different medication context")

        elif wants_hospital_context:
            if is_hospital_context:
                score += 6.0
            elif is_discharge or is_pre_admission:
                score -= 1.5
                reasons.append("different medication context")

        # Category is a relevance signal, not an absolute filter.
        if claim_category and claim_category == fact_category:
            score += 2.0
            reasons.append("category matches")

        # Prefer facts stating the same attribute or relation.
        fact_text = normalize_text(
            f"{fact_subject} {fact_value} {fact_relation}"
        )

        if claim_relation and _canonical_retrieval_relation(claim_relation) == _canonical_retrieval_relation(fact_relation):
            score += 1.5
            reasons.append("relation family matches")

        elif claim_relation and claim_relation in fact_text:
            score += 1.0
            reasons.append("relation appears in fact text")

        # Resolve medication regimen variants using a contextual marker from
        # the sentence and sibling facts within the same exact regimen group.
        # Example: "transition dose" appears in a frequency sibling fact, but
        # the dose itself is stored as a separate fact with the same subject.
        claim_context_markers = {
            "transition dose", "loading dose", "maintenance dose",
            "rescue dose", "stat dose", "one time dose", "one-time dose",
        }
        claim_marker_text = normalize_text(
            f"{claim.get('source_text', '')} {claim.get('relation', '')}"
        )
        markers_present = [marker for marker in claim_context_markers if marker in claim_marker_text]
        sibling_group_for_context = facts_by_subject.get(str(fact_subject or ""), [])
        sibling_context_text = normalize_text(" ".join(
            f"{sibling.get('relation', '')} {sibling.get('value', '')}"
            for sibling in sibling_group_for_context
        ))
        if markers_present and any(marker in sibling_context_text for marker in markers_present):
            score += 5.0
            reasons.append("medication regimen context marker matches sibling facts")

        # Compare claim and fact dates, including month-level references.
        fact_dates = set(
            _extract_temporal_dates(
                fact.get("date_time", ""),
                fact.get("value", ""),
                fact.get("subject", ""),
            )
        )
        sibling_group = facts_by_subject.get(str(fact_subject or ""), [])
        sibling_relations = set()
        sibling_start_dates = set()
        sibling_stop_dates = set()
        for sibling in sibling_group:
            sibling_relation = normalize_text(sibling.get("relation", ""))
            sibling_relations.add(sibling_relation)
            sibling_value = str(sibling.get("value", "") or "")
            if sibling_relation in {"start date", "started on", "date started"}:
                sibling_start_dates.update(_extract_temporal_dates(sibling_value, sibling.get("date_time", "")))
            elif sibling_relation in {"stop date", "stopped on", "discontinued on", "date stopped"}:
                sibling_stop_dates.update(_extract_temporal_dates(sibling_value, sibling.get("date_time", "")))
            elif sibling_relation == "duration":
                fact_dates.update(_extract_temporal_dates(sibling_value, sibling.get("date_time", "")))
        fact_dates.update(sibling_start_dates)
        fact_dates.update(sibling_stop_dates)
        # As with claim dates, month-year references parsed from a full source
        # date must not make another day in the same month appear compatible.
        full_fact_dates = {
            item for item in fact_dates
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", item)
        }
        if full_fact_dates:
            fact_dates = {
                item for item in fact_dates
                if not (
                    re.fullmatch(r"\d{4}-\d{2}", item)
                    and any(full.startswith(item + "-") for full in full_fact_dates)
                )
            }
        partial_fact_months = {
            item[2:4] for item in fact_dates
            if re.fullmatch(r"--\d{2}-\d{2}", item)
        }
        if partial_fact_months:
            fact_dates = {
                item for item in fact_dates
                if not (
                    re.fullmatch(r"\d{4}-\d{2}", item)
                    and item[-2:] in partial_fact_months
                )
            }

        def dates_compatible(left, right):
            """Match full dates and month/day or month/year references."""
            if left == right:
                return True
            if left.startswith("--") and len(left) == 7 and len(right) == 10:
                return left[2:] == right[5:]
            if right.startswith("--") and len(right) == 7 and len(left) == 10:
                return right[2:] == left[5:]

            if len(left) == 7 and len(right) == 10:
                return right.startswith(left + "-")

            if len(right) == 7 and len(left) == 10:
                return left.startswith(right + "-")

            return False

        if is_lab_claim and claim_timepoint_markers:
            fact_timepoint_markers = lab_timepoint_markers(
                fact_subject, fact_value, fact.get("date_time", ""), fact_section
            )
            if claim_timepoint_markers & fact_timepoint_markers:
                score += 5.0
                reasons.append("explicit laboratory timepoint matches")
            elif fact_timepoint_markers and not (claim_timepoint_markers & fact_timepoint_markers):
                # Penalize only an explicitly different timepoint. A fact with
                # no timepoint metadata remains eligible but cannot outrank a
                # fact with a matching marker solely on this signal.
                score -= 3.0
                reasons.append("explicit laboratory timepoint differs")

        if claim_dates and fact_dates:
            ordered_claim_dates = sorted(
                date for date in claim_dates
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)
            )
            relation_dates = claim_dates
            if len(ordered_claim_dates) >= 2 and fact_relation in {
                "start date", "started on", "date started"
            }:
                relation_dates = {ordered_claim_dates[0]}
            elif len(ordered_claim_dates) >= 2 and fact_relation in {
                "stop date", "stopped on", "discontinued on", "date stopped"
            }:
                relation_dates = {ordered_claim_dates[-1]}
            date_match = any(
                dates_compatible(claim_date, fact_date)
                for claim_date in relation_dates
                for fact_date in fact_dates
            )

            if date_match:
                score += 3.0
                reasons.append("temporal date matches")
            else:
                score -= 2.0
                reasons.append("temporal date differs")

        # Where a claim describes a medication being started or completed on a
        # date, a regimen's explicit start/stop-date facts are informative cues.
        # They rank otherwise similar regimens but never decide the verdict.
        if claim_dates and start_action and sibling_start_dates:
            if any(dates_compatible(cdate, fdate) for cdate in claim_dates for fdate in sibling_start_dates):
                score += 3.0
                reasons.append("matching regimen start date")
            elif sibling_start_dates and not any(dates_compatible(cdate, fdate) for cdate in claim_dates for fdate in sibling_stop_dates):
                score -= 1.0
        if claim_dates and stop_action and sibling_stop_dates:
            if any(dates_compatible(cdate, fdate) for cdate in claim_dates for fdate in sibling_stop_dates):
                score += 3.0
                reasons.append("matching regimen stop date")
            elif sibling_stop_dates and not any(dates_compatible(cdate, fdate) for cdate in claim_dates for fdate in sibling_start_dates):
                score -= 1.0
        # Return only facts that have some relevance signal.
        # Identify the temporal context represented by the fact.
        fact_subject_context = normalize_text(
            fact.get("subject", "")
        )

        fact_relation_context = normalize_text(
            fact.get("relation", "")
        )

        fact_markers = set()

        if (
            "baseline" in fact_subject_context
            or fact_relation_context in {
                "baseline value",
                "baseline result",
            }
        ):
            fact_markers.add("baseline")

        if re.search(r"\bpeak\b", fact_subject_context):
            fact_markers.add("peak")

        if "pre discharge" in fact_subject_context:
            fact_markers.add("pre_discharge")
        elif (
            "discharge" in fact_subject_context
            or "discharge" in fact_section
        ):
            fact_markers.add("discharge")

        admission_date = (
            record.get("admission", {}).get("admission_date")
            if record is not None
            else None
        )

        if admission_date and admission_date in fact_dates:
            fact_markers.add("admission")

        incompatible_contexts = {
            "baseline": {
                "admission", "peak", "pre_discharge", "discharge"
            },
            "admission": {
                "baseline", "peak", "pre_discharge", "discharge"
            },
            "peak": {
                "baseline", "admission", "pre_discharge", "discharge"
            },
            "pre_discharge": {
                "baseline", "admission", "peak"
            },
            "discharge": {
                "baseline", "admission", "peak"
            },
        }

        for marker in claim_markers:
            if marker in fact_markers:
                score += 3.0
                reasons.append(f"{marker} context matches")

            elif fact_markers & incompatible_contexts.get(marker, set()):
                score -= 2.0
                reasons.append(f"{marker} context differs")
        if score > 0:
            results.append({
                "fact": fact,
                "score": round(score, 3),
                "match_reasons": reasons,
            })

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:top_k]


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