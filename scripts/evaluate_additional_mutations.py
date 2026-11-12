"""Create and score additional deterministic mutations from saved atomic claims.

The mutation generator uses only claims already extracted from original summary
text. For each eligible baseline claim that the verifier supports, it creates a
copy of the summary with one value/attribute changed and a corresponding copy
of that atomic claim. Mutation labels are held by this evaluation script and are
used only after predictions; they are never passed to retrieval or verification.

IMPORTANT: this is a verification-stage perturbation test, not a fresh LLM
claim-extraction experiment. The claim fields are updated deterministically to
represent what an extractor would see in the mutated source. Fresh LLM extraction
requires the configured Gemini credentials and should be evaluated separately.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_builder import GROUND_TRUTH_PATH, normalize_clinical_entity, retrieve_candidate_facts
from src.verification_engine import _canonical_relation, verify_claim_against_facts

CLAIMS_DIR = ROOT / "data" / "claims"
SUMMARY_DIR = ROOT / "data" / "discharge_summaries"
OUT = ROOT / "results" / "additional_error_injection"
SUMMARY_OUT = OUT / "summaries"
CLAIM_OUT = OUT / "claims"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def norm(text: Any) -> str:
    return " ".join(str(text or "").casefold().replace("–", "-").replace("—", "-").split())


def case_from_filename(name: str) -> str:
    m = re.match(r"(SYN-DS-\d+)", name)
    if not m:
        raise ValueError(f"Unrecognized claim file: {name}")
    return m.group(1)


def predict(claim: dict, record: dict) -> tuple[dict, list[dict]]:
    candidates = retrieve_candidate_facts(
        claim, record.get("ground_truth_facts", []), top_k=20, record=record
    )
    result = verify_claim_against_facts(claim, candidates, record=record)
    return result, candidates


def _fact_by_id(record: dict, fact_id: str) -> dict | None:
    return next((f for f in record.get("ground_truth_facts", []) if f.get("fact_id") == fact_id), None)


def _matched_facts(claim: dict, result: dict, record: dict) -> list[dict]:
    ids = set(result.get("matched_fact_ids", []))
    return [f for f in record.get("ground_truth_facts", []) if f.get("fact_id") in ids]


def _replace_source_span(summary_text: str, old_span: str, new_span: str) -> str | None:
    """Replace a claim span exactly, then tolerate Unicode whitespace/dashes.

    Source spans can include narrow no-break spaces or typographic dashes while
    the visible summary uses ordinary spaces/dashes. This fallback only relaxes
    formatting; it does not change word order or permit fuzzy semantic matches.
    """
    if not old_span:
        return None
    if old_span in summary_text:
        return summary_text.replace(old_span, new_span, 1)

    pieces = re.split(r"(\s+|[–—−-])", old_span)
    pattern_parts = []
    for piece in pieces:
        if not piece:
            continue
        if piece.isspace():
            pattern_parts.append(r"\s+")
        elif piece in {"–", "—", "−", "-"}:
            pattern_parts.append(r"[–—−-]")
        else:
            pattern_parts.append(re.escape(piece))
    pattern = re.compile("".join(pattern_parts), flags=re.IGNORECASE)
    match = pattern.search(summary_text)
    if not match:
        return None
    return summary_text[:match.start()] + new_span + summary_text[match.end():]


def _numeric_mutation(claim: dict) -> tuple[dict, str, str] | None:
    raw = str(claim.get("value", "")).strip()
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", raw):
        return None
    old = float(raw)
    # Relative perturbation keeps small measurements realistic and is always nonzero.
    new = old * 1.2 if old != 0 else 1.0
    decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
    new_s = f"{new:.{decimals}f}" if decimals else str(int(round(new)))
    if new_s == raw:
        new_s = f"{old + 1:.{decimals}f}" if decimals else str(int(old) + 1)
    old_pat = re.compile(rf"(?<![0-9]){re.escape(raw)}(?![0-9])")
    source = str(claim.get("source_text", ""))
    replaced, n = old_pat.subn(new_s, source, count=1)
    if n != 1:
        return None
    mutated = copy.deepcopy(claim)
    mutated["value"] = new_s
    mutated["source_text"] = replaced
    return mutated, source, replaced


def _route_mutation(claim: dict) -> tuple[dict, str, str] | None:
    old = str(claim.get("route", "")).strip()
    source = str(claim.get("source_text", ""))
    # Choose a route incompatibility rather than a compatible spelling variant.
    if re.search(r"\b(?:oral|po|by mouth)\b", old, flags=re.I):
        replacements = [
            (re.compile(r"\b(?:oral|PO|by mouth)\b", re.I), "IV"),
        ]
        new_value = "intravenous"
    elif re.search(r"\b(?:iv|intravenous|sc|subcutaneous|im|intramuscular|inhaled|inhalation)\b", old, flags=re.I):
        replacements = [
            (re.compile(r"\b(?:intravenous|IV|subcutaneous|SC|intramuscular|IM|inhaled|inhalation)\b", re.I), "oral"),
        ]
        new_value = "oral"
    else:
        return None
    replaced = source
    count = 0
    for pattern, repl in replacements:
        replaced, count = pattern.subn(repl, source, count=1)
        if count:
            break
    if not count:
        return None
    mutated = copy.deepcopy(claim)
    mutated["route"] = new_value
    mutated["source_text"] = replaced
    return mutated, source, replaced


def _frequency_mutation(claim: dict) -> tuple[dict, str, str] | None:
    old = str(claim.get("frequency", "")).strip()
    source = str(claim.get("source_text", ""))
    operations = [
        (re.compile(r"\bq\s*6\s*h\b", re.I), "q12h", r"\bq\s*6\s*h\b", "q12h"),
        (re.compile(r"\bq\s*8\s*h\b", re.I), "q12h", r"\bq\s*8\s*h\b", "q12h"),
        (re.compile(r"\bevery\s+6\s+hours?\b", re.I), "every 12 hours", r"\bevery\s+6\s+hours?\b", "every 12 hours"),
        (re.compile(r"\bevery\s+8\s+hours?\b", re.I), "every 12 hours", r"\bevery\s+8\s+hours?\b", "every 12 hours"),
        (re.compile(r"\b(?:once daily|once a day|daily|nightly|qd)\b", re.I), "twice daily", r"\b(?:once daily|once a day|daily|nightly|qd)\b", "twice daily"),
        (re.compile(r"\b(?:twice daily|twice a day|bid|BID)\b", re.I), "once daily", r"\b(?:twice daily|twice a day|bid|BID)\b", "once daily"),
    ]
    for source_pattern, source_replacement, old_pattern, claim_replacement in operations:
        replaced, count = source_pattern.subn(source_replacement, source, count=1)
        if not count:
            continue
        mutated = copy.deepcopy(claim)
        old_freq = str(mutated.get("frequency", ""))
        if re.search(old_pattern, old_freq, re.I):
            mutated["frequency"] = re.sub(old_pattern, claim_replacement, old_freq, count=1, flags=re.I)
        else:
            # Keep additional modifiers (PRN, with food) when the source wording
            # and normalized claim value differ, but update the cadence itself.
            mutated["frequency"] = claim_replacement
        mutated["source_text"] = replaced
        return mutated, source, replaced
    return None


def _duration_mutation(claim: dict) -> tuple[dict, str, str] | None:
    old = str(claim.get("duration", "")).strip()
    source = str(claim.get("source_text", ""))
    m = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*(days?|weeks?|months?)\b", old, re.I)
    if not m:
        return None
    number = float(m.group(1))
    decimals = len(m.group(1).split(".", 1)[1]) if "." in m.group(1) else 0
    new_num = number + 1
    new_num_s = f"{new_num:.{decimals}f}" if decimals else str(int(new_num))
    unit = m.group(2)
    new_duration = old[:m.start(1)] + new_num_s + old[m.end(1):]
    source_pat = re.compile(rf"(?<!\d){re.escape(m.group(1))}\s*{re.escape(unit)}\b", re.I)
    replaced, count = source_pat.subn(f"{new_num_s} {unit}", source, count=1)
    if not count:
        return None
    mutated = copy.deepcopy(claim)
    mutated["duration"] = new_duration
    mutated["source_text"] = replaced
    return mutated, source, replaced


def _next_date_string(value: str) -> str | None:
    raw = str(value or "").strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3))) + timedelta(days=1)
        return d.isoformat()
    m = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)(?:\s+(\d{4}))?", raw)
    if m:
        months = {"jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,"may":5,"jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,"september":9,"oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12}
        month = months.get(m.group(2).lower())
        if not month: return None
        year = int(m.group(3) or 2026)
        d = date(year, month, int(m.group(1))) + timedelta(days=1)
        result = f"{d.day} {d.strftime('%B')}"
        if m.group(3): result += f" {d.year}"
        return result
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", raw)
    if m:
        first, second, year = map(int, m.groups())
        if first > 12:
            d = date(year, second, first) + timedelta(days=1)
            return d.strftime("%d/%m/%Y")
        if second > 12:
            d = date(year, first, second) + timedelta(days=1)
            return d.strftime("%m/%d/%Y")
    return None


def _date_mutation(claim: dict) -> tuple[dict, str, str] | None:
    rel = _canonical_relation(claim.get("relation", ""))
    if rel not in {"performed on", "date", "start date", "stop date", "date of birth"}:
        return None
    source = str(claim.get("source_text", ""))
    old_value = str(claim.get("value", "")).strip()
    if not old_value:
        old_value = str(claim.get("date_time", "")).strip()
    new_value = _next_date_string(old_value)
    # If `value` omits a year but date_time has it, derive the new date in the
    # same day/month/year representation as value and use the updated year in context.
    if not new_value:
        dt = str(claim.get("date_time", "")).strip()
        new_value = _next_date_string(dt)
        old_value = dt
    if not new_value:
        return None
    # Replace the date occurrence in the statement (first exact occurrence).
    if old_value not in source:
        # Date-time can include an ISO date while source uses written date; find
        # an unambiguous written day/month token as a fallback.
        return None
    replaced = source.replace(old_value, new_value, 1)
    mutated = copy.deepcopy(claim)
    mutated["value"] = new_value
    mutated["date_time"] = new_value
    mutated["source_text"] = replaced
    return mutated, source, replaced


def _temporal_value_swap_mutation(claim: dict, record: dict, result: dict) -> tuple[dict, str, str] | None:
    """Swap a numeric result with another timepoint for the same clinical entity.

    Unlike a simple +20% perturbation, this creates a clinically plausible but
    temporally wrong value (e.g. a peak value attributed to admission). The
    claim's original date/time is deliberately retained so the verifier must
    use time-aware evidence selection to find the conflict.
    """
    category = str(claim.get("category", "")).casefold()
    if not any(tag in category for tag in ("laboratory", "lab result", "investigation", "vital")):
        return None
    value = str(claim.get("value", "")).strip()
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value):
        return None
    target_ids = set(result.get("matched_fact_ids", []))
    target_facts = [f for f in record.get("ground_truth_facts", []) if f.get("fact_id") in target_ids]
    if not target_facts:
        return None
    target = next((f for f in target_facts if _canonical_relation(f.get("relation", "")) in {"result", "value", "measurement"}), None)
    if not target:
        return None
    target_entity = normalize_clinical_entity(target.get("subject", ""))
    target_date = norm(target.get("date_time", ""))
    target_unit = norm(target.get("unit", ""))
    if not target_entity or not target_date:
        return None

    alternatives = []
    for fact in record.get("ground_truth_facts", []):
        if _canonical_relation(fact.get("relation", "")) not in {"result", "value", "measurement"}:
            continue
        if normalize_clinical_entity(fact.get("subject", "")) != target_entity:
            continue
        if norm(fact.get("date_time", "")) == target_date or not fact.get("date_time"):
            continue
        if norm(fact.get("unit", "")) != target_unit:
            continue
        alternative_value = str(fact.get("value", "")).strip()
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", alternative_value) or alternative_value == value:
            continue
        if not re.search(rf"(?<![0-9]){re.escape(value)}(?![0-9])", str(claim.get("source_text", ""))):
            continue
        alternatives.append(fact)

    if not alternatives:
        return None
    alternatives.sort(key=lambda f: (str(f.get("date_time", "")), str(f.get("fact_id", ""))))
    alternative = alternatives[0]
    new_value = str(alternative.get("value", "")).strip()
    source = str(claim.get("source_text", ""))
    replaced, count = re.subn(
        rf"(?<![0-9]){re.escape(value)}(?![0-9])",
        new_value,
        source,
        count=1,
    )
    if count != 1:
        return None
    mutated = copy.deepcopy(claim)
    mutated["value"] = new_value
    # Keep the original timepoint/date, intentionally making the copied value
    # temporally inconsistent with the assertion context.
    mutated["source_text"] = replaced
    return mutated, source, replaced


def _narrative_polarity_mutation(claim: dict) -> tuple[dict, str, str] | None:
    """Invert a clear negated clinical finding while preserving the sentence context."""
    value = str(claim.get("value", ""))
    source = str(claim.get("source_text", ""))
    mappings = [
        (re.compile(r"\bno extraluminal air\b", re.I), "extraluminal air"),
        (re.compile(r"\bwithout perforation\b", re.I), "with perforation"),
        (re.compile(r"\bwithout abscess\b", re.I), "with abscess"),
        (re.compile(r"\bno growth\b", re.I), "growth"),
        (re.compile(r"\bnegative for\b", re.I), "positive for"),
        (re.compile(r"\bno evidence of\b", re.I), "evidence of"),
        (re.compile(r"\bnot detected\b", re.I), "detected"),
        (re.compile(r"\babsence of\b", re.I), "presence of"),
    ]
    for pattern, replacement in mappings:
        if not pattern.search(value) or not pattern.search(source):
            continue
        mutated = copy.deepcopy(claim)
        mutated["value"] = pattern.sub(replacement, value, count=1)
        mutated["source_text"] = pattern.sub(replacement, source, count=1)
        # The new assertion is affirmative; do not retain the original negation flag.
        mutated["negated"] = False
        return mutated, source, mutated["source_text"]
    return None


def _multi_field_medication_mutation(claim: dict) -> tuple[dict, str, str] | None:
    """Change dose and schedule together to test interacting field checks."""
    original_source = str(claim.get("source_text", ""))
    dose_mutation = _numeric_mutation(claim)
    if dose_mutation is None:
        return None
    frequency_mutation = _frequency_mutation(dose_mutation[0])
    if frequency_mutation is None:
        return None
    mutated_claim = frequency_mutation[0]
    if str(mutated_claim.get("value", "")) == str(claim.get("value", "")):
        return None
    if str(mutated_claim.get("frequency", "")) == str(claim.get("frequency", "")):
        return None
    return mutated_claim, original_source, frequency_mutation[2]


def _allergy_polarity_mutation(claim: dict) -> tuple[dict, str, str] | None:
    if "allergy" not in str(claim.get("category", "")).lower():
        return None
    old = str(claim.get("value", ""))
    source = str(claim.get("source_text", ""))
    phrase = re.compile(r"\bno known drug allergies\b", re.I)
    if not phrase.search(old) or not phrase.search(source):
        return None
    mutated = copy.deepcopy(claim)
    mutated["value"] = "Known drug allergy"
    mutated["negated"] = False
    replaced = phrase.sub("Known drug allergy", source, count=1)
    mutated["source_text"] = replaced
    return mutated, source, replaced


def _unit_mutation(claim: dict) -> tuple[dict, str, str] | None:
    old = str(claim.get("unit", "")).strip()
    value = str(claim.get("value", "")).strip()
    source = str(claim.get("source_text", ""))
    if not old or not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value):
        return None
    alternatives = {
        "%": "mg/L",
        "mg/dL": "mmol/L",
        "mmol/L": "mg/dL",
        "mg/L": "g/L",
        "g/dL": "mg/dL",
        "°C": "°F",
        "×10⁹/L": "mg/L",
        "bpm": "mmHg",
        "kg": "lb",
    }
    new = alternatives.get(old)
    if not new:
        return None
    # Replace unit after the first occurrence of the asserted scalar to avoid
    # changing unrelated units elsewhere in the same statement.
    number_match = re.search(rf"(?<!\d){re.escape(value)}(?!\d)", source)
    if not number_match:
        return None
    tail = source[number_match.end():]
    unit_match = re.search(re.escape(old), tail, re.I)
    if not unit_match:
        return None
    start = number_match.end() + unit_match.start()
    end = number_match.end() + unit_match.end()
    replaced = source[:start] + new + source[end:]
    mutated = copy.deepcopy(claim)
    mutated["unit"] = new
    mutated["source_text"] = replaced
    return mutated, source, replaced


MUTATORS: list[tuple[str, Callable[[dict], tuple[dict, str, str] | None]]] = [
    ("laboratory_numeric", _numeric_mutation),
    ("medication_dose", _numeric_mutation),
    ("route", _route_mutation),
    ("frequency", _frequency_mutation),
    ("duration", _duration_mutation),
    ("event_date", _date_mutation),
    ("allergy_polarity", _allergy_polarity_mutation),
    ("narrative_polarity", _narrative_polarity_mutation),
    ("multi_field_medication", _multi_field_medication_mutation),
    ("temporal_value_swap", lambda claim: None),  # handled with record/context in main loop
    ("unit", _unit_mutation),
]


def is_eligible(kind: str, claim: dict, result: dict, record: dict) -> bool:
    if result.get("verdict") != "SUPPORTED":
        return False
    facts = _matched_facts(claim, result, record)
    relations = {_canonical_relation(f.get("relation", "")) for f in facts}
    category = str(claim.get("category", "")).casefold().replace("_", " ")
    val = str(claim.get("value", "")).strip()
    unit = str(claim.get("unit", "")).strip()
    if kind == "laboratory_numeric":
        return ("laboratory" in category or "lab result" in category or "investigation" in category) and bool(unit) and bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", val)) and bool(relations & {"result", "value", "measurement"})
    if kind == "medication_dose":
        return "medication" in category and bool(unit) and bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", val)) and "dose" in relations
    if kind == "route":
        return "medication" in category and bool(claim.get("route")) and "route" in relations
    if kind == "frequency":
        freq = str(claim.get("frequency", ""))
        has_cadence = bool(re.search(r"\b(?:q\s*\d+\s*h|every\s+\d+\s+hours?|daily|once daily|twice daily|three times daily|four times daily|bid|tid|qid|nightly)\b", freq, re.I))
        return "medication" in category and has_cadence and "frequency" in relations
    if kind == "duration":
        dur = str(claim.get("duration", ""))
        has_days = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:days?|weeks?|months?)\b", dur, re.I))
        return "medication" in category and has_days and "duration" in relations
    if kind == "event_date":
        return _canonical_relation(claim.get("relation", "")) in {"performed on", "date", "start date", "stop date", "date of birth"} and bool(claim.get("value") or claim.get("date_time")) and bool(relations & {"performed on", "date", "start date", "stop date", "date of birth"})
    if kind == "temporal_value_swap":
        return (
            ("laboratory" in category or "lab result" in category or "investigation" in category or "vital" in category)
            and bool(unit)
            and bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", val))
            and bool(relations & {"result", "value", "measurement"})
            and bool(claim.get("date_time"))
        )
    if kind == "narrative_polarity":
        value_text = str(claim.get("value", ""))
        source_text = str(claim.get("source_text", ""))
        has_negated_finding = any(
            pattern.search(value_text) and pattern.search(source_text)
            for pattern in [
                re.compile(r"\bno extraluminal air\b", re.I),
                re.compile(r"\bwithout perforation\b", re.I),
                re.compile(r"\bwithout abscess\b", re.I),
                re.compile(r"\bno growth\b", re.I),
                re.compile(r"\bnegative for\b", re.I),
                re.compile(r"\bno evidence of\b", re.I),
                re.compile(r"\bnot detected\b", re.I),
                re.compile(r"\babsence of\b", re.I),
            ]
        )
        return bool(has_negated_finding) and result.get("verdict") == "SUPPORTED"
    if kind == "multi_field_medication":
        freq = str(claim.get("frequency", ""))
        has_cadence = bool(re.search(r"\b(?:q\s*\d+\s*h|every\s+\d+\s+hours?|daily|once daily|twice daily|three times daily|four times daily|bid|tid|qid|nightly)\b", freq, re.I))
        return (
            "medication" in category and bool(unit)
            and bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", val))
            and "dose" in relations and "frequency" in relations and has_cadence
        )
    if kind == "allergy_polarity":
        return "allergy" in category and "status" in relations
    if kind == "unit":
        return ("laboratory" in category or "lab result" in category or "investigation" in category or "vital" in category) and bool(unit) and bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", val)) and bool(relations & {"result", "value", "measurement"})
    return False


def main() -> int:
    records = read_json(GROUND_TRUTH_PATH)
    records_by_id = {r["case_metadata"]["case_id"]: r for r in records}
    claims_payloads: list[tuple[Path, str, dict, dict]] = []
    baseline_predictions: dict[tuple[str, str], dict] = {}

    # Baseline phase: only the eight original summaries are candidate sources.
    for path in sorted(CLAIMS_DIR.glob("*_claims.json")):
        if re.search(r"_E\d+_claims$", path.stem):
            continue
        case_id = case_from_filename(path.name)
        record = records_by_id[case_id]
        payload = read_json(path)
        claims = payload.get("claims", payload)
        summary = SUMMARY_DIR / f"{case_id}.txt"
        summary_text = summary.read_text(encoding="utf-8")
        for claim in claims:
            key = (path.name, str(claim.get("claim_id")))
            result, candidates = predict(claim, record)
            baseline_predictions[key] = {
                "verdict": result.get("verdict"),
                "error_type": result.get("error_type"),
                "matched_fact_ids": result.get("matched_fact_ids", []),
                "candidate_count": len(candidates),
            }
            claims_payloads.append((path, case_id, claim, {"record": record, "summary_text": summary_text, "result": result, "candidates": candidates}))

    # Select high-confidence supported source claims. Selection uses only the
    # original claim+record, never a proposed mutation or its expected verdict.
    max_per_type = 12
    selected: list[dict] = []
    per_type_counts: Counter = Counter()
    seen_pairs: set[tuple[str, str, str]] = set()
    for kind, mutator in MUTATORS:
        for path, case_id, claim, ctx in claims_payloads:
            if per_type_counts[kind] >= max_per_type:
                break
            if not is_eligible(kind, claim, ctx["result"], ctx["record"]):
                continue
            dedupe_key = (kind, path.name, str(claim.get("claim_id")))
            if dedupe_key in seen_pairs:
                continue
            if kind == "temporal_value_swap":
                mutated_result = _temporal_value_swap_mutation(claim, ctx["record"], ctx["result"])
            else:
                mutated_result = mutator(claim)
            if mutated_result is None:
                continue
            mutated_claim, old_span, new_span = mutated_result
            mutated_summary = _replace_source_span(ctx["summary_text"], old_span, new_span)
            if mutated_summary is None:
                continue
            if norm(mutated_claim.get("source_text", "")) not in norm(mutated_summary):
                continue
            # A control and mutation are now frozen; expected labels are not
            # included in the input passed to predict().
            selected.append({
                "test_id": f"ADD-{len(selected)+1:03d}",
                "mutation_type": kind,
                "case_id": case_id,
                "source_claim_file": path.name,
                "source_claim_id": str(claim.get("claim_id")),
                "original_summary_path": str((SUMMARY_DIR / f"{case_id}.txt").relative_to(ROOT)),
                "mutated_summary_text": mutated_summary,
                "mutated_claim": mutated_claim,
                "original_claim": copy.deepcopy(claim),
                "original_claim_source_text": old_span,
                "mutated_claim_source_text": new_span,
                "control_verdict_before_run": ctx["result"].get("verdict"),
                "control_fact_ids": ctx["result"].get("matched_fact_ids", []),
                "expected_verdict": "CONTRADICTED",
            })
            per_type_counts[kind] += 1
            seen_pairs.add(dedupe_key)

    if not selected:
        raise RuntimeError("No eligible mutations were generated; inspect data/schema.")

    SUMMARY_OUT.mkdir(parents=True, exist_ok=True)
    CLAIM_OUT.mkdir(parents=True, exist_ok=True)
    manifest_for_scoring = []
    predictions = []
    control_results = []

    # Prediction phase: no mutation type, target fact ID, expected verdict, or
    # expected value is passed to the retrieval/verifier API.
    for item in selected:
        record = records_by_id[item["case_id"]]
        mutated_claim = item["mutated_claim"]
        result, candidates = predict(mutated_claim, record)
        control_claim = item["original_claim"]
        control_result, control_candidates = predict(control_claim, record)
        test_id = item["test_id"]
        summary_path = SUMMARY_OUT / f"{test_id}_{item['case_id']}.txt"
        claim_path = CLAIM_OUT / f"{test_id}_claims.json"
        summary_path.write_text(item["mutated_summary_text"], encoding="utf-8")
        claim_payload = {
            "summary_file": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
            "claim_count": 1,
            "claims": [mutated_claim],
        }
        claim_path.write_text(json.dumps(claim_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        predictions.append({
            "test_id": test_id,
            "predicted_verdict": result.get("verdict"),
            "predicted_error_type": result.get("error_type"),
            "candidate_count": len(candidates),
            "matched_fact_ids": result.get("matched_fact_ids", []),
            "field_results": result.get("field_results", {}),
            "top_candidates": [{
                "fact_id": c.get("fact", {}).get("fact_id"),
                "subject": c.get("fact", {}).get("subject"),
                "relation": c.get("fact", {}).get("relation"),
                "value": c.get("fact", {}).get("value"),
                "score": c.get("score"),
            } for c in candidates[:5]],
            "source_text_valid_in_mutated_summary": norm(mutated_claim.get("source_text", "")) in norm(item["mutated_summary_text"]),
            "control_verdict_rerun": control_result.get("verdict"),
            "control_false_contradiction": control_result.get("verdict") == "CONTRADICTED",
            "claim_file": str(claim_path.relative_to(ROOT)).replace("\\", "/"),
            "summary_file": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
        })
        # Expected labels are collected only after each prediction is produced.
        manifest_for_scoring.append({
            "test_id": test_id,
            "mutation_type": item["mutation_type"],
            "case_id": item["case_id"],
            "source_claim_file": item["source_claim_file"],
            "source_claim_id": item["source_claim_id"],
            "expected_verdict": item["expected_verdict"],
            "baseline_matched_fact_ids": item["control_fact_ids"],
            "claim_file": str(claim_path.relative_to(ROOT)).replace("\\", "/"),
            "summary_file": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
        })
        control_results.append({"test_id": test_id, "verdict": control_result.get("verdict")})

    # Scoring phase, isolated from prediction inputs.
    by_type: dict[str, list[dict]] = defaultdict(list)
    pred_by_id = {p["test_id"]: p for p in predictions}
    for label in manifest_for_scoring:
        pred = pred_by_id[label["test_id"]]
        row = {**label, **pred}
        row["detected_as_contradicted"] = pred["predicted_verdict"] == "CONTRADICTED"
        row["missed_mutation"] = not row["detected_as_contradicted"]
        by_type[label["mutation_type"]].append(row)

    type_summary = {}
    for kind, rows in sorted(by_type.items()):
        detected = sum(row["detected_as_contradicted"] for row in rows)
        type_summary[kind] = {
            "mutations": len(rows),
            "detected_contradicted": detected,
            "missed": len(rows) - detected,
            "detection_rate": detected / len(rows) if rows else None,
            "verdicts": dict(Counter(row["predicted_verdict"] for row in rows)),
        }

    detected = sum(row["detected_as_contradicted"] for rows in by_type.values() for row in rows)
    total = len(manifest_for_scoring)
    controls_supported = sum(c["verdict"] == "SUPPORTED" for c in control_results)
    controls_other = len(control_results) - controls_supported
    overall = {
        "generated_mutations": total,
        "detected_as_contradicted": detected,
        "not_detected_as_contradicted": total - detected,
        "mutation_detection_rate": detected / total if total else None,
        "mutations_by_type": type_summary,
        "control_claims_rerun": len(control_results),
        "controls_supported_on_rerun": controls_supported,
        "controls_with_other_verdict": controls_other,
        "control_false_contradictions": sum(c["verdict"] == "CONTRADICTED" for c in control_results),
        "warning": "Controls were selected because the baseline verifier marked them SUPPORTED. Their rerun is a stability check, not an unbiased specificity estimate. All cases use saved extractor claims with deterministic field mutations; claim extraction was not rerun by an LLM.",
    }

    detail = {
        "title": "Additional controlled mutations from saved summary-derived atomic claims",
        "generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": {
            "synthetic_only": True,
            "source_claims_from_original_summary_files_only": True,
            "claim_extraction_rerun": False,
            "mutations_change_summary_text_and_corresponding_atomic_claim": True,
            "mutation_labels_passed_to_verifier": False,
            "manifest_scored_after_predictions": True,
            "record_ids_and_error_ids_passed_to_verifier": False,
            "selection_rule": "Original claim must be SUPPORTED by baseline verifier and have matching retrieved relation-family evidence for the attribute being mutated.",
        },
        "aggregate": overall,
        "mutation_manifest_used_for_scoring_after_prediction": manifest_for_scoring,
        "predictions": predictions,
    }
    (OUT / "additional_mutation_evaluation_report.json").write_text(json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Additional controlled error-injection evaluation",
        "",
        f"Run time: {detail['generated_at_local']}",
        "",
        "## Scope and fairness",
        "",
        "This test starts with atomic claims previously extracted from the original synthetic discharge summaries. It creates paired mutated summaries and mutated claim objects by changing one field at a time. The verifier receives only the mutated claim and the corresponding ground-truth record; mutation type, target fact ID, expected verdict, and changed value are kept in the scoring manifest and attached after predictions.",
        "",
        "This is an additional **verification-stage controlled perturbation test**, not a fresh LLM extraction test. The baseline claims were selected because this verifier marked them SUPPORTED, so the original controls are a stability check and must not be interpreted as an unbiased specificity estimate. Cases come from the existing eight records, so the test is not an independent record-level holdout.",
        "",
        "## Overall result",
        "",
        f"- Mutations generated: {total}",
        f"- Detected as CONTRADICTED: {detected}/{total} ({(detected/total*100 if total else 0):.1f}%)",
        f"- Not detected as CONTRADICTED: {total-detected}",
        f"- Controls rerun: {len(control_results)}; still SUPPORTED: {controls_supported}; other verdict: {controls_other}; falsely CONTRADICTED: {overall['control_false_contradictions']}",
        "",
        "## By error type",
        "",
        "| Mutation type | N | Detected | Missed | Detection rate | Verdicts |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for kind, stats in sorted(type_summary.items()):
        verdicts = ", ".join(f"{k}: {v}" for k, v in stats["verdicts"].items())
        lines.append(f"| {kind} | {stats['mutations']} | {stats['detected_contradicted']} | {stats['missed']} | {stats['detection_rate']*100:.1f}% | {verdicts} |")
    lines += ["", "## Missed mutations", ""]
    missed = [row for rows in by_type.values() for row in rows if not row["detected_as_contradicted"]]
    if not missed:
        lines.append("No mutated claims were missed in this generated set.")
    else:
        lines += ["| Test | Type | Case/claim | Predicted verdict | Error type | Source text |", "|---|---|---|---|---|---|"]
        for row in missed:
            source = str(row.get("source_text", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['test_id']} | {row['mutation_type']} | {row['case_id']} / {row['source_claim_id']} | {row['predicted_verdict']} | {row.get('predicted_error_type')} | {source} |")
    lines += [
        "",
        "## Interpretation and limitations",
        "",
        "A mutation detection rate is reported instead of a global F1 score. The mutations are selected from verifier-supported baseline claims, which creates selection bias. The source-claim fields are deterministically updated to reflect the injected sentence; the LLM extractor is not rerun. This test evaluates whether the retrieval/verification stage catches additional controlled discrepancies; it does not establish end-to-end performance on independently generated summaries or real-world clinical notes.",
        "",
    ]
    (OUT / "ADDITIONAL_MUTATION_EVALUATION_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "mutation_scoring_manifest.json").write_text(json.dumps(manifest_for_scoring, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== ADDITIONAL MUTATION EVALUATION ===")
    print(f"Mutations generated: {total}")
    print(f"Detected as contradicted: {detected}/{total} ({detected/total*100:.1f}%)")
    print(f"Controls: {controls_supported}/{len(control_results)} still SUPPORTED; {overall['control_false_contradictions']} falsely contradicted")
    for kind, stats in sorted(type_summary.items()):
        print(f"{kind}: {stats['detected_contradicted']}/{stats['mutations']} ({stats['detection_rate']*100:.1f}%) | {stats['verdicts']}")
    print(f"Report: {OUT / 'additional_mutation_evaluation_report.json'}")
    print(f"Summary: {OUT / 'ADDITIONAL_MUTATION_EVALUATION_SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
