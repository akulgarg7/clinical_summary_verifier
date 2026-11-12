"""Evaluate verification on held-out synthetic records without target/error hints.

This is a controlled fact-level benchmark, not a clinical gold-standard study.
It constructs an exact control claim from each eligible source fact and a paired
contradiction by changing one eligible field. The verifier receives the claim
and a record's facts only; it is not given target fact IDs, mutation status, or
mutated field metadata. Entire case IDs are reserved for evaluation.

Usage:
    python scripts/evaluate_heldout.py
    python scripts/evaluate_heldout.py --records SYN-DS-0007 SYN-DS-0008
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_builder import GROUND_TRUTH_PATH, retrieve_candidate_facts
from src.verification_engine import verify_claim_against_facts

DEFAULT_HELD_OUT = ["SYN-DS-0001", "SYN-DS-0002", "SYN-DS-0004", "SYN-DS-0005"]
FINAL_EVALUATION_RECORDS = {"SYN-DS-0001", "SYN-DS-0002", "SYN-DS-0004", "SYN-DS-0005"}
DEVELOPMENT_RECORDS = {"SYN-DS-0007", "SYN-DS-0008"}
REGRESSION_RECORDS = {"SYN-DS-0003", "SYN-DS-0006"}
ROUTE_RELATIONS = {"route", "administration route"}
FREQUENCY_RELATIONS = {"frequency", "dosing frequency", "administration frequency"}
DATE_RELATIONS = {"date", "admission date", "discharge date", "start date", "stop date", "performed on"}


def _norm_relation(value: Any) -> str:
    return re.sub(r"[_\s]+", " ", str(value or "").strip().casefold())


def _mutation(value: str, relation: str, unit: str) -> tuple[str, str] | None:
    """Return (mutated_value, mutation_kind) using general, label-independent rules."""
    val = str(value or "").strip()
    rel = _norm_relation(relation)

    # Numeric facts: perturb the value while preserving the original format.
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", val):
        if "." in val:
            decimals = len(val.split(".", 1)[1])
            altered = f"{float(val) + (10 ** -decimals):.{decimals}f}"
        else:
            altered = str(int(val) + 1)
        return altered, "numeric"

    # ISO date values are perturbed as dates, not by swapping a known patient/error.
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})", val)
    if m or rel in DATE_RELATIONS and re.fullmatch(r"\d{4}-\d{2}-\d{2}", val):
        d = date.fromisoformat(val[:10]) + timedelta(days=1)
        return d.isoformat(), "date"

    if rel in ROUTE_RELATIONS:
        low = val.casefold()
        # Mutate a route to a clinically distinct general route category.
        if re.search(r"\b(?:oral|po|by mouth|per os)\b", low):
            return "intravenous", "route"
        if re.search(r"\b(?:intravenous|i\.v\.|iv)\b", low):
            return "oral", "route"
        if re.search(r"\b(?:inhaled|inhalation)\b", low):
            return "oral", "route"
        if re.search(r"\b(?:intramuscular|i\.m\.|im)\b", low):
            return "subcutaneous", "route"
        if re.search(r"\b(?:subcutaneous|subcut|s\.c\.|sc)\b", low):
            return "intramuscular", "route"

    if rel in FREQUENCY_RELATIONS:
        low = val.casefold()
        # Test the longer/more specific phrase first: "daily" is contained in
        # "twice daily" and would otherwise create a no-op mutation.
        if re.search(r"\b(?:twice daily|twice a day|bid)\b", low):
            return "once daily", "frequency"
        if re.search(r"\b(?:once daily|daily|once a day|every 24 hours|qd)\b", low):
            return "twice daily", "frequency"
        m = re.search(r"\bevery\s+(\d+)\s+hours?\b", low)
        if m:
            n = int(m.group(1))
            alternative = 6 if n != 6 else 8
            return re.sub(r"\bevery\s+\d+\s+hours?\b", f"every {alternative} hours", val, count=1, flags=re.I), "frequency"
        if re.search(r"\bq\s*\d+\s*h\b", low):
            m = re.search(r"\bq\s*(\d+)\s*h\b", low)
            n = int(m.group(1)); alternative = 6 if n != 6 else 8
            return re.sub(r"\bq\s*\d+\s*h\b", f"q{alternative}h", val, count=1, flags=re.I), "frequency"

    if rel in {"sex", "gender"}:
        if val.casefold() == "male": return "Female", "categorical"
        if val.casefold() == "female": return "Male", "categorical"

    if rel in {"result", "value", "status", "interpretation", "finding"}:
        binary = {
            "positive": "negative", "negative": "positive",
            "present": "absent", "absent": "present",
            "detected": "not detected", "not detected": "detected",
            "yes": "no", "no": "yes",
            "normal": "abnormal", "abnormal": "normal",
        }
        if val.casefold() in binary:
            return binary[val.casefold()], "categorical"

    return None


def _mutation_options(fact: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Return independent, plausible contradictions for an eligible fact.

    The third tuple item is evaluation metadata only. It is never copied into
    the claim or passed to the verifier.
    """
    value = str(fact.get("value", "")).strip()
    relation = _norm_relation(fact.get("relation", ""))
    unit = str(fact.get("unit", "") or "").strip()
    category = _norm_relation(fact.get("category", ""))
    options: list[tuple[str, str, str]] = []

    base_mutation = _mutation(value, relation, unit)
    if base_mutation and base_mutation[0].strip().casefold() != value.casefold():
        options.append((base_mutation[0], unit, base_mutation[1]))

    # Additional mutation families are deliberately independent of the first
    # one: dose-unit changes, medication status changes, and duration changes.
    if category == "medication" and relation == "dose" and unit.casefold() in {
        "mg", "g", "mcg", "μg", "µg", "mg/hour", "mcg/kg/min", "mg/kg"
    }:
        unit_map = {
            "mg": "g", "g": "mg", "mcg": "mg", "μg": "mg", "µg": "mg",
            "mg/hour": "mg/min", "mcg/kg/min": "mg/kg/min", "mg/kg": "mcg/kg",
        }
        alternate_unit = unit_map[unit.casefold()]
        if alternate_unit.casefold() != unit.casefold():
            options.append((value, alternate_unit, "unit"))

    if relation in {"status", "status relative to pre admission regimen", "medication status", "drug status"}:
        status_value = value.casefold().strip()
        status_alternatives = {
            "started": "discontinued", "discontinued": "started", "stopped": "started",
            "continued": "discontinued", "withheld": "continued", "held": "continued",
            "completed": "ongoing", "ongoing": "completed", "resumed": "discontinued",
            "dose_changed": "continued", "route_changed": "continued", "frequency_changed": "continued",
        }
        alternative = status_alternatives.get(status_value)
        if alternative and alternative != status_value:
            options.append((alternative, unit, "status"))

    if "duration" in relation:
        # Change a duration quantity (e.g. 14 days -> 15 days) without
        # touching embedded calendar dates in the same source value.
        duration_match = re.search(r"\b(\d{1,3})(?=\s*(?:more\s*)?(?:hours?|days?|weeks?|months?))", value, re.I)
        if duration_match:
            old_quantity = int(duration_match.group(1))
            new_quantity = old_quantity + 1
            new_value = value[:duration_match.start(1)] + str(new_quantity) + value[duration_match.end(1):]
            if new_value != value:
                options.append((new_value, unit, "duration"))

    # De-duplicate mutations with the same effective value/unit/type.
    deduplicated = []
    seen = set()
    for item in options:
        signature = (item[0].casefold().strip(), item[1].casefold().strip(), item[2])
        if signature not in seen:
            deduplicated.append(item)
            seen.add(signature)
    return deduplicated


def _eligible_fact(fact: dict[str, Any]) -> bool:
    if str(fact.get("should_appear_in_discharge_summary", "")).casefold() != "yes":
        return False
    if not all(str(fact.get(k, "")).strip() for k in ("subject", "relation", "value")):
        return False
    return bool(_mutation_options(fact))


def _claim_from_fact(
    fact: dict[str, Any],
    claim_id: str,
    value: str | None = None,
    unit_override: str | None = None,
) -> dict[str, Any]:
    claim_value = str(fact.get("value", "") if value is None else value)
    unit = str(fact.get("unit", "") or "") if unit_override is None else str(unit_override)
    subject = str(fact.get("subject", "") or "")
    relation = str(fact.get("relation", "") or "")
    dt = str(fact.get("date_time", "") or "")
    # source_text is neutral evidence of what this claim says; it does not reveal
    # whether this is a control or a mutation.
    components = [subject, relation, claim_value]
    if unit:
        components.append(unit)
    if dt:
        components.append(dt)
    source_text = " ".join(components)
    return {
        "claim_id": claim_id,
        "category": fact.get("category", ""),
        "subject": subject,
        "relation": relation,
        "value": claim_value,
        "unit": unit,
        "route": "",
        "frequency": "",
        "duration": "",
        "date_time": dt,
        "negated": False,
        "source_text": source_text,
    }


def _verify(claim: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    candidates = retrieve_candidate_facts(
        claim,
        record.get("ground_truth_facts", []),
        top_k=10,
        record=record,
    )
    result = verify_claim_against_facts(claim, candidates, record=record)
    # Keep evidence IDs for audit; never pass an intended target fact ID to verifier.
    return {
        "verdict": result.get("verdict", "INSUFFICIENT_EVIDENCE"),
        "error_type": result.get("error_type"),
        "matched_fact_ids": result.get("matched_fact_ids", []),
        "candidate_fact_ids": [
            (item.get("fact") or {}).get("fact_id") for item in candidates
        ],
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Positive class means a claim contains an explicit contradiction.
    tp = sum(r["expected"] == "CONTRADICTED" and r["predicted"] == "CONTRADICTED" for r in rows)
    fn = sum(r["expected"] == "CONTRADICTED" and r["predicted"] != "CONTRADICTED" for r in rows)
    fp = sum(r["expected"] != "CONTRADICTED" and r["predicted"] == "CONTRADICTED" for r in rows)
    tn = sum(r["expected"] != "CONTRADICTED" and r["predicted"] != "CONTRADICTED" for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    return {
        "n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "specificity": specificity, "accuracy": accuracy,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "verdict_counts": dict(Counter(r["predicted"] for r in rows)),
        "expected_counts": dict(Counter(r["expected"] for r in rows)),
    }


def run(records_to_hold_out: list[str]) -> dict[str, Any]:
    with GROUND_TRUTH_PATH.open("r", encoding="utf-8") as f:
        all_records = json.load(f)
    records_by_id = {r["case_metadata"]["case_id"]: r for r in all_records}
    missing = sorted(set(records_to_hold_out) - set(records_by_id))
    if missing:
        raise SystemExit(f"Unknown held-out record IDs: {missing}")

    rows: list[dict[str, Any]] = []
    by_mutation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = 0
    eligible_facts = 0
    mutation_count = 0
    for case_id in records_to_hold_out:
        record = records_by_id[case_id]
        facts = record.get("ground_truth_facts", [])
        for fact_index, fact in enumerate(facts):
            if not _eligible_fact(fact):
                skipped += 1
                continue
            mutations = _mutation_options(fact)
            if not mutations:
                skipped += 1
                continue
            eligible_facts += 1
            fact_key = f"HOLDOUT-{case_id[-4:]}-{fact_index:03d}"

            # Each source fact contributes one exact control; additional
            # contradiction mutations are scored separately. This avoids
            # artificially multiplying the control count when a fact has
            # more than one possible mutation family.
            control_claim = _claim_from_fact(fact, f"{fact_key}-control")
            observed = _verify(control_claim, record)
            control_row = {
                "case_id": case_id,
                "fact_category": fact.get("category", "unknown"),
                "relation": fact.get("relation", ""),
                "mutation_type": "control",
                "condition": "control",
                "expected": "SUPPORTED",
                "predicted": observed["verdict"],
                "error_type": observed["error_type"],
                "matched_fact_ids": observed["matched_fact_ids"],
                "candidate_fact_ids": observed["candidate_fact_ids"],
                "claim": control_claim,
            }
            rows.append(control_row)
            by_category[str(fact.get("category", "unknown"))].append(control_row)

            for mutation_index, (mutated_value, mutated_unit, mutation_type) in enumerate(mutations, start=1):
                mutation_count += 1
                claim = _claim_from_fact(
                    fact,
                    f"{fact_key}-mutation-{mutation_index:02d}",
                    value=mutated_value,
                    unit_override=mutated_unit,
                )
                observed = _verify(claim, record)
                row = {
                    "case_id": case_id,
                    "fact_category": fact.get("category", "unknown"),
                    "relation": fact.get("relation", ""),
                    "mutation_type": mutation_type,
                    "condition": "mutated",
                    "expected": "CONTRADICTED",
                    "predicted": observed["verdict"],
                    "error_type": observed["error_type"],
                    "matched_fact_ids": observed["matched_fact_ids"],
                    "candidate_fact_ids": observed["candidate_fact_ids"],
                    "claim": claim,
                }
                rows.append(row)
                by_mutation[mutation_type].append(row)
                by_category[str(fact.get("category", "unknown"))].append(row)

    metrics = _metrics(rows)
    mutated_rows = [r for r in rows if r["condition"] == "mutated"]
    control_rows = [r for r in rows if r["condition"] == "control"]
    # These are separate because they answer different practical questions.
    per_mutation = {k: _metrics(v) for k, v in sorted(by_mutation.items())}
    per_category = {k: _metrics(v) for k, v in sorted(by_category.items())}
    return {
        "benchmark": "held-out synthetic fact-level mutation benchmark",
        "version": 2,
        "held_out_record_ids": records_to_hold_out,
        "evaluation_split": (
            "development/validation set used during rule iteration"
            if set(records_to_hold_out) and set(records_to_hold_out).issubset(DEVELOPMENT_RECORDS)
            else "targeted regression-case check (not an independent evaluation)"
            if set(records_to_hold_out) and set(records_to_hold_out).issubset(REGRESSION_RECORDS)
            else "post-hoc controlled-mutation validation (records outside targeted regression cases; not a blind final holdout)"
            if set(records_to_hold_out) and set(records_to_hold_out).issubset(FINAL_EVALUATION_RECORDS)
            else "custom or mixed record selection; determine split status from record IDs"
        ),
        "previously_targeted_regression_record_ids": ["SYN-DS-0003", "SYN-DS-0006"],
        "development_record_ids_used_during_rule_iteration": ["SYN-DS-0007", "SYN-DS-0008"],
        "note": "No target fact ID, mutation label, expected verdict, or deliberately changed value is passed to the verifier. The control/mutation labels are used only by the scorer. This run is post-hoc validation, not a blind holdout: one exact control was inspected during debugging and prompted a general lab-versus-medication relation-selection fix before this rerun.",
        "eligibility_rule": "Source facts marked should_appear_in_discharge_summary=yes with a supported general mutation. Each eligible source fact contributes one exact control; a fact may contribute multiple contradiction mutations. Narrative-only values are generally excluded except where a controlled numeric/status/date/duration mutation is applicable.",
        "scoring_definition": "CONTRADICTED is positive; exact controls are expected SUPPORTED. A deliberately modified claim is counted as detected only when predicted CONTRADICTED. Target IDs and mutation labels are retained only in the evaluator's report and are not passed into retrieval or verification.",
        "eligible_fact_count": eligible_facts,
        "mutation_count": mutation_count,
        "skipped_facts": skipped,
        "overall_metrics": metrics,
        "mutated_claim_metrics": _metrics(mutated_rows),
        "control_claim_metrics": _metrics(control_rows),
        "per_mutation_type": per_mutation,
        "per_fact_category": per_category,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", default=DEFAULT_HELD_OUT)
    parser.add_argument("--output", default="results/final_holdout_evaluation_report.json")
    args = parser.parse_args()
    report = run(args.records)
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    m = report["overall_metrics"]
    print(f"Held-out records: {', '.join(report['held_out_record_ids'])}")
    print(f"Eligible source facts: {report['eligible_fact_count']}; contradiction mutations: {report['mutation_count']}; skipped facts: {report['skipped_facts']}")
    print("Overall (control + mutation samples):")
    print(f"  TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
    print(f"  precision={m['precision']:.4f} recall={m['recall']:.4f} F1={m['f1']:.4f} specificity={m['specificity']:.4f} accuracy={m['accuracy']:.4f}")
    mm = report["mutated_claim_metrics"]
    print(f"Mutation detection: precision={mm['precision']:.4f} recall={mm['recall']:.4f} F1={mm['f1']:.4f}; TP={mm['tp']} FN={mm['fn']}")
    cm = report["control_claim_metrics"]
    print(f"Exact controls: false positives={cm['fp']}/{cm['n']} (FPR={cm['false_positive_rate']:.4f}); non-contradicted={cm['tn']}/{cm['n']}")
    print(f"Saved full claim-level report: {out}")
    print("Per mutation type:")
    for k, v in report["per_mutation_type"].items():
        print(f"  {k}: n={v['n']} TP={v['tp']} FN={v['fn']} FP={v['fp']} recall={v['recall']:.3f} precision={v['precision']:.3f} F1={v['f1']:.3f}")

if __name__ == "__main__":
    main()
