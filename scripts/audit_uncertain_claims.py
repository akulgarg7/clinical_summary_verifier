"""Generate a reproducible review file for PARTIALLY_SUPPORTED and INSUFFICIENT_EVIDENCE claims.

This script flags potential review targets from verifier outputs. It deliberately does
not assign gold verdicts based on string overlap; such flags need independent review.
"""
from __future__ import annotations
import csv, json, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BASELINE = RESULTS / "end_to_end_evaluation_report.json"
CURRENT = RESULTS / "end_to_end_after_fixes.json"
OUT_CSV = RESULTS / "uncertain_claims_after_fixes.csv"
OUT_MD = RESULTS / "UNCERTAIN_VERDICT_AUDIT_AFTER_FIXES.md"


def key(x):
    return (x.get("claim_file", ""), x.get("claim_id", ""))


def norm(s):
    s = str(s or "").casefold()
    s = s.replace("–", "-").replace("—", "-").replace("‑", "-").replace("−", "-")
    return " ".join(re.findall(r"[a-z0-9.%:+/-]+", s))


def values_equal(a, b):
    na, nb = norm(a), norm(b)
    if na == nb and na:
        return True
    try:
        return float(na) == float(nb)
    except (ValueError, TypeError):
        return False


def candidate_flags(row):
    claim_value = row.get("value", "")
    subject = norm(row.get("subject", ""))
    rel = norm(row.get("relation", "")).replace("_", " ")
    top = row.get("top_candidates") or []
    exact_value = []
    same_subject_value = []
    same_relation_value = []
    for c in top:
        csub = norm(c.get("subject", ""))
        crel = norm(c.get("relation", "")).replace("_", " ")
        cval = c.get("value", "")
        if values_equal(claim_value, cval):
            exact_value.append(c.get("fact_id", ""))
            if subject and (subject in csub or csub in subject):
                same_subject_value.append(c.get("fact_id", ""))
            if rel and (rel == crel or rel in crel or crel in rel):
                same_relation_value.append(c.get("fact_id", ""))
    if row.get("candidate_count", 0) == 0:
        priority = "HIGH: no candidates retrieved"
    elif same_subject_value:
        priority = "HIGH: exact value among subject-compatible candidates"
    elif exact_value:
        priority = "MEDIUM: exact value appears in retrieved evidence; verify entity/timepoint"
    elif row.get("verdict") == "PARTIALLY_SUPPORTED":
        priority = "MEDIUM: inspect field-level mix of supported/uncertain attributes"
    else:
        priority = "REVIEW: no simple exact-value signal"
    return priority, exact_value, same_subject_value, same_relation_value


def main():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    old_map = {key(x): x for x in baseline["claim_predictions"]}
    current_rows = current["claim_predictions"]
    transition = Counter((old_map[key(x)].get("verdict", ""), x.get("verdict", "")) for x in current_rows if key(x) in old_map)
    uncertain = [x for x in current_rows if x.get("verdict") in {"PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE"}]
    fields = ["claim_file", "summary_file", "case_id", "claim_id", "category", "subject", "relation", "value", "unit", "route", "frequency", "duration", "date_time", "source_text", "verdict", "previous_verdict", "verdict_changed", "candidate_count", "matched_fact_ids", "error_type", "source_text_valid", "review_priority", "exact_value_candidate_fact_ids", "subject_compatible_exact_value_fact_ids", "relation_compatible_exact_value_fact_ids", "field_results_json", "top_candidates_json"]
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in uncertain:
            old = old_map.get(key(row), {})
            priority, exact, subject_exact, rel_exact = candidate_flags(row)
            writer.writerow({
                "claim_file": row.get("claim_file", ""), "summary_file": row.get("summary_file", ""),
                "case_id": row.get("case_id", ""), "claim_id": row.get("claim_id", ""),
                "category": row.get("category", ""), "subject": row.get("subject", ""),
                "relation": row.get("relation", ""), "value": row.get("value", ""), "unit": row.get("unit", ""),
                "route": row.get("route", ""), "frequency": row.get("frequency", ""), "duration": row.get("duration", ""),
                "date_time": row.get("date_time", ""), "source_text": row.get("source_text", ""),
                "verdict": row.get("verdict", ""), "previous_verdict": old.get("verdict", ""),
                "verdict_changed": old.get("verdict") != row.get("verdict"),
                "candidate_count": row.get("candidate_count", 0),
                "matched_fact_ids": ";".join(row.get("matched_fact_ids", [])),
                "error_type": row.get("error_type", ""), "source_text_valid": row.get("source_text_valid", ""),
                "review_priority": priority, "exact_value_candidate_fact_ids": ";".join(exact),
                "subject_compatible_exact_value_fact_ids": ";".join(subject_exact),
                "relation_compatible_exact_value_fact_ids": ";".join(rel_exact),
                "field_results_json": json.dumps(row.get("field_results", {}), ensure_ascii=False),
                "top_candidates_json": json.dumps(row.get("top_candidates", []), ensure_ascii=False),
            })
    verdicts = Counter(x.get("verdict", "") for x in current_rows)
    priorities = Counter(candidate_flags(x)[0].split(":", 1)[0] for x in uncertain)
    no_candidate = sum(1 for x in uncertain if not x.get("candidate_count", 0))
    exact_value = sum(1 for x in uncertain if candidate_flags(x)[1])
    subject_exact = sum(1 for x in uncertain if candidate_flags(x)[2])
    lines = [
        "# Uncertain-verdict audit after generalized fixes", "",
        "This report was generated from the latest saved-claim verification output. It is an audit aid, not independent adjudication.", "",
        "## Current verdicts", "",
        f"- Claims verified: **{len(current_rows)}**", f"- Supported: **{verdicts['SUPPORTED']}**",
        f"- Contradicted: **{verdicts['CONTRADICTED']}**", f"- Partially supported: **{verdicts['PARTIALLY_SUPPORTED']}**",
        f"- Insufficient evidence: **{verdicts['INSUFFICIENT_EVIDENCE']}**",
        f"- Uncertain claims listed in CSV: **{len(uncertain)}**", "",
        "## Change from initial saved-claim report", "",
        "| Previous verdict | Current verdict | Count |", "|---|---|---:|",
    ]
    for (old, new), count in sorted(transition.items()):
        lines.append(f"| {old} | {new} | {count} |")
    lines += ["", "## Review-priority signals in uncertain claims", "",
              f"- No candidates retrieved: **{no_candidate}**",
              f"- At least one exact value among top retrieved candidates: **{exact_value}**",
              f"- Exact value in a candidate with subject-compatible wording: **{subject_exact}**", "",
              "These are automatic review flags only. Exact numeric/text overlap does not prove support if the entity, unit, relation, timepoint, polarity, or context differs.", "",
              "## Interpretation and limitations", "",
              "The number of `INSUFFICIENT_EVIDENCE` and `PARTIALLY_SUPPORTED` verdicts is not itself an error count. Some are justified abstentions (for example, a normality claim when the source gives a number but no normal range or interpretation); others indicate retrieval or schema-alignment failures. The CSV includes source text, field-level results, candidate facts, and previous verdicts to support manual review.", "",
              "Do not calculate overall precision/recall/F1 using unlabelled claims as if the system's own verdicts were gold labels. The three registered mutation targets are the only authoritative end-to-end injected-error labels in this dataset.", "",
              "## Reproducibility", "",
              "Run `python scripts/evaluate_end_to_end.py --output results/end_to_end_after_fixes.json` first, then run `python scripts/audit_uncertain_claims.py`. The script uses only the saved ground-truth-derived evaluation report and performs no API calls.", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rows in uncertain audit: {len(uncertain)}")
    print(f"CSV: {OUT_CSV}")
    print(f"Markdown: {OUT_MD}")
    print("Transitions:")
    for (old, new), count in sorted(transition.items()): print(f"  {old} -> {new}: {count}")
    print("Review flags:", dict(priorities))
    print("No candidates:", no_candidate, "exact value candidate:", exact_value, "subject-compatible exact value:", subject_exact)

if __name__ == "__main__":
    main()
