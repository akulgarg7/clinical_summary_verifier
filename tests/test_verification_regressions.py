"""Offline regression tests for independent clinical evidence retrieval/verification.

The tests use only the project's synthetic ground-truth records and saved claims.
They never call Gemini or another external API.
"""
from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from src.evidence_builder import (
    GROUND_TRUTH_PATH,
    match_subject,
    retrieve_candidate_facts,
)
from src.verification_engine import (
    _canonical_frequency,
    _canonical_route,
    _canonical_unit,
    verify_claim_against_facts,
)

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_DIR = ROOT / "data" / "claims"


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def claim_list(filename: str):
    payload = read_json(CLAIMS_DIR / filename)
    return payload.get("claims", payload) if isinstance(payload, dict) else payload


class VerificationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        records = read_json(GROUND_TRUTH_PATH)
        cls.records = {
            item["case_metadata"]["case_id"]: item for item in records
        }
        cls.e003_claims = {
            item["claim_id"]: item
            for item in claim_list("SYN-DS-0003_E003_claims.json")
        }

    def verify(self, case_id: str, claim: dict):
        record = self.records[case_id]
        candidates = retrieve_candidate_facts(
            claim,
            record.get("ground_truth_facts", []),
            top_k=10,
            record=record,
        )
        result = verify_claim_against_facts(
            claim,
            candidates,
            record=record,
        )
        return result, candidates

    def test_potassium_does_not_match_creatinine_peak(self):
        claim = self.e003_claims["C045"]
        result, candidates = self.verify("SYN-DS-0003", claim)
        candidate_ids = {
            item.get("fact", {}).get("fact_id") for item in candidates
        }
        self.assertIn("SYN-DS-0003-F063", candidate_ids)
        self.assertNotIn("SYN-DS-0003-F057", candidate_ids)
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0003-F063", result["matched_fact_ids"])

    def test_injected_admission_creatinine_error_stays_contradicted(self):
        result, _ = self.verify("SYN-DS-0003", self.e003_claims["C041"])
        self.assertEqual(result["verdict"], "CONTRADICTED")
        self.assertEqual(result["error_type"], "result_mismatch")
        self.assertIn("SYN-DS-0003-F055", result["matched_fact_ids"])

    def test_reported_false_positive_family_is_no_longer_contradicted(self):
        # C063 is deliberately only partial: the record stores the bolus as
        # 30 mL/kg and as 1860 mL in separate facts; do not overstate that match.
        for claim_id in ("C035", "C045", "C058", "C065", "C066", "C067", "C068", "C069", "C070", "C071", "C072"):
            with self.subTest(claim_id=claim_id):
                result, _ = self.verify(
                    "SYN-DS-0003", self.e003_claims[claim_id]
                )
                self.assertNotEqual(result["verdict"], "CONTRADICTED")
        bolus_result, _ = self.verify("SYN-DS-0003", self.e003_claims["C063"])
        self.assertEqual(bolus_result["verdict"], "PARTIALLY_SUPPORTED")

    def test_e001_and_e002_each_keep_only_the_injected_medication_contradiction(self):
        expected = {
            "SYN-DS-0006_E001_claims.json": {"C065"},
            "SYN-DS-0006_E002_claims.json": {"C107"},
        }
        for filename, expected_ids in expected.items():
            with self.subTest(filename=filename):
                claims = claim_list(filename)
                contradicted = set()
                for claim in claims:
                    result, _ = self.verify("SYN-DS-0006", claim)
                    if result["verdict"] == "CONTRADICTED":
                        contradicted.add(claim.get("claim_id"))
                self.assertEqual(contradicted, expected_ids)

    def test_subject_matching_uses_clinical_tokens_not_substrings(self):
        matched, *_ = match_subject("K⁺", "Serum creatinine (peak)", "2.1")
        self.assertFalse(matched)
        matched, *_ = match_subject("K⁺", "Serum potassium", "4.6")
        self.assertTrue(matched)

    def test_common_frequency_route_and_unit_variants_normalize(self):
        self.assertEqual(
            _canonical_frequency("q6h PRN"),
            _canonical_frequency("every 6 hours as needed"),
        )
        self.assertEqual(
            _canonical_frequency("every 48 hours (renally adjusted)"),
            _canonical_frequency("every 48 hours"),
        )
        self.assertEqual(
            _canonical_frequency("once daily before breakfast"),
            _canonical_frequency("daily"),
        )
        self.assertEqual(
            _canonical_route("IV"),
            _canonical_route("intravenous via central line"),
        )
        self.assertEqual(_canonical_unit("×10⁹/L"), _canonical_unit("x10^9/L"))

    def test_injected_rivaroxaban_dose_error_stays_detected(self):
        claims = claim_list("SYN-DS-0006_E001_claims.json")
        claim = next(
            item for item in claims
            if item.get("claim_id") == "C065"
            and "rivaroxaban" in str(item.get("subject", "")).lower()
        )
        result, _ = self.verify("SYN-DS-0006", claim)
        self.assertEqual(result["verdict"], "CONTRADICTED")
        self.assertEqual(result["error_type"], "dose_mismatch")

    def test_injected_rivaroxaban_frequency_error_stays_detected(self):
        claims = claim_list("SYN-DS-0006_E002_claims.json")
        claim = next(item for item in claims if item.get("claim_id") == "C107")
        result, _ = self.verify("SYN-DS-0006", claim)
        self.assertEqual(result["verdict"], "CONTRADICTED")
        self.assertEqual(result["error_type"], "frequency_mismatch")

    def test_e003_current_verdict_distribution_has_only_known_numeric_conflict(self):
        results = [
            self.verify("SYN-DS-0003", claim)[0]
            for claim in self.e003_claims.values()
        ]
        contradicted = [
            item["claim_id"] for item in results
            if item["verdict"] == "CONTRADICTED"
        ]
        # This injected creatinine mismatch is the intentionally modified fact.
        self.assertEqual(contradicted, ["C041"])
        counts = Counter(item["verdict"] for item in results)
        self.assertEqual(sum(counts.values()), len(self.e003_claims))

    def _verify_saved_claim(self, case_id, claims_filename, claim_id):
        claims = claim_list(claims_filename)
        claim = next(item for item in claims if item.get("claim_id") == claim_id)
        return self.verify(case_id, claim)

    def test_gram_and_milligram_representation_does_not_create_dose_mismatch(self):
        from src.verification_engine import _compare_claim_field
        direct_verdict = _compare_claim_field(
            "1", "g", {"value": "1000", "unit": "mg"}, "dose"
        )
        self.assertEqual(direct_verdict, "SUPPORTED")
        result, _ = self._verify_saved_claim(
            "SYN-DS-0001", "SYN-DS-0001_claims.json", "C068"
        )
        # The full claim may still abstain because the original sentence
        # combines IV-to-oral transition details into one extracted claim.
        self.assertNotEqual(result["verdict"], "CONTRADICTED")

    def test_ecg_rate_is_matched_to_the_ecg_record_not_later_hospital_rate(self):
        result, _ = self._verify_saved_claim(
            "SYN-DS-0005", "SYN-DS-0005_claims.json", "C049"
        )
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0005-F068", result["matched_fact_ids"])

    def test_same_day_post_transfusion_blood_pressure_is_retrieved(self):
        result, _ = self._verify_saved_claim(
            "SYN-DS-0007", "SYN-DS-0007_claims.json", "C072"
        )
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0007-F074", result["matched_fact_ids"])

    def test_positive_imaging_finding_conflicts_with_explicitly_negated_source(self):
        from src.verification_engine import _compare_claim_field
        fact = {
            "fact_id": "TEST-FINDING-1",
            "category": "imaging",
            "subject": "CT abdomen and pelvis",
            "relation": "findings",
            "value": "No extraluminal air; no periappendiceal fluid collection",
            "unit": "",
        }
        verdict = _compare_claim_field(
            "Extraluminal air or fluid collection", "", fact, "is"
        )
        self.assertEqual(verdict, "CONTRADICTED")

    def test_imaging_finding_polarity_is_scoped_to_its_own_clause(self):
        result, _ = self._verify_saved_claim(
            "SYN-DS-0007", "SYN-DS-0007_claims.json", "C053"
        )
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0007-F063", result["matched_fact_ids"])

    def test_narrative_symptom_duration_is_not_compared_to_medication_duration(self):
        result, _ = self._verify_saved_claim(
            "SYN-DS-0006", "SYN-DS-0006_claims.json", "C009"
        )
        self.assertNotEqual(result["verdict"], "CONTRADICTED")

    def test_composite_procedure_result_date_does_not_become_performed_date(self):
        claims = claim_list("SYN-DS-0003_claims.json")
        claim = next(item for item in claims if item.get("claim_id") == "C068")
        result, _ = self.verify("SYN-DS-0003", claim)
        self.assertNotEqual(result["verdict"], "CONTRADICTED")

    def test_admission_date_and_type_generic_is_relation_resolve_source_slots(self):
        date_result, _ = self.verify("SYN-DS-0001", claim_list("SYN-DS-0001_claims.json")[0])
        claims = {item["claim_id"]: item for item in claim_list("SYN-DS-0001_claims.json")}
        date_result, _ = self.verify("SYN-DS-0001", claims["C001"])
        type_result, _ = self.verify("SYN-DS-0001", claims["C002"])
        self.assertEqual(date_result["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0001-F007", date_result["matched_fact_ids"])
        self.assertIn("SYN-DS-0001-F008", date_result["matched_fact_ids"])
        self.assertEqual(type_result["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0001-F006", type_result["matched_fact_ids"])

    def test_wbc_same_month_different_day_does_not_create_ambiguity(self):
        claims = {item["claim_id"]: item for item in claim_list("SYN-DS-0001_claims.json")}
        result, _ = self.verify("SYN-DS-0001", claims["C031"])
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0001-F037", result["matched_fact_ids"])
        self.assertNotIn("SYN-DS-0001-F050", result["matched_fact_ids"])

    def test_troponin_timepoint_selects_matching_result(self):
        claims = {item["claim_id"]: item for item in claim_list("SYN-DS-0002_claims.json")}
        first, _ = self.verify("SYN-DS-0002", claims["C040"])
        later, _ = self.verify("SYN-DS-0002", claims["C041"])
        self.assertEqual(first["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0002-F048", first["matched_fact_ids"])
        self.assertNotIn("SYN-DS-0002-F050", first["matched_fact_ids"])
        self.assertEqual(later["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0002-F050", later["matched_fact_ids"])
        self.assertNotIn("SYN-DS-0002-F048", later["matched_fact_ids"])

    def test_initial_potassium_uses_initial_not_six_hour_result(self):
        claims = {item["claim_id"]: item for item in claim_list("SYN-DS-0004_claims.json")}
        initial, _ = self.verify("SYN-DS-0004", claims["C044"])
        later, _ = self.verify("SYN-DS-0004", claims["C045"])
        self.assertEqual(initial["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0004-F051", initial["matched_fact_ids"])
        self.assertNotIn("SYN-DS-0004-F060", initial["matched_fact_ids"])
        self.assertEqual(later["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0004-F060", later["matched_fact_ids"])

    def test_laboratory_concentration_is_not_misresolved_as_medication_dose(self):
        claim = {
            "claim_id": "CONTROL-K-POTASSIUM-6H",
            "category": "laboratory",
            "subject": "Serum potassium (6 hours)",
            "relation": "result",
            "value": "3.8",
            "unit": "mmol/L",
            "route": "",
            "frequency": "",
            "duration": "",
            "date_time": "2026-01-10",
            "negated": False,
            "source_text": "Serum potassium (6 hours) result 3.8 mmol/L 2026-01-10",
        }
        result, _ = self.verify("SYN-DS-0004", claim)
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertNotEqual(result.get("error_type"), "dose_mismatch")
        self.assertIn("SYN-DS-0004-F060", result["matched_fact_ids"])

    def test_mmol_per_litre_iv_fluid_unit_maps_to_concentration_unit(self):
        self.assertEqual(_canonical_unit("mmol per litre of IV fluid"), "mmol/l")
        claim = next(item for item in claim_list("SYN-DS-0004_claims.json") if item["claim_id"] == "C067")
        result, _ = self.verify("SYN-DS-0004", claim)
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0004-F113", result["matched_fact_ids"])

    def test_symptom_duration_claim_retrieves_chief_complaint_fact(self):
        claims = claim_list("SYN-DS-0001_claims.json")
        claim = next(item for item in claims if item.get("claim_id") == "C005")
        result, candidates = self.verify("SYN-DS-0001", claim)
        candidate_ids = {item["fact"].get("fact_id") for item in candidates}
        self.assertIn("SYN-DS-0001-F011", candidate_ids)
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0001-F011", result["matched_fact_ids"])

    def test_allergy_presence_relation_matches_source_status(self):
        claims = claim_list("SYN-DS-0001_claims.json")
        claim = next(item for item in claims if item.get("claim_id") == "C017")
        result, _ = self.verify("SYN-DS-0001", claim)
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0001-F020", result["matched_fact_ids"])

    def test_duration_embedded_in_supported_instruction_does_not_downgrade_verdict(self):
        claims = claim_list("SYN-DS-0001_claims.json")
        claim = next(item for item in claims if item.get("claim_id") == "C0101")
        result, _ = self.verify("SYN-DS-0001", claim)
        self.assertEqual(result["verdict"], "SUPPORTED")

    def test_discharge_condition_relation_alias_supports_discharge_status(self):
        claims = claim_list("SYN-DS-0002_claims.json")
        claim = next(item for item in claims if item.get("claim_id") == "C092")
        result, _ = self.verify("SYN-DS-0002", claim)
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0002-F179", result["matched_fact_ids"])

    def test_normal_glucose_remains_uncertain_without_explicit_interpretation(self):
        # A numeric glucose fact alone does not prove the qualitative label
        # "normal" without a reference interval or an explicit interpretation.
        claims = claim_list("SYN-DS-0005_claims.json")
        claim = next(item for item in claims if item.get("claim_id") == "C051")
        result, _ = self.verify("SYN-DS-0005", claim)
        self.assertEqual(result["verdict"], "INSUFFICIENT_EVIDENCE")


    def test_baseline_creatinine_relation_alias_resolves_exact_source_fact(self):
        claims = claim_list("SYN-DS-0003_claims.json")
        claim = next(item for item in claims if item.get("claim_id") == "C021")
        result, _ = self.verify("SYN-DS-0003", claim)
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0003-F207", result["matched_fact_ids"])

    def test_admission_creatinine_has_value_relation_resolves_result(self):
        claims = claim_list("SYN-DS-0003_claims.json")
        claim = next(item for item in claims if item.get("claim_id") == "C043")
        result, _ = self.verify("SYN-DS-0003", claim)
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0003-F055", result["matched_fact_ids"])

    def test_blood_pressure_does_not_match_white_blood_cell_count(self):
        matched, *_ = match_subject("BP", "White blood cell count", "19.6")
        self.assertFalse(matched)

    def test_matching_date_with_unavailable_clock_time_is_not_contradicted(self):
        from src.verification_engine import _compare_claim_field
        verdict = _compare_claim_field(
            "14/02/2026 09:05", "",
            {"value": "2026-02-14", "unit": ""},
            "date",
        )
        self.assertEqual(verdict, "INSUFFICIENT_EVIDENCE")

    def test_admission_date_and_time_in_slash_format_are_combined(self):
        claims = claim_list("SYN-DS-0005_claims.json")
        claim = next(item for item in claims if item.get("claim_id") == "C006")
        result, _ = self.verify("SYN-DS-0005", claim)
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0005-F007", result["matched_fact_ids"])

    def test_ct_confirmed_indication_matches_on_ct_source_wording(self):
        claims = claim_list("SYN-DS-0001_claims.json")
        claim = next(item for item in claims if item.get("claim_id") == "C077")
        result, _ = self.verify("SYN-DS-0001", claim)
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0001-F069", result["matched_fact_ids"])

    def test_iv_dextrose_does_not_retrieve_potassium_chloride(self):
        matched, *_ = match_subject(
            "IV dextrose 10%",
            "Potassium chloride (inpatient, 40 mmol per litre of IV fluid intravenous, from 2026-01-10)",
            "40",
        )
        self.assertFalse(matched)

    def test_nacl_alias_matches_sodium_chloride_source(self):
        matched, *_ = match_subject(
            "0.9% NaCl",
            "Sodium chloride 0.9% (inpatient, 125 mL/hour intravenous, from 2025-03-12)",
            "125",
        )
        self.assertTrue(matched)

    def test_transition_dose_context_disambiguates_same_drug_regimen(self):
        claims = claim_list("SYN-DS-0004_claims.json")
        claim = next(item for item in claims if item.get("claim_id") == "C079")
        result, _ = self.verify("SYN-DS-0004", claim)
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertIn("SYN-DS-0004-F134", result["matched_fact_ids"])

    def test_embedded_frequency_in_supported_instruction_does_not_downgrade(self):
        claims = claim_list("SYN-DS-0001_claims.json")
        claim = next(item for item in claims if item.get("claim_id") == "C0102")
        result, _ = self.verify("SYN-DS-0001", claim)
        self.assertEqual(result["verdict"], "SUPPORTED")

    def test_wound_care_synonym_preserves_supported_instruction(self):
        claims = claim_list("SYN-DS-0001_claims.json")
        claim = next(item for item in claims if item.get("claim_id") == "C0103")
        result, _ = self.verify("SYN-DS-0001", claim)
        self.assertEqual(result["verdict"], "SUPPORTED")

    def test_diagnosis_relation_uses_diagnosis_facts_not_negative_history(self):
        claims = claim_list("SYN-DS-0005_claims.json")
        claim = next(item for item in claims if item.get("claim_id") == "C010")
        result, _ = self.verify("SYN-DS-0005", claim)
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertTrue(
            set(result["matched_fact_ids"]) & {"SYN-DS-0005-F070", "SYN-DS-0005-F145"}
        )


class SemanticFallbackSafetyTests(unittest.TestCase):
    def _batch_item(self, verdict="INSUFFICIENT_EVIDENCE"):
        return {
            "claim": {"claim_id": "C-X", "source_text": "sample claim"},
            "candidates": [{
                "score": 2.0,
                "fact": {
                    "fact_id": "F-X",
                    "subject": "sample entity",
                    "relation": "result",
                    "value": "sample value",
                    "unit": "",
                },
            }],
            "result": {
                "claim_id": "C-X",
                "verdict": verdict,
                "error_type": "value_mismatch" if verdict == "CONTRADICTED" else None,
                "matched_fact_ids": [],
                "field_results": {
                    "result": {
                        "claim_value": "sample value",
                        "source_value": None,
                        "verdict": "INSUFFICIENT_EVIDENCE",
                        "matched_fact_id": None,
                        "matched_fact_ids": [],
                    }
                },
            },
            "error": None,
        }

    def test_semantic_fallback_requires_citation_to_submitted_evidence(self):
        from src.semantic_adjudicator import apply_adjudication_payload

        item = self._batch_item()
        applied = apply_adjudication_payload(
            [item],
            {"adjudications": [{
                "claim_id": "C-X",
                "verdict": "SUPPORTED",
                "supporting_fact_ids": ["MADE-UP-FACT"],
            }]},
            {"C-X": {"F-X"}},
        )
        self.assertEqual(applied, 0)
        self.assertEqual(item["result"]["verdict"], "INSUFFICIENT_EVIDENCE")

    def test_semantic_fallback_never_overrides_deterministic_contradiction(self):
        from src.semantic_adjudicator import apply_adjudication_payload

        item = self._batch_item(verdict="CONTRADICTED")
        applied = apply_adjudication_payload(
            [item],
            {"adjudications": [{
                "claim_id": "C-X",
                "verdict": "SUPPORTED",
                "supporting_fact_ids": ["F-X"],
            }]},
            {"C-X": {"F-X"}},
        )
        self.assertEqual(applied, 0)
        self.assertEqual(item["result"]["verdict"], "CONTRADICTED")

    def test_semantic_fallback_can_resolve_one_uncertain_field_with_valid_citation(self):
        from src.semantic_adjudicator import apply_adjudication_payload

        item = self._batch_item()
        applied = apply_adjudication_payload(
            [item],
            {"adjudications": [{
                "claim_id": "C-X",
                "verdict": "SUPPORTED",
                "supporting_fact_ids": ["F-X"],
                "rationale": "Source fact entails the unresolved assertion.",
            }]},
            {"C-X": {"F-X"}},
        )
        self.assertEqual(applied, 1)
        self.assertEqual(item["result"]["verdict"], "SUPPORTED")
        self.assertTrue(item["result"]["field_results"]["result"]["semantic_adjudicated"])

class ConfigurationSafetyTests(unittest.TestCase):
    def test_claim_extractor_import_does_not_require_api_key(self):
        # Import itself is safe; only an attempted model call needs credentials.
        import os
        from unittest.mock import patch
        from src.claim_extractor import _get_gemini_client_and_model

        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GEMINI_MODEL": ""}):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY is not configured"):
                _get_gemini_client_and_model()

    def test_semantic_fallback_is_disabled_by_default_without_network_access(self):
        import os
        from unittest.mock import patch
        from src.semantic_adjudicator import maybe_apply_semantic_adjudications

        batch = []
        with patch.dict(os.environ, {"SEMANTIC_VERIFICATION_ENABLED": "false"}):
            returned, status = maybe_apply_semantic_adjudications(batch)
        self.assertIs(returned, batch)
        self.assertFalse(status["enabled"])
        self.assertEqual(status["reason"], "disabled")



class GeneralizedStructuredComparisonTests(unittest.TestCase):
    """Generic comparison rules; no patient/case-specific expectations."""

    def test_changed_date_is_contradicted_even_for_noncanonical_date_relation(self):
        from src.verification_engine import _compare_claim_field
        verdict = _compare_claim_field(
            "2026-05-21", "",
            {"value": "2026-05-20", "unit": ""},
            "discontinued on",
        )
        self.assertEqual(verdict, "CONTRADICTED")

    def test_scalar_measurement_mismatch_is_contradicted_for_generic_relation(self):
        from src.verification_engine import _compare_claim_field
        verdict = _compare_claim_field(
            "13", "mm", {"value": "12", "unit": "mm"}, "size"
        )
        self.assertEqual(verdict, "CONTRADICTED")

    def test_numeric_change_inside_matching_duration_phrase_is_not_supported(self):
        from src.verification_engine import _compare_claim_field
        verdict = _compare_claim_field(
            "15 days (until 2026-06-05)", "",
            {"value": "14 days (until 2026-06-05)", "unit": ""},
            "duration",
        )
        self.assertEqual(verdict, "CONTRADICTED")

    def test_known_opposite_demographic_categories_are_contradicted(self):
        from src.verification_engine import _compare_claim_field
        verdict = _compare_claim_field(
            "Female", "", {"value": "Male", "unit": ""}, "sex"
        )
        self.assertEqual(verdict, "CONTRADICTED")

    def test_binary_result_polarity_conflict_is_contradicted(self):
        from src.verification_engine import _compare_claim_field
        verdict = _compare_claim_field(
            "positive", "", {"value": "negative", "unit": ""}, "result"
        )
        self.assertEqual(verdict, "CONTRADICTED")

    def test_medication_status_relation_variant_normalizes_to_status(self):
        from src.verification_engine import _canonical_relation
        self.assertEqual(
            _canonical_relation("status relative to pre-admission regimen"),
            "status",
        )

    def test_frequency_equivalence_covers_single_dose_and_nightly(self):
        self.assertEqual(
            _canonical_frequency("single pre-operative dose"),
            _canonical_frequency("once (pre-operative)"),
        )
        self.assertEqual(
            _canonical_frequency("nightly"),
            _canonical_frequency("once daily at night"),
        )
        self.assertEqual(
            _canonical_frequency("BID 30 min before breakfast & dinner × 14 days"),
            _canonical_frequency("twice daily before breakfast and dinner"),
        )

    def test_prn_indication_and_interval_notation_are_equivalent(self):
        self.assertEqual(
            _canonical_frequency("every 4 hours PRN for severe pain"),
            _canonical_frequency("every 4 hours as needed for severe pain"),
        )
        self.assertEqual(
            _canonical_frequency("every 20 min × 3 doses"),
            _canonical_frequency("every 20 minutes for 3 doses"),
        )
        self.assertEqual(
            _canonical_frequency("q6h PRN (max 3 g/24h)"),
            _canonical_frequency("every 6 hours as needed"),
        )

    def test_inhaled_device_variants_do_not_create_route_contradictions(self):
        self.assertEqual(_canonical_route("via MDI/spacer"), "inhaled")
        self.assertEqual(
            _canonical_route("via MDI/spacer"),
            _canonical_route("inhaled via metered-dose inhaler with spacer"),
        )
        self.assertEqual(
            _canonical_route("nebulised"),
            _canonical_route("nebulised (oxygen-driven)"),
        )

    def test_written_date_equivalent_to_iso_date(self):
        from src.verification_engine import _compare_claim_field
        verdict = _compare_claim_field(
            "12 March", "", {"value": "2025-03-12", "unit": ""}, "performed on"
        )
        self.assertEqual(verdict, "SUPPORTED")

    def test_status_phrase_variant_tolerating_is_not_a_contradiction(self):
        from src.verification_engine import _compare_claim_field
        verdict = _compare_claim_field(
            "tolerated", "",
            {"value": "Improved: Afebrile, tolerating soft diet, mobilising independently", "unit": ""},
            "status",
        )
        self.assertEqual(verdict, "SUPPORTED")

    def test_frequency_field_containing_only_a_time_note_does_not_claim_a_conflict(self):
        from src.verification_engine import _compare_claim_field
        verdict = _compare_claim_field(
            "first dose at 22:10", "",
            {"value": "every 48 hours (renally adjusted)", "unit": ""},
            "frequency",
        )
        self.assertEqual(verdict, "INSUFFICIENT_EVIDENCE")

    def test_total_course_duration_is_not_contradicted_by_discharge_remainder(self):
        from src.verification_engine import _compare_claim_field
        verdict = _compare_claim_field(
            "5-day course to 2026-07-06", "",
            {"value": "2 more days (2026-07-05 and 2026-07-06; completes 5-day course)", "unit": ""},
            "duration",
        )
        self.assertNotEqual(verdict, "CONTRADICTED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
