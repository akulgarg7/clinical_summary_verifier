"""Offline tests for the optional, evidence-constrained semantic fallback."""
from __future__ import annotations

import json
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from src.semantic_adjudicator import (
    _eligible_items,
    apply_adjudication_payload,
    maybe_apply_semantic_adjudications,
)
from src.verification_engine import verify_claim_against_facts
from src.verification_pipeline import verify_claims_batch


def _frequency_batch_item(claim_value: str, source_value: str = "once daily") -> dict:
    claim = {
        "claim_id": "C-FREQ-1",
        "category": "medication",
        "subject": "test medication",
        "relation": "frequency",
        "value": claim_value,
        "unit": "",
        "route": "",
        "frequency": "",
        "duration": "",
        "date_time": "",
        "time_of_day": "",
        "negated": False,
        "source_text": f"test medication frequency {claim_value}",
    }
    fact = {
        "fact_id": "F-FREQ-1",
        "category": "medication",
        "subject": "test medication",
        "relation": "frequency",
        "value": source_value,
        "unit": "",
        "source_section": "discharge_medications",
    }
    candidate = {"fact": fact, "score": 1.0, "match_reasons": ["subject", "relation"]}
    result = verify_claim_against_facts(claim, [candidate])
    return {
        "claim": claim,
        "candidates": [candidate],
        "result": result,
        "error": None,
    }


def _mock_gemini_response(payload: dict):
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(payload))
        )]
    )
    return client



class SemanticAdjudicationTests(unittest.TestCase):
    def test_unfamiliar_frequency_wording_is_eligible(self):
        item = _frequency_batch_item("one time in a day")
        self.assertEqual(item["result"]["verdict"], "CONTRADICTED")
        self.assertTrue(item["result"]["field_results"]["frequency"]["semantic_review_candidate"])
        self.assertEqual(len(_eligible_items([item])), 1)

    def test_clear_frequency_conflict_is_not_eligible(self):
        item = _frequency_batch_item("once daily", "twice daily")
        self.assertEqual(item["result"]["verdict"], "CONTRADICTED")
        self.assertFalse(item["result"]["field_results"]["frequency"]["semantic_review_candidate"])
        self.assertEqual(_eligible_items([item]), [])

    def test_every_24_hours_matches_once_daily_without_llm(self):
        item = _frequency_batch_item("every 24 hours", "once daily")
        self.assertEqual(item["result"]["verdict"], "SUPPORTED")
        self.assertEqual(_eligible_items([item]), [])

    def test_interval_versus_daily_count_is_escalated_for_semantic_review(self):
        item = _frequency_batch_item("every 12 hours", "twice daily")
        self.assertEqual(item["result"]["verdict"], "CONTRADICTED")
        self.assertTrue(item["result"]["field_results"]["frequency"]["semantic_review_candidate"])
        self.assertEqual(len(_eligible_items([item])), 1)

    def test_two_different_explicit_intervals_remain_deterministic(self):
        item = _frequency_batch_item("every 6 hours", "every 8 hours")
        self.assertEqual(item["result"]["verdict"], "CONTRADICTED")
        self.assertFalse(item["result"]["field_results"]["frequency"]["semantic_review_candidate"])
        self.assertEqual(_eligible_items([item]), [])

    def test_supported_adjudication_requires_submitted_evidence_citation(self):
        item = _frequency_batch_item("one time in a day")
        applied = apply_adjudication_payload(
            [item],
            {"adjudications": [{
                "claim_id": "C-FREQ-1",
                "verdict": "SUPPORTED",
                "evidence_fact_ids": ["NOT-SUBMITTED"],
                "rationale": "Untrusted or invented citation.",
            }]},
            {"C-FREQ-1": {"F-FREQ-1"}},
        )
        self.assertEqual(applied, 0)
        self.assertEqual(item["result"]["verdict"], "CONTRADICTED")

    def test_supported_adjudication_resolves_semantic_frequency_mismatch(self):
        item = _frequency_batch_item("one time in a day")
        applied = apply_adjudication_payload(
            [item],
            {"adjudications": [{
                "claim_id": "C-FREQ-1",
                "verdict": "SUPPORTED",
                "evidence_fact_ids": ["F-FREQ-1"],
                "rationale": "The expressions describe the same once-daily cadence.",
            }]},
            {"C-FREQ-1": {"F-FREQ-1"}},
        )
        self.assertEqual(applied, 1)
        self.assertEqual(item["result"]["verdict"], "SUPPORTED")
        self.assertTrue(item["result"]["field_results"]["frequency"]["semantic_adjudicated"])
        self.assertTrue(item["result"]["semantic_adjudication"]["verdict_changed"])
        self.assertEqual(item["result"]["semantic_adjudication"]["final_verdict"], "SUPPORTED")

    def test_semantic_contradiction_can_be_applied_only_to_flagged_candidate(self):
        item = _frequency_batch_item("one time in a day")
        applied = apply_adjudication_payload(
            [item],
            {"adjudications": [{
                "claim_id": "C-FREQ-1",
                "verdict": "CONTRADICTED",
                "evidence_fact_ids": ["F-FREQ-1"],
                "rationale": "The source schedule conflicts with the asserted schedule.",
            }]},
            {"C-FREQ-1": {"F-FREQ-1"}},
        )
        self.assertEqual(applied, 1)
        self.assertEqual(item["result"]["verdict"], "CONTRADICTED")
        self.assertEqual(item["result"]["semantic_adjudication"]["verdict"], "CONTRADICTED")
        self.assertFalse(item["result"]["semantic_adjudication"]["verdict_changed"])

    def test_llm_cannot_override_clear_deterministic_contradiction(self):
        item = _frequency_batch_item("once daily", "twice daily")
        applied = apply_adjudication_payload(
            [item],
            {"adjudications": [{
                "claim_id": "C-FREQ-1",
                "verdict": "SUPPORTED",
                "evidence_fact_ids": ["F-FREQ-1"],
                "rationale": "Attempted override.",
            }]},
            {"C-FREQ-1": {"F-FREQ-1"}},
        )
        self.assertEqual(applied, 0)
        self.assertEqual(item["result"]["verdict"], "CONTRADICTED")

    def test_insufficient_evidence_is_preserved_as_an_abstention(self):
        item = _frequency_batch_item("one time in a day")
        applied = apply_adjudication_payload(
            [item],
            {"adjudications": [{
                "claim_id": "C-FREQ-1",
                "verdict": "INSUFFICIENT_EVIDENCE",
                "evidence_fact_ids": [],
                "rationale": "The retrieved facts do not establish the meaning confidently.",
            }]},
            {"C-FREQ-1": {"F-FREQ-1"}},
        )
        self.assertEqual(applied, 1)
        self.assertEqual(item["result"]["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(item["result"]["semantic_adjudication"]["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertTrue(item["result"]["semantic_adjudication"]["verdict_changed"])

    def test_enabled_fallback_uses_gemini_response_without_network_access(self):
        item = _frequency_batch_item("one time in a day")
        client = _mock_gemini_response({"adjudications": [{
            "claim_id": "C-FREQ-1",
            "verdict": "SUPPORTED",
            "evidence_fact_ids": ["F-FREQ-1"],
            "rationale": "Both phrases express one administration per day.",
        }]})
        fake_openai = ModuleType("openai")
        fake_openai.OpenAI = Mock(return_value=client)
        with patch.dict(os.environ, {
            "SEMANTIC_VERIFICATION_ENABLED": "true",
            "GEMINI_API_KEY": "test-key",
            "GEMINI_MODEL": "test-model",
            "SEMANTIC_VERIFICATION_MAX_CLAIMS": "10",
        }), patch.dict(sys.modules, {"openai": fake_openai}):
            returned, status = maybe_apply_semantic_adjudications([item])

        self.assertIs(returned[0], item)
        self.assertEqual(returned[0]["result"]["verdict"], "SUPPORTED")
        self.assertEqual(status["supported"], 1)
        self.assertEqual(status["applied"], 1)
        client.chat.completions.create.assert_called_once()

    def test_accepted_abstention_is_counted_consistently_in_batch_status(self):
        item = _frequency_batch_item("one time in a day")
        client = _mock_gemini_response({"adjudications": [{
            "claim_id": "C-FREQ-1",
            "verdict": "INSUFFICIENT_EVIDENCE",
            "evidence_fact_ids": [],
            "rationale": "The evidence does not establish the schedule with sufficient certainty.",
        }]})
        fake_openai = ModuleType("openai")
        fake_openai.OpenAI = Mock(return_value=client)
        with patch.dict(os.environ, {
            "SEMANTIC_VERIFICATION_ENABLED": "true",
            "GEMINI_API_KEY": "test-key",
            "GEMINI_MODEL": "test-model",
            "SEMANTIC_VERIFICATION_MAX_CLAIMS": "10",
        }), patch.dict(sys.modules, {"openai": fake_openai}):
            returned, status = maybe_apply_semantic_adjudications([item])

        self.assertEqual(returned[0]["result"]["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(status["accepted"], 1)
        self.assertEqual(status["applied"], 1)
        self.assertEqual(status["decisive"], 0)
        self.assertEqual(status["abstained"], 1)

    def test_fallback_is_disabled_by_default(self):
        item = _frequency_batch_item("one time in a day")
        with patch.dict(os.environ, {"SEMANTIC_VERIFICATION_ENABLED": "false"}, clear=False):
            returned, status = maybe_apply_semantic_adjudications([item])
        self.assertIs(returned[0], item)
        self.assertFalse(status["enabled"])
        self.assertEqual(status["reason"], "disabled")

    def test_pipeline_keeps_deterministic_results_when_gemini_fallback_fails(self):
        claim = {"claim_id": "C-PIPE", "subject": "test"}
        deterministic_result = {"claim_id": "C-PIPE", "verdict": "SUPPORTED", "field_results": {}}
        with patch("src.verification_pipeline.retrieve_candidate_facts", return_value=[]), \
             patch("src.verification_pipeline.verify_claim_against_facts", return_value=deterministic_result), \
             patch("src.verification_pipeline.maybe_apply_semantic_adjudications", side_effect=RuntimeError("API unavailable")):
            batch, status = verify_claims_batch([claim], [], record={})

        self.assertEqual(batch[0]["result"]["verdict"], "SUPPORTED")
        self.assertEqual(status["reason"], "semantic fallback failed; deterministic results retained")
        self.assertEqual(status["error"], "RuntimeError")

    def test_pipeline_isolates_one_bad_claim_and_continues(self):
        claims = [{"claim_id": "C-GOOD"}, "not-a-claim"]
        deterministic_result = {"claim_id": "C-GOOD", "verdict": "SUPPORTED", "field_results": {}}
        with patch("src.verification_pipeline.retrieve_candidate_facts", return_value=[]), \
             patch("src.verification_pipeline.verify_claim_against_facts", return_value=deterministic_result), \
             patch("src.verification_pipeline.maybe_apply_semantic_adjudications", side_effect=lambda results: (results, {"enabled": False, "applied": 0})):
            batch, _ = verify_claims_batch(claims, [], record={})

        self.assertEqual(batch[0]["result"]["verdict"], "SUPPORTED")
        self.assertIsNone(batch[0]["error"])
        self.assertIsNone(batch[1]["result"])
        self.assertIn("not a JSON object", batch[1]["error"])


if __name__ == "__main__":
    unittest.main()
