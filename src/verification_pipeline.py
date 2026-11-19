"""Reusable batch pipeline for independent evidence retrieval and verification.

This module centralizes the steps previously performed directly in the
Streamlit page: retrieve evidence, run deterministic claim verification, then
optionally run the evidence-constrained Gemini semantic fallback. It is safe to
use from the UI, evaluation scripts, and tests.
"""
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from src.evidence_builder import retrieve_candidate_facts
from src.semantic_adjudicator import maybe_apply_semantic_adjudications
from src.verification_engine import verify_claim_against_facts


def verify_claims_batch(
    claims: list[dict],
    ground_truth_facts: list[dict],
    *,
    record: dict | None = None,
    top_k: int = 20,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Retrieve, verify, and optionally adjudicate a batch of extracted claims.

    Claim-level retrieval/verification errors are retained as individual error
    entries so one malformed claim cannot stop the remaining batch. If the
    optional Gemini request fails, all deterministic verdicts are retained.
    """
    if not isinstance(claims, list):
        raise TypeError("claims must be a list of claim dictionaries")
    if not isinstance(ground_truth_facts, list):
        raise TypeError("ground_truth_facts must be a list of source fact dictionaries")

    batch_results: list[dict] = []
    total_claims = len(claims)

    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            batch_results.append({
                "claim": {},
                "candidates": [],
                "result": None,
                "error": "Claim entry is not a JSON object.",
                "source_type": "Discharge summary",
                "test_id": "",
            })
        else:
            try:
                candidates = retrieve_candidate_facts(
                    claim,
                    ground_truth_facts,
                    top_k=top_k,
                    record=record,
                )
                result = verify_claim_against_facts(
                    claim,
                    candidates,
                    record=record,
                )
                batch_results.append({
                    "claim": claim,
                    "candidates": candidates,
                    "result": result,
                    "error": None,
                    "source_type": "Discharge summary",
                    "test_id": "",
                })
            except Exception as exc:  # Keep the previous per-claim isolation behavior.
                batch_results.append({
                    "claim": claim,
                    "candidates": [],
                    "result": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "source_type": "Discharge summary",
                    "test_id": "",
                })

        if progress_callback is not None:
            progress_callback(index + 1, total_claims)

    # Work on a copy so even an unexpected adjudicator exception cannot mutate
    # the deterministic results that must be retained as the safe fallback.
    deterministic_results = batch_results
    try:
        semantic_working_copy = deepcopy(deterministic_results)
        semantic_results, semantic_status = maybe_apply_semantic_adjudications(
            semantic_working_copy
        )
        batch_results = semantic_results
    except Exception as exc:
        batch_results = deterministic_results
        semantic_status = {
            "enabled": True,
            "applied": 0,
            "supported": 0,
            "contradicted": 0,
            "abstained": 0,
            "error": type(exc).__name__,
            "reason": "semantic fallback failed; deterministic results retained",
        }

    return batch_results, semantic_status
