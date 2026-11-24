"""Validate hosted completion evidence without inference from displayed bars."""
from __future__ import annotations

import hashlib
import re

from _mcp import Failure


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inference_state(status, prompt):
    state = status.get("inference")
    if not isinstance(state, dict):
        raise Failure("Hosted inference evidence is absent.")
    for key in ("prompt_hash", "interventions_hash"):
        if not isinstance(state.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", state[key]):
            raise Failure("Hosted inference state has an invalid hash.")
    if state["prompt_hash"] != text_hash(prompt):
        raise Failure("Hosted inference evidence belongs to another prompt.")
    return state


def bound_completion(record, state):
    return (isinstance(record, dict)
            and all(record.get(key) == state[key] for key in ("prompt_hash", "interventions_hash"))
            and type(record.get("completed_step")) is int and record["completed_step"] >= 0)


def comparison_evidence(status, prompt, bars):
    state = inference_state(status, prompt)
    record = state.get("comparison")
    if not bound_completion(record, state):
        raise Failure("Output comparison evidence is stale or incomplete.")
    ids = record.get("argmax_token_ids")
    if (not isinstance(ids, list) or len(ids) != 2
            or any(type(token) is not int or token < 0 for token in ids)):
        raise Failure("Output comparison lacks two actual argmax token IDs.")
    if not set(ids) <= {row["token"] for row in bars}:
        raise Failure("The displayed bars omit an actual argmax token.")
    return record


def generation_baseline(status, prompt):
    state = inference_state(status, prompt)
    generation = state.get("generation")
    if (not isinstance(generation, dict) or type(generation.get("id")) is not int or generation["id"] < 0):
        raise Failure("The baseline generation identity is absent or invalid.")
    return state


def generation_evidence(prior, status, params, prompt):
    """Return no result while the prior identity remains, then validate completion."""
    state = inference_state(status, prompt)
    if any(state[key] != prior[key] for key in ("prompt_hash", "interventions_hash")):
        raise Failure("The prompt or interventions changed during generation.")
    record = state.get("generation")
    if not isinstance(record, dict) or type(record.get("id")) is not int:
        raise Failure("Generation completion has no valid identity.")
    if record["id"] == prior["generation"]["id"]:
        return None
    if record["id"] != prior["generation"]["id"] + 1 or not bound_completion(record, state):
        raise Failure("Generation completion is stale or belongs to another action.")
    previous_step = prior["generation"].get("completed_step")
    if previous_step is not None and record["completed_step"] <= previous_step:
        raise Failure("Generation completion did not advance its step.")
    rows = record.get("token_ids")
    if (record.get("counts") != [24, 24] or not isinstance(rows, list) or len(rows) != 2
            or any(not isinstance(row, list) or len(row) != 24
                   or any(type(token) is not int or token < 0 for token in row) for row in rows)):
        raise Failure("Generation must supply exactly two rows of 24 integer token IDs.")
    texts = [params.get("output.baseline"), params.get("output.steered")]
    if (params.get("prompt") != prompt or any(not isinstance(text, str) or not text for text in texts)
            or record.get("text_hashes") != [text_hash(text) for text in texts]):
        raise Failure("Generation completion does not match both current output text rows.")
    return {"completion": record, "texts": texts, "prior_id": prior["generation"]["id"]}
