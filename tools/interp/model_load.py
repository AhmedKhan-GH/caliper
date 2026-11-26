"""Pinned local text-model load and logits-free decoder forwards."""
from __future__ import annotations

import json
import os
from pathlib import Path

from caliper.interp._dictionary import _canonical_device
from caliper.interp._memory import BudgetMemory, MemoryGuard
from caliper.interp._admission import GEMMA_ID, GEMMA_REVISION, _load_gemma_model

MODEL_ID, MODEL_REVISION = GEMMA_ID, GEMMA_REVISION


def load_text_model(*, device="mps", memory=None, model_id=MODEL_ID,
                    reference_profile=None):
    """Load the pinned Gemma snapshot with a 40 GB ceiling.

    Missing, unexpected, and mismatched keys are errors.
    Supplied lower limits remain active.
    The explicit reference profile selects bfloat16 inference.
    """
    if model_id != GEMMA_ID:
        raise ValueError("unsupported text model selection")
    if reference_profile is not None:
        from caliper.interp._gemma_reference import validate_profile
        validate_profile(reference_profile)
    memory = MemoryGuard(40, device) if memory is None else BudgetMemory(memory, 40)
    if _canonical_device(memory.device) != _canonical_device(device):
        raise ValueError("text model requires a memory guard for its device")
    memory.check("text model admission")
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot = Path(snapshot_download(GEMMA_ID, revision=GEMMA_REVISION, local_files_only=True))
    selection = {}
    if reference_profile is not None:
        import torch
        selection = {"dtype": torch.bfloat16, "reference_profile": reference_profile}
    model = _load_gemma_model(snapshot, device=device, memory=memory, **selection)
    if reference_profile is not None:
        from caliper.interp._reference_snapshots import remember_model_snapshot
        remember_model_snapshot(model, snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    print(json.dumps({"stage": "text_model_ready", "model_id": GEMMA_ID, "revision": GEMMA_REVISION,
                      "fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")}), flush=True)
    return model, tokenizer


def hidden_forward(model, input_ids, attention_mask=None):
    """Return final normalized hidden states without a vocabulary projection."""
    from caliper.interp._index import _reference_runtime
    _reference_runtime(model)
    if getattr(model.config, "model_type", None) == "caliper_tiny":
        x = model.embed_tokens(input_ids)
        for layer in model.layers:
            x = layer(x)
        return model.norm(x)
    options = {"attention_mask": attention_mask} if attention_mask is not None else {}
    result = model.model(input_ids=input_ids, use_cache=False, return_dict=True, **options).last_hidden_state
    _reference_runtime(model)
    return result
