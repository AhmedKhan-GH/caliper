"""Pinned local text-model load and logits-free decoder forwards."""
from __future__ import annotations

import json
import os
from pathlib import Path

from caliper.interp import _qwen_family
from caliper.interp._qwen_family import MODEL_ID, MODEL_REVISION
from caliper.interp._dictionary import _canonical_device, _record_verified_hf_load
from caliper.interp._memory import BudgetMemory, MemoryGuard
from caliper.interp._admission import (GEMMA_ID, GEMMA_REVISION, _admit_model_load,
                                       _admit_model_transfer, _load_gemma_model)


def load_text_model(*, device="mps", memory=None, model_id=MODEL_ID, mode="inference",
                    reference_profile=None):
    """Load only the text class from the complete cached snapshot.

    Transformers resolves the shard index. It applies the same prefix mapping
    as the lead preflight. Missing, unexpected, and mismatched keys are errors.
    Qwen inference has a 12 GB ceiling. Explicit Qwen training uses 24 GB.
    Gemma inference uses 40 GB. Supplied lower limits remain active.
    The explicit Gemma reference profile selects bfloat16 inference.
    """
    if model_id not in (MODEL_ID, GEMMA_ID):
        raise ValueError("unsupported text model selection")
    if mode not in ("inference", "training"):
        raise ValueError("unsupported text model load mode")
    if mode == "training" and model_id != MODEL_ID:
        raise ValueError("training mode requires the Qwen text model")
    if reference_profile is not None:
        if model_id != GEMMA_ID or mode != "inference":
            raise ValueError("reference profile requires Gemma inference")
        from caliper.interp._gemma_reference import validate_profile
        validate_profile(reference_profile)
    ceiling = (_qwen_family.TRAINING_MEMORY_MAX_GB if mode == "training" else
               40 if model_id == GEMMA_ID else _qwen_family.INFERENCE_MEMORY_MAX_GB)
    memory = MemoryGuard(ceiling, device) if memory is None else BudgetMemory(memory, ceiling)
    if _canonical_device(memory.device) != _canonical_device(device):
        raise ValueError("text model requires a memory guard for its device")
    memory.check("text model admission")
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    if model_id == GEMMA_ID:
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

    return _qwen_family.load_text_model(
        device=device, memory=memory, snapshot_download=snapshot_download, tokenizer_type=AutoTokenizer,
        admit_load=_admit_model_load, admit_transfer=_admit_model_transfer, record_load=_record_verified_hf_load)


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
