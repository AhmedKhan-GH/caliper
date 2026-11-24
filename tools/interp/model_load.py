"""Pinned local text-model load and logits-free decoder forwards."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from caliper.interp._dictionary import _canonical_device, _record_verified_hf_load
from caliper.interp._memory import BudgetMemory, MemoryGuard
from caliper.interp._admission import (GEMMA_ID, GEMMA_REVISION, _admit_model_load,
                                       _admit_model_transfer, _load_gemma_model)


MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def load_text_model(*, device="mps", memory=None, model_id=MODEL_ID):
    """Load only the text class from the complete cached snapshot.

    Transformers resolves the shard index. It applies the same prefix mapping
    as the lead preflight. Missing, unexpected, and mismatched keys are errors.
    """
    if model_id not in (MODEL_ID, GEMMA_ID):
        raise ValueError("unsupported text model selection")
    ceiling = 28 if model_id == GEMMA_ID else 12
    memory = MemoryGuard(ceiling, device) if memory is None else BudgetMemory(memory, ceiling)
    if _canonical_device(memory.device) != _canonical_device(device):
        raise ValueError("text model requires a memory guard for its device")
    memory.check("text model admission")
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    if model_id == GEMMA_ID:
        snapshot = Path(snapshot_download(GEMMA_ID, revision=GEMMA_REVISION, local_files_only=True))
        model = _load_gemma_model(snapshot, device=device, memory=memory)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        print(json.dumps({"stage": "text_model_ready", "model_id": GEMMA_ID, "revision": GEMMA_REVISION,
                          "fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")}), flush=True)
        return model, tokenizer

    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    config = Qwen3_5TextConfig(**json.loads((snapshot / "config.json").read_text())["text_config"])
    if (config.num_hidden_layers, config.hidden_size) != (24, 1024):
        raise ValueError("the pinned text model has unexpected dimensions")
    estimate = _admit_model_load(snapshot, memory, device)
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        snapshot, config=config, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True, output_loading_info=True,
        key_mapping={r"^model\.language_model\.": "model."},
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"text checkpoint load is incomplete: {info}")
    # These fields come from the resolved snapshot, never from a dictionary.
    model.config._name_or_path = MODEL_ID
    model.config._commit_hash = MODEL_REVISION
    model.requires_grad_(False).eval()
    if memory:
        memory.check("after_model_cpu_load")
    # The complete CPU load establishes the digest in the final float32 dtype.
    _record_verified_hf_load(model, memory=memory)
    _admit_model_transfer(memory, estimate, device)
    model.to(device=device, dtype=torch.float32)
    if memory:
        memory.check("after_model_device_load")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    print(json.dumps({"stage": "text_model_ready", "model_id": MODEL_ID, "revision": MODEL_REVISION,
                      "fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")}), flush=True)
    return model, tokenizer


def hidden_forward(model, input_ids, attention_mask=None):
    """Return final normalized hidden states without a vocabulary projection."""
    if getattr(model.config, "model_type", None) == "caliper_tiny":
        x = model.embed_tokens(input_ids)
        for layer in model.layers:
            x = layer(x)
        return model.norm(x)
    options = {"attention_mask": attention_mask} if attention_mask is not None else {}
    return model.model(input_ids=input_ids, use_cache=False, return_dict=True, **options).last_hidden_state
