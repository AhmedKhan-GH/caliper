"""Bounded prompt tokenization across Backend, HookedTransformer, and Hugging Face models."""
from __future__ import annotations

import torch

from .._interpret_protocol import MAX_STRING, text as _valid_text
from ._graph import TOKEN_CAP, bounded_integer


def prompt_inputs(model_or_backend, tokenizer, prompt: str) -> dict[str, torch.Tensor]:
    """Encode one bounded UTF-8 prompt into ``{"input_ids": (1, T)}`` with ``1 <= T <= 32``."""
    if not _valid_text(prompt, MAX_STRING, empty=False):
        raise ValueError(f"prompt must contain 1 to {MAX_STRING} valid UTF-8 bytes without NUL")
    if hasattr(model_or_backend, "tokenize"):
        return {"input_ids": model_or_backend.tokenize(prompt)}
    model = model_or_backend
    if hasattr(model, "to_tokens") and tokenizer is getattr(model, "tokenizer", None):
        prepend = bool(getattr(getattr(model, "cfg", None), "default_prepend_bos", True))
        ids = model.to_tokens(prompt, prepend_bos=prepend)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        vocab = int(model.cfg.d_vocab)
        device = next(model.parameters()).device
        bounded_integer(int(ids.shape[1]), "prompt tokens", 1, TOKEN_CAP)
        if int(ids.min().item()) < 0 or int(ids.max().item()) >= vocab:
            raise ValueError("prompt contains a token outside the vocabulary")
        return {"input_ids": ids.to(device, dtype=torch.long)}

    encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
    if not isinstance(ids, torch.Tensor):
        ids = torch.as_tensor(ids, dtype=torch.long)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("prompt must contain 1 to 32 integer tokens in one row")
    bounded_integer(int(ids.shape[1]), "prompt tokens", 1, TOKEN_CAP)
    cfg = getattr(model, "cfg", None) or getattr(model, "config", None)
    vocab = getattr(cfg, "d_vocab", None) or getattr(cfg, "vocab_size", None)
    if vocab is not None and (int(ids.min().item()) < 0 or int(ids.max().item()) >= int(vocab)):
        raise ValueError("prompt contains a token outside the vocabulary")
    param = next(model.parameters(), None)
    dev = param.device if param is not None else torch.device("cpu")
    out = {"input_ids": ids.to(dev, dtype=torch.long)}
    if isinstance(encoded, dict) and "attention_mask" in encoded:
        mask = encoded["attention_mask"]
        if mask.shape != ids.shape or not torch.all(mask == 1):
            raise ValueError("one unpadded prompt is required")
        out["attention_mask"] = mask.to(dev)
    return out
