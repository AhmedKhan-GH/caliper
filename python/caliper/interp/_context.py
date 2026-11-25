"""Gemma context prefixes with separate source and model positions."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ._memory import autorelease_scope


def context_policy(model, context_length, *, tokenizer=None):
    """Return the Gemma policy, or None for the unchanged Qwen and fixture paths."""
    if getattr(model.config, "model_type", None) != "gemma2":
        return None
    if type(context_length) is not int or not 2 <= context_length <= 256:
        raise ValueError("Gemma context must reserve a prefix within two to 256 positions")
    bos, pad = model.config.bos_token_id, model.config.pad_token_id
    vocab = model.config.vocab_size
    if (type(vocab) is not int or vocab < 1
            or any(type(value) is not int or not 0 <= value < vocab for value in (bos, pad))):
        raise ValueError("Gemma context requires valid BOS and padding IDs")
    for name, value in (("bos_token_id", bos), ("pad_token_id", pad)):
        if tokenizer is not None and (type(getattr(tokenizer, name, None)) is not int
                                      or getattr(tokenizer, name) != value):
            raise ValueError("Gemma tokenizer special IDs differ from the model")
    return {"version": 1, "family": "gemma2", "prefix_ids": [bos], "pad_id": pad,
            "model_context": context_length, "content_context": context_length - 1,
            "segmentation": "contiguous_document_positions",
            "observations": "source_tokens_only", "replacement": "source_positions_only",
            "cross_entropy": "within_context_content_targets_including_bos_to_first"}


@dataclass(frozen=True)
class ContextBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    content_mask: torch.Tensor

    @property
    def prediction_mask(self):
        """Select predictors whose next position is observed content in the same row."""
        return self.attention_mask[:, :-1].bool() & self.content_mask[:, 1:]


def prepare_contexts(rows, policy, *, device, pad_to=None):
    """Prefix nonempty source rows without a change to their IDs or order."""
    with autorelease_scope(device):
        return _prepare_contexts(rows, policy, device=device, pad_to=pad_to)


def _prepare_contexts(rows, policy, *, device, pad_to):
    lengths = [len(row) for row in rows]
    if not lengths or min(lengths) < 1 or max(lengths) > policy["content_context"]:
        raise ValueError("invalid content context lengths")
    width = max(lengths) + 1 if pad_to is None else pad_to
    if type(width) is not int or not max(lengths) + 1 <= width <= policy["model_context"]:
        raise ValueError("invalid padded model context")
    ids = torch.full((len(rows), width), policy["pad_id"], dtype=torch.int64, device=device)
    attention = torch.zeros_like(ids)
    content = torch.zeros_like(ids, dtype=torch.bool)
    ids[:, 0] = policy["prefix_ids"][0]
    attention[:, 0] = 1
    for index, row in enumerate(rows):
        end = len(row) + 1
        ids[index, 1:end] = torch.as_tensor(row, dtype=torch.int64, device=device)
        attention[index, 1:end] = 1
        content[index, 1:end] = True
    return ContextBatch(ids, attention, content)


def source_segments(document_ids, positions, capacity, *, final=False):
    """Return complete source spans. Retain an incomplete final span for more input."""
    if type(capacity) is not int or not 1 <= capacity < 2**63 or type(final) is not bool:
        raise ValueError("invalid source context capacity or final flag")
    if len(document_ids) != len(positions):
        raise ValueError("source document and position counts differ")
    start = 0
    for index in range(len(positions)):
        if any(type(value) is not int or not 0 <= value < 2**63
               for value in (document_ids[index], positions[index])):
            raise ValueError("invalid source document position")
        if index > 0:
            if document_ids[index] != document_ids[index - 1] and positions[index] != 0:
                raise ValueError("new source document must start at position zero")
            boundary = document_ids[index] != document_ids[index - 1] or positions[index] == 0
            if not boundary and positions[index] != positions[index - 1] + 1:
                raise ValueError("source document positions are not contiguous")
            if boundary and index > start:
                yield start, index
                start = index
        if index + 1 - start == capacity:
            yield start, index + 1
            start = index + 1
    if final and start < len(positions):
        yield start, len(positions)
