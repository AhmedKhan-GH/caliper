"""Contiguous document context segmentation for feature indexing."""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ContextBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    content_mask: torch.Tensor


def source_segments(document_ids, positions, capacity: int, *, final: bool = False):
    """Yield contiguous ``(start, end)`` spans of at most ``capacity`` tokens within each document."""
    if type(capacity) is not int or not 1 <= capacity < 2**63 or type(final) is not bool:
        raise ValueError("invalid source context capacity or final flag")
    if len(document_ids) != len(positions):
        raise ValueError("source document and position counts differ")
    start = 0
    for index in range(len(positions)):
        if any(type(v) is not int or not 0 <= v < 2**63 for v in (document_ids[index], positions[index])):
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
