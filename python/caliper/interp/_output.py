"""Paired next-token probabilities and bounded greedy continuations."""
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from ._memory import autorelease_scope

_INTERNAL = ContextVar("caliper_inference_forward", default=False)


@contextmanager
def internal_forward():
    """Keep internal forwards within one managed run boundary."""
    token = _INTERNAL.set(True)
    try:
        yield
    finally:
        _INTERNAL.reset(token)


def is_internal_forward():
    return _INTERNAL.get()


def prompt_inputs(model, tokenizer, prompt):
    """Encode one bounded prompt without truncation or a device change."""
    if not isinstance(prompt, str) or not 1 <= len(prompt.encode("utf-8")) <= 4096:
        raise ValueError("prompt must contain 1 to 4096 UTF-8 bytes")
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
    ids = encoded["input_ids"]
    if (ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= 32
            or ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("prompt must contain 1 to 32 integer tokens in one row")
    if ids.min() < 0 or ids.max() >= model.config.vocab_size:
        raise ValueError("prompt contains a token outside the vocabulary")
    device = next(model.parameters()).device
    inputs = {"input_ids": ids.to(device)}
    if "attention_mask" in encoded:
        mask = encoded["attention_mask"]
        if mask.shape != ids.shape or not torch.all(mask == 1):
            raise ValueError("one unpadded prompt is required")
        inputs["attention_mask"] = mask.to(device)
    return inputs


def forward_logits(model, inputs):
    """Run real model math without a cache, a gradient, or another run step."""
    if model.training:
        raise ValueError("inference requires an evaluation model")
    with internal_forward(), torch.no_grad(), autorelease_scope(next(model.parameters()).device):
        output = model(**inputs, use_cache=False)
        logits = output.logits
        if (logits.ndim != 3 or logits.shape[:2] != inputs["input_ids"].shape
                or logits.shape[-1] != model.config.vocab_size or not torch.isfinite(logits).all()):
            raise ValueError("model logits must be finite batch-token-vocabulary values")
        return logits.detach()


def bounded_text(text, limit=1024):
    """Keep complete UTF-8 characters and remove embedded NUL characters."""
    return str(text).replace("\0", "").encode("utf-8", errors="replace")[:limit].decode("utf-8", errors="ignore")


def packed_labels(labels, device):
    """Pack at most 64 UTF-8 bytes per label as sixteen little-endian words."""
    rows = []
    for label in labels:
        raw = bounded_text(label, 64).encode("utf-8").ljust(64, b"\0")
        rows.append([int.from_bytes(raw[i:i + 4], "little") for i in range(0, 64, 4)])
    return torch.tensor(rows, dtype=torch.uint32, device=device)


class Output:
    """Publish baseline row zero and intervention row one from one batch."""

    def __init__(self, model, tokenizer, interventions):
        self.model, self.tokenizer, self.interventions = model, tokenizer, interventions
        self.prompt = None

    def _inputs(self, prompt):
        inputs = prompt_inputs(self.model, self.tokenizer, prompt)
        if prompt != self.prompt:
            self.interventions.invalidate_source_cache()
            self.prompt = prompt
        return {name: value.repeat(2, 1) for name, value in inputs.items()}

    def compare(self, prompt):
        from . import publish
        inputs = self._inputs(prompt)
        with self.interventions.rows([1]):
            logits = forward_logits(self.model, inputs)[:, -1]
        probabilities = logits.softmax(-1)
        if probabilities.shape[-1] < 10:
            raise ValueError("output comparison requires at least ten vocabulary tokens")
        scores = probabilities.max(dim=0).values
        ids = torch.argsort(scores, descending=True, stable=True)[:10].clone()
        baseline_top = probabilities[0].argmax()
        if not torch.any(ids == baseline_top):
            ids[-1] = baseline_top
        values = probabilities[:, ids].to(torch.float32)
        labels = [bounded_text(self.tokenizer.convert_ids_to_tokens(i), 64) for i in ids.tolist()]
        publish("output.probs", values)
        publish("output.tokens", ids.to(torch.int32))
        publish("output.label_text", packed_labels(labels, ids.device))
        return {"tokens": ids.tolist(), "probabilities": values.tolist(), "labels": labels}

    def generate(self, prompt):
        """Produce exactly 24 greedy tokens for both rows, including EOS tokens."""
        inputs = self._inputs(prompt)
        start = inputs["input_ids"].shape[1]
        with self.interventions.rows([1]):
            for _ in range(24):
                token = forward_logits(self.model, inputs)[:, -1].argmax(-1, keepdim=True)
                inputs["input_ids"] = torch.cat((inputs["input_ids"], token), dim=1)
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(token)), dim=1)
        ids = inputs["input_ids"][:, start:].tolist()
        return {"baseline": bounded_text(self.tokenizer.decode(ids[0])),
                "steered": bounded_text(self.tokenizer.decode(ids[1])), "token_ids": ids}
