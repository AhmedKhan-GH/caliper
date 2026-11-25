"""Paired next-token probabilities and bounded greedy continuations."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from inspect import Parameter, signature

import torch

from .._interpret_protocol import MAX_STRING, text
from ._labels import display_text, token_label
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
    if not text(prompt, MAX_STRING, empty=False):
        raise ValueError(f"prompt must contain 1 to {MAX_STRING} valid UTF-8 bytes without NUL")
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


def forward_logits(model, inputs, *, last_token=True, guard=None):
    """Run real model math without a cache, a gradient, or another run step."""
    if model.training:
        raise ValueError("inference requires an evaluation model")
    with internal_forward(), torch.no_grad(), autorelease_scope(next(model.parameters()).device):
        if guard is not None:
            guard.check("inference forward start")
        kwargs = {"use_cache": False}
        keep_last = last_token and model.config.model_type in ("qwen3_5_text", "gemma2")
        if keep_last:
            kwargs["logits_to_keep"] = 1
        output = model(**inputs, **kwargs)
        logits = output.logits
        shape = (inputs["input_ids"].shape[0], 1) if keep_last else inputs["input_ids"].shape
        if (logits.ndim != 3 or logits.shape[:2] != shape
                or logits.shape[-1] != model.config.vocab_size or not torch.isfinite(logits).all()):
            raise ValueError("model logits must be finite batch-token-vocabulary values")
        if guard is not None:
            guard.check("inference forward complete")
        return logits.detach()


def forward_reference_logit(model, inputs, *, position, token_id, guard=None):
    """Measure the traced centered direction before the qualified model's softcap."""
    from ._gemma_reference import profile_metadata
    if profile_metadata(model) is None:
        raise ValueError("reference validation requires a Gemma reference profile")
    last_token = position == inputs["input_ids"].shape[1] - 1
    values = []

    def measure(module, args):
        # Center weights before contraction, with the same dtype as traced edges.
        weight = module.weight.detach()
        direction = weight[token_id] - weight.mean(0)
        row = args[0][0, -1 if last_token else position]
        values.append(float(row @ direction))

    handle = model.lm_head.register_forward_pre_hook(measure)
    try:
        forward_logits(model, inputs, last_token=last_token, guard=guard)
        if len(values) != 1 or not torch.isfinite(torch.tensor(values)).all():
            raise ValueError("reference validation requires one finite output objective")
        return values[0]
    finally:
        handle.remove()


def bounded_text(text, limit=1024):
    """Keep complete UTF-8 characters and the shared NUL display escape."""
    return display_text(text).encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def packed_labels(labels, device):
    """Pack 63 UTF-8 bytes and a zero terminator as sixteen little-endian words."""
    rows = []
    for label in labels:
        raw = bounded_text(label, 63).encode("utf-8").ljust(64, b"\0")
        rows.append([int.from_bytes(raw[i:i + 4], "little") for i in range(0, 64, 4)])
    return torch.tensor(rows, dtype=torch.uint32, device=device)


class Output:
    """Publish baseline row zero and intervention row one from one batch."""

    def __init__(self, model, tokenizer, interventions, *, guard=None):
        self.model, self.tokenizer, self.interventions = model, tokenizer, interventions
        self.guard = guard
        self.prompt = None
        self.attention_scope = None

    def _inputs(self, prompt):
        inputs = prompt_inputs(self.model, self.tokenizer, prompt)
        if prompt != self.prompt:
            self.interventions.invalidate_source_cache()
            self.prompt = prompt
        return {name: value.repeat(2, 1) for name, value in inputs.items()}

    def compare(self, prompt):
        from ._publish import _publication_scope
        with _publication_scope(self.model), autorelease_scope(next(self.model.parameters()).device):
            inputs = self._inputs(prompt)
            scope = (self.attention_scope(prompt, self.interventions.specifications())
                     if self.attention_scope is not None else nullcontext())
            with scope, self.interventions.rows([1]):
                logits = forward_logits(self.model, inputs, guard=self.guard)[:, -1]
            return self.publish_logits(logits)

    def publish_logits(self, logits):
        """Publish bounded rows from actual baseline and candidate logits."""
        publication, result = self.prepare_logits(logits)
        publication.commit()
        return result

    def prepare_logits(self, logits, *, transform=None):
        """Prepare all output tensors without a change to public bindings."""
        from ._publish import _publication_scope
        with _publication_scope(self.model), autorelease_scope(next(self.model.parameters()).device):
            from ._publish import _prepare_many
            probabilities = logits.softmax(-1)
            if probabilities.shape[-1] < 10:
                raise ValueError("output comparison requires at least ten vocabulary tokens")
            scores = probabilities.max(dim=0).values
            ids = torch.argsort(scores, descending=True, stable=True)[:10].clone()
            argmax_ids = probabilities.argmax(dim=-1)
            baseline_top = int(argmax_ids[0])
            decode = self.tokenizer.decode
            parameters = signature(decode).parameters
            accepts_kwargs = any(p.kind == Parameter.VAR_KEYWORD for p in parameters.values())
            options = {name: False for name in ("skip_special_tokens", "clean_up_tokenization_spaces")
                       if accepts_kwargs or (name in parameters and parameters[name].kind in
                                             (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY))}
            baseline_top_text = decode([baseline_top], **options)
            baseline_top_probability = float(probabilities[0, baseline_top])
            # Reserve both row winners before filling the fixed comparison slots.
            winners = set(argmax_ids.tolist())
            for winner in sorted(winners):
                if not torch.any(ids == winner):
                    replace = next(i for i in range(9, -1, -1) if int(ids[i]) not in winners)
                    ids[replace] = winner
            values = probabilities[:, ids].to(torch.float32)
            labels = [bounded_text(token_label(self.tokenizer, i), 63) for i in ids.tolist()]
            tensors = {"output.probs": values, "output.tokens": ids.to(torch.int32),
                       "output.label_text": packed_labels(labels, ids.device)}
            result = {"tokens": ids.tolist(), "probabilities": values.tolist(), "labels": labels,
                    "argmax_token_ids": argmax_ids.tolist(), "baseline_top_text": baseline_top_text,
                    "baseline_top_probability": baseline_top_probability}
            if transform is not None:
                tensors = transform(tensors)
            return _prepare_many(tensors), result

    def generate(self, prompt):
        """Produce exactly 24 greedy tokens for both rows, including EOS tokens."""
        inputs = self._inputs(prompt)
        start = inputs["input_ids"].shape[1]
        for spec in self.interventions.specifications():
            if spec["op"] != "patch" or "source_position" in spec:
                continue
            source = prompt_inputs(self.model, self.tokenizer, spec["source_prompt"])
            # The last forward sees the prompt and 23 generated tokens.
            last_source = max(spec.get("positions", [start + 22]), default=-1)
            if last_source >= source["input_ids"].shape[1]:
                raise ValueError("generation requires patch positions within the clean source or a fixed source_position")
        with self.interventions.rows([1]):
            for _ in range(24):
                with autorelease_scope(next(self.model.parameters()).device):
                    token = forward_logits(self.model, inputs, guard=self.guard)[:, -1].argmax(-1, keepdim=True)
                    inputs["input_ids"] = torch.cat((inputs["input_ids"], token), dim=1)
                    if "attention_mask" in inputs:
                        inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(token)), dim=1)
        with autorelease_scope(next(self.model.parameters()).device):
            ids = inputs["input_ids"][:, start:].tolist()
            return {"baseline": bounded_text(self.tokenizer.decode(ids[0])),
                    "steered": bounded_text(self.tokenizer.decode(ids[1])), "token_ids": ids}
