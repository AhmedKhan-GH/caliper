"""Paired next-token probabilities and bounded greedy continuations over Backend."""
from __future__ import annotations

from inspect import Parameter, signature

import torch

from .._intervention_protocol import internal_forward, is_internal_forward
from ._labels import display_text, token_label
from ._memory import autorelease_scope
from ._prompt import prompt_inputs


def bounded_text(value: str, limit: int = 1024) -> str:
    """Keep complete UTF-8 characters and visible control escapes."""
    return display_text(value).encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def packed_labels(labels, device) -> torch.Tensor:
    """Pack 63 UTF-8 bytes and a zero terminator as sixteen little-endian words."""
    rows = []
    for label in labels:
        raw = bounded_text(label, 63).encode("utf-8").ljust(64, b"\0")
        rows.append([int.from_bytes(raw[i:i + 4], "little") for i in range(0, 64, 4)])
    return torch.tensor(rows, dtype=torch.uint32, device=device)


def _decode_raw_token(tokenizer, token_id: int) -> str:
    decode = tokenizer.decode
    parameters = signature(decode).parameters
    accepts_kwargs = any(p.kind == Parameter.VAR_KEYWORD for p in parameters.values())
    options = {
        name: False
        for name in ("skip_special_tokens", "clean_up_tokenization_spaces")
        if accepts_kwargs or (name in parameters and parameters[name].kind in (
            Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY
        ))
    }
    return decode([int(token_id)], **options)


class Output:
    """Publish baseline row zero and intervention row one for ``logit_bars@1``."""

    def __init__(self, backend, interventions, *, guard=None):
        self.backend = backend
        self.model = backend.model
        self.tokenizer = backend.tokenizer
        self.interventions = interventions
        self.guard = guard
        self.prompt: str | None = None

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        ids = prompt_inputs(self.backend, self.tokenizer, prompt)["input_ids"]
        if prompt != self.prompt:
            if hasattr(self.interventions, "invalidate_source_cache"):
                self.interventions.invalidate_source_cache()
            elif hasattr(getattr(self.interventions, "executor", None), "invalidate_cache"):
                self.interventions.executor.invalidate_cache()
            self.prompt = prompt
            self.backend.record_prompt(prompt)
        return ids

    @staticmethod
    def _pure_feature_specs(specs: list[dict]) -> bool:
        return bool(specs) and all(s.get("op") in ("clamp", "scale") for s in specs)

    def compare(self, prompt: str, *, transform=None) -> dict:
        from ._publish import _publication_scope, _publish_many

        with _publication_scope(self.model), autorelease_scope(self.backend.device):
            ids = self._encode_prompt(prompt)
            specs = self.interventions.specifications()
            if not specs:
                with self.interventions.suspend():
                    base_last = self.backend.forward_logits(ids, guard=self.guard)[0, -1]
                logits = torch.stack((base_last, base_last), dim=0)
            elif self.backend.is_replacement_model and self._pure_feature_specs(specs):
                with self.interventions.suspend():
                    base_last = self.backend.forward_logits(ids, guard=self.guard)[0, -1]
                    steered_last = self.backend.intervene_logits(
                        ids[0], specs, freeze_attention=True, guard=self.guard
                    )[0][0, -1]
                logits = torch.stack((base_last, steered_last), dim=0)
            else:
                paired = ids.repeat(2, 1)
                with self.interventions.rows([1]):
                    logits = self.backend.forward_logits(paired, guard=self.guard)[:, -1]
            tensors, result = self.prepare_logits(logits, transform=transform)
            _publish_many(tensors)
            return result

    def prepare_logits(self, logits: torch.Tensor, *, transform=None) -> tuple[dict, dict]:
        probabilities = logits.float().softmax(-1)
        if probabilities.shape[-1] < 10:
            raise ValueError("output comparison requires at least ten vocabulary tokens")
        scores = probabilities.max(dim=0).values
        ids = torch.argsort(scores, descending=True, stable=True)[:10].clone()
        argmax_ids = probabilities.argmax(dim=-1)
        baseline_top = int(argmax_ids[0].item())
        baseline_top_text = _decode_raw_token(self.tokenizer, baseline_top)
        baseline_top_probability = float(probabilities[0, baseline_top].item())
        winners = set(argmax_ids.tolist())
        for winner in sorted(winners):
            if not torch.any(ids == winner):
                replace = next(i for i in range(9, -1, -1) if int(ids[i].item()) not in winners)
                ids[replace] = winner
        # Preserve bfloat16/float32 probability values from the model's native softmax for baseline reporting.
        native_probs = logits.softmax(-1)
        baseline_top_probability = float(native_probs[0, baseline_top].item())
        values = native_probs[:, ids].to(torch.float32)
        labels = [bounded_text(token_label(self.tokenizer, i), 63) for i in ids.tolist()]
        tensors = {
            "output.probs": values.to(self.backend.device),
            "output.tokens": ids.to(device=self.backend.device, dtype=torch.int32),
            "output.label_text": packed_labels(labels, self.backend.device),
        }
        if transform is not None:
            tensors = transform(tensors)
        result = {
            "tokens": ids.cpu().tolist(),
            "probabilities": values.cpu().tolist(),
            "labels": labels,
            "argmax_token_ids": argmax_ids.cpu().tolist(),
            "baseline_top_text": baseline_top_text,
            "baseline_top_probability": baseline_top_probability,
        }
        return tensors, result

    def generate(self, prompt: str) -> dict:
        """Produce 24 greedy tokens for baseline and steered rows."""
        ids = self._encode_prompt(prompt)
        start = int(ids.shape[1])
        specs = self.interventions.specifications()
        for spec in specs:
            if spec["op"] != "patch" or "source_position" in spec:
                continue
            src_ids = prompt_inputs(self.backend, self.tokenizer, spec["source_prompt"])["input_ids"]
            last_source = max(spec.get("positions", [start + 22]), default=-1)
            if last_source >= int(src_ids.shape[1]):
                raise ValueError("generation requires patch positions within the clean source or a fixed source_position")

        if not specs or (self.backend.is_replacement_model and self._pure_feature_specs(specs)):
            with self.interventions.suspend():
                gen = self.backend.generate(prompt, specs, max_new_tokens=24, guard=self.guard)
            row0, row1 = gen["token_ids"]
            return {
                "baseline": bounded_text(self.tokenizer.decode(row0)),
                "steered": bounded_text(self.tokenizer.decode(row1)),
                "token_ids": [row0, row1],
            }

        paired = ids.repeat(2, 1)
        with self.interventions.rows([1]):
            for _ in range(24):
                with autorelease_scope(self.backend.device):
                    nxt = self.backend.forward_logits(paired, guard=self.guard)[:, -1].argmax(-1, keepdim=True)
                    paired = torch.cat((paired, nxt), dim=1)
        gen_ids = paired[:, start:].cpu().tolist()
        return {
            "baseline": bounded_text(self.tokenizer.decode(gen_ids[0])),
            "steered": bounded_text(self.tokenizer.decode(gen_ids[1])),
            "token_ids": gen_ids,
        }
