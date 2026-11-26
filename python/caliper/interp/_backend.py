"""Library adapter over TransformerLens, circuit-tracer, and SAELens."""
from __future__ import annotations

from collections.abc import Sequence
import gc
import heapq
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import torch

from .._ablate import suspend
from .._interpret_protocol import MAX_STRING, text as _valid_text
from .._intervention_protocol import InterventionError, internal_forward
from ._graph import (
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_NODE_THRESHOLD,
    Graph,
    LARGE_EDGE_CAP,
    LARGE_NODE_CAP,
    SHOW_MORE_EDGE_THRESHOLD,
    SHOW_MORE_NODE_THRESHOLD,
    resolve_pruning_thresholds,
)
from ._labels import token_label
from ._memory import autorelease_scope, mps_batch_pool_scope

os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")

WINDOW_TOKENS = 33
DEFAULT_CORPUS = (
    "Fact: the capital of the state containing Dallas is",
    "Fact: the capital of the state containing Oakland is",
    "Fact: the capital of the territory containing Vancouver is",
    "Fact: the capital of the country containing Shanghai is",
)

# Canonical lowest-L0 width_16k transcoders from the public Gemma Scope alias config.
_GEMMA_2B_CANONICAL_L0 = (
    76, 65, 49, 54, 88, 87, 95, 70, 52, 72, 88, 5, 6,
    8, 8, 8, 10, 12, 13, 12, 11, 13, 15, 25, 37, 41,
)


def default_device() -> torch.device:
    """Select CUDA, MPS, or CPU explicitly."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mps_cleanup(device=None) -> None:
    """Release Python garbage and flush the Metal allocator cache."""
    gc.collect()
    dev = torch.device(device) if device is not None else None
    if (dev is not None and dev.type == "mps") or (dev is None and torch.backends.mps.is_available()):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()


class _LayerWeightsShim:
    """Expose per-layer dimensions and lazy decoder slice lookup."""

    def __init__(self, owner: "DictionaryView", layer: int):
        self._owner = owner
        self._layer = layer

    @property
    def W_enc(self):
        return self._owner.encoder_matrix(self._layer)

    @property
    def W_dec(self):
        return self._owner.decoder_proxy(self._layer)


class _DecoderRowProxy:
    """Slice decoder rows via ``_get_decoder_vectors`` without loading full lazy matrices."""

    def __init__(self, owner: "DictionaryView", layer: int):
        self._owner = owner
        self._layer = layer

    @property
    def shape(self) -> tuple[int, int]:
        meta = self._owner.metadata
        return (int(meta["latents"]), int(meta["width"]))

    @property
    def device(self) -> torch.device:
        return self._owner.device

    def __getitem__(self, idx):
        return self._owner.decoder_vectors(self._layer, idx)


class DictionaryView:
    """Uniform metadata and encode/decode view over ``TranscoderSet`` and ``SAE``."""

    def __init__(self, raw, *, kind: str, n_model_layers: int, dictionary_id: str, device: torch.device):
        self.raw = raw
        self.kind = kind
        self.device = torch.device(device)
        if kind == "transcoder":
            covered = sorted(int(k) for k in raw.transcoders.keys()) if isinstance(raw.transcoders, dict) else list(range(len(raw.transcoders)))
            first = raw[covered[0]]
            latents = int(first.d_transcoder)
            width = int(first.d_model)
            read_points = [f"layers.{l}.mlp_in" for l in covered]
            write_points = [f"layers.{l}.mlp_out" for l in covered]
            neuronpedia_id = None
            scan_name = getattr(raw, "scan_name", None) or dictionary_id
        elif kind == "sae":
            sae_map = raw if isinstance(raw, dict) else {self._infer_sae_layer(raw): raw}
            self.raw = sae_map
            covered = sorted(sae_map.keys())
            first = sae_map[covered[0]]
            latents = int(first.cfg.d_sae)
            width = int(first.cfg.d_in)
            read_points = [str(sae_map[l].cfg.metadata.hook_name or f"blocks.{l}.hook_resid_pre") for l in covered]
            write_points = [str(sae_map[l].cfg.metadata.hook_name_out or sae_map[l].cfg.metadata.hook_name or f"blocks.{l}.hook_resid_pre") for l in covered]
            neuronpedia_id = getattr(first.cfg.metadata, "neuronpedia_id", None)
            scan_name = dictionary_id
        else:
            raise ValueError(f"unsupported dictionary kind {kind!r}")

        self.covered_layers = tuple(covered)
        self.layers = [_LayerWeightsShim(self, l) if l in covered else None for l in range(n_model_layers)]
        self.metadata = {
            "dictionary_id": str(dictionary_id),
            "scan_name": str(scan_name),
            "kind": kind,
            "layers": int(n_model_layers),
            "covered_layers": list(covered),
            "latents": latents,
            "width": width,
            "read_points": read_points,
            "write_points": write_points,
            "neuronpedia_id": neuronpedia_id,
        }

    @staticmethod
    def _infer_sae_layer(sae) -> int:
        hook = str(getattr(sae.cfg.metadata, "hook_name", "") or "")
        m = re.search(r"(?:blocks|layers)\.(\d+)\.", hook)
        return int(m.group(1)) if m else 0

    def check_feature(self, layer: int, feature: int) -> None:
        if type(layer) is not int or layer not in self.covered_layers:
            raise InterventionError(f"dictionary has no layer {layer}", "intervene.bad_request")
        latents = self.metadata["latents"]
        if type(feature) is not int or not 0 <= feature < latents:
            raise InterventionError(f"feature {feature} must be in 0..{latents - 1}", "intervene.bad_request")

    def encoder_matrix(self, layer: int) -> torch.Tensor:
        if self.kind == "transcoder":
            return self.raw[layer].W_enc
        return self.raw[layer].W_enc

    def decoder_proxy(self, layer: int) -> _DecoderRowProxy:
        return _DecoderRowProxy(self, layer)

    def decoder_vectors(self, layer: int, feat_ids) -> torch.Tensor:
        if self.kind == "transcoder":
            single = isinstance(feat_ids, int)
            idx = torch.tensor([feat_ids], dtype=torch.long) if single else feat_ids
            vecs = self.raw[layer]._get_decoder_vectors(idx)
            return vecs[0] if single else vecs
        sae = self.raw[layer]
        return sae.W_dec[feat_ids]

    def encode(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "transcoder":
            return self.raw.encode_layer(x, layer)
        return self.raw[layer].encode(x)

    def parameters(self):
        if self.kind == "transcoder":
            return self.raw.parameters()
        params = []
        for sae in self.raw.values():
            params.extend(sae.parameters())
        return iter(params)

    def state_dict(self):
        if self.kind == "transcoder":
            return self.raw.state_dict()
        out = {}
        for l, sae in self.raw.items():
            for k, v in sae.state_dict().items():
                out[f"layer_{l}.{k}"] = v
        return out


class _FeatureLayerGroup:
    """Ordered live feature hooks on one layer with last-edit-wins at shared coordinates."""

    def __init__(self, backend: "Backend", executor, layer: int, target_module):
        self.backend = backend
        self.executor = executor
        self.layer = layer
        self.target_module = target_module
        self.members: list[_FeatureHook] = []
        self._cached_input: torch.Tensor | None = None
        self.handles: list = []
        self._install()

    def _install(self) -> None:
        model = self.backend.model
        mods = dict(model.named_modules())
        if self.backend.dictionary.kind == "transcoder":
            in_hook = f"blocks.{self.layer}.{self.backend.dictionary.raw.feature_input_hook}"
            out_hook = f"blocks.{self.layer}.{self.backend.dictionary.raw.feature_output_hook}"
            in_mod = mods[in_hook]
            out_mod = mods[out_hook]
            self.handles.append(in_mod.register_forward_hook(self._capture_input))
            self.handles.append(out_mod.register_forward_hook(self._apply_transcoder))
        else:
            sae = self.backend.dictionary.raw[self.layer]
            in_hook = str(sae.cfg.metadata.hook_name or f"blocks.{self.layer}.hook_resid_pre")
            out_hook = str(sae.cfg.metadata.hook_name_out or in_hook)
            if in_hook == out_hook:
                self.handles.append(mods[in_hook].register_forward_hook(self._apply_sae_same_point))
            else:
                self.handles.append(mods[in_hook].register_forward_hook(self._capture_input))
                self.handles.append(mods[out_hook].register_forward_hook(self._apply_sae_split_point))

    def close_if_empty(self) -> None:
        if self.members:
            return
        while self.handles:
            self.handles.pop().remove()
        self._cached_input = None
        self.executor._feature_groups.pop(self.layer, None)

    def _active_members(self, is_muted) -> list["_FeatureHook"]:
        active = [m for m in self.members if not m.removed and not is_muted(m)]
        active.sort(key=lambda m: self.executor.execution_order(m))
        return active

    def _capture_input(self, module, args, output):
        self._cached_input = output
        return None

    def _select_rows(self, tensor: torch.Tensor, selected_rows_fn) -> list[int]:
        chosen = selected_rows_fn(self.executor.model)
        batch = list(range(tensor.shape[0])) if chosen is None else list(chosen)
        if any(r >= tensor.shape[0] for r in batch):
            raise InterventionError("selected batch row is outside the live tensor")
        return batch

    def _apply_edits(self, acts: torch.Tensor, batch: list[int], active: list["_FeatureHook"]) -> tuple[torch.Tensor, set[int]]:
        acts_mod = acts.clone()
        touched_features: set[int] = set()
        seq_len = acts.shape[1]
        for member in active:
            member.calls += 1
            spec = member.spec
            positions = spec.get("positions", list(range(seq_len)))
            if any(p >= seq_len for p in positions):
                member.fail(InterventionError("selected position is outside the live tensor"))
                continue
            if not positions:
                continue
            feat = int(spec["feature"])
            touched_features.add(feat)
            b_idx = torch.tensor(batch, device=acts.device, dtype=torch.long)[:, None]
            p_idx = torch.tensor(positions, device=acts.device, dtype=torch.long)[None, :]
            if spec["op"] == "clamp":
                acts_mod[b_idx, p_idx, feat] = float(spec["value"])
            elif spec["op"] == "scale":
                # Scale relative to current activation in the chain so ordered edits compose,
                # while clamp overwrites (last-edit-wins at shared coordinates).
                acts_mod[b_idx, p_idx, feat] = acts_mod[b_idx, p_idx, feat] * float(spec["multiplier"])
        return acts_mod, touched_features

    def _apply_transcoder(self, module, args, output):
        from ._memory import CleanStop
        x_in = self._cached_input
        self._cached_input = None
        if x_in is None:
            return None
        active = self._active_members(self.members[0].is_muted) if self.members else []
        if not active:
            return None
        try:
            batch = self._select_rows(output, active[0].selected_rows)
            if not batch:
                return None
            with torch.no_grad(), autorelease_scope(output.device):
                acts = self.backend.dictionary.encode(self.layer, x_in)
                acts_mod, touched = self._apply_edits(acts, batch, active)
                if not touched:
                    return None
                feat_list = sorted(touched)
                feat_t = torch.tensor(feat_list, device=acts.device, dtype=torch.long)
                deltas = (acts_mod.index_select(-1, feat_t) - acts.index_select(-1, feat_t)).to(output.dtype)
                if not torch.any(deltas != 0):
                    return output
                dec_vecs = self.backend.dictionary.decoder_vectors(self.layer, feat_t).to(
                    device=output.device, dtype=output.dtype
                )
                mlp_delta = deltas @ dec_vecs
                if not torch.isfinite(mlp_delta).all().item():
                    raise InterventionError("feature intervention produced a nonfinite tensor")
                return output + mlp_delta
        except CleanStop:
            raise
        except Exception as exc:
            for m in list(active):
                m.fail(exc)
            return None

    def _apply_sae_same_point(self, module, args, output):
        return self._run_sae(output, output)

    def _apply_sae_split_point(self, module, args, output):
        x_in = self._cached_input
        self._cached_input = None
        if x_in is None:
            return None
        return self._run_sae(x_in, output)

    def _run_sae(self, x_in: torch.Tensor, output: torch.Tensor):
        from ._memory import CleanStop
        active = self._active_members(self.members[0].is_muted) if self.members else []
        if not active:
            return None
        try:
            batch = self._select_rows(output, active[0].selected_rows)
            if not batch:
                return None
            sae = self.backend.dictionary.raw[self.layer]
            with torch.no_grad(), autorelease_scope(output.device):
                acts = sae.encode(x_in)
                acts_mod, touched = self._apply_edits(acts, batch, active)
                if not touched or torch.equal(acts_mod, acts):
                    return output
                base_dec = sae.decode(acts).to(output.dtype)
                mod_dec = sae.decode(acts_mod).to(output.dtype)
                res = output + (mod_dec - base_dec)
                if not torch.isfinite(res).all().item():
                    raise InterventionError("SAE intervention produced a nonfinite tensor")
                return res
        except CleanStop:
            raise
        except Exception as exc:
            for m in list(active):
                m.fail(exc)
            return None


class _FeatureHook:
    """Removable handle for one ``clamp`` or ``scale`` specification."""

    def __init__(self, group: _FeatureLayerGroup, spec: dict, on_error, is_muted, selected_rows):
        self.group = group
        self.spec = spec
        self.on_error = on_error
        self.is_muted = is_muted
        self.selected_rows = selected_rows
        self.calls = 0
        self.errors = 0
        self.error: str | None = None
        self.removed = False

    def __call__(self) -> None:
        if self.removed:
            return
        self.removed = True
        self.on_error = None
        if self in self.group.members:
            self.group.members.remove(self)
        self.group.close_if_empty()

    def fail(self, exc: BaseException) -> None:
        from ._memory import CleanStop
        self.errors += 1
        self.error = f"{type(exc).__name__}: {exc}"[:300]
        cb = self.on_error
        self()
        if cb is not None:
            try:
                cb(self.error)
            except CleanStop:
                raise
            except Exception:
                pass


class Backend:
    """Unified adapter over ``ReplacementModel`` / ``HookedTransformer`` and ``TranscoderSet`` / ``SAE``."""

    def __init__(self, model, tokenizer, dictionary: DictionaryView | None, *, device: torch.device,
                 dtype: torch.dtype, model_id: str = "model", guard=None):
        self.model = model
        self.tokenizer = tokenizer
        self.dictionary = dictionary
        self.device = torch.device(device)
        self.dtype = dtype
        self.model_id = model_id
        self.guard = guard
        self.recent_prompts: list[str] = []

    @property
    def is_replacement_model(self) -> bool:
        return hasattr(self.model, "feature_intervention") and hasattr(self.model, "transcoders")

    def record_prompt(self, prompt: str) -> None:
        if isinstance(prompt, str) and prompt and prompt not in self.recent_prompts:
            self.recent_prompts.append(prompt)
            if len(self.recent_prompts) > 16:
                self.recent_prompts.pop(0)

    def tokenize(self, prompt: str) -> torch.Tensor:
        """Encode one prompt into ``(1, T)`` token IDs on ``self.device`` with ``1 <= T <= 32``."""
        if not _valid_text(prompt, MAX_STRING, empty=False):
            raise ValueError(f"prompt must contain 1 to {MAX_STRING} valid UTF-8 bytes without NUL")
        prepend = bool(getattr(self.model.cfg, "default_prepend_bos", True))
        ids = self.model.to_tokens(prompt, prepend_bos=prepend)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= int(ids.shape[1]) <= 32:
            raise ValueError("prompt must contain 1 to 32 integer tokens in one row")
        vocab = int(self.model.cfg.d_vocab)
        if int(ids.min().item()) < 0 or int(ids.max().item()) >= vocab:
            raise ValueError("prompt contains a token outside the vocabulary")
        return ids.to(self.device, dtype=torch.long)

    def forward_logits(self, inputs: torch.Tensor | dict | str, *, guard=None) -> torch.Tensor:
        """Run a no-grad forward pass returning finite ``(B, T, V)`` logits."""
        active_guard = guard if guard is not None else self.guard
        if isinstance(inputs, str):
            ids = self.tokenize(inputs)
        elif isinstance(inputs, dict):
            ids = inputs["input_ids"].to(self.device)
        else:
            ids = inputs.to(self.device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        with internal_forward(), torch.no_grad(), autorelease_scope(self.device):
            if active_guard is not None:
                active_guard.check("backend forward start")
            logits = self.model(ids, prepend_bos=False)
            if logits.ndim != 3 or not torch.isfinite(logits).all().item():
                raise ValueError("model logits must be finite batch-token-vocabulary values")
            if active_guard is not None:
                active_guard.check("backend forward complete")
            return logits.detach()

    def activations(self, prompt_or_ids: str | torch.Tensor, *, sparse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ``(logits, activations)`` via ``ReplacementModel.get_activations`` or ``SAE.encode``."""
        if self.dictionary is None:
            raise ValueError("activations require a loaded dictionary")
        ids = self.tokenize(prompt_or_ids) if isinstance(prompt_or_ids, str) else prompt_or_ids.to(self.device)
        if ids.ndim == 2:
            ids_1d = ids[0]
        else:
            ids_1d = ids
            ids = ids.unsqueeze(0)
        with suspend(self.model), internal_forward(), torch.inference_mode(), autorelease_scope(self.device):
            if self.is_replacement_model and self.dictionary.kind == "transcoder":
                logits, acts = self.model.get_activations(ids_1d, sparse=sparse)
                if logits.ndim == 2:
                    logits = logits.unsqueeze(0)
                return logits.detach(), acts.detach()
            sae_map = self.dictionary.raw
            hook_names = [str(sae.cfg.metadata.hook_name) for sae in sae_map.values()]
            logits, cache = self.model.run_with_cache(ids, prepend_bos=False, names_filter=hook_names)
            layers_out = []
            for l in range(self.dictionary.metadata["layers"]):
                if l in sae_map:
                    h = str(sae_map[l].cfg.metadata.hook_name)
                    enc = sae_map[l].encode(cache[h])[0]
                else:
                    enc = torch.zeros((ids.shape[1], self.dictionary.metadata["latents"]),
                                      device=self.device, dtype=self.dtype)
                layers_out.append(enc.to_sparse() if sparse else enc)
            acts = torch.stack(layers_out).coalesce() if sparse else torch.stack(layers_out)
            return logits.detach(), acts.detach()

    def specs_to_tuples(
        self,
        prompt_or_ids: str | torch.Tensor,
        interventions: Sequence,
        *,
        base_activations: torch.Tensor | None = None,
    ) -> list[tuple[int, int | slice, int, float | torch.Tensor]]:
        """Normalize Caliper spec dicts or ``(layer, pos, feature, value)`` tuples into ``circuit-tracer`` tuples."""
        if not interventions:
            return []
        ids = self.tokenize(prompt_or_ids) if isinstance(prompt_or_ids, str) else prompt_or_ids
        seq_len = int(ids.shape[-1])
        needs_base = any(isinstance(iv, dict) and iv.get("op") == "scale" for iv in interventions)
        acts = base_activations
        if needs_base and acts is None:
            _, acts = self.activations(ids, sparse=False)

        # Track resolved values per (layer, pos, feature) so ordered scale/clamp compose cleanly.
        resolved: dict[tuple[int, int, int], float | torch.Tensor] = {}
        order_keys: list[tuple[int, int, int]] = []
        raw_tuples: list[tuple[int, int | slice, int, float | torch.Tensor]] = []

        for item in interventions:
            if not isinstance(item, dict):
                layer, pos, feat, val = item
                if isinstance(pos, slice):
                    raw_tuples.append((int(layer), pos, int(feat), val))
                else:
                    key = (int(layer), int(pos), int(feat))
                    if key not in resolved:
                        order_keys.append(key)
                    resolved[key] = val
                continue
            op = item["op"]
            if op not in ("clamp", "scale"):
                raise ValueError(f"feature tuple conversion does not accept tensor op {op!r}")
            layer = self._spec_layer(item)
            feat = int(item["feature"])
            positions = item.get("positions", list(range(seq_len)))
            for pos in positions:
                p = int(pos)
                key = (layer, p, feat)
                if key not in resolved:
                    order_keys.append(key)
                if op == "clamp":
                    resolved[key] = float(item["value"])
                else:
                    cur = resolved.get(key)
                    if cur is None:
                        cur = float(acts[layer, p, feat].item())
                    resolved[key] = float(cur) * float(item["multiplier"])

        return [*raw_tuples, *((l, p, f, resolved[(l, p, f)]) for (l, p, f) in order_keys)]

    def _spec_layer(self, spec: dict) -> int:
        if "layer" in spec and isinstance(spec["layer"], int):
            return int(spec["layer"])
        return self.resolve_feature_layer(str(spec.get("point", "")))

    def resolve_feature_layer(self, point: str) -> int:
        if self.dictionary is None:
            raise InterventionError("features require a configured dictionary", "intervene.unknown_dictionary")
        writes = self.dictionary.metadata["write_points"]
        if point in writes:
            idx = writes.index(point)
            return self.dictionary.covered_layers[idx]
        m_sem = re.fullmatch(r"layers\.(\d+)\.mlp_out", point)
        if m_sem:
            layer = int(m_sem.group(1))
            if layer in self.dictionary.covered_layers and self.dictionary.kind == "transcoder":
                return layer
        if self.dictionary.kind == "transcoder":
            out_hook = self.dictionary.raw.feature_output_hook
            m_tl = re.fullmatch(rf"blocks\.(\d+)\.{re.escape(out_hook)}", point)
            if m_tl and int(m_tl.group(1)) in self.dictionary.covered_layers:
                return int(m_tl.group(1))
        raise InterventionError("features require a configured dictionary write point")

    def feature_count(self, layer: int, *, point: str | None = None) -> int:
        if self.dictionary is None or layer not in self.dictionary.covered_layers:
            raise InterventionError("dictionary has no features at the requested layer", "intervene.unknown_dictionary")
        return int(self.dictionary.metadata["latents"])

    def intervene_logits(
        self,
        prompt_or_ids: str | torch.Tensor,
        interventions: Sequence,
        *,
        freeze_attention: bool = True,
        return_activations: bool = False,
        guard=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run ``ReplacementModel.feature_intervention`` or delta-form ``SAE`` hooks."""
        if self.dictionary is None:
            raise ValueError("feature interventions require a loaded dictionary")
        active_guard = guard if guard is not None else self.guard
        ids = self.tokenize(prompt_or_ids) if isinstance(prompt_or_ids, str) else prompt_or_ids.to(self.device)
        ids_1d = ids[0] if ids.ndim == 2 else ids
        if interventions and all(not isinstance(iv, dict) for iv in interventions):
            tuples = list(interventions)
        else:
            tuples = self.specs_to_tuples(ids_1d, interventions)
        with suspend(self.model), internal_forward(), torch.no_grad(), autorelease_scope(self.device):
            if active_guard is not None:
                active_guard.check("backend feature intervention start")
            if self.is_replacement_model and self.dictionary.kind == "transcoder":
                logits, acts = self.model.feature_intervention(
                    ids_1d,
                    tuples,
                    freeze_attention=freeze_attention,
                    return_activations=return_activations,
                )
                if logits.ndim == 2:
                    logits = logits.unsqueeze(0)
                if active_guard is not None:
                    active_guard.check("backend feature intervention complete")
                return logits.detach(), (acts.detach() if acts is not None else None)

            # Standalone HookedTransformer + SAELens delta-form hook
            edits_by_layer: dict[int, list[tuple[int | slice, int, float]]] = {}
            for layer, pos, feat, val in tuples:
                edits_by_layer.setdefault(int(layer), []).append((pos, int(feat), float(val)))
            fwd_hooks = []
            for layer, edits in edits_by_layer.items():
                sae = self.dictionary.raw[layer]
                hook_name = str(sae.cfg.metadata.hook_name)

                def _hook(x, hook, *, _sae=sae, _edits=edits):
                    acts = _sae.encode(x)
                    base_dec = _sae.decode(acts).to(x.dtype)
                    acts_mod = acts.clone()
                    for pos, feat, val in _edits:
                        acts_mod[:, pos, feat] = val
                    return x + (_sae.decode(acts_mod).to(x.dtype) - base_dec)

                fwd_hooks.append((hook_name, _hook))
            logits = self.model.run_with_hooks(ids_1d.unsqueeze(0), prepend_bos=False, fwd_hooks=fwd_hooks)
            if active_guard is not None:
                active_guard.check("backend feature intervention complete")
            _, acts_out = self.activations(ids_1d) if return_activations else (None, None)
            return logits.detach(), acts_out

    def install_feature_hook(self, executor, spec: dict, *, layer: int, width: int, module,
                             on_error=None, is_muted=lambda h: False, selected_rows=lambda m: None):
        """Install a live removable ``clamp`` or ``scale`` hook for ``InterventionExecutor``."""
        if self.dictionary is None:
            raise InterventionError("feature interventions require a configured dictionary",
                                    "intervene.unknown_dictionary")
        self.dictionary.check_feature(layer, int(spec["feature"]))
        if self.dictionary.metadata["width"] != width:
            raise InterventionError("dictionary width does not match target point width")
        group = executor._feature_groups.get(layer)
        if group is None:
            group = _FeatureLayerGroup(self, executor, layer, module)
            executor._feature_groups[layer] = group
        hook = _FeatureHook(group, spec, on_error, is_muted, selected_rows)
        group.members.append(hook)
        return hook

    def generate(
        self,
        prompt: str,
        interventions: Sequence = (),
        *,
        max_new_tokens: int = 24,
        freeze_attention: bool = True,
        open_ended: bool = False,
        guard=None,
    ) -> dict:
        """Generate paired greedy continuations while preserving permanent model hooks."""
        ids = self.tokenize(prompt)
        start = int(ids.shape[1])
        tuples = self.specs_to_tuples(ids, interventions) if interventions and self.dictionary is not None else []
        if open_ended:
            tuples = [(l, slice(None), f, v) for (l, _p, f, v) in tuples]

        with suspend(self.model), internal_forward(), torch.no_grad(), autorelease_scope(self.device):
            if guard is not None:
                guard.check("backend generate baseline")
            try:
                base_out = self.model.generate(
                    ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    verbose=False,
                    stop_at_eos=False,
                    prepend_bos=False,
                )
            finally:
                self.model.reset_hooks(including_permanent=False)
            base_ids = base_out[0, start:].tolist()

            if not tuples:
                steered_ids = list(base_ids)
            elif self.is_replacement_model and self.dictionary is not None and self.dictionary.kind == "transcoder":
                if guard is not None:
                    guard.check("backend generate steered")
                prepend = bool(getattr(self.model.cfg, "default_prepend_bos", True))
                try:
                    steered_out, _, _ = self.model.feature_intervention_generate(
                        prompt,
                        tuples,
                        freeze_attention=freeze_attention,
                        return_activations=False,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        verbose=False,
                        stop_at_eos=False,
                        prepend_bos=prepend,
                        return_type="tokens",
                    )
                finally:
                    self.model.reset_hooks(including_permanent=False)
                steered_ids = steered_out[0, start:].tolist()
            else:
                edits_by_layer: dict[int, list[tuple[int | slice, int, float]]] = {}
                for layer, pos, feat, val in tuples:
                    edits_by_layer.setdefault(int(layer), []).append((pos, int(feat), float(val)))
                cur = ids.clone()
                try:
                    for step_idx in range(max_new_tokens):
                        fwd_hooks = []
                        for layer, edits in edits_by_layer.items():
                            sae = self.dictionary.raw[layer]
                            hook_name = str(sae.cfg.metadata.hook_name)

                            def _step_hook(x, hook, *, _sae=sae, _edits=edits):
                                acts = _sae.encode(x)
                                base_dec = _sae.decode(acts).to(x.dtype)
                                acts_mod = acts.clone()
                                for pos, feat, val in _edits:
                                    if isinstance(pos, slice) or (isinstance(pos, int) and pos < acts_mod.shape[1]):
                                        acts_mod[:, pos, feat] = val
                                return x + (_sae.decode(acts_mod).to(x.dtype) - base_dec)

                            fwd_hooks.append((hook_name, _step_hook))
                        step_logits = self.model.run_with_hooks(cur, prepend_bos=False, fwd_hooks=fwd_hooks)
                        nxt = step_logits[:, -1].argmax(dim=-1, keepdim=True)
                        cur = torch.cat((cur, nxt), dim=1)
                finally:
                    self.model.reset_hooks(including_permanent=False)
                steered_ids = cur[0, start:].tolist()

        return {
            "baseline": self.tokenizer.decode(base_ids),
            "steered": self.tokenizer.decode(steered_ids),
            "token_ids": [base_ids, steered_ids],
        }

    def patch_sweep(
        self,
        clean_prompt_or_ids: str | torch.Tensor,
        corrupt_prompt_or_ids: str | torch.Tensor,
        *,
        positive_token: int,
        negative_token: int,
        batch_size: int = 32,
        guard=None,
    ) -> dict:
        """Run a bounded ``hook_resid_pre`` activation patching sweep (at most ``batch_size`` cells per forward)."""
        vocab = int(self.model.cfg.d_vocab)
        if (type(positive_token) is not int or type(negative_token) is not int
                or not 0 <= positive_token < vocab or not 0 <= negative_token < vocab
                or positive_token == negative_token):
            raise ValueError("metric requires two distinct vocabulary token IDs")
        clean_ids = self.tokenize(clean_prompt_or_ids) if isinstance(clean_prompt_or_ids, str) else clean_prompt_or_ids.to(self.device)
        corrupt_ids = self.tokenize(corrupt_prompt_or_ids) if isinstance(corrupt_prompt_or_ids, str) else corrupt_prompt_or_ids.to(self.device)
        if clean_ids.shape != corrupt_ids.shape:
            raise ValueError("clean and corrupted prompts must have equal token counts")

        n_layers = int(self.model.cfg.n_layers)
        n_pos = int(clean_ids.shape[1])
        active_guard = guard if guard is not None else self.guard

        def _score(logits: torch.Tensor) -> torch.Tensor:
            return logits[:, -1, positive_token].float() - logits[:, -1, negative_token].float()

        with suspend(self.model), internal_forward(), torch.no_grad(), autorelease_scope(self.device):
            if active_guard is not None:
                active_guard.check("patch sweep clean cache")
            clean_logits, clean_cache = self.model.run_with_cache(
                clean_ids,
                prepend_bos=False,
                names_filter=lambda n: n.endswith("hook_resid_pre"),
            )
            clean_val = float(_score(clean_logits)[0].item())
            corrupt_logits = self.model(corrupt_ids, prepend_bos=False)
            corrupt_val = float(_score(corrupt_logits)[0].item())
            denom = clean_val - corrupt_val
            if not math.isfinite(denom) or abs(denom) <= 1e-12:
                raise ValueError("patch recovery is undefined because the clean metric equals the corrupted metric")

            cells = [(l, p) for l in range(n_layers) for p in range(n_pos)]
            raw_diffs: list[torch.Tensor] = []
            for start in range(0, len(cells), batch_size):
                chunk = cells[start:start + batch_size]
                if active_guard is not None:
                    active_guard.check("patch sweep batch")
                by_layer: dict[int, list[tuple[int, int]]] = {}
                for row_idx, (layer, pos) in enumerate(chunk):
                    by_layer.setdefault(layer, []).append((row_idx, pos))
                fwd_hooks = []
                for layer, pairs in by_layer.items():
                    h_name = f"blocks.{layer}.hook_resid_pre"
                    src = clean_cache[h_name][0]
                    rows_t = torch.tensor([r for r, _ in pairs], device=self.device, dtype=torch.long)
                    pos_t = torch.tensor([p for _, p in pairs], device=self.device, dtype=torch.long)

                    def _patch_fn(acts, hook, *, _src=src, _r=rows_t, _p=pos_t):
                        out = acts.clone()
                        out[_r, _p] = _src[_p]
                        return out

                    fwd_hooks.append((h_name, _patch_fn))
                tiled = corrupt_ids.expand(len(chunk), -1).clone()
                patched_logits = self.model.run_with_hooks(tiled, prepend_bos=False, fwd_hooks=fwd_hooks)
                raw_diffs.append((_score(patched_logits) - corrupt_val) / denom)

            grid = torch.cat(raw_diffs).reshape(n_layers, n_pos).to(torch.float32)
            if not torch.isfinite(grid).all().item():
                raise ValueError("patch recovery contains a nonfinite value")
            del clean_cache

        return {
            "clean_metric": clean_val,
            "corrupted_metric": corrupt_val,
            "normalizer": denom,
            "grid": grid,
            "recovery": grid.cpu().tolist(),
            "layers": n_layers,
            "positions": n_pos,
        }

    def top_examples(
        self,
        layer: int,
        feature: int,
        *,
        prompts: Sequence[str] | None = None,
        limit: int = 16,
        allow_online: bool = False,
    ) -> list[dict]:
        """Return up to ``limit`` centered 33-token activation windows via Neuronpedia or a local min-heap."""
        if self.dictionary is None:
            raise ValueError("top_examples requires a loaded dictionary")
        self.dictionary.check_feature(layer, feature)
        if limit == 0:
            return []

        np_id = self.dictionary.metadata.get("neuronpedia_id")
        if allow_online and np_id and os.environ.get("CALIPER_AGENT_OFFLINE") != "1":
            try:
                from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_feature
                np_model, np_layer_ds = str(np_id).split("/", 1)
                np_layer_s, np_ds = np_layer_ds.split("-", 1)
                payload = get_neuronpedia_feature(
                    feature=feature, layer=int(np_layer_s), model=np_model, dataset=np_ds
                )
            except Exception:
                payload = None
            if isinstance(payload, dict):
                rows = []
                for rank, item in enumerate((payload.get("activations") or [])[:limit]):
                    toks = list(item.get("tokens") or [])
                    vals = [float(v) for v in (item.get("values") or [0.0] * len(toks))]
                    if not toks:
                        continue
                    pos = int(max(range(len(vals)), key=vals.__getitem__))
                    rows.append(self._pack_str_window(rank, f"neuronpedia-{rank}", toks, vals, pos, self.tokenizer))
                if rows:
                    return rows

        corpus = list(prompts) if prompts is not None else [*self.recent_prompts, *DEFAULT_CORPUS]
        seen: list[str] = []
        for p in corpus:
            if isinstance(p, str) and p and p not in seen:
                seen.append(p)

        heap: list[tuple[float, int, int, list[int], list[float], str]] = []
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        with internal_forward(), torch.inference_mode(), autorelease_scope(self.device):
            for doc_idx, text_prompt in enumerate(seen):
                try:
                    ids = self.tokenize(text_prompt)
                except ValueError:
                    continue
                _, acts = self.activations(ids, sparse=False)
                feat_acts = acts[layer, :, feature].float().cpu()
                if feat_acts.numel() > 1 and bos_id is not None and int(ids[0, 0].item()) == int(bos_id):
                    feat_acts = feat_acts.clone()
                    feat_acts[0] = 0.0
                pos = int(feat_acts.argmax().item())
                val = float(feat_acts[pos].item())
                if val <= 0.0:
                    continue
                entry = (val, -doc_idx, pos, ids[0].cpu().tolist(), feat_acts.tolist(), f"prompt-{doc_idx}")
                if len(heap) < limit:
                    heapq.heappush(heap, entry)
                elif (val, -doc_idx) > (heap[0][0], heap[0][1]):
                    heapq.heapreplace(heap, entry)

        ordered = sorted(heap, key=lambda it: (-it[0], -it[1], it[2]))
        return [
            self._pack_id_window(rank, doc_id, tok_ids, vals, pos)
            for rank, (_val, _neg_doc, pos, tok_ids, vals, doc_id) in enumerate(ordered)
        ]

    def _pack_id_window(self, rank: int, doc_id: str, tok_ids: list[int], vals: list[float], position: int) -> dict:
        begin, end = max(0, position - 16), min(len(tok_ids), position + 17)
        left = 16 - (position - begin)
        sub_ids = tok_ids[begin:end]
        sub_vals = vals[begin:end]
        slots: list[int | None] = [None] * WINDOW_TOKENS
        slots[left:left + len(sub_ids)] = sub_ids
        act_slots: list[float | None] = [None] * WINDOW_TOKENS
        act_slots[left:left + len(sub_vals)] = [float(v) for v in sub_vals]
        pos_slots: list[int | None] = [None] * WINDOW_TOKENS
        pos_slots[left:left + len(sub_ids)] = list(range(begin, end))
        return {
            "example_id": rank,
            "document_id": doc_id,
            "position": position,
            "activation": float(vals[position]),
            "window_start": begin,
            "center": 16,
            "token_ids": slots,
            "tokens": [token_label(self.tokenizer, i) if i is not None else None for i in slots],
            "positions": pos_slots,
            "valid": [i is not None for i in slots],
            "activations": act_slots,
            "document_truncated": False,
        }

    @staticmethod
    def _str_token_id(tokenizer, tok: str) -> int | None:
        if tokenizer is None or not isinstance(tok, str):
            return None
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            try:
                ids = encode(tok, add_special_tokens=False)
                if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], int) and ids[0] >= 0:
                    return ids[0]
            except Exception:
                pass
        conv = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(conv):
            try:
                tid = conv(tok)
                unk = getattr(tokenizer, "unk_token_id", None)
                unk_tok = getattr(tokenizer, "unk_token", None)
                if isinstance(tid, int) and tid >= 0 and (tid != unk or tok == unk_tok):
                    return tid
            except Exception:
                pass
        return None

    @classmethod
    def _pack_str_window(cls, rank: int, doc_id: str, toks: list[str], vals: list[float], position: int,
                         tokenizer=None) -> dict:
        begin, end = max(0, position - 16), min(len(toks), position + 17)
        left = 16 - (position - begin)
        sub_toks = toks[begin:end]
        sub_vals = vals[begin:end]
        id_slots: list[int | None] = [None] * WINDOW_TOKENS
        id_slots[left:left + len(sub_toks)] = [cls._str_token_id(tokenizer, tok) for tok in sub_toks]
        tok_slots: list[str | None] = [None] * WINDOW_TOKENS
        tok_slots[left:left + len(sub_toks)] = sub_toks
        act_slots: list[float | None] = [None] * WINDOW_TOKENS
        act_slots[left:left + len(sub_vals)] = [float(v) for v in sub_vals]
        pos_slots: list[int | None] = [None] * WINDOW_TOKENS
        pos_slots[left:left + len(sub_toks)] = list(range(begin, end))
        return {
            "example_id": rank,
            "document_id": doc_id,
            "position": position,
            "activation": float(vals[position]),
            "window_start": begin,
            "center": 16,
            "token_ids": id_slots,
            "tokens": tok_slots,
            "positions": pos_slots,
            "valid": [t is not None for t in tok_slots],
            "activations": act_slots,
            "document_truncated": False,
        }

    def attribute_and_prune(
        self,
        prompt: str,
        *,
        max_n_logits: int = 10,
        desired_logit_prob: float = 0.95,
        max_feature_nodes: int | None = 5000,
        batch_size: int | None = None,
        update_interval: int = 8,
        node_threshold: float | None = None,
        edge_threshold: float | None = None,
        show_more: bool = False,
        max_nodes: int = LARGE_NODE_CAP,
        max_edges: int = LARGE_EDGE_CAP,
        trace_id: str = "trace-1",
        index=None,
        guard=None,
    ) -> Graph:
        """Run ``circuit_tracer.attribute``, transfer the graph to CPU, prune, and build a Caliper ``Graph``."""
        if not self.is_replacement_model:
            raise ValueError("circuit attribution requires a ReplacementModel with transcoders")
        from circuit_tracer import attribute
        from circuit_tracer.graph import compute_graph_scores, compute_node_influence, prune_graph

        eff_node_threshold, eff_edge_threshold = resolve_pruning_thresholds(
            show_more=show_more,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )
        ids = self.tokenize(prompt)
        self.record_prompt(prompt)
        active_guard = guard if guard is not None else self.guard
        eff_batch = int(batch_size) if batch_size is not None else (256 if self.device.type == "mps" else 512)
        settings = {
            "max_n_logits": max_n_logits,
            "desired_logit_prob": float(desired_logit_prob),
            "max_feature_nodes": max_feature_nodes,
            "batch_size": eff_batch,
            "update_interval": update_interval,
            "node_threshold": eff_node_threshold,
            "edge_threshold": eff_edge_threshold,
            "show_more": bool(show_more),
        }

        if max_feature_nodes == 0:
            # Degenerate zero-feature trace: keep only tokens, errors, and salient logits.
            return self._logit_only_graph(
                ids, max_n_logits=max_n_logits, desired_logit_prob=desired_logit_prob,
                max_nodes=max_nodes, max_edges=max_edges, trace_id=trace_id,
                settings=settings, guard=active_guard,
            )

        with internal_forward(), autorelease_scope(self.device), mps_batch_pool_scope(self.device):
            if active_guard is not None:
                active_guard.check("attribute start")
            ct_graph = attribute(
                prompt,
                self.model,
                max_n_logits=max_n_logits,
                desired_logit_prob=desired_logit_prob,
                batch_size=eff_batch,
                max_feature_nodes=max_feature_nodes,
                offload=None,
                verbose=False,
                update_interval=update_interval,
            )
            if active_guard is not None:
                active_guard.check("attribute complete")

        ct_graph.to("cpu")
        ct_graph.input_tokens = ct_graph.input_tokens.cpu()
        ct_graph.activation_values = ct_graph.activation_values.cpu()
        ct_graph.selected_features = ct_graph.selected_features.cpu()
        mps_cleanup(self.device)

        pruned = prune_graph(
            ct_graph,
            node_threshold=eff_node_threshold,
            edge_threshold=eff_edge_threshold,
        )
        rep_score, comp_score = compute_graph_scores(ct_graph)
        n_logits = len(ct_graph.logit_targets)
        logit_weights = torch.zeros(ct_graph.adjacency_matrix.shape[0], dtype=torch.float32)
        if n_logits > 0:
            logit_weights[-n_logits:] = ct_graph.logit_probabilities.float().cpu()
        infl = compute_node_influence(ct_graph.adjacency_matrix, logit_weights)

        feature_labels = None
        if index is not None and hasattr(index, "labels"):
            n_feats = len(ct_graph.selected_features)
            kept_mask = pruned.node_mask[:n_feats].tolist()
            coords = ct_graph.active_features[ct_graph.selected_features].tolist()
            kept_pairs = []
            seen_pairs = set()
            for (l, _p, f), keep in zip(coords, kept_mask):
                pair = (int(l), int(f))
                if keep and pair not in seen_pairs and len(kept_pairs) < 512:
                    seen_pairs.add(pair)
                    kept_pairs.append(pair)
            if kept_pairs:
                feature_labels = index.labels(kept_pairs)

        return Graph.from_circuit_tracer(
            ct_graph,
            pruned,
            replacement_score=rep_score,
            completeness_score=comp_score,
            node_influence=infl,
            tokenizer=self.tokenizer,
            device=self.device,
            trace_id=trace_id,
            max_nodes=max_nodes,
            max_edges=max_edges,
            feature_labels=feature_labels,
            settings=settings,
            memory=active_guard,
        )

    def _logit_only_graph(self, ids: torch.Tensor, *, max_n_logits: int, desired_logit_prob: float,
                          max_nodes: int, max_edges: int, trace_id: str, settings: dict | None = None,
                          guard=None) -> Graph:
        logits = self.forward_logits(ids, guard=guard)[0, -1].float().cpu()
        probs = logits.softmax(-1)
        order = torch.argsort(probs, descending=True, stable=True)[:max_n_logits]
        cum = probs[order].cumsum(0)
        take = min(len(order), int((cum < desired_logit_prob).sum().item()) + 1)
        chosen = order[:max(1, take)].tolist()
        input_ids = ids[0].cpu().tolist()
        n_layers = int(self.model.cfg.n_layers)
        tokens = [
            {"position": p, "token_id": int(t), "text": token_label(self.tokenizer, int(t))}
            for p, t in enumerate(input_ids)
        ]
        logit_nodes = [
            {
                "id": f"logit:{len(tokens) - 1}:{int(t)}",
                "kind": "logit",
                "layer": n_layers,
                "position": len(tokens) - 1,
                "token_id": int(t),
                "probability": float(probs[t].item()),
                "label": token_label(self.tokenizer, int(t)),
                "activation": float(probs[t].item()),
                "index": i,
                "influence": 0.0,
                "cumulative_influence": 0.0,
                "effect": 0.0,
            }
            for i, t in enumerate(chosen)
        ]
        kept = logit_nodes[:max_nodes]
        for i, n in enumerate(kept):
            n["index"] = i
        return Graph.build(
            nodes=kept,
            edges=[],
            tokens=tokens,
            layers=n_layers,
            completeness={
                "replacement_score": 0.0,
                "completeness_score": 0.0,
                "explained_fraction": 0.0,
                "full_nodes": len(logit_nodes),
                "kept_nodes": len(kept),
                "kept_features": 0,
                "kept_nonzero_edges": 0,
            },
            omitted_influence=0.0,
            error_share=0.0,
            feature_share=0.0,
            trace_id=trace_id,
            device=self.device,
            logits=logit_nodes,
            settings=settings,
            memory=guard,
        )


def _build_tiny_tokenizer(d_vocab: int = 128):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
    for code in range(32, 127):
        ch = chr(code)
        if len(vocab) < d_vocab and ch not in vocab:
            vocab[ch] = len(vocab)
    while len(vocab) < d_vocab:
        vocab[f"t{len(vocab)}"] = len(vocab)

    tok_obj = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tok_obj.pre_tokenizer = Split("", behavior="isolated")
    tok_obj.post_processor = TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
    return PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        add_bos_token=True,
        clean_up_tokenization_spaces=False,
        name_or_path="tiny-lm",
    )


def make_tiny_backend(
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    n_layers: int = 2,
    d_model: int = 16,
    n_heads: int = 2,
    d_mlp: int = 32,
    n_features: int = 32,
    d_vocab: int = 128,
    seed: int = 42,
    guard=None,
) -> Backend:
    """Create a deterministic offline 2-layer ``ReplacementModel`` + ``TranscoderSet`` fixture."""
    from circuit_tracer import ReplacementModel
    from circuit_tracer.transcoder.activation_functions import JumpReLU
    from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder, TranscoderSet
    from transformer_lens import HookedTransformerConfig

    dev = torch.device(device)
    tok = _build_tiny_tokenizer(d_vocab=d_vocab)
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_head=d_model // n_heads,
        n_heads=n_heads,
        d_mlp=d_mlp,
        n_ctx=64,
        d_vocab=d_vocab,
        act_fn="relu",
        normalization_type="RMS",
        dtype=dtype,
        device=str(dev),
        seed=seed,
        model_name="tiny-lm",
        default_prepend_bos=True,
    )
    rng = np.random.default_rng(seed)
    tc_modules = {}
    for layer in range(n_layers):
        thresh = torch.full((n_features,), 0.05, device=dev, dtype=dtype)
        act_fn = JumpReLU(thresh, 0.1)
        tc = SingleLayerTranscoder(d_model, n_features, act_fn, layer, device=dev, dtype=dtype)
        state = {
            "W_enc": torch.tensor(rng.normal(0.0, 0.25, size=(n_features, d_model)), device=dev, dtype=dtype),
            "W_dec": torch.tensor(rng.normal(0.0, 0.25, size=(n_features, d_model)), device=dev, dtype=dtype),
            "b_enc": torch.full((n_features,), 0.4, device=dev, dtype=dtype),
            "b_dec": torch.zeros((d_model,), device=dev, dtype=dtype),
            "activation_function.threshold": thresh,
        }
        tc.load_state_dict(state, assign=True)
        tc_modules[layer] = tc

    tc_set = TranscoderSet(
        tc_modules,
        feature_input_hook="ln2.hook_normalized",
        feature_output_hook="hook_mlp_out",
        scan_name="tiny-lm",
    )
    rm = ReplacementModel.from_config(cfg, tc_set, backend="transformerlens", tokenizer=tok)
    rm.eval()
    view = DictionaryView(tc_set, kind="transcoder", n_model_layers=n_layers, dictionary_id="tiny-lm", device=dev)
    return Backend(rm, tok, view, device=dev, dtype=dtype, model_id="tiny-lm", guard=guard)


def make_tiny_sae_backend(
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    layer: int = 0,
    d_model: int = 16,
    n_features: int = 32,
    d_vocab: int = 128,
    seed: int = 42,
    guard=None,
) -> Backend:
    """Create a deterministic offline 2-layer ``HookedTransformer`` + ``SAELens`` ``StandardSAE`` fixture."""
    from sae_lens import StandardSAE, StandardSAEConfig
    from sae_lens.saes.sae import SAEMetadata
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    dev = torch.device(device)
    tok = _build_tiny_tokenizer(d_vocab=d_vocab)
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=d_model,
        d_head=d_model // 2,
        n_heads=2,
        d_mlp=32,
        n_ctx=64,
        d_vocab=d_vocab,
        act_fn="relu",
        normalization_type="RMS",
        dtype=dtype,
        device=str(dev),
        seed=seed,
        model_name="tiny-sae-lm",
        default_prepend_bos=True,
    )
    model = HookedTransformer(cfg, tokenizer=tok)
    model.eval()
    str_dtype = "bfloat16" if dtype == torch.bfloat16 else ("float16" if dtype == torch.float16 else "float32")
    sae_cfg = StandardSAEConfig(
        d_in=d_model,
        d_sae=n_features,
        dtype=str_dtype,
        device="cpu",
        metadata=SAEMetadata(
            model_name="tiny-sae-lm",
            hook_name=f"blocks.{layer}.hook_resid_pre",
            hook_layer=layer,
        ),
    )
    # Stage on CPU first, initialize deterministic positive bias so features fire, then move to target device.
    sae = StandardSAE(sae_cfg)
    rng = np.random.default_rng(seed + 7)
    with torch.no_grad():
        sae.W_enc.copy_(torch.tensor(rng.normal(0.0, 0.25, size=(d_model, n_features))))
        sae.W_dec.copy_(torch.tensor(rng.normal(0.0, 0.25, size=(n_features, d_model))))
        sae.b_enc.fill_(0.35)
        sae.b_dec.zero_()
    sae = sae.to(device=dev, dtype=dtype)
    sae.eval()
    view = DictionaryView(sae, kind="sae", n_model_layers=2, dictionary_id="tiny-sae", device=dev)
    return Backend(model, tok, view, device=dev, dtype=dtype, model_id="tiny-sae-lm", guard=guard)


def _infer_official_tl_name(model_dir: Path) -> str:
    for parent in (model_dir, *model_dir.parents):
        if parent.name.startswith("models--"):
            parts = parent.name.split("--")
            if len(parts) == 3:
                return f"{parts[1]}/{parts[2]}"
    cfg_path = model_dir / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        name = cfg.get("_name_or_path") or ""
        if "/" in name and not Path(name).exists():
            return str(name)
        mtype = cfg.get("model_type")
        if mtype == "gemma2":
            hidden = cfg.get("hidden_size")
            if hidden == 2304:
                return "google/gemma-2-2b"
    raise ValueError(f"cannot infer TransformerLens architecture name from {model_dir}")


def _resolve_gemma_scope_npz(layer_dir: Path, layer: int) -> Path:
    width_dir = layer_dir / "width_16k"
    search_root = width_dir if width_dir.is_dir() else layer_dir
    if layer < len(_GEMMA_2B_CANONICAL_L0):
        preferred = search_root / f"average_l0_{_GEMMA_2B_CANONICAL_L0[layer]}" / "params.npz"
        if preferred.is_file():
            return preferred
    candidates: list[tuple[int, Path]] = []
    for child in sorted(search_root.glob("average_l0_*")):
        npz = child / "params.npz"
        m = re.match(r"average_l0_(\d+)$", child.name)
        if npz.is_file() and m:
            candidates.append((int(m.group(1)), npz))
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]
    direct = layer_dir / "params.npz"
    if direct.is_file():
        return direct
    raise FileNotFoundError(f"no Gemma Scope params.npz found under {layer_dir}")


def _load_local_transcoder_set(dict_dir: Path, n_layers: int, *, device: torch.device, dtype: torch.dtype):
    from circuit_tracer.transcoder.single_layer_transcoder import (
        TranscoderSet,
        load_gemma_scope_transcoder,
        load_transcoder_set,
    )

    st_files = {l: str(dict_dir / f"layer_{l}.safetensors") for l in range(n_layers) if (dict_dir / f"layer_{l}.safetensors").is_file()}
    if len(st_files) == n_layers:
        return load_transcoder_set(
            st_files,
            scan_name=dict_dir.name,
            feature_input_hook="ln2.hook_normalized",
            feature_output_hook="hook_mlp_out",
            device=device,
            dtype=dtype,
            lazy_encoder=False,
            lazy_decoder=True,
        )

    tc_modules = {}
    for layer in range(n_layers):
        npz_path = _resolve_gemma_scope_npz(dict_dir / f"layer_{layer}", layer)
        with autorelease_scope(device):
            tc_modules[layer] = load_gemma_scope_transcoder(str(npz_path), layer, device=device, dtype=dtype)
    mps_cleanup(device)
    scan = "gemma" if "gemma-scope" in str(dict_dir) else dict_dir.name
    return TranscoderSet(
        tc_modules,
        feature_input_hook="ln2.hook_normalized",
        feature_output_hook="hook_mlp_out",
        scan_name=scan,
    )


def _is_transcoder_source(dictionary_ref) -> bool:
    if dictionary_ref is None:
        return False
    from circuit_tracer.transcoder.single_layer_transcoder import TranscoderSet
    if isinstance(dictionary_ref, TranscoderSet):
        return True
    s = str(dictionary_ref)
    if s in {"gemma", "llama"}:
        return True
    p = Path(s)
    if p.is_dir() and ((p / "layer_0").is_dir() or (p / "layer_0.safetensors").is_file()):
        return True
    return False


def load_backend(
    model_ref: str | Path,
    dictionary_ref=None,
    *,
    sae_id: str | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    guard=None,
) -> Backend:
    """Load a staged ``ReplacementModel`` or ``HookedTransformer`` (+ optional ``SAE``) backend."""
    if str(model_ref) == "tiny-lm":
        dev = torch.device(device) if device is not None else torch.device("cpu")
        dt = dtype if dtype is not None else torch.float32
        backend = make_tiny_backend(device=dev, dtype=dt, guard=guard)
        if dictionary_ref is not None:
            p = Path(str(dictionary_ref))
            if p.is_dir() and (p / "layer_0.safetensors").is_file():
                tc_set = _load_local_transcoder_set(p, int(backend.model.cfg.n_layers), device=dev, dtype=dt)
                backend.model._configure_replacement_model(tc_set)
                backend.dictionary = DictionaryView(
                    tc_set, kind="transcoder", n_model_layers=int(backend.model.cfg.n_layers),
                    dictionary_id="tiny-lm", device=dev,
                )
        return backend

    dev = torch.device(device) if device is not None else default_device()
    dt = dtype if dtype is not None else (torch.bfloat16 if dev.type in ("mps", "cuda") else torch.float32)
    model_path = Path(str(model_ref))
    is_local_model = model_path.is_dir()
    official_name = _infer_official_tl_name(model_path) if is_local_model else str(model_ref)

    if _is_transcoder_source(dictionary_ref):
        from circuit_tracer.replacement_model.replacement_model_transformerlens import (
            TransformerLensReplacementModel,
        )
        from circuit_tracer.transcoder.single_layer_transcoder import TranscoderSet
        from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with autorelease_scope(dev):
            if is_local_model:
                tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
                hf_model = AutoModelForCausalLM.from_pretrained(
                    str(model_path), dtype=dt, low_cpu_mem_usage=True, local_files_only=True
                )
                hf_model.eval()
                model = super(TransformerLensReplacementModel, TransformerLensReplacementModel).from_pretrained(
                    official_name,
                    hf_model=hf_model,
                    tokenizer=tok,
                    fold_ln=False,
                    center_writing_weights=False,
                    center_unembed=False,
                    device=dev,
                    dtype=dt,
                )
                del hf_model
            else:
                model = super(TransformerLensReplacementModel, TransformerLensReplacementModel).from_pretrained(
                    official_name,
                    fold_ln=False,
                    center_writing_weights=False,
                    center_unembed=False,
                    device=dev,
                    dtype=dt,
                )
            mps_cleanup(dev)
            if guard is not None:
                guard.check("backend model loaded")

            if isinstance(dictionary_ref, TranscoderSet):
                transcoders = dictionary_ref
                dict_id = getattr(transcoders, "scan_name", None) or "transcoders"
            elif Path(str(dictionary_ref)).is_dir():
                dict_dir = Path(str(dictionary_ref))
                transcoders = _load_local_transcoder_set(dict_dir, int(model.cfg.n_layers), device=dev, dtype=dt)
                dict_id = getattr(transcoders, "scan_name", None) or dict_dir.name
            else:
                transcoders, _ = load_transcoder_from_hub(
                    str(dictionary_ref),
                    device=dev,
                    dtype=dt,
                    lazy_encoder=False,
                    lazy_decoder=True,
                )
                dict_id = str(dictionary_ref)

            model._configure_replacement_model(transcoders)
            model.eval()
            mps_cleanup(dev)
            if guard is not None:
                guard.check("backend transcoders loaded")

        view = DictionaryView(
            transcoders, kind="transcoder", n_model_layers=int(model.cfg.n_layers),
            dictionary_id=str(dict_id), device=dev,
        )
        return Backend(model, model.tokenizer, view, device=dev, dtype=dt, model_id=official_name, guard=guard)

    # Standalone HookedTransformer (+ optional CPU-staged SAELens SAE)
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with autorelease_scope(dev):
        if is_local_model:
            tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
            hf_model = AutoModelForCausalLM.from_pretrained(
                str(model_path), dtype=dt, low_cpu_mem_usage=True, local_files_only=True
            )
            hf_model.eval()
            model = HookedTransformer.from_pretrained(
                official_name,
                hf_model=hf_model,
                tokenizer=tok,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
                fold_value_biases=False,
                device=dev,
                dtype=dt,
            )
            del hf_model
        else:
            model = HookedTransformer.from_pretrained(
                official_name,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
                fold_value_biases=False,
                device=dev,
                dtype=dt,
            )
        model.eval()
        mps_cleanup(dev)
        if guard is not None:
            guard.check("backend model loaded")

    view = None
    if dictionary_ref is not None:
        from sae_lens import SAE
        str_dtype = "bfloat16" if dt == torch.bfloat16 else ("float16" if dt == torch.float16 else "float32")
        with autorelease_scope(dev):
            if isinstance(dictionary_ref, SAE):
                sae = dictionary_ref.to(device=dev, dtype=dt)
                dict_id = str(getattr(sae.cfg.metadata, "neuronpedia_id", None) or "sae")
            elif Path(str(dictionary_ref)).is_dir():
                sae = SAE.load_from_disk(str(dictionary_ref), device="cpu").to(device=dev, dtype=dt)
                dict_id = Path(str(dictionary_ref)).name
            else:
                if not sae_id:
                    raise ValueError("sae_id is required when loading a SAELens release by name")
                sae = SAE.from_pretrained(
                    release=str(dictionary_ref), sae_id=sae_id, device="cpu", dtype=str_dtype
                ).to(device=dev, dtype=dt)
                dict_id = f"{dictionary_ref}:{sae_id}"
            sae.eval()
            mps_cleanup(dev)
            if guard is not None:
                guard.check("backend sae loaded")
        view = DictionaryView(sae, kind="sae", n_model_layers=int(model.cfg.n_layers), dictionary_id=dict_id, device=dev)

    return Backend(model, model.tokenizer, view, device=dev, dtype=dt, model_id=official_name, guard=guard)
