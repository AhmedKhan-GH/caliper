"""Published Dallas circuit intervention presets for HostedInference."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import time

import torch

from .._intervention_protocol import internal_forward
from ._memory import CleanStop, autorelease_scope


class _HostedPresetsMixin:
    """Preset actions loaded from an external circuit intervention manifest."""

    def _dallas_manifest_path(self) -> Path | None:
        raw = os.environ.get("CALIPER_PRESET_MANIFEST", "").strip()
        if not raw:
            return None
        manifest = Path(raw).expanduser()
        return manifest if manifest.is_file() else None

    def _register_dallas_actions(self, caliper) -> None:
        manifest_path = self._dallas_manifest_path()
        if (manifest_path is None or self.dictionary.kind != "transcoder"
                or self.dictionary.metadata["layers"] != 26 or self.dictionary.metadata["latents"] < 16384):
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for case in manifest["interventions"]:
            cell = int(case["cell"])
            self.actions.append(
                caliper.action(f"dallas.cell{cell}", lambda _c=cell: self._run_dallas_cell(_c), help=case["label"])
            )

    def _ensure_dallas_cache(self) -> dict:
        if self._dallas_cache is not None:
            return self._dallas_cache
        manifest_path = self._dallas_manifest_path()
        if manifest_path is None:
            raise ValueError("Published Dallas manifest is unavailable.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        t0 = time.perf_counter()
        prompt_acts: dict[str, torch.Tensor] = {}
        dallas_logits = None
        items = list(manifest["prompts"].items())
        for idx, (key, prompt_text) in enumerate(items):
            self._report_progress(
                state="running",
                stage="intervention",
                label=f"Caching intervention prompts ({idx + 1}/{len(items)})",
                done=idx + 1,
                total=len(items),
                panels=self._active_panel_set(("output",)),
            )
            logits, acts = self.backend.activations(prompt_text, sparse=False)
            prompt_acts[key] = acts.detach().cpu()
            if key == "dallas":
                dallas_logits = logits.detach()
            del logits, acts
        self._dallas_cache = {
            "manifest": manifest,
            "acts": prompt_acts,
            "dallas_logits": dallas_logits,
            "cache_seconds": time.perf_counter() - t0,
        }
        return self._dallas_cache

    def _run_dallas_cell(self, cell: int) -> str:
        from ._output import _decode_raw_token
        from ._publish import _publication_scope, _publish_many

        self._check_stop()
        with internal_forward():
            self._set_progress("tracing")
            try:
                with self.controls.lock:
                    self._discard_owned_clamp()
                self._apply_interventions()
                cache = self._ensure_dallas_cache()
                manifest = cache["manifest"]
                case = next(c for c in manifest["interventions"] if int(c["cell"]) == cell)
                tuples = []
                for op in case["operations"]:
                    src_acts = cache["acts"][op["activation_prompt"]]
                    factor = op["factor"]
                    for node in manifest["groups"][op["group"]]:
                        orig_val = src_acts[node["layer"], node["position"], node["feature"]]
                        tuples.append((node["layer"], node["position"], node["feature"], factor * orig_val))

                dallas_prompt = manifest["prompts"]["dallas"]
                if self.prompt.value != dallas_prompt:
                    self.prompt.set(dallas_prompt)
                self.checkpoint_prompt = dallas_prompt
                self._last_prompt = dallas_prompt
                if self.traced_prompt != dallas_prompt:
                    self._in_refresh = True
                    try:
                        response = self.service._retrace(
                            {"prompt": dallas_prompt, "show_more": bool(self.show_more.value)},
                            self.controls.n,
                        )
                        if response is not None and response[0] == "error":
                            raise ValueError(response[2]["message"])
                    finally:
                        self._in_refresh = False

                t0 = time.perf_counter()
                self._report_progress(
                    state="running",
                    stage="intervention",
                    label=f"Applying intervention ({case['label']})",
                    panels=self._active_panel_set(("output",)),
                )
                iv_logits, iv_acts = self.backend.intervene_logits(
                    dallas_prompt,
                    tuples,
                    freeze_attention=True,
                    return_activations=(cell == 29),
                    guard=self.guard,
                )
                elapsed = time.perf_counter() - t0

                with autorelease_scope(self.backend.device):
                    base_last = cache["dallas_logits"].squeeze(0)[-1]
                    iv_last = iv_logits.squeeze(0)[-1]
                    base_probs = base_last.softmax(-1)
                    iv_probs = iv_last.softmax(-1)
                    base_p, base_i = base_probs.topk(1)
                    obs_p, obs_i = iv_probs.topk(1)
                    base_id = int(base_i[0].item())
                    obs_id = int(obs_i[0].item())
                    base_top = {
                        "id": base_id,
                        "text": _decode_raw_token(self.tokenizer, base_id),
                        "probability": float(base_p[0].item()),
                        "logit": float(base_last[base_id].item()),
                    }
                    steered_top = {
                        "id": obs_id,
                        "text": _decode_raw_token(self.tokenizer, obs_id),
                        "probability": float(obs_p[0].item()),
                        "logit": float(iv_last[obs_id].item()),
                    }
                    say_vic = float(iv_acts[21, 10, 2236].item()) if cell == 29 and iv_acts is not None else None
                    paired = torch.stack((base_last, iv_last), dim=0)
                del iv_logits, iv_acts

                with _publication_scope(self.model):
                    tensors, comp = self.output.prepare_logits(paired)
                    _publish_many(tensors)

                self._preset_tuples = tuples
                self._preset_cell = cell
                state = self._input_state()
                self._generate_for_state(state, force=True)
                with self._inference_lock:
                    self._comparison = {
                        **self._input_hashes(state),
                        "argmax_token_ids": [base_id, obs_id],
                        "baseline_top_text": base_top["text"],
                        "baseline_top_probability": base_top["probability"],
                        "baseline_top": base_top,
                        "steered_top": steered_top,
                        "cell": cell,
                        "label": case["label"],
                        "say_victoria_activation": say_vic,
                        "dallas_cache_seconds": cache["cache_seconds"],
                        "completed_step": self.controls.n,
                        "compute_publish_seconds": elapsed,
                    }
                self._output_key = deepcopy(state)
                summary = (
                    f"Cell {cell} ({case['label']}): {base_top['text']!r} ({base_top['probability']:.4f}) -> "
                    f"{steered_top['text']!r} ({steered_top['probability']:.4f})"
                )
                self._set_readonly_param(self.clamp_status, summary)
                self._set_progress("ready")
                return summary
            except CleanStop:
                raise
            except Exception as exc:
                self._set_progress("error", str(exc))
                raise
