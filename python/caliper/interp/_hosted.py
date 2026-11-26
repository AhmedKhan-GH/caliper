"""Hosted inference run loop, controls, and checkpoints over Backend."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from .._controls import RUN_PAUSING, RUN_STEPPING, intervention_key
from .._intervention_protocol import internal_forward, specification
from ._graph import LARGE_NODE_CAP
from ._hosted_controls import _HostedControlsMixin
from ._hosted_presets import _HostedPresetsMixin
from ._hosted_progress import _HostedProgressMixin
from ._hosted_resources import (
    ALL_PANELS,
    CHECKPOINT_BYTES,
    CHECKPOINT_FORMAT,
    CHECKPOINT_VALUES,
    DEFAULT_PROMPT,
    READ_ONLY_PARAMS,
    _FixtureDisplay,
    _accepts_kwarg,
    checkpoint_metadata,
    fixture_resources,
    load_resources,
)
from ._hosted_service import HostedService
from ._memory import CleanStop, MemoryGuard
from ._output import Output, bounded_text
from ._prompt import prompt_inputs
from ._service import Refused, register_service, unregister_service


class HostedInference(_HostedProgressMixin, _HostedControlsMixin, _HostedPresetsMixin):
    """Own one ``Backend``, its live controls, and ``.json`` checkpoint metadata."""

    READ_ONLY_PARAMS = READ_ONLY_PARAMS

    def __init__(self, model_name: str, dictionary_path: str | Path, prompt: str = DEFAULT_PROMPT, *,
                 resources=None, defer_load: bool = False, progress=None):
        self.model_name = str(model_name)
        dict_p = Path(str(dictionary_path))
        self.dictionary_path = str(dict_p.resolve()) if dict_p.exists() else str(dictionary_path)
        self.initial_prompt = prompt
        self.checkpoint_prompt = prompt
        self._last_prompt = prompt
        self.restored = None
        self.handle = self.service = self.managed = None
        self._failure = None
        self._stop_checkpointed = False
        self._inference_lock = threading.Lock()
        self._inference_ready = False
        self._progress_state = "starting"
        self._progress_started_at: float | None = time.time()
        self._panels_ready: set[str] = set()
        self._active_panels: tuple[str, ...] = ALL_PANELS
        self._progress = self._build_progress(
            state="starting",
            stage="starting",
            label="Starting inference",
            panels=ALL_PANELS,
        )
        self._comparison = None
        self._restore_generation(0)
        self.selection = None
        self._owned_clamp = None
        self._clamp_selection = None
        self._selection_graph = None
        self.traced_prompt = None
        self.traced_specifications = None
        self.traced_state = None
        self.traced_show_more = None
        self._output_key = None
        self._clamp_version = None
        self._selected_version = None
        self._retrace = False
        self._in_refresh = False
        self._failed_trace_key = None
        self._readonly_values: dict[str, str] = {}
        self._readonly_versions: dict[str, int] = {}
        self._preset_tuples = None
        self._preset_cell = None
        self._dallas_cache = None
        self.parameters: list = []
        self.actions: list = []
        self.backend = self.index = self.guard = None
        self.model = self.tokenizer = self.dictionary = None
        self.fixture_display = None
        self._early_programs_enabled = False

        if resources is not None:
            self._bind_loaded_resources(*resources)
        elif not defer_load:
            self._bind_loaded_resources(*load_resources(self.model_name, self.dictionary_path, progress=progress))

    def _bind_loaded_resources(self, backend, index, guard) -> None:
        self.backend, self.index, self.guard = backend, index, guard
        self.model = self.backend.model
        self.tokenizer = self.backend.tokenizer
        self.dictionary = self.backend.dictionary
        self.backend.progress_callback = self._on_backend_progress
        mps_display = os.environ.get("CALIPER_FIXTURE_DISPLAY_MPS", "0") == "1" and self.model_name == "tiny-lm"
        self.fixture_display = _FixtureDisplay(MemoryGuard(12, "mps")) if mps_display else None
        prompt_inputs(self.backend, self.tokenizer, self.initial_prompt)
        self.backend.record_prompt(self.initial_prompt)

    def _latch_failure(self, failure) -> None:
        if self._failure is None:
            self._failure = failure
        if self.service is not None:
            with self.service.lock:
                if self.service.failure is None:
                    self.service.failure = self._failure

    def _bindings(self) -> dict:
        return {
            "model": self.model_name,
            "model_id": self.backend.model_id,
            "dictionary": self.dictionary_path,
            "dictionary_id": self.dictionary.metadata["dictionary_id"],
            "index_hash": self.index.metadata["index_hash"],
        }

    def load_checkpoint(self, record: dict, *, params: dict | None = None) -> None:
        if (not checkpoint_metadata(record) or record.get("format") != CHECKPOINT_FORMAT
                or record.get("version") != 1 or record.get("bindings") != self._bindings()):
            raise ValueError("inference checkpoint resource bindings do not match")
        if type(record.get("step")) is not int or not 0 <= record["step"] < 2**53:
            raise ValueError("inference checkpoint step is invalid")
        if type(record.get("params")) is not dict or type(record.get("interventions")) is not dict:
            raise ValueError("inference checkpoint state is invalid")
        prompt = (params or {}).get("prompt", record["params"].get("prompt"))
        prompt_inputs(self.backend, self.tokenizer, prompt)
        selection = record.get("selection")
        if selection is not None:
            if type(selection) is not dict or set(selection) != {"layer", "feature", "position"}:
                raise ValueError("inference checkpoint selection is invalid")
            limits = {
                "layer": self.dictionary.metadata["layers"],
                "feature": self.dictionary.metadata["latents"],
                "position": 32,
            }
            if any(type(selection[k]) is not int or not 0 <= selection[k] < lim for k, lim in limits.items()):
                raise ValueError("inference checkpoint selection is out of range")
        owned = record.get("owned_clamp")
        if owned is not None and (not specification(owned) or owned.get("op", owned.get("kind")) != "clamp"):
            raise ValueError("inference checkpoint clamp ownership is invalid")
        self.restored = deepcopy(record)
        self.restored["params"]["prompt"] = prompt

    def state_dict(self) -> dict:
        bindings = self._bindings()
        with self.controls.lock, self._inference_lock:
            owned = self._owned_record()
            return {
                "bindings": bindings,
                "interventions": self.interventions._state_dict(),
                "selection": deepcopy(self.selection),
                "owned_clamp": deepcopy(owned.spec) if owned is not None else None,
                "generation_id": self._generation["id"],
            }

    def _restore_generation(self, generation_id: int) -> None:
        with self._inference_lock:
            self._generation = {
                "id": generation_id,
                "prompt_hash": None,
                "interventions_hash": None,
                "token_ids": [[], []],
                "counts": [0, 0],
                "completed_step": None,
                "text_hashes": [None, None],
            }

    def _input_state(self) -> tuple[str, list[dict]]:
        with self.controls.lock:
            specs = [
                deepcopy(iv.applied_spec)
                for iv in self.controls._ivs.values()
                if iv.state == "active" and iv.hook is not None and iv.applied_spec is not None
            ]
            return self.prompt.value, specs

    @staticmethod
    def _input_hashes(state: tuple[str, list[dict]]) -> dict:
        prompt, specs = state
        encoded = json.dumps(specs, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
        return {
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "interventions_hash": hashlib.sha256(encoded).hexdigest(),
        }

    def inference_status(self) -> dict:
        with self._inference_lock:
            if not self._inference_ready:
                return {
                    "state": self._progress_state,
                    "progress": deepcopy(self._progress),
                    "prompt_hash": None,
                    "interventions_hash": None,
                    "traced_prompt": None,
                    "traced_show_more": None,
                    "selected_node": -1,
                    "comparison": None,
                    "generation": deepcopy(self._generation),
                }
        state = self._input_state()
        hashes = self._input_hashes(state)
        selected_node = int(self.selected.value) if hasattr(self, "selected") else -1
        with self._inference_lock:
            return {
                **hashes,
                "state": self._progress_state,
                "progress": deepcopy(self._progress),
                "traced_prompt": self.traced_prompt,
                "traced_show_more": self.traced_show_more,
                "selected_node": selected_node,
                "comparison": deepcopy(self._comparison),
                "generation": deepcopy(self._generation),
            }

    def start(self):
        import caliper
        from .. import _managed

        self.controls = caliper._CTL
        self.managed = _managed.current()
        if self.managed is not None:
            self.managed.inference = self
            self.managed.kept.extension = ".json"

        def add_param(name, value, **kwargs):
            if name in self.READ_ONLY_PARAMS:
                if _accepts_kwarg(caliper.param, "read_only"):
                    kwargs.setdefault("read_only", True)
                elif _accepts_kwarg(caliper.param, "readonly"):
                    kwargs.setdefault("readonly", True)
            p = caliper.param(name, value, **kwargs)
            self.parameters.append(p)
            if name in self.READ_ONLY_PARAMS:
                self._readonly_values[name] = value if name == "output.progress" else bounded_text(str(value))
                self._readonly_versions[name] = p._version
            return p

        try:
            self.status = add_param("output.status", "Loading model weights" if self.backend is None else "ready")
            self.progress = add_param("output.progress", self._progress_json())
            # Register the interpretation service and enable watch capabilities before
            # serving so the initial HELLO/WELCOME announces interpret, programs,
            # resources, and interventions even while model weights are loading.
            self.service = register_service(HostedService(self), controls=self.controls)
            if self.backend is None:
                from .. import _net_programs

                _net_programs.enable()
                self._early_programs_enabled = True
                if self.managed is not None:
                    self.managed.write_link(caliper.serve().url)
                self._progress_started_at = time.time()
                self._report_progress(
                    state="loading",
                    stage="weights",
                    label="Loading model weights (stage 1/3)",
                    panels=ALL_PANELS,
                )
                try:
                    loaded = load_resources(
                        self.model_name,
                        self.dictionary_path,
                        progress=self._on_load_progress,
                    )
                except Exception as exc:
                    self._set_progress("error", f"Model load failed: {exc}")
                    raise
                self._bind_loaded_resources(*loaded)
                self.service.attach_loaded_resources()
            else:
                self.backend.progress_callback = self._on_backend_progress

            self.handle = caliper.watch(self.model, activations=False)
            if self._early_programs_enabled:
                from .. import _net_programs

                _net_programs.disable()
                self._early_programs_enabled = False
            self.interventions = self.controls.intervention_service
            self.interventions.configure(
                dictionary=self.dictionary,
                tokenizer=self.tokenizer,
                resource_guard=self.guard,
                backend=self.backend,
            )
            self.output = Output(self.backend, self.interventions, guard=self.guard)

            self.prompt = add_param("prompt", self.initial_prompt)
            prompt_inputs(self.backend, self.tokenizer, self.prompt.value)
            self.checkpoint_prompt = self.prompt.value
            self._last_prompt = self.prompt.value
            self.selected = add_param("selected_node", -1, min=-1, max=LARGE_NODE_CAP - 1)
            self.clamp = add_param("clamp", 0.0, min=-4.0, max=4.0)
            self.clamp_status = add_param("clamp.status", "No feature selected. No slider clamp.")
            self.retrace = add_param("retrace", False)
            self.show_more = add_param("show_more", False)
            self.baseline_text = add_param("output.baseline", "")
            self.steered_text = add_param("output.steered", "")
            self.actions = [
                caliper.action("run", self.run_prompt),
                caliper.action("generate", self.generate),
                caliper.action("reset", self.reset),
                caliper.action("apply_clamp", self.apply_clamp),
            ]
            self._register_dallas_actions(caliper)
            if self.restored is not None:
                self.interventions.load_state_dict(self.restored["interventions"], restore_active=True)
                self.selection = deepcopy(self.restored.get("selection"))
                owned = self.restored.get("owned_clamp")
                if owned is not None:
                    with self.controls.lock:
                        iv = self.controls._ivs.get(intervention_key(owned))
                        if iv is not None and iv.spec == owned and not iv.clear and iv.state != "ended":
                            self._owned_clamp = iv.spec
                self._restore_generation(self.restored.get("generation_id", 0))
            if self.managed is not None and self.managed.interventions:
                self.managed._intervene()
            self._clamp_version = self.clamp._version
            self._selected_version = self.selected._version
            with self._inference_lock:
                self._inference_ready = True
                self._progress_state = "ready"
            self.boundary()
            return self
        except BaseException:
            self.close()
            raise

    def _compute_output(self, prompt: str, state: tuple[str, list[dict]]) -> dict:
        started = time.perf_counter()
        comparison = self.output.compare(prompt)
        seconds = time.perf_counter() - started
        self._panels_ready.add("output")
        with self._inference_lock:
            self._comparison = {
                **self._input_hashes(state),
                "argmax_token_ids": comparison["argmax_token_ids"],
                "baseline_top_text": comparison["baseline_top_text"],
                "baseline_top_probability": comparison["baseline_top_probability"],
                "completed_step": self.controls.n,
                "compute_publish_seconds": seconds,
            }
        return comparison

    def _generate_for_state(self, state: tuple[str, list[dict]], *, force: bool = False) -> dict:
        prompt, _specs = state
        hashes = self._input_hashes(state)
        with self._inference_lock:
            if (
                not force
                and self._preset_tuples is None
                and self._generation["prompt_hash"] == hashes["prompt_hash"]
                and self._generation["interventions_hash"] == hashes["interventions_hash"]
                and self.baseline_text.value != ""
            ):
                return {
                    "baseline": self.baseline_text.value,
                    "steered": self.steered_text.value,
                    "token_ids": deepcopy(self._generation["token_ids"]),
                }
            gen_id = self._generation["id"] + 1
        t0 = time.perf_counter()
        with internal_forward():
            dallas_prompt = (
                self._dallas_cache["manifest"]["prompts"]["dallas"]
                if self._dallas_cache is not None
                else None
            )
            if self._preset_tuples is not None and prompt == dallas_prompt:
                with self.interventions.suspend():
                    raw = self.backend.generate(prompt, self._preset_tuples, max_new_tokens=24, guard=self.guard)
                row0, row1 = raw["token_ids"]
                result = {
                    "baseline": bounded_text(raw["baseline"] if "baseline" in raw else self.tokenizer.decode(row0)),
                    "steered": bounded_text(raw["steered"] if "steered" in raw else self.tokenizer.decode(row1)),
                    "token_ids": [row0, row1],
                }
            else:
                result = self.output.generate(prompt)
        gen_seconds = time.perf_counter() - t0
        ids = result["token_ids"]
        texts = [result["baseline"], result["steered"]]
        self._set_readonly_param(self.baseline_text, texts[0])
        self._set_readonly_param(self.steered_text, texts[1])
        completion = {
            **self._input_hashes(state),
            "id": gen_id,
            "token_ids": deepcopy(ids),
            "counts": [len(ids[0]), len(ids[1])],
            "completed_step": self.controls.n,
            "seconds": gen_seconds,
            "text_hashes": [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts],
        }
        with self._inference_lock:
            self._generation = completion
        return result

    def _refresh(
        self,
        *,
        prompt: str,
        specs: list[dict],
        show_more: bool,
        do_trace: bool,
        do_output: bool,
        do_generate: bool,
        force_generate: bool = False,
        raise_on_trace_error: bool = False,
    ) -> bool:
        key = (prompt, specs)
        self._in_refresh = True
        try:
            if do_trace:
                self._set_progress("tracing")
                response = self.service._retrace({"prompt": prompt, "show_more": show_more}, self.controls.n)
                if response is None:
                    return False
                if response[0] == "error":
                    msg = response[2]["message"]
                    self._failed_trace_key = (prompt, deepcopy(specs), show_more)
                    self._set_progress("error", msg)
                    if raise_on_trace_error:
                        raise ValueError(msg)
                    return False
                self._failed_trace_key = None
            if do_output:
                self._compute_output(prompt, key)
                self._output_key = deepcopy(key)
            if do_generate:
                self._generate_for_state(key, force=force_generate)
            if self.fixture_display is not None:
                self.fixture_display.publish()
            if self._preset_cell is None:
                self._sync_clamp_status()
            self._set_progress("ready")
            return True
        except CleanStop:
            raise
        except Exception as exc:
            self._set_progress("error", str(exc))
            raise
        finally:
            self._in_refresh = False

    def update_output(self, *, force: bool = False) -> None:
        key = self._input_state()
        prompt, specs = key
        prompt_inputs(self.backend, self.tokenizer, prompt)
        show_more = bool(self.show_more.value)
        initial = self._output_key is None
        prev_prompt = self._output_key[0] if self._output_key is not None else None
        changed = self._output_key != key
        prompt_changed = (not initial) and (prompt != prev_prompt)
        show_more_changed = (self.traced_show_more is not None) and (show_more != self.traced_show_more)
        if changed:
            self._preset_tuples = None
            self._preset_cell = None

        if not force and self._failed_trace_key == (prompt, specs, show_more):
            if self.guard is not None:
                self.guard.check("inference boundary", force=False)
            return

        need_trace = False
        if initial and self.restored is None:
            need_trace = True
        elif prompt_changed and (self.traced_prompt != prompt or self.traced_show_more != show_more):
            need_trace = True
        elif show_more_changed:
            need_trace = True
        elif self.retrace.value and (changed or not self._retrace):
            if (self.traced_prompt, self.traced_specifications, self.traced_show_more) != (prompt, specs, show_more):
                need_trace = True

        need_output = changed or force
        need_generate = changed or force or (self._generation["prompt_hash"] is None)
        self._retrace = self.retrace.value

        if need_trace or need_output or need_generate:
            self._refresh(
                prompt=prompt,
                specs=specs,
                show_more=show_more,
                do_trace=need_trace,
                do_output=need_output,
                do_generate=need_generate,
            )
        else:
            if self._preset_cell is None:
                self._sync_clamp_status()
            if self._progress_state == "error":
                self._set_progress("ready")
        if self.guard is not None:
            self.guard.check("inference boundary", force=False)

    def run_prompt(self) -> str:
        """Recompute graph trace, output bars, generated text, and examples for the current prompt and clamp."""
        self._check_stop()
        with internal_forward():
            self._preset_tuples = None
            self._preset_cell = None
            self.update_controls()
            key = self._input_state()
            prompt, specs = key
            prompt_inputs(self.backend, self.tokenizer, prompt)
            show_more = bool(self.show_more.value)
            ok = self._refresh(
                prompt=prompt,
                specs=specs,
                show_more=show_more,
                do_trace=True,
                do_output=True,
                do_generate=True,
                force_generate=True,
                raise_on_trace_error=True,
            )
            if not ok:
                return "A trace is already pending."
        return "Traced prompt, updated outputs, regenerated text, and refreshed examples."

    def generate(self) -> str:
        self._check_stop()
        self._progress_started_at = time.time()
        try:
            self._generate_for_state(self._input_state(), force=True)
            self._set_progress("ready")
        except CleanStop:
            raise
        except Exception as exc:
            self._set_progress("error", str(exc))
            raise
        return "Generated 24 tokens for each output."

    def reset(self) -> str:
        self._check_stop()
        with internal_forward():
            self._preset_tuples = None
            self._preset_cell = None
            with self.controls.lock:
                self._discard_owned_clamp()
            self._apply_interventions()
            self.clamp.set(0.0)
            self._clamp_version = self.clamp._version
            try:
                try:
                    prompt_inputs(self.backend, self.tokenizer, self.prompt.value)
                except ValueError:
                    self._set_progress("error", "The clamp is reset. Correct the prompt to update the output.")
                else:
                    self.update_output(force=True)
            finally:
                self._sync_clamp_status()
        return "The feature clamp is reset."

    def sweep(self, *, clean_prompt: str, corrupted_prompt: str, metric: dict) -> dict:
        from ._sweep import patch_sweep
        result = patch_sweep(
            self.backend, self.interventions,
            clean_prompt=clean_prompt, corrupted_prompt=corrupted_prompt,
            metric=metric, guard=self.guard,
        )
        if self.fixture_display is not None:
            self.fixture_display.publish()
        return result

    def validate(self) -> dict:
        from ._validate import validate_trace
        if self.service.graph is None:
            raise Refused("interpret.missing_trace", "No completed trace is available.")
        with internal_forward(), self.interventions.temporary_state(self.traced_state or []) as replay:
            return validate_trace(
                self.backend, replay, self.service.graph,
                prompt=self.traced_prompt, guard=self.guard,
            )

    def _check_stop(self) -> None:
        if self._failure is not None:
            raise self._failure

    @contextmanager
    def _stop_boundary(self):
        try:
            self._check_stop()
            yield
            self._check_stop()
        except CleanStop as exc:
            if self.service is not None:
                self.service.stop(exc, self.controls.n)
            if self._failure is None:
                self._failure = exc
            if self.managed is not None and not self._stop_checkpointed:
                self._stop_checkpointed = True
                try:
                    self.managed.checkpoint(self.model, None)
                except Exception as checkpoint_error:
                    raise self._failure from checkpoint_error
            raise self._failure

    def boundary(self) -> None:
        import caliper
        with self._stop_boundary():
            self.controls.busy = True
            caliper.step()
            if self.guard is not None:
                self.guard.check("inference actions", force=False)
            if self.managed is not None:
                if self.managed.stopping:
                    self.managed.exit()
                self.managed._keep(self.model, None)

    def run(self) -> None:
        from .. import _programs

        while True:
            def pending():
                due = any(
                    iv.clear or iv.replace or (
                        iv.state == "pending" and (iv.from_step is None or iv.from_step <= self.controls.n + 1)
                    )
                    for iv in self.controls._ivs.values()
                )
                return bool(
                    self.controls._npending or self.controls.queue or self.service.pending or due
                    or self.controls.state in (RUN_PAUSING, RUN_STEPPING)
                )

            blocked = _programs.inference_blocked()
            publication = _programs.inference_pending() and not blocked
            with self.controls.cond:
                if not publication:
                    self.controls.cond.wait_for(pending, timeout=0.05 if blocked else 1.0)
                work = pending()
            if not publication:
                blocked = _programs.inference_blocked()
                publication = _programs.inference_pending() and not blocked
            try:
                with self._stop_boundary():
                    if work or publication:
                        self.boundary()
                    else:
                        self.controls.checkpoint()
            except CleanStop:
                raise
            except Exception as exc:
                self._set_progress("error", f"{type(exc).__name__}: {exc}")

    def close(self) -> None:
        if self.service is not None:
            unregister_service(self.service, controls=getattr(self, "controls", None))
            self.service = None
        for item in self.actions + self.parameters:
            item.remove()
        self.actions.clear()
        self.parameters.clear()
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if self._early_programs_enabled:
            from .. import _net_programs

            _net_programs.disable()
            self._early_programs_enabled = False
        if self.index is not None:
            self.index.close()
        if self.managed is not None and self.managed.inference is self:
            self.managed.inference = None


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Local Inference Run")
    parser.add_argument("--model")
    parser.add_argument("--dictionary")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--fixture-dir")
    args = parser.parse_args(argv)
    if args.fixture_dir:
        if any((args.model, args.dictionary, args.prompt)):
            parser.error("fixture creation takes no run arguments")
        fixture_resources(args.fixture_dir)
        return
    if not all((args.model, args.dictionary)):
        parser.error("model and dictionary are required")
    prompt = args.prompt if args.prompt is not None else DEFAULT_PROMPT
    run = HostedInference(args.model, args.dictionary, prompt, defer_load=True)
    try:
        run.start()
        print(f"caliper: inference ready at step {run.controls.n}", flush=True)
        run.run()
    finally:
        run.close()


if __name__ == "__main__":
    from caliper.interp._hosted import main as hosted_main
    hosted_main()
