"""Hosted inference run loop, controls, and checkpoints over Backend."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from inspect import Parameter, signature
import json
import math
import os
from pathlib import Path
import threading
import time

import torch

from .._controls import RUN_PAUSING, RUN_STEPPING, intervention_key
from .._intervention_protocol import internal_forward, specification
from ._backend import Backend, load_backend
from ._graph import LARGE_NODE_CAP, Tracer
from ._index import FeatureIndex, LiveFeatureIndex, build_index
from ._memory import CleanStop, MAX_PROCESS_GB, MemoryGuard, autorelease_scope
from ._output import Output, bounded_text
from ._prompt import prompt_inputs
from ._service import InterpretationService, Refused, register_service, unregister_service

CHECKPOINT_FORMAT = "caliper.inference"
CHECKPOINT_BYTES = 4 * 1024 * 1024
CHECKPOINT_VALUES = 262144
DEFAULT_PROMPT = "He saw a carrot and had to grab it, his hunger was like a starving"
READ_ONLY_PARAMS = frozenset({
    "output.baseline",
    "output.steered",
    "output.status",
    "output.progress",
    "clamp.status",
})
ALL_PANELS = ("graph", "output", "examples")


def _accepts_kwarg(fn, name: str) -> bool:
    parameters = signature(fn).parameters
    return any(p.kind == Parameter.VAR_KEYWORD for p in parameters.values()) or (
        name in parameters
        and parameters[name].kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    )


def checkpoint_metadata(value) -> bool:
    """Validate bounded JSON checkpoint metadata."""
    count = 0

    def valid(item, depth=0):
        nonlocal count
        count += 1
        if depth > 8 or count > CHECKPOINT_VALUES:
            return False
        if item is None or type(item) is bool:
            return True
        if type(item) is int:
            return abs(item) < 2**53
        if type(item) is float:
            return math.isfinite(item)
        if type(item) is str:
            return len(item.encode("utf-8")) <= 4096
        if type(item) is list:
            return len(item) <= 16384 and all(valid(v, depth + 1) for v in item)
        if type(item) is dict:
            return len(item) <= 64 and all(
                type(k) is str and len(k.encode("utf-8")) <= 63 and valid(v, depth + 1)
                for k, v in item.items()
            )
        return False

    try:
        if type(value) is dict and valid(value):
            raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            return len(raw) <= CHECKPOINT_BYTES
    except (ValueError, UnicodeError):
        pass
    return False


class _FixtureDisplay:
    """Copy bounded CPU fixture display tensors onto MPS when ``CALIPER_FIXTURE_DISPLAY_MPS=1``."""

    TENSORS = (
        "trace.counts", "trace.node_i", "trace.node_f", "trace.edge_i", "trace.edge_f",
        "trace.label_text", "trace.probs", "trace.logit_tokens", "trace.logit_label_text",
        "examples.tokens", "examples.activations", "examples.example_ids",
        "examples.valid_tokens", "examples.label_text",
        "output.probs", "output.tokens", "output.label_text",
        "patch.grid",
    )

    def __init__(self, guard: MemoryGuard, *, device="mps"):
        self.guard = guard
        self.device = torch.device(device)
        if self.device.type == "mps" and self.device.index is None:
            self.device = torch.device("mps:0")

    def publish(self) -> int:
        from .. import _programs
        from . import _publish

        with _publish._publication_scope():
            with _programs._LOCK:
                values = {n: _publish._TENSORS[n] for n in self.TENSORS if n in _publish._TENSORS}
            pending = [(n, t) for n, t in values.items() if t.device.type == "cpu"]
            if not pending:
                return 0
            with autorelease_scope(self.device):
                self.guard.check("fixture display copy start")
                converted = {n: t.detach().contiguous().to(device=self.device, copy=True) for n, t in pending}
                self.guard.check("fixture display copy complete")
            _publish._publish_many(converted)
            return sum(t.numel() * t.element_size() for _, t in pending)


def fixture_resources(directory: str | Path) -> Path:
    """Materialize deterministic ``tiny-lm`` transcoders and an example index on disk."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    backend = load_backend("tiny-lm")
    backend.dictionary.raw.to_safetensors(str(root))
    if not (root / "index").exists():
        docs = [
            ("doc-0", backend.tokenize("ABCD")[0].tolist()),
            ("doc-1", backend.tokenize("EFGH")[0].tolist()),
            ("doc-2", backend.tokenize("ABEF")[0].tolist()),
        ]
        build_index(
            root,
            docs,
            model=backend.model,
            tokenizer=backend.tokenizer,
            dictionary=backend.dictionary,
            provenance={"partition": "tiny-lm-fixture"},
            context_length=16,
        )
    return root


def load_resources(model_name: str, dictionary_path: str | Path, *, device=None, memory_limit_gb: float = MAX_PROCESS_GB, progress=None):
    """Load ``(backend, index, guard)`` for hosted inference."""
    is_tiny = str(model_name) == "tiny-lm"
    if device is None:
        dev = torch.device("cpu") if is_tiny else None
    else:
        dev = torch.device(device)
    guard = None
    if not is_tiny:
        from ._backend import default_device
        actual_dev = dev if dev is not None else default_device()
        guard = MemoryGuard(memory_limit_gb, actual_dev)
        guard.check("hosted load start")

    backend = load_backend(model_name, dictionary_path, device=dev, guard=guard, progress=progress)
    if callable(progress):
        progress(stage="index", label="Loading example index (stage 3/3)", done=None, total=None)
    env_index = os.environ.get("CALIPER_FEATURE_INDEX", "").strip()
    resolved_index = Path(env_index).expanduser() if env_index else FeatureIndex.resolve(dictionary_path)
    if resolved_index is not None:
        index = FeatureIndex(
            resolved_index,
            model=backend.model,
            tokenizer=backend.tokenizer,
            dictionary=backend.dictionary,
            memory=guard,
        )
    else:
        index = LiveFeatureIndex(backend, memory=guard, allow_online=os.environ.get("CALIPER_AGENT_OFFLINE") != "1")
    if callable(progress):
        progress(stage="index", label="Model resources ready (3/3)", done=3, total=3)
    return backend, index, guard


class HostedService(InterpretationService):
    """Run step-boundary controls, interpretation requests, and output comparisons together."""

    def __init__(self, owner: "HostedInference"):
        self.owner = owner
        self.boundary_running = False
        self.boundary_thread = None
        tracer = (
            Tracer(
                owner.model,
                owner.tokenizer,
                owner.dictionary,
                backend=owner.backend,
                index=owner.index,
                guard=owner.guard,
            )
            if owner.backend is not None
            else None
        )
        super().__init__(
            model=owner.model,
            tokenizer=owner.tokenizer,
            dictionary=owner.dictionary,
            index=owner.index,
            tracer=tracer,
            handlers={
                "patch.sweep": owner.sweep,
                "trace.validate": owner.validate,
            },
            memory=owner.guard,
        )
        self.owner = owner
        self.progress_callback = owner._on_service_progress

    def attach_loaded_resources(self) -> None:
        with self.lock:
            self.model = self.owner.model
            self.tokenizer = self.owner.tokenizer
            self.dictionary = self.owner.dictionary
            self.index = self.owner.index
            self.memory = self.owner.guard
            self.tracer = Tracer(
                self.owner.model,
                self.owner.tokenizer,
                self.owner.dictionary,
                backend=self.owner.backend,
                index=self.owner.index,
                guard=self.owner.guard,
                controls=getattr(self, "controls", None),
            )
            self._bind_resources()

    def _bind_controls(self, controls):
        if self.tracer is not None:
            self.tracer.controls = controls
        super()._bind_controls(controls)

    def _check_intervention_budget(self, ivs):
        for iv in ivs.values():
            if not checkpoint_metadata({"spec": iv.spec}):
                raise ValueError("intervention exceeds the checkpoint metadata limits")

    def submit(self, metadata, reply, *, acknowledge=None):
        from .. import _interpret_protocol as protocol

        if protocol.request(metadata) and metadata["method"] != "trace.status":
            with self.owner._inference_lock:
                ready = self.owner._inference_ready
            if not ready:
                raise Refused("interpret.busy", "Model resources are still loading.")
        status, result = super().submit(metadata, reply, acknowledge=acknowledge)
        if metadata["method"] == "trace.status":
            inference = self.owner.inference_status()
            result["inference"] = inference
            result["progress"] = deepcopy(inference.get("progress"))
        return status, result

    def stop(self, failure, step=None):
        error = super().stop(failure, step)
        self.owner._latch_failure(self.failure)
        self.owner.controls._terminate_actions(self.failure)
        return error

    def _run_request(self, method, params, step):
        status, result, error = super()._run_request(method, params, step)
        if method == "trace.run" and status == "error" and not self.owner._in_refresh:
            self.owner._set_progress("error", error["message"])
        return status, result, error

    def _execute(self, method, params, step, *, complete=None):
        if method != "trace.run":
            return super()._execute(method, params, step, complete=complete)

        prompt_param = self.owner.prompt
        show_more_param = self.owner.show_more
        has_show_more = "show_more" in params
        show_more = bool(params["show_more"]) if has_show_more else bool(show_more_param.value)
        with self.owner.controls.lock:
            prompt_version = prompt_param._version
            show_more_version = show_more_param._version
            entry = prompt_param.entry()
            entry["value"] = params["prompt"]
            self.owner.controls._check_param_budget(prompt_param, entry)
        prompt_inputs(self.owner.backend, self.tokenizer, params["prompt"])

        if not self.owner._in_refresh:
            self.owner._set_progress("tracing")
            if params["prompt"] != self.owner._last_prompt:
                self.owner._reconcile_interventions_for_prompt(params["prompt"])

        traced_specs = self.owner.interventions.specifications()
        exec_params = dict(params)
        exec_params["show_more"] = show_more

        def on_complete(_result):
            publish_controls = False
            with self.owner.controls.lock:
                if (self.owner.controls.params.get(prompt_param.name) is prompt_param
                        and prompt_param._version == prompt_version and prompt_param.pending is None):
                    prompt_param._value = params["prompt"]
                    prompt_param._version += 1
                    self.owner.checkpoint_prompt = params["prompt"]
                    self.owner.controls._update_busy()
                    publish_controls = True
                if (has_show_more
                        and self.owner.controls.params.get(show_more_param.name) is show_more_param
                        and show_more_param._version == show_more_version and show_more_param.pending is None):
                    show_more_param._value = show_more
                    show_more_param._version += 1
                    self.owner.controls._update_busy()
                    publish_controls = True
            self.owner.traced_prompt = params["prompt"]
            self.owner.traced_specifications = traced_specs
            self.owner.traced_state = deepcopy(traced_specs)
            self.owner.traced_show_more = show_more
            self.owner._failed_trace_key = None
            self.owner._panels_ready.add("graph")
            self.owner._reconcile_selection(self.graph, auto_select=True)
            self.owner._refresh_examples()
            if not self.owner._in_refresh:
                if params["prompt"] != self.owner._last_prompt:
                    self.owner._preset_tuples = None
                    self.owner._preset_cell = None
                if self.owner.prompt.value == params["prompt"]:
                    self.owner._last_prompt = params["prompt"]
                state = (params["prompt"], deepcopy(traced_specs))
                with internal_forward():
                    self.owner._compute_output(params["prompt"], state)
                    self.owner._generate_for_state(state)
                if self.owner.prompt.value == params["prompt"]:
                    self.owner._output_key = deepcopy(self.owner._input_state())
                if self.owner._preset_cell is None:
                    self.owner._sync_clamp_status()
                if self.owner.fixture_display is not None:
                    self.owner.fixture_display.publish()
                self.owner._set_progress("ready")
            elif self.owner.fixture_display is not None:
                self.owner.fixture_display.publish()
            if publish_controls:
                self.owner.controls._publish()

        return super()._execute(method, exec_params, step, complete=on_complete)

    def step(self, step=None):
        with self.lock:
            if self.running or self.boundary_running or self.closed:
                return 0
            self.boundary_running = True
            self.boundary_thread = threading.get_ident()
        try:
            self.owner._check_stop()
            with internal_forward():
                try:
                    self.owner.update_controls()
                except ValueError as exc:
                    self.owner._set_progress("error", str(exc))
                count = super().step(step)
                if self.owner.fixture_display is not None and count:
                    self.owner.fixture_display.publish()
                try:
                    self.owner.update_output()
                except ValueError as exc:
                    self.owner._set_progress("error", str(exc))
                return count
        except self.stop_exception as exc:
            self.stop(exc, step)
            raise self.failure
        finally:
            with self.idle:
                self.boundary_running = False
                self.boundary_thread = None
                self.idle.notify_all()

    def _check_close_thread(self):
        super()._check_close_thread()
        with self.lock:
            if self.boundary_thread == threading.get_ident():
                raise Refused("interpret.busy", "An active hosted boundary cannot close its own service.")

    def close(self):
        with self.idle:
            self._check_close_thread()
            self.closed = True
            while self.boundary_running:
                self.idle.wait()
        super().close()


class HostedInference:
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

    def _build_progress(
        self,
        *,
        state: str,
        stage: str,
        label: str,
        done: int | None = None,
        total: int | None = None,
        panels: tuple[str, ...] | dict | None = None,
        error: str | None = None,
    ) -> dict:
        clean_label = bounded_text(str(label), 120)
        clean_error = bounded_text(str(error), 240) if error else None
        active = state not in ("ready", "idle")
        started_at = self._progress_started_at if active else None
        elapsed_s = round(max(0.0, time.time() - self._progress_started_at), 2) if self._progress_started_at else 0.0
        panel_map: dict[str, dict] = {}
        if isinstance(panels, dict):
            panel_map = panels
        elif panels:
            for pid in panels:
                panel_map[pid] = {
                    "state": state,
                    "stage": stage,
                    "label": clean_label,
                    "done": done,
                    "total": total,
                    "error": clean_error,
                }
        return {
            "state": state,
            "stage": stage,
            "label": clean_label,
            "done": done,
            "total": total,
            "started_at": round(started_at, 3) if started_at is not None else None,
            "elapsed_s": elapsed_s,
            "error": clean_error,
            "panels": panel_map,
        }

    def _progress_json(self, payload: dict | None = None) -> str:
        data = self._progress if payload is None else payload
        return json.dumps(data, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

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

    def _owned_record(self):
        if self._owned_clamp is None:
            return None
        iv = self.controls._ivs.get(intervention_key(self._owned_clamp))
        if iv is None or iv.clear or iv.state == "ended" or iv.spec is not self._owned_clamp:
            self._owned_clamp = None
            return None
        return iv

    def _discard_owned_clamp(self) -> None:
        iv = self._owned_record()
        if iv is None:
            return
        self._owned_clamp = None
        iv.clear = True
        self.controls._iv_revision += 1
        self.controls._iv_due = True
        self.controls.counters["interventions_in"] += 1
        self.controls._update_busy()
        self.controls.cond.notify_all()

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

    def _set_readonly_param(self, param, value: str, *, raw: bool = False) -> None:
        text = value if raw else bounded_text(value)
        self._readonly_values[param.name] = text
        if param.value != text:
            param.set(text)
        self._readonly_versions[param.name] = param._version

    def _enforce_readonly_outputs(self) -> bool:
        reverted = False
        for param in (self.baseline_text, self.steered_text, self.status, self.progress, self.clamp_status):
            expected = self._readonly_values.get(param.name)
            if expected is None:
                continue
            if param.value != expected:
                param.set(expected)
                reverted = True
            self._readonly_versions[param.name] = param._version
        if reverted:
            self.controls._publish()
        return reverted

    def _active_panel_set(self, primary: tuple[str, ...]) -> tuple[str, ...]:
        ordered = list(primary)
        for pid in ALL_PANELS:
            if pid not in self._panels_ready and pid not in ordered:
                ordered.append(pid)
        return tuple(ordered)

    def _report_progress(
        self,
        *,
        stage: str,
        label: str,
        done: int | None = None,
        total: int | None = None,
        state: str | None = None,
        panels: tuple[str, ...] | dict | None = None,
        error: str | None = None,
        status_text: str | None = None,
        baseline: str | None = None,
        steered: str | None = None,
    ) -> None:
        with self._inference_lock:
            eff_state = state if state is not None else (
                self._progress_state if self._progress_state not in ("ready", "idle") else "running"
            )
            if eff_state in ("loading", "tracing", "running", "starting") and self._progress_started_at is None:
                self._progress_started_at = time.time()
            self._progress_state = eff_state
            if isinstance(panels, tuple):
                self._active_panels = panels
            payload = self._build_progress(
                state=eff_state,
                stage=stage,
                label=label,
                done=done,
                total=total,
                panels=panels,
                error=error,
            )
            self._progress = payload
        if hasattr(self, "progress"):
            self._set_readonly_param(self.progress, self._progress_json(payload), raw=True)
        if hasattr(self, "status"):
            if status_text is not None:
                self._set_readonly_param(self.status, status_text)
            elif eff_state == "ready":
                self._set_readonly_param(self.status, "ready")
            elif eff_state == "error":
                self._set_readonly_param(self.status, error or label)
            else:
                elapsed = payload["elapsed_s"]
                self._set_readonly_param(self.status, f"{payload['label']} ({elapsed:.1f}s)")
        if baseline is not None and hasattr(self, "baseline_text"):
            self._set_readonly_param(self.baseline_text, baseline)
        if steered is not None and hasattr(self, "steered_text"):
            self._set_readonly_param(self.steered_text, steered)
        if getattr(self, "controls", None) is not None:
            self.controls._publish()

    def _on_load_progress(self, *, stage: str, label: str, done: int | None = None, total: int | None = None, **_extra) -> None:
        self._report_progress(
            state="loading",
            stage=stage,
            label=label,
            done=done,
            total=total,
            panels=ALL_PANELS,
        )

    def _on_backend_progress(
        self,
        *,
        stage: str,
        label: str,
        done: int | None = None,
        total: int | None = None,
        baseline: str | None = None,
        steered: str | None = None,
        **_extra,
    ) -> None:
        if stage in ("attribution", "pruning", "publishing"):
            panels = self._active_panel_set(("graph",))
            state = "tracing"
        elif stage in ("output", "intervention", "generation", "sweep"):
            panels = self._active_panel_set(("output",))
            state = "running" if self._progress_state == "ready" else self._progress_state
        elif stage == "examples":
            panels = ("examples",)
            state = "running" if self._progress_state == "ready" else self._progress_state
        else:
            panels = ALL_PANELS
            state = "running"
        self._report_progress(
            state=state,
            stage=stage,
            label=label,
            done=done,
            total=total,
            panels=panels,
            baseline=baseline,
            steered=steered,
        )

    def _on_service_progress(self, *, stage: str, label: str, done: int | None = None, total: int | None = None, **_extra) -> None:
        if stage == "examples":
            if self._progress_state == "ready":
                self._progress_started_at = time.time()
            self._report_progress(
                state="running" if self._progress_state == "ready" else self._progress_state,
                stage="examples",
                label=label,
                done=done,
                total=total,
                panels=("examples",),
            )
        elif stage == "examples_done":
            self._panels_ready.add("examples")
            if not self._in_refresh and self._progress_state != "tracing":
                self._report_progress(state="ready", stage="ready", label="Ready", panels={}, status_text="ready")

    def _set_progress(self, state: str, message: str | None = None) -> None:
        text = state if message is None else message
        if state in ("loading", "tracing", "running", "starting"):
            self._progress_started_at = time.time()
            stage = "attribution" if state == "tracing" else state
            label = "Tracing prompt" if state == "tracing" else (text or state)
            panels = self._active_panel_set(("graph",)) if state == "tracing" else ALL_PANELS
            self._report_progress(
                state=state,
                stage=stage,
                label=label,
                panels=panels,
                status_text=text,
            )
        elif state == "error":
            err_panels = self._active_panels or ALL_PANELS
            self._report_progress(
                state="error",
                stage="error",
                label="Error",
                panels=err_panels,
                error=text,
                status_text=text,
            )
        else:
            self._report_progress(
                state=state,
                stage="ready" if state == "ready" else state,
                label="Ready" if state == "ready" else text,
                panels={},
                status_text=text,
            )
            self._progress_started_at = None

    def _apply_interventions(self) -> None:
        with self.controls.lock:
            changed, _ready = self.controls._intervention_boundary()
            self._owned_record()
            self.controls._update_busy()
        if changed:
            self.controls._publish()

    def _reconcile_interventions_for_prompt(self, prompt: str) -> bool:
        """Keep active interventions coherent with the new prompt's token count."""
        ids = prompt_inputs(self.backend, self.tokenizer, prompt)["input_ids"]
        seq_len = int(ids.shape[1])
        changed = False
        reset_clamp_param = False

        with self.controls.lock:
            owned = self._owned_record()
            for iv in list(self.controls._ivs.values()):
                if iv.clear or iv.state == "ended":
                    continue
                spec = iv.applied_spec if iv.applied_spec is not None else iv.spec
                if not isinstance(spec, dict):
                    continue
                positions = spec.get("positions")
                if positions is None:
                    continue
                valid_positions = [p for p in positions if isinstance(p, int) and 0 <= p < seq_len]
                if len(valid_positions) == len(positions):
                    continue
                is_owned = owned is not None and iv is owned
                if not valid_positions:
                    if is_owned:
                        self._discard_owned_clamp()
                        reset_clamp_param = True
                    else:
                        iv.clear = True
                        self.controls._iv_revision += 1
                        self.controls._iv_due = True
                        self.controls.counters["interventions_in"] += 1
                        self.controls._update_busy()
                        self.controls.cond.notify_all()
                    changed = True
                else:
                    updated = deepcopy(spec)
                    updated["positions"] = valid_positions
                    try:
                        validated = self.interventions.executor.validate(updated)
                    except ValueError:
                        validated = None
                    if validated is None:
                        if is_owned:
                            self._discard_owned_clamp()
                            reset_clamp_param = True
                        else:
                            iv.clear = True
                            self.controls._iv_revision += 1
                            self.controls._iv_due = True
                            self.controls.counters["interventions_in"] += 1
                    else:
                        old_key = intervention_key(iv.spec)
                        new_key = intervention_key(validated)
                        if new_key != old_key:
                            if is_owned:
                                self._discard_owned_clamp()
                            else:
                                iv.clear = True
                                self.controls._iv_revision += 1
                                self.controls._iv_due = True
                                self.controls.counters["interventions_in"] += 1
                        refused = self.controls._intervene({"set": validated})
                        if refused is None and is_owned:
                            record = self.controls._ivs.get(new_key)
                            if record is not None:
                                self._owned_clamp = record.spec
                        elif refused is not None:
                            if is_owned:
                                self._discard_owned_clamp()
                                reset_clamp_param = True
                            else:
                                iv.clear = True
                                self.controls._iv_revision += 1
                                self.controls._iv_due = True
                                self.controls.counters["interventions_in"] += 1
                    self.controls._update_busy()
                    self.controls.cond.notify_all()
                    changed = True

        if changed:
            self._apply_interventions()
        if reset_clamp_param:
            self.clamp.set(0.0)
            self._clamp_version = self.clamp._version
        if self.selection is not None and self.selection["position"] >= seq_len:
            self.selection = None
        if hasattr(self.interventions, "invalidate_source_cache"):
            self.interventions.invalidate_source_cache()
        elif hasattr(getattr(self.interventions, "executor", None), "invalidate_cache"):
            self.interventions.executor.invalidate_cache()
        return changed

    @staticmethod
    def _pick_important_feature(graph) -> int:
        if graph is None:
            return -1
        features = [(i, n) for i, n in enumerate(graph.nodes) if n.get("kind") == "feature"]
        if not features:
            return -1

        def score(item):
            idx, node = item
            influence = node.get("influence")
            if influence is None:
                influence = node.get("score")
            if influence is None:
                influence = node.get("importance")
            inf_val = (
                abs(float(influence))
                if isinstance(influence, (int, float)) and math.isfinite(influence)
                else 0.0
            )
            act = node.get("activation", 0.0)
            act_val = abs(float(act)) if isinstance(act, (int, float)) and math.isfinite(act) else 0.0
            non_initial_layer = 1 if int(node.get("layer", 0)) > 0 else 0
            return (non_initial_layer, inf_val, act_val, -idx)

        best_idx, _ = max(features, key=score)
        return best_idx

    def _reconcile_selection(self, graph, *, auto_select: bool = True) -> None:
        row = -1
        if graph is not None and self.selection is not None:
            row = next(
                (i for i, n in enumerate(graph.nodes)
                 if n["kind"] == "feature" and all(n[k] == v for k, v in self.selection.items())),
                -1,
            )
        if row == -1 and auto_select and graph is not None:
            row = self._pick_important_feature(graph)
        if row == -1:
            self.selection = None
        else:
            record = graph.nodes[row]
            self.selection = {k: record[k] for k in ("layer", "feature", "position")}
        if self.selected.value != row:
            self.selected.set(row)
        self._selected_version = self.selected._version
        self._selection_graph = graph

    def _refresh_examples(self) -> list:
        if self.service is None:
            return []
        if self.selection is None or self.index is None:
            self.service.publish_empty_examples()
            return []
        return self.service.publish_feature_examples(
            self.selection["layer"],
            self.selection["feature"],
            limit=16,
        )

    def _update_selection(self) -> None:
        graph = self.service.graph
        if graph is not self._selection_graph:
            self._reconcile_selection(graph)
            self._refresh_examples()
        if self.selected._version != self._selected_version:
            node = self.selected.value
            self.selection = None
            self._selected_version = self.selected._version
            if node == -1:
                self._refresh_examples()
                return
            if graph is None or node >= len(graph.nodes):
                raise ValueError("selected_node requires a completed trace")
            record = graph.nodes[node]
            if record["kind"] != "feature":
                raise ValueError("selected_node must identify a feature")
            self.selection = {k: record[k] for k in ("layer", "feature", "position")}
            self._refresh_examples()

    def update_controls(self) -> None:
        self._enforce_readonly_outputs()
        clamp_changed = self.clamp._version != self._clamp_version
        self._clamp_version = self.clamp._version
        try:
            prompt = self.prompt.value
            prompt_inputs(self.backend, self.tokenizer, prompt)
            if prompt != self._last_prompt:
                self._reconcile_interventions_for_prompt(prompt)
                self._last_prompt = prompt
            self.checkpoint_prompt = prompt
            self._update_selection()
            if clamp_changed:
                self._set_clamp()
        finally:
            self._sync_clamp_status(preserve_edit=clamp_changed)

    def _feature_maximum(self, layer: int, feature: int, position: int) -> float:
        maximum = float(self.index.show(layer, feature)["maximum"])
        if maximum > 0 and math.isfinite(maximum):
            return maximum
        graph = self.service.graph
        if graph is not None:
            for node in graph.nodes:
                if (node["kind"] == "feature" and node["layer"] == layer
                        and node["feature"] == feature and node["position"] == position):
                    act = abs(float(node["activation"]))
                    if act > 0 and math.isfinite(act):
                        return act
        raise ValueError("selected feature has no positive activation maximum")

    def _set_clamp(self) -> bool:
        self._check_stop()
        if self.selection is None:
            raise ValueError("clamp requires a selected feature")
        layer, feature, position = (self.selection[k] for k in ("layer", "feature", "position"))
        maximum = self._feature_maximum(layer, feature, position)
        point = f"layers.{layer}.mlp_out" if self.dictionary.kind == "transcoder" else self.dictionary.metadata["write_points"][0]
        spec = self.interventions.executor.validate({
            "kind": "clamp",
            "point": point,
            "feature": feature,
            "positions": [position],
            "value": float(self.clamp.value) * maximum,
            "dictionary": self.interventions.executor.dictionary_id,
        })
        with self.controls.lock:
            owned = self._owned_record()
            if owned is not None and intervention_key(owned.spec) != intervention_key(spec):
                self._discard_owned_clamp()
            refused = self.controls._intervene({"set": spec})
            if refused is not None:
                raise ValueError(refused[1])
            record = self.controls._ivs[intervention_key(spec)]
            self._owned_clamp = record.spec
        self._apply_interventions()
        return True

    def _sync_clamp_status(self, *, preserve_edit: bool = False) -> None:
        reset_input = False
        if self.selection != self._clamp_selection:
            self._clamp_selection = deepcopy(self.selection)
            with self.controls.lock:
                if not preserve_edit and self.clamp._version == self._clamp_version and self.clamp.pending is None:
                    self.clamp._value = 0.0
                    self.clamp._version += 1
                    self._clamp_version = self.clamp._version
                    reset_input = True
            if reset_input:
                self.controls._publish()
        selected = "No feature selected."
        if self.selection is not None:
            selected = (
                f"Selected layer {self.selection['layer']}, feature {self.selection['feature']}, "
                f"position {self.selection['position']}."
            )
        live = "No slider clamp."
        with self.controls.lock:
            owned = self._owned_record()
            if owned is not None and owned.hook is not None and owned.applied_spec is not None:
                s = owned.applied_spec
                live = f"Live slider clamp: {s['point']}, feature {s['feature']}, position {s['positions'][0]}, value {s['value']:g}."
        msg = bounded_text(f"{selected} Clamp input: {self.clamp.value:g} times maximum. {live}")
        self._set_readonly_param(self.clamp_status, msg)

    def apply_clamp(self) -> str:
        self._check_stop()
        with internal_forward():
            self._preset_tuples = None
            self._preset_cell = None
            self.update_controls()
            self._set_clamp()
            self._sync_clamp_status()
            self.update_output(force=True)
        return "The selected feature clamp is applied."

    def _dallas_manifest_path(self) -> Path | None:
        manifest = Path(__file__).resolve().parents[3] / "demos" / "circuit_trace" / "published_dallas_v1.json"
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
