"""Hosted inference run loop, controls, and checkpoints over Backend."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
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


def load_resources(model_name: str, dictionary_path: str | Path, *, device=None, memory_limit_gb: float = MAX_PROCESS_GB):
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

    backend = load_backend(model_name, dictionary_path, device=dev, guard=guard)
    dict_dir = Path(str(dictionary_path))
    if dict_dir.is_dir() and (dict_dir / "index" / "index.json").is_file():
        index = FeatureIndex(
            dict_dir,
            model=backend.model,
            tokenizer=backend.tokenizer,
            dictionary=backend.dictionary,
            memory=guard,
        )
    else:
        index = LiveFeatureIndex(backend, memory=guard, allow_online=os.environ.get("CALIPER_AGENT_OFFLINE") != "1")
    return backend, index, guard


class HostedService(InterpretationService):
    """Run step-boundary controls, interpretation requests, and output comparisons together."""

    def __init__(self, owner: "HostedInference"):
        self.owner = owner
        self.boundary_running = False
        self.boundary_thread = None
        tracer = Tracer(
            owner.model,
            owner.tokenizer,
            owner.dictionary,
            backend=owner.backend,
            index=owner.index,
            guard=owner.guard,
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

    def _bind_controls(self, controls):
        if self.tracer is not None:
            self.tracer.controls = controls
        super()._bind_controls(controls)

    def _check_intervention_budget(self, ivs):
        for iv in ivs.values():
            if not checkpoint_metadata({"spec": iv.spec}):
                raise ValueError("intervention exceeds the checkpoint metadata limits")

    def submit(self, metadata, reply, *, acknowledge=None):
        status, result = super().submit(metadata, reply, acknowledge=acknowledge)
        if metadata["method"] == "trace.status":
            result["inference"] = self.owner.inference_status()
        return status, result

    def stop(self, failure, step=None):
        error = super().stop(failure, step)
        self.owner._latch_failure(self.failure)
        self.owner.controls._terminate_actions(self.failure)
        return error

    def _execute(self, method, params, step, *, complete=None):
        if method != "trace.run":
            return super()._execute(method, params, step, complete=complete)

        prompt_param = self.owner.prompt
        with self.owner.controls.lock:
            prompt_version = prompt_param._version
            entry = prompt_param.entry()
            entry["value"] = params["prompt"]
            self.owner.controls._check_param_budget(prompt_param, entry)
        prompt_inputs(self.owner.backend, self.tokenizer, params["prompt"])
        traced_specs = self.owner.interventions.specifications()

        def on_complete(_result):
            publish_prompt = False
            with self.owner.controls.lock:
                if (self.owner.controls.params.get(prompt_param.name) is prompt_param
                        and prompt_param._version == prompt_version and prompt_param.pending is None):
                    prompt_param._value = params["prompt"]
                    prompt_param._version += 1
                    self.owner.checkpoint_prompt = params["prompt"]
                    self.owner.controls._update_busy()
                    publish_prompt = True
            self.owner.traced_prompt = params["prompt"]
            self.owner.traced_specifications = traced_specs
            self.owner.traced_state = deepcopy(traced_specs)
            self.owner._reconcile_selection(self.graph)
            if self.owner.fixture_display is not None:
                self.owner.fixture_display.publish()
            if publish_prompt:
                self.owner.controls._publish()

        return super()._execute(method, params, step, complete=on_complete)

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
                    self.owner.status.set(bounded_text(str(exc)))
                count = super().step(step)
                if self.owner.fixture_display is not None and count:
                    self.owner.fixture_display.publish()
                try:
                    self.owner.update_output()
                except ValueError as exc:
                    self.owner.status.set(bounded_text(str(exc)))
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

    def __init__(self, model_name: str, dictionary_path: str | Path, prompt: str, *, resources=None):
        self.model_name = str(model_name)
        dict_p = Path(str(dictionary_path))
        self.dictionary_path = str(dict_p.resolve()) if dict_p.exists() else str(dictionary_path)
        if resources is None:
            self.backend, self.index, self.guard = load_resources(self.model_name, self.dictionary_path)
        else:
            self.backend, self.index, self.guard = resources
        self.model = self.backend.model
        self.tokenizer = self.backend.tokenizer
        self.dictionary = self.backend.dictionary

        mps_display = os.environ.get("CALIPER_FIXTURE_DISPLAY_MPS", "0") == "1" and self.model_name == "tiny-lm"
        self.fixture_display = _FixtureDisplay(MemoryGuard(12, "mps")) if mps_display else None

        prompt_inputs(self.backend, self.tokenizer, prompt)
        self.backend.record_prompt(prompt)
        self.initial_prompt = prompt
        self.checkpoint_prompt = prompt
        self.restored = None
        self.handle = self.service = self.managed = None
        self._failure = None
        self._stop_checkpointed = False
        self._inference_lock = threading.Lock()
        self._inference_ready = False
        self._comparison = None
        self._restore_generation(0)
        self.selection = None
        self._owned_clamp = None
        self._clamp_selection = None
        self._selection_graph = None
        self.traced_prompt = None
        self.traced_specifications = None
        self.traced_state = None
        self._output_key = None
        self._clamp_version = None
        self._selected_version = None
        self._retrace = False
        self._preset_tuples = None
        self._preset_cell = None
        self._dallas_cache = None
        self.parameters: list = []
        self.actions: list = []

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
                    "state": "starting",
                    "prompt_hash": None,
                    "interventions_hash": None,
                    "comparison": None,
                    "generation": deepcopy(self._generation),
                }
        state = self._input_state()
        hashes = self._input_hashes(state)
        with self._inference_lock:
            return {
                **hashes,
                "state": "ready",
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
        try:
            self.service = register_service(HostedService(self), controls=self.controls)
            self.handle = caliper.watch(self.model, activations=False)
            self.interventions = self.controls.intervention_service
            self.interventions.configure(
                dictionary=self.dictionary,
                tokenizer=self.tokenizer,
                resource_guard=self.guard,
                backend=self.backend,
            )
            self.output = Output(self.backend, self.interventions, guard=self.guard)

            def add_param(name, value, **kwargs):
                p = caliper.param(name, value, **kwargs)
                self.parameters.append(p)
                return p

            self.prompt = add_param("prompt", self.initial_prompt)
            prompt_inputs(self.backend, self.tokenizer, self.prompt.value)
            self.checkpoint_prompt = self.prompt.value
            self.selected = add_param("selected_node", -1, min=-1, max=LARGE_NODE_CAP - 1)
            self.clamp = add_param("clamp", 0.0, min=-4.0, max=4.0)
            self.clamp_status = add_param("clamp.status", "No feature selected. No slider clamp.")
            self.retrace = add_param("retrace", False)
            self.baseline_text = add_param("output.baseline", "")
            self.steered_text = add_param("output.steered", "")
            self.status = add_param("output.status", "ready")
            self.actions = [
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
            self.boundary()
            return self
        except BaseException:
            self.close()
            raise

    def _apply_interventions(self) -> None:
        with self.controls.lock:
            changed, _ready = self.controls._intervention_boundary()
            self._owned_record()
            self.controls._update_busy()
        if changed:
            self.controls._publish()

    def _reconcile_selection(self, graph) -> None:
        row = -1
        if graph is not None and self.selection is not None:
            row = next(
                (i for i, n in enumerate(graph.nodes)
                 if n["kind"] == "feature" and all(n[k] == v for k, v in self.selection.items())),
                -1,
            )
        if row == -1:
            self.selection = None
        if self.selected.value != row:
            self.selected.set(row)
        self._selected_version = self.selected._version
        self._selection_graph = graph

    def _update_selection(self) -> None:
        graph = self.service.graph
        if graph is not self._selection_graph:
            self._reconcile_selection(graph)
        if self.selected._version != self._selected_version:
            node = self.selected.value
            self.selection = None
            self._selected_version = self.selected._version
            if node == -1:
                return
            if graph is None or node >= len(graph.nodes):
                raise ValueError("selected_node requires a completed trace")
            record = graph.nodes[node]
            if record["kind"] != "feature":
                raise ValueError("selected_node must identify a feature")
            self.selection = {k: record[k] for k in ("layer", "feature", "position")}

    def update_controls(self) -> None:
        clamp_changed = self.clamp._version != self._clamp_version
        self._clamp_version = self.clamp._version
        try:
            prompt = self.prompt.value
            prompt_inputs(self.backend, self.tokenizer, prompt)
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
        if self.clamp_status.value != msg:
            self.clamp_status.set(msg)

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
        for key, prompt_text in manifest["prompts"].items():
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
            t0 = time.perf_counter()
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
            self.clamp_status.set(bounded_text(summary))
            self.status.set("ready")
            return summary

    def update_output(self, *, force: bool = False) -> None:
        key = self._input_state()
        changed = self._output_key != key
        if changed:
            self._preset_tuples = None
            self._preset_cell = None
        if changed or force:
            started = time.perf_counter()
            comparison = self.output.compare(key[0])
            seconds = time.perf_counter() - started
            if self.fixture_display is not None:
                self.fixture_display.publish()
            with self._inference_lock:
                self._comparison = {
                    **self._input_hashes(key),
                    "argmax_token_ids": comparison["argmax_token_ids"],
                    "baseline_top_text": comparison["baseline_top_text"],
                    "baseline_top_probability": comparison["baseline_top_probability"],
                    "completed_step": self.controls.n,
                    "compute_publish_seconds": seconds,
                }
            self._output_key = deepcopy(key)
            self.status.set("ready")
        if self.retrace.value and (changed or not self._retrace):
            if (self.traced_prompt, self.traced_specifications) != key:
                response = self.service._retrace({"prompt": key[0]}, self.controls.n)
                if response is not None and response[0] == "error":
                    self.status.set(response[2]["message"])
                elif self.fixture_display is not None:
                    self.fixture_display.publish()
        self._retrace = self.retrace.value
        if self._preset_cell is None:
            self._sync_clamp_status()
        if self.guard is not None:
            self.guard.check("inference boundary", force=False)

    def generate(self) -> str:
        self._check_stop()
        state = self._input_state()
        with self._inference_lock:
            gen_id = self._generation["id"] + 1
        t0 = time.perf_counter()
        with internal_forward():
            if self._preset_tuples is not None:
                raw = self.backend.generate(state[0], self._preset_tuples, max_new_tokens=24, guard=self.guard)
                result = {
                    "baseline": bounded_text(raw["baseline"]),
                    "steered": bounded_text(raw["steered"]),
                    "token_ids": raw["token_ids"],
                }
            else:
                result = self.output.generate(state[0])
        gen_seconds = time.perf_counter() - t0
        ids = result["token_ids"]
        texts = [result["baseline"], result["steered"]]
        self.baseline_text.set(texts[0])
        self.steered_text.set(texts[1])
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
                    self.status.set("The clamp is reset. Correct the prompt to update the output.")
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
                self.status.set(bounded_text(f"{type(exc).__name__}: {exc}"))

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
        if self.index is not None:
            self.index.close()
        if self.managed is not None and self.managed.inference is self:
            self.managed.inference = None


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Local Inference Run")
    parser.add_argument("--model")
    parser.add_argument("--dictionary")
    parser.add_argument("--prompt")
    parser.add_argument("--fixture-dir")
    args = parser.parse_args(argv)
    if args.fixture_dir:
        if any((args.model, args.dictionary, args.prompt)):
            parser.error("fixture creation takes no run arguments")
        fixture_resources(args.fixture_dir)
        return
    if not all((args.model, args.dictionary, args.prompt)):
        parser.error("model, dictionary, and prompt are required")
    run = HostedInference(args.model, args.dictionary, args.prompt)
    try:
        run.start()
        print(f"caliper: inference ready at step {run.controls.n}", flush=True)
        run.run()
    finally:
        run.close()


if __name__ == "__main__":
    from caliper.interp._hosted import main as hosted_main
    hosted_main()
