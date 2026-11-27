"""Hosted interpretation service bridging step-boundary controls and trace requests."""
from __future__ import annotations

from copy import deepcopy
import threading
from typing import TYPE_CHECKING

from .._intervention_protocol import internal_forward
from ._graph import Tracer
from ._hosted_resources import checkpoint_metadata
from ._prompt import prompt_inputs
from ._service import InterpretationService, Refused

if TYPE_CHECKING:
    from ._hosted import HostedInference


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
