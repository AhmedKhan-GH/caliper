"""Step-boundary service for feature indexes, attribution graphs, sweeps, and validation."""
from collections import deque
from contextlib import nullcontext
from copy import deepcopy
import threading
import time

from .. import _interpret_protocol as protocol
from ._labels import pack_labels


class Refused(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class InterpretationService:
    """Own bounded requests on the run thread, with no model work on a network thread."""

    def __init__(self, *, model=None, tokenizer=None, dictionary=None, index=None, tracer=None,
                 dictionary_factory=None, index_factory=None, tracer_factory=None, handlers=None,
                 memory_limit_gb=None, memory=None):
        self.model, self.tokenizer = model, tokenizer
        self.dictionary, self.index, self.tracer = dictionary, index, tracer
        self.memory_limit_gb, self.memory = memory_limit_gb, memory
        self.factories = {"dictionary": dictionary_factory, "index": index_factory, "tracer": tracer_factory}
        self.handlers = dict(handlers or {})
        self.graph = None
        self.status = {"state": "idle", "seconds": 0., "step": None}
        self.lock = threading.RLock()
        self.idle = threading.Condition(self.lock)
        self.queue = deque()
        self.running = False
        self.running_thread = None
        self.closed = False
        self.failure = None
        self.owner = None
        from ._memory import CleanStop
        self.stop_exception = CleanStop
        self._bind_resources()

    def _bind_resources(self):
        """Use one model, tokenizer, and dictionary for all supplied resources."""
        self._check_tracer_controls(self.tracer, getattr(self, "controls", None))
        if self.index is None and self.factories["index"] is None:
            self.index = getattr(self.tracer, "index", None)
        for resource in (self.index, self.tracer):
            if resource is None:
                continue
            for name in ("model", "tokenizer", "dictionary"):
                value = getattr(resource, name, None)
                current = getattr(self, name)
                if value is not None:
                    if current is not None and current is not value:
                        raise Refused("interpret.resource_mismatch", "Interpretation resources must share one " + name + ".")
                    setattr(self, name, value)

    @staticmethod
    def _check_tracer_controls(tracer, controls):
        if controls is not None and getattr(tracer, "controls", controls) is not controls:
            raise Refused("interpret.controls_mismatch", "The tracer must use the service control owner.")

    def _bind_controls(self, controls):
        with self.lock:
            if self.closed:
                raise Refused("interpret.closed", "The interpretation service is closed.")
            if getattr(self, "controls", controls) is not controls:
                raise Refused("interpret.controls_mismatch", "An interpretation service cannot change its control owner.")
            self._check_tracer_controls(self.tracer, controls)
            self.controls = controls

    @property
    def pending(self):
        with self.lock:
            return bool(self.queue) or self.failure is not None

    def submit(self, metadata, reply, *, acknowledge=None):
        """Return a status and result; send deferred results through ``reply``."""
        if not protocol.request(metadata):
            raise Refused("interpret.bad_request", "Invalid interpretation request.")
        method, params = metadata["method"], deepcopy(metadata["params"])
        with self.lock:
            if self.closed:
                raise Refused("interpret.closed", "The interpretation service is closed.")
            if method == "trace.status":
                return "ok", deepcopy(self.status)
            if self.failure is not None:
                raise Refused("interpret.stopped", "The interpretation run stopped.")
            if len(self.queue) >= 32:
                raise Refused("interpret.busy", "At most 32 requests can wait for a step.")
            if method == "trace.run":
                if self.status["state"] in ("pending", "running"):
                    raise Refused("interpret.busy", "A trace already waits for completion.")
            if acknowledge is not None:
                acknowledge()
            if method == "trace.run":
                self.status = {"state": "pending", "seconds": 0., "step": None}
            self.queue.append((method, params, reply))
        return "queued", None

    def _resource(self, name):
        self._bind_resources()
        value = getattr(self, name)
        if value is None:
            factory = self.factories[name]
            if factory is not None:
                value = factory()
            elif name == "dictionary":
                for owner in ("index", "tracer"):
                    if self.factories[owner] is not None:
                        self._resource(owner)
                        value = self.dictionary
                        if value is not None:
                            break
            elif name == "tracer":
                if self.model is None or self.tokenizer is None:
                    raise Refused("interpret.missing_model", "A model and tokenizer are required.")
                from . import Tracer
                value = Tracer(
                    self.model,
                    self.tokenizer,
                    self._resource("dictionary"),
                    controls=getattr(self, "controls", None),
                    guard=self.memory,
                )
            if value is None:
                raise Refused("interpret.missing_" + name, "The run has no " + name + ".")
            with self.lock:
                if name == "tracer":
                    self._check_tracer_controls(value, getattr(self, "controls", None))
                setattr(self, name, value)
                self._bind_resources()
        return value

    def _feature(self, params):
        if "node" not in params:
            return params["layer"], params["feature"]
        if self.graph is None:
            raise Refused("interpret.missing_trace", "No completed trace is available.")
        node = params["node"]
        if node >= len(self.graph.nodes) or self.graph.nodes[node]["kind"] != "feature":
            raise Refused("interpret.not_feature", "The selected node is not a feature.")
        record = self.graph.nodes[node]
        return record["layer"], record["feature"]

    def _baseline(self):
        from .._controls import CONTROLS
        owner = getattr(self, "controls", CONTROLS)
        service = getattr(owner, "intervention_service", None)
        if service is not None:
            return service.suspend()
        with owner.lock:
            if any(iv.hook is not None for iv in owner._ivs.values()):
                raise Refused("interpret.baseline_required", "Clear active ablations before a baseline feature query.")
        return nullcontext()

    def _execute(self, method, params, step, *, complete=None):
        from ._publish import _publication_scope
        scope = (
            _publication_scope(self.model)
            if method in ("trace.run", "feature.examples", "patch.sweep")
            else nullcontext()
        )
        with scope:
            result = self._execute_request(method, params, step)
            if complete is not None:
                complete(result)
            return result

    def _execute_request(self, method, params, step):
        if method == "feature.dictionary":
            dictionary = self._resource("dictionary")
            if self.index is None and self.factories["index"] is not None:
                self._resource("index")
            meta = deepcopy(getattr(dictionary, "metadata", {}))
            if self.index is None:
                return {"metadata": meta}
            idx_meta = getattr(self.index, "metadata", {}) or {}
            identity = {"dictionary_id": meta.get("dictionary_id", "default")}
            if "index_hash" in idx_meta:
                identity["index_hash"] = idx_meta["index_hash"]
            if "tokenizer_hash" in idx_meta:
                identity["tokenizer_hash"] = idx_meta["tokenizer_hash"]
            return {"metadata": meta, "identity": identity}
        if method in ("feature.show", "feature.examples"):
            with self._baseline():
                self._resource("dictionary")
                index = self._resource("index")
                layer, feature = self._feature(params)
                if method == "feature.show":
                    return index.show(layer, feature)
                rows = index.examples(layer, feature, limit=params.get("limit", 16))
            result = {
                "layer": layer,
                "feature": feature,
                "examples": rows,
                "bindings": {
                    k: "live:examples." + k
                    for k in ("tokens", "activations", "example_ids", "valid_tokens", "label_text")
                },
            }
            if not protocol.bounded(result):
                raise Refused("interpret.result_limit", "The example result exceeds the metadata limits.")
            self._publish_examples(rows)
            return result
        if method == "trace.run":
            started = time.monotonic()
            tracer = self._resource("tracer")
            index = self._resource("index") if self.index is not None or self.factories["index"] is not None else None
            tracer.bind_index(index)
            graph = tracer.trace(**params)
            summary = graph.read(node_limit=0, edge_limit=0)
            result = {
                "state": "done",
                "seconds": time.monotonic() - started,
                "step": step,
                **{k: v for k, v in summary.items() if k not in ("nodes", "edges", "tokens")},
            }
            if not protocol.bounded(result):
                raise Refused("interpret.result_limit", "The trace result exceeds the metadata limits.")
            with self.lock:
                self.graph = graph
            return result
        if method == "trace.read":
            if self.graph is None:
                raise Refused("interpret.missing_trace", "No completed trace is available.")
            return self.graph.read(**params)
        handler = self.handlers.get(method)
        if handler is None:
            raise Refused("interpret.unsupported", "This run does not support the requested operation.")
        return handler(**params)

    def _run_request(self, method, params, step):
        from ._memory import CleanStop
        if method == "trace.run":
            with self.lock:
                self.status = {"state": "running", "seconds": 0., "step": step}
        try:
            result = self._execute(method, params, step)
            if not protocol.bounded(result):
                raise Refused("interpret.result_limit", "The result exceeds the structured metadata limits.")
        except CleanStop:
            raise
        except BaseException as exc:
            code = getattr(exc, "code", "interpret.refused")
            if not protocol.text(code, 63, False):
                code = "interpret.refused"
            error = {
                "code": code,
                "message": str(exc).replace("\0", "").encode("utf-8", errors="replace")[:1024].decode("utf-8", errors="ignore"),
            }
            if method == "trace.run":
                with self.lock:
                    self.status = {"state": "error", "seconds": 0., "step": step, "error": error}
            if not isinstance(exc, Exception):
                raise
            return "error", None, error
        if method == "trace.run":
            with self.lock:
                self.status = result
        return "ok", result, None

    def _retrace(self, params, step):
        """Run an automatic retrace when idle."""
        from ._memory import CleanStop
        with self.lock:
            if self.running or self.closed:
                return None
            if self.failure is not None:
                raise self.failure
            if self.status["state"] in ("pending", "running"):
                return None
            self.running = True
            self.running_thread = threading.get_ident()
            self.status = {"state": "running", "seconds": 0., "step": step}
        try:
            return self._run_request("trace.run", params, step)
        except CleanStop as exc:
            self.stop(exc, step)
            raise self.failure
        finally:
            with self.idle:
                self.running = False
                self.running_thread = None
                self.idle.notify_all()

    def stop(self, failure, step=None):
        """Latch a resource stop and terminate every queued request."""
        with self.lock:
            if self.failure is None:
                self.failure = failure
            error = {
                "code": "interpret.stopped",
                "message": str(self.failure).replace("\0", "").encode("utf-8", errors="replace")[:1024].decode("utf-8", errors="ignore"),
            }
            self.status = {"state": "error", "seconds": 0., "step": step, "error": error}
            requests, self.queue = self.queue, deque()
        for _, _, reply in requests:
            try:
                reply("error", None, error)
            except Exception:
                pass
        return error

    def step(self, step=None):
        """Execute the requests present at entry, once, on the run thread."""
        from ._memory import CleanStop
        with self.lock:
            if self.running or self.closed:
                return 0
            if self.failure is not None:
                raise self.failure
            self.running = True
            self.running_thread = threading.get_ident()
            count = len(self.queue)
        try:
            for _ in range(count):
                with self.lock:
                    if not self.queue:
                        break
                    method, params, reply = self.queue.popleft()
                try:
                    response = self._run_request(method, params, step)
                except CleanStop as exc:
                    error = self.stop(exc, step)
                    try:
                        reply("error", None, error)
                    except Exception:
                        pass
                    raise self.failure
                else:
                    reply(*response)
            return count
        finally:
            with self.lock:
                self.running = False
                self.running_thread = None
                self.idle.notify_all()

    def _check_close_thread(self):
        with self.lock:
            if self.running_thread == threading.get_ident():
                raise Refused("interpret.busy", "An active service request cannot close its own service.")

    def close(self):
        """Wait for active work before the caller disposes of external resources."""
        with self.idle:
            self._check_close_thread()
            self.closed = True
            requests, self.queue = self.queue, deque()
            while self.running:
                self.idle.wait()
        for _, _, reply in requests:
            try:
                reply("error", None, {"code": "interpret.closed", "message": "The interpretation service closed."})
            except Exception:
                pass

    def _publish_examples(self, rows):
        from ._publish import _publication_scope
        with _publication_scope(self.model):
            self._publish_example_tensors(rows)

    def _publish_example_tensors(self, rows):
        import torch
        from ._publish import _publish_many
        device = next(self.model.parameters()).device if self.model is not None else "cpu"
        tokens = torch.zeros((16, 25), dtype=torch.int32)
        activations = torch.zeros((16, 25), dtype=torch.float32)
        ids = torch.full((16,), -1, dtype=torch.int32)
        valid = torch.zeros(16, dtype=torch.int32)
        labels = torch.zeros((400, 16), dtype=torch.uint32)
        for i, row in enumerate(rows):
            ids[i] = row["example_id"]
            slots = [j for j in range(4, 29) if row["valid"][j]]
            valid[i] = len(slots)
            for col, j in enumerate(slots):
                tokens[i, col] = row["token_ids"][j]
                activations[i, col] = row["activations"][j]
                labels[i * 25 + col] = pack_labels([row["tokens"][j]], 1)[0]
        _publish_many({
            "examples." + name: value.to(device)
            for name, value in zip(
                ("tokens", "activations", "example_ids", "valid_tokens", "label_text"),
                (tokens, activations, ids, valid, labels),
            )
        })


def _refresh_native_capability(controls):
    import sys
    from .._controls import CONTROLS
    if controls is not CONTROLS:
        return
    native = sys.modules.get("caliper._net_native")
    if native is not None:
        native.refresh_interpretation()


def register_service(service=None, *, controls=None, **resources):
    """Register one service; Controls checkpoints call its step after interventions."""
    if controls is None:
        from .._controls import CONTROLS
        controls = CONTROLS
    service = service or InterpretationService(**resources)
    with controls.lock:
        old = controls.interpretation_service
    if old is not None:
        old._check_close_thread()
    with controls.interpretation_registration_lock:
        service._bind_controls(controls)
        with controls.lock:
            old = controls.interpretation_service
        if old is not None and old is not service:
            old.close()
        with controls.lock:
            controls.interpretation_service = service
            controls._update_busy()
            if service.pending:
                controls.cond.notify_all()
        _refresh_native_capability(controls)
    return service


def unregister_service(service, *, controls=None):
    if controls is None:
        from .._controls import CONTROLS
        controls = CONTROLS
    service._check_close_thread()
    with controls.interpretation_registration_lock:
        service.close()
        with controls.lock:
            if controls.interpretation_service is service:
                controls.interpretation_service = None
                controls._update_busy()
        _refresh_native_capability(controls)
