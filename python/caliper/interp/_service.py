"""Step-boundary service for real feature indexes and attribution graphs."""
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
    """Own bounded requests on the run thread, with no model work on a network thread.

    Resources can be actual objects or zero-argument lazy factories. Explicit
    factories are resolved once, on the thread that calls ``step``.
    """

    def __init__(self, *, model=None, tokenizer=None, dictionary=None, index=None, tracer=None,
                 dictionary_factory=None, index_factory=None, tracer_factory=None, handlers=None):
        self.model, self.tokenizer = model, tokenizer
        self.dictionary, self.index, self.tracer = dictionary, index, tracer
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

    @property
    def pending(self):
        with self.lock:
            return bool(self.queue)

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
            if len(self.queue) >= 32:
                raise Refused("interpret.busy", "At most 32 requests can wait for a step.")
            if method == "trace.run":
                if self.status["state"] in ("pending", "running"):
                    raise Refused("interpret.busy", "A trace already waits for completion.")
            # The transport queues its acknowledgment before the run can see work.
            if acknowledge is not None:
                acknowledge()
            if method == "trace.run":
                self.status = {"state": "pending", "seconds": 0., "step": None}
            self.queue.append((method, params, reply))
        return "queued", None

    def _resource(self, name):
        value = getattr(self, name)
        if value is None:
            factory = self.factories[name]
            if factory is not None:
                value = factory()
            elif name == "tracer":
                if self.model is None or self.tokenizer is None:
                    raise Refused("interpret.missing_model", "A model and tokenizer are required.")
                from . import Tracer
                value = Tracer(self.model, self.tokenizer, self._resource("dictionary"),
                               controls=getattr(self, "controls", None))
            if value is None:
                raise Refused("interpret.missing_" + name, "The run has no " + name + ".")
            setattr(self, name, value)
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
        # Legacy M7 hooks have no suspension API. Refuse before any index access.
        with owner.lock:
            if any(iv.hook is not None for iv in owner._ivs.values()):
                raise Refused("interpret.baseline_required", "Clear active ablations before a baseline feature query.")
        return nullcontext()

    def _execute(self, method, params, step):
        if method == "feature.dictionary":
            return {"metadata": deepcopy(self._resource("dictionary").metadata)}
        if method in ("feature.show", "feature.examples"):
            with self._baseline():
                self._resource("dictionary")
                index = self._resource("index")
                layer, feature = self._feature(params)
                if method == "feature.show":
                    return index.show(layer, feature)
                rows = index.examples(layer, feature, limit=params.get("limit", 16))
            result = {"layer": layer, "feature": feature, "examples": rows,
                      "bindings": {k: "live:examples." + k for k in
                                   ("tokens", "activations", "example_ids", "valid_tokens", "label_text")}}
            if not protocol.bounded(result):
                raise Refused("interpret.result_limit", "The example result exceeds the metadata limits.")
            self._publish_examples(rows)
            return result
        if method == "trace.run":
            started = time.monotonic()
            with self.lock:
                self.status = {"state": "running", "seconds": 0., "step": step}
            graph = self._resource("tracer").trace(**params)
            summary = graph.read(node_limit=0, edge_limit=0)
            result = {"state": "done", "seconds": time.monotonic() - started, "step": step,
                      **{k: v for k, v in summary.items() if k not in ("nodes", "edges", "tokens")}}
            if not protocol.bounded(result):
                raise Refused("interpret.result_limit", "The trace result exceeds the metadata limits.")
            self._publish_trace_panels(graph)
            with self.lock:
                self.graph, self.status = graph, result
            return result
        if method == "trace.read":
            if self.graph is None:
                raise Refused("interpret.missing_trace", "No completed trace is available.")
            return self.graph.read(**params)
        handler = self.handlers.get(method)
        if handler is None:
            raise Refused("interpret.unsupported", "This run does not support the requested operation.")
        return handler(**params)

    def step(self, step=None):
        """Execute the requests present at entry, once, on the run thread."""
        with self.lock:
            if self.running or self.closed:
                return 0
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
                    result = self._execute(method, params, step)
                    if not protocol.bounded(result):
                        raise Refused("interpret.result_limit", "The result exceeds the structured metadata limits.")
                except Exception as exc:
                    error = {"code": getattr(exc, "code", "interpret.refused"),
                             "message": str(exc).encode("utf-8", errors="replace")[:1024].decode("utf-8", errors="ignore")}
                    if method == "trace.run":
                        with self.lock:
                            self.status = {"state": "error", "seconds": 0., "step": step, "error": error}
                    reply("error", None, error)
                else:
                    reply("ok", result, None)
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
            reply("error", None, {"code": "interpret.closed", "message": "The interpretation service closed."})

    def _publish_examples(self, rows):
        import torch
        from . import publish
        device = next(self.model.parameters()).device if self.model is not None else "cpu"
        tokens = torch.zeros((16, 25), dtype=torch.int32)
        activations = torch.zeros((16, 25), dtype=torch.float32)
        ids = torch.full((16,), -1, dtype=torch.int32)
        valid = torch.zeros(16, dtype=torch.int32)
        labels = torch.zeros((400, 16), dtype=torch.uint32)
        for i, row in enumerate(rows):
            ids[i] = row["example_id"]
            # The piece uses a valid prefix. Compact document-edge padding.
            slots = [j for j in range(4, 29) if row["valid"][j]]
            valid[i] = len(slots)
            for col, j in enumerate(slots):
                tokens[i, col] = row["token_ids"][j]
                activations[i, col] = row["activations"][j]
                labels[i * 25 + col] = pack_labels([row["tokens"][j]], 1)[0]
        for name, value in zip(("tokens", "activations", "example_ids", "valid_tokens", "label_text"),
                               (tokens, activations, ids, valid, labels)):
            publish("examples." + name, value.to(device))

    @staticmethod
    def _pack_labels(values, count):
        return pack_labels(values, count)

    def _publish_trace_panels(self, graph):
        """Publish display labels and the existing traced probabilities without a forward."""
        import math
        import torch
        from . import publish
        device = graph.tensors["trace.counts"].device
        labels = [token["text"] for token in graph.tokens]
        labels.extend([""] * (32 - len(labels)))
        for node in graph.nodes:
            labels.append(node.get("label") or node["id"])
        logits = [node for node in graph.nodes if node["kind"] == "logit"][:10]
        probabilities = [float(node["probability"]) for node in logits]
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
            raise Refused("interpret.result_limit", "Traced probabilities must be finite values from zero to one.")
        probs = torch.zeros((2, 10), dtype=torch.float32)
        tokens = torch.full((10,), -1, dtype=torch.int32)
        if logits:
            probs[:, :len(logits)] = torch.tensor(probabilities, dtype=torch.float32)
            tokens[:len(logits)] = torch.tensor([node["token_id"] for node in logits], dtype=torch.int32)
        values = {"label_text": self._pack_labels(labels, 544), "probs": probs,
                  "logit_tokens": tokens,
                  "logit_label_text": self._pack_labels([node["label"] for node in logits], 10)}
        for name, value in values.items():
            publish("trace." + name, value.to(device))


def _refresh_native_capability(controls):
    # Registration does not load an unused native extension or start a server.
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
        with service.lock:
            if service.closed:
                raise Refused("interpret.closed", "The interpretation service is closed.")
        with controls.lock:
            old = controls.interpretation_service
        if old is not None and old is not service:
            old.close()
        with controls.lock:
            controls.interpretation_service = service
            service.controls = controls
            controls.busy = True
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
        _refresh_native_capability(controls)
