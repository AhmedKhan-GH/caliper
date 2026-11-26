"""Event-controlled publication across watch lifetimes on small CPU fixtures."""
import threading
from types import SimpleNamespace

import pytest
import torch

from caliper import _programs as P
from caliper.interp import Tracer, _publish, publish
from caliper.interp._backend import make_tiny_backend
from caliper.interp._graph import Graph
from caliper.interp._index import FeatureIndex, build_index
from caliper.interp._service import InterpretationService
from .test_publication import registry


@pytest.mark.parametrize("transition", ["replace", "remove", "reattach", "first_attach"])
@pytest.mark.parametrize("operation", ["trace", "cached_trace", "service_trace", "examples"])
def test_replaced_watch_refuses_late_publication(registry, monkeypatch, tmp_path, operation, transition):
    backend = make_tiny_backend(n_layers=1)
    model, tokenizer, dictionary = backend.model, backend.tokenizer, backend.dictionary
    guard = SimpleNamespace(check=lambda stage, **kw: None)
    tracer = Tracer(model, tokenizer, dictionary, backend=backend, guard=guard)
    if transition != "first_attach":
        P.attach(model, None)
    options = dict(prompt="A", max_targets=0, max_nodes=8, max_edges=8)
    entered, resume, changed = (threading.Event() for _ in range(3))
    errors, replies, changes = [], [], []
    index = service = None

    def pause():
        entered.set()
        assert resume.wait(5), "The watch change did not release the request."

    if operation == "examples":
        build_index(
            tmp_path, [("fixture", [2, 37, 38, 39, 40])], model=model,
            tokenizer=tokenizer, dictionary=dictionary, provenance={"partition": "fixture"},
        )
        index = FeatureIndex(tmp_path, model=model, tokenizer=tokenizer, dictionary=dictionary)
        orig_examples = index.examples

        def query(*args, **kwargs):
            rows = orig_examples(*args, **kwargs)
            pause()
            return rows

        monkeypatch.setattr(index, "examples", query)
        service = InterpretationService(index=index)
        service.submit(
            {"version": 1, "method": "feature.examples", "params": {"layer": 0, "feature": int(index.counts[0].argmax())}},
            lambda *args: replies.append(args),
        )
        work = service.step
    else:
        if operation == "cached_trace":
            tracer.trace(**options)
            _publish._clear()
            monkeypatch.setattr(guard, "check", lambda stage, **kw: pause() if stage == "cached publication" else None)
        else:
            orig_trace = tracer._trace

            def slow_trace(*args, **kwargs):
                res = orig_trace(*args, **kwargs)
                pause()
                return res

            monkeypatch.setattr(tracer, "_trace", slow_trace)
        if operation == "service_trace":
            service = InterpretationService(tracer=tracer)
            service.submit({"version": 1, "method": "trace.run", "params": options}, lambda *args: replies.append(args))
            work = service.step
        else:
            work = lambda: tracer.trace(**options)

    def run():
        try:
            work()
        except BaseException as exc:
            errors.append(exc)

    replacement = torch.nn.Linear(2, 2) if transition == "replace" else (None if transition == "remove" else model)

    def replace():
        try:
            if transition in ("remove", "reattach"):
                P.detach(model)
            if replacement is not None:
                P.attach(replacement, None)
            publish("custom.current", torch.ones(1))
            changed.set()
        except BaseException as exc:
            changes.append(exc)

    worker = threading.Thread(target=run)
    changer = threading.Thread(target=replace)
    worker.start()
    try:
        assert entered.wait(5), errors
        changer.start()
        assert changed.wait(5)
        assert P._STATE["model"] is replacement
    finally:
        resume.set()
        worker.join(6)
        if changer.ident is not None:
            changer.join(6)
        if service is not None:
            service.close()
        if index is not None:
            index.close()
    assert not changes
    assert set(_publish._TENSORS) == {"custom.current"}
    if service is not None:
        assert replies[0][0] == "error" and replies[0][2]["code"] == "interpret.watch_changed"
    else:
        assert len(errors) == 1 and getattr(errors[0], "code", None) == "interpret.watch_changed"


def test_other_watched_model_refuses_before_trace_work(registry, monkeypatch):
    backend = make_tiny_backend(n_layers=1)
    tracer = Tracer(backend.model, backend.tokenizer, backend.dictionary, backend=backend)
    P.attach(torch.nn.Linear(2, 2), None)
    monkeypatch.setattr(tracer, "_trace", lambda *args, **kwargs: pytest.fail("A foreign model reached trace work."))
    with pytest.raises(RuntimeError) as caught:
        tracer.trace("A")
    assert caught.value.code == "interpret.watch_changed"
    assert not _publish._TENSORS


def test_same_watch_attachment_keeps_lease(registry):
    backend = make_tiny_backend(n_layers=1)
    tracer = Tracer(backend.model, backend.tokenizer, backend.dictionary, backend=backend)
    graph = tracer.trace("A", max_targets=0)
    before = dict(_publish._TENSORS)
    P.attach(backend.model, None)
    assert _publish._TENSORS == before
    with _publish._publication_scope(backend.model):
        P.attach(backend.model, None)
        assert graph.publish() is graph
    P.detach(backend.model)
    assert not _publish._TENSORS
