"""Event-controlled publication across watch lifetimes on small CPU fixtures."""
import threading

import pytest
import torch

from caliper import _programs as P
from caliper.interp import Tracer, _publish, publish
from caliper.interp._service import InterpretationService
from .fixture import TinyLM, TinyTokenizer
from .test_publication import registry
from .test_trace import exact_dictionary


@pytest.mark.parametrize("transition", ["replace", "remove", "reattach", "first_attach"])
@pytest.mark.parametrize("operation", ["trace", "cached_trace", "service_trace", "examples"])
def test_replaced_watch_refuses_late_publication(registry, monkeypatch, tmp_path, operation, transition):
    from caliper.interp import _trace
    from caliper.interp._index import FeatureIndex, build_index
    from .test_dictionary import write_fixture

    model = TinyLM(layers=1, width=8, heads=2, latents=8, k=2)
    tokenizer, dictionary = TinyTokenizer(), exact_dictionary(model)
    tracer = Tracer(model, tokenizer, dictionary)
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
        write_fixture(tmp_path, model)
        build_index(tmp_path, [("fixture", [65, 66, 67, 68])], model=model,
                    tokenizer=tokenizer, dictionary=dictionary, provenance={"partition": "fixture"})
        index = FeatureIndex(tmp_path, model=model, tokenizer=tokenizer, dictionary=dictionary)
        examples = index.examples

        def query(*args, **kwargs):
            rows = examples(*args, **kwargs)
            pause()
            return rows

        monkeypatch.setattr(index, "examples", query)
        service = InterpretationService(index=index)
        service.submit({"version": 1, "method": "feature.examples",
                        "params": {"layer": 0, "feature": int(index.counts[0].argmax())}},
                       lambda *args: replies.append(args))
        work = service.step
    else:
        if operation == "cached_trace":
            tracer.trace(**options)
            _publish._clear()
            check = tracer.guard.check

            def guard(stage):
                result = check(stage)
                if stage == "cached publication":
                    pause()
                return result

            monkeypatch.setattr(tracer.guard, "check", guard)
        else:
            backward = _trace._DirectEffects.backward

            def effects(self, targets):
                result = backward(self, targets)
                pause()
                return result

            monkeypatch.setattr(_trace._DirectEffects, "backward", effects)
        if operation == "service_trace":
            service = InterpretationService(tracer=tracer)
            service.submit({"version": 1, "method": "trace.run", "params": options},
                           lambda *args: replies.append(args))
            work = service.step
        else:
            work = lambda: tracer.trace(**options)

    def run():
        try:
            work()
        except BaseException as exc:
            errors.append(exc)

    replacement = torch.nn.Linear(2, 2) if transition == "replace" else (
        None if transition == "remove" else model)

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
        assert changed.wait(5), "Model work retained the program lock."
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
    assert not worker.is_alive() and not changer.is_alive()
    assert not changes
    assert set(_publish._TENSORS) == {"custom.current"}
    if service is not None:
        assert not errors
        assert replies[0][0] == "error", replies
        assert replies[0][2]["code"] == "interpret.watch_changed"
        if operation == "service_trace":
            assert service.graph is None
            assert service.status["state"] == "error"
    else:
        assert len(errors) == 1
        assert getattr(errors[0], "code", None) == "interpret.watch_changed"


@pytest.mark.parametrize("method", ["trace.run", "feature.examples"])
def test_lazy_resource_retains_request_lease(registry, monkeypatch, method):
    from types import SimpleNamespace

    model = TinyLM(layers=1, width=8, heads=2, latents=8, k=2)
    tokenizer, dictionary = TinyTokenizer(), exact_dictionary(model)
    tracer = Tracer(model, tokenizer, dictionary)
    P.attach(model, None)
    entered, resume, changed = (threading.Event() for _ in range(3))
    replies, errors = [], []

    def factory():
        entered.set()
        assert resume.wait(5)
        if method == "trace.run":
            return tracer
        return SimpleNamespace(model=model, tokenizer=tokenizer, dictionary=dictionary,
                               examples=lambda *args, **kwargs: [])

    resources = {"tracer_factory" if method == "trace.run" else "index_factory": factory}
    service = InterpretationService(**resources)
    params = dict(prompt="A", max_targets=0) if method == "trace.run" else dict(layer=0, feature=0)

    def reply(*args):
        replies.append(args)
        # Reply callbacks run after the expired scope exits.
        publish("custom.callback", torch.ones(1))

    service.submit({"version": 1, "method": method, "params": params}, reply)

    def replace():
        try:
            # An identity-only check misses a removal followed by the same model.
            P.detach(model)
            P.attach(model, None)
            changed.set()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=service.step)
    changer = threading.Thread(target=replace)
    worker.start()
    try:
        assert entered.wait(5)
        changer.start()
        assert changed.wait(5), "The factory retained the program lock."
    finally:
        resume.set()
        worker.join(6)
        if changer.ident is not None:
            changer.join(6)
        service.close()
    assert not worker.is_alive() and not changer.is_alive()
    assert not errors
    assert replies[0][0] == "error", replies
    assert replies[0][2]["code"] == "interpret.watch_changed"
    assert set(_publish._TENSORS) == {"custom.callback"}


def empty_graph():
    from caliper.interp._graph import Graph
    return Graph.build(nodes=[], edges=[], tokens=[{"token_id": 65, "text": "A"}], layers=1,
                       completeness={}, omitted_influence=0., error_share=0., feature_share=0.,
                       trace_id="fixture", device="cpu")


@pytest.mark.parametrize("kind", ["graph", "examples"])
def test_tensor_group_conflict_preserves_previous_publication(registry, kind):
    model = torch.nn.Linear(2, 2)
    namespace = torch.nn.Module()
    namespace.register_buffer("logit_label_text" if kind == "graph" else "label_text", torch.zeros(1))
    model.add_module("trace" if kind == "graph" else "examples", namespace)
    P.attach(model, None)
    name = "trace.counts" if kind == "graph" else "examples.tokens"
    previous = torch.tensor([-1], dtype=torch.int32)
    publish(name, previous)
    if kind == "graph":
        work = empty_graph().publish
    else:
        service = InterpretationService(model=model)
        work = lambda: service._publish_examples([])
    with pytest.raises(ValueError, match="conflicts with a model tensor"):
        work()
    assert set(_publish._TENSORS) == {name}
    assert _publish._TENSORS[name].data_ptr() == previous.data_ptr()


def test_same_watch_attachment_keeps_lease_and_pre_watch_results(registry):
    model = TinyLM(layers=1, width=8, heads=2, latents=8, k=2)
    tracer = Tracer(model, TinyTokenizer(), exact_dictionary(model))
    graph = tracer.trace("A", max_targets=0)
    before = dict(_publish._TENSORS)
    P.attach(model, None)
    assert _publish._TENSORS == before
    with _publish._publication_scope(model):
        P.attach(model, None)
        assert graph.publish() is graph
    assert set(_publish._TENSORS) == set(before)
    P.detach(model)
    assert not _publish._TENSORS


def test_other_watched_model_refuses_before_trace_work(registry, monkeypatch):
    model = TinyLM(layers=1, width=8, heads=2, latents=8, k=2)
    tracer = Tracer(model, TinyTokenizer(), exact_dictionary(model))
    P.attach(torch.nn.Linear(2, 2), None)
    monkeypatch.setattr(tracer, "_trace", lambda *args: pytest.fail("A foreign model reached trace work."))
    with pytest.raises(RuntimeError) as caught:
        tracer.trace("A")
    assert caught.value.code == "interpret.watch_changed"
    assert not _publish._TENSORS


@pytest.mark.parametrize("kind", ["graph", "examples"])
def test_watch_change_after_tensor_preparation_refuses_whole_group(registry, monkeypatch, kind):
    model, replacement = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    P.attach(model, None)
    entered, resume, changed = (threading.Event() for _ in range(3))
    errors, changes = [], []
    publish_many = _publish._publish_many

    def commit(values):
        if "custom.current" not in values:
            entered.set()
            assert resume.wait(5)
        publish_many(values)

    monkeypatch.setattr(_publish, "_publish_many", commit)

    def run():
        try:
            if kind == "graph":
                empty_graph().publish()
            else:
                InterpretationService(model=model)._publish_examples([])
        except BaseException as exc:
            errors.append(exc)

    def replace():
        try:
            P.attach(replacement, None)
            publish("custom.current", torch.ones(1))
            changed.set()
        except BaseException as exc:
            changes.append(exc)

    worker, changer = threading.Thread(target=run), threading.Thread(target=replace)
    worker.start()
    try:
        assert entered.wait(5), errors
        changer.start()
        assert changed.wait(5), "Tensor preparation retained the program lock."
    finally:
        resume.set()
        worker.join(6)
        if changer.ident is not None:
            changer.join(6)
    assert not worker.is_alive() and not changer.is_alive()
    assert not changes
    assert len(errors) == 1 and getattr(errors[0], "code", None) == "interpret.watch_changed"
    assert set(_publish._TENSORS) == {"custom.current"}
