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


@pytest.mark.parametrize("operation", ["trace", "cached_trace", "examples"])
def test_replaced_watch_refuses_late_publication(registry, monkeypatch, tmp_path, operation):
    from caliper.interp import _trace
    from caliper.interp._index import FeatureIndex, build_index
    from .test_dictionary import write_fixture

    model = TinyLM(layers=1, width=8, heads=2, latents=8, k=2)
    tokenizer, dictionary = TinyTokenizer(), exact_dictionary(model)
    tracer = Tracer(model, tokenizer, dictionary)
    P.attach(model, None)
    options = dict(prompt="A", max_targets=0, max_nodes=8, max_edges=8)
    entered, resume, changed = (threading.Event() for _ in range(3))
    errors, replies, changes = [], [], []
    index = None

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
        work = lambda: tracer.trace(**options)

    def run():
        try:
            work()
        except BaseException as exc:
            errors.append(exc)

    replacement = torch.nn.Linear(2, 2)

    def replace():
        try:
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
        if index is not None:
            service.close()
            index.close()
    assert not worker.is_alive() and not changer.is_alive()
    assert not changes
    assert set(_publish._TENSORS) == {"custom.current"}
    if operation == "examples":
        assert not errors
        assert replies[0][0] == "error", replies
        assert replies[0][2]["code"] == "interpret.watch_changed"
    else:
        assert len(errors) == 1
        assert getattr(errors[0], "code", None) == "interpret.watch_changed"
