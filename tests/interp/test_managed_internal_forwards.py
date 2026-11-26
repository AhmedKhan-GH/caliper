"""Managed actions and fork state across internal model calls."""
from copy import deepcopy
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

import caliper
from caliper import _managed, _wire
from caliper.interp import Tracer
from caliper.interp._index import FeatureIndex, _forward, build_index
from caliper.interp._output import internal_forward, is_internal_forward
from caliper.interp._service import InterpretationService, register_service, unregister_service
from .fixture import make_fixture
from .test_dictionary import write_fixture
from .test_managed_startup import OVERRIDE, SAVED, environment, manual_logits
from .test_service import Transport
from .test_trace import exact_dictionary


def test_trace_defers_queued_action(environment, monkeypatch, tmp_path):
    model, tokenizer = make_fixture(layers=1, width=8, heads=2, latents=16, k=2)
    dictionary = exact_dictionary(model)
    managed = _managed.ManagedRun("script", str(tmp_path), str(tmp_path / "link"))
    monkeypatch.setattr(_managed, "current", lambda: managed)
    with caliper.watch(model, activations=False):
        service = environment.intervention_service
        service.configure(dictionary=dictionary, tokenizer=tokenizer)
        calls = []

        def apply():
            calls.append(environment.n)

        service.set(SAVED[0])
        action = caliper.action("apply", apply)
        transport = Transport()
        route = environment.attach(transport)
        try:
            request = _wire.Control(1, "run", name="apply")
            assert route.on_control(None, request.to_meta())["status"] == "queued"
            tracer = Tracer(model, tokenizer, dictionary)
            step = environment.n
            graph = tracer.trace("AB", max_targets=2)
            assert graph.nodes and not calls and action.queued
            assert environment.n == step and not service.specifications()
            assert not is_internal_forward()
            # Standalone publication still needs the caller's explicit boundary.
            caliper.step()
            assert calls == [step + 1] and not action.queued
            assert service.specifications() == [dict(SAVED[0], op="add")]
        finally:
            action.remove()
            route.detach()


@pytest.mark.parametrize("operation", ["trace", "service_trace", "index_capture", "decoder_capture"])
@pytest.mark.parametrize("mode", ["saved", "override", "both"])
@pytest.mark.parametrize("training", [False, True])
def test_internal_call_retains_fork_restore(environment, monkeypatch, tmp_path, operation, mode, training):
    model, tokenizer = make_fixture(layers=1, width=8, heads=2, latents=16, k=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1) if training else None
    dictionary = exact_dictionary(model)
    baseline = manual_logits(model, tokenizer, [])
    expected = manual_logits(model, tokenizer, SAVED[:1] if mode == "saved" else [OVERRIDE])
    checkpoint = None
    if mode != "override":
        parent_service = environment.interventions_on(model)
        parent_service.configure(dictionary=dictionary, tokenizer=tokenizer)
        parent_service.set(SAVED[0])
        environment.step(5)
        parent = _managed.ManagedRun("parent", str(tmp_path))
        parent._pkg = caliper
        checkpoint = parent.checkpoint(model, optimizer)
        parent_service.remove()
    managed = _managed.ManagedRun("child", str(tmp_path), str(tmp_path / "link"), resume=checkpoint,
                                  interventions=[] if mode == "saved" else [OVERRIDE])
    monkeypatch.setattr(_managed, "current", lambda: managed)
    with caliper.watch(model, optimizer, activations=False, scene=False):
        service = environment.intervention_service
        service.configure(dictionary=dictionary, tokenizer=tokenizer)
        restores = []
        restore = service.load_state_dict

        def load(*args, **kwargs):
            restores.append(True)
            return restore(*args, **kwargs)

        monkeypatch.setattr(service, "load_state_dict", load)
        overrides = []
        apply_overrides = service.apply_overrides

        def apply(specifications):
            overrides.append(True)
            return apply_overrides(specifications)

        monkeypatch.setattr(service, "apply_overrides", apply)
        first, pending, step = managed._first, managed._restored_interventions, environment.n
        saved = deepcopy(pending) if pending is not _managed.NONE else pending
        if operation == "trace":
            graph = Tracer(model, tokenizer, dictionary).trace("AB", max_targets=2)
            assert graph.nodes
        elif operation == "service_trace":
            interpretation = InterpretationService(model=model, tokenizer=tokenizer, dictionary=dictionary)
            register_service(interpretation, controls=environment)
            replies = []
            try:
                interpretation.submit({"version": 1, "method": "trace.run",
                                       "params": {"prompt": "AB", "max_targets": 2}},
                                      lambda *reply: replies.append(reply))
                assert interpretation.step(step) == 1
                assert replies[0][0] == "ok", replies
            finally:
                unregister_service(interpretation, controls=environment)
        else:
            inputs = tokenizer("AB")
            forward_model = model
            if operation == "decoder_capture":
                # Exercise decoder dispatch with the same random fixture.
                forward_model = torch.nn.Module()
                forward_model.config = SimpleNamespace(model_type="qwen3_5_text")
                forward_model.model = model
            with torch.no_grad():
                result = _forward(forward_model, inputs["input_ids"], torch.ones_like(inputs["input_ids"]))
            torch.testing.assert_close(result.logits[0, -1], baseline, rtol=0, atol=1e-12)
        assert managed._first is first and managed._restored_interventions is pending
        assert pending == saved and not restores and not overrides and not service.specifications()
        assert environment.n == step and not is_internal_forward()
        for offset in (1, 2):
            with torch.no_grad():
                actual = model(**tokenizer("AB")).logits[0, -1]
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
            if optimizer is not None:
                assert environment.n == step + offset - 1
                optimizer.step()
            assert environment.n == step + offset
            assert restores == ([] if mode == "override" else [True])
            assert overrides == ([] if mode == "saved" else [True])
            assert managed._first is None and managed._restored_interventions is _managed.NONE
    assert all(not module._forward_pre_hooks and not module._forward_hooks for module in model.modules())


@pytest.mark.parametrize("operation", ["trace", "service_trace", "index_capture", "decoder_capture"])
@pytest.mark.parametrize("outer", [False, True])
def test_internal_exception_restores_scope(environment, monkeypatch, tmp_path, operation, outer):
    model, tokenizer = make_fixture(layers=1, width=8, heads=2, latents=16, k=2)
    dictionary = exact_dictionary(model)
    expected = manual_logits(model, tokenizer, [OVERRIDE])
    managed = _managed.ManagedRun("child", str(tmp_path), str(tmp_path / "link"), interventions=[OVERRIDE])
    monkeypatch.setattr(_managed, "current", lambda: managed)
    with caliper.watch(model, activations=False):
        first, step = managed._first, environment.n
        observed = []

        def fail(module, args):
            observed.append(is_internal_forward())
            with internal_forward():
                raise RuntimeError("fixture forward failure")

        hook = model.layers[0].mlp.register_forward_pre_hook(fail)
        try:
            with internal_forward() if outer else nullcontext():
                if operation == "service_trace":
                    service = InterpretationService(model=model, tokenizer=tokenizer, dictionary=dictionary)
                    register_service(service, controls=environment)
                    replies = []
                    try:
                        service.submit({"version": 1, "method": "trace.run", "params": {"prompt": "AB"}},
                                       lambda *reply: replies.append(reply))
                        assert service.step(step) == 1
                        assert replies[0][0] == "error"
                        assert replies[0][2]["message"] == "fixture forward failure"
                    finally:
                        unregister_service(service, controls=environment)
                else:
                    with pytest.raises(RuntimeError, match="fixture forward failure"):
                        if operation == "trace":
                            Tracer(model, tokenizer, dictionary).trace("AB", max_targets=2)
                        else:
                            ids = tokenizer("AB")["input_ids"]
                            forward_model = model
                            if operation == "decoder_capture":
                                forward_model = torch.nn.Module()
                                forward_model.config = SimpleNamespace(model_type="qwen3_5_text")
                                forward_model.model = model
                            _forward(forward_model, ids, torch.ones_like(ids))
                assert is_internal_forward() is outer
        finally:
            hook.remove()
        assert observed == [True] and not is_internal_forward()
        assert managed._first is first and environment.n == step
        assert not environment.intervention_service.specifications()
        with torch.no_grad():
            actual = model(**tokenizer("AB")).logits[0, -1]
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
        assert managed._first is None and environment.n == step + 1
    assert all(not module._forward_pre_hooks and not module._forward_hooks for module in model.modules())


def test_trace_retains_concurrent_state_error(environment, monkeypatch, tmp_path):
    model, tokenizer = make_fixture(layers=1, width=8, heads=2, latents=16, k=2)
    dictionary = exact_dictionary(model)
    managed = _managed.ManagedRun("script", str(tmp_path), str(tmp_path / "link"))
    monkeypatch.setattr(_managed, "current", lambda: managed)
    with caliper.watch(model, activations=False), ThreadPoolExecutor(max_workers=1) as worker:
        service = environment.intervention_service
        service.configure(dictionary=dictionary, tokenizer=tokenizer)
        tracer = Tracer(model, tokenizer, dictionary)

        def change():
            assert not is_internal_forward()
            service.apply_overrides([OVERRIDE])

        def concurrent_change(module, args, output):
            assert is_internal_forward()
            worker.submit(change).result(timeout=5)

        hook = model.register_forward_hook(concurrent_change)
        try:
            with pytest.raises(RuntimeError, match="intervention state changed during the trace forward"):
                tracer.trace("AB", max_targets=2)
        finally:
            hook.remove()
        assert not tracer._cache and tracer._forward_cache is None
        assert environment.n == 0 and not is_internal_forward()
        assert service.specifications() == [dict(OVERRIDE, op="add")]
        with torch.no_grad():
            model(**tokenizer("AB"))
        assert environment.n == 1


def test_index_build_and_examples_preserve_pending_work(environment, monkeypatch, tmp_path):
    model, tokenizer = make_fixture(layers=1, width=8, heads=2, latents=16, k=2)
    folder = tmp_path / "dictionary"
    write_fixture(folder, model)
    dictionary = exact_dictionary(model)
    expected = manual_logits(model, tokenizer, [OVERRIDE])
    managed = _managed.ManagedRun("child", str(tmp_path), str(tmp_path / "link"), interventions=[OVERRIDE])
    monkeypatch.setattr(_managed, "current", lambda: managed)
    with caliper.watch(model, activations=False):
        first, step = managed._first, environment.n
        calls = []
        action = caliper.action("observe", lambda: calls.append(environment.n))
        route = environment.attach(Transport())
        try:
            request = _wire.Control(1, "run", name="observe")
            assert route.on_control(None, request.to_meta())["status"] == "queued"
            build_index(folder, [("fixture", [65, 66, 67])], model=model, tokenizer=tokenizer,
                        dictionary=dictionary, provenance={"partition": "fixture"}, context_length=3)
            assert managed._first is first and environment.n == step and not calls
            with FeatureIndex(folder, model=model, tokenizer=tokenizer, dictionary=dictionary) as index:
                feature = int(index.counts[0].argmax())
                rows = index.examples(0, feature)
                assert rows and managed._first is first and environment.n == step and not calls
                index.clear_cache()
                service = InterpretationService(model=model, tokenizer=tokenizer, dictionary=dictionary, index=index)
                register_service(service, controls=environment)
                replies = []
                try:
                    service.submit({"version": 1, "method": "feature.examples",
                                    "params": {"layer": 0, "feature": feature}},
                                   lambda *reply: replies.append(reply))
                    assert service.step(step) == 1
                    assert replies[0][0] == "ok", replies
                    assert replies[0][1]["examples"] == rows
                finally:
                    unregister_service(service, controls=environment)
            assert managed._first is first and environment.n == step and not calls
            assert action.queued and not is_internal_forward()
            with torch.no_grad():
                actual = model(**tokenizer("AB")).logits[0, -1]
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
            assert managed._first is None and calls == [step + 1] and not action.queued
        finally:
            action.remove()
            route.detach()
