"""Request admission and borrowed guard lifetimes with small CPU resources."""
from types import SimpleNamespace
import threading

import pytest
import torch

from caliper._controls import Controls
from caliper.interp import Tracer, _dictionary, _frozen, _index, _memory
from caliper.interp._memory import CleanStop
from caliper.interp._service import InterpretationService, register_service, unregister_service
from .fixture import TinyLM, TinyTokenizer
from .test_dictionary import write_fixture
from .test_memory_phases import process_guard


class RequestGuard:
    def __init__(self, limit=1_000_000_000):
        self.readings = {"current": 1_000, "peak": 1_000}
        self._guard = process_guard(limit, self.readings)
        self.stages = []
        self.failure = None
        self.at_stage = lambda stage: None

    def __getattr__(self, name):
        return getattr(self._guard, name)

    def check(self, stage):
        self.stages.append(stage)
        self.at_stage(stage)
        if self.failure is not None:
            raise self.failure
        self._guard.check(stage)

    def _check_interval(self, stage):
        self.check(stage)


@pytest.fixture
def resources(tmp_path, monkeypatch):
    readings = {"current": 1_000, "peak": 1_000}

    def guard(limit, device, **kwargs):
        assert torch.device(device).type == "cpu"
        return process_guard(int(limit * 1e9), readings)

    for module in (_dictionary, _index, _memory):
        monkeypatch.setattr(module, "MemoryGuard", guard)
    monkeypatch.setattr(_frozen, "ProcessMemoryGuard", guard)
    monkeypatch.setattr("caliper.interp._publish._TENSORS", {})
    model, _ = write_fixture(tmp_path, TinyLM(layers=2, width=8, heads=2, latents=8, k=2))
    tokenizer, controls = TinyTokenizer(), Controls()
    dictionary = _dictionary.load_dictionary(tmp_path, model=model)
    _index.build_index(tmp_path, [("fixture", [65, 66, 67, 68] * 2)], model=model,
                       tokenizer=tokenizer, dictionary=dictionary, provenance={"partition": "fixture"})
    index = _index.FeatureIndex(tmp_path, model=model, tokenizer=tokenizer, dictionary=dictionary,
                                memory_limit_gb=2)
    tracer = Tracer(model, tokenizer, dictionary, controls=controls, index=index, memory_limit_gb=2)
    case = SimpleNamespace(model=model, tokenizer=tokenizer, dictionary=dictionary, index=index,
                           tracer=tracer, controls=controls, readings=readings,
                           index_memory=index.memory, trace_guard=tracer.guard)
    yield case
    index.close()


def service_for(case, source, **options):
    if source == "supplied":
        args = {"index": case.index, "tracer": case.tracer}
    elif source == "factory":
        args = {"index_factory": lambda: case.index, "tracer_factory": lambda: case.tracer}
    else:
        args = {"index": case.index}
    return register_service(controls=case.controls, **args, **options)


def submit(service, case, method, **overrides):
    params = ({"prompt": "AB", "max_targets": 1, "max_nodes": 32} if method == "trace.run" else
              {} if method == "feature.dictionary" else {"layer": 0, "feature": int(case.index.counts[0].argmax())})
    replies = []
    service.submit({"version": 1, "method": method, "params": {**params, **overrides}},
                   lambda *args: replies.append(args))
    return replies


def restored(case):
    assert case.index.memory is case.index_memory
    assert case.tracer.guard is case.trace_guard
    assert case.index.memory.limit == case.tracer.guard.limit == 2_000_000_000
    if case.tracer._forward_cache is not None:
        assert case.tracer._forward_cache[1].guard is case.trace_guard
    assert not case.index._closed


@pytest.mark.parametrize("source", ["supplied", "factory", "default"])
def test_real_requests_use_lower_guard_and_restore_shared_owners(resources, source):
    case, lower = resources, RequestGuard()
    service = service_for(case, source, memory=lower, memory_limit_gb=1.5)
    try:
        for method in ("feature.examples", "feature.show", "feature.dictionary", "trace.run"):
            replies = submit(service, case, method)
            assert service.step() == 1
            assert replies[0][0] == "ok", replies
            assert replies[0][1]
            restored(case)
        assert service.index is case.index and service.dictionary is case.dictionary
        assert service.model is case.model and service.tokenizer is case.tokenizer
        assert {"example windows admission", "feature labels admission", "dictionary qualification",
                "dictionary admission", "target batch admission"} <= set(lower.stages)
        baseline = service.graph
        replies = submit(service, case, "trace.run")
        service.step()
        assert replies[0][0] == "ok" and service.graph is baseline
        assert "cached publication" in lower.stages
        if source != "default":
            assert service.tracer is case.tracer
        # Reuse the frozen path with a different graph request.
        forward = service.tracer._forward_cache[1]
        replies = submit(service, case, "trace.run", max_nodes=31)
        service.step()
        assert replies[0][0] == "ok" and service.tracer._forward_cache[1] is forward
        restored(case)
    finally:
        unregister_service(service, controls=case.controls)
    restored(case)


@pytest.mark.parametrize("source", ["supplied", "factory"])
@pytest.mark.parametrize("method", ["feature.examples", "feature.show", "feature.dictionary", "trace.run"])
@pytest.mark.parametrize("failure", ["stop", "historical"])
def test_service_stop_precedes_resources_and_drains_queue(resources, source, method, failure):
    case, lower = resources, RequestGuard()
    if failure == "stop":
        lower.failure = CleanStop("explicit request stop")
    else:
        lower.readings.update(current=1_000, peak=1_000_000_001)
    service = service_for(case, source, memory=lower)
    replies = submit(service, case, method)
    queued = submit(service, case, "feature.dictionary")
    try:
        with pytest.raises(CleanStop) as caught:
            service.step()
        assert service.failure is caught.value
        assert replies[0][2]["code"] == queued[0][2]["code"] == "interpret.stopped"
        assert not service.queue
        assert not case.index._cache and not case.index._summaries and not case.tracer._cache
        if source == "factory":
            assert service.index is service.tracer is None
        if failure == "stop":
            assert caught.value is lower.failure
        restored(case)
    finally:
        unregister_service(service, controls=case.controls)


@pytest.mark.parametrize("source", ["supplied", "factory", "default"])
@pytest.mark.parametrize("method,limit", [("feature.examples", .5), ("feature.show", .06),
                                         ("feature.dictionary", .004), ("trace.run", .004)])
@pytest.mark.parametrize("supplied_memory", [False, True])
def test_request_reserves_enforce_service_cap(resources, source, method, limit, supplied_memory):
    case, lower = resources, RequestGuard(int(limit * 1e9))
    service = service_for(case, source, **({"memory": lower, "memory_limit_gb": 1.5} if supplied_memory else
                                          {"memory_limit_gb": limit}))
    replies = submit(service, case, method)
    try:
        with pytest.raises(CleanStop, match="allocation exceeds"):
            service.step()
        assert replies[0][2]["code"] == "interpret.stopped"
        assert not case.index._summaries and not case.tracer._cache
        assert all("examples" not in entry for entry in case.index._cache.values())
        restored(case)
    finally:
        unregister_service(service, controls=case.controls)


@pytest.mark.parametrize("method,stage,limit", [
    ("feature.examples", "example windows admission", 500_000_000),
    ("feature.show", "feature labels admission", 60_000_000),
    ("trace.run", "target batch admission", 500_000_000),
])
def test_dynamic_lower_guard_controls_real_allocation_reserves(resources, method, stage, limit):
    case, lower = resources, RequestGuard()

    def shrink(current):
        if current == stage:
            lower._guard.limit = limit

    lower.at_stage = shrink
    service = service_for(case, "supplied", memory=lower)
    replies = submit(service, case, method)
    try:
        with pytest.raises(CleanStop):
            service.step()
        assert stage in lower.stages and replies[0][2]["code"] == "interpret.stopped"
        restored(case)
    finally:
        unregister_service(service, controls=case.controls)


@pytest.mark.parametrize("method,stage", [
    ("feature.examples", "example windows complete"),
    ("feature.show", "feature label examples"),
    ("feature.dictionary", "index qualification file"),
    ("trace.run", "target backward"),
])
def test_stage_stop_reaches_service_queue_and_restores_owners(resources, method, stage):
    case, lower = resources, RequestGuard()
    failure = CleanStop("request stage stop")

    def stop(current):
        if current == stage:
            raise failure

    lower.at_stage = stop
    service = service_for(case, "factory", memory=lower)
    replies = submit(service, case, method)
    queued = submit(service, case, "feature.dictionary")
    try:
        with pytest.raises(CleanStop) as caught:
            service.step()
        assert caught.value is service.failure is failure
        assert stage in lower.stages
        assert replies[0][2]["code"] == queued[0][2]["code"] == "interpret.stopped"
        assert not case.tracer._cache and not case.index._summaries
        assert all("examples" not in entry for entry in case.index._cache.values())
        restored(case)
    finally:
        unregister_service(service, controls=case.controls)


def test_default_service_trace_retains_twelve_gb_budget(resources):
    case = resources
    service = register_service(controls=case.controls, model=case.model, tokenizer=case.tokenizer,
                               dictionary=case.dictionary)
    replies = submit(service, case, "trace.run", max_targets=0)
    try:
        service.step()
        assert replies[0][0] == "ok"
        assert service.tracer.guard.limit == 12_000_000_000
        assert service.memory is service._process_memory is service._memory_budget is None
    finally:
        unregister_service(service, controls=case.controls)


@pytest.mark.parametrize("source", ["supplied", "factory"])
def test_service_budget_preserves_owned_guard_stops(resources, source):
    case, lower = resources, RequestGuard()
    case.index_memory.limit = 500
    service = service_for(case, source, memory=lower)
    replies = submit(service, case, "feature.examples")
    try:
        with pytest.raises(CleanStop):
            service.step()
        assert replies[0][2]["code"] == "interpret.stopped"
        assert case.index.memory is case.index_memory and case.index.memory.limit == 500
        assert case.tracer.guard is case.trace_guard
    finally:
        unregister_service(service, controls=case.controls)


@pytest.mark.parametrize("source", ["supplied", "factory"])
def test_cached_trace_stop_restores_frozen_path_and_drains_queue(resources, source):
    case, lower = resources, RequestGuard()
    baseline = case.tracer.trace("AB", max_targets=1, max_nodes=32)
    forward = case.tracer._forward_cache[1]
    failure = CleanStop("cached path stop")

    def stop(stage):
        if stage == "source nodes":
            raise failure

    lower.at_stage = stop
    service = service_for(case, source, memory=lower)
    replies = submit(service, case, "trace.run", max_nodes=31)
    queued = submit(service, case, "feature.dictionary")
    try:
        with pytest.raises(CleanStop) as caught:
            service.step()
        assert caught.value is failure and service.failure is failure
        assert replies[0][2]["code"] == queued[0][2]["code"] == "interpret.stopped"
        assert case.tracer._forward_cache[1] is forward
        restored(case)
    finally:
        unregister_service(service, controls=case.controls)
    assert case.tracer.trace("AB", max_targets=1, max_nodes=32) is baseline


def test_cached_examples_reserve_publication_before_allocation(resources, monkeypatch):
    from caliper.interp import _publish
    case, lower = resources, RequestGuard()
    service = service_for(case, "supplied", memory=lower)
    try:
        replies = submit(service, case, "feature.examples")
        service.step()
        assert replies[0][0] == "ok"
        published = dict(_publish._TENSORS)
        lower._guard.limit = 64_000
        monkeypatch.setattr(torch, "zeros", lambda *a, **kw: pytest.fail("unadmitted publication"))
        replies = submit(service, case, "feature.examples")
        with pytest.raises(CleanStop, match="example publication allocation"):
            service.step()
        assert replies[0][2]["code"] == "interpret.stopped"
        assert all(_publish._TENSORS[name] is value for name, value in published.items())
        restored(case)
    finally:
        unregister_service(service, controls=case.controls)


@pytest.mark.parametrize("method", ["feature.show", "feature.examples"])
@pytest.mark.parametrize("failure", ["stop", "limit"])
def test_default_service_checks_cached_resource_guard_before_publication(resources, monkeypatch, method, failure):
    from caliper.interp import _publish
    case = resources
    service = service_for(case, "default")
    try:
        replies = submit(service, case, method)
        service.step()
        assert replies[0][0] == "ok"
        published = dict(_publish._TENSORS)
        stopped = CleanStop("cached resource stop")

        def stop(stage):
            raise stopped

        if failure == "stop":
            monkeypatch.setattr(case.index_memory, "check", stop)
        else:
            case.index_memory.limit = 500
        monkeypatch.setattr(torch, "zeros", lambda *a, **kw: pytest.fail("publication followed resource stop"))
        replies = submit(service, case, method)
        queued = submit(service, case, "feature.dictionary")
        with pytest.raises(CleanStop) as caught:
            service.step()
        assert service.failure is caught.value
        assert replies[0][2]["code"] == queued[0][2]["code"] == "interpret.stopped"
        assert all(_publish._TENSORS[name] is value for name, value in published.items())
        assert not service.queue
        assert case.index.memory is case.index_memory
    finally:
        case.index_memory.limit = 2_000_000_000
        unregister_service(service, controls=case.controls)


def test_default_service_reserves_cached_example_publication(resources, monkeypatch):
    case = resources
    service = service_for(case, "default")
    try:
        replies = submit(service, case, "feature.examples")
        service.step()
        assert replies[0][0] == "ok"
        case.index_memory.limit = 64_000
        monkeypatch.setattr(torch, "zeros", lambda *a, **kw: pytest.fail("unadmitted cached publication"))
        replies = submit(service, case, "feature.examples")
        with pytest.raises(CleanStop, match="example publication allocation"):
            service.step()
        assert replies[0][2]["code"] == "interpret.stopped"
    finally:
        case.index_memory.limit = 2_000_000_000
        unregister_service(service, controls=case.controls)


def test_factory_result_retains_stop_before_first_use(resources):
    case, lower = resources, RequestGuard()
    failure = CleanStop("factory result stop")

    def factory():
        lower.failure = failure
        return case.index

    service = register_service(controls=case.controls, index_factory=factory, memory=lower)
    replies = submit(service, case, "feature.examples")
    try:
        with pytest.raises(CleanStop) as caught:
            service.step()
        assert caught.value is failure and service.index is case.index
        assert replies[0][2]["code"] == "interpret.stopped"
        assert not case.index._cache
        restored(case)
    finally:
        unregister_service(service, controls=case.controls)


def test_ordinary_refusal_restores_guard_and_allows_another_request(resources, monkeypatch):
    case, lower = resources, RequestGuard()
    service = service_for(case, "supplied", memory=lower)

    def refuse(*args):
        assert case.index.memory.limit == lower.limit
        raise ValueError("query refusal")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(case.index, "show", refuse)
            replies = submit(service, case, "feature.show")
            service.step()
        assert replies[0][2]["code"] == "interpret.refused" and service.failure is None
        restored(case)
        replies = submit(service, case, "feature.show")
        service.step()
        assert replies[0][0] == "ok"
        restored(case)
    finally:
        unregister_service(service, controls=case.controls)


def test_service_rejects_larger_family_budget_before_existing_resources(resources):
    case = resources
    service = service_for(case, "supplied", memory_limit_gb=28)
    replies = submit(service, case, "feature.examples")
    try:
        service.step()
        assert replies[0][2]["code"] == "interpret.refused"
        assert "at most 12 GB" in replies[0][2]["message"]
        assert not case.index._cache
        restored(case)
    finally:
        unregister_service(service, controls=case.controls)


@pytest.mark.parametrize("mismatch", ["service", "index", "tracer"])
@pytest.mark.parametrize("source", ["supplied", "factory"])
def test_service_refuses_device_mismatch_before_resource_work(resources, source, mismatch):
    case, lower = resources, RequestGuard()
    target = {"service": lower._guard, "index": case.index_memory, "tracer": case.trace_guard}[mismatch]
    target.device = torch.device("meta")
    service = service_for(case, source, memory=lower)
    replies = submit(service, case, "trace.run" if mismatch == "tracer" else "feature.examples")
    try:
        service.step()
        assert replies[0][2]["code"] == "interpret.device_mismatch"
        assert not case.index._cache and not case.tracer._cache
        restored(case)
    finally:
        target.device = torch.device("cpu")
        unregister_service(service, controls=case.controls)


def test_shared_resource_refuses_overlap_and_restores_before_close(resources, monkeypatch):
    case, lower = resources, RequestGuard()
    service = service_for(case, "supplied", memory=lower)
    other = InterpretationService(index=case.index)
    entered, release, waiting, closed = (threading.Event() for _ in range(4))
    failures = []
    original = case.index.examples

    def examples(*args, **kwargs):
        assert case.index.memory.limit == lower.limit
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    def work():
        try:
            service.step()
        except BaseException as error:
            failures.append(error)

    wait = service.idle.wait

    def close_wait(timeout=None):
        waiting.set()
        return wait(timeout)

    def close():
        service.close()
        restored(case)
        closed.set()

    monkeypatch.setattr(case.index, "examples", examples)
    monkeypatch.setattr(service.idle, "wait", close_wait)
    replies = submit(service, case, "feature.examples")
    worker, closer = threading.Thread(target=work), threading.Thread(target=close)
    worker.start()
    try:
        assert entered.wait(5)
        overlap = submit(other, case, "feature.show")
        other.step()
        assert overlap[0][2]["code"] == "interpret.busy"
        closer.start()
        assert waiting.wait(5) and not closed.is_set()
    finally:
        release.set()
        worker.join(5)
        if closer.ident is not None:
            closer.join(5)
        unregister_service(service, controls=case.controls)
        other.close()
    assert not worker.is_alive() and not closer.is_alive() and not failures
    assert closed.is_set() and replies[0][0] == "ok"
    restored(case)
