"""Small CPU checks for finite trace inputs and request recovery."""
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import Tracer, _frozen, _publish, _trace
from .fixture import TinyLM, TinyTokenizer
from .test_trace import exact_dictionary


def resources():
    model = TinyLM(layers=2, width=8, heads=2, latents=8, k=2)
    return model, TinyTokenizer(), exact_dictionary(model)


@pytest.mark.parametrize("name", ["W_enc", "b_enc", "W_dec", "b_dec"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_dictionary_refuses_before_forward_and_cache(monkeypatch, name, bad):
    model, tokenizer, dictionary = resources()
    tracer = Tracer(model, tokenizer, dictionary)
    monkeypatch.setattr(_publish, "_TENSORS", {})
    tracer.trace("A", max_targets=0)
    published = dict(_publish._TENSORS)
    # An alias write leaves the version counter unchanged.
    value = getattr(dictionary.layers[1], name)
    version = value._version
    value.data.view(-1)[-1] = bad
    assert value._version == version
    monkeypatch.setattr(model, "forward", lambda *a, **k: pytest.fail("nonfinite dictionary reached the model"))
    message = f"dictionary layer 1 {name} must be finite"
    with pytest.raises(ValueError, match=message):
        tracer.trace("A", max_targets=0)
    assert not tracer._cache and tracer._forward_cache is None
    with pytest.raises(ValueError, match=message):
        Tracer(model, tokenizer, dictionary)
    with pytest.raises(ValueError, match=message):
        _frozen.FrozenPath(model, torch.tensor([[1, 2]]), dictionary)
    assert all(_publish._TENSORS[key] is value for key, value in published.items())
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(_trace, "time", SimpleNamespace(monotonic=lambda: clock.now))
    return clock


def test_forward_timeout_stops_at_first_layer_and_removes_hooks(monkeypatch, clock):
    model, tokenizer, dictionary = resources()
    tracer = Tracer(model, tokenizer, dictionary)
    def expire(*args):
        clock.now += 20
    first = model.layers[0].register_forward_hook(expire)
    second = model.layers[1].register_forward_pre_hook(lambda *a: pytest.fail("expired trace reached the next layer"))
    try:
        with pytest.raises(_trace.TraceTimeout):
            tracer.trace("A", max_targets=0)
        assert len(model.layers[0]._forward_hooks) == 1
    finally:
        first.remove()
        second.remove()
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    assert not tracer._cache and tracer._forward_cache is None
    assert tracer.trace("A", max_targets=0).completeness["max_absolute_residual"] < 1e-9


def test_backward_timeout_precedes_edge_products_and_recovers(monkeypatch, clock):
    model, tokenizer, dictionary = resources()
    tracer = Tracer(model, tokenizer, dictionary)
    grad = torch.autograd.grad
    def expire(*args, **kwargs):
        result = grad(*args, **kwargs)
        clock.now += 20
        return result
    with monkeypatch.context() as patch:
        patch.setattr(torch.autograd, "grad", expire)
        patch.setattr(torch, "bmm", lambda *a: pytest.fail("expired backward reached edge products"))
        with pytest.raises(_trace.TraceTimeout):
            tracer.trace("A", max_targets=2)
    assert not tracer._cache and tracer._forward_cache is None
    assert tracer.trace("A", max_targets=2).completeness["max_absolute_residual"] < 1e-9


@pytest.mark.parametrize("stage", ["graph labels admission", "graph labels complete"])
def test_indexless_labels_check_request_deadline(clock, stage):
    from caliper.interp._feature_labels import label_graph_nodes
    deadline = _trace._Deadline()
    def check(current):
        if current == stage:
            clock.now += 20
        deadline(current)
    nodes = [{"kind": "feature", "layer": 0, "feature": 1, "label": "prior"}]
    with pytest.raises(_trace.TraceTimeout):
        label_graph_nodes(nodes, check=check)
    assert nodes[0]["label"] == "prior"


@pytest.mark.parametrize("stage", [
    "finite block", "model_identity", "record layer 0", "freeze layer 0",
    "frozen layer 0", "replacement layer 0", "source layer 1",
    "target backward", "target batch complete", "graph influence",
    "graph pruning", "graph construction", "graph publication", "cached publication",
])
def test_deadline_stages_preserve_prior_publication_and_recover(monkeypatch, clock, stage):
    model, tokenizer, dictionary = resources()
    tracer = Tracer(model, tokenizer, dictionary)
    monkeypatch.setattr(_publish, "_TENSORS", {})
    prior = tracer.trace("A", max_targets=2)
    forward_cache = tracer._forward_cache
    published = dict(_publish._TENSORS)
    cache = dict(tracer._cache)
    deadline = _trace._Deadline.__call__
    seen = []
    def check(self, current):
        seen.append(current)
        if current == stage:
            clock.now += 20
        deadline(self, current)
    prompt = "A" if stage == "cached publication" else "B"
    with monkeypatch.context() as patch:
        patch.setattr(_trace._Deadline, "__call__", check)
        with pytest.raises(_trace.TraceTimeout):
            tracer.trace(prompt, max_targets=2)
    assert seen[-1] == stage
    assert tracer._cache == cache
    assert all(_publish._TENSORS[key] is value for key, value in published.items())
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    assert tracer._forward_cache is None or tracer._forward_cache is forward_cache
    recovered = tracer.trace(prompt, max_targets=2)
    assert recovered.completeness["max_absolute_residual"] < 1e-9
    assert tracer.trace("A", max_targets=2) is prior


@pytest.mark.parametrize("indexed,stage", [
    (False, "record layer 0"), (True, "record layer 0"),
    (False, "target backward"), (True, "target backward"),
    (False, "graph publication"), (True, "graph publication"),
    (False, "cached publication"), (True, "cached publication"),
    (True, "feature label chunk"), (True, "feature labels complete"),
])
def test_service_timeout_preserves_queue_status_and_recovery(monkeypatch, tmp_path, clock, indexed, stage):
    from caliper import _controls, _interpret_protocol as protocol
    from caliper.interp._index import FeatureIndex, build_index
    from caliper.interp._service import InterpretationService, register_service, unregister_service
    from .test_dictionary import write_fixture
    from .test_service import Transport, call

    model, tokenizer, dictionary = resources()
    index = None
    if indexed:
        write_fixture(tmp_path, model)
        build_index(tmp_path, [("fixture", [65, 66, 67, 68] * 2)], model=model, tokenizer=tokenizer,
                    dictionary=dictionary, provenance={"partition": "fixture"}, context_length=4)
        index = FeatureIndex(tmp_path, model=model, tokenizer=tokenizer, dictionary=dictionary)
    controls, transport = _controls.Controls(), Transport()
    route = controls.attach(transport)
    tracer = Tracer(model, tokenizer, dictionary, controls=controls, index=index)
    service = register_service(InterpretationService(tracer=tracer, index=index), controls=controls)
    monkeypatch.setattr(_publish, "_TENSORS", {})
    try:
        assert call(route, "trace.run", {"prompt": "A", "max_targets": 2}, 1)["status"] == "queued"
        controls.checkpoint()
        assert transport.replies[-1].status == "ok"
        prior = service.graph
        published = dict(_publish._TENSORS)
        cache = dict(tracer._cache)
        if index is not None:
            index.clear_cache()
        deadline = _trace._Deadline.__call__
        def check(self, current):
            if current == stage:
                clock.now += 20
            deadline(self, current)
        prompt = "A" if stage == "cached publication" else "B"
        with monkeypatch.context() as patch:
            patch.setattr(_trace._Deadline, "__call__", check)
            assert call(route, "trace.run", {"prompt": prompt, "max_targets": 2}, 2)["status"] == "queued"
            assert call(route, "trace.read", {"node_limit": 1, "edge_limit": 1}, 3)["status"] == "queued"
            controls.checkpoint()
        terminal = {r.request_id: r for r in transport.replies if r.status != "queued"}
        assert terminal[2].status == "error"
        assert terminal[2].error["code"] == "interpret.trace_timeout"
        assert len(terminal[2].error["message"].encode()) <= 1024
        assert terminal[3].status == "ok" and terminal[3].result["trace_id"] == prior.trace_id
        status = call(route, "trace.status")["result"]
        assert protocol.bounded(status) and status["error"] == terminal[2].error
        assert service.failure is None and not service.closed and not service.pending
        assert service.graph is prior and tracer._cache == cache
        assert all(_publish._TENSORS[key] is value for key, value in published.items())
        if index is not None and stage in ("feature label chunk", "feature labels complete"):
            assert not index._summaries
        assert call(route, "trace.run", {"prompt": prompt, "max_targets": 2}, 4)["status"] == "queued"
        controls.checkpoint()
        assert transport.replies[-1].status == "ok"
        assert call(route, "trace.status")["result"]["state"] == "done"
    finally:
        unregister_service(service, controls=controls)
        route.detach()
        if index is not None:
            index.close()


def test_finite_scan_bounds_strided_views_without_copies(monkeypatch):
    value = torch.arange(144, dtype=torch.float64).reshape(12, 12).T
    finite = torch.isfinite
    sizes = []
    def bounded(block):
        sizes.append(block.numel())
        assert block.numel() <= 16
        assert block.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        return finite(block)
    monkeypatch.setattr(_frozen, "_FINITE_BLOCK_BYTES", 16 * 32)
    monkeypatch.setattr(torch, "isfinite", bounded)
    _frozen._require_finite(value, "fixture")
    assert sum(sizes) == value.numel()
    value[-1, -1] = float("nan")
    with pytest.raises(ValueError, match="fixture must be finite"):
        _frozen._require_finite(value, "fixture")


@pytest.mark.parametrize("method", ["run", "replacement"])
@pytest.mark.parametrize("location", ["later_read", "logits"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_forward_residual_never_reaches_edges_or_cache(monkeypatch, method, location, bad):
    model, tokenizer, dictionary = resources()
    tracer = Tracer(model, tokenizer, dictionary)
    original = getattr(_frozen.FrozenPath, method)
    def corrupt(path, *args, **kwargs):
        reads, final = original(path, *args, **kwargs)
        assert torch.isfinite(reads[0]).all()
        if location == "later_read":
            reads[1] = torch.full_like(reads[1], bad)
        else:
            final = torch.full_like(final, bad)
        return reads, final
    monkeypatch.setattr(_frozen.FrozenPath, method, corrupt)
    monkeypatch.setattr(_trace, "_DirectEffects", lambda *a: pytest.fail("nonfinite forward reached edges"))
    with pytest.raises(ValueError, match="forward residual must be finite"):
        tracer.trace("A", max_targets=0)
    assert not tracer._cache and tracer._forward_cache is None
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_reconstruction_residual_refuses_before_edges(monkeypatch, bad):
    model, tokenizer, dictionary = resources()
    original = _frozen._trace_reconstruction
    def corrupt(layer, features):
        value = original(layer, features)
        return torch.full_like(value, bad) if layer is dictionary.layers[1] else value
    monkeypatch.setattr(_frozen, "_trace_reconstruction", corrupt)
    with pytest.raises(ValueError, match="layer 1 reconstruction residual must be finite"):
        Tracer(model, tokenizer, dictionary).trace("A", max_targets=0)


@pytest.mark.parametrize("field", ["absolute_residual", "relative_residual"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_target_residual_refuses_before_influence_and_cache(monkeypatch, field, bad):
    model, tokenizer, dictionary = resources()
    tracer = Tracer(model, tokenizer, dictionary)
    original = _trace._DirectEffects.backward
    def corrupt(effects, targets):
        incoming, rows = original(effects, targets)
        assert len(rows) > 1 and rows[0][field] < 1e-9
        rows[1][field] = bad
        return incoming, rows
    monkeypatch.setattr(_trace._DirectEffects, "backward", corrupt)
    monkeypatch.setattr(_trace, "influence", lambda *a: pytest.fail("nonfinite residual reached influence"))
    with pytest.raises(ValueError, match=f"direct-effect {field} must be finite"):
        tracer.trace("A", max_targets=0)
    assert not tracer._cache and tracer._forward_cache is None
