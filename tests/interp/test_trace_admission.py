"""CPU checks for bounded finite scans and exact request admission."""
from contextlib import ExitStack
import math
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _frozen, _publish, _trace
from caliper.interp._memory import CleanStop
from caliper._controls import Controls
from .fixture import TinyTokenizer
from .test_trace import exact_dictionary
from .test_trace_safety import resources


def test_finite_batches_bound_masks_and_readbacks_for_strided_views(monkeypatch):
    value = torch.arange(324, dtype=torch.float64).reshape(18, 18).T[::2]
    finite, item = torch.isfinite, torch.Tensor.item
    sizes, readbacks, stages = [], [], []

    def bounded(block):
        sizes.append(block.numel())
        assert block.numel() <= 8
        assert block.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        return finite(block)

    def readback(result):
        readbacks.append(result.numel())
        return item(result)

    monkeypatch.setattr(_frozen, "_FINITE_BLOCK_BYTES", 8 * 32)
    monkeypatch.setattr(_frozen, "_FINITE_BATCH_BLOCKS", 4)
    monkeypatch.setattr(torch, "isfinite", bounded)
    monkeypatch.setattr(torch.Tensor, "item", readback)
    _frozen._require_finite(value, "fixture", stages.append)
    assert sum(sizes) == value.numel()
    assert readbacks == [1] * math.ceil(len(sizes) / 4)
    assert stages.count("finite block") == stages.count("finite block complete") == len(sizes)
    assert stages.count("finite block readback") == len(readbacks)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("position", [0, 7, 8, 16])
def test_finite_batches_reject_nonfinite_blocks_and_partial_tail(monkeypatch, bad, position):
    value = torch.ones(34, dtype=torch.float64)[::2]
    value[position] = bad
    monkeypatch.setattr(_frozen, "_FINITE_BLOCK_BYTES", 32)
    monkeypatch.setattr(_frozen, "_FINITE_BATCH_BLOCKS", 8)
    with pytest.raises(ValueError, match="fixture must be finite"):
        _frozen._require_finite(value, "fixture")


@pytest.mark.parametrize("error", [CleanStop, _trace.TraceTimeout])
def test_finite_batches_keep_checks_inside_batch(monkeypatch, error):
    seen = []

    def check(stage):
        seen.append(stage)
        if seen.count("finite block") == 3:
            raise error("fixture stop")

    monkeypatch.setattr(_frozen, "_FINITE_BLOCK_BYTES", 24)
    monkeypatch.setattr(torch.Tensor, "item", lambda value: pytest.fail("stop followed a readback"))
    with pytest.raises(error, match="fixture stop"):
        _frozen._require_finite(torch.ones(8), "fixture", check)


def test_finite_batches_check_deadline_after_readback(monkeypatch):
    item = torch.Tensor.item
    copied = False

    def readback(value):
        nonlocal copied
        result = item(value)
        copied = True
        return result

    def check(stage):
        if copied:
            assert stage == "finite block readback"
            raise _trace.TraceTimeout("fixture readback")

    monkeypatch.setattr(torch.Tensor, "item", readback)
    with pytest.raises(_trace.TraceTimeout, match="fixture readback"):
        _frozen._require_finite(torch.ones(8), "fixture", check)


def test_finite_batches_preserve_exact_graph(monkeypatch):
    model, tokenizer, dictionary = resources()
    tracer = _trace.Tracer(model, tokenizer, dictionary)
    monkeypatch.setattr(_frozen, "_FINITE_BLOCK_BYTES", 8 * 32)
    with monkeypatch.context() as patch:
        patch.setattr(_frozen, "_FINITE_BATCH_BLOCKS", 1)
        reference = tracer.trace("AB", max_targets=4)
    tracer._cache.clear()
    tracer._forward_cache = None
    batched = tracer.trace("AB", max_targets=4)
    assert reference.trace_id == batched.trace_id
    assert reference.nodes == batched.nodes and reference.edges == batched.edges
    assert reference.completeness == batched.completeness
    assert reference.tokens == batched.tokens
    for name, tensor in reference.tensors.items():
        assert torch.equal(tensor, batched.tensors[name])


def test_each_request_checks_content_once_before_both_caches(monkeypatch):
    model, tokenizer, dictionary = resources()
    tracer = _trace.Tracer(model, tokenizer, dictionary)
    counts = dict(identity=0, finite=0, content=0, forwards=0)
    identity, finite, content = _frozen.model_identity, _frozen._require_finite, _trace.weights_hash
    forward = model.forward
    standalone = False

    def checked_identity(value, **kwargs):
        assert value is model
        active = kwargs["memory"]
        if standalone:
            from caliper.interp._memory import BudgetMemory
            assert isinstance(active, BudgetMemory)
            assert active.memory is tracer.guard and active.limit == tracer.guard.limit
        else:
            assert active is tracer.guard
        counts["identity"] += 1
        return identity(value, **kwargs)

    def checked_finite(value, name, *args):
        if name.startswith("dictionary layer"):
            counts["finite"] += 1
        return finite(value, name, *args)

    def checked_content(value, **kwargs):
        assert value is dictionary
        counts["content"] += 1
        return content(value, **kwargs)

    def checked_forward(*args, **kwargs):
        counts["forwards"] += 1
        return forward(*args, **kwargs)

    monkeypatch.setattr(_frozen, "model_identity", checked_identity)
    monkeypatch.setattr(_frozen, "_require_finite", checked_finite)
    monkeypatch.setattr(_trace, "weights_hash", checked_content)
    monkeypatch.setattr(model, "forward", checked_forward)
    first = tracer.trace("AB", max_targets=4)
    assert counts == dict(identity=1, finite=8, content=1, forwards=1)
    assert tracer.trace("AB", max_targets=4) is first
    assert counts == dict(identity=2, finite=16, content=2, forwards=1)
    path = tracer._forward_cache[1]
    tracer.trace("AB", max_targets=4, max_nodes=511)
    assert tracer._forward_cache[1] is path
    assert counts == dict(identity=3, finite=24, content=3, forwards=1)
    tracer.trace("CD", max_targets=4)
    assert counts == dict(identity=4, finite=32, content=4, forwards=2)
    standalone = True
    _frozen.FrozenPath(model, tokenizer("AB")["input_ids"], dictionary, guard=tracer.guard)
    assert counts == dict(identity=5, finite=40, content=4, forwards=3)


def test_request_resource_stop_precedes_prompt_callback(monkeypatch):
    model, tokenizer, dictionary = resources()
    tracer = _trace.Tracer(model, tokenizer, dictionary)

    def stopped(stage):
        assert stage == "tokenization admission"
        raise CleanStop("fixture resource stop")

    monkeypatch.setattr(tracer.guard, "check", stopped)
    tracer.tokenizer = lambda *a, **kw: pytest.fail("resource stop reached the tokenizer")
    monkeypatch.setattr(model, "forward", lambda *a, **kw: pytest.fail("resource stop reached the forward"))
    with pytest.raises(CleanStop, match="fixture resource stop"):
        tracer.trace("AB", max_targets=0)
    assert not tracer._cache and tracer._forward_cache is None


@pytest.mark.parametrize("cache", ["cold", "graph", "forward"])
@pytest.mark.parametrize("resource", ["model", "dictionary"])
def test_tokenizer_alias_mutation_refuses_before_forward_or_publication(monkeypatch, cache, resource):
    model, _, dictionary = resources()
    tensor = model.embed_tokens.weight if resource == "model" else dictionary.layers[1].W_dec

    class MutatingTokenizer(TinyTokenizer):
        armed = False

        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            if self.armed:
                version = tensor._version
                tensor.data[-1, -1] = float("nan") if resource == "dictionary" else 7.
                assert tensor._version == version
            return result

    tokenizer = MutatingTokenizer()
    tracer = _trace.Tracer(model, tokenizer, dictionary)
    monkeypatch.setattr(_publish, "_TENSORS", {})
    if cache != "cold":
        tracer.trace("AB", max_targets=0)
    published = dict(_publish._TENSORS)
    tokenizer.armed = True
    monkeypatch.setattr(_frozen, "_FINITE_BLOCK_BYTES", 2 * 32)
    monkeypatch.setattr(model, "forward", lambda *a, **kw: pytest.fail("alias mutation reached the forward"))
    message = "model identity mismatch" if resource == "model" else "dictionary layer 1 W_dec must be finite"
    with pytest.raises(ValueError, match=message):
        tracer.trace("AB", max_targets=0, max_nodes=511 if cache == "forward" else 512)
    assert not tracer._cache and tracer._forward_cache is None
    assert _publish._TENSORS.keys() == published.keys()
    assert all(_publish._TENSORS[name] is value for name, value in published.items())
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("resource", ["model", "dictionary"])
def test_patch_tokenizer_alias_mutation_precedes_request_admission(monkeypatch, cached, resource):
    model, _, dictionary = resources()
    controls = Controls()
    tensor = model.embed_tokens.weight if resource == "model" else dictionary.layers[1].W_dec

    class SourceTokenizer(TinyTokenizer):
        armed = False

        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if self.armed and text == "CD":
                version = tensor._version
                tensor.data[-1, -1] = float("nan") if resource == "dictionary" else 7.
                assert tensor._version == version
            return result

    tokenizer = SourceTokenizer()
    service = controls.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    tracer = _trace.Tracer(model, tokenizer, dictionary, controls=controls)
    monkeypatch.setattr(_publish, "_TENSORS", {})
    try:
        service.set({"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "CD"})
        controls.step()
        if cached:
            tracer.trace("AB", max_targets=0)
        published = dict(_publish._TENSORS)
        hooks = [(tuple(module._forward_hooks), tuple(module._forward_pre_hooks)) for module in model.modules()]
        tokenizer.armed = True
        monkeypatch.setattr(model, "forward", lambda *a, **kw: pytest.fail("source alias reached a forward"))
        message = "model identity mismatch" if resource == "model" else "dictionary layer 1 W_dec must be finite"
        with pytest.raises(ValueError, match=message):
            tracer.trace("AB", max_targets=0)
        assert not tracer._cache and tracer._forward_cache is None
        assert _publish._TENSORS.keys() == published.keys()
        assert all(_publish._TENSORS[name] is value for name, value in published.items())
        assert hooks == [(tuple(module._forward_hooks), tuple(module._forward_pre_hooks)) for module in model.modules()]
    finally:
        service.remove()


@pytest.mark.parametrize("cache", ["graph", "forward"])
def test_tokenizer_finite_alias_change_refreshes_exact_dictionary_content(monkeypatch, cache):
    model, _, dictionary = resources()

    class MutatingTokenizer(TinyTokenizer):
        armed = False

        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            if self.armed:
                dictionary.layers[1].b_dec.data[-1] = -0.
                self.armed = False
            return result

    tokenizer = MutatingTokenizer()
    tracer = _trace.Tracer(model, tokenizer, dictionary)
    first = tracer.trace("AB", max_targets=4)
    path = tracer._forward_cache[1]
    tokenizer.armed = True
    nodes = 511 if cache == "forward" else 512
    changed = tracer.trace("AB", max_targets=4, max_nodes=nodes)
    assert changed is not first and changed.trace_id != first.trace_id
    assert tracer._forward_cache[1] is not path
    tracer._cache.clear()
    tracer._forward_cache = None
    frozen = _trace.FrozenPath

    def standalone(*args, **kwargs):
        kwargs.pop("_admission")
        return frozen(*args, **kwargs)

    monkeypatch.setattr(_trace, "FrozenPath", standalone)
    reference = tracer.trace("AB", max_targets=4, max_nodes=nodes)
    assert reference.trace_id == changed.trace_id
    assert reference.nodes == changed.nodes and reference.edges == changed.edges
    assert reference.completeness == changed.completeness
    for name, tensor in reference.tensors.items():
        assert torch.equal(tensor, changed.tensors[name])


@pytest.mark.parametrize("fault", ["model", "dictionary", "reused", "expired", "boolean"])
def test_private_admission_refuses_wrong_resources_or_lifetime(monkeypatch, fault):
    model, _, dictionary = resources()
    other_model, _, _ = resources()
    other_dictionary = exact_dictionary(model)
    target_model = other_model if fault == "model" else model
    target_dictionary = other_dictionary if fault == "dictionary" else dictionary
    monkeypatch.setattr(target_model, "forward", lambda *a, **kw: pytest.fail("invalid admission reached the forward"))
    with ExitStack() as scope:
        admission = scope.enter_context(_frozen._DictionaryAdmission(
            model, dictionary, lambda stage: None, memory=None))
        if fault == "reused":
            admission.consume(model, dictionary)
        elif fault == "expired":
            scope.close()
        with pytest.raises(TypeError if fault == "boolean" else ValueError, match="admission"):
            _frozen.FrozenPath(target_model, torch.tensor([[1, 2]]), target_dictionary,
                               _admission=True if fault == "boolean" else admission)
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in target_model.modules())


@pytest.mark.parametrize("cache", ["cold", "graph", "forward", "failure"])
def test_request_admission_expires_on_every_exit(monkeypatch, cache):
    model, tokenizer, dictionary = resources()
    tracer = _trace.Tracer(model, tokenizer, dictionary)
    if cache in ("graph", "forward"):
        tracer.trace("AB", max_targets=0)
    admissions = []
    admission_class = _trace._DictionaryAdmission

    def remember(*args, **kwargs):
        admission = admission_class(*args, **kwargs)
        admissions.append(admission)
        return admission

    monkeypatch.setattr(_trace, "_DictionaryAdmission", remember)
    if cache == "failure":
        def fail(*args, **kwargs):
            raise _trace.TraceTimeout("fixture content deadline")
        monkeypatch.setattr(_trace, "weights_hash", fail)
        with pytest.raises(_trace.TraceTimeout):
            tracer.trace("AB", max_targets=0)
    else:
        tracer.trace("AB", max_targets=0, max_nodes=511 if cache == "forward" else 512)
    assert len(admissions) == 1
    with pytest.raises(ValueError, match="expired"):
        admissions[0].consume(model, dictionary)


@pytest.mark.parametrize("boundary", ["chunk", "complete"])
def test_dictionary_hash_deadline_reports_stage_and_preserves_publication(monkeypatch, boundary):
    model, tokenizer, dictionary = resources()
    tracer = _trace.Tracer(model, tokenizer, dictionary)
    monkeypatch.setattr(_publish, "_TENSORS", {})
    tracer.trace("AB", max_targets=0)
    published, cache = dict(_publish._TENSORS), dict(tracer._cache)
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(_trace, "time", SimpleNamespace(monotonic=lambda: clock.now))
    weights_hash = _trace.weights_hash

    def expired(value, **kwargs):
        if boundary == "complete":
            result = weights_hash(value, **kwargs)
        clock.now += _trace.TRACE_TIMEOUT_SECONDS
        return weights_hash(value, **kwargs) if boundary == "chunk" else result

    monkeypatch.setattr(_trace, "weights_hash", expired)
    monkeypatch.setattr(model, "forward", lambda *a, **kw: pytest.fail("expired hash reached the forward"))
    stage = "model_identity" if boundary == "chunk" else "complete"
    with pytest.raises(_trace.TraceTimeout, match="600 second safety timeout at dictionary content " + stage):
        tracer.trace("AB", max_targets=0)
    assert tracer._cache == cache
    assert all(_publish._TENSORS[name] is value for name, value in published.items())
