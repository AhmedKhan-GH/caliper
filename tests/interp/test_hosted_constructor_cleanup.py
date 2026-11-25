"""Constructor cleanup through CPU resource seams, without model execution.

The memmap seam bypasses index admission but uses real FeatureIndex.close.
"""
import gc
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from caliper.interp import _hosted, _memory
from caliper.interp._index import FeatureIndex
from caliper.interp._service import Refused


class Resource:
    pass


class Model:
    config = SimpleNamespace(vocab_size=512)

    def parameters(self):
        return iter((torch.empty(0),))


class Index:
    def __init__(self, calls, *resources):
        self.calls, self.resources = calls, resources

    def close(self):
        self.calls.append("close")


def resource_tuple(calls, *, guard=True):
    model, tokenizer, dictionary = Model(), _hosted.TinyTokenizer(), Resource()
    memory = Resource() if guard else None
    index = Index(calls, model, tokenizer, dictionary, memory)
    return model, tokenizer, dictionary, index, memory


@pytest.fixture(autouse=True)
def cpu_display(monkeypatch):
    monkeypatch.setattr(_hosted, "_fixture_display_guard", lambda name: None)


@pytest.mark.parametrize("owned", [True, False], ids=["owned", "borrowed"])
@pytest.mark.parametrize("prompt, message", [
    ("", "valid UTF-8"),
    ("A\0B", "valid UTF-8"),
    (None, "valid UTF-8"),
    ("A" * 32, "1 to 32 integer tokens"),
])
def test_invalid_prompt_closes_only_owned_index(monkeypatch, tmp_path, owned, prompt, message):
    calls = []
    resources = resource_tuple(calls)
    monkeypatch.setattr(_hosted, "load_resources", lambda *args: resources)
    with pytest.raises(ValueError, match=message) as caught:
        _hosted.HostedInference("tiny-lm", tmp_path, prompt, resources=None if owned else resources)
    assert caught.value.__traceback__ is not None
    assert calls == (["close"] if owned else [])
    gc.collect()
    assert calls == (["close"] if owned else [])
    if not owned:
        resources[3].close()
        assert calls == ["close"]


@pytest.mark.parametrize("owned", [True, False], ids=["owned", "borrowed"])
@pytest.mark.parametrize("phase", ["memory_guard", "run_guard", "display", "prompt", "generation"])
def test_setup_refusal_closes_only_owned_index(monkeypatch, tmp_path, owned, phase):
    calls = []
    resources = resource_tuple(calls, guard=phase != "memory_guard")
    refusal = Refused("interpret.unavailable", "constructor setup refused")

    def refuse(*args, **kwargs):
        raise refusal

    monkeypatch.setattr(_hosted, "load_resources", lambda *args: resources)
    if phase in ("memory_guard", "display"):
        monkeypatch.setattr(_hosted, "_fixture_display_guard", lambda name: Resource())
    target, name = {
        "memory_guard": (_memory, "MemoryGuard"),
        "run_guard": (_hosted, "_RunGuard"),
        "display": (_hosted, "_FixtureDisplay"),
        "prompt": (_hosted, "prompt_inputs"),
        "generation": (_hosted.HostedInference, "_restore_generation"),
    }[phase]
    monkeypatch.setattr(target, name, refuse)
    with pytest.raises(Refused) as caught:
        _hosted.HostedInference("tiny-lm", tmp_path, "AB", resources=None if owned else resources)
    assert caught.value is refusal
    assert refusal.code == "interpret.unavailable"
    assert str(refusal) == "constructor setup refused"
    assert calls == (["close"] if owned else [])


@pytest.mark.parametrize("failure_type", [ValueError, _memory.CleanStop, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("cleanup_type", [RuntimeError, KeyboardInterrupt])
def test_cleanup_failure_preserves_original_refusal(monkeypatch, tmp_path, failure_type, cleanup_type):
    calls = []
    refusal = failure_type("original refusal")
    cause = ValueError("original cause")

    def refuse(*args):
        raise refusal from cause

    def close(index):
        calls.append("close")
        raise cleanup_type("cleanup refusal")

    monkeypatch.setattr(Index, "close", close)
    monkeypatch.setattr(_hosted, "load_resources", lambda *args: resource_tuple(calls))
    monkeypatch.setattr(_hosted, "prompt_inputs", refuse)
    with pytest.raises(failure_type) as caught:
        _hosted.HostedInference("tiny-lm", tmp_path, "AB")
    assert caught.value is refusal
    assert str(refusal) == "original refusal"
    assert refusal.__cause__ is cause and refusal.__context__ is None
    assert calls == ["close"]


@pytest.mark.parametrize("display", [False, True])
def test_retained_prompt_exception_releases_owned_resources(monkeypatch, tmp_path, display):
    calls, references = [], []

    def load(*args):
        resources = resource_tuple(calls)
        references.extend(weakref.ref(value) for value in resources)
        return resources

    def display_resource(guard):
        value = Resource()
        value.guard = guard
        references.append(weakref.ref(value))
        return value

    if display:
        monkeypatch.setattr(_hosted, "_fixture_display_guard", lambda name: display_resource(None))
        monkeypatch.setattr(_hosted, "_FixtureDisplay", display_resource)
    monkeypatch.setattr(_hosted, "load_resources", load)
    with pytest.raises(ValueError, match="valid UTF-8") as caught:
        _hosted.HostedInference("tiny-lm", tmp_path, "")
    gc.collect()
    assert caught.value.__traceback__ is not None
    assert calls == ["close"]
    assert len(references) == (7 if display else 5)
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("explicit_cause", [True, False], ids=["cause", "context"])
def test_retained_exception_chain_releases_owned_resources(monkeypatch, tmp_path, explicit_cause):
    calls, references, causes = [], [], []
    refusal = Refused("interpret.unavailable", "prompt setup refused")

    def load(*args):
        resources = resource_tuple(calls)
        references.extend(weakref.ref(value) for value in resources)
        return resources

    def inner(model, tokenizer):
        raise ValueError("inner refusal")

    def refuse(model, tokenizer, prompt):
        try:
            inner(model, tokenizer)
        except ValueError as cause:
            causes.append(cause)
            if explicit_cause:
                raise refusal from cause
            raise refusal

    monkeypatch.setattr(_hosted, "load_resources", load)
    monkeypatch.setattr(_hosted, "prompt_inputs", refuse)
    with pytest.raises(Refused) as caught:
        _hosted.HostedInference("tiny-lm", tmp_path, "AB")
    gc.collect()
    assert caught.value is refusal
    assert refusal.__context__ is causes[0]
    assert refusal.__cause__ is (causes[0] if explicit_cause else None)
    assert causes[0].__traceback__.tb_next.tb_frame.f_code.co_name == "inner"
    assert calls == ["close"]
    assert len(references) == 5 and all(reference() is None for reference in references)


def test_retained_cleanup_exception_releases_owned_resources(monkeypatch, tmp_path):
    calls, references = [], []
    cleanup_error = RuntimeError("cleanup refusal")

    def load(*args):
        resources = resource_tuple(calls)
        references.extend(weakref.ref(value) for value in resources)
        return resources

    def close(index):
        calls.append("close")
        raise cleanup_error

    monkeypatch.setattr(Index, "close", close)
    monkeypatch.setattr(_hosted, "load_resources", load)
    with pytest.raises(ValueError, match="valid UTF-8") as caught:
        _hosted.HostedInference("tiny-lm", tmp_path, "")
    gc.collect()
    assert cleanup_error.__context__ is caught.value
    assert caught.value.__context__ is None
    assert calls == ["close"]
    assert len(references) == 5 and all(reference() is None for reference in references)


@pytest.mark.parametrize("owned", [True, False], ids=["owned", "borrowed"])
def test_retained_exception_preserves_memmap_ownership(monkeypatch, tmp_path, owned):
    resources = resource_tuple([])
    # Admission is outside this seam. Only map lifetime and close use real code.
    index = FeatureIndex.__new__(FeatureIndex)
    index._closed, index._cache, index._summaries = False, {}, {}
    index.model, index.tokenizer, index.dictionary = resources[:3]
    names = ("table", "counts", "tokens", "documents")
    for name in names:
        setattr(index, name, np.lib.format.open_memmap(
            tmp_path / f"{name}.npy", mode="w+", dtype="uint32", shape=(2,)))
    mappings = [getattr(index, name)._mmap for name in names]
    resources = (*resources[:3], index, resources[4])
    calls = []
    original_close = FeatureIndex.close

    def close(value):
        calls.append("close")
        original_close(value)

    monkeypatch.setattr(FeatureIndex, "close", close)
    monkeypatch.setattr(_hosted, "load_resources", lambda *args: resources)
    assert all(not mapping.closed for mapping in mappings)
    try:
        with pytest.raises(ValueError, match="valid UTF-8") as caught:
            _hosted.HostedInference("tiny-lm", tmp_path, "", resources=None if owned else resources)
        assert caught.value.__traceback__ is not None
        assert calls == (["close"] if owned else [])
        assert index._closed is owned
        assert all(mapping.closed is owned for mapping in mappings)
        if owned:
            assert all(getattr(index, name) is None for name in names)
            for name in names:
                (tmp_path / f"{name}.npy").unlink()
        else:
            assert index.tokens.tolist() == [0, 0]
    finally:
        if not index._closed:
            index.close()


@pytest.mark.parametrize("owned", [True, False], ids=["owned", "borrowed"])
def test_success_keeps_existing_close_behavior(monkeypatch, tmp_path, owned):
    calls = []
    resources = resource_tuple(calls)
    monkeypatch.setattr(_hosted, "load_resources", lambda *args: resources)
    run = _hosted.HostedInference("tiny-lm", tmp_path, "AB", resources=None if owned else resources)
    assert (run.model, run.tokenizer, run.dictionary, run.index) == resources[:4]
    assert run.guard.guard is resources[4]
    assert not calls
    run.close()
    assert calls == ["close"]
