"""CPU seams for the explicit fixture display bridge."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _hosted, _memory, publish
from caliper.interp._output import forward_logits
from caliper.interp._publish import _clear, _resolve
from .fixture import forward_hash
from .test_hosted import hosted_run
from .test_intervene import dictionary_for


@pytest.fixture(autouse=True)
def isolated_display(monkeypatch):
    monkeypatch.delenv("CALIPER_FIXTURE_DISPLAY_MPS", raising=False)
    _clear()
    yield
    _clear()


def test_fixture_display_defaults_off_without_a_device_query(monkeypatch):
    def forbidden():
        pytest.fail("the default must not query MPS")
    monkeypatch.setattr(torch.backends.mps, "is_available", forbidden)
    assert _hosted._fixture_display_guard("tiny-lm") is None
    monkeypatch.setenv("CALIPER_FIXTURE_DISPLAY_MPS", "0")
    assert _hosted._fixture_display_guard("tiny-lm") is None


def test_fixture_display_refuses_invalid_opt_in_and_missing_mps(monkeypatch):
    monkeypatch.setenv("CALIPER_FIXTURE_DISPLAY_MPS", "yes")
    with pytest.raises(ValueError, match="0 or 1"):
        _hosted._fixture_display_guard("tiny-lm")
    monkeypatch.setenv("CALIPER_FIXTURE_DISPLAY_MPS", "1")
    with pytest.raises(ValueError, match="tiny-lm"):
        _hosted._fixture_display_guard("other")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="require MPS"):
        _hosted._fixture_display_guard("tiny-lm")


@pytest.mark.parametrize("supplied_guard", [False, True])
def test_fixture_opt_in_adds_the_shared_twelve_gb_guard_without_changes_to_math(tmp_path, monkeypatch, supplied_guard):
    from caliper.interp._validate import _validation_guard
    stages, limits, original_stages = [], [], []
    def memory_guard(limit, device):
        limits.append((limit, device))
        calls = stages if torch.device(device).type == "mps" else original_stages
        return SimpleNamespace(check=lambda stage, **kwargs: calls.append(stage),
                               device=torch.device(device), limit=limit * 1e9)
    monkeypatch.setenv("CALIPER_FIXTURE_DISPLAY_MPS", "1")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(_memory, "MemoryGuard", memory_guard)
    model, tokenizer = _hosted.TinyLM(), _hosted.TinyTokenizer()
    dictionary = dictionary_for(model)
    expected = forward_hash(model)
    run = _hosted.HostedInference("tiny-lm", tmp_path, "AB", resources=(
        model, tokenizer, dictionary, SimpleNamespace(close=lambda: None),
        SimpleNamespace(check=lambda stage, **kwargs: original_stages.append(stage),
                        device=torch.device("cpu"), limit=12e9) if supplied_guard else None))
    try:
        assert limits == [(12, "mps")] + ([] if supplied_guard else [(12, torch.device("cpu"))])
        assert run.fixture_display.guard is run.guard
        assert run.guard.device == torch.device("cpu")
        assert run.guard.extra.device == torch.device("mps")
        run.guard.check("test")
        assert stages == original_stages == ["test"]
        run.guard.extra.limit = 11e9
        validation = _validation_guard(model, dictionary, None, run.guard)
        assert validation.device == torch.device("cpu") and validation.limit == 11e9
        assert forward_hash(model) == expected
        assert all(parameter.device.type == "cpu" and parameter.dtype == torch.float64 for parameter in model.parameters())
    finally:
        run.close()


def _display_tensors():
    values = {}
    for name, (shape, dtype) in _hosted._FixtureDisplay.TENSORS.items():
        shape = tuple(32 if dim is None else dim for dim in shape)
        values[name] = torch.arange(torch.Size(shape).numel()).reshape(shape).to(dtype)
        publish(name, values[name])
    return values


def test_fixture_display_copies_only_fixed_tensors_within_the_aggregate_cap(monkeypatch):
    sources = _display_tensors()
    arbitrary = torch.zeros(4096, dtype=torch.float64)
    publish("trace.arbitrary", arbitrary)
    stages, scopes = [], []
    @contextmanager
    def scope(device):
        assert device.type == "cpu"
        scopes.append("enter")
        try:
            yield
        finally:
            scopes.append("exit")
    monkeypatch.setattr(_memory, "autorelease_scope", scope)
    display = _hosted._FixtureDisplay(SimpleNamespace(check=lambda stage: stages.append(stage)), device="cpu")
    copied = display.publish()
    assert copied == sum(t.numel() * t.element_size() for t in sources.values())
    assert 0 < copied < 256 * 1024 and len(sources) == 18
    assert stages == ["fixture display copy start", "fixture display copy complete"]
    assert scopes == ["enter", "exit"]
    for name, source in sources.items():
        shown = _resolve(name)
        assert shown.device.type == "cpu" and shown.dtype == source.dtype
        assert shown.data_ptr() != source.data_ptr()
        torch.testing.assert_close(shown, source, rtol=0, atol=0)
    assert _resolve("trace.arbitrary").data_ptr() == arbitrary.data_ptr()


@pytest.mark.parametrize("device,accepted", [("mps:0", True), ("mps:1", False)])
def test_fixture_display_recognizes_the_concrete_mps_device(monkeypatch, device, accepted):
    from caliper.interp import _publish
    tensor = SimpleNamespace(dtype=torch.float32, ndim=2, shape=(2, 10),
                             device=torch.device(device), numel=lambda: 20, element_size=lambda: 4)
    monkeypatch.setattr(_publish, "_TENSORS", {"output.probs": tensor})
    display = _hosted._FixtureDisplay(None)
    assert display.device == torch.device("mps:0")
    if accepted:
        assert display.publish() == 0
    else:
        with pytest.raises(ValueError, match="invalid device"):
            display.publish()


@pytest.mark.parametrize("name,tensor", [
    ("trace.node_f", torch.zeros(513, 4)),
    ("output.probs", torch.zeros(2, 10, dtype=torch.float64)),
    ("patch.grid", torch.zeros(4, 33)),
    ("patch.grid", torch.zeros(3, 2)),
])
def test_fixture_display_refuses_invalid_shapes_and_dtypes_before_copy(name, tensor, monkeypatch):
    publish(name, tensor)
    display = _hosted._FixtureDisplay(None, device="cpu")
    monkeypatch.setattr(display, "_transfer", lambda tensor: pytest.fail("invalid display tensor reached transfer"))
    with pytest.raises(ValueError, match="shape or dtype"):
        display.publish()
    assert _resolve(name).data_ptr() == tensor.data_ptr()


def test_fixture_display_refuses_aggregate_overflow_before_copy(monkeypatch):
    _display_tensors()
    display = _hosted._FixtureDisplay(None, device="cpu")
    monkeypatch.setattr(display, "MAX_BYTES", 128)
    with pytest.raises(ValueError, match="aggregate byte limit"):
        display.publish()


def test_fixture_display_guard_failure_preserves_every_publication():
    sources = _display_tensors()
    def check(stage):
        if stage == "fixture display copy complete":
            raise _memory.CleanStop("fixture display limit")
    guard = _hosted._RunGuard(None, extra=SimpleNamespace(check=check))
    display = _hosted._FixtureDisplay(guard, device="cpu")
    with pytest.raises(_memory.CleanStop, match="fixture display limit"):
        display.publish()
    for name, tensor in sources.items():
        assert _resolve(name).data_ptr() == tensor.data_ptr()
    with pytest.raises(_memory.CleanStop, match="fixture display limit"):
        guard.check("later source forward")


def test_fixture_transfer_failure_checks_resources_and_preserves_publications(monkeypatch):
    sources = _display_tensors()
    stages, calls = [], []
    display = _hosted._FixtureDisplay(SimpleNamespace(check=lambda stage: stages.append(stage)), device="cpu")
    def transfer(tensor):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("fixture transfer failure")
        return tensor.clone()
    monkeypatch.setattr(display, "_transfer", transfer)
    with pytest.raises(RuntimeError, match="fixture transfer failure"):
        display.publish()
    assert stages == ["fixture display copy start", "fixture display copy complete"]
    for name, tensor in sources.items():
        assert _resolve(name).data_ptr() == tensor.data_ptr()


def test_fixture_display_publishes_before_the_producer_without_changes_to_cpu_math(hosted_run, monkeypatch):
    from caliper import _programs
    run, _ = hosted_run
    expected = forward_logits(run.model, run.tokenizer("AB")).clone()
    parameters = {name: tensor.data_ptr() for name, tensor in run.model.named_parameters()}
    stages, producer = [], []
    run.fixture_display = _hosted._FixtureDisplay(SimpleNamespace(check=lambda stage: stages.append(stage)), device="cpu")
    def at_step(step):
        graph = run.service.graph
        assert stages[-1] == "fixture display copy complete"
        assert _resolve("trace.node_f").data_ptr() != graph.tensors["trace.node_f"].data_ptr()
        torch.testing.assert_close(_resolve("trace.node_f"), graph.tensors["trace.node_f"], rtol=0, atol=0)
        producer.append(step)
    monkeypatch.setattr(_programs, "ACTIVE", True)
    monkeypatch.setattr(_programs, "at_step", at_step)
    replies = []
    run.service.submit({"version": 1, "method": "trace.run", "params": {"prompt": "AB", "max_targets": 16}},
                       lambda *args: replies.append(args))
    run.boundary()
    assert replies[-1][0] == "ok", replies
    assert producer == [run.controls.n]
    assert parameters == {name: tensor.data_ptr() for name, tensor in run.model.named_parameters()}
    assert all(t.device.type == "cpu" for t in run.service.graph.tensors.values())
    assert all(layer.W_enc.device.type == "cpu" and layer.W_enc.dtype == torch.float64 for layer in run.dictionary.layers)
    torch.testing.assert_close(forward_logits(run.model, run.tokenizer("AB")), expected, rtol=0, atol=0)
