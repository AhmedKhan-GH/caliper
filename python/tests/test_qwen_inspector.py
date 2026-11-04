"""Inspector behavior with NumPy backend doubles; no MLX/Metal/model validation.

These tests exercise editing, undo, bounds, sampling and generation orchestration
without importing MLX or loading the downloaded model. check_qwen_inspector.py
is the real-model integration check and must run where Metal is available.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import subprocess
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "caliper_qwen_inspector_example",
    Path(__file__).resolve().parents[2] / "examples/python/qwen_inspector.py")
example = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = example
spec.loader.exec_module(example)


class Array(np.ndarray):
    def reshape(self, *shape, **kwargs):
        # MLX reshape returns a separate immutable-data handle: assignment
        # updates that handle without changing its source array. NumPy views
        # instead alias mutable data, so the double copies at this boundary.
        return super().reshape(*shape, **kwargs).copy()


def array(value, dtype=None):
    return np.array(value, dtype=dtype).view(Array)


class Backend:
    float32 = np.float32
    array = staticmethod(array)
    all = staticmethod(np.all)
    isfinite = staticmethod(np.isfinite)
    zeros_like = staticmethod(np.zeros_like)

    def __init__(self):
        self.clears = 0

    def eval(self, value):
        pass

    def clear_cache(self):
        self.clears += 1

    def get_peak_memory(self):
        return 123456


class Model:
    def __init__(self, weights):
        self.weights = dict(weights)

    def load_weights(self, values, strict):
        assert strict is False
        self.weights.update(values)


def inspector(weights=None):
    result = example.Inspector.__new__(example.Inspector)
    result.mx = Backend()
    result.weights = weights or {"language_model.model.layers.0.test.weight":
                                array(np.arange(1, 21, dtype=np.float32).reshape(4, 5))}
    result.model = Model(result.weights)
    result.model_name = "Test model"
    result.undo = []
    result.undo_bytes = 0
    result.history = []
    result.next_edit = 1
    result.tied = True
    return result


def edit(host, action, **kwargs):
    return host.edit(dict(name=next(iter(host.weights)), action=action, row=0, col=0, **kwargs))


def test_overlapping_edits_revert_original_and_update_live_model():
    host = inspector()
    name = next(iter(host.weights))
    original = host.weights[name].copy()
    edit(host, "set", value=0.25)
    assert host.model.weights[name][0, 0] == 0.25
    assert host.model.weights[name] is host.weights[name]
    edit(host, "scale_row", value=2)
    np.testing.assert_array_equal(host.weights[name][0], [0.5, 4, 6, 8, 10])
    edit(host, "scale_column", value=0.5)
    np.testing.assert_array_equal(host.weights[name][:, 0], [0.25, 3, 5.5, 8])
    edit(host, "zero_row")
    assert not host.weights[name][0].any()
    edit(host, "zero_column")
    assert not host.weights[name][:, 0].any()
    assert host.undo_bytes > 0
    result = host.revert(dict(name=name))
    np.testing.assert_array_equal(host.model.weights[name], original)
    assert host.undo_bytes == 0 and not host.undo
    assert len(result["edits"]) == 6
    assert result["edits"][-1]["description"] == "Reverted to original"


def test_revert_one_tensor_preserves_other_edits():
    host = inspector({"first": array([[1., 2.]]), "second": array([[3., 4.]])})
    host.edit(dict(name="first", action="set", row=0, col=0, value=7))
    host.edit(dict(name="second", action="set", row=0, col=0, value=8))
    host.revert(dict(name="first"))
    assert host.weights["first"][0, 0] == 1
    assert host.weights["second"][0, 0] == 8
    assert len(host.undo) == 1 and host.undo[0].name == "second"
    assert host.undo_bytes == 4


def test_manifest_matrix_layout_sample_indices_and_reconnect_history():
    large = array(np.arange(513 * 259, dtype=np.float32).reshape(513, 259))
    host = inspector({"language_model.model.embed_tokens.weight": large,
                      "language_model.model.layers.3.norm.weight": array([1., 2., 3.]),
                      "conv": array(np.zeros((2, 3, 4))), "scalar": array(1.)})
    call = lambda op, **kw: host(dict(op=op, **kw), lambda _: None, threading.Event())
    manifest = call("list_tensors")
    metadata = {t["name"]: t for t in manifest["tensors"]}
    embedding = metadata["language_model.model.embed_tokens.weight"]
    assert embedding["tied"] and embedding["bytes"] == large.nbytes and embedding["size"] == large.size
    assert metadata["conv"]["rows"] == 6 and metadata["conv"]["cols"] == 4
    assert metadata["scalar"]["rows"] == metadata["scalar"]["cols"] == 1
    norm = metadata["language_model.model.layers.3.norm.weight"]
    assert (norm["rows"], norm["cols"], norm["group"]) == (3, 1, "Layer 03")
    shown = call("show_tensor", name=embedding["name"])
    assert shown["sample_rows"] <= 128 and shown["sample_cols"] <= 128
    sample = large[np.ix_(shown["row_indices"], shown["col_indices"])]
    np.testing.assert_array_equal(np.array(shown["values"]).reshape(sample.shape), sample)
    assert shown["min"] == sample.min() and shown["max"] == sample.max()
    host.edit(dict(name=norm["name"], action="set", row=2, col=0, value=9))
    assert call("read_weight", name=norm["name"], row=2, col=0)["value"] == 9
    assert len(call("show_tensor", name=norm["name"])["edits"]) == 1


@pytest.mark.parametrize("arguments", [
    dict(name="missing"), dict(row=-1), dict(row=4), dict(col=5), dict(row=True),
    dict(value=float("nan")), dict(value=float("inf")), dict(action="unknown"),
])
def test_invalid_edit_leaves_weights_unchanged(arguments):
    host = inspector()
    name = next(iter(host.weights))
    before = host.weights[name].copy()
    request = dict(name=name, action="set", row=0, col=0, value=7)
    request.update(arguments)
    with pytest.raises(ValueError):
        host.edit(request)
    np.testing.assert_array_equal(host.weights[name], before)
    assert not host.undo and not host.history


def test_overflow_and_undo_limits_are_refused(monkeypatch):
    host = inspector({"weight": array([[40000., 2.]], dtype=np.float16)})
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="non-finite"):
        edit(host, "scale_row", value=2)
    assert host.weights["weight"][0, 0] == 40000
    monkeypatch.setattr(example, "MAX_UNDO_BYTES", 1)
    with pytest.raises(ValueError, match="undo limit"):
        edit(host, "set", value=7)
    assert host.undo_bytes == 0
    monkeypatch.setattr(example, "MAX_UNDO_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(example, "MAX_UNDO_EDITS", 0)
    with pytest.raises(ValueError, match="undo limit"):
        edit(host, "set", value=7)


@pytest.fixture
def generation(monkeypatch):
    state = SimpleNamespace(calls=[], sampler=[], templates=[], closed=0, prefill_stop=None)
    def make_sampler(**kwargs):
        state.sampler.append(kwargs)
        return "sampler"
    def stream_generate(model, tokenizer, prompt, **kwargs):
        state.calls.append(dict(model=model, prompt=prompt, kwargs=kwargs,
                                weight=float(next(iter(model.weights.values()))[0, 0])))
        try:
            if state.prefill_stop:
                state.prefill_stop.set()
            kwargs["prompt_progress_callback"](0, len(prompt))
            for count in (1, 2, 3):
                yield SimpleNamespace(text=f"text {count}", generation_tokens=count,
                                      generation_tps=12.5, finish_reason="length" if count == 3 else None)
        finally:
            state.closed += 1
    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.stream_generate = stream_generate
    sampling = ModuleType("mlx_lm.sample_utils")
    sampling.make_sampler = make_sampler
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sampling)
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            state.templates.append((messages, kwargs))
            return [11, 22, 33]
    host = inspector()
    host.tokenizer = Tokenizer()
    return host, state


def test_generation_uses_edited_weights_and_forwards_sampler_thinking(generation):
    host, state = generation
    name = next(iter(host.weights))
    edit(host, "set", value=0.25)
    events = []
    request = dict(prompt="Hello", max_new_tokens=7, temperature=0.6, top_p=0.8, top_k=9, thinking=True)
    result = host.generate(request, events.append, threading.Event())
    assert state.calls[0]["model"] is host.model and state.calls[0]["weight"] == 0.25
    assert state.sampler == [dict(temp=0.6, top_p=0.8, top_k=9)]
    assert state.templates[0][1]["enable_thinking"] is True
    assert state.calls[0]["kwargs"]["max_tokens"] == 7
    assert state.calls[0]["kwargs"]["prefill_step_size"] == 128
    assert "prompt_cache" not in state.calls[0]["kwargs"]
    assert len(events) == 3 and result["tokens"] == 3 and not result["cancelled"]
    assert state.closed == host.mx.clears == 1
    host.revert(dict(name=name))
    host.generate(dict(request, thinking=False), lambda _: None, threading.Event())
    assert state.calls[1]["weight"] == 1
    assert state.templates[1][1]["enable_thinking"] is False
    assert state.closed == 2


def test_stop_during_stream_closes_generator_before_next_operation(generation):
    host, state = generation
    cancelled = threading.Event()
    events = []
    def emit(event):
        events.append(event)
        cancelled.set()
    result = host.generate(dict(prompt="Hello"), emit, cancelled)
    assert result["cancelled"] and result["tokens"] == 1
    assert len(events) == 1 and state.closed == 1 and host.mx.clears == 1


def test_stop_during_prefill(generation):
    host, state = generation
    cancelled = threading.Event()
    state.prefill_stop = cancelled
    result = host.generate(dict(prompt="Hello"), lambda _: pytest.fail("unexpected token"), cancelled)
    assert result["cancelled"] and result["tokens"] == 0 and state.closed == 1


@pytest.mark.parametrize("arguments", [dict(max_new_tokens=0), dict(max_new_tokens=1025),
    dict(temperature=3), dict(top_p=-1), dict(top_k=1001), dict(thinking="yes"), dict(prompt="")])
def test_generation_settings_validated_before_work(generation, arguments):
    host, state = generation
    with pytest.raises(ValueError):
        host.generate(dict(prompt="Hello", **arguments) if "prompt" not in arguments else arguments,
                      lambda _: None, threading.Event())
    assert not state.calls


def test_model_lease_rejects_second_process_before_loading(tmp_path):
    tmp_path.chmod(0o700)
    lock = tmp_path / "inspector.lock"
    code = """
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location('inspector_lease_test', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
try:
    with module.model_lease(pathlib.Path(sys.argv[2])):
        sys.exit(0)
except RuntimeError as exc:
    print(exc)
    sys.exit(4)
"""
    args = [sys.executable, "-c", code, str(Path(example.__file__)), str(lock)]
    with example.model_lease(lock):
        child = subprocess.run(args, capture_output=True, text=True, timeout=10)
        assert child.returncode == 4 and "another inspector" in child.stdout
    child = subprocess.run(args, capture_output=True, text=True, timeout=10)
    assert child.returncode == 0, child.stderr
    assert lock.exists() and lock.stat().st_size == 0
