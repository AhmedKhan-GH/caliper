"""Inspector behavior with NumPy backend doubles; no MLX/Metal/model validation.

These tests exercise editing, undo, bounds, sampling and generation orchestration
without importing MLX or loading the downloaded model. check_qwen_inspector.py
is the real-model integration check and must run where Metal is available.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import socket
import sys
import subprocess
import tempfile
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
    result.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    result.revisions = dict.fromkeys(result.weights, 0)
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
    sample = box_means(large, shown["region"], shown["sample_rows"], shown["sample_cols"])
    np.testing.assert_allclose(np.array(shown["values"]).reshape(sample.shape), sample, rtol=1e-6)
    assert shown["min"] == np.float32(sample.min()) and shown["max"] == np.float32(sample.max())
    host.edit(dict(name=norm["name"], action="set", row=2, col=0, value=9))
    assert call("read_weight", name=norm["name"], row=2, col=0)["value"] == 9
    assert len(call("show_tensor", name=norm["name"])["edits"]) == 1


def box_means(a, region, sample_rows, sample_cols):
    """Each bin's mean by plain loops, independent of the host's chunked reduceat."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, a.shape[-1]) if np.ndim(a) > 1 else np.asarray(a, dtype=np.float64)[:, None]
    r, c, h, w = region["row"], region["col"], region["rows"], region["cols"]
    rs = [r + (i * h) // sample_rows for i in range(sample_rows + 1)]
    cs = [c + (j * w) // sample_cols for j in range(sample_cols + 1)]
    return np.array([[a[rs[i]:rs[i + 1], cs[j]:cs[j + 1]].mean() for j in range(sample_cols)] for i in range(sample_rows)])


def test_area_samples_are_bin_means_across_chunks(monkeypatch):
    monkeypatch.setattr(example, "CHUNK_CELLS", 50)   # several chunks, with bins that end mid-way
    data = np.random.default_rng(0).normal(size=(103, 29)).astype(np.float32)
    region = dict(row=7, col=3, rows=90, cols=25)
    got = example.area_sample(lambda r0, r1: data[r0:r1, 3:28], region, 13, 6)
    np.testing.assert_allclose(got, box_means(data, region, 13, 6), rtol=1e-6)


def test_region_uses_bin_centres_in_tensor_coordinates():
    weight = array(np.arange(37 * 23, dtype=np.float32).reshape(37, 23))
    host = inspector({"weight": weight})
    shown = host(dict(op="show_tensor", name="weight",
                      region=dict(row=11, col=5, rows=9, cols=7),
                      target=dict(width=3, height=2)), lambda _: None, threading.Event())
    assert shown["region"] == dict(row=11, col=5, rows=9, cols=7)
    assert shown["row_indices"] == [13, 17]
    assert shown["col_indices"] == [6, 8, 10]
    assert (shown["sample_rows"], shown["sample_cols"], shown["exact"]) == (2, 3, False)
    # Rows 11-14 and 15-19, columns 5-6, 7-8 and 9-11: each value the mean of its bin, not one cell.
    expected = [[weight[r0:r1, c0:c1].mean() for c0, c1 in ((5, 7), (7, 9), (9, 12))] for r0, r1 in ((11, 15), (15, 20))]
    np.testing.assert_allclose(np.array(shown["values"]).reshape(2, 3), expected, rtol=1e-6)


def test_single_cells_and_terminal_region_are_exact():
    weight = array(np.arange(37 * 23, dtype=np.float32).reshape(37, 23))
    host = inspector({"weight": weight})
    call = lambda op, **kw: host(dict(op=op, name="weight", **kw), lambda _: None, threading.Event())
    shown = call("show_tensor", region=dict(row=36, col=22, rows=1, cols=1),
                 target=dict(width=100, height=200))
    assert shown["exact"] and shown["values"] == [weight[36, 22]]
    assert shown["values"][0] == call("read_weight", row=36, col=22)["value"]
    shown = call("show_tensor", region=dict(row=35, col=20, rows=2, cols=3),
                 target=dict(width=3, height=2))
    assert shown["row_indices"] == [35, 36] and shown["col_indices"] == [20, 21, 22]
    assert shown["exact"]


def test_vector_sampling_preserves_wire_coordinates():
    host = inspector({"bias": array(np.arange(1024, dtype=np.float32))})
    shown = host(dict(op="show_tensor", name="bias",
                      region=dict(row=400, col=0, rows=25, cols=1),
                      target=dict(width=1, height=25)), lambda _: None, threading.Event())
    assert (shown["rows"], shown["cols"], shown["sample_rows"], shown["sample_cols"]) == (1024, 1, 25, 1)
    assert shown["row_indices"] == list(range(400, 425)) and shown["col_indices"] == [0]
    assert shown["values"] == list(range(400, 425)) and shown["exact"]


@pytest.mark.parametrize("rows,cols,target", [
    (248320, 1024, dict(width=2048, height=2048)),
    (6144, 1024, dict(width=1500, height=400)),
    (1024, 1, dict(width=1, height=2048)),
    (1, 1, dict(width=2048, height=2048)),
])
def test_sample_budget_and_indices(rows, cols, target):
    region, ri, ci = example.sample_plan(rows, cols, dict(target=target))
    assert region == dict(row=0, col=0, rows=rows, cols=cols)
    assert len(ri) * len(ci) <= example.MAX_SAMPLES
    assert 1 <= len(ri) <= min(rows, target["height"])
    assert 1 <= len(ci) <= min(cols, target["width"])
    assert ri == sorted(set(ri)) and ci == sorted(set(ci))
    assert 0 <= ri[0] <= ri[-1] < rows and 0 <= ci[0] <= ci[-1] < cols
    if min(rows, target["height"]) * min(cols, target["width"]) > example.MAX_SAMPLES:
        requested_ratio = min(rows, target["height"]) / min(cols, target["width"])
        assert len(ri) / len(ci) == pytest.approx(requested_ratio, rel=0.01)


@pytest.mark.parametrize("arguments", [
    dict(region=None), dict(region=[]), dict(region={}),
    dict(region=dict(row=-1, col=0, rows=1, cols=1)),
    dict(region=dict(row=4, col=0, rows=1, cols=1)),
    dict(region=dict(row=3, col=0, rows=2, cols=1)),
    dict(region=dict(row=0, col=0, rows=0, cols=1)),
    dict(region=dict(row=True, col=0, rows=1, cols=1)),
    dict(region=dict(row=0, col=0.5, rows=1, cols=1)),
    dict(target=None), dict(target=[]), dict(target={}),
    dict(target=dict(width=0, height=1)), dict(target=dict(width=2049, height=1)),
    dict(target=dict(width=True, height=1)), dict(target=dict(width=1, height=float("nan"))),
])
def test_invalid_sampling_requests_do_not_change_model(arguments):
    host = inspector()
    name = next(iter(host.weights))
    before = host.weights[name].copy()
    with pytest.raises(ValueError):
        host(dict(op="show_tensor", name=name, **arguments), lambda _: None, threading.Event())
    np.testing.assert_array_equal(host.weights[name], before)
    assert host.revisions[name] == 0 and not host.undo


def test_large_sample_reply_fits_the_request_frame_budget():
    from caliper.requests import MAX_MESSAGE, encode
    # Exercise long float reprs across every finite exponent, plus a full
    # history. All values are generated on CPU; no MLX import is needed.
    indices = np.arange(example.MAX_SAMPLES, dtype=np.uint32)
    exponent = (indices % 254 + 1) << 23
    mantissa = (indices * np.uint32(2654435761)) & np.uint32(0x7fffff)
    bits = np.uint32(0x80000000) | exponent | mantissa
    weight = array(bits.view(np.float32).reshape(512, 256))
    host = inspector({"weight": weight})
    host.history = [dict(id=i + 1, name="weight", description="Scale row 248319 by -3.40282347e+38" + " " * 96)
                    for i in range(512)]
    result = host(dict(op="show_tensor", name="weight", target=dict(width=2048, height=2048)),
                  lambda _: None, threading.Event())
    assert len(result["values"]) == example.MAX_SAMPLES
    assert len(result["edits"]) == 512
    assert all(value < 0 and np.isfinite(value) for value in result["values"])
    assert len(encode(dict(id=1, done=True, result=result))) < MAX_MESSAGE


def test_layer_roles_tied_alias_and_active_edit_revisions():
    host = inspector({"language_model.model.embed_tokens.weight": array([[1., 2.], [3., 4.]]),
                      "language_model.model.layers.0.mlp.down_proj.weight": array([[1., 2.]]),
                      "language_model.model.layers.3.self_attn.q_proj.weight": array([[1., 2.]]),
                      "language_model.model.norm.weight": array([1., 2.])})
    call = lambda op, **kw: host(dict(op=op, **kw), lambda _: None, threading.Event())
    manifest = call("list_tensors")
    tensors = {tensor["name"]: tensor for tensor in manifest["tensors"]}
    name = manifest["output_alias"]["tensor"]
    assert tensors[name]["role"] == "embedding" and tensors[name]["tied"]
    assert tensors["language_model.model.norm.weight"]["role"] == "final_norm"
    linear = tensors["language_model.model.layers.0.mlp.down_proj.weight"]
    full = tensors["language_model.model.layers.3.self_attn.q_proj.weight"]
    assert (linear["layer_index"], linear["layer_type"]) == (0, "linear_attention")
    assert (full["layer_index"], full["layer_type"]) == (3, "full_attention")
    assert all(not tensor["edited"] and tensor["revision"] == 0 for tensor in tensors.values())
    first = call("edit_weight", name=name, action="set", row=0, col=0, value=7)
    assert first["edited"] and first["revision"] == 1
    shown = call("show_tensor", name=name)
    assert shown["edited"] and shown["revision"] == 1
    assert next(t for t in call("list_tensors")["tensors"] if t["name"] == name)["edited"]
    reverted = call("revert_tensor", name=name)
    assert not reverted["edited"] and reverted["revision"] == 2
    assert reverted["edits"]  # History remains, but it no longer means active edits.
    assert not call("show_tensor", name=name)["edited"]
    assert not call("revert_tensor", name=name)["edited"]
    assert call("show_tensor", name=name)["revision"] == 2


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


def test_backend_double_over_real_request_channel(generation, monkeypatch):
    import caliper.requests as transport

    host, state = generation
    name = next(iter(host.weights))
    # Use a short, private socket path unrelated to any real live session.
    # A pytest tmp_path can exceed macOS's Unix-socket path-length limit.
    with tempfile.TemporaryDirectory(prefix="caliper-wire-", dir="/tmp") as directory:
        monkeypatch.setattr(transport, "socket_path", lambda _: str(Path(directory) / "host.sock"))
        with transport.RequestServer("backend-double", example.CAPABILITIES) as server:
            worker = threading.Thread(target=server.serve, args=(host,), daemon=True)
            worker.start()
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as controller:
                    controller.settimeout(3)
                    controller.connect(server.path)
                    with controller.makefile("rb") as replies:
                        serial = 0

                        def request(op, **arguments):
                            nonlocal serial
                            serial += 1
                            controller.sendall(transport.encode(dict(id=serial, op=op, **arguments)))
                            events = []
                            while True:
                                line = replies.readline(transport.MAX_MESSAGE + 1)
                                assert line.endswith(b"\n") and len(line) <= transport.MAX_MESSAGE
                                reply = json.loads(line)
                                assert reply["id"] == serial and "error" not in reply, reply
                                if reply["done"]:
                                    return reply["result"], events
                                events.append(reply["event"])

                        hello, _ = request("hello")
                        assert hello["protocol"] == 1 and "tensor_regions" in hello["capabilities"]
                        manifest, _ = request("list_tensors")
                        assert manifest["tensors"][0]["name"] == name
                        region = dict(row=0, col=0, rows=2, cols=3)
                        target = dict(width=3, height=2)
                        original, _ = request("show_tensor", name=name, region=region, target=target)
                        assert original["region"] == region and original["exact"]
                        assert original["values"] == [1., 2., 3., 6., 7., 8.]
                        assert original["revision"] == 0 and not original["edited"]

                        result, events = request("generate", prompt="Hello", max_new_tokens=3)
                        assert "".join(event["text"] for event in events) == "text 1text 2text 3"
                        assert result["tokens"] == 3 and not result["cancelled"]

                        edited, _ = request("edit_weight", name=name, action="set", row=0, col=0, value=0.25)
                        assert edited["edited"] and edited["revision"] == 1
                        value, _ = request("read_weight", name=name, row=0, col=0)
                        assert value["value"] == 0.25 and value["revision"] == 1
                        changed, _ = request("show_tensor", name=name, region=region, target=target)
                        assert changed["edited"] and changed["revision"] != original["revision"]
                        assert changed["values"][0] == 0.25
                        request("generate", prompt="Hello", max_new_tokens=3)
                        assert state.calls[-1]["model"] is host.model and state.calls[-1]["weight"] == 0.25

                        reverted, _ = request("revert_tensor", name=name)
                        assert not reverted["edited"] and reverted["revision"] == 2
                        restored, _ = request("read_weight", name=name, row=0, col=0)
                        assert restored["value"] == 1 and restored["revision"] == 2
                        final, _ = request("show_tensor", name=name, region=region, target=target)
                        assert final["values"] == original["values"] and not final["edited"]
                        assert final["revision"] != changed["revision"]
            finally:
                server.close()
                worker.join(timeout=3)
                assert not worker.is_alive()
        assert not Path(server.path).exists()
        assert not Path(server.advertisement_path).exists()


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
