"""Named tensor lifecycle and a real producer derive pass."""
from contextlib import contextmanager
import ctypes
import gc
import struct
import weakref

import pytest
import torch

import caliper
from caliper import interp
from caliper import _programs as P
from caliper._adapter import TORCH
from caliper._watch import WatchHandle
from caliper.interp import _publish


@pytest.fixture(autouse=True)
def clean_registry():
    P.detach()
    yield
    P.detach()
    runtime = P._STATE["rt"]
    if runtime is not None:
        runtime.close()
        P._STATE["rt"] = None
    P._CAD.reset()


def attach(model):
    handle = WatchHandle()
    P.attach(model, None, handle)
    return handle


def test_publication_preserves_name_storage_dtype_and_device(monkeypatch):
    model = torch.nn.Linear(2, 2)
    handle = attach(model)
    tensor = torch.arange(12, dtype=torch.float32, requires_grad=True).reshape(3, 4)[:, ::2]
    def forbidden(*args, **kwargs):
        raise AssertionError("publication must not copy or transfer a tensor")
    with monkeypatch.context() as patch:
        for method in ("cpu", "numpy", "clone", "contiguous", "to"):
            patch.setattr(torch.Tensor, method, forbidden)
        assert interp.publish("trace.node_f", tensor) is None
    resolved = P._resolve("trace.node_f")
    assert resolved.data_ptr() == tensor.data_ptr()
    assert resolved.stride() == tensor.stride()
    assert resolved.dtype == tensor.dtype and resolved.device == tensor.device
    assert not resolved.requires_grad
    assert set(_publish._TENSORS) == {"trace.node_f"}
    assert P._resolve("weight") is model.weight
    handle.remove()
    assert _publish._resolve("trace.node_f") is None
    assert not _publish._TENSORS


def test_republication_releases_old_reference_and_watch_replacement_clears():
    first, second = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    first_handle = attach(first)
    interp.publish("trace.counts", torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    old = weakref.ref(P._resolve("trace.counts"))
    replacement = torch.tensor([4, 3, 2, 1], dtype=torch.int32)
    interp.publish("trace.counts", replacement)
    assert old() is None
    current = weakref.ref(P._resolve("trace.counts"))
    second_handle = attach(second)
    assert current() is None
    with pytest.raises(P._Waiting):
        P._resolve("trace.counts")
    interp.publish("output.probs", torch.ones(2, 10))
    first_handle.remove()
    assert P._resolve("output.probs") is not None
    second_handle.remove()
    assert _publish._resolve("output.probs") is None
    with pytest.raises(P._Waiting):
        P._resolve("output.probs")


def test_publication_before_watch_joins_its_lifecycle():
    interp.publish("trace.counts", torch.zeros(4, dtype=torch.int32))
    assert P._resolve("trace.counts") is not None
    handle = attach(torch.nn.Linear(2, 2))
    assert P._resolve("trace.counts") is not None
    handle.remove()
    assert _publish._resolve("trace.counts") is None


@pytest.mark.parametrize("name", ["", "live:trace.x", "trace x", "trace\nx", 3, None])
def test_invalid_publication_names_fail(name):
    with pytest.raises(ValueError, match="publication name"):
        interp.publish(name, torch.ones(1))


def test_publication_type_and_model_name_conflicts():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    handle = attach(model)
    with pytest.raises(TypeError, match="torch.Tensor"):
        interp.publish("trace.x", [1, 2])
    for name in ("0.weight", "0.weight.grad", "0.act"):
        with pytest.raises(ValueError, match="conflicts"):
            interp.publish(name, torch.ones(1))
    handle.remove()
    interp.publish("0.weight", torch.ones(1))
    with pytest.raises(ValueError, match="conflicts"):
        attach(model)


def test_public_inference_watch_clears_publications_and_capture_hooks():
    model = torch.nn.Sequential(torch.nn.Embedding(8, 4), torch.nn.Linear(4, 8))
    watch = caliper.watch(model, activations=False, groups=False, scene=False)
    try:
        with pytest.raises(P._Waiting):
            P._resolve("1.act")
        model(torch.tensor([[1, 2]]))
        act_tensor = P._resolve("1.act")
        captured = weakref.ref(act_tensor)
        interp.publish("trace.node_f", act_tensor.clone())
        del act_tensor
        published = weakref.ref(P._resolve("trace.node_f"))
        watch.remove()
        gc.collect()
        assert captured() is None and published() is None
        assert not P._ACTS and not _publish._TENSORS
        assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    finally:
        watch.remove()


@contextmanager
def autorelease_pool():
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    objc.objc_autoreleasePoolPush.restype = ctypes.c_void_p
    objc.objc_autoreleasePoolPop.argtypes = [ctypes.c_void_p]
    pool = objc.objc_autoreleasePoolPush()
    try:
        yield
    finally:
        objc.objc_autoreleasePoolPop(pool)


MANIFEST = {
    "name": "published_values", "version": 1,
    "inputs": [{"name": "payload", "kind": "tensor", "dtype": "f32", "rank_min": 1, "rank_max": 1},
               {"name": "counts", "kind": "tensor", "dtype": "i32", "rank_min": 1, "rank_max": 1}],
    "resources": [{"id": "result", "kind": "buffer", "element": "f32", "count": 4}],
    "derive": {"passes": [{"entry": "derive_values", "writes": ["result"]}]},
}
DERIVE = """@compute @workgroup_size(4)
fn derive_values(@builtin(global_invocation_id) id: vec3<u32>) {
    result[id.x] = 2.0 * payload[id.x] + f32(counts[0]);
}
"""


def install(derive=DERIVE):
    return P.install_piece(
        "published_values", piece={"name": "published_values", "version": 1,
                                    "hash": "sha256:" + "0" * 64, "from": "workspace"},
        manifest=MANIFEST, bind={"payload": "live:trace.node_f", "counts": "live:trace.counts"},
        params={}, panel="probe", segments=[], common="", derive=derive, draw="",
    )


def derive_pass(step):
    # The existing facade resolves and rebinds storage, then its native runtime dispatches WGSL.
    with autorelease_pool(), P._LOCK:
        program = P._PROGS["published_values"]
        P._refresh(program)
        TORCH.framework_done()
        runtime = P._STATE["rt"]
        result = runtime.run([program.handle], step, float(step), 1)
        assert result["ran"] == 1 and result["stopped"] == 0
        data, count, stride, *_ = runtime.resource(program.handle, "result")
        assert count == 4 and stride == 4
        return struct.unpack("<4f", data)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="the producer runtime requires MPS")
def test_named_tensor_reaches_real_derive_and_rebinds_without_cpu_copy(monkeypatch):
    # An available device must load the real backend. A missing module fails this test.
    from caliper import _producer
    assert _producer.ABI_VERSION >= 3
    model = torch.nn.Linear(2, 2)
    handle = attach(model)
    tensor = torch.arange(1, 5, dtype=torch.float32, device="mps")
    counts = torch.tensor([2, 0, 0, 0], dtype=torch.int32, device="mps")
    def forbidden(*args, **kwargs):
        raise AssertionError("input publication must not copy to CPU")
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "numpy", forbidden)
        interp.publish("trace.node_f", tensor)
        interp.publish("trace.counts", counts)
        assert P._resolve("trace.node_f").data_ptr() == tensor.data_ptr()
        with autorelease_pool():
            installed = install()
        assert installed["state"] == "compiled", installed
        assert derive_pass(0) == (4.0, 6.0, 8.0, 10.0)
        program = P._PROGS["published_values"]
        compiled = program.handle
        original_import = program.bound["payload"].handle
        replacement = tensor + 10
        interp.publish("trace.node_f", replacement)
        assert derive_pass(1) == (24.0, 26.0, 28.0, 30.0)
        assert program.handle == compiled
        assert program.bound["payload"].handle != original_import
        assert original_import not in P._LIVE
    torch.testing.assert_close(tensor.cpu(), torch.arange(1, 5, dtype=torch.float32))
    with autorelease_pool():
        refused = install(DERIVE.replace("result[id.x] =", "payload[id.x] ="))
    assert refused["state"] == "refused", refused
    handle.remove()
    assert not P._PROGS and not P._IMPORTS and not P._LIVE and not _publish._TENSORS


def test_conflicting_watch_attach_removes_its_partial_hooks():
    model = torch.nn.Linear(2, 2)
    interp.publish("weight", torch.ones(1))
    handle = WatchHandle()
    handle._handles.append(model.register_forward_hook(lambda *args: None))
    with pytest.raises(ValueError, match="conflicts"):
        P.attach(model, None, handle)
    assert not model._forward_hooks
    assert not handle.active
