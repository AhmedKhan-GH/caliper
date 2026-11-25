"""Bounded CPU events for producer readiness and input delivery."""
from collections import OrderedDict
import itertools
import struct
from types import SimpleNamespace

import pytest
import torch

import caliper
from caliper import _programs as P
from caliper._watch import WatchHandle
from caliper.interp import publish


class Runtime:
    """Read CPU imports through the facade's compile, rebind, and output paths."""

    def __init__(self):
        self.ids = itertools.count(1)
        self.imports, self.programs, self.values = {}, {}, {}
        self.runs, self.bindings = [], []
        self.ran = True

    def import_tensor(self, name, tensor):
        key = P._storage_key(tensor)
        previous = P._IMPORTS.get(name)
        if previous is not None and previous.key == key:
            return previous
        handle = next(self.ids)
        self.imports[handle] = tensor
        imported = P._Import(name, handle, key)
        P._IMPORTS[name] = P._LIVE[handle] = imported
        return imported

    def program(self, pid, source, inputs, kind, **kwargs):
        handle = next(self.ids)
        self.programs[handle] = dict(inputs)
        return handle

    def derive(self, pid, source, common, inputs, resources, passes, params, **kwargs):
        self.bindings.append((inputs, kwargs["unbound"]))
        return self.program(pid, source, [(name, h) for name, h, _ in inputs if h], "piece")

    def rebind(self, handle, old, new):
        self.programs[handle] = {name: new if h == old else h
                                 for name, h in self.programs[handle].items()}

    def run(self, handles, step, *args):
        if self.ran:
            for handle in handles:
                self.values[handle] = sum(float(self.imports[h].sum())
                                          for h in self.programs[handle].values())
            self.runs.append(step)
        return {"ran": self.ran}

    def status(self, handle):
        return {"state": "compiled", "memory_bytes": 4}

    def output(self, handle, name):
        return torch.tensor([self.values[handle]]), 1

    def resource(self, handle, name):
        return struct.pack("<f", self.values[handle]), 1, 4, 0, 0, 0

    def remove(self, handle):
        self.programs.pop(handle, None)
        self.values.pop(handle, None)

    def release(self, handle):
        self.imports.pop(handle, None)


@pytest.fixture
def producer(monkeypatch):
    P.detach()
    runtime = Runtime()
    clock = SimpleNamespace(now=0.)
    shown = []
    monkeypatch.setitem(P._STATE, "rt", runtime)
    monkeypatch.setitem(P._STATE, "marks", False)
    monkeypatch.setitem(P._STATE, "step", None)
    monkeypatch.setitem(P._STATE, "pass", None)
    monkeypatch.setitem(P._STATE, "pass_budget", True)
    monkeypatch.setattr(P, "_clock", lambda: clock.now)
    monkeypatch.setattr(P, "_import", runtime.import_tensor)
    monkeypatch.setattr(P, "_controls_ports", lambda: [])
    adapter = SimpleNamespace(framework_done=lambda: None)
    monkeypatch.setattr(P, "_torch_adapter", lambda: adapter)
    monkeypatch.setattr(caliper, "show", lambda t, **kwargs: shown.append(t.clone()))
    monkeypatch.setattr(caliper, "remove", lambda *args: None)
    model = torch.nn.Sequential(OrderedDict(main=torch.nn.Identity(), other=torch.nn.Identity()))
    watch = WatchHandle()
    P.attach(model, None, watch)

    def step(number):
        clock.now = float(number)
        P.at_step(number)

    yield SimpleNamespace(runtime=runtime, model=model, watch=watch, step=step, shown=shown)
    P.detach()
    P._CAD.reset()
    assert not runtime.imports and not P._ACT_HOOKS and not P._ACTS


def install_piece(*, source="live:trace.node_f", extra=None, every=1):
    inputs = [{"name": "payload", "kind": "tensor", "dtype": "f32"}, *(extra or [])]
    manifest = {
        "name": "readiness", "version": 1, "inputs": inputs,
        "resources": [{"id": "result", "kind": "buffer", "element": "f32", "count": 1}],
        "derive": {"every": every, "passes": [{"entry": "derive_values", "writes": ["result"]}]},
    }
    return P.install_piece(
        "readiness", piece={"name": "readiness", "version": 1,
                            "hash": "sha256:" + "0" * 64, "from": "workspace"},
        manifest=manifest, bind={"payload": source}, derive="fake derive", panel="probe")


def warmup(producer):
    # These eight bounded events retain the production warmup threshold.
    for number in range(1, P.WARMUP_STEPS + 1):
        producer.step(number)
        assert not producer.runtime.runs


@pytest.mark.parametrize("extra", [
    [{"name": "label_text", "optional": True}],
    [{"name": "viewer_data", "kind": "buffer"}],
    [{"name": "label_text", "optional": True}, {"name": "viewer_data", "kind": "buffer"}],
])
def test_unbound_piece_inputs_wait_for_required_publication(producer, extra):
    assert install_piece(extra=extra)["state"] == "waiting"
    assert not P.inference_pending()
    warmup(producer)
    producer.step(9)
    assert not producer.runtime.runs and not P.inference_pending()
    publish("trace.edge_f", torch.ones(1))
    assert not P.inference_pending()
    publish("trace.node_f", torch.tensor([7.]))
    assert P.inference_pending()
    producer.step(10)
    assert producer.runtime.runs == [10] and not P.inference_pending()
    inputs, unbound = producer.runtime.bindings[-1]
    assert all(handle == 0 for name, handle, _ in inputs if name != "payload")
    assert unbound == [x["name"] for x in extra if x.get("optional")]


def test_readiness_resolves_inputs_without_native_calls(producer, monkeypatch):
    install_piece(extra=[{"name": "label_text", "optional": True}])

    def forbidden(*args, **kwargs):
        raise AssertionError("readiness must not call the native runtime or import tensors")

    monkeypatch.setattr(P, "_runtime", forbidden)
    monkeypatch.setattr(P, "_import", forbidden)
    monkeypatch.setattr(P, "_torch_adapter", forbidden)
    assert not P.inference_pending()
    publish("trace.node_f", torch.ones(1))
    assert P.inference_pending()


def test_unknown_bound_source_still_reaches_a_terminal_refusal(producer):
    install_piece()
    program = P._PROGS["readiness"]
    # An unknown bound source must take precedence over a missing publication.
    program.inputs.append(("unknown", "live:unknown"))
    program.types["unknown"] = ("tensor", "f32")
    assert P.inference_pending()
    warmup(producer)
    producer.step(9)
    assert program.state == "refused"
    assert program.error["code"] == "program.unknown_input"
    assert not producer.runtime.runs and not P.inference_pending()
