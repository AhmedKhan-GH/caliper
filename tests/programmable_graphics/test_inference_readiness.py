"""Bounded CPU events for producer readiness and input delivery."""
from collections import OrderedDict
import itertools
import struct
from types import SimpleNamespace

import pytest
import torch

import caliper
from caliper import _programs as P, _wire
from caliper._watch import WatchHandle
from caliper.interp import _publish, publish


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
    monkeypatch.setattr(caliper, "show", lambda t, *args, **kwargs: shown.append(t.clone()))
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


def install_piece(*, source="live:trace.node_f", extra=None, every=1, bindings=None):
    inputs = [{"name": "payload", "kind": "tensor", "dtype": "f32"}, *(extra or [])]
    manifest = {
        "name": "readiness", "version": 1, "inputs": inputs,
        "resources": [{"id": "result", "kind": "buffer", "element": "f32", "count": 1}],
        "derive": {"every": every, "passes": [{"entry": "derive_values", "writes": ["result"]}]},
    }
    return P.install_piece(
        "readiness", piece={"name": "readiness", "version": 1,
                            "hash": "sha256:" + "0" * 64, "from": "workspace"},
        manifest=manifest, bind={"payload": source, **(bindings or {})},
        derive="fake derive", panel="probe")


def install_program(source="live:main.act", *, pid="activation", every=1):
    return P.install(pid, {"inputs": [{"var": "payload", "source": source}],
                          "output": {"kind": "tensor", "shape": [1]},
                          "limits": {"every": every}}, "fake program")


def warmup(producer):
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


@pytest.mark.parametrize("source", ["live:trace.node_f", "live:main.act"])
def test_readiness_resolves_inputs_without_native_calls(producer, monkeypatch, source):
    install_piece(source=source, extra=[{"name": "label_text", "optional": True}])

    def forbidden(*args, **kwargs):
        raise AssertionError("readiness must not call the native runtime or import tensors")

    monkeypatch.setattr(P, "_runtime", forbidden)
    monkeypatch.setattr(P, "_import", forbidden)
    monkeypatch.setattr(P, "_torch_adapter", forbidden)
    assert not P.inference_pending()
    if source == "live:main.act":
        producer.model.main(torch.ones(1))
    else:
        publish("trace.node_f", torch.ones(1))
    assert P.inference_pending()


def test_unknown_bound_source_still_reaches_a_terminal_refusal(producer):
    install_piece()
    program = P._PROGS["readiness"]
    program.inputs.append(("unknown", "live:unknown"))
    program.types["unknown"] = ("tensor", "f32")
    assert P.inference_pending()
    warmup(producer)
    producer.step(9)
    assert program.state == "refused"
    assert program.error["code"] == "program.unknown_input"
    assert not producer.runtime.runs and not P.inference_pending()


@pytest.mark.parametrize("kind", ["program", "piece"])
@pytest.mark.parametrize("every", [2, 5])
def test_activation_changes_stay_pending_through_cadence(producer, kind, every):
    if kind == "program":
        installed = install_program(every=every)
    else:
        installed = install_piece(source="live:main.act", every=every,
                                  extra=[{"name": "label_text", "optional": True}])
    assert installed["state"] == "waiting" and not P.inference_pending()
    producer.model.main(torch.tensor([1.]))
    assert P.inference_pending()
    warmup(producer)
    assert P.inference_pending()
    producer.step(9)
    assert producer.runtime.runs == [9] and not P.inference_pending()
    publish("trace.edge_f", torch.ones(1))
    with pytest.raises(P._Waiting):
        P._resolve("other.act")
    producer.model.other(torch.tensor([99.]))
    assert not P.inference_pending()
    producer.model.main(torch.tensor([7.]))
    assert P.inference_pending()
    next_pass = 9 + max(P.N_MIN, every)
    for number in range(10, next_pass):
        producer.step(number)
        assert producer.runtime.runs == [9] and P.inference_pending()
    producer.step(next_pass)
    assert producer.runtime.runs == [9, next_pass] and not P.inference_pending()
    assert list(producer.runtime.values.values()) == [7.]
    assert not _publish._VERSIONS.get("main.act")


def test_same_activation_storage_and_shared_consumers_keep_separate_evidence(producer):
    install_program(pid="fast", every=2)
    install_program(pid="slow", every=5)
    tensor = torch.tensor([1.])
    producer.model.main(tensor)
    warmup(producer)
    producer.step(9)
    tensor.add_(6)
    producer.model.main((None, tensor))
    assert P.inference_pending()
    producer.step(10)
    producer.step(11)
    producer.step(12)
    assert P._PROGS["fast"].last_step == 12
    assert P._PROGS["slow"].last_step == 9 and P.inference_pending()
    producer.step(13)
    producer.step(14)
    assert P.inference_pending()
    producer.step(15)
    assert not P.inference_pending()
    assert [float(t.item()) for t in producer.shown] == [1., 1., 7., 7., 7.]


def test_activation_change_still_waits_for_a_bound_optional_publication(producer):
    install_piece(source="live:main.act", extra=[{"name": "labels", "optional": True}],
                  bindings={"labels": "live:trace.label_text"})
    producer.model.main(torch.ones(1))
    assert not P.inference_pending()
    warmup(producer)
    producer.step(9)
    assert not producer.runtime.runs and not P.inference_pending()
    publish("trace.label_text", torch.ones(1))
    assert P.inference_pending()
    producer.step(10)
    assert producer.runtime.runs == [10] and not P.inference_pending()


def test_dispatch_without_a_successful_pass_retains_activation_evidence(producer):
    install_program()
    producer.model.main(torch.ones(1))
    warmup(producer)
    producer.step(9)
    producer.model.main(torch.tensor([7.]))
    producer.runtime.ran = False
    producer.step(10)
    producer.step(11)
    producer.step(12)
    assert producer.runtime.runs == [9] and P.inference_pending()
    producer.runtime.ran = True
    producer.step(13)
    assert producer.runtime.runs == [9, 13] and not P.inference_pending()
    assert float(producer.shown[-1].item()) == 7.


@pytest.mark.parametrize("delivery", ["backlog", "busy"])
def test_resource_delivery_retry_retains_pending_work(producer, monkeypatch, delivery):
    posts, backlog = [], SimpleNamespace(bytes=0)
    link = SimpleNamespace(caps={_wire.CAP_RESOURCES}, attached=True, closed=False,
                           backlog=lambda: backlog.bytes)
    control = SimpleNamespace(server=SimpleNamespace(producer=SimpleNamespace(links=[link])),
                              post=lambda ln, frame, cap: posts.append(_wire.decode_frame(bytes(frame))))
    monkeypatch.setattr(P, "_controls_ports", lambda: [control])
    install_piece(source="live:main.act")
    producer.model.main(torch.ones(1))
    warmup(producer)
    producer.step(9)
    assert not P.inference_pending()
    producer.model.main(torch.tensor([7.]))
    if delivery == "backlog":
        backlog.bytes = P.RESOURCE_BACKLOG_BYTES
    with monkeypatch.context() as patch:
        if delivery == "busy":
            patch.setattr(P, "_all_busy", lambda p: True)
        producer.step(10)
        producer.step(11)
        producer.step(12)
    assert P.inference_pending()
    assert len([frame for frame in posts if frame.type == P.RESOURCE_TYPE]) == 1
    backlog.bytes = 0
    producer.step(13)
    producer.step(14)
    assert P.inference_pending()
    producer.step(15)
    assert not P.inference_pending()
    assert len([frame for frame in posts if frame.type == P.RESOURCE_TYPE]) == 2
    assert P._PROGS["readiness"].skipped == 1


@pytest.mark.parametrize("action", ["remove", "detach", "replacement", "refusal", "stop"])
def test_activation_evidence_ends_with_its_program_or_watch(producer, monkeypatch, action):
    install_program()
    producer.model.main(torch.ones(1))
    warmup(producer)
    producer.step(9)
    program = P._PROGS["activation"]
    old_hook = next(iter(producer.model.main._forward_hooks.values()))
    producer.model.main(torch.tensor([7.]))
    assert P.inference_pending()
    if action == "remove":
        P.remove(program.id)
    elif action == "detach":
        publish("trace.node_f", torch.ones(1))
        producer.watch.remove()
        assert not _publish._TENSORS and not _publish._VERSIONS
    elif action == "replacement":
        replacement = torch.nn.Sequential(OrderedDict(main=torch.nn.Identity()))
        P.attach(replacement, None)
        producer.watch.remove()
        assert P._STATE["model"] is replacement
    else:
        if action == "refusal":
            def refuse(*args):
                raise P._Refused("program.import_refused", "CPU fixture refusal")
            monkeypatch.setattr(P, "_import", refuse)
        else:
            monkeypatch.setattr(producer.runtime, "status", lambda h: {"state": "stopped"})
        producer.step(10)
        producer.step(11)
        producer.step(12)
        assert program.state == ("refused" if action == "refusal" else "stopped")
    assert not P.inference_pending()
    assert not P._ACTS and not P._ACT_HOOKS and not producer.model.main._forward_hooks
    assert not program.act_dirty
    if action == "replacement":
        assert install_program()["state"] == "waiting"
        producer.model.main(torch.tensor([99.]))
        old_hook(producer.model.main, (), torch.tensor([99.]))
        assert not P.inference_pending()
        replacement.main(torch.ones(1))
        assert P.inference_pending()


def test_hosted_readiness_follows_service_registration(producer, monkeypatch):
    from caliper import _controls
    from caliper.interp._service import InterpretationService, register_service, unregister_service

    controls = _controls.Controls()
    monkeypatch.setattr(_controls, "CONTROLS", controls)
    service = InterpretationService(model=producer.model)
    service.owner = object()
    register_service(service, controls=controls)
    try:
        install_piece()
        publish("trace.node_f", torch.tensor([7.]))
        warmup(producer)
        producer.step(9)
        assert producer.runtime.runs == [9]
        for number in range(10, 16):
            producer.step(number)
        assert producer.runtime.runs == [9]
        publish("trace.node_f", torch.tensor([8.]))
        producer.step(16)
        producer.step(17)
        assert producer.runtime.runs == [9, 17]
    finally:
        unregister_service(service, controls=controls)
    assert controls.interpretation_service is None
    for number in range(18, 21):
        producer.step(number)
    assert producer.runtime.runs == [9, 17, 20]


def test_new_watch_starts_without_closed_hosted_budget_history(producer, monkeypatch):
    from caliper import _controls
    from caliper.interp._service import InterpretationService, register_service, unregister_service

    controls = _controls.Controls()
    monkeypatch.setattr(_controls, "CONTROLS", controls)
    monkeypatch.setattr(caliper, "_CTL", controls)
    service = InterpretationService(model=producer.model)
    service.owner = object()
    register_service(service, controls=controls)
    install_program()
    caliper.step()
    P._CAD.base.append(1.)
    P._CAD.cost.append(100.)
    P._CAD.n, P._CAD.want, P._CAD.ceiling, P._CAD.over = 10, 100, 10, P.OVER_UPDATES
    unregister_service(service, controls=controls)
    P.detach()
    assert not P.ACTIVE and not P._PROGS
    P.attach(producer.model, None, WatchHandle())
    assert install_program()["state"] == "waiting"
    assert not P._CAD.base and not P._CAD.cost and not P._CAD.over_budget(1)
    producer.model.main(torch.ones(1))
    warmup(producer)
    producer.step(9)
    assert producer.runtime.runs == [9]
    assert P._PROGS["activation"].state == "running"
