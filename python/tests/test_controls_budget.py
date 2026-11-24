"""CPU checks for the complete CONTROLS metadata budget."""
from copy import deepcopy

import pytest

from caliper import _controls as controls
from caliper import _wire as wire
from caliper.interp._intervene import InterventionError


VECTOR = [0.1234567890123456] * 1024


class Model:
    def named_modules(self):
        yield "", self

    def parameters(self):
        return iter(())


class Adapter:
    def points(self, model):
        return [{"point": f"layers.{layer}.{probe}", "kind": "module",
                 "module": f"layers.{layer}", "head": None, "type": "Fixture", "shape": None}
                for layer in range(24) for probe in ("resid_pre", "mlp_in", "mlp_out")]


class Hook:
    def __init__(self, callback):
        self.callback = callback
        self.removed = False

    def __call__(self):
        self.removed = True


@pytest.fixture
def case(monkeypatch):
    ctl, model = controls.Controls(), Model()
    service = ctl.interventions_on(model, adapter=Adapter())

    def target(point):
        assert point in service.names
        return model, None, 1024, None

    def install(spec, *, on_error):
        service.executor.validate(spec)
        return Hook(on_error)

    # Keep the real specification validator, including vector width and wire bounds.
    # Only target resolution and hook installation use a model stub.
    monkeypatch.setattr(service.executor, "_target", target)
    monkeypatch.setattr(service.executor, "install", install)
    yield ctl, service
    service.remove()


def spec(index, **changes):
    layer, probe = divmod(index, 3)
    return {"point": f"layers.{layer}.{('resid_pre', 'mlp_in', 'mlp_out')[probe]}",
            "kind": "add", "vector": list(VECTOR), "coefficient": 0,
            "from_step": 100, **changes}


def encoded(ctl):
    meta = ctl.controls_meta()
    wire.decode(wire.encode_frame(wire.MsgType.CONTROLS, meta))
    return len(wire.canonical_json(meta))


def state(service, count, *, lifecycle="ended", **changes):
    records = []
    for index in range(count):
        request = service.executor.validate(spec(index, **changes))
        records.append({"specification": request, "state": lifecycle, "clear": False,
                        "since_step": None if lifecycle == "pending" else 1,
                        "applied_specification": deepcopy(request) if lifecycle == "active" else None,
                        "error": None, "error_code": None})
    return {"version": 1, "interventions": records}


def fill_budget(ctl):
    """Find the admission boundary through public calls, without a size formula."""
    for index in range(controls.MAX_PARAMS):
        param = ctl.param(f"padding{index}", "")
        low, high = 0, controls.PARAM_TEXT_MAX
        try:
            param.set("\x01" * high)
            continue
        except ValueError:
            pass
        while low < high:
            middle = (low + high + 1) // 2
            try:
                param.set("\x01" * middle)
                low = middle
            except ValueError:
                high = middle - 1
        param.set("\x01" * low)
        return param
    pytest.fail("the wire budget did not refuse a bounded text value")


@pytest.mark.parametrize("admission", ["request", "restore"])
def test_smaller_replacement_reserves_the_retained_specification(case, monkeypatch, admission):
    ctl, service = case
    service.set(spec(0, from_step=1, until_step=1000))
    ctl.step()
    service.set(spec(0, from_step=None, until_step=None, vector=[0.] * 1024))
    if admission == "restore":
        service.load_state_dict(service.state_dict(), restore_active=True)
    observed = []
    action = ctl.action("inspect", lambda: observed.append(service.specifications()))
    pad = fill_budget(ctl)
    with pytest.raises(ValueError, match="wire limit"):
        pad.set(pad.value + "\x01")
    record = next(iter(ctl._ivs.values()))
    hook, applied, request = record.hook, record.applied_spec, record.spec
    since = record.since_step
    saved = service.state_dict()
    assert control(ctl, "run", name="inspect")["status"] == "queued"

    def fail(*args, **kwargs):
        raise RuntimeError("\x01" * controls.TEXT_MAX)

    with monkeypatch.context() as patch:
        patch.setattr(service.executor, "install", fail)
        with pytest.warns(RuntimeWarning, match="failed"):
            ctl.step()
    assert record.hook is hook and not hook.removed and record.applied_spec is applied
    assert record.spec is request and record.spec == applied and not record.replace
    assert (record.from_step, record.until_step, record.since_step) == (1, 1000, since)
    assert record.error_code == "intervene.hook_failed"
    assert record.error == ctl._stop_message("RuntimeError: " + "\x01" * controls.TEXT_MAX)
    assert service.state_dict() != saved
    assert not observed and action.queued
    size = encoded(ctl)
    assert wire.MAX_META_BYTES - 3000 < size <= wire.MAX_META_BYTES
    resolved = service.state_dict()
    ctl.checkpoint()
    assert observed == [[applied]] and not action.queued
    for restore_active in (False, True):
        service.load_state_dict(resolved, restore_active=restore_active)
        ctl.checkpoint()
        assert service.state_dict() == resolved
        assert service.specifications() == [applied]
        encoded(ctl)
    ctl.step(999)
    assert not service.specifications()
    assert next(iter(ctl._ivs.values())).state == "ended"
    encoded(ctl)


class Transport:
    def __init__(self):
        self.frames = []

    def broadcast(self, kind, meta, **kwargs):
        self.frames.append(wire.decode(wire.encode_frame(kind, meta)))

    def reply(self, connection, meta):
        wire.decode(wire.encode_frame(wire.MsgType.CONTROL_REPLY, meta))

    def controllers(self):
        return 1

    def may_control(self, connection):
        return True


def control(ctl, op, **fields):
    route = controls.Route(ctl, Transport())
    result = route.on_control("viewer", {"request_id": 1, "op": op, **fields})
    wire.decode(wire.encode_frame(wire.MsgType.CONTROL_REPLY, result))
    return result


def test_aggregate_vectors_refuse_before_the_wire_cap(case):
    assert controls.MAX_META_BYTES == wire.MAX_META_BYTES
    ctl, service = case
    refused = None
    for index in range(64):
        request = spec(index)
        wire.decode(wire.encode(wire.Control(index + 1, "intervene", intervene={"set": request})))
        before = service.state_dict()
        try:
            service.set(request)
        except InterventionError as error:
            assert error.code == wire.ERR_LIMIT
            assert service.state_dict() == before
            refused = index
            break
    size = len(wire.canonical_json(ctl.controls_meta()))
    print(f"admitted={len(ctl._ivs)} metadata_bytes={size} wire_cap={wire.MAX_META_BYTES}")
    assert refused is not None, f"all 64 requests were admitted with {size} metadata bytes"
    assert 1 < refused < 64
    encoded(ctl)


def test_near_boundary_history_preserves_exact_vectors(case):
    ctl, service = case
    saved = state(service, 50)
    service.load_state_dict(saved)
    pad = fill_budget(ctl)
    size = encoded(ctl)
    print(f"near_boundary_bytes={size} wire_cap={wire.MAX_META_BYTES}")
    assert wire.MAX_META_BYTES - 100 < size <= wire.MAX_META_BYTES
    assert service.state_dict() == saved
    assert all(entry["vector"] == VECTOR for entry in ctl.controls_meta()["interventions"]["active"])
    before = ctl.controls_meta()
    with pytest.raises(ValueError, match="wire limit"):
        pad.set(pad.value + "\x01")
    assert ctl.controls_meta() == before
    ctl.step(controls._INT_EXACT - 2)
    assert control(ctl, "step", count=2**32 - 1)["status"] == "ok"
    encoded(ctl)
    assert control(ctl, "resume")["status"] == "ok"


@pytest.mark.parametrize("route", ["script", "viewer", "override"])
def test_replacement_refusal_retains_installed_hook_and_pending_state(case, route):
    ctl, service = case
    service.set(spec(0, from_step=None, vector=[0.] * 1024))
    ctl.step()
    service.set(spec(0, from_step=None, vector=[0.] * 1024, coefficient=1))
    fill_budget(ctl)
    before = service.state_dict()
    record = next(iter(ctl._ivs.values()))
    hook = record.hook
    request = spec(0, from_step=None)
    if route == "viewer":
        reply = control(ctl, "intervene", intervene={"set": request})
        assert reply["error"]["code"] == wire.ERR_LIMIT
    else:
        with pytest.raises(InterventionError) as error:
            service.set(request) if route == "script" else service.apply_overrides([request])
        assert error.value.code == wire.ERR_LIMIT
    assert service.state_dict() == before
    assert next(iter(ctl._ivs.values())) is record
    assert record.hook is hook and not hook.removed and record.replace
    encoded(ctl)
    hook.callback("failure after refused replacement")
    assert record.state == "ended" and hook.removed
    encoded(ctl)


@pytest.mark.parametrize("restore_active", [False, True])
@pytest.mark.parametrize("lifecycle", ["pending", "active", "ended"])
def test_restore_overflow_is_atomic_before_hook_installation(case, monkeypatch, restore_active, lifecycle):
    ctl, service = case
    service.set(spec(0, from_step=None, vector=[0.] * 1024))
    ctl.step()
    before = service.state_dict()
    hook = next(iter(ctl._ivs.values())).hook
    saved = state(service, 64, lifecycle=lifecycle)
    monkeypatch.setattr(service.executor, "install", lambda *a, **k: pytest.fail("restore installed a hook"))
    with pytest.raises(InterventionError) as error:
        service.load_state_dict(saved, restore_active=restore_active)
    assert error.value.code == wire.ERR_LIMIT
    assert service.state_dict() == before
    assert next(iter(ctl._ivs.values())).hook is hook and not hook.removed
    encoded(ctl)


@pytest.mark.parametrize("route", ["set", "override"])
def test_budget_eviction_removes_only_oldest_ended_records(case, route):
    ctl, service = case
    service.load_state_dict(state(service, 49))
    service.set(spec(49, from_step=None, vector=[0.] * 1024))
    ctl.step()
    live = ctl._ivs[(spec(49)["point"], "add")]
    fill_budget(ctl)
    request = spec(50)
    service.set(request) if route == "set" else service.apply_overrides([request])
    assert (spec(0)["point"], "add") not in ctl._ivs
    assert (spec(48)["point"], "add") in ctl._ivs
    assert (spec(50)["point"], "add") in ctl._ivs
    assert service.specifications() == [live.spec]
    if route == "set":
        assert ctl._ivs[(live.point, live.op)] is live and not live.hook.removed
    encoded(ctl)


def test_refusal_does_not_evict_history_when_it_cannot_free_enough_space(case):
    ctl, service = case
    service.load_state_dict(state(service, 1, vector=[0.] * 1024))
    fill_budget(ctl)
    before = service.state_dict()
    with pytest.raises(InterventionError) as error:
        service.set(spec(1))
    assert error.value.code == wire.ERR_LIMIT
    assert service.state_dict() == before
    encoded(ctl)


@pytest.mark.parametrize("transition", ["callback", "install_error", "poll_error", "configure", "linger", "expire", "clear"])
def test_lifecycle_publication_fits_after_near_boundary_admission(case, monkeypatch, transition):
    ctl, service = case
    service.load_state_dict(state(service, 49))
    service.set(spec(49, until_step=controls._INT_EXACT - 1))
    fill_budget(ctl)
    transport = Transport()
    route = ctl.attach(transport)
    message = "\x01" * 1024 + "\0\ud800é"
    try:
        if transition == "install_error":
            def fail(*args, **kwargs):
                raise ValueError(message)
            monkeypatch.setattr(service.executor, "install", fail)
            with pytest.warns(RuntimeWarning, match="failed"):
                ctl.step(controls._INT_EXACT - 3)
        else:
            ctl.step(controls._INT_EXACT - 3)
            record = ctl._ivs[(spec(49)["point"], "add")]
            assert record.state == "active"
            if transition in ("callback", "poll_error"):
                if transition == "callback":
                    record.hook.callback(message)
                else:
                    record.hook.errors, record.hook.error = 1, message
                with pytest.warns(RuntimeWarning, match="failed"):
                    ctl.checkpoint()
            elif transition == "configure":
                service.configure()
            elif transition == "linger":
                with ctl.lock, pytest.warns(RuntimeWarning, match="no connection"):
                    ctl._alone_since = float("-inf")
                    assert ctl._linger(0)
                ctl._publish()
            elif transition == "expire":
                ctl.step(controls._INT_EXACT - 2)
            else:
                service.clear(spec(49)["point"])
                ctl.step()
        assert not service.specifications()
        assert transport.frames
        for entry in ctl.controls_meta()["interventions"]["active"]:
            if "error" in entry:
                assert len(entry["error"].encode("utf-8")) <= 1024
                assert "\0" not in entry["error"]
        encoded(ctl)
    finally:
        route.detach()


def test_every_live_hook_can_publish_maximum_error_together(case):
    ctl, service = case
    for index in range(39):
        service.set(spec(index))
    fill_budget(ctl)
    ctl.step(controls._INT_EXACT - 2)
    for record in ctl._ivs.values():
        record.hook.callback("\x01" * 1024)
    with pytest.warns(RuntimeWarning, match="failed"):
        ctl.checkpoint()
    size = encoded(ctl)
    print(f"simultaneous_errors_bytes={size} records={len(ctl._ivs)}")
    assert wire.MAX_META_BYTES - 20000 < size <= wire.MAX_META_BYTES
    assert len(ctl._ivs) == 39
    assert all(record.state == "ended" for record in ctl._ivs.values())


@pytest.mark.parametrize("operation", ["param", "param_replace", "script_value", "viewer_value", "action", "action_replace"])
def test_later_control_growth_is_transactional(case, operation):
    ctl, service = case
    param = ctl.param("text", "")
    action = ctl.action("task", lambda: None)
    service.set(spec(0, from_step=None))
    ctl.step()
    assert control(ctl, "set", name=param.name, value="old")["status"] == "ok"
    assert control(ctl, "run", name=action.name)["status"] == "queued"
    fill_budget(ctl)
    before = ctl.controls_meta()
    version, pending, queue = param._version, ctl._npending, list(ctl.queue)
    hook = next(iter(ctl._ivs.values())).hook
    if operation == "viewer_value":
        reply = control(ctl, "set", name=param.name, value="é" * 2048)
        assert reply["error"]["code"] == wire.ERR_LIMIT
    else:
        with pytest.raises(ValueError, match="wire limit"):
            if operation in ("param", "param_replace"):
                ctl.param("new" if operation == "param" else param.name, "é" * 2048)
            elif operation == "script_value":
                param.set("é" * 2048)
            else:
                ctl.action("new" if operation == "action" else action.name, lambda: None, help="\x01" * 1024)
    assert ctl.controls_meta() == before
    assert (param._version, ctl._npending, list(ctl.queue)) == (version, pending, queue)
    assert not param.removed and not action.removed and not hook.removed
    assert ctl.params[param.name] is param and ctl.actions[action.name] is action
    encoded(ctl)
    ctl.step()
    assert param.value == "old" and not action.queued
    encoded(ctl)


def test_point_refresh_refusal_preserves_effects_and_previous_advertisement(case, monkeypatch):
    ctl, service = case
    service.set(spec(0))
    fill_budget(ctl)
    before = deepcopy(service.points)
    larger = [dict(point, shape=[controls._INT_EXACT - 1] * wire.MAX_DIMS) for point in before]
    monkeypatch.setattr(service.adapter, "points", lambda model: larger)
    with pytest.warns(RuntimeWarning, match="point metadata update refused"):
        ctl.step(100)
    assert service.points == before
    assert len(service.specifications()) == 1
    encoded(ctl)


def test_oversize_point_advertisement_keeps_previous_model_and_hooks(case):
    ctl, service = case
    service.set(spec(0, from_step=None))
    ctl.step()
    hook = next(iter(ctl._ivs.values())).hook
    before = ctl.controls_meta()

    class LargeAdapter(Adapter):
        def points(self, model):
            return [{"point": f"{index:04}" + "p" * 251, "module": "m" * 255,
                     "kind": "module", "head": None, "type": "Fixture", "shape": None}
                    for index in range(2048)]

    with pytest.raises(ValueError, match="wire limit"):
        ctl.interventions_on(Model(), adapter=LargeAdapter())
    assert ctl.intervention_service is service
    assert ctl.controls_meta() == before and not hook.removed
    encoded(ctl)


def test_existing_controls_leave_no_room_for_another_effect(case):
    ctl, service = case
    ctl.action("task", lambda: None, help="\x01" * 1024)
    fill_budget(ctl)
    before = ctl.controls_meta()
    with pytest.raises(InterventionError) as error:
        service.set(spec(0))
    assert error.value.code == wire.ERR_LIMIT
    assert ctl.controls_meta() == before
    encoded(ctl)


def test_one_valid_large_vector_needs_lifecycle_headroom(case, monkeypatch):
    from caliper import _intervention_protocol as metadata
    ctl, service = case
    monkeypatch.setattr(service.executor, "_target", lambda point: (service.model, None, 16384, None))
    request = spec(0, vector=[0.1234567890123] * 16300 + [0.] * 84)
    validated = service.executor.validate(request)
    assert metadata.specification(validated)
    assert len(wire.canonical_json(validated)) > metadata.MAX_BYTES - 2000
    before = ctl.controls_meta()
    with pytest.raises(InterventionError, match="lifecycle headroom") as error:
        service.set(request)
    assert error.value.code == wire.ERR_LIMIT
    assert ctl.controls_meta() == before


@pytest.mark.parametrize("error_fields", [{"error": "retired"}, {"error_code": "retired"}])
def test_retained_optional_error_fields_remain_wire_valid(case, error_fields):
    ctl, service = case
    saved = state(service, 1)
    saved["interventions"][0].update(error_fields)
    service.load_state_dict(saved)
    assert service.state_dict() == saved
    entry, = ctl.controls_meta()["interventions"]["active"]
    assert all(entry[key] == value for key, value in error_fields.items())
    assert all(entry[key] is not None for key in ("error", "error_code") if key in entry)
    encoded(ctl)


def test_existing_large_step_is_part_of_admission():
    ctl = controls.Controls()
    ctl.step(10**100)
    fill_budget(ctl)
    size = encoded(ctl)
    assert wire.MAX_META_BYTES - 100 < size <= wire.MAX_META_BYTES


def test_all_64_small_vectors_remain_valid_and_visible(case):
    ctl, service = case
    for index in range(64):
        service.set(spec(index, vector=[0.] * 1024))
    ctl.step(100)
    assert len(service.specifications()) == len(ctl.controls_meta()["interventions"]["active"]) == 64
    hooks = [record.hook for record in ctl._ivs.values()]
    before = service.state_dict()
    with pytest.raises(InterventionError) as error:
        service.set(spec(64, vector=[0.] * 1024))
    assert error.value.code == wire.ERR_LIMIT
    assert service.state_dict() == before
    assert all(not hook.removed for hook in hooks)
    encoded(ctl)
    service.clear()
    ctl.step()
    assert not ctl._ivs and all(hook.removed for hook in hooks)


@pytest.mark.parametrize("restore_active", [False, True])
def test_near_boundary_restore_retains_history_and_applied_effect(case, restore_active):
    ctl, service = case
    service.load_state_dict(state(service, 49))
    service.set(spec(49, from_step=None))
    ctl.step()
    fill_budget(ctl)
    saved = service.state_dict()
    old_hook = ctl._ivs[(spec(49)["point"], "add")].hook
    service.load_state_dict(saved, restore_active=restore_active)
    assert old_hook.removed
    if not restore_active:
        ctl.checkpoint()
    assert service.state_dict() == saved
    assert len(service.specifications()) == 1
    assert not ctl._ivs[(spec(49)["point"], "add")].hook.removed
    encoded(ctl)
