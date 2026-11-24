"""File imports retain the control codec rules without the Caliper package."""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
WIRE = ROOT / "python" / "caliper" / "_wire.py"


def _load(path, name, patch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    patch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def standalone(monkeypatch):
    # Refuse package imports even when another test uses Caliper.
    for name in tuple(sys.modules):
        if name.startswith("caliper."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "caliper", None)
    # An unrelated module on the import path must not supply the validators.
    monkeypatch.setitem(sys.modules, "_interpret_protocol", None)
    return monkeypatch


@pytest.fixture
def wire(standalone):
    return _load(WIRE, "standalone_wire", standalone)


@pytest.fixture(params=["encode", "encode_parts"])
def encode(wire, request):
    if request.param == "encode":
        return wire.encode

    def parts(message):
        buffers, size = wire.encode_parts(message)
        data = b"".join(buffers)
        assert len(data) == size
        return data

    return parts


def _raw(wire, kind, meta):
    # Escaped surrogates and nonfinite numbers must reach the inbound checks.
    data = json.dumps(meta, ensure_ascii=True, separators=(",", ":")).encode()
    return wire.HEADER.pack(wire.MAGIC, wire.MAJOR, wire.MINOR, kind, 0,
                            wire.HEADER_BYTES + len(data), len(data), 0, 0) + data


def test_browser_control_consumer(standalone):
    mock = _load(ROOT / "web" / "test" / "mock_server.py", "wire_mock", standalone)
    standalone.setitem(sys.modules, "_wire", None)
    wire = mock.W = mock.load_wire(str(WIRE))
    assert not wire.__package__
    received = []
    broadcasts = []

    async def send(message):
        received.append(wire.decode(wire.encode(message)))

    async def broadcast():
        broadcasts.append(wire.decode(wire.encode(controls.message())))

    controls = mock.Controls(broadcast)

    async def exercise():
        for name, value in [("lr", 0.02), ("batch", 32), ("augment", False),
                            ("optimizer", "sgd")]:
            data = wire.encode_frame(wire.MsgType.CONTROL,
                                     {"request_id": 1, "op": "set", "name": name,
                                      "value": value})
            await controls.handle(wire.decode(data), send, allowed=True)
            assert received[-1] == wire.ControlReply(1, "set", "ok", name=name, value=value)
            entry = next(p for p in broadcasts[-1].params if p["name"] == name)
            assert entry["value"] == value

    asyncio.run(exercise())
    assert len(received) == len(broadcasts) == 4


def test_client_codec_consumer(standalone):
    client = _load(ROOT / "python" / "bench" / "net_client.py", "wire_client", standalone)
    standalone.setitem(sys.modules, "caliper_wire_copy", None)
    encode, decode = client.wire_codec("v3", str(WIRE))
    wire = sys.modules["caliper_wire_copy"]
    assert not wire.__package__
    assert decode(encode({"type": "pong", "nonce": 7})).meta == {"nonce": 7}
    controls = wire.Controls({"state": "none"},
                             [{"name": "prompt", "type": "text", "value": "é"}])
    assert decode(wire.encode(controls)).meta["params"] == controls.params
    reply = wire.ControlReply(1, "interpret", "ok", result={"state": "idle"})
    decoded = decode(wire.encode(reply))
    assert decoded.type == "control_reply" and decoded.meta["result"] == reply.result
    decoded = decode(wire.encode(wire.Control(2, "set", name="prompt", value="é")))
    assert decoded.type == "control" and decoded.meta["value"] == "é"


@pytest.mark.parametrize("value", [0.25, 32, True, False, "", "adam", "é" * 2048,
                                  "😀" * 1024])
def test_scalar_control_round_trips(wire, encode, value):
    control = wire.Control(1, "set", name="value", value=value)
    reply = wire.ControlReply(1, "set", "ok", name="value", value=value)
    for message in (control, reply):
        decoded = wire.decode(encode(message))
        assert decoded == message
        assert type(decoded.value) is type(value)


@pytest.mark.parametrize("op,fields", [("pause", {}), ("resume", {}),
                                      ("step", {"count": 2}), ("run", {"name": "save"})])
def test_other_control_round_trips(wire, encode, op, fields):
    message = wire.Control(1, op, **fields)
    assert wire.decode(encode(message)) == message
    reply = wire.ControlReply(1, op, "ok")
    assert wire.decode(encode(reply)) == reply


def test_text_parameter_round_trip(wire, encode):
    param = {"name": "prompt", "type": "text", "value": "é" * 2048,
             "pending": "😀" * 1024}
    message = wire.Controls({"state": "none"}, [param])
    assert wire.decode(encode(message)) == message
    assert wire.param_value(param, param["pending"]) == param["pending"]


@pytest.mark.parametrize("value", [None, {}, [], float("nan"), float("inf"),
                                  "é" * 2048 + "x", "a\0b", "\ud800"])
def test_invalid_scalar_controls(wire, value):
    for kind, meta in [
        (wire.MsgType.CONTROL, {"request_id": 1, "op": "set", "name": "p", "value": value}),
        (wire.MsgType.CONTROL_REPLY, {"request_id": 1, "op": "set", "status": "ok",
                                     "value": value}),
    ]:
        # An optional reply value can be null, but a set request needs a value.
        if value is None and kind == wire.MsgType.CONTROL_REPLY:
            assert wire.decode(_raw(wire, kind, meta)).value is None
        else:
            with pytest.raises((wire.MessageError, wire.WireError)):
                wire.decode(_raw(wire, kind, meta))


@pytest.mark.parametrize("field", ["value", "pending"])
@pytest.mark.parametrize("value", [True, 1, "x" * 4097, "a\0b", "\ud800"])
def test_invalid_text_parameters(wire, field, value):
    param = {"name": "prompt", "type": "text", "value": "ok", field: value}
    with pytest.raises(wire.MessageError):
        wire.decode(_raw(wire, wire.MsgType.CONTROLS,
                         {"run": {"state": "none"}, "params": [param], "actions": []}))


@pytest.mark.parametrize("payload", [
    {"version": 1, "method": "trace.status", "params": {}},
    {"version": 1, "method": "trace.run", "params": {"prompt": "é" * 2048}},
    {"version": 1, "method": "feature.examples", "params": {"node": 511, "limit": 16}},
])
def test_interpretation_round_trips(wire, encode, payload):
    control = wire.Control(1, "interpret", interpretation=payload)
    reply = wire.ControlReply(1, "interpret", "ok",
                              result={"text": "é", "finite": 1.5e16, "bool": True})
    assert wire.decode(encode(control)) == control
    assert wire.decode(encode(reply)) == reply


@pytest.mark.parametrize("payload", [
    None, {}, {"version": True, "method": "trace.status", "params": {}},
    {"version": 1, "method": "trace.status", "params": {"unknown": 1}},
    {"version": 1, "method": "trace.run", "params": {"prompt": "\ud800"}},
    {"version": 1, "method": "trace.run", "params": {"prompt": "x", "max_nodes": True}},
])
def test_invalid_interpretation_requests(wire, encode, payload):
    with pytest.raises(wire.MessageError):
        encode(wire.Control(1, "interpret", interpretation=payload))
    with pytest.raises(wire.MessageError):
        wire.decode(_raw(wire, wire.MsgType.CONTROL,
                         {"request_id": 1, "op": "interpret", "interpretation": payload}))


@pytest.mark.parametrize("result", [None, [], {"x": 2**53}, {"x": float("nan")},
                                   {"x": float("inf")}, {"x": "a\0b"}, {"x": "\ud800"}])
def test_invalid_interpretation_results(wire, encode, result):
    with pytest.raises(wire.MessageError):
        encode(wire.ControlReply(1, "interpret", "ok", result=result))
    with pytest.raises((wire.MessageError, wire.WireError)):
        wire.decode(_raw(wire, wire.MsgType.CONTROL_REPLY,
                         {"request_id": 1, "op": "interpret", "status": "ok", "result": result}))


@pytest.mark.parametrize("op,status", [("pause", "ok"), ("interpret", "queued"),
                                      ("interpret", "error")])
def test_interpretation_result_requires_success(wire, encode, op, status):
    error = {"code": "refused"} if status == "error" else None
    fields = {"request_id": 1, "op": op, "status": status, "error": error}
    valid = wire.ControlReply(**fields)
    assert wire.decode(encode(valid)) == valid
    with pytest.raises(wire.MessageError):
        encode(wire.ControlReply(**fields, result={}))
    with pytest.raises(wire.MessageError):
        wire.decode(_raw(wire, wire.MsgType.CONTROL_REPLY, {**fields, "result": {}}))


def test_interpretation_request_requires_its_operation(wire, encode):
    payload = {"version": 1, "method": "trace.status", "params": {}}
    with pytest.raises(wire.MessageError):
        encode(wire.Control(1, "pause", interpretation=payload))
    with pytest.raises(wire.MessageError):
        wire.decode(_raw(wire, wire.MsgType.CONTROL,
                         {"request_id": 1, "op": "pause", "interpretation": payload}))


@pytest.mark.parametrize("bound", ["string", "fields", "list", "depth", "values", "bytes"])
def test_interpretation_result_bounds(wire, encode, bound):
    if bound == "string":
        valid, invalid = {"x": "é" * 2048}, {"x": "é" * 2048 + "x"}
    elif bound == "fields":
        valid = {f"k{i}": None for i in range(64)}
        invalid = {**valid, "extra": None}
    elif bound == "list":
        valid, invalid = {"x": [None] * 512}, {"x": [None] * 513}
    elif bound == "depth":
        valid = {"x": None}
        for _ in range(7):
            valid = {"x": valid}
        invalid = {"x": valid}
    elif bound == "values":
        valid = {f"k{i}": [None] * (479 if i == 31 else 512) for i in range(32)}
        invalid = {**valid, "k31": [None] * 480}
    else:
        valid = {f"k{i}": "x" * 4096 for i in range(64)}
        size = len(json.dumps(valid, separators=(",", ":")).encode())
        valid["k0"] = valid["k0"][:4096 - (size - 262144)]
        invalid = {**valid, "k0": valid["k0"] + "x"}
    reply = wire.ControlReply(1, "interpret", "ok", result=valid)
    assert wire.decode(encode(reply)) == reply
    with pytest.raises(wire.MessageError):
        encode(wire.ControlReply(1, "interpret", "ok", result=invalid))
    with pytest.raises(wire.MessageError):
        wire.decode(_raw(wire, wire.MsgType.CONTROL_REPLY,
                         {"request_id": 1, "op": "interpret", "status": "ok", "result": invalid}))


@pytest.mark.parametrize("intervene", [
    {"set": {"point": "blocks.0.attn#3", "op": "ablate", "from_step": 1}},
    {"clear": "blocks.0.attn#3"}, {"clear_all": True},
])
def test_intervention_round_trips(wire, encode, intervene):
    message = wire.Control(1, "intervene", intervene=intervene)
    assert wire.decode(encode(message)) == message
    reply = wire.ControlReply(1, "intervene", "ok")
    assert wire.decode(encode(reply)) == reply


@pytest.mark.parametrize("intervene", [
    {}, {"clear_all": 1}, {"clear": "x", "clear_all": True},
    {"set": {"point": "x", "op": "ablate", "from_step": True}},
])
def test_invalid_interventions(wire, intervene):
    with pytest.raises(wire.MessageError):
        wire.decode(_raw(wire, wire.MsgType.CONTROL,
                         {"request_id": 1, "op": "intervene", "intervene": intervene}))
