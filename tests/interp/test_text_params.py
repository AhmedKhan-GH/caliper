"""Exact UTF-8 byte bounds for the free-text parameter transport."""
import pytest

from caliper import _controls, _wire as wire
from caliper import _interpret_protocol as protocol
from .test_service import Transport


@pytest.mark.parametrize("value", ["", "x" * 1025, "x" * 4096, "é" * 2048, "😀" * 1024])
def test_text_parameter_preserves_bytes_at_each_boundary(value):
    controls, transport = _controls.Controls(), Transport()
    parameter = controls.param("prompt", value)
    route = controls.attach(transport)
    try:
        metadata = wire.Controls({"state": "running"}, [parameter.entry()])
        assert wire.decode(wire.encode(metadata)).params[0]["value"].encode() == value.encode()
        controls.step(1)
        parameter.set("initial")
        request = wire.decode(wire.encode(wire.Control(1, "set", name="prompt", value=value)))
        answer = route.on_control(None, request.to_meta())
        assert answer["status"] == "ok" and answer["value"] == value
        assert parameter.value == "initial" and parameter.pending == value
        metadata = wire.Controls({"state": "running"}, [parameter.entry()])
        assert wire.decode(wire.encode(metadata)).params[0]["pending"] == value
        reply = wire.ControlReply.from_frame(wire.Frame(wire.MsgType.CONTROL_REPLY, answer))
        assert wire.decode(wire.encode(reply)).value == value
        controls.step(2)
        assert parameter.value.encode() == value.encode()
        assert parameter.set(value) == value
        if value:
            assert protocol.request({"version": 1, "method": "trace.run", "params": {"prompt": value}})
    finally:
        route.detach()
        parameter.remove()


@pytest.mark.parametrize("value", ["x" * 4097, "é" * 2048 + "x", "😀" * 1024 + "x", "a\0b", "\ud800"])
def test_text_parameter_refuses_invalid_bytes_without_mutation(value):
    controls = _controls.Controls()
    parameter = controls.param("prompt", "original")
    with pytest.raises(ValueError):
        controls.param("invalid", value)
    with pytest.raises(ValueError):
        parameter.set(value)
    assert parameter.value == "original" and parameter.pending is None
    for metadata in (
        wire.Controls({"state": "none"}, [{"name": "prompt", "type": "text", "value": value}]),
        wire.Controls({"state": "none"}, [{"name": "prompt", "type": "text", "value": "ok", "pending": value}]),
        wire.Control(1, "set", name="prompt", value=value),
        wire.ControlReply(1, "set", "ok", name="prompt", value=value),
    ):
        with pytest.raises(wire.MessageError):
            type(metadata).from_frame(wire.Frame(metadata.TYPE, metadata.to_meta()))
    assert not protocol.request({"version": 1, "method": "trace.run", "params": {"prompt": value}})


def test_older_string_bounds_remain_separate():
    controls = _controls.Controls()
    assert controls.param("help", "", help="x" * 1024).help == "x" * 1024
    with pytest.raises(ValueError):
        controls.param("invalid", "", help="x" * 1025)
    with pytest.raises(ValueError):
        controls.param("invalid", "x" * 64, choices=["x" * 64])
    assert wire.TEXT_MAX == _controls.TEXT_MAX == 1024
    assert wire.PARAM_TEXT_MAX == _controls.PARAM_TEXT_MAX == protocol.MAX_STRING == 4096
