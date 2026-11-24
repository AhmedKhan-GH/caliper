"""Outbound interpretation metadata uses the inbound wire limits."""
import json

import pytest

from caliper import _wire as wire


@pytest.mark.parametrize("encode", [wire.encode, wire.encode_parts])
def test_outbound_set_values_match_inbound_limits(encode):
    for value in ["", "é" * 2048, "line\n\tbreak", True, 1, 0.5]:
        for message in [wire.Control(1, "set", name="prompt", value=value),
                        wire.ControlReply(1, "set", "ok", name="prompt", value=value)]:
            encode(message)
            assert wire.decode(wire.encode(message)).value == value
    for value in ["é" * 2048 + "x", "a\0b", "\ud800", float("inf"), [], {}]:
        for message in [wire.Control(1, "set", name="prompt", value=value),
                        wire.ControlReply(1, "set", "ok", name="prompt", value=value)]:
            with pytest.raises(wire.MessageError):
                encode(message)
    with pytest.raises(wire.MessageError):
        encode(wire.Control(1, "set", name="prompt"))
    encode(wire.ControlReply(1, "set", "error", error={"code": "unsupported"}))


@pytest.mark.parametrize("encode", [wire.encode, wire.encode_parts])
def test_outbound_requests_require_the_interpretation_operation(encode):
    request = {"version": 1, "method": "trace.status", "params": {}}
    encode(wire.Control(1, "interpret", interpretation=request))
    for value in [None, {}, {**request, "version": True},
                  {**request, "params": {"unknown": 1}},
                  {"version": 1, "method": "trace.run", "params": {"prompt": "x" * 4097}}]:
        with pytest.raises(wire.MessageError):
            encode(wire.Control(1, "interpret", interpretation=value))
    with pytest.raises(wire.MessageError):
        encode(wire.Control(1, "pause", interpretation=request))


@pytest.mark.parametrize("encode", [wire.encode, wire.encode_parts])
def test_outbound_results_require_success(encode):
    for op, status in [("pause", "ok"), ("interpret", "queued"), ("interpret", "error")]:
        with pytest.raises(wire.MessageError):
            encode(wire.ControlReply(1, op, status, result={}))
    for value in [None, [], "text", {"x": 2**53}, {"x": float("nan")},
                  {"x": float("inf")}, {"x": "\0"}, {"x": "\ud800"}]:
        with pytest.raises(wire.MessageError):
            encode(wire.ControlReply(1, "interpret", "ok", result=value))
    for value in [2**53 - 1, -(2**53 - 1), 1.5e16, 1e20]:
        encode(wire.ControlReply(1, "interpret", "ok", result={"x": value}))
    encode(wire.ControlReply(1, "interpret", "queued"))
    encode(wire.ControlReply(1, "interpret", "error", error={"code": "refused"}))


@pytest.mark.parametrize("encode", [wire.encode, wire.encode_parts])
@pytest.mark.parametrize("bound", ["string", "fields", "list", "depth", "values", "bytes"])
def test_outbound_structured_limits(encode, bound):
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
        assert len(json.dumps(valid, separators=(",", ":")).encode()) == 262144
        invalid = {**valid, "k0": valid["k0"] + "x"}
    encode(wire.ControlReply(1, "interpret", "ok", result=valid))
    with pytest.raises(wire.MessageError):
        encode(wire.ControlReply(1, "interpret", "ok", result=invalid))
