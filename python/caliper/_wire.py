"""Wire protocol v3: frames and messages, encoded and decoded in pure Python.

The protocol is defined by ``include/caliper/live_net.h`` and
``docs/reference/wire-protocol.md``; this module follows them and mirrors
their constants (``python/tests/test_wire.py`` checks that the three agree).
It has no sockets and imports no torch: :func:`encode` turns a message object
into the bytes of one WebSocket message, :func:`decode` turns those bytes back
into a message object, and :func:`resolve_data_view` and :func:`payload_layout`
apply the data view rules a producer needs. :func:`param_value` and
:func:`selection_items` apply the rules of 3.1's controls and linked selection.
numpy is used only for payload sections (a decoded section is a zero-copy view
of the received buffer).

A frame is a 24-byte little-endian header, ``section_count`` uint32 section
lengths, a UTF-8 JSON metadata object, then the payload sections, each starting
at an offset that is a multiple of 8 from the start of the frame.
"""
from __future__ import annotations

import numpy as np

if __package__:
    from . import (
        _interpret_protocol,
        _intervention_protocol,
        _wire_controls,
        _wire_dataview,
        _wire_frame,
        _wire_messages,
        _wire_programs,
        _wire_recordings,
        _wire_scenes,
    )
else:
    # Browser fixtures and codec clients load this file without the package.
    import importlib.util as _import_util
    from pathlib import Path as _Path
    import sys as _sys

    def _load_sibling(name):
        mod_name = f"{__name__}.{name}"
        spec = _import_util.spec_from_file_location(
            mod_name, _Path(__file__).with_name(name + ".py"))
        module = _import_util.module_from_spec(spec)
        _sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module

    _interpret_protocol = _load_sibling("_interpret_protocol")
    _intervention_protocol = _load_sibling("_intervention_protocol")
    _wire_frame = _load_sibling("_wire_frame")
    _wire_programs = _load_sibling("_wire_programs")
    _wire_recordings = _load_sibling("_wire_recordings")
    _wire_scenes = _load_sibling("_wire_scenes")
    _wire_controls = _load_sibling("_wire_controls")
    _wire_messages = _load_sibling("_wire_messages")
    _wire_dataview = _load_sibling("_wire_dataview")


def _export(module):
    g = globals()
    for name, value in vars(module).items():
        if name.startswith("__") or name in ("np", "dataclass", "field", "Any", "ClassVar",
                                            "enum", "json", "math", "re", "struct",
                                            "_f", "_p", "_r", "_s", "_sys", "_prefix"):
            continue
        g[name] = value


for _mod in (
    _wire_frame,
    _wire_programs,
    _wire_recordings,
    _wire_scenes,
    _wire_controls,
    _wire_messages,
    _wire_dataview,
):
    _export(_mod)

__all__ = [
    "MAJOR", "MINOR", "MAGIC", "MsgType", "Kind", "DType", "Encoding", "WireError",
    "MessageError", "Section", "Frame", "encode_frame", "decode_frame", "canonical_json",
    "encode", "decode", "Hello", "Welcome", "Bye", "Error", "Ping", "Pong", "Sessions",
    "Attach", "Views", "Subscribe", "Unsubscribe", "ViewRequest", "ViewReply", "Snapshot",
    "Series", "Event", "Unknown", "resolve_data_view", "payload_layout",
    "Controls", "Control", "ControlReply", "Selection", "param_value", "selection_items",
    "SceneUpdate", "SceneTransforms", "SceneClear", "SceneClasses", "entity_path_ok",
    "scene_view_info", "resolve_scene_data_view",
    "Text", "Recordings", "RecordingOpen", "Cursor", "TimeState", "stamp_text_ok",
    "SplitUpdates",
    "Program", "ProgramState", "Resource", "SplitResources", "resource_parts",
]

MESSAGES = {
    cls.TYPE: cls
    for cls in (
        Hello, Welcome, Bye, Error, Ping, Pong, Sessions, Attach,
        Views, Subscribe, Unsubscribe, ViewRequest, ViewReply,
        Snapshot, Series, Event,
        Controls, Control, ControlReply, Selection,
        SceneUpdate, SceneTransforms, SceneClear, SceneClasses,
        Text, Recordings, RecordingOpen, Cursor, TimeState,
        Program, ProgramState, Resource,
    )
}


def encode(msg, *, minor: int = MINOR, as_bytearray: bool = False):
    """The frame bytes of a message object (a bytearray with ``as_bytearray``)."""
    return encode_frame(msg.TYPE, msg.to_meta(), msg.to_sections(), minor=minor,
                        as_bytearray=as_bytearray)


def encode_parts(msg, *, minor: int = MINOR):
    """A message object as :func:`frame_parts`: ``(buffers, frame_len)``."""
    return frame_parts(msg.TYPE, msg.to_meta(), msg.to_sections(), minor=minor)


def decode(buf):
    """A message object (or :class:`Unknown`) from one frame's bytes. Raises
    :class:`WireError` for a frame that is not well formed and
    :class:`MessageError` for an invalid message."""
    f = decode_frame(buf)
    cls = MESSAGES.get(f.type)
    if cls is None:
        return Unknown(f.type, f.meta, f.sections)
    return cls.from_frame(f)
