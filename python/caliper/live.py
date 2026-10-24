"""caliper.live: the live-session transport ("attach to app").

A training script's first show opens a LOCAL LIVE SESSION: shared GPU snapshot
memory (POSIX shm, one segment per view) plus a session file
``<data_root>/live/<pid>.json`` (pid, script, start time, views, session token,
segment names, protocol version; data_root = ``$CALIPER_DATA_DIR`` or Caliper's
app-data folder). A viewer process (``caliper-viewer``, the Caliper app's
"Attach to process") lists the session files, attaches, and draws the newest
completed snapshot of each view at up to 60 fps; it can detach again. Training
never waits for it. Same user, local only, no network (caliper/live.h).

:class:`SessionTransport` implements the ``Transport`` protocol of
``caliper/_ring.py`` (attached / alloc / reading_slot / publish / remove). The
snapshot ring itself (slot choice, rate limit, torch.mps.Event polling) is
``caliper._ring.SnapshotRing``, and the public ``caliper.show`` routing lives in
``caliper/__init__.py``; this module only knows where slots live and where a
completed snapshot goes:

* ``alloc`` hands out torch tensors ALIASING a view's segment (payload, aux and
  stats regions of each slot): for MPS a kDLMetal DLPack capsule over the
  segment's MTLBuffer at the region's offset, so ``copy_`` / ``aminmax(out=)``
  run on torch's own stream straight into the shared pages (no host copy); for
  CPU a kDLCPU capsule over the mapping. The segment is created lazily on the
  first ``alloc`` of a view, sized from its ``ViewSpec``.
* ``publish`` writes the slot metadata (colormap, range, flags, series_total,
  points_dims, label_count, stats_count, stamp) and the publish seq.
* ``attached()`` is near-free (one atomic load and a clock read in C): True
  while a viewer's heartbeat is younger than ~1 s. With no viewer the ring takes
  no snapshot at all.
* If the viewer dies nothing blocks: ``attached()`` turns False and snapshots
  stop. If the script dies the viewer shows "session ended" and keeps what it
  received; ``close()`` (also at interpreter exit) removes the session file and
  unlinks the shm; a SIGKILLed script's leftovers are reclaimed by the next
  listing (``list_sessions``).

Not imported by ``import caliper``; ``from caliper import live`` loads it.

Naming: this glue is ``caliper.live`` (live.py), not ``caliper._live``, on
purpose. The C++ extension built from python/src/_live.mm is
``caliper/_live.cpython-*.so``, and Python's FileFinder tries extension suffixes
before ``.py``, so a ``_live.py`` next to it would be shadowed by the .so.
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path

from . import _ring

__all__ = ["SessionTransport", "list_sessions", "data_root", "viewer_executable"]


def _ext():
    from . import _live
    return _live


def data_root() -> str:
    """The default data root ($CALIPER_DATA_DIR, else Caliper's app-data folder)."""
    return _ext().data_root()


def list_sessions(data_root: str | None = None) -> list[dict]:
    """Live sessions under ``<data_root>/live``, newest first. Stale sessions
    (their process is gone) are reclaimed as a side effect: shm unlinked, file
    removed, not listed."""
    return _ext().list_sessions(data_root)


def viewer_executable(path: str | None = None) -> str:
    """``path``, else $CALIPER_VIEWER, else the viewer shipped in the package."""
    exe = path or os.environ.get("CALIPER_VIEWER") or \
        str(Path(__file__).parent / "bin" / "caliper-viewer")   # not resolve(): staged symlinks
    if not os.path.exists(exe):
        raise FileNotFoundError(f"caliper-viewer not found at {exe} (set CALIPER_VIEWER)")
    return exe


def _script_name() -> str:
    a = sys.argv[0] if sys.argv and sys.argv[0] else ""
    if not a or a == "-c":
        return "python -c" if a == "-c" else "python"
    return os.path.basename(a)


def _device_type(device) -> str:
    t = getattr(device, "type", None) or str(device).split(":")[0]
    if t not in ("mps", "cpu"):
        raise ValueError(f"live sessions take MPS or CPU slots, not {device!r}")
    return t


def _dtype_name(dtype) -> str:
    import torch
    if dtype in (torch.float32, "float32"):
        return "float32"
    if dtype in (torch.uint8, "uint8"):
        return "uint8"
    raise TypeError(f"live slots are float32 or uint8, not {dtype!r}")


def _snap_flags(flags: int) -> int:
    """caliper/_ring.py FLAG_* (== CALIPER_LIVE_APPLY_*) -> CALIPER_LIVE_SNAP_*."""
    ext = _ext()
    f = 0
    if flags & _ring.FLAG_AUTO_VMIN:
        f |= ext.SNAP_AUTO_VMIN
    if flags & _ring.FLAG_AUTO_VMAX:
        f |= ext.SNAP_AUTO_VMAX
    if flags & _ring.FLAG_RAW:
        f |= ext.SNAP_RAW
    return f


@dataclass
class _View:
    name: str
    key: tuple
    segment: object
    published: int = 0


_OPEN: "weakref.WeakSet[SessionTransport]" = weakref.WeakSet()


@atexit.register
def _close_all() -> None:
    for t in list(_OPEN):
        try:
            t.close()
        except Exception:
            pass


class _ClosedClient:
    """Stand-in for the native client after close(): the transport object may
    outlive the session (the package keeps it until interpreter exit), and a
    live reference to the nanobind instance there is reported as a leak at
    shutdown. Every accessor keeps the closed semantics."""
    closed = True
    session_file = ""
    session = ""

    def attached(self) -> bool:
        return False

    def viewer(self):
        return (0, None)

    def close(self) -> None:
        pass

    def remove_view(self, segment) -> None:
        pass

    def add_view(self, *args, **kwargs):
        raise RuntimeError("caliper live session is closed")


class SessionTransport:
    """One live session of this process (the ``Transport`` of caliper/_ring.py).

    Use :meth:`open`. Calls are serialised by the caller (the ring runs on the
    training thread); ``attached()`` may be called from any thread."""

    def __init__(self, client, script: str, viewer_path: str | None, data_dir: str | None):
        self._client = client
        self._token = client.session
        # The hot path: bind the C method itself (no Python frame in between).
        self.attached = client.attached
        self.script = script
        self._viewer_path = viewer_path
        self._data_dir = data_dir
        self._views: dict[str, _View] = {}
        self._viewer: subprocess.Popen | None = None
        self._viewer_log = None
        self._lock = threading.Lock()
        self.opened_ns = time.monotonic_ns()
        _OPEN.add(self)

    # -- lifecycle -------------------------------------------------------------
    @classmethod
    def open(cls, auto_launch: bool = False, viewer_path: str | None = None,
             script_name: str | None = None, session: str | None = None,
             data_dir: str | None = None, viewer_args=(), viewer_log: str | None = None,
             session_file: bool = True) -> "SessionTransport":
        """Open the session: control segment + session file; with
        ``auto_launch`` also spawn ``caliper-viewer --session <token>``.

        ``session``: the token (default ``clp-<pid>-<6 hex>``). ``data_dir``: the
        session-file root (default $CALIPER_DATA_DIR or Caliper's app data).
        ``viewer_log``: None discards the viewer's stderr, "-" inherits it, a
        path writes it there."""
        ptr = 0
        try:
            import torch
            if torch.backends.mps.is_available():
                # Create the segment MTLBuffers on torch's own device.
                ptr = torch.empty(1, device="mps").untyped_storage().data_ptr()
        except ImportError:
            pass
        script = script_name or _script_name()
        client = _ext().Client(session, ptr, script, data_dir, session_file)
        t = cls(client, script, viewer_path, data_dir)
        if auto_launch:
            t.launch_viewer(args=viewer_args, log=viewer_log)
        return t

    @property
    def session(self) -> str:
        """The session token (what ``caliper-viewer --session`` takes)."""
        return self._token

    @property
    def session_file(self) -> str:
        """Path of the session file ('' when opened without one or closed)."""
        return "" if self._client.closed else self._client.session_file

    @property
    def closed(self) -> bool:
        return self._client.closed

    def close(self) -> None:
        """Remove the session file and unlink the shared memory. Never blocks
        on the viewer (it keeps what it received and shows "session ended").
        Tensors handed out by alloc() keep their pages until they are freed."""
        with self._lock:
            self._views.clear()
            if not self._client.closed:
                self._client.close()
            # Drop the native instance (and the bound C method) now.
            self._client = _ClosedClient()
            self.attached = self._client.attached
            if self._viewer_log is not None:
                self._viewer_log.close()
                self._viewer_log = None
        _OPEN.discard(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- Transport (caliper/_ring.py) -------------------------------------------
    # attached() is bound per instance in __init__ (Client.attached).

    def alloc(self, name: str, spec, slot: int, role: str, shape: tuple, dtype):
        """A torch tensor aliasing one region of slot ``slot`` of view ``name``
        (role "payload" | "aux" | "stats"), on ``spec.device``. Creates the
        view's segment on first use (and recreates it if the spec changed)."""
        import torch
        dev = _device_type(spec.device)
        key = (int(spec.kind), tuple(int(d) for d in spec.shape), _dtype_name(spec.dtype), dev,
               tuple(spec.aux_shape) if spec.aux_shape else None, int(spec.slots))
        with self._lock:
            v = self._views.get(name)
            if v is not None and v.key != key:
                self._remove_locked(name)
                v = None
            if v is None:
                aux = int(spec.aux_shape[0]) if spec.aux_shape else 0
                seg = self._client.add_view(name, key[0], key[2], list(key[1]), key[5], aux)
                v = self._views[name] = _View(name, key, seg)
            cap = v.segment.region(role, int(slot), dev, [int(d) for d in shape], _dtype_name(dtype))
        return torch.from_dlpack(cap)

    def reading_slot(self, name: str) -> int:
        """The slot the viewer may still be reading, or -1. A points view can
        hold two claims for about a frame (the current snapshot and the one it
        superseded, until its last frame retired); while the client has not
        published past the current one, the superseded one is returned (the
        ring avoids the published slot anyway). :meth:`claimed_slots` gives
        both."""
        v = self._views.get(name)
        if v is None:
            return -1
        r, prev = v.segment.claims()
        if prev < 0 or prev == r:
            return r
        if r < 0:
            return prev
        p, _ = v.segment.published()
        return prev if r == p else r

    def claimed_slots(self, name: str) -> tuple:
        """Every slot of ``name`` the viewer holds (0, 1 or 2 of them)."""
        v = self._views.get(name)
        if v is None:
            return ()
        r, prev = v.segment.claims()
        return tuple(s for s in dict.fromkeys((r, prev)) if s >= 0)

    def publish(self, name: str, slot, meta) -> None:
        """Hand over a completed snapshot (its GPU writes completed). Never
        blocks. ``slot`` is a caliper._ring.Slot (or a slot index)."""
        v = self._views.get(name)
        if v is None:
            return   # removed meanwhile
        idx = slot.index if hasattr(slot, "index") else int(slot)
        v.segment.publish_slot(idx, int(meta.stamp_ns), int(meta.colormap), _snap_flags(int(meta.flags)),
                               float(meta.vmin), float(meta.vmax), int(meta.series_total),
                               int(meta.points_dims), int(meta.label_count), int(meta.stats_count))
        v.published += 1

    def remove(self, name: str) -> None:
        """Drop the view: its table row and shm name go now; the pages live on
        while the viewer's mapping or a tensor from alloc() holds them."""
        with self._lock:
            self._remove_locked(name)

    def _remove_locked(self, name: str) -> None:
        v = self._views.pop(name, None)
        if v is not None and not self._client.closed:
            self._client.remove_view(v.segment)

    def count_drop(self, name: str) -> None:
        """Count a snapshot the ring dropped for want of a free slot (shown by
        the viewer as dropped@client)."""
        v = self._views.get(name)
        if v is not None:
            v.segment.count_drop()

    # -- the viewer process ----------------------------------------------------------
    def launch_viewer(self, exit_after: float | None = None, args=(), log: str | None = None
                      ) -> subprocess.Popen:
        """Spawn ``caliper-viewer --session <token>`` in its own session (so a
        Ctrl-C in the script's terminal does not close it). It attaches at once
        and outlives the script."""
        exe = viewer_executable(self._viewer_path)
        cmd = [exe, "--session", self.session]
        if self._data_dir:
            cmd += ["--data-dir", self._data_dir]
        if exit_after:
            cmd += ["--exit-after", str(exit_after)]
        cmd += [str(a) for a in args]
        stderr = subprocess.DEVNULL
        if log == "-":
            stderr = None
        elif log:
            if self._viewer_log is not None:
                self._viewer_log.close()
            self._viewer_log = open(log, "w")
            stderr = self._viewer_log
        self._viewer = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=stderr, start_new_session=True)
        return self._viewer

    @property
    def viewer(self) -> subprocess.Popen | None:
        """The viewer this transport spawned, if any."""
        return self._viewer

    def viewer_alive(self) -> bool:
        """The spawned viewer is running; without one, whether any viewer is
        attached (heartbeat)."""
        if self._viewer is not None:
            return self._viewer.poll() is None
        return bool(self.attached())

    def wait_viewer(self, timeout: float | None = None) -> bool:
        """Wait until the viewer is gone (the spawned process exited, or with
        none spawned, the heartbeat stopped). True if it is gone."""
        if self._viewer is not None:
            try:
                self._viewer.wait(timeout)
                return True
            except subprocess.TimeoutExpired:
                return False
        end = None if timeout is None else time.monotonic() + timeout
        while self.attached():
            if end is not None and time.monotonic() >= end:
                return False
            time.sleep(0.05)
        return True

    def wait_attached(self, timeout: float = 10.0) -> bool:
        """Wait until a viewer is attached (heartbeat). True if it is."""
        end = time.monotonic() + timeout
        while not self.attached():
            if time.monotonic() >= end or (self._viewer is not None and self._viewer.poll() is not None):
                return False
            time.sleep(0.01)
        return True

    # -- telemetry -------------------------------------------------------------------
    def stats(self) -> dict:
        closed = self._client.closed
        pid, age = (0, None) if closed else self._client.viewer()
        out = {"session": self.session, "session_file": self.session_file, "script": self.script,
               "closed": closed, "attached": bool(self.attached()),
               "viewer_pid": pid, "viewer_heartbeat_age_s": age,
               "spawned_viewer_pid": self._viewer.pid if self._viewer is not None else None,
               "spawned_viewer_running": self._viewer is not None and self._viewer.poll() is None,
               "views": {}}
        for name, v in list(self._views.items()):
            info = v.segment.info()
            out["views"][name] = {
                "kind": info["kind"], "published": v.published, "seq": info["seq"],
                "published_slot": info["published_slot"],
                "viewer_reading_slot": info["viewer_reading_slot"],
                "viewer_prev_slot": info["viewer_prev_slot"],
                "dropped_client": info["dropped_client"], "viewer_seq": info["viewer_seq"],
                "viewer_skipped": info["skipped_viewer"], "segment": info["segment"],
                "segment_bytes": info["segment_bytes"]}
        return out

    def segment(self, name: str):
        """The _live.Segment of a view (diagnostics, tests), or None."""
        v = self._views.get(name)
        return v.segment if v is not None else None
