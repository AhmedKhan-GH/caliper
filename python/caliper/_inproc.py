"""The in-process window transport (``caliper.window()``, ``caliper.offscreen()``).

The second of caliper's two transports (the first is the attach-to-app session,
caliper.live). Every slot the shared :class:`~caliper._ring.SnapshotRing` writes
is a caliper-made shared-storage MTLBuffer (``_native.SharedBuffer``) handed to
torch through DLPack, so torch writes it on its own stream and the core reads
the same pages: zero-copy on the GPU for heatmap/image/points, on the CPU for
stats, series and histogram values. CPU tensors are copied once into the slot
and then read by the core in place too (path ``compute-imported``).

A completed snapshot goes to the render thread (``Core.publish_live``), which
applies it through ``caliper_core_live_apply`` (heatmap/image through the embed
v1.2 view until the typed view layer delegates them: one switch in _native.mm)
and renders at 60 Hz. Slots the core may still read are claimed and reported
by ``claimed_slots``; the ring never rewrites them.

Limits of the in-process window, by design of macOS and of one process:

* AppKit delivers input (mouse, keys, resize, the close button) only while the
  MAIN thread runs the event loop. The render thread keeps the canvas
  repainting whatever the training loop does, but input waits during long
  steps unless ``caliper.run(train_fn)`` runs the training on a worker thread
  and gives the main thread to Caliper. Without run(), show() pumps events at
  most at 60 Hz when it is called on the main thread.
* A crash of the training process takes the window down with it (and a hung
  window hangs the script). The attach-to-app session has neither problem.
* Opening a window makes the script an AppKit app, which macOS may App Nap
  when its windows are occluded and the main thread idles in the event loop
  (caliper.run): every thread is throttled then, training included (measured
  before the fix: render 9 fps, training at 56-60 % of the control). The
  native window holds an NSProcessInfo user-initiated, latency-critical
  activity while it is open, which prevents that.
* One core per process: one window (or one offscreen canvas) at a time.

``Window(sync=True)`` (``caliper.offscreen``) is the synchronous test mode: no
render thread; a show waits for its snapshot's event, applies it and renders on
the calling thread, so view readbacks are deterministic.
"""
from __future__ import annotations

import ctypes
import math
import os
import shutil
import tempfile
import threading
import time
from typing import Callable

from ._ring import (LOCK, ROLE_AUX, ROLE_PAYLOAD, ROLE_STATS, SnapshotRing, Slot, SnapshotMeta,
                    ViewSpec)

__all__ = ["Window", "WindowTransport"]

_FRAME_INTERVAL = 1.0 / 60.0


def _native():
    from . import _native as n
    return n


def _dtype_name(dtype) -> str:
    import torch
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.uint8:
        return "uint8"
    raise TypeError(f"window transport slots are float32 or uint8, not {dtype}")


class WindowTransport:
    """Transport protocol over the native core: slots are SharedBuffers,
    publish hands a completed slot to the render thread (or applies it at once
    in the synchronous mode)."""

    def __init__(self, core):
        self.core = core
        self._bufs: dict[tuple, object] = {}       # (name, slot, role) -> SharedBuffer
        self._closed = False

    def attached(self) -> bool:
        return not self._closed

    def alloc(self, name: str, spec: ViewSpec, slot: int, role: str, shape: tuple, dtype):
        import torch
        dt = _dtype_name(dtype)
        nbytes = max(4, math.prod(shape) * (4 if dt == "float32" else 1))
        buf = _native().SharedBuffer(nbytes)
        dev = "mps" if spec.device.type == "mps" else "cpu"
        t = torch.from_dlpack(buf.dlpack(dev, dt, list(shape)))
        self._bufs[(name, slot, role)] = buf
        return t

    def reading_slot(self, name: str) -> int:
        return -1 if self._closed else self.core.reading_slot(name)

    def claimed_slots(self, name: str) -> list:
        """Every slot of ``name`` the core may still read (see _native.mm)."""
        return [] if self._closed else self.core.reading_slots(name)

    reading_slots = claimed_slots

    def publish(self, name: str, slot: Slot, meta: SnapshotMeta) -> None:
        if self._closed:
            return
        b = self._bufs
        i = slot.index
        p = slot.payload
        aux = b.get((name, i, ROLE_AUX)) if slot.aux is not None else None
        self.core.publish_live(
            name, meta.kind, i, b[(name, i, ROLE_PAYLOAD)], list(p.shape), _dtype_name(p.dtype),
            aux, slot.aux.shape[0] if aux is not None else 0, b.get((name, i, ROLE_STATS)),
            meta.stats_count, meta.colormap, meta.vmin, meta.vmax, meta.flags, meta.series_total,
            meta.points_dims, meta.label_count, meta.seq, meta.stamp_ns)

    def remove(self, name: str) -> None:
        for k in [k for k in self._bufs if k[0] == name]:
            del self._bufs[k]
        if not self._closed:
            self.core.forget(name)

    def close(self) -> None:
        self._closed = True
        self._bufs.clear()


class Window:
    """A Caliper canvas in this process fed by a SnapshotRing.

    * ``Window()``: a GLFW/AppKit window repainted by the render thread.
    * ``Window(offscreen=True)``: the render thread on an offscreen canvas
      (benchmarks, CI, :meth:`render`).
    * ``Window(offscreen=True, sync=True)``: no render thread (caliper.offscreen).

    Create on the main thread."""

    def __init__(self, width: int = 1280, height: int = 800, title: str = "caliper", *,
                 offscreen: bool = False, sync: bool = False, fps: float = 60.0,
                 rate: float | None = None, activate: bool = True, heartbeat: bool = False,
                 data_dir: str | None = None, on_pending: Callable | None = None):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("caliper.window()/offscreen() must be called from the main thread "
                               "(macOS AppKit/GLFW windows are main-thread only)")
        if sync and not offscreen:
            raise ValueError("sync=True needs offscreen=True")
        n = _native()
        self._owned_dir = None
        if data_dir is None:
            data_dir = os.environ.get("CALIPER_DATA_DIR")
        if data_dir is None:
            # A private per-process store root: the viewer needs no persistence,
            # and a shared root would contend for DuckDB's file lock.
            data_dir = self._owned_dir = tempfile.mkdtemp(prefix="caliper-py-")
        self.core = n.Core(data_dir)
        try:
            if offscreen:
                self.core.open_offscreen(width, height)
            else:
                self.core.open_window(width, height, title, activate)
                # AppKit puts a new window on screen only once the main thread
                # has run the event loop.
                for _ in range(5):
                    self.core.wait_events(0.01)
            if not sync:
                self.core.start_render_thread(fps, heartbeat)
        except BaseException:
            self.core.close()
            self._cleanup_dir()
            raise
        self.offscreen = offscreen
        self.sync = sync
        self.transport = WindowTransport(self.core)
        if rate is None:
            rate = 0.0 if sync else 60.0       # the test mode shows every call
        self.ring = SnapshotRing(self.transport, rate, on_pending=None if sync else on_pending)
        self._in_run = False
        self._next_pump = 0.0
        self._last_frame = 0.0
        self._closed = False

    # -- the synchronous mode -----------------------------------------------------
    def complete(self, name: str) -> None:
        """Sync mode: wait for view ``name``'s snapshot, apply it (on this
        thread) and render a frame if one is due (60 Hz)."""
        self.ring.flush(name)
        now = time.perf_counter()
        if now - self._last_frame >= _FRAME_INTERVAL:
            self._last_frame = now
            self.core.frame()

    # -- main-thread event pumping -----------------------------------------------------
    def pump_due(self) -> bool:
        """Cheap check for show(): a real window, not inside run(), >= 1/60 s
        since the last pump (the caller checks the main thread)."""
        return (not self.offscreen and not self._in_run and not self._closed
                and time.perf_counter() >= self._next_pump)

    def pump(self) -> bool:
        """Main thread: process pending window events. Returns False once the
        window was closed by the user (the window is then closed)."""
        if self._closed:
            return False
        self._next_pump = time.perf_counter() + _FRAME_INTERVAL
        if self.offscreen:
            return True
        self.core.poll()
        if self.core.should_close():
            self.close()
            return False
        return True

    def loop(self, seconds: float | None = None) -> None:
        """Main thread: run the event loop until the window is closed or
        ``seconds`` passed."""
        if self._closed:
            return
        if self.offscreen:
            if seconds:
                time.sleep(seconds)
            return
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("caliper.spin() with a window must run on the main thread")
        closed = self.core.run_event_loop(0.1, -1.0 if seconds is None else float(seconds))
        if closed:
            self.close()

    def run(self, fn: Callable, *args, **kwargs):
        """Run ``fn(*args, **kwargs)`` on a worker thread while the main thread
        sits in the AppKit event loop, so the window stays fully interactive
        whatever fn does. Returns fn's result or re-raises its exception. If the
        user closes the window, fn keeps running (its show() calls become
        no-ops) and the window closes when fn returns. Ctrl-C is forwarded to
        the worker as KeyboardInterrupt."""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("caliper.run must be called from the main thread")
        result: dict = {}

        def target():
            try:
                result["value"] = fn(*args, **kwargs)
            except BaseException as e:      # re-raised on the main thread
                result["error"] = e
            finally:
                if not self._closed and not self.offscreen:
                    self.core.stop_event_loop()

        worker = threading.Thread(target=target, name="caliper.train", daemon=True)
        self._in_run = True
        worker.start()
        user_closed = False
        try:
            while worker.is_alive():
                if self.offscreen or self._closed or user_closed:
                    worker.join(0.1)
                    continue
                # Bounded: run_event_loop clears its stop flag on entry, so a
                # worker that finished just before re-entry must not be missed.
                if self.core.run_event_loop(0.1, 0.25):
                    user_closed = True
                    with LOCK:
                        self.transport.close()   # show() becomes a no-op at once
        except KeyboardInterrupt:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(worker.ident),
                                                       ctypes.py_object(KeyboardInterrupt))
            worker.join()
            raise
        finally:
            self._in_run = False
        worker.join()
        if user_closed:
            self.close()
        if "error" in result:
            raise result["error"]
        return result.get("value")

    # -- telemetry / readback -------------------------------------------------------------
    def stats(self, reset: bool = False) -> dict:
        with LOCK:
            d = dict(self.core.render_stats(reset)) if not self._closed else {}
            d["ring"] = dict(self.ring.counters)
            if reset:
                self.ring.counters.clear()
        return d

    def render(self):
        """(H, W, 4) uint8 of a freshly composited frame (offscreen canvases)."""
        if not self.offscreen:
            raise RuntimeError("render() needs an offscreen canvas (caliper.offscreen() or "
                               "caliper.window(offscreen=True))")
        with LOCK:
            if self.sync:
                self.ring.flush()
            self.core.frame()
            return self.core.read_pixels()

    # -- teardown -------------------------------------------------------------------------
    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        with LOCK:
            if self._closed:
                return
            self._closed = True
            try:
                self.ring.flush(timeout=1.0)
            except Exception:
                pass
            self.ring.close()
            self.transport.close()
            self.core.close()
        self._cleanup_dir()

    def _cleanup_dir(self):
        if self._owned_dir:
            shutil.rmtree(self._owned_dir, ignore_errors=True)
            self._owned_dir = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
