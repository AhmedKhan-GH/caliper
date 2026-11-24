"""The managed run: a script that the model runner of the Caliper app started.

The runner starts the script as a child process and sets these variables, and nothing else that
is Caliper-specific:

| Variable | Meaning |
|---|---|
| ``CALIPER_RUN_ID`` | The run's id, unique in the workspace |
| ``CALIPER_RUN_DIR`` | An existing folder for the run's files; a managed run is one where it is set |
| ``CALIPER_RUN_LINK_FILE`` | The file that receives the stream link of ``caliper.serve()`` |
| ``CALIPER_RUN_PARAMS`` | Optional JSON object of parameter name to value |
| ``CALIPER_RUN_RESUME`` | Optional checkpoint file to load before the first step (a fork) |
| ``CALIPER_RUN_PARENT`` | Optional id of the parent run (a fork) |
| ``CALIPER_RUN_INTERVENTIONS`` | Optional JSON array of the interventions that a fork applies before its first step |
| ``CALIPER_RUN_APP_PID`` | The process id of the app whose model runner started the run |

In a managed run, ``caliper.watch(model, optimizer)`` and ``caliper.watch(model)`` also do these
things, in this order:

1. ``prepare``: start the server if it is not started, and load ``CALIPER_RUN_RESUME`` into the
   model and the optimizer. The step count then continues from the checkpoint: the first
   checkpoint of the fork has the step of the file, so the fork redoes that step with its own
   parameter values, as a value that a viewer sets at a checkpoint applies to that same step.
2. The parameter values of the checkpoint and then of ``CALIPER_RUN_PARAMS`` apply to every
   parameter that is registered (``caliper._controls`` asks ``value`` at each registration, also
   for parameters that the script registers later). A value is checked and clamped like a value
   from a viewer. The parameter ``lr`` of ``watch(model, optimizer)`` applies its value at the
   first step.
3. ``attach``: register the actions ``caliper.checkpoint`` and ``caliper.stop``. Without an
   optimizer, each forward pass of the model becomes the checkpoint of the run (it calls
   ``caliper.step()``), so an inference loop with ``watch(model)`` alone takes actions too; such a
   script does not also call ``caliper.step()``. With an optimizer, the run keeps checkpoints
   for a fork from a past step (``KeptCheckpoints``), and a fork with
   ``CALIPER_RUN_INTERVENTIONS`` applies them at the first forward pass of the model, thus before
   its first step. Last, write the link file, so that the app that attaches finds the parameters
   and the actions there.

``caliper.checkpoint`` writes ``<run dir>/checkpoint-<step>.pt`` at the checkpoint where it runs,
before the weights change, and returns the path. The file is a ``torch.save`` dict with the keys
``model``, ``optimizer``, ``format``, ``version``, ``run``, ``parent``, ``step`` and ``params``
(``caliper._adapter.TorchAdapter.state``). ``caliper.stop`` makes the hook of that checkpoint
send what is pending, close the stream server and raise ``SystemExit(0)`` before the weights
change. The close sends the final reply of each action and the last run state before BYE and
the close code 1001, and waits a bounded time for each viewer. The stop also ends a pause.

A managed run does not outlive the app that started it. On POSIX, ``import caliper`` in a process
with ``CALIPER_RUN_APP_PID`` imports this module and starts a daemon thread (``watch_parent``), and
so does ``prepare``. Without the variable, ``import caliper`` imports nothing of this module. The
thread looks every ``PARENT_POLL_S`` seconds whether the app still runs: while the app is this
process's parent, whether the parent changed, else whether a process of that id exists. When the
app ended, also in a crash where the app cannot end its children and also before the import, the
thread ends the run at once with exit code ``PARENT_GONE_EXIT``. Nothing reads the run after that,
so the thread does not wait for a checkpoint. Without ``CALIPER_RUN_APP_PID`` the thread watches
the parent's process id, from ``prepare`` on.

A run with an optimizer keeps checkpoints in ``<run dir>/kept/checkpoint-<step>.pt``, in the format
of ``caliper.checkpoint``, at the step boundary before the weights change, so that ``run.fork``
with a step starts from the newest kept checkpoint at or before it (``KeptCheckpoints``). The
first step is always kept. Later steps are kept at least a stride apart, and only while the time
of the kept checkpoints stays within ``KEEP_SHARE`` of the run's wall clock. When more than
``KEEP_MAX`` are kept, every second one goes, the oldest and the newest stay, and the stride
doubles, thus the kept checkpoints span the whole run evenly, as the kept versions of a resource
do. They hold at most ``KEEP_BYTES`` together: a run whose checkpoint is too large for two keeps
only its first.
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import time
import warnings

from ._adapter import TORCH

__all__ = ["ManagedRun", "current", "value", "NONE"]

ENV_ID = "CALIPER_RUN_ID"
ENV_DIR = "CALIPER_RUN_DIR"
ENV_LINK = "CALIPER_RUN_LINK_FILE"
ENV_PARAMS = "CALIPER_RUN_PARAMS"
ENV_RESUME = "CALIPER_RUN_RESUME"
ENV_PARENT = "CALIPER_RUN_PARENT"
ENV_APP_PID = "CALIPER_RUN_APP_PID"
ENV_INTERVENTIONS = "CALIPER_RUN_INTERVENTIONS"
_VARS = (ENV_ID, ENV_DIR, ENV_LINK, ENV_PARAMS, ENV_RESUME, ENV_PARENT, ENV_INTERVENTIONS)

CHECKPOINT = "caliper.checkpoint"
"""The action that saves the run's state and returns the path of the file."""
STOP = "caliper.stop"
"""The action that ends the script at its checkpoint."""
FORMAT = "caliper.checkpoint"
"""The ``format`` field of a checkpoint file."""
VERSION = 1
"""The ``version`` field of a checkpoint file."""
FLUSH_S = 2.0
"""Seconds that a stop waits at most for pending snapshots before the script ends."""
# How often the run checks that the app that started it still runs, and the exit code when it
# does not.
PARENT_POLL_S = 0.5
PARENT_GONE_EXIT = 1
NONE = object()
"""What ``value`` returns for a parameter that the runner gives no value."""
KEEP_DIR = "kept"
"""The folder of the kept checkpoints in the run folder."""
KEEP_MAX = 32
"""The most kept checkpoints; one more thins them."""
KEEP_SHARE = 0.01
"""The most share of the run's wall clock that the kept checkpoints take."""
KEEP_BYTES = 1 << 30
"""The most bytes of the kept checkpoints together."""
INTERVENTIONS_MAX = 16
"""The most interventions of ``CALIPER_RUN_INTERVENTIONS``."""


def _params(text):
    if not text:
        return {}
    try:
        got = json.loads(text)
    except ValueError as e:
        warnings.warn(f"caliper: {ENV_PARAMS} is not JSON ({e}); no parameter value applies",
                      RuntimeWarning, stacklevel=3)
        return {}
    if not isinstance(got, dict):
        warnings.warn(f"caliper: {ENV_PARAMS} is not a JSON object; no parameter value applies",
                      RuntimeWarning, stacklevel=3)
        return {}
    return got


def _interventions(text):
    """The interventions of ``CALIPER_RUN_INTERVENTIONS``: a list of objects, each with the keys
    of ``intervene.set``. Text that is not such a list raises, because a fork must never run
    without the interventions that it was started with."""
    if not text:
        return []
    got = json.loads(text)
    if (not isinstance(got, list) or len(got) > INTERVENTIONS_MAX
            or not all(isinstance(x, dict) for x in got)):
        raise ValueError(f"{ENV_INTERVENTIONS} is not a JSON array of at most "
                         f"{INTERVENTIONS_MAX} interventions")
    return got


def _temp_name(folder: str, base: str) -> str:
    """A new name for the temporary file of ``_write`` beside ``base``."""
    return os.path.join(folder, f".{base}.{os.getpid()}.{secrets.token_hex(8)}.tmp")


_TEMP_FLAGS = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
               | getattr(os, "O_BINARY", 0))


def _write(path: str, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically: a new file of mode 0600 beside it, then a rename.

    The temporary file is created with ``O_EXCL | O_NOFOLLOW``, so that no file or link that is
    there already, such as one that another user put into the run folder, is written through.

    Raises:
        FileExistsError: When eight new names were all taken.
    """
    folder, base = os.path.split(os.path.abspath(path))
    for _ in range(8):
        tmp = _temp_name(folder, base)
        try:
            fd = os.open(tmp, _TEMP_FLAGS, 0o600)
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError(f"no new temporary name beside {path}")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class KeptCheckpoints:
    """The checkpoints that a managed run keeps for a fork from a past step, thinned evenly.

    ``keep`` runs at each step boundary, before the weights change. It writes
    ``<folder>/checkpoint-<step>.pt`` for the first step it sees, and later for a step at least
    ``stride`` steps past the newest kept one while the time of the kept checkpoints is at most
    ``share`` of the wall clock since the first. When more than ``limit`` are kept, the second,
    fourth and every other second one goes, the oldest and the newest stay, and the stride
    doubles. The limit is also at most ``bytes_max`` over the size of one checkpoint; below 2, the
    run keeps no more than its first.

    Attributes:
        kept (list[tuple[int, str]]): The kept steps and their files, oldest first.
        stride (int): The fewest steps between two kept checkpoints.
        off (bool): ``True`` once the run keeps no more.
    """

    def __init__(self, folder, clock=time.monotonic, limit=KEEP_MAX, share=KEEP_SHARE,
                 bytes_max=KEEP_BYTES):
        self.folder = folder
        self.extension = ".pt"
        self.clock = clock
        self.limit = limit
        self.share = share
        self.bytes_max = bytes_max
        self.kept = []
        self.stride = 1
        self.off = False
        self._spent = 0.0
        self._since = None

    def due(self, step: int) -> bool:
        """Whether ``keep`` writes the checkpoint of ``step``."""
        if self.off:
            return False
        if not self.kept:
            return True
        if step < self.kept[-1][0] + self.stride:
            return False
        return self._spent <= self.share * (self.clock() - self._since)

    def keep(self, step: int, state) -> str | None:
        """Write the checkpoint of ``step`` when it is due, and thin the kept ones.

        Args:
            step (int): The step of the boundary.
            state (Callable[[], bytes]): Gives the bytes of the checkpoint; it runs only when the
                checkpoint is due.

        Returns:
            str | None: The new file's path, or ``None`` when the step is not kept.

        Raises:
            OSError: When the folder or the file cannot be written.
        """
        if not self.due(step):
            return None
        start = self.clock()
        if self._since is None:
            self._since = start
        if os.path.islink(self.folder):
            raise OSError(f"{self.folder} is a link")
        os.makedirs(self.folder, mode=0o700, exist_ok=True)
        data = state()
        path = os.path.join(self.folder, f"checkpoint-{step}{self.extension}")
        _write(path, data)
        self.kept.append((step, path))
        self._spent += self.clock() - start
        limit = min(self.limit, self.bytes_max // max(len(data), 1))
        if limit < 2:
            self.off = True
            print(f"caliper: the run keeps no checkpoint after step {step}: one holds "
                  f"{len(data)} bytes, and two pass {self.bytes_max}", flush=True)
        elif len(self.kept) > limit:
            last = len(self.kept) - 1
            gone = [kp for i, kp in enumerate(self.kept) if i % 2 == 1 and i != last]
            self.kept = [kp for i, kp in enumerate(self.kept) if i % 2 == 0 or i == last]
            self.stride *= 2
            for _, p in gone:
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass
        return path


class ManagedRun:
    """The managed run of this process, from the variables that the model runner set.

    Attributes:
        id (str | None): ``CALIPER_RUN_ID``.
        dir (str): ``CALIPER_RUN_DIR``.
        link_file (str | None): ``CALIPER_RUN_LINK_FILE``.
        params (dict): ``CALIPER_RUN_PARAMS``, parsed.
        resume (str | None): ``CALIPER_RUN_RESUME``.
        parent (str | None): ``CALIPER_RUN_PARENT``.
        interventions (list[dict]): ``CALIPER_RUN_INTERVENTIONS``, parsed.
        kept (KeptCheckpoints): The checkpoints that the run keeps for a fork from a past step.
        values (dict): The value each parameter gets: the checkpoint's after a resume, then
            ``params``.
        step (int | None): The step of the checkpoint that the run resumed from.
        stopping (bool): ``True`` once ``caliper.stop`` ran.
    """

    def __init__(self, run_id, run_dir, link_file=None, params=None, resume=None, parent=None,
                 interventions=None):
        self.id = run_id
        self.dir = run_dir
        self.link_file = link_file
        self.params = dict(params or {})
        self.resume = resume
        self.parent = parent
        self.interventions = list(interventions or [])
        self.kept = KeptCheckpoints(os.path.join(run_dir, KEEP_DIR))
        self._first = None               # the hook that applies the interventions
        self._restored_interventions = NONE
        self._intervening = False
        self.values = dict(self.params)
        self.step = None
        self.stopping = False
        self._resumed = False
        self._pkg = None
        self.inference = None           # built-in hosted inference metadata owner

    @classmethod
    def from_env(cls, env=None):
        """The managed run of the variables in ``env`` (``os.environ``), or ``None``.

        Args:
            env (Mapping[str, str] | None): The environment. ``None`` reads ``os.environ``.

        Returns:
            ManagedRun | None: ``None`` when ``CALIPER_RUN_DIR`` is not set.

        Raises:
            ValueError: When ``CALIPER_RUN_INTERVENTIONS`` is not a JSON array of interventions.
        """
        env = os.environ if env is None else env
        vals = tuple(env.get(k) or None for k in _VARS)
        return cls._make(vals)

    @classmethod
    def _make(cls, vals):
        run_id, run_dir, link, params, resume, parent, interventions = vals
        if run_dir is None:
            return None
        return cls(run_id, run_dir, link, _params(params), resume, parent,
                   _interventions(interventions))

    def value(self, name: str):
        """The value that parameter ``name`` gets at its registration, or ``NONE``."""
        return self.values.get(name, NONE)

    # -- watch(): before the hooks ---------------------------------------------------
    def prepare(self, pkg, model, optimizer) -> None:
        """Start the server, and load the checkpoint of a fork once per process.

        Args:
            pkg (module): The caliper package.
            model (torch.nn.Module): The watched model.
            optimizer (torch.optim.Optimizer | None): Its optimizer, or ``None``.

        Raises:
            OSError: When the checkpoint file cannot be read.
            ValueError: When the file is not a checkpoint of a managed run.
            RuntimeError: When the model does not match the checkpoint.
        """
        self._pkg = pkg
        watch_parent()
        with pkg.LOCK:
            pkg._auto_tried = True       # the server carries the views, not the shared memory
        pkg.serve()
        if self.resume and not self._resumed:
            self._resumed = True
            self._load(pkg, model, optimizer)

    def _load(self, pkg, model, optimizer) -> None:
        if self.inference is not None:
            from .interp._hosted import CHECKPOINT_BYTES, checkpoint_metadata
        with open(self.resume, "rb") as f:
            data = f.read(CHECKPOINT_BYTES + 1) if self.inference is not None else f.read()
        if self.inference is not None:
            if len(data) > CHECKPOINT_BYTES:
                raise ValueError("inference checkpoint exceeds the metadata limit")
            extra = json.loads(data)
            if not checkpoint_metadata(extra):
                raise ValueError("inference checkpoint contains invalid metadata")
            self.inference.load_checkpoint(extra)
        else:
            if optimizer is not None and not hasattr(optimizer, "load_state_dict"):
                optimizer = None         # watch() refuses it after this
            extra = TORCH.restore(data, model, optimizer)
            if extra.get("format") != FORMAT:
                raise ValueError(f"{self.resume} is not a checkpoint of a managed run")
            self._restored_interventions = extra.get("interventions", NONE)
        step = int(extra.get("step") or 0)
        self.step = step
        ctl = pkg._CTL
        ctl.n = max(step - 1, 0)         # the first checkpoint of this run has the file's step
        saved = extra.get("params")
        self.values = {**(saved if isinstance(saved, dict) else {}), **self.params}
        with ctl.lock:
            registered = [(p, self.values[n]) for n, p in ctl.params.items() if n in self.values]
        for p, v in registered:
            _set(p, v)
        print(f"caliper: run {self.id} resumes from {self.resume} at step {step}", flush=True)

    # -- watch(): after the hooks ----------------------------------------------------
    def attach(self, handle, model, optimizer) -> None:
        """Register the two actions and the checkpoint of an inference run, then write the link.

        Args:
            handle (caliper._watch.WatchHandle): The handle of the watch, which removes the
                actions and the hook with its own.
            model (torch.nn.Module): The watched model.
            optimizer (torch.optim.Optimizer | None): Its optimizer, or ``None``.
        """
        pkg = self._pkg
        handle._actions.append(pkg.action(
            CHECKPOINT, lambda: self.checkpoint(model, optimizer),
            help="Saves the model, the optimizer, the step and the parameter values into the run "
                 "folder at the next checkpoint and returns the path of the file."))
        handle._actions.append(pkg.action(
            STOP, self.request_stop,
            help="Ends the script cleanly at the next checkpoint, before the weights change."))
        if optimizer is None:
            handle._handles.append(model.register_forward_hook(self._forward))
        elif hasattr(optimizer, "register_step_pre_hook"):
            # after the pre-hook of caliper._watch_run, which calls caliper.step() first
            handle._handles.append(optimizer.register_step_pre_hook(
                lambda opt, args, kwargs: self._keep(model, optimizer)))
        if self.interventions or self._restored_interventions is not NONE:
            self._first = model.register_forward_pre_hook(lambda mod, args: self._intervene())
            handle._handles.append(self._first)
        self.write_link(pkg.serve().url)

    def write_link(self, url: str) -> None:
        """Write ``url`` into the link file atomically, as the only content, without a newline."""
        if not self.link_file:
            warnings.warn(f"caliper: {ENV_DIR} is set but {ENV_LINK} is not, so the model "
                          "runner cannot find the stream link", RuntimeWarning, stacklevel=2)
            return
        _write(self.link_file, url.encode("utf-8"))

    # -- the actions and the checkpoint ------------------------------------------------
    def _state(self, model, optimizer, step) -> bytes:
        """The bytes of a checkpoint of ``step`` (``caliper._adapter.TorchAdapter.state``)."""
        ctl = self._pkg._CTL
        with ctl.lock:
            params = {n: p.value for n, p in ctl.params.items()}
        extra = {"format": FORMAT, "version": VERSION, "run": self.id, "parent": self.parent,
                 "step": step, "params": params}
        if self.inference is not None:
            from .interp._hosted import CHECKPOINT_FORMAT, checkpoint_metadata
            extra.update(self.inference.state_dict(), format=CHECKPOINT_FORMAT)
            if not checkpoint_metadata(extra):
                raise ValueError("inference checkpoint exceeds the metadata limits")
            return json.dumps(extra, ensure_ascii=False, allow_nan=False).encode("utf-8")
        interventions = getattr(ctl, "intervention_service", None)
        if interventions is not None and interventions.model is model:
            extra["interventions"] = interventions.state_dict()
        return TORCH.state(model, optimizer, extra)

    def checkpoint(self, model, optimizer) -> str:
        """The action ``caliper.checkpoint``: write the state and return the file's path."""
        step = self._pkg._CTL.n
        data = self._state(model, optimizer, step)
        extension = ".json" if self.inference is not None else ".pt"
        path = os.path.join(self.dir, f"checkpoint-{step}{extension}")
        _write(path, data)
        return path

    def _keep(self, model, optimizer) -> None:
        """At each step boundary before the weights change: keep a checkpoint when one is due.
        An error stops the keeping with a warning and never stops the step."""
        step = self._pkg._CTL.n
        if self.stopping or self.kept.off or not isinstance(step, int) or isinstance(step, bool) \
                or step < 0:
            return
        try:
            self.kept.keep(step, lambda: self._state(model, optimizer, step))
        except Exception as e:           # noqa: BLE001 - the training goes on
            self.kept.off = True
            warnings.warn(f"caliper: run {self.id} keeps no more checkpoints: "
                          f"{type(e).__name__}: {e}", RuntimeWarning, stacklevel=2)

    def _intervene(self) -> None:
        """Restore saved effects after resource binding and before the first script forward."""
        from .interp._output import internal_forward
        if self._first is None or self._intervening:
            return
        self._intervening = True
        try:
            with internal_forward():
                if self._restored_interventions is not NONE:
                    service = self._pkg._CTL.intervention_service
                    if service is None:
                        raise RuntimeError("the fork has no intervention service for its saved state")
                    service.load_state_dict(self._restored_interventions, restore_active=True)
                if self.interventions:
                    self._apply_fork_overrides()
            self._restored_interventions = NONE
            first, self._first = self._first, None
            first.remove()
        finally:
            # A refused restore must also refuse a later forward after a caught error.
            self._intervening = False

    def _apply_fork_overrides(self) -> None:
        """Apply explicit overrides before the first forward and retain unrelated requests.

        Raises:
            RuntimeError: When the script takes no interventions, or one is refused; the fork
                must not train without them.
        """
        from .interp._intervene import InterventionError
        service = self._pkg._CTL.intervention_service
        error = None
        if service is None:
            error = "the script watches no model with caliper.watch(model, optimizer)"
        else:
            try:
                service.apply_overrides(self.interventions)
            except InterventionError as exc:
                error = str(exc)
        if error is not None:
            print(f"caliper: run {self.id} cannot apply its interventions: {error}", flush=True)
            raise RuntimeError(f"the fork's interventions cannot apply: {error}")
        print(f"caliper: run {self.id} applies "
              + ", ".join(f"{iv.get('kind', iv.get('op'))} {iv.get('point')}" for iv in self.interventions)
              + " from its first step", flush=True)

    def request_stop(self) -> str:
        """The action ``caliper.stop``: end the script when its checkpoint returns."""
        self.stopping = True
        ctl = self._pkg._CTL
        ctl._release()                   # a paused run goes on to the end of its checkpoint
        return f"stopping at step {ctl.n}"

    def exit(self) -> None:
        """Send what is pending, close the stream server, then raise ``SystemExit(0)`` (the hook
        of the checkpoint).

        The close sends the final replies of the actions and the last run state, then BYE and
        the close 1001 to each viewer, with a bounded wait (``NetServer.close``). It comes before
        the ``SystemExit``, so that neither the script's own cleanup nor a thread that the
        interpreter waits for at its exit delays it.
        """
        pkg = self._pkg
        try:
            pkg.flush(FLUSH_S)
        except Exception:                # noqa: BLE001 - the stop goes on
            pass
        print(f"caliper: run {self.id} stops at step {pkg._CTL.n}", flush=True)
        from ._net import current_server
        server = current_server()
        if server is not None:
            try:
                server.close()
            except Exception:            # noqa: BLE001 - the stop goes on
                pass
        raise SystemExit(0)

    def _forward(self, module, inputs, output) -> None:
        from .interp._output import is_internal_forward
        ctl = self._pkg._CTL
        interpretation = ctl.interpretation_service
        interventions = ctl.intervention_service
        if (is_internal_forward() or (interpretation is not None and interpretation.running)
                or (interventions is not None and interventions.executor.preparing_source)):
            return
        self._pkg.step()
        if self.stopping:
            self.exit()


def _set(p, v) -> None:
    try:
        p.set(v)
    except ValueError as e:
        warnings.warn(f"caliper: the managed run's value of parameter {p.name!r} does not apply: "
                      f"{e}", RuntimeWarning, stacklevel=3)


_key = None
_run = None
_parent_watch = None


def _alive(pid: int) -> bool:
    """Whether a process of id ``pid`` exists (POSIX; ``os.kill`` ends a process on Windows)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _app_pid(env=None):
    """``CALIPER_RUN_APP_PID`` as a process id, or ``None`` when it is not set or not valid."""
    text = (os.environ if env is None else env).get(ENV_APP_PID) or ""
    try:
        pid = int(text)
    except ValueError:
        return None
    return pid if pid > 1 and pid != os.getpid() else None


def _parent_gone(app: int, ppid: int, interval: float) -> None:
    # While the app is the parent, a new parent says that it ended, also before its parent reaped it.
    while (os.getppid() == ppid) if app == ppid else _alive(app):
        time.sleep(interval)
    try:
        sys.stderr.write(f"caliper: the process {app} that started this managed run ended, so "
                         "the run ends\n")
        sys.stderr.flush()
    except Exception:                    # noqa: BLE001 - the exit goes on
        pass
    os._exit(PARENT_GONE_EXIT)


def watch_parent(interval: float = PARENT_POLL_S) -> bool:
    """End this process when the app that started it ends (POSIX), once per process.

    The app ends its children by their process ids when it closes. After a crash of the app
    nothing does, so the child itself sees that the app ended and exits with
    ``PARENT_GONE_EXIT``. The app is ``CALIPER_RUN_APP_PID``, else the parent.

    Args:
        interval (float): Seconds between two looks.

    Returns:
        bool: Whether a watch runs now. ``False`` on Windows, where the runner's job object ends
        the children, and without ``CALIPER_RUN_APP_PID`` when the parent is already the init
        process.
    """
    global _parent_watch
    if _parent_watch is not None:
        return True
    if os.name != "posix":
        return False
    ppid = os.getppid()
    app = _app_pid()
    if app is None:
        if ppid <= 1:
            return False
        app = ppid
    _parent_watch = threading.Thread(target=_parent_gone, args=(app, ppid, interval),
                                     name="caliper-parent-watch", daemon=True)
    _parent_watch.start()
    return True


def current():
    """The managed run of this process, or ``None`` when ``CALIPER_RUN_DIR`` is not set.

    The object stays the same while the variables stay the same.
    """
    global _key, _run
    vals = tuple(os.environ.get(k) or None for k in _VARS)
    if vals != _key:
        _key = vals
        _run = ManagedRun._make(vals)
    return _run


def value(name: str):
    """The value that the managed run gives parameter ``name`` at its registration, or ``NONE``."""
    run = current()
    return NONE if run is None else run.values.get(name, NONE)

