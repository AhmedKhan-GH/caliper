"""Run notebooks headless, and screenshot them in JupyterLab under headless Chromium.

`run_notebook` executes a notebook with nbclient. `notebook_session` starts a
real JupyterLab server (``--no-browser``, 127.0.0.1, a free port, a random
token) on a temporary copy of the notebook, opens it in headless Chromium,
runs all cells and hands back a `NotebookSession` for screenshots and checks.
`screenshot_notebook` does all of that and saves one screenshot.

Jupyter and IPython state (config, data, runtime, kernel specs, JupyterLab
settings and workspaces, IPython profile) goes under a session directory in
the harness work directory, never into the user's own Jupyter folders. The
work directory is `$CALIPER_HEADLESS_DIR`, or the system temporary folder.

Command line, from the repository root:

    python -m tests.headless.notebook <notebook.ipynb> <out.png> [--nonblank SEL] [--outputs]
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import nbformat
from playwright.sync_api import Page

from .browser import DEFAULT_MODE, MODES, launch, new_page, probe_page, work_dir
from .screenshot import PixelStats, WaitFor, check, element_png, nonblank_now, png_stats

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class NotebookCellError(RuntimeError):
    """A notebook cell raised. Carries the cell index, source and traceback."""

    def __init__(self, cell: int, source: str, ename: str, evalue: str, traceback: str,
                 notebook=None):
        self.cell, self.source, self.ename, self.evalue = cell, source, ename, evalue
        self.traceback = _ANSI.sub("", traceback)
        self.notebook = notebook
        super().__init__(f"cell {cell} raised {ename}: {evalue}\n"
                         f"--- source ---\n{source}\n--- traceback ---\n{self.traceback}")


def _kernel_json(kernel_name: str, python: str | None, extra_env: dict | None) -> dict:
    if python is None and kernel_name != "python3":
        # A kernel spec installed elsewhere: read it from the user's normal
        # Jupyter paths (read only) and copy it into the isolated ones.
        from jupyter_client.kernelspec import KernelSpecManager

        spec = KernelSpecManager().get_kernel_spec(kernel_name).to_dict()
    else:
        spec = {
            "argv": [python or sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": f"Python ({kernel_name})",
            "language": "python",
            "metadata": {"debugger": False},
        }
    if extra_env:
        spec["env"] = {**spec.get("env", {}), **extra_env}
    return spec


def jupyter_env(state: Path, *, kernel_name: str = "python3", python: str | None = None,
                kernel_env: dict | None = None) -> dict[str, str]:
    """Create isolated Jupyter folders under `state` and return the variables naming them.

    The folder also receives a kernel spec named `kernel_name` that starts
    `python` (default: this interpreter, which must have ipykernel).
    """
    dirs = {
        "JUPYTER_CONFIG_DIR": state / "config",
        "JUPYTER_DATA_DIR": state / "data",
        "JUPYTER_RUNTIME_DIR": state / "runtime",
        "JUPYTER_PATH": state / "path",
        "IPYTHONDIR": state / "ipython",
        "JUPYTERLAB_SETTINGS_DIR": state / "lab-settings",
        "JUPYTERLAB_WORKSPACES_DIR": state / "lab-workspaces",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    spec_dir = dirs["JUPYTER_PATH"] / "kernels" / kernel_name
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "kernel.json").write_text(json.dumps(_kernel_json(kernel_name, python, kernel_env), indent=1))
    env = {k: str(v) for k, v in dirs.items()}
    env["JUPYTER_PLATFORM_DIRS"] = "1"
    return env


@contextlib.contextmanager
def _state_dir(work: str | Path | None, prefix: str, keep: bool = False,
               keep_on_error: bool = True) -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=work_dir(work)))
    ok = False
    try:
        yield path
        ok = True
    finally:
        if not keep and (ok or not keep_on_error):
            shutil.rmtree(path, ignore_errors=True)


def _pythonpath(notebooks: Iterable[Path]) -> str:
    """The notebooks' folders, then any PYTHONPATH this process already has."""
    parts = [str(Path(n).resolve().parent) for n in notebooks]
    parts += [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    return os.pathsep.join(dict.fromkeys(parts))


def _first_error(nb) -> tuple[int, dict] | None:
    for i, cell in enumerate(nb.cells):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                return i, out
    return None


def run_notebook(
    path: str | Path,
    *,
    timeout: float = 600,
    kernel_name: str = "python3",
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    python: str | None = None,
    work: str | Path | None = None,
):
    """Execute a notebook with nbclient and return the executed notebook.

    `timeout` is per cell in seconds. The kernel runs in `cwd`, by default a
    scratch folder in the work directory that is removed afterwards, so a
    notebook never writes into the repository; the notebook's own folder is
    on the kernel's PYTHONPATH, so modules beside it import as usual. The
    kernel sees `env` on top of this process's environment and starts
    `python` (default: this interpreter). Raises `NotebookCellError` with the
    failing cell's traceback when a cell raises.
    """
    from jupyter_client.kernelspec import KernelSpecManager
    from jupyter_client.manager import KernelManager
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    path = Path(path)
    nb = nbformat.read(path, as_version=4)
    with _state_dir(work, "run-", keep_on_error=False) as state:
        jenv = jupyter_env(state / "jupyter", kernel_name=kernel_name, python=python,
                           kernel_env={"PYTHONPATH": _pythonpath([path])})
        if cwd is None:
            cwd = state / "cwd"
            cwd.mkdir()
        ksm = KernelSpecManager(kernel_dirs=[str(Path(jenv["JUPYTER_PATH"]) / "kernels")])
        km = KernelManager(
            kernel_name=kernel_name,
            kernel_spec_manager=ksm,
            connection_file=str(Path(jenv["JUPYTER_RUNTIME_DIR"]) / f"kernel-{secrets.token_hex(6)}.json"),
        )
        client = NotebookClient(
            nb, km=km, timeout=int(timeout), kernel_name=kernel_name,
            resources={"metadata": {"path": str(Path(cwd).resolve())}},
        )
        try:
            client.execute(env={**os.environ, **jenv, **(env or {})})
        except CellExecutionError as exc:
            found = _first_error(nb)
            if found is None:
                raise
            i, out = found
            raise NotebookCellError(i, nb.cells[i].source, out.get("ename", ""), out.get("evalue", ""),
                                    "\n".join(out.get("traceback", [])), notebook=nb) from exc
        finally:
            # nbclient leaves a kernel manager it did not create running.
            if client.kc is not None:
                client.kc.stop_channels()
            if km.has_kernel:
                km.shutdown_kernel(now=False)
    return nb


# --- JupyterLab under headless Chromium ---------------------------------------

_LAB_SETTINGS = {
    # Render every cell, so output screenshots never meet a virtualised cell,
    # and leave no blank space after the last cell in a fitted screenshot.
    "@jupyterlab/notebook-extension/tracker.jupyterlab-settings": {
        "windowingMode": "none", "scrollPastEnd": False,
    },
    "@jupyterlab/apputils-extension/notification.jupyterlab-settings": {
        "fetchNews": "false", "checkForUpdates": False,
    },
    "@jupyterlab/docmanager-extension/plugin.jupyterlab-settings": {"autosave": False},
}

# Finds the panel of the notebook named `name` in the JupyterLab shell.
_PANEL_JS = """
window.__nbPanel = (name) => {
  for (const w of jupyterapp.shell.widgets('main')) {
    if (w.context && w.context.path === name && w.content && w.content.widgets) return w;
  }
  return null;
};
"""

_STATE_JS = """
(name) => {
  const p = __nbPanel(name);
  if (!p) return null;
  const kernel = p.sessionContext.session && p.sessionContext.session.kernel;
  const cells = p.content.widgets.map((c, i) => ({
    index: i, type: c.model.type,
    empty: c.model.sharedModel.getSource().trim() === '',
    count: c.model.type === 'code' ? c.model.executionCount : null,
    running: c.model.type === 'code' && c.model.executionState === 'running',
  }));
  const errors = [];
  p.content.widgets.forEach((c, i) => {
    if (c.model.type !== 'code') return;
    for (const o of c.model.outputs.toJSON()) {
      if (o.output_type === 'error') errors.push({cell: i, ename: o.ename, evalue: o.evalue,
        traceback: (o.traceback || []).join('\\n'), source: c.model.sharedModel.getSource()});
    }
  });
  return {kernel: kernel ? kernel.status : null, cells, errors};
}
"""

_FIT_JS = """
(name) => {
  const p = __nbPanel(name);
  const scrollers = [p.content.node, ...p.content.node.querySelectorAll('.jp-WindowedPanel-outer')];
  let extra = 0;
  for (const s of scrollers) extra = Math.max(extra, s.scrollHeight - s.clientHeight);
  return extra;
}
"""


# The page that opens a notebook in each front end; the token is appended.
_FRONTENDS = {"lab": "/lab/tree/{name}?reset&", "notebook": "/notebooks/{name}?"}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _children(pid: int) -> list[int]:
    try:
        out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(x) for x in out.split()]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _kill_group(pid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        if hasattr(os, "killpg"):
            os.killpg(pid, sig)
        else:
            os.kill(pid, sig)


@dataclass
class LabServer:
    """A running JupyterLab server started by `jupyter_lab`."""

    url: str
    token: str
    root: Path
    state: Path
    log: Path
    process: subprocess.Popen

    def api(self, method: str, route: str, body: dict | None = None, timeout: float = 10.0):
        data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
        req = urllib.request.Request(f"{self.url}{route}", data=data, method=method,
                                     headers={"Authorization": f"token {self.token}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode() or "null"
        return json.loads(text)

    def log_tail(self, lines: int = 40) -> str:
        try:
            return "\n".join(self.log.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""


def _prepare_root(root: Path, notebooks: Iterable[Path], extra_files: Iterable[Path],
                  kernel_name: str) -> list[str]:
    names = []
    for nb_path in notebooks:
        nb = nbformat.read(nb_path, as_version=4)
        # A fresh run: no stale outputs, and the kernel spec this server offers.
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.outputs, cell.execution_count = [], None
        nb.metadata["kernelspec"] = {"name": kernel_name, "display_name": f"Python ({kernel_name})",
                                     "language": "python"}
        nbformat.write(nb, root / nb_path.name)
        names.append(nb_path.name)
    for f in extra_files:
        f = Path(f)
        (shutil.copytree if f.is_dir() else shutil.copy2)(f, root / f.name)
    return names


@contextlib.contextmanager
def jupyter_lab(
    notebooks: Iterable[str | Path],
    *,
    extra_files: Iterable[str | Path] = (),
    kernel_name: str = "python3",
    python: str | None = None,
    env: dict[str, str] | None = None,
    theme: str = "light",
    work: str | Path | None = None,
    startup_timeout: float = 60.0,
    keep: bool = False,
) -> Iterator[LabServer]:
    """Start JupyterLab with ``--no-browser`` on 127.0.0.1 and stop it on exit.

    The server root is a fresh folder holding copies of `notebooks` (outputs
    cleared) and `extra_files`. Kernels start `python` and see `env`, and the
    folder of each original notebook is on their PYTHONPATH, so modules
    beside a notebook import as they would in place. The session folder is
    removed afterwards unless `keep` is true or an error occurred.
    """
    notebooks = [Path(n).resolve() for n in notebooks]
    with _state_dir(work, "lab-", keep=keep) as state:
        root = state / "root"
        root.mkdir()
        _prepare_root(root, notebooks, extra_files, kernel_name)
        jenv = jupyter_env(state / "jupyter", kernel_name=kernel_name, python=python,
                           kernel_env={"PYTHONPATH": _pythonpath(notebooks)})
        settings = Path(jenv["JUPYTERLAB_SETTINGS_DIR"])
        lab_settings = dict(_LAB_SETTINGS)
        if theme == "dark":
            lab_settings["@jupyterlab/apputils-extension/themes.jupyterlab-settings"] = {
                "theme": "JupyterLab Dark"}
        for rel, value in lab_settings.items():
            target = settings / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(value))

        token = secrets.token_hex(24)
        log = state / "server.log"
        server_env = {**os.environ, **jenv, **(env or {})}
        last_error = ""
        for _attempt in range(3):
            port = _free_port()
            config = {
                "ServerApp": {
                    "ip": "127.0.0.1", "port": port, "port_retries": 0, "open_browser": False,
                    "root_dir": str(root), "terminals_enabled": False,
                    "jpserver_extensions": {"jupyter_lsp": False},
                },
                "IdentityProvider": {"token": token},
                "LabApp": {
                    "expose_app_in_browser": True, "news_url": None,
                    "check_for_updates_class": "jupyterlab.handlers.announcements.NeverCheckForUpdate",
                    "extension_manager": "readonly",
                },
                # The Notebook 7 front end, served by the same server when installed.
                "JupyterNotebookApp": {"expose_app_in_browser": True},
            }
            (Path(jenv["JUPYTER_CONFIG_DIR"]) / "jupyter_server_config.json").write_text(json.dumps(config))
            with open(log, "ab") as log_file:
                process = subprocess.Popen(
                    [sys.executable, "-m", "jupyterlab", "--no-browser"],
                    cwd=root, env=server_env, stdout=log_file, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                )
            server = LabServer(f"http://127.0.0.1:{port}", token, root, state, log, process)
            try:
                if _wait_ready(server, startup_timeout):
                    break
                last_error = server.log_tail()
            except BaseException:
                _stop_server(server)
                raise
            _stop_server(server)
        else:
            raise RuntimeError(f"JupyterLab did not start; log ({log}):\n{last_error}")

        try:
            yield server
        except BaseException as exc:
            exc.add_note(f"JupyterLab session folder: {state}")
            raise
        finally:
            _stop_server(server)


def _wait_ready(server: LabServer, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            return False
        try:
            server.api("GET", "/api/status", timeout=2)
            return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            time.sleep(0.2)
    return False


def _stop_server(server: LabServer) -> None:
    """Shut down the server and its kernels; escalate to signals if needed."""
    proc = server.process
    kernels = _children(proc.pid)
    if proc.poll() is None:
        with contextlib.suppress(Exception):
            server.api("POST", "/api/shutdown", timeout=10)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _kill_group(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _kill_group(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
    # Kernels run in their own process groups; make sure none outlives the server.
    deadline = time.monotonic() + 5
    while any(_alive(k) for k in kernels) and time.monotonic() < deadline:
        time.sleep(0.1)
    for k in kernels:
        if _alive(k):
            _kill_group(k, signal.SIGKILL)


@dataclass
class NotebookSession:
    """A notebook open in JupyterLab in headless Chromium."""

    page: Page
    server: LabServer
    name: str
    headers: dict = field(default_factory=dict)
    console: list[str] = field(default_factory=list)

    def state(self) -> dict | None:
        return self.page.evaluate(_STATE_JS, self.name)

    def raise_errors(self) -> None:
        state = self.state()
        if state and state["errors"]:
            e = state["errors"][0]
            raise NotebookCellError(e["cell"], e["source"], e["ename"], e["evalue"], e["traceback"])
        if state and state["kernel"] == "dead":
            raise RuntimeError(f"the kernel died; server log:\n{self.server.log_tail()}")

    def kernel_busy(self) -> bool:
        """True while a cell runs or waits to run.

        This looks at each cell's execution state as well as the kernel
        status, because with ipykernel 7 and a widget on the page JupyterLab
        can show the kernel as idle while a cell is running.
        """
        state = self.state()
        return bool(state) and (state["kernel"] == "busy" or any(c["running"] for c in state["cells"]))

    def finished(self) -> bool:
        """True when every non-empty code cell has run and the kernel is idle."""
        state = self.state()
        if not state:
            return False
        code = [c for c in state["cells"] if c["type"] == "code" and not c["empty"]]
        return state["kernel"] == "idle" and all(c["count"] is not None and not c["running"] for c in code)

    def run_all(self) -> None:
        """Start running all cells and return at once."""
        self.page.evaluate("() => { window.__runAll = jupyterapp.commands.execute("
                           "'notebook:run-all-cells').then(() => 'done', e => String(e)); }")

    def wait(self, wait_for: WaitFor = None, *, nonblank: str | None = None, idle: bool = False,
             timeout: float = 120.0) -> list[PixelStats]:
        """Wait for a predicate, for non-blank elements and/or for all cells to finish.

        Fails at once with `NotebookCellError` if a cell raises meanwhile.
        Returns the pixel statistics of the `nonblank` elements, if any.
        """
        deadline = time.monotonic() + timeout
        stats: list[PixelStats] = []
        while True:
            self.raise_errors()
            ok = wait_for is None or check(self.page, wait_for)
            if ok and nonblank:
                stats = nonblank_now(self.page, nonblank)
                ok = bool(stats) and all(s.nonblank() for s in stats)
            if ok and idle:
                ok = self.finished()
            if ok:
                return stats
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"notebook {self.name}: wait_for={wait_for!r} nonblank={nonblank!r} idle={idle} "
                    f"did not hold within {timeout} s; state {self.state()}; "
                    f"console {self.console[-5:]}")
            self.page.wait_for_timeout(200)

    def screenshot(self, out_png: str | Path, *, outputs: bool = False, fit: bool = True,
                   max_height: int = 8000) -> list[Path]:
        """Save the notebook view; with `outputs`, also every cell output on its own.

        With `fit`, the viewport grows so the whole notebook is in the shot.
        Returns the paths written, the full view first.
        """
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        size = self.page.viewport_size
        if fit:
            extra = self.page.evaluate(_FIT_JS, self.name)
            if extra > 0:
                self.page.set_viewport_size({"width": size["width"],
                                             "height": min(max_height, size["height"] + extra + 8)})
                self.page.wait_for_timeout(300)
        written = [out_png]
        self.page.screenshot(path=str(out_png), animations="allow")
        if outputs:
            cells = self.page.locator(".jp-NotebookPanel:visible .jp-Notebook .jp-Cell")
            for ci in range(cells.count()):
                outs = cells.nth(ci).locator(".jp-Cell-outputArea > .jp-OutputArea-child > .jp-OutputArea-output")
                for oi in range(outs.count()):
                    box = outs.nth(oi).bounding_box()
                    if not box or box["width"] < 1 or box["height"] < 1:
                        continue
                    target = out_png.with_name(f"{out_png.stem}-cell{ci}-out{oi}.png")
                    outs.nth(oi).screenshot(path=str(target), animations="allow")
                    written.append(target)
        if fit and size:
            self.page.set_viewport_size(size)
        return written

    def element_stats(self, selector: str) -> PixelStats:
        """Pixel statistics of the first element matching `selector` (shadow DOM included)."""
        return png_stats(element_png(self.page, selector))

    def probe_webgpu(self) -> dict:
        """What WebGPU offers inside this JupyterLab page."""
        return probe_page(self.page)


@contextlib.contextmanager
def notebook_session(
    path: str | Path,
    *,
    extra_files: Iterable[str | Path] = (),
    kernel_name: str = "python3",
    python: str | None = None,
    env: dict[str, str] | None = None,
    mode: str = DEFAULT_MODE,
    viewport: tuple[int, int] = (1280, 900),
    device_scale_factor: float = 1,
    theme: str = "light",
    run_all: bool = True,
    frontend: str = "lab",
    timeout: float = 120.0,
    work: str | Path | None = None,
    keep: bool = False,
) -> Iterator[NotebookSession]:
    """Open a copy of the notebook at `path` in JupyterLab in headless Chromium.

    `frontend` is ``"lab"`` (JupyterLab) or ``"notebook"`` (the Jupyter
    Notebook 7 front end, which needs the ``notebook`` package). Waits for the
    kernel, collapses the side panels and, with `run_all`, starts running all
    cells. Everything (browser, kernel, server) is shut down on exit, also
    after an error.
    """
    if frontend not in _FRONTENDS:
        raise ValueError(f"frontend must be one of {sorted(_FRONTENDS)}, not {frontend!r}")
    path = Path(path).resolve()
    with jupyter_lab([path], extra_files=extra_files, kernel_name=kernel_name, python=python,
                     env=env, theme=theme, work=work, keep=keep) as server, launch(mode) as browser:
        page = new_page(browser, viewport=viewport, device_scale_factor=device_scale_factor)
        page.set_default_timeout(timeout * 1000)
        session = NotebookSession(page=page, server=server, name=path.name)
        page.on("console", lambda m: session.console.append(f"{m.type}: {m.text}")
                if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: session.console.append(f"pageerror: {e}"))
        page.add_init_script(_PANEL_JS)
        response = page.goto(f"{server.url}{_FRONTENDS[frontend].format(name=urllib.parse.quote(path.name))}"
                             f"token={server.token}")
        session.headers = response.all_headers() if response else {}
        page.wait_for_function("() => window.jupyterapp && jupyterapp.shell", timeout=timeout * 1000)
        page.wait_for_function(
            "(name) => { const p = __nbPanel(name); const s = p && p.sessionContext;"
            " const k = s && s.session && s.session.kernel; return !!k && k.status === 'idle'; }",
            arg=path.name, timeout=timeout * 1000)
        page.evaluate("() => { const s = jupyterapp.shell;"
                      " if (s.collapseLeft) s.collapseLeft(); if (s.collapseRight) s.collapseRight(); }")
        if run_all:
            session.run_all()
        yield session


@dataclass
class NotebookScreenshot:
    """What `screenshot_notebook` produced."""

    pngs: list[Path]
    stats: list[PixelStats]
    headers: dict
    webgpu: dict
    console: list[str]


def screenshot_notebook(
    path: str | Path,
    out_png: str | Path,
    *,
    wait_for: WaitFor = None,
    nonblank: str | None = None,
    idle: bool | None = None,
    outputs: bool = False,
    settle: float = 0.0,
    timeout: float = 120.0,
    **session_kwargs,
) -> NotebookScreenshot:
    """Run a notebook in JupyterLab under headless Chromium and screenshot it.

    Waits for `wait_for` (selector, JavaScript predicate or callable) and for
    the `nonblank` elements (open shadow roots included) to be non-blank.
    `idle` waits for all cells to finish; by default it does so only when no
    other condition is given. Takes the full notebook view and, with
    `outputs`, one PNG per cell output. `session_kwargs` go to
    `notebook_session`.
    """
    if idle is None:
        idle = wait_for is None and nonblank is None
    with notebook_session(path, timeout=timeout, **session_kwargs) as session:
        stats = session.wait(wait_for, nonblank=nonblank, idle=idle, timeout=timeout)
        if settle:
            session.page.wait_for_timeout(settle * 1000)
        pngs = session.screenshot(out_png, outputs=outputs)
        return NotebookScreenshot(pngs=pngs, stats=stats, headers=session.headers,
                                  webgpu=session.probe_webgpu(), console=list(session.console))


def _main(argv: list[str] | None = None) -> int:
    import argparse

    from .screenshot import _size

    parser = argparse.ArgumentParser(
        prog="python -m tests.headless.notebook",
        description="Run a notebook in JupyterLab under headless Chromium and screenshot it, "
                    "or run it with nbclient (--run-only).",
    )
    parser.add_argument("notebook")
    parser.add_argument("out", nargs="?", help="output PNG path (not used with --run-only)")
    parser.add_argument("--run-only", action="store_true", help="execute with nbclient, no browser")
    parser.add_argument("--wait-for", help="CSS selector or JavaScript predicate to wait for")
    parser.add_argument("--nonblank", help="selector whose elements must be non-blank first")
    parser.add_argument("--idle", action="store_true", help="also wait until all cells finished")
    parser.add_argument("--outputs", action="store_true", help="also save one PNG per cell output")
    parser.add_argument("--mode", choices=sorted(MODES), default=DEFAULT_MODE)
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    parser.add_argument("--frontend", choices=["lab", "notebook"], default="lab")
    parser.add_argument("--viewport", type=_size, default=(1280, 900), help="WIDTHxHEIGHT")
    parser.add_argument("--scale", type=float, default=1.0, help="device scale factor")
    parser.add_argument("--settle", type=float, default=0.0, help="extra seconds before the shot")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds (default 120)")
    parser.add_argument("--python", help="interpreter for the kernel (default: this one)")
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                        help="environment variable for the kernel (repeatable)")
    ns = parser.parse_args(argv)
    env = dict(item.split("=", 1) for item in ns.env)
    if ns.run_only:
        nb = run_notebook(ns.notebook, timeout=ns.timeout, env=env, python=ns.python)
        print(json.dumps({"cells": len(nb.cells), "ok": True}))
        return 0
    if not ns.out:
        parser.error("out is required unless --run-only is given")
    result = screenshot_notebook(
        ns.notebook, ns.out, wait_for=ns.wait_for, nonblank=ns.nonblank,
        idle=True if ns.idle else None, outputs=ns.outputs, settle=ns.settle, timeout=ns.timeout,
        mode=ns.mode, theme=ns.theme, frontend=ns.frontend, viewport=ns.viewport,
        device_scale_factor=ns.scale,
        python=ns.python, env=env,
    )
    print(json.dumps({
        "pngs": [str(p.resolve()) for p in result.pngs],
        "nonblank": [s.to_dict() for s in result.stats],
        "webgpu": result.webgpu,
        "csp": result.headers.get("content-security-policy"),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
