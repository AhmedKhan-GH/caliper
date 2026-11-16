# Headless test harness

These tools check browser pages and notebook widgets without ever showing a window. A page with live views is opened in headless Chromium, and a notebook is run either with nbclient or in a real JupyterLab server that headless Chromium drives. Every check that something was drawn works on screenshots.

The harness runs on macOS. It needs nothing from the rest of the repository.

## Setup

Run these commands from the repository root. Any Python from 3.12 on works; 3.13 is what the harness was measured with.

```sh
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -r tests/headless/requirements.txt
.venv/bin/python -m playwright install chromium
export CALIPER_HEADLESS_DIR=/path/to/a/scratch/folder
```

`playwright install chromium` downloads Chromium and the Chromium headless shell into Playwright's shared browser cache (`~/Library/Caches/ms-playwright` on macOS), but only when the revision that the installed Playwright expects is missing. Builds of other revisions already in the cache are not used. Playwright 1.65 expects revision 1255 (Chromium 159).

`CALIPER_HEADLESS_DIR` is where the harness keeps everything it writes: JupyterLab sessions, notebook working folders, test pages and the screenshots of the self-tests (under `shots/`). Without it the harness uses a `caliper-headless` folder in the system temporary folder. Nothing is ever written into the repository.

To run a notebook that imports `caliper`, install the package into the same environment. Kernels start this environment's Python unless you pass another interpreter (`--python` or `python=`), which must have `ipykernel`.

## Commands

Screenshot a URL or a local HTML file (a local file is served from 127.0.0.1, because Chromium does not let a `file://` page import ES modules beside it):

```sh
.venv/bin/python -m tests.headless.screenshot http://127.0.0.1:8765/?token=T out.png --nonblank "caliper-view canvas"
```

Run a notebook in JupyterLab under headless Chromium and screenshot it, with one extra PNG per cell output (for a Caliper notebook, wait for `--nonblank "caliper-view canvas"` instead):

```sh
.venv/bin/python -m tests.headless.notebook tests/headless/mock_widget/selftest.ipynb out.png --nonblank ".mock-view canvas" --outputs
```

Run a notebook with nbclient only, without a browser:

```sh
.venv/bin/python -m tests.headless.notebook tests/headless/mock_widget/selftest.ipynb --run-only --env MOCK_LOOP_SECONDS=1
```

Probe WebGPU in every mode, and run the self-tests:

```sh
.venv/bin/python -m tests.headless.browser
.venv/bin/python -m pytest tests/headless
```

Both screenshot commands print pixel statistics as JSON. The notebook command also prints what WebGPU offers inside the JupyterLab page and JupyterLab's Content-Security-Policy. Common options are `--wait-for` (a CSS selector or a JavaScript predicate such as `"() => window.ready === true"`), `--nonblank` (a selector whose elements must all be non-blank), `--mode`, `--viewport 1280x800`, `--scale 2`, `--settle` and `--timeout`. The notebook command adds `--frontend lab|notebook`, `--theme light|dark`, `--idle`, `--python` and `--env NAME=VALUE`.

## Python API

| Module | Function | What it does |
|---|---|---|
| `browser` | `launch(mode)`, `new_page(browser)`, `open_page(url, mode=...)` | Start headless Chromium. `headless=False` raises `HeadedBrowserRefused`. |
| `browser` | `probe_webgpu(mode)` | Returns `has_gpu`, `adapter`, `device`, `software` and the adapter `info`. |
| `browser` | `static_server(folder)` | Serves a folder on 127.0.0.1 and yields its base URL. |
| `screenshot` | `screenshot_page(url, out_png, wait_for=, nonblank=, timeout=, viewport=, device_scale_factor=)` | Saves a page screenshot and returns its `PixelStats`. |
| `screenshot` | `screenshot_element(url, selector, out_png, ...)` | The same for one element. |
| `screenshot` | `png_stats(png)`, `png_difference(a, b)`, `canvas_stats(page, selector)`, `wait_nonblank(page, selector)` | Pixel checks. `PixelStats.nonblank()` is true when at least 0.5 % of the pixels differ from the background colour and there are at least two colours. |
| `notebook` | `run_notebook(path, timeout=, kernel_name=, cwd=, env=)` | Executes with nbclient and returns the executed notebook. A failing cell raises `NotebookCellError` with its index, source and traceback. |
| `notebook` | `screenshot_notebook(path, out_png, wait_for=, nonblank=, outputs=)` | One call: start JupyterLab, run all cells, wait, screenshot, shut down. |
| `notebook` | `notebook_session(path, env=, mode=, frontend=)` | The same as a context manager that yields a `NotebookSession` for several screenshots, interaction and checks. |

A test that needs more than one screenshot uses the session:

```python
with notebook_session(notebook_path, env={"STEPS": "500"}) as nb:
    nb.wait(nonblank="caliper-view canvas")
    first = nb.page.locator("caliper-view canvas").screenshot()
    nb.page.wait_for_timeout(500)
    second = nb.page.locator("caliper-view canvas").screenshot()
    assert png_difference(first, second) > 0.05
    nb.screenshot(out_dir / "demo.png", outputs=True)
```

`NotebookSession.wait()` fails at once with `NotebookCellError` when a cell raises. `wait(idle=True)` waits until every cell has run. `session.server.root` is the server's root folder, which is also the kernel's working folder, so a test can read files the notebook wrote or create a file the notebook polls for.

Playwright's CSS selectors search open shadow roots, so `"caliper-view canvas"` finds a canvas inside the element's shadow root. Closed shadow roots cannot be searched. JavaScript predicates can call `__caliperHarness.deepQueryAll(selector)`, which searches open shadow roots too. It matches the selector inside each root separately, so `deepQueryAll('canvas')` finds a canvas inside a shadow root, while `deepQueryAll('caliper-view canvas')` does not.

## How a notebook session is isolated

`notebook_session` starts `python -m jupyterlab --no-browser` bound to 127.0.0.1 on a free port, with a random token, terminals off and no news or update checks. The server's root is a fresh folder holding a copy of the notebook with its outputs cleared. The variables `JUPYTER_CONFIG_DIR`, `JUPYTER_DATA_DIR`, `JUPYTER_RUNTIME_DIR`, `JUPYTER_PATH`, `IPYTHONDIR`, `JUPYTERLAB_SETTINGS_DIR` and `JUPYTERLAB_WORKSPACES_DIR` point into the session folder, so the user's own Jupyter configuration is neither read nor written. The one exception is a kernel spec named other than `python3` with no interpreter given, which is looked up in the user's Jupyter folders (read only) and copied into the session folder. The session folder also holds a kernel spec named `python3` that starts the chosen interpreter. The folder of the original notebook is on the kernel's `PYTHONPATH`, so modules beside the notebook import as they would in place. `run_notebook` isolates its kernel the same way and runs it in a scratch folder.

On exit the browser closes, the server is asked to shut down through `POST /api/shutdown` (which shuts down its kernels), and any process that remains is killed. The session folder is removed, except after an error, when its path is added to the exception and `server.log` in it holds the server's log.

## Findings

These were measured on an Apple silicon Mac with Playwright 1.65.2 (Chromium 159 headless shell), JupyterLab 4.6.6, Notebook 7.6.5, ipykernel 7.4.0 and anywidget 0.11.0.

### WebGPU in headless Chromium

WebGPU needs a secure context. A page from `http://127.0.0.1` or from a `file://` URL is one. A page written with Playwright's `page.set_content()` is not, and has no `navigator.gpu` at all.

| Flags | Adapter | Canvas in screenshots |
|---|---|---|
| none | `requestAdapter()` returns null | not applicable |
| `--enable-unsafe-webgpu` | SwiftShader, fallback | black: the WebGPU canvas does not composite |
| `--enable-unsafe-webgpu --enable-features=Vulkan,WebGPU` | SwiftShader, fallback | black |
| `--use-angle=metal` | Apple Metal (`metal-3`), hardware | correct |
| `--enable-unsafe-webgpu --use-angle=metal` | Apple Metal (`metal-3`), hardware | correct |
| `--enable-unsafe-webgpu --use-angle=swiftshader` | SwiftShader, fallback | correct |
| `--enable-unsafe-webgpu --use-angle=swiftshader --use-webgpu-adapter=swiftshader` | SwiftShader, fallback | correct |

The harness has three modes. `metal` is the default (`--enable-unsafe-webgpu --use-angle=metal`), and its adapter is the Mac's GPU. `swiftshader` (`--enable-unsafe-webgpu --use-angle=swiftshader --use-webgpu-adapter=swiftshader`) gives a software adapter. `canvas2d` uses no WebGPU flags, so no adapter is available, and it also removes `navigator.gpu` before any page script runs, because no Chromium flag removes it (`--disable-features=WebGPU`, `--disable-blink-features=WebGPU` and `--disable-gpu` all leave it in place). A viewer in `canvas2d` mode therefore takes its Canvas 2D path.

The Chromium headless shell is enough for WebGPU, so the harness does not use the new headless mode of the full Chromium build. The headless shell has no app bundle; macOS registers it as a `UIElement` process, which has no Dock icon, menu bar or window.

A WebGPU canvas cannot be read back through the page: after a frame is presented, `drawImage()` of the canvas gives transparent black. The harness therefore measures pixels on screenshots, which show exactly what was composited.

### Notebooks and the stream

The self-test in `mock_widget/` shows that a widget in a notebook cell can be fed directly by a WebSocket from the kernel process. `selftest.ipynb` starts a `websockets` server on 127.0.0.1 in a background thread of the kernel, which pushes binary frames at 30 Hz, and displays an anywidget that connects to it and draws each frame on a canvas inside an open shadow root. A later cell then runs a busy loop. The widget keeps drawing during the loop at the full rate (30.0 frames per second measured in both front ends), because its data does not travel through Jupyter's message channel.

- A WebSocket from the JupyterLab page at `http://127.0.0.1:<jupyter port>` to `ws://127.0.0.1:<other port>` works in headless Chromium. Both are loopback and neither is secure, so this is neither mixed content nor blocked by Chromium's local network rules.
- The connection carries `Origin: http://127.0.0.1:<jupyter port>`, the origin of the JupyterLab page, and the same in the Notebook 7 front end. A stream server that checks the Origin must therefore accept the notebook server's origin, not only its own. A notebook server opened as `localhost` gives `http://localhost:<port>`, and other front ends such as VS Code give their own origins; these were not measured.
- JupyterLab's only Content-Security-Policy is `frame-ancestors 'self'; report-uri /api/security/csp-report`. It has no `connect-src` or `default-src`, so it restricts neither sockets nor anywidget's ES modules.
- WebGPU works inside the JupyterLab page in headless Chromium, with the same Metal adapter as on a plain page, both from the page and from inside the widget.
- With ipykernel 7 and a widget on the page, JupyterLab shows the kernel as idle while a cell is running. The harness therefore decides whether cells are running from each cell's execution state, not from the kernel status.
