"""The demo notebook examples/notebooks/live_training.ipynb, run in JupyterLab
under headless Chromium with the harness in tests/headless: while it trains,
the loss, attention and embedding widgets draw and change; a pan of the
loss stops it following the data and holds while training continues, and a
double click follows again; choosing a layer and a head in the attention
widget's picker changes its data view; the attention heatmap zooms and pans,
and a hover there names the cell under the pointer; the controls of wire
protocol 3.1 work while the cell trains: Pause in the loss widget's header
holds the loop, Step runs it one step, Resume lets it go on, the learning
rate set in the controls panel comes back as the value the script took and
is the optimizer's afterwards, and the save action answers with its result; a
selection in the embedding widget highlights the same sequences in the
predictions widget, shows their details, and reaches
``embedding_widget.selection``; and ``attention_widget.data_view = {...}``
from Python reaches the widget. Run
in the "metal" mode, where Caliper's WebAssembly views draw all three widgets
on WebGPU and make the selection, and the "canvas2d" mode, where the
element's own TypeScript views do. Without the WebAssembly views in the
package (scripts/build_wasm.py and scripts/fetch_wasm.py put them there), the
"metal" mode checks the TypeScript views on WebGPU instead and says so.

    CALIPER_HEADLESS_DIR=<scratch> CALIPER_TEST_ARTIFACTS=<shots> \\
    PYTHONPATH=python .venv/bin/python -m pytest --noconftest \\
        python/tests/test_live_training_notebook.py

``CALIPER_NOTEBOOK_STEPS`` sets the training steps (default here 6000, about
15 s on MPS); screenshots go to ``CALIPER_TEST_ARTIFACTS``."""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "examples" / "notebooks" / "live_training.ipynb"

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="the harness manages POSIX process groups"),
    pytest.mark.skipif(not NOTEBOOK.exists(), reason="the demo notebook is not in this checkout"),
]

STATE = """(name) => { const v = __caliperHarness.deepQueryAll('caliper-view')
                        .find((e) => e.getAttribute('view') === name);
                       return v ? v.debugState() : null; }"""
PANEL = """() => { const p = __caliperHarness.deepQueryAll('caliper-controls')[0];
                   return p ? p.debugState() : null; }"""


def cell_index(text: str) -> int:
    """The index of the first code cell that contains ``text``."""
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    return next(i for i, c in enumerate(cells)
                if c["cell_type"] == "code" and text in "".join(c["source"]))


def changed(a: bytes, b: bytes) -> bool:
    from PIL import Image, ImageChops
    return ImageChops.difference(Image.open(io.BytesIO(a)).convert("RGB"),
                                 Image.open(io.BytesIO(b)).convert("RGB")).getbbox() is not None


def run_cells(nb, index: int, above: bool) -> None:
    """Start running the cells above ``index``, or that cell alone, through
    JupyterLab's commands with that cell selected, and return."""
    command = "notebook:run-all-above" if above else "notebook:run-cell"
    nb.page.evaluate("([name, i, command]) => { const p = __nbPanel(name); const n = p.content;"
                     " jupyterapp.shell.activateById(p.id); n.mode = 'command'; n.deselectAll();"
                     " n.activeCellIndex = i; n.select(n.widgets[i]);"
                     " window.__run = jupyterapp.commands.execute(command); }",
                     [nb.name, index, command])


def page_point(box: dict, d: dict, x: float, y: float) -> tuple[float, float]:
    """Page coordinates of data point (x, y) in a view's drawing area, from its
    debug state's area and window (rows grow down in a heatmap)."""
    a, w = d["area"], d["window"]
    fy = (y - w["y0"]) / (w["y1"] - w["y0"])
    py = a["y"] + fy * a["h"] if d["kind"] in ("heatmap", "image") else a["y"] + a["h"] - fy * a["h"]
    return box["x"] + a["x"] + (x - w["x0"]) / (w["x1"] - w["x0"]) * a["w"], box["y"] + py


def first_state(nb, view: str, pred: str, timeout: float = 30.0) -> dict:
    """The debug state of ``view`` read in the first animation frame in which
    the JavaScript predicate ``pred`` holds for it: the same reading the
    predicate saw, not a later one."""
    handle = nb.page.wait_for_function(
        f"() => {{ const s = ({STATE})({json.dumps(view)}); return s && ({pred})(s) ? s : null; }}",
        timeout=timeout * 1000)
    return handle.json_value()


def loss_values(state: dict) -> int:
    """The number of loss values a loss view holds, from its info line
    ("1801 steps  last 1.208")."""
    return int(state["info"].split()[0])


def settled(nb, state: dict, values: int) -> dict:
    """The loss view's state once it holds ``values`` values. A view drawn by
    the element's own views shares the connection that carries the run state,
    which follows the data on that connection, so it must hold them in the
    same frame. Caliper's WebAssembly views read the stream on a connection of
    their own and draw on their own frame: the producer sends their data
    before the run state too, but nothing orders two connections, so they get
    a bounded moment to show it, and then exactly that many values."""
    if state["drawer"] != "wasm":
        return state
    return first_state(nb, "loss", f"(s) => parseInt(s.info) === {values}", timeout=5.0)


def cell_done(nb, index: int) -> bool:
    state = nb.state()
    cell = next(c for c in state["cells"] if c["index"] == index)
    return cell["count"] is not None and not cell["running"]


@pytest.mark.parametrize("mode", ["metal", "canvas2d"])
def test_live_training_notebook(tmp_path, mode):
    for mod in ("jupyterlab", "ipykernel", "anywidget", "ipywidgets", "websockets", "PIL",
                "torch", "playwright"):
        pytest.importorskip(mod)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tests.headless.notebook import notebook_session
    from tests.headless.screenshot import png_stats
    shots = Path(os.environ.get("CALIPER_TEST_ARTIFACTS") or tmp_path / "shots")
    shots.mkdir(parents=True, exist_ok=True)
    steps = os.environ.get("CALIPER_NOTEBOOK_STEPS", "6000")
    env = {"PYTHONPATH": os.pathsep.join(filter(None, [str(REPO / "python"),
                                                       os.environ.get("PYTHONPATH")])),
           "CALIPER_NOTEBOOK_STEPS": steps, "CALIPER_SESSION": "0"}
    train_cell, select_cell = cell_index("def train("), cell_index("embedding_widget.selection")
    close_cell = cell_index("server.close()")

    with notebook_session(NOTEBOOK, env=env, mode=mode, run_all=False,
                          viewport=(1280, 1100), timeout=180) as nb:
        run_cells(nb, select_cell, above=True)            # everything up to and with training
        names = ("loss", "attention", "embedding")
        nb.wait("() => " + " && ".join(
            f"(({STATE})('{n}') || {{}}).status === 'connected'" for n in names), timeout=150)
        assert not cell_done(nb, train_cell), "training ended before the checks; raise the steps"
        renderers = {n: nb.page.evaluate(STATE, n)["renderer"] for n in names}
        from caliper import _assets
        views = mode == "metal" and not _assets.missing_views()
        if mode == "canvas2d":
            assert set(renderers.values()) == {"canvas2d"}, renderers
        elif views:
            # The WebAssembly views draw every kind the demo shows.
            assert set(renderers.values()) == {"wasm"}, renderers
        else:
            print(f"the WebAssembly views are not in {_assets.asset_dir()}: the metal mode checks "
                  "the TypeScript views on WebGPU")
            # The TypeScript views: the heatmap and the points on WebGPU, the series with Canvas 2D.
            assert renderers == {"loss": "canvas2d", "attention": "webgpu", "embedding": "webgpu"}, renderers

        # Each widget draws, and updates while the cell trains. The embedding
        # barely moves during the loss plateau, so for it the view's own count
        # shows the updates: the snapshots the WebAssembly views applied, or
        # the frames the element's points view drew.
        for n in names:
            area = nb.page.locator(f'caliper-view[view="{n}"] canvas:visible').first
            before = nb.page.evaluate(STATE, n)
            first = area.screenshot()
            nb.page.wait_for_timeout(1500)
            second = area.screenshot()
            after = nb.page.evaluate(STATE, n)
            (shots / f"notebook-{mode}-{n}.png").write_bytes(second)
            least = 0.001 if n == "embedding" else 0.005          # 256 small points
            assert png_stats(first).nonblank(least), (n, png_stats(first).to_dict())
            if n == "embedding":
                key = "updates" if views else "frames"
                assert after[key] > before[key], (before, after)
            else:
                assert changed(first, second), f"{n} did not change while training"
        nb.screenshot(shots / f"notebook-{mode}-training.png")

        # A pan of the loss stops it following the data: its window holds while
        # training goes on, and a double click follows the newest steps again.
        loss = nb.page.locator('caliper-view[view="loss"] canvas:visible').first
        loss.scroll_into_view_if_needed()
        lbox = loss.bounding_box()
        assert nb.page.evaluate(STATE, "loss")["follow"] is True
        lx, ly = lbox["x"] + lbox["width"] * 0.6, lbox["y"] + lbox["height"] * 0.5
        nb.page.mouse.move(lx, ly)
        nb.page.mouse.down()
        nb.page.mouse.move(lx + 80, ly, steps=5)
        nb.page.mouse.up()
        nb.wait(f"() => (({STATE})('loss') || {{}}).follow === false", timeout=10)
        nb.page.wait_for_timeout(300)
        held = nb.page.evaluate(STATE, "loss")
        nb.page.wait_for_timeout(1500)
        later = nb.page.evaluate(STATE, "loss")
        assert not cell_done(nb, train_cell), "training ended before the loss check; raise the steps"
        assert later["window"] == held["window"], (held["window"], later["window"])
        assert later["info"] != held["info"], "the loss got no new steps while it was held"
        nb.page.locator('caliper-view[view="loss"]').screenshot(path=str(shots / f"notebook-{mode}-loss-held.png"))
        nb.page.mouse.dblclick(lx, ly)
        nb.wait(f"() => {{ const s = ({STATE})('loss'); return !!s && s.follow === true"
                f" && s.window.x1 > {held['window']['x1']}; }}", timeout=10)

        # The picker: layer 0, head 2 (sequence 0); the heatmap's data view follows.
        att = nb.page.locator('caliper-view[view="attention"]')
        before = att.locator("canvas:visible").first.screenshot()
        att.locator('select[data-dim="layer"]').select_option("i:0")
        att.locator('select[data-dim="head"]').select_option("i:2")
        nb.wait(f"() => {{ const s = ({STATE})('attention'); const i = s && s.dataView.index;"
                " return !!i && i.layer === 0 && i.sequence === 0 && i.head === 2; }", timeout=30)
        nb.page.wait_for_timeout(300)
        after = att.locator("canvas:visible").first.screenshot()
        assert changed(before, after)
        att.screenshot(path=str(shots / f"notebook-{mode}-attention-layer0-head2.png"))

        # The heatmap zooms about the pointer and pans with a drag; a hover then
        # names the cell under the pointer; a double click fits it again.
        heat = att.locator("canvas:visible").first
        heat.scroll_into_view_if_needed()
        hbox = heat.bounding_box()
        fit = nb.page.evaluate(STATE, "attention")["window"]
        hx, hy = hbox["x"] + hbox["width"] * 0.5, hbox["y"] + hbox["height"] * 0.5
        nb.page.mouse.move(hx, hy)
        nb.page.mouse.wheel(0, -400)
        span = (fit["x1"] - fit["x0"]) * 0.8
        nb.wait(f"() => {{ const w = ({STATE})('attention').window; return w.x1 - w.x0 < {span}; }}", timeout=10)
        zoomed = nb.page.evaluate(STATE, "attention")["window"]
        nb.page.mouse.down()
        nb.page.mouse.move(hx + 60, hy + 30, steps=5)
        nb.page.mouse.up()
        nb.wait(f"() => ({STATE})('attention').window.x0 < {zoomed['x0']}", timeout=10)
        d = nb.page.evaluate(STATE, "attention")
        w = d["window"]
        col = int((w["x0"] + w["x1"]) / 2)
        row = int((w["y0"] + w["y1"]) / 2)
        px, py = page_point(hbox, d, col + 0.5, row + 0.5)
        nb.page.mouse.move(px, py)
        nb.page.mouse.move(px + 1, py)
        nb.wait(f"() => {{ const h = ({STATE})('attention').hover; return !!h && h.r === {row} && h.c === {col}; }}",
                timeout=10)
        att.screenshot(path=str(shots / f"notebook-{mode}-attention-zoomed.png"))
        nb.page.mouse.dblclick(px, py)
        nb.wait(f"() => {{ const w = ({STATE})('attention').window; return Math.abs(w.x0 - {fit['x0']}) < 1e-6"
                f" && Math.abs(w.x1 - {fit['x1']}) < 1e-6 && Math.abs(w.y0 - {fit['y0']}) < 1e-6; }}", timeout=10)

        # The controls (3.1), while the cell trains. Pause in the loss widget's
        # header holds the loop at its next caliper.step(), and the run state
        # follows the data of that step: in the very frame in which the header
        # first says "paused at N", the loss already shows its N - 1 values
        # (the loop calls caliper.step() before each step's work), and nothing
        # changes while it stays paused. Step runs one step, whose value is
        # there when the header says "paused at N + 1"; Resume lets it go on.
        assert not cell_done(nb, train_cell), "training ended before the controls; raise the steps"
        loss_view = nb.page.locator('caliper-view[view="loss"]')
        toggle, step = loss_view.locator("button.run-toggle"), loss_view.locator("button.run-step")
        nb.wait(f"() => (({STATE})('loss') || {{}}).run === 'running'", timeout=30)
        assert toggle.inner_text() == "Pause" and toggle.is_enabled()
        toggle.click()
        paused = first_state(nb, "loss", "(s) => /^paused at \\d+$/.test(s.run || '')")
        at = int(paused["run"].rsplit(" ", 1)[1])
        paused = settled(nb, paused, at - 1)
        assert loss_values(paused) == at - 1, paused
        nb.page.wait_for_timeout(1200)
        still = nb.page.evaluate(STATE, "loss")
        assert still["run"] == paused["run"] and still["info"] == paused["info"], (paused, still)
        assert toggle.inner_text() == "Resume"
        loss_view.screenshot(path=str(shots / f"notebook-{mode}-loss-paused.png"))
        step.click()
        stepped = first_state(nb, "loss", f"(s) => s.run === 'paused at {at + 1}'")
        stepped = settled(nb, stepped, at)
        assert loss_values(stepped) == at, stepped
        nb.page.wait_for_timeout(500)
        still = nb.page.evaluate(STATE, "loss")
        assert still["run"] == stepped["run"] and still["info"] == stepped["info"], (stepped, still)
        # The learning rate in the controls panel, set while the run is paused:
        # the panel shows the value the script took, at the checkpoint it waits in.
        panel = nb.page.locator("caliper-controls")
        field = panel.locator('input[aria-label="lr"]')
        field.fill("0.004")
        field.dispatch_event("change")
        nb.wait(f"() => {{ const c = ({PANEL})().controls; const p = c && c.params.find((x) => x.name === 'lr');"
                " return !!p && p.value === 0.004 && p.pending === undefined; }", timeout=30)
        # The save action runs at the checkpoint and answers with its result.
        panel.locator('button[data-action="save checkpoint"]').click()
        nb.wait(f"() => {{ const r = ({PANEL})().results['save checkpoint'];"
                " return !!r && /^saved /.test(r.text || ''); }", timeout=30)
        panel.screenshot(path=str(shots / f"notebook-{mode}-controls.png"))
        toggle.click()
        nb.wait(f"() => (({STATE})('loss') || {{}}).run === 'running'", timeout=30)
        resumed = nb.page.evaluate(STATE, "loss")
        nb.wait(f"() => (({STATE})('loss') || {{}}).info !== {json.dumps(resumed['info'])}", timeout=30)

        # A rectangle selection in the embedding (shift + alt drag), after a
        # double click fits the view to the points, which move while it trains.
        # The visible canvas takes the events: the WebAssembly views' own, or
        # the overlay above the element's points view.
        emb = nb.page.locator('caliper-view[view="embedding"] canvas:visible').first
        emb.scroll_into_view_if_needed()
        box = emb.bounding_box()
        nb.page.mouse.dblclick(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        nb.page.wait_for_timeout(300)
        x0, y0 = box["x"] + box["width"] * 0.03, box["y"] + box["height"] * 0.03
        x1, y1 = box["x"] + box["width"] * 0.97, box["y"] + box["height"] * 0.97
        nb.page.mouse.move(x0, y0)
        nb.page.keyboard.down("Shift")
        nb.page.keyboard.down("Alt")
        nb.page.mouse.down()
        for k in range(1, 6):
            nb.page.mouse.move(x0 + (x1 - x0) * k / 5, y0 + (y1 - y0) * k / 5)
        nb.page.mouse.up()
        nb.page.keyboard.up("Alt")
        nb.page.keyboard.up("Shift")
        nb.wait(f"() => (({STATE})('embedding') || {{}}).selected > 0", timeout=20)
        assert nb.page.evaluate(STATE, "embedding")["renderer"] == renderers["embedding"]
        nb.page.wait_for_timeout(300)
        nb.page.locator('caliper-view[view="embedding"]').screenshot(
            path=str(shots / f"notebook-{mode}-embedding-selection.png"))
        # Linked selection: a second rectangle over the left part of the
        # embedding selects some of its sequences; the predictions widget
        # marks the same sequences of the item space "eval", and the embedding
        # shows their details.
        x2 = box["x"] + box["width"] * 0.45
        nb.page.mouse.move(x0, y0)
        nb.page.keyboard.down("Shift")
        nb.page.keyboard.down("Alt")
        nb.page.mouse.down()
        for k in range(1, 6):
            nb.page.mouse.move(x0 + (x2 - x0) * k / 5, y0 + (y1 - y0) * k / 5)
        nb.page.mouse.up()
        nb.page.keyboard.up("Alt")
        nb.page.keyboard.up("Shift")
        nb.wait(f"() => {{ const l = (({STATE})('predictions') || {{}}).linked;"
                " return !!l && l.space === 'eval' && l.items.length > 0 && l.items.length < 512; }",
                timeout=20)
        nb.page.wait_for_timeout(300)
        nb.wait(f"() => /class/.test((({STATE})('embedding') || {{}}).details || '')", timeout=20)
        linked = nb.page.evaluate(STATE, "predictions")["linked"]
        details = nb.page.evaluate(STATE, "embedding")["details"]
        assert "marker at" in details, details
        nb.page.locator('caliper-view[view="predictions"]').screenshot(
            path=str(shots / f"notebook-{mode}-predictions-linked.png"))
        nb.page.locator('caliper-view[view="embedding"]').screenshot(
            path=str(shots / f"notebook-{mode}-embedding-details.png"))

        # The training cell ends; the selection is in Python.
        nb.wait(lambda page: cell_done(nb, train_cell), timeout=240)
        sel = json.loads(nb.execute("import json; print(json.dumps(embedding_widget.selection))"))
        assert sel["view"] == "embedding" and sel["type"] == "select", sel
        assert sel["detail"]["indices"] and len(sel["detail"]["row_ids"]) == len(sel["detail"]["indices"])
        assert sel["detail"]["space"] == "eval" and sel["detail"]["items"], sel["detail"].keys()
        # The script took the learning rate, used it, and saved once; it knows
        # the selected sequences too.
        got = json.loads(nb.execute(
            "import json; print(json.dumps([lr.value, opt.param_groups[0]['lr'], os.path.exists(CHECKPOINT),"
            " evaluation.selection[:5]]))"))
        assert got[0] == got[1] == 0.004 and got[2] is True, got
        assert got[3] == linked["items"][:5], (got[3], linked["items"][:5])

        # The next cell sets the attention widget's data view from Python and trains again.
        run_cells(nb, select_cell, above=False)
        nb.wait(f"() => {{ const s = ({STATE})('attention'); const d = s && s.dataView;"
                " return !!d && d.reduce && d.reduce.head === 'max' && d.index.layer === 0; }",
                timeout=60)
        assert att.locator('select[data-dim="head"]').input_value() == "r:max"
        nb.wait(lambda page: cell_done(nb, select_cell), timeout=240)
        out = nb.page.locator(".jp-Cell").nth(select_cell).locator(".jp-OutputArea-output").first
        assert "selection: {" in out.inner_text()
        nb.screenshot(shots / f"notebook-{mode}-after.png", outputs=True)

        # Closing the server ends every widget's stream.
        run_cells(nb, close_cell, above=False)
        nb.wait("() => " + " && ".join(
            f"(({STATE})('{n}') || {{}}).status === 'ended'" for n in names), timeout=60)
