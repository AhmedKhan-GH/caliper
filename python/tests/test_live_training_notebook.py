"""The demo notebook examples/notebooks/live_training.ipynb, run in JupyterLab
under headless Chromium with the harness in tests/headless: while it trains,
the loss, attention and embedding widgets draw and change; choosing a layer
and a head in the attention widget's picker changes its data view; a
selection in the embedding widget reaches ``embedding_widget.selection``; and
``attention_widget.data_view = {...}`` from Python reaches the widget. Run
in the "metal" mode, where Caliper's WebAssembly views draw all three widgets
on WebGPU and make the selection, and the "canvas2d" mode, where the
element's own TypeScript views do.

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
        if mode == "canvas2d":
            assert set(renderers.values()) == {"canvas2d"}, renderers
        else:
            # The WebAssembly views draw every kind the demo shows.
            assert set(renderers.values()) == {"wasm"}, renderers

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
                key = "updates" if mode == "metal" else "frames"
                assert after[key] > before[key], (before, after)
            else:
                assert changed(first, second), f"{n} did not change while training"
        nb.screenshot(shots / f"notebook-{mode}-training.png")

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

        # The training cell ends; the selection is in Python.
        nb.wait(lambda page: cell_done(nb, train_cell), timeout=240)
        sel = json.loads(nb.execute("import json; print(json.dumps(embedding_widget.selection))"))
        assert sel["view"] == "embedding" and sel["type"] == "select", sel
        assert sel["detail"]["indices"] and len(sel["detail"]["row_ids"]) == len(sel["detail"]["indices"])

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
