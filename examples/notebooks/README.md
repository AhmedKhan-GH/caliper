# Notebooks

`live_training.ipynb` trains a small transformer on synthetic data and shows
three live views in the notebook while it trains: the loss as a series, the
attention weights as a heatmap with a choice of layer, sequence and head, and
the model's 3-D embedding as points coloured by class. The link that
`caliper.serve()` prints opens the same views in a browser tab.
[Watch a training run in a notebook](../../docs/python/notebook-training-runs.md)
walks through it cell by cell.

## Run it

The notebook needs an Apple Silicon Mac, Python 3.12 and PyTorch with MPS. On
a machine without MPS it trains on the CPU, more slowly. Build the wheel as
[Watch a PyTorch training run live](../../docs/introduction/python-quickstart.md#wheel-installation)
describes, then install it with the extra `caliper[notebook]` and a notebook
front end, build the native sender once, and open the notebook, from the
repository root:

```sh
uv pip install --python .venv/bin/python --reinstall-package caliper "$(ls dist/caliper-*.whl)[notebook]" jupyterlab
.venv/bin/python scripts/build_netio.py
.venv/bin/jupyter lab examples/notebooks/live_training.ipynb
```

`scripts/build_netio.py` compiles `caliper._netio`, the native sender of
`caliper.serve()`, in a few seconds and copies it into the installed package.
Without it, `serve()` uses a slower sender written in Python and says so once.
Reinstalling the wheel removes the module, so run the script again after a
reinstall.

The widgets draw with Caliper's own views compiled to WebAssembly, which the
repository does not hold. `python scripts/fetch_wasm.py` downloads CI's build
and `python scripts/build_wasm.py` builds them with Emscripten; both copy them
into `python/caliper/_web/`, from where a wheel built afterwards ships them,
and into the installed package. Without them the widgets draw with views
written in TypeScript, and `caliper.serve()` says so once.

Run the cells from top to bottom. In VS Code, open the notebook and choose the
`.venv` interpreter as its kernel.

The default run is 8000 steps. The environment variable
`CALIPER_NOTEBOOK_STEPS` shortens it, which is how to run the notebook headless
as a test:

```sh
CALIPER_NOTEBOOK_STEPS=200 .venv/bin/jupyter execute examples/notebooks/live_training.ipynb
```

`jupyter execute` runs every cell and leaves the file unchanged.

## Changing the notebook

The notebook is kept in the repository without outputs or execution counts, and
its metadata names only the `python3` kernel. Clear all outputs before you
commit a change to it.
