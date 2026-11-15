# Notebooks

`live_training.ipynb` trains a small transformer on synthetic data and shows
three live views in the notebook while it trains: the loss as a series, the
attention weights as a heatmap with a choice of layer, sequence and head, and
the model's 3-D embedding as points coloured by class. The link that
`caliper.serve()` prints opens the same views in a browser tab.
[Watch a training run in a notebook](../../docs/tutorials/watch-a-training-run-in-a-notebook.md)
walks through it cell by cell.

## Run it

The notebook needs an Apple Silicon Mac, Python 3.12 and PyTorch with MPS. On
a machine without MPS it trains on the CPU, more slowly. Build the wheel as
[Watch a PyTorch training run live](../../docs/tutorials/live-pytorch.md#build-and-install-the-wheel)
describes, then install it with the extra `caliper[notebook]` and a notebook
front end, from the repository root:

```sh
uv pip install --python .venv/bin/python --reinstall-package caliper "$(ls dist/caliper-*.whl)[notebook]" jupyterlab
.venv/bin/jupyter lab examples/notebooks/live_training.ipynb
```

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
