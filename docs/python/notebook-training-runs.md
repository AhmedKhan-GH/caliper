# Notebook Training Runs

This tutorial trains a model in a Jupyter notebook and shows four live views
of it inside the notebook while it trains: the loss as a series, the attention
weights as a heatmap in which you choose a layer, a sequence and a head, the
model's embedding of an evaluation set as a point cloud coloured by class, and
its predictions for the same evaluation set as a heatmap. A selection in the
embedding marks the same sequences in the predictions. A panel of controls
pauses and steps the training loop, changes its learning rate and saves a
checkpoint while the cell runs. The same views open in a browser tab and in
the Caliper app. The training loop runs at full speed, and Caliper samples it.

The worked example is `examples/notebooks/live_training.ipynb`, a two-layer
transformer that learns to classify synthetic token sequences, with nothing to
download. The excerpts below come from it; run the notebook itself for the full
version.

Prereqs: Apple Silicon, macOS 14 (Sonoma) or later, Python 3.12, PyTorch with MPS, and
JupyterLab or VS Code with its Jupyter extension. [Watch a PyTorch training run
live](../introduction/python-quickstart.md) introduces the view calls in a plain script, and
[Network streaming](network-streaming.md) explains the design
that this page uses.

<!-- SPRINT2-FILL screenshot: the controls panel and the four widgets (loss,
attention, embedding, predictions) in JupyterLab after the loss has dropped,
with a selection in the embedding, taken headless. Place the image here. -->

## Install

Build the wheel as [Build and install the wheel](../introduction/python-quickstart.md#build-and-install-the-wheel)
describes. Then install it with the extra `caliper[notebook]`, which adds
anywidget for the widgets, together with a notebook front end. From the
repository root:

```sh
uv pip install --python .venv/bin/python --reinstall-package caliper "$(ls dist/caliper-*.whl)[notebook]" jupyterlab
.venv/bin/python scripts/build_netio.py
.venv/bin/jupyter lab examples/notebooks/live_training.ipynb
```

The second command builds the native sender of `caliper.serve()`,
`caliper._netio`, in a few seconds and copies it into the installed package.
Run it once, and again after you reinstall the wheel. Without it, the server
uses a slower sender written in Python and says so once when it starts; the
[cost of streaming](network-streaming.md#cost-of-streaming)
compares the two.

The widgets draw with Caliper's own views compiled to WebAssembly. They are
build outputs that the repository does not hold: CI builds them, and a wheel
ships them when they are in `python/caliper/_web/` before it is built. In a
checkout, `python scripts/fetch_wasm.py` downloads CI's build and
`python scripts/build_wasm.py` builds them with Emscripten. Both put them
there and into the installed package, so either one works before or after
you install the wheel, and the module's README, `web/wasm/README.md`, gives
what each needs. Without them the widgets draw with views written in
TypeScript, and the server says so once when it starts.

The server needs nothing beyond Caliper LIB, which
depends on `websockets`. In
VS Code, open the notebook and choose `.venv/bin/python` as its kernel.

The notebook reads one setting from the environment: `CALIPER_NOTEBOOK_STEPS`
sets the number of training steps, 8000 by default. A small value, such as
200, makes a run of a few seconds for a quick check.

## Serve

The first cell imports the packages and starts a server:

```python
import caliper

server = caliper.serve()
```

`serve()` starts a server inside the notebook's kernel, on 127.0.0.1 and a free
port, and prints a link:

```text
http://127.0.0.1:<port>/?token=<token>
```

The token is a random **session token**, made new for each server, and the
server refuses a stream connection that does not carry it. The kernel is now a
**producer**: every view call in it is offered to the server, which sends the
view's snapshots to each viewer that subscribes to it. `server.url`,
`server.port` and `server.token` hold the parts of the link, `server.stats()`
shows the connected viewers and their subscriptions, and `server.close()` stops
the server.

## The model and the data

Each example is a sequence of 24 random tokens. One position holds a marker
token, and the token right after the marker decides the class, one of six. The
six class tokens also appear at random elsewhere in the sequence, so the model
has to find the marker to get the class right. Batches are generated on the
device at every step.

The first eight sequences of every batch are the same fixed probe sequences, so
that the attention heatmap follows the same inputs from step to step.

The model is a transformer with two layers of four heads. It computes attention
explicitly, so each layer returns its attention weights, and the output at the
start token passes through a 3-D embedding layer before the classifier. Its
forward pass returns all three things the notebook shows:

```python
def forward(self, x):
    h = self.tokens(x) + self.positions
    attention = []
    for block in self.blocks:
        h, weights = block(h)
        attention.append(weights)
    z = self.embed(self.norm(h[:, 0]))
    return self.classify(z), z, torch.stack(attention)  # attention: (layer, batch, head, query, key)
```

Because the embedding layer is 3-D, the point cloud is the model's own
embedding, not a projection of it.

## Controls and the evaluation set

The next cell registers what a viewer can change or run while the model
trains, and a fixed evaluation set that two views show:

```python
EVAL = 512
eval_x, eval_y = make_batch(EVAL)
eval_marker = (eval_x == MARKER).int().argmax(1)
evaluation = caliper.items("eval", details={"class": eval_y, "marker at": eval_marker})

lr = caliper.param("lr", 2e-3, min=1e-4, max=1e-1, log=True, help="learning rate")
CHECKPOINT = os.path.join(tempfile.gettempdir(), "caliper-live-training.pt")


@caliper.action("save checkpoint", help="saves the model's weights")
def save_checkpoint():
    torch.save(model.state_dict(), CHECKPOINT)
    return f"saved {CHECKPOINT}"
```

`caliper.param` registers the learning rate as a **parameter**, a value that
the viewers show as a slider, here on a log scale between 0.0001 and 0.1. The
script reads `lr.value`; a viewer's change reaches it at the next step.
`caliper.action` registers a function that a viewer runs with a button, here
one that saves the model's weights. `caliper.items` names the 512 evaluation
sequences an **item space**, `eval`, so that a selection of some of them in one
view marks the same sequences in every view that shows them, and gives each
sequence two details, its class and the position of its marker, which a viewer
shows for the selected sequences. [Control a running script](script-control.md)
explains each call.

## The widgets

The widget cell creates one widget for each view and a panel of controls, and
lays them out with ipywidgets:

```python
from ipywidgets import HBox, VBox

loss_widget = caliper.widget("loss", "series", height=200)
attention_widget = caliper.widget("attention", "heatmap")
embedding_widget = caliper.widget("embedding", "points")
predictions_widget = caliper.widget("predictions", "heatmap", height=160)
attention_widget.layout.flex = embedding_widget.layout.flex = "1 1 0"

VBox([caliper.controls(), loss_widget, HBox([attention_widget, embedding_widget]), predictions_widget])
```

`caliper.widget(name, kind)` returns an anywidget widget that shows the live
view `name` from the server of `serve()`. The kind is optional; without it, the
widget takes the kind from the stream. The widget is created before training
starts and shows the view from its first snapshot. `caliper.controls()` is the
panel of the run's controls: its state with Pause or Resume and Step, the
learning rate and the save action. The header of every view widget shows the
run state and the same Pause and Step buttons.

Each widget reads the server's WebSocket stream directly. The snapshots do not
pass through the notebook's own message channel, so the widgets keep updating
while the training cell runs, and the kernel does no drawing work for them.

## The training loop

The loop is the body of a function, `train(steps)`, so that the notebook can
train again later. Each step starts with `caliper.step()`, the checkpoint of
the controls, and copies the learning rate into the optimizer. The training
step is plain PyTorch, and the Caliper calls follow it:

```python
ATTENTION_DIMS = ("layer", "sequence", "head", "query", "key")
ATTENTION_DEFAULT = {"index": {"layer": 1, "sequence": 0, "head": 0}}

for step in range(steps):
    caliper.step()  # pauses, the new learning rate and actions take effect here
    for group in opt.param_groups:
        group["lr"] = lr.value
    x, y = make_batch(BATCH)
    x[:PROBES] = probe_x  # the same probe sequences at the top of every batch
    y[:PROBES] = probe_y
    logits, z, attention = model(x)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    caliper.series(loss.detach(), "loss")
    caliper.show(attention[:, :PROBES], "attention", cmap="magma",
                 dims=ATTENTION_DIMS, data_view=ATTENTION_DEFAULT)
    if step % 10 == 0:
        with torch.no_grad():
            eval_logits, eval_z, _ = model(eval_x)
        caliper.points(eval_z, eval_y, "embedding", classes=CLASSES, items="eval")
        caliper.show(eval_logits.softmax(1).T, "predictions", dims=("class", "sequence"),
                     items={"sequence": "eval"})
```

| Call | Tensor | Live view |
|---|---|---|
| `series(loss.detach(), "loss")` | 0-d tensor on MPS | A series that receives every value |
| `show(attention[:, :PROBES], "attention", ...)` | `(2, 8, 4, 24, 24)`: layer, sequence, head, query, key | A 24x24 heatmap of the data view the viewer chose |
| `points(eval_z, eval_y, "embedding", ...)` | `(512, 3)` positions and `(512,)` labels of the evaluation set | 3-D points, coloured by class |
| `show(eval_logits.softmax(1).T, "predictions", ...)` | `(6, 512)`: class, sequence | The predicted probability of each class for each evaluation sequence |

`caliper.step()` costs about 22 ns when nothing waits for it. `items="eval"`
says that the points of the embedding are the sequences of the item space
`eval`, and `items={"sequence": "eval"}` that the `sequence` dimension of the
predictions indexes the same sequences.

None of these calls reads a value back to the CPU. `series` takes the loss
tensor itself, not `loss.item()`, so the loop never waits for the GPU. The
notebook calls `loss.item()` once, after the loop, to print the final loss.

`dims` names each dimension of the attention tensor. The names turn the tensor
into a source that a viewer can take a **data view** of: a slice, selection or
reduction that the producer computes on the device before anything is sent.
`data_view` sets the view's default. An integer index selects one position and
drops that dimension, so the default, layer 1, sequence 0 and head 0, leaves a
24x24 heatmap of queries against keys. A few other data views of the same
tensor:

| Data view | Heatmap |
|---|---|
| `{"index": {"layer": 1, "sequence": 0, "head": 0}}` | Head 0 of layer 1 on probe sequence 0, the default |
| `{"index": {"layer": 0, "sequence": 0}, "reduce": {"head": "max"}}` | The largest weight over the four heads of layer 0, on probe sequence 0 |
| `{"index": {"layer": 1, "head": 2}, "reduce": {"sequence": "mean"}}` | Head 2 of layer 1, averaged over the eight probe sequences |

The producer applies the index first, then the reduction, then the budget,
which pools a heatmap or samples points that would exceed the viewer's limit.
The [Caliper LIB](caliper-lib-api.md#data-views) reference lists every field.

## What you see

The loss stays on a plateau above 1 for the first one to a few thousand steps
and then falls to almost zero within a few hundred steps, once the model has
learned to find the marker. At the same time the embedding separates into six clusters,
one for each class colour, and each column of the predictions turns to one
bright class. In layer 1 the row of the start token, row 0, puts
nearly all its weight on one column, the class token after the marker. The
training cell ends by printing the number of steps, the steps per second, the
final loss and the accuracy on the last batch.

## Interaction

Every widget pans, zooms and shows the value under the pointer. The heatmap
also has a picker for its data view, built from the dimensions that `dims`
named and their sizes. Choosing a layer, a sequence or a head in the picker
sends a data view request over the stream to the producer, so the choice takes
effect at the view's next snapshot, while the training cell is still running.
The choice belongs to that widget's subscription: a browser tab on the same
view keeps its own data view, and the producer computes each data view that a
viewer asks for once per snapshot.

Selecting points in the embedding widget, with a Shift drag, selects those
sequences of the evaluation set everywhere: the
predictions widget marks their columns, and a table under the embedding shows
their class and marker position. The selection travels on the stream, so it
works while the training cell runs, and the script knows it at once:
`evaluation.selection` holds the selected sequences.

The selection is also a `select` event. The widget stores the latest event in
its `selection` trait, which Python reads:

```python
print("selection:", embedding_widget.selection)
print("selected sequences:", evaluation.selection[:20])
```

The selection is a dict with the view's name, the event's type and its detail.
For points, the detail holds `indices`, the positions of the selected points in
the snapshot that was shown, `row_ids`, their indices in the source tensor,
and `space` and `items`, the item space and the selected sequences in it. To
run code on every new selection instead, pass a function to
`embedding_widget.on_select()`.

The `data_view` trait works in both directions. The widget writes the data view
the user picked, and setting it from Python makes the widget request that data
view:

```python
attention_widget.data_view = {"index": {"layer": 0, "sequence": 0}, "reduce": {"head": "max"}}
train(max(1, STEPS // 10))
```

In layer 0, at least one head usually learns to make the class token attend to
the marker just before it, and the maximum over the heads shows that pattern
whichever head has learned it.

Widget traits travel over the notebook's message channel, and the kernel reads
that channel only while no cell runs. A selection made during training
therefore reaches `selection` once the training cell has finished, and a data
view set from Python takes effect at the next snapshot, which is why the
notebook trains for a few more seconds after setting it. Events from the
viewer are data for your code to read; nothing in them runs as a command.
Besides a widget's traits, `server.on_event(fn)` calls `fn` with every event of
every viewer, on a thread of its own, also while a cell runs, and
`server.events()` returns the events received since its last call.

## Pause, step and change the learning rate

The controls act on the training cell while it runs. Pause, in the panel or in
the header of any widget, stops the loop at its next `caliper.step()`: the
loss stops growing, and every header reads `paused at` the step it stopped at.
Step runs one more step and pauses again, and Resume lets the loop go on.
Typing a learning rate in the panel's field, or dragging its slider, sends the
value to the script, which clamps it to the range and takes it at its next
step; the panel then shows the value the script took, and the loop copies it
into the optimizer. Save checkpoint runs `save_checkpoint()` at the next step,
on the training thread, and the panel shows what it returned. While the loop
is paused, the learning rate changes and the action runs at once, in the step
where the loop waits.

## Open the stream in a browser tab

The link that `serve()` printed opens the viewer page: every view of the
notebook in a grid, with the same pan, zoom, hover and data view picker as the
widgets, and the same panel of controls above them. A tab and the widgets can watch at the same time. Each viewer
subscribes to the views it shows, and the producer does no work for a view that
no viewer subscribes to.

The server listens only on 127.0.0.1, so the browser that shows the tab or the
notebook must run on the same machine as the kernel. A notebook on a remote
Jupyter server cannot show the widgets.

## Open the views in the Caliper app

To watch the same views in the Caliper app as well, open a live session in the
notebook before training:

```python
caliper.session(script_name="live_training")
```

Every open route receives each view call, so the app, the widgets and the tab
update together. Start the Caliper app and attach to the notebook's kernel with
`F4 Attach to process` on its landing screen, as
[Attach from Caliper](../introduction/python-quickstart.md#attach-from-caliper) describes. Without
`script_name`, the session list shows the kernel's launcher script instead of a
useful name.

## Performance

The view calls behave the same in a notebook as in a script. A call that the
rate limit skips returns after a clock check, and a call that takes a snapshot
queues a few operations on the device and returns at once
([Cost of a call](caliper-lib-api.md#cost-of-a-call)). Each view takes
at most 60 snapshots per second by default. A widget draws the newest snapshot
it has, so most training steps are skipped by design, while the series receives
every value, sent as deltas.

The network route adds a copy of each snapshot from the device into the
socket. Data views keep that copy small: the attention heatmap travels as 24x24
values out of a tensor of 2 x 8 x 4 x 24 x 24, the embedding as 512 points
and the predictions as 6 x 512 values.
Encoding and sending happen outside the training loop, and a viewer that falls
behind receives fewer snapshots rather than slowing training down.

The widgets draw in the browser, with WebGPU where it is available and Canvas
2D otherwise, so drawing costs the kernel nothing.

<!-- SPRINT2-FILL measured overhead: training throughput of this notebook with
the four views served and watched, as a percentage of the same loop without
Caliper calls, on MPS, with the machine and torch version. Place the number
here. -->

## Close

```python
server.close()
```

`server.close()` stops the server and ends the stream to every widget and
browser tab.

## Next

- [Caliper LIB](caliper-lib-api.md#serve) documents `serve`, `widget`
  and the data view arguments, and its [Controls](caliper-lib-api.md#controls)
  the calls of the controls.
- [Control a running script](script-control.md) adds the
  controls to a script of your own.
- [Network streaming](network-streaming.md) explains the
  transport, data views and the security position.
- [Wire protocol](../reference/wire-protocol.md) defines every message on the
  stream.
