# Control a running script

A training loop that streams its views with `caliper.serve()` can also take controls from its viewers while it trains: a viewer pauses the loop, runs it one step at a time, changes a learning rate, saves a checkpoint, and selects samples in one view to see them highlighted in the others. This guide adds each of these to a loop with Caliper LIB, the `caliper` Python package. The calls are listed in [Caliper LIB](../references/python-api.md#controls), and the messages that carry them are [wire protocol 3.1](../references/wire-protocol.md#controls).

Controls travel on the same stream as the views, so they reach the script while a notebook cell runs. A notebook's own widget messages cannot, because the kernel reads them only between cells.

## Add a checkpoint

Every control acts at one place in the loop: `caliper.step()`, the checkpoint. Call it once per step, on the training thread:

```python
import caliper

server = caliper.serve()

for step, (x, y) in enumerate(loader):
    caliper.step(step)
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    caliper.series(loss.detach(), "loss")
```

`caliper.step()` is where parameter values change, where actions run and where a paused run waits, so the script never sees a change in the middle of a step. When nothing waits for it, the call only counts the step, which costs about 22 ns. The number you pass is the step that viewers show when the run pauses; without it, `caliper.step()` counts its calls.

A script that never calls `caliper.step()` can still change parameters, which then take effect at once, but viewers cannot pause it, and its actions wait.

## Change a parameter while it trains

`caliper.param()` registers a value that viewers can change. Its type comes from the first value: a float is a number, an int an integer, a bool a toggle, and a str with `choices` a menu. Read `.value` in the loop:

```python
lr = caliper.param("lr", 2e-3, min=1e-5, max=1e-1, log=True, help="learning rate")
augment = caliper.param("augment", True)

for step, (x, y) in enumerate(loader):
    caliper.step(step)
    for group in opt.param_groups:
        group["lr"] = lr.value
    if augment.value:
        x = jitter(x)
    ...
```

A viewer shows `lr` as a slider on a log scale between `min` and `max`. The script checks every value a viewer sends and clamps it to the range, and the new value reaches `lr.value` at the next `caliper.step()`, so a whole step always uses one value. Every viewer then shows the value that the script took. To change a value from the script, call `lr.set(1e-3)`.

## Run an action

`caliper.action()` registers a function without arguments that a viewer can run with a button:

```python
@caliper.action("save checkpoint", help="saves the model's weights")
def save_checkpoint():
    torch.save(model.state_dict(), "checkpoint.pt")
    return "saved checkpoint.pt"
```

A viewer's request runs the function at the next `caliper.step()`, on the training thread, so it sees the model between two steps. The text it returns goes back to the viewer that asked, and every viewer sees that the action is queued or running. An exception goes back to the viewer with its message and appears in the script as a warning; the loop goes on. Requests run in the order they arrive, and a second request for an action that already waits joins the first. The viewer can only ask for an action by its name: it sends no code and no arguments.

## Pause, resume and step

A viewer pauses the run with Pause in the header of a notebook widget, or in the controls panel of a notebook or of the viewer page. The loop stops in its next `caliper.step()` and waits there. The views show the step at which it paused, and the controls keep working: values change and actions run during the pause. Resume continues the loop, and Step runs it one step, or a given number of steps, and pauses it again. Every header shows the run state, such as `paused at 1897`.

The script can pause itself too, for example when its loss stops being finite:

```python
if not torch.isfinite(loss):
    caliper.pause()
```

A paused run does not wait forever. It continues when the server closes, or when no viewer that may control it has been connected for 10 seconds, which leaves time to reload a page. Ctrl-C, or a notebook's interrupt, ends the wait at once.

## Link selections between views

When several views show the same items, such as the samples of an evaluation set, a selection in one of them can highlight the same samples in the others. Name the item space of each view with `items=`:

```python
eval_x, eval_y = make_eval_set(512)

with torch.no_grad():
    logits, z = model(eval_x)
caliper.points(z, eval_y, "embedding", classes=10, items="eval")
caliper.show(logits.softmax(1), "predictions", items="eval")
```

The points of `embedding` and the rows of `predictions` are both samples of the item space `eval`, numbered from 0. Select a cluster of points in the embedding, and the predictions heatmap marks the rows of the same samples. For `show` the name applies to the rows, for `points` to the points, and for `series` to its index; with `dims`, `items={"sample": "eval"}` names a dimension instead.

`caliper.items()` adds details that a viewer shows in a small table under the view in which the selection was made:

```python
caliper.items("eval", details={"class": eval_y, "marker at": marker_positions})
```

A table of columns is copied once, when you call `caliper.items()`. A function `details=lambda ids: {...}` can compute them instead, but it runs on a thread of its own, never on the training thread, so it must read only data that the loop does not change in place. `caliper.items("eval").selection` holds the samples that are selected now, even while the training cell runs.

## Show the controls in a notebook

`caliper.controls()` is a notebook widget with the run controls, every parameter and every action:

```python
from ipywidgets import VBox

VBox([caliper.controls(), caliper.widget("loss"), caliper.widget("embedding")])
```

The viewer page that the link of `caliper.serve()` opens shows the same panel above its views. The demo notebook `examples/notebooks/live_training.ipynb` uses every control in this guide.

## Who may control

Every viewer that holds the session token of `caliper.serve()` may control the script. The token is in the link that `caliper.serve()` prints, and the server listens on 127.0.0.1 only. To let viewers watch without changing anything, start the server with `caliper.serve(control=False)`: the viewers still show the run state, the parameters and the actions, with every control disabled. Controls cannot run code that the script did not register: a viewer can only set a registered parameter to a value that the script checks, and run a registered action by its name.

A script that streams to the hub of Caliper CLI with `caliper.connect()` takes the same controls, and the hub, `caliper serve`, forwards them to it only when it was started with `--allow-control`; [Run a hub](run-a-hub.md#controls-and-selections) explains that option.
