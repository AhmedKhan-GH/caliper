# Build a dashboard from a live training run

This walk-through builds a dashboard in Caliper from a real training run, in six steps, with every way the app offers
to build one. A training script trains a small transformer on your Mac's GPU and streams its loss and its attention to
the app. You then build a panel by hand, change it with the widget agent (⌘K), ask the side chat for a dashboard of
the attention entropy per layer while the model trains, let an outside agent add a panel through `caliper-mcp`, and
save the dashboard, open it again and open it in a browser.

| Step | What you do | Dry run |
|---|---|---|
| 1 | Train a model and attach the app to its stream | `dry_run/step1_train_and_attach.py` |
| 2 | Drop the data files and build a panel by hand | |
| 3 | Change a panel with ⌘K, accept it and undo it | |
| 4 | Ask the side chat for the attention entropy per layer | |
| 5 | Let an outside agent add a panel through `caliper-mcp` | |
| 6 | Save, open again and open in a browser | |

The files in this folder:

| File | What it is |
|---|---|
| `train.py` | The training script of step 1 |
| `make_data.py` | Writes the two data files of step 2 |
| `data/run_metrics.parquet` | A table of a training run's `step`, `loss`, `accuracy` and `lr`, for step 2 |
| `data/attention.safetensors` | One attention tensor `attn`, layers × heads × T × T, for step 2 |
| `dry_run/` | One script per step that runs it in the hidden app and writes captures |

## What you need

You need a Mac with Apple silicon and macOS 14 or later, and the repository built as below. Everything runs on your
machine: the model of the widget agent and of the side chat is a local model served by Ollama, and nothing needs an
account or a key.

### The app

Build the app from the root of the repository, as its README describes:

```sh
cmake --preset release
cmake --build --preset release
```

This builds `cmake-build-release/caliper` and `cmake-build-release/src/builder/mcp/caliper-mcp`, the MCP server that
the side chat and outside agents use.

### A Python for the training script

The training script needs a Python with PyTorch, which uses the GPU through MPS, and Caliper LIB, the `caliper`
package. Build and install the package's wheel into that Python as
[Watch a PyTorch training run live](../../docs/tutorials/live-pytorch.md#build-and-install-the-wheel) describes, then
build the package's native network sender with the same Python:

```sh
.venv/bin/python scripts/build_netio.py
```

From a checkout, `PYTHONPATH=python` instead of the wheel is enough for this script, since it only streams with
`caliper.serve()`; `scripts/build_netio.py` then puts the sender into `python/caliper/`. Without the sender the script
still works, sending from Python, and says so when it starts.

### The widget agent's Python and the local model

The widget agent of ⌘K is `caliper-agent`, a Python program that the app starts. Give it a Python of its own and tell
the app where it is:

```sh
uv venv .venv-agent
uv pip install --python .venv-agent/bin/python ./src/builder/agent
export CALIPER_AGENT_PYTHON=$PWD/.venv-agent/bin/python
```

Install Ollama and download the model that both agents use:

```sh
ollama pull qwen3.8:27b
```

The model takes about 20 GB of memory while it is loaded, beside the training run, which shares the same GPU and
memory.

> TODO(orchestrator): the context length that the Ollama preset caps the model at, 8,192 or 16,384 tokens, in one
> sentence.

In the app, open the widget agent's settings with ⌘, (or **⌘, Agent** on the landing screen), choose the **Ollama**
preset, whose address is `http://localhost:11434/v1` (`http://127.0.0.1:11434/v1` works the same), type
`qwen3.8:27b` under **Model**, and choose **Probe** to check that the model calls tools.
[Set up the widget agent](../../docs/guides/set-up-the-widget-agent.md) explains each setting.

### The side chat's agent

The side chat runs an agent that speaks the Agent Client Protocol (ACP). This walk-through uses `caliper-agent` on the
same local model, added as a custom command. In the side chat's agent picker, choose **Custom command**, name it
`caliper-agent (Ollama)` and give it this command line, with the absolute path of the Python above:

> Not yet confirmed: the exact command line, and how this agent reaches the session's commands.

```text
/absolute/path/to/.venv-agent/bin/python -m caliper_agent --preset ollama --model qwen3.8:27b
```

### Starting the app

Start the app with Caliper API on, which the widget agent needs:

```sh
cmake-build-release/caliper --api
```

It prints two lines: the API's address with your session token, and the link of the spec page in a browser. You need
neither later, since step 6 opens the dashboard in your browser from the app.

```text
[api] Caliper API on http://127.0.0.1:7341/v0 with the session token <token> (agents read <API file>, whose commands are reviewed)
[api] Spec applets in a browser at http://127.0.0.1:7341/v0/spec?token=<viewer token>&applet=<id>
```

## 1. Train a model and attach the app

In a terminal, start the training script with the Python that has PyTorch and Caliper LIB:

```sh
.venv/bin/python examples/build_a_dashboard/train.py
```

It trains for five minutes; `--seconds 1200` gives it twenty, so that it still trains when you reach step 4, and
`--cpu` trains on the CPU instead of the GPU. The model is a transformer of 4 layers with 4 heads each. Each of its
sequences is 24 random tokens followed by the same 24 tokens again, so the second half can be predicted by looking
back, and the model learns to do that; the data is made by the script, so nothing is downloaded.

The script first prints its stream link, a line such as

```text
http://127.0.0.1:53634/?token=<token>
```

and then a line every 10 seconds with the step, the loss and the entropy of each layer's attention, in nats. A note
that the WebAssembly views are not built concerns browsers only, and the app does not need them.

In Caliper, on the landing screen, press F4, or click **F4 Attach to process** in the Projects pane. The script is
not in the list of sessions, because it streams with `caliper.serve()` rather than through a session file; click the
field under the list, **or a stream link: http://127.0.0.1:&lt;port&gt;/?token=...**, paste the link, and press Enter
or click **Attach link**. You can also open the app on the stream directly, with the link on the command line:

```sh
cmake-build-release/caliper --api --url "http://127.0.0.1:53634/?token=<token>"
```

The live page opens. Its bar reads **esc Home**, `train.py` and a diamond with **live**, with **⌘L Chat** at the
right. Under it, the run bar says **running** with **Pause** and **Step**, which hold the training loop at
its next step and run one step at a time. Seven views fill the page:

| View | What it shows |
|---|---|
| `loss` | The training loss of every step, from the first step on, even if you attached later |
| `entropy.l0` to `entropy.l3` | Each layer's attention entropy, averaged over its heads, at every step |
| `attn` | The attention of the batch's first sequence, the mean over the heads, one map per layer in a 2 × 2 grid, with menus to choose a layer, a head, a reduction or a range |
| `attn.entropy` | The entropy of each head, layers as rows and heads as columns |

The status bar shows the GPU, the script's pid, `7 views` and the frame rate. On an M-series Mac with nothing else on
the GPU, the loss falls slowly from 4.3 to about 3.5 over the first 2,500 steps, then drops to near 0 by about step
3,500, some 15 to 20 seconds after the start. At the same time the entropy of the deeper layers falls, layer 1 to
about 2.4 nats, layer 2 to about 1.6 and layer 3 below 0.8, while layer 0 stays near 2.9: the heads of the deeper
layers learn to look at one earlier token each. In `attn`, the lower two maps, of layers 2 and 3, turn into thin
diagonal lines as that happens.

The app's catalog now lists the stream's views as live sources, which the side chat finds in step 4: `live:loss`,
`live:attn` (shape `[4, 47, 47]`, one map per layer), `live:attn.entropy` (`[4, 4]`) and `live:entropy.l0` to
`live:entropy.l3`, each with its rate. A heatmap on the stream carries at most three dimensions and a dashboard's spec
cannot reshape its data, which is why the script computes the entropy per head itself.

Home detaches the app from the stream; the script keeps training, and the link works again while it runs. Step 2
starts from the landing screen, so press **esc Home** now.

The dry run of this step starts the script and the app hidden on the same app-data folder, presses F4 and Enter on
the listed stream, checks the catalog's live sources over Caliper API, goes Home and checks that the app stays
attached, writes captures of the session list and the live page and the catalog's live entries to
`build/build_a_dashboard/` (`--out` changes it), and prints one line per check:

```sh
python examples/build_a_dashboard/dry_run/step1_train_and_attach.py --train-python .venv/bin/python --size 1440x900
```

## 2. Drop the data files and build a panel by hand

Not yet verified by a dry run.

Write the two data files once, with any Python that has NumPy and PyArrow:

```sh
python examples/build_a_dashboard/make_data.py
```

On the landing screen, drop `data/run_metrics.parquet` and `data/attention.safetensors` from the Finder onto the
window. Caliper imports them into the workspace and opens a new Untitled dashboard in build mode: Sources on the left
lists the table under Tables, with its rows and columns, and the tensor under Tensors, with its dtype and dimensions;
Primitives is under it, the canvas in the middle and Properties on the right.

Select the table in Sources and click **Series** in Primitives. A series panel appears on the canvas, with `step` as
x and the other numeric columns as lines. Drag a source onto the canvas instead to place the panel where you drop it,
and drag a panel's title strip onto another panel's edge to place it beside that panel. Press ⌘T **Text** to show the
spec text under the canvas: it shows the new panel at once, and every change you make on the canvas or in Properties
shows there as you make it. Each gesture is one undo step, with ⌘Z and ⇧⌘Z.

## 3. Change a panel with ⌘K, accept it and undo it

Not yet verified by a dry run.

Click the series panel to focus it; its frame turns bronze. Press ⌘K: a one-line prompt opens at the top of the
panel, with the model, `qwen3.8:27b`, at its end. Type a change, such as "Use a log scale for the y axis", and press
Enter. While the agent works, the prompt row shows its latest step. When it has finished, the panel is drawn with the
change and **PREVIEW** in its title strip, with the agent's reply and the changed keys with their old and new values.
Nothing has changed yet: press Enter or choose **⏎ Accept** to apply it as one undo step, and the spec text shows it.
Press ⌘Z to undo it, and the panel returns to what it was.

## 4. Ask the side chat for the attention entropy per layer

Not yet verified by a dry run.

The side chat builds on the live sources, so the app must be attached to the stream again. If `train.py` has
finished, start it again; it prints a new link. On the landing screen press F4, paste the link and press Enter, as in
step 1.

On the live page, press ⌘L **Chat**. The side chat opens as the right column of the window. Choose the custom agent of
the setup in its agent picker, type "Plot attention entropy per layer while it trains" and press Enter. The agent
starts, reads the catalog's live sources, and makes a dashboard over them: a new spec, its sources, its panels and
their layout. Each edit of the open spec waits on an approval card at the bottom of the window, with its diff; choose **Approve** on each. In the
conversation, calls of the same command fold into one row with their count and the approvals' outcome, such as
`source.add ×4 · approved · done`. When the agent opens the dashboard, the spec page shows it beside the chat, still
attached to the stream, and its panels follow the training run: the entropy by layer and head, and each layer's mean
entropy by step.

## 5. Let an outside agent add a panel through caliper-mcp

Not yet verified by a dry run.

An agent outside the app works on the same workspace through `caliper-mcp`, an MCP server on standard input and
output that reads the app's API file. Add it to your MCP client as a stdio server, with the absolute path of
`cmake-build-release/src/builder/mcp/caliper-mcp` as the command and `--docs` and the repository's `docs` folder as
its arguments. With the dashboard open in the app, ask the agent to add a panel, such as a series of the loss. Its edit
does not change the dashboard at once: it waits on an approval card in the app, with the diff, and the agent's tool
call waits too. Choose **Approve**, and the panel appears on the dashboard and the agent's call returns its result.
Choose **Deny** instead, and the dashboard stays as it was and the agent is told so.

## 6. Save, open again and open in a browser

Not yet verified by a dry run.

Press ⌘S **Save**. The dashboard is saved as a spec applet of the workspace and appears under Projects on the landing
screen. Press ⌘O **Open in browser** in the page bar: your default browser opens the saved dashboard, drawn with the
layout and look of the app's spec page, and the page asks the app for each panel's data once a second. Had you
started the app without `--api`, it starts Caliper API on 127.0.0.1 for this and shows its address in the status bar.
The link carries a viewer token, which reads the workspace's saved dashboards and their data and nothing else, and it
works until the app quits. Press **esc Home**: the landing selects the dashboard under Projects, and Enter opens it
again with the same panels.
