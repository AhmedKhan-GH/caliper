# Build a dashboard from a live training run

This walk-through builds dashboards in Caliper from a real training run and from two data files, in six steps, with
every way the app offers to build one. A training script trains a small transformer on your Mac's GPU and streams its
loss and its attention to the app. You then build a dashboard by hand from the data files, change one of its panels
with the widget agent (⌘K), ask the side chat for a dashboard of the attention entropy per layer while the model
trains, let an outside agent add a panel through `caliper-mcp`, and save a dashboard, open it again and open it in a
browser.

| Step | What you do | Dry run |
|---|---|---|
| 1 | Train a model and attach the app to its stream | `dry_run/step1_train_and_attach.py` |
| 2 | Drop the data files and build a dashboard by hand | `dry_run/step2_drop_and_build.py` |
| 3 | Change a panel with ⌘K, accept it and undo it | `dry_run/step3_cmdk.py` |
| 4 | Ask the side chat for the attention entropy per layer | `dry_run/step4_side_chat.py` |
| 5 | Let an outside agent add a panel through `caliper-mcp` | `dry_run/step5_outside_agent.py` |
| 6 | Save, open again and open in a browser | `dry_run/step6_save_reopen_browser.py` |

Each dry run plays its step in the hidden app with real components and writes captures. [Dry runs](#dry-runs) at the
end says how to run them, one at a time or all six with one command.

The files in this folder:

| File | What it is |
|---|---|
| `train.py` | The training script of step 1 |
| `make_data.py` | Writes the two data files of step 2 again, the same on every run |
| `data/run_metrics.parquet` | A table of a training run's `step`, `loss`, `accuracy` and `lr`, 300 rows |
| `data/attention.safetensors` | One attention tensor `attn`, float32, 4 layers × 4 heads × 32 × 32 |
| `chat-agent.sh` | Starts Caliper's own agent on a local model, as the side chat's agent in step 4 |
| `outside_agent.py` | A small MCP client that acts as the outside agent of step 5 |
| `dry_run/` | One script per step, `run_all.py`, which runs them in order, and the helpers they share |

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
outside agents use.

### A Python for the training script

The training script needs a Python with PyTorch, which uses the GPU through MPS, and Caliper LIB, the `caliper`
package. Build and install the package's wheel into that Python as
[Watch a PyTorch training run live](../../docs/tutorials/live-pytorch.md#build-and-install-the-wheel) describes. From
a checkout, `PYTHONPATH=python` instead of the wheel is enough for this script, since it only streams with
`caliper.serve()`.

### A Python for the agent, and the local model

The widget agent of ⌘K and the side chat's agent in this walk-through are both `caliper-agent`, a Python program that
runs from the repository's `src/builder/agent` folder. Give it a Python with its dependencies, and tell the app where
that Python is:

```sh
python3 -m venv .venv-agent
.venv-agent/bin/pip install ./src/builder/agent
export CALIPER_AGENT_PYTHON=$PWD/.venv-agent/bin/python
```

Install Ollama and download the model that both agents use:

```sh
ollama pull qwen3.8:27b
```

Caliper asks Ollama for a context length of 16,384 tokens for ⌘K and 32,768 tokens for the side chat, with every
request, and Ollama unloads the model a minute after its last request. Left to itself, Ollama would load this model
with room for its whole context of 262,144 tokens, which took 34 GB here. When a conversation no longer fits its
context length, the agent does not send the next request and says so, and the turn ends as **LIMIT**.

**What to expect in memory.** In a run of this whole walk-through on a 64 GB Mac, with the model at the side chat's
32,768 tokens, Ollama's model process took about 22 GB and the training script under 1 GB, and free memory fell from
60 % to 24 % while the model worked. If you see memory pressure, close other heavy apps before steps 3 and 4.

### Starting the app

Start the app with Caliper API on, which the widget agent needs:

```sh
cmake-build-release/caliper --api
```

The side chat and **⌘O Open in browser** start Caliper API themselves when it is not running, on this machine only,
and the status bar then shows its address.

## 1. Train a model and attach the app

In a terminal, start the training script with the Python that has PyTorch and Caliper LIB:

```sh
.venv/bin/python examples/build_a_dashboard/train.py --seconds 1200
```

It trains for five minutes by default; `--seconds 1200` gives it twenty, so that it still trains when you reach step
4, and `--cpu` trains on the CPU instead of the GPU. The model is a transformer of 4 layers with 4 heads each. Each of
its sequences is 24 random tokens followed by the same 24 tokens again, so the second half can be predicted from the
token 23 positions back, and the model learns to look there. The script makes the data itself, so nothing is
downloaded.

The script prints its stream link, then a line every 10 seconds with the step, the loss and the entropy of each
layer's attention, in nats.

In Caliper, on the landing screen, press F4, or click **F4 Attach to process** in the Projects pane. The list shows
the script with its stream. Select it and press Enter. A stream on another machine is not in the list: paste its link
into the field under the list. You can also open the app on a stream directly, with the link on the command line
(`caliper --api --url "<link>"`).

The live page opens. Its bar reads **esc Home**, **⌘D Detach**, `train.py` and a diamond with **live**, with
**⌘L Chat** at the right. Under it, the run bar says **running** with **Pause** and **Step**, which hold the training
loop at its next step and run one step at a time. Seven views fill the page:

| View | What it shows |
|---|---|
| `loss` | The training loss of every step, from the first step on, even if you attached later |
| `entropy.l0` to `entropy.l3` | Each layer's attention entropy, averaged over its heads, at every step |
| `attn` | The attention of the batch's first sequence, the mean over the heads, one map per layer in a 2 × 2 grid, with menus to choose a layer, a head, a reduction or a range |
| `attn.entropy` | The entropy of each head, layers as rows and heads as columns |

On an M-series Mac with nothing else on the GPU, the loss falls slowly from 4.3 at first, then drops to near 0 some
15 to 20 seconds after the start. At the same time the entropy of the deeper layers falls, while layer 0 stays near
2.9 nats, which is what attention spread evenly over the earlier tokens gives: the heads of the deeper layers learn
to look at one earlier token each. In `attn`, the maps of the deeper layers turn into thin diagonal lines as that
happens.

The app's catalog now lists the stream's views as live sources, which the side chat finds in step 4: `live:loss`,
`live:attn` (shape `[4, 47, 47]`, one map per layer), `live:attn.entropy` (`[4, 4]`) and `live:entropy.l0` to
`live:entropy.l3`, each with its rate. A heatmap on the stream carries at most three dimensions and a dashboard's spec
cannot reshape its data, which is why the script computes the entropy per head itself.

Press **esc Home**. The app stays attached to the stream on every page, and the landing screen shows the script with
**F5 Live page**, which goes back to it. Only **⌘D Detach** on the live page, attaching another stream, or quitting
ends the attachment.

## 2. Drop the data files and build a dashboard by hand

The two data files are in `data/`. The table holds the logged metrics of a training run, one row every 10 steps for
3,000 steps. The tensor holds the attention weights of four layers of four heads over 32 tokens, as layers × heads ×
query × key.

On the landing screen, press ⌘K, type the title `Training run` and press F3 **New dashboard**, which takes the
command line's text as the title. An empty dashboard opens in build mode: Sources and Primitives on the left, the
canvas in the middle, which says to drop a data file or pick a source, and Properties on the right.

Drag `data/run_metrics.parquet` and `data/attention.safetensors` from the Finder onto the window. Caliper imports
both into the workspace, and Sources lists `run_metrics` (300 × 4) under Tables and `attention_attn`
(f32 4×4×32×32) under Tensors. Press ⌘T **Text** to show the spec text under the canvas: it shows the TOML that will
be saved, and it scrolls to each change as you make it. Then build three panels:

- **A series of loss and accuracy.** Click `run_metrics`, then **Series** under Primitives. The series takes `step` as
  x and the other columns as lines. Remove the `lr` line with its × under Bindings in Properties, then type the title
  `Loss and accuracy` and the x label `Step`.
- **A heatmap of one attention head.** Click `attention_attn`. **Heatmap** is faint, because a heatmap takes a tensor
  of two dimensions and this one has four; its tooltip says so and says to slice or reduce it in Properties. Under
  Transforms in Properties, click **Slice** twice. Each slice shows its axis with the axis's range, such as
  `axis 0 · 0–3`. Set the first slice's index to 1, which picks layer 1, and the second slice's index to 2, which
  picks head 2 of that layer; the shape after each slice shows at its right and ends at 32×32. Click **Heatmap**,
  then type the title `Attention of layer 1, head 2` and the labels `Key` and `Query`.
- **A table of the metrics.** Click `run_metrics` under Spec in Sources and drag **Table** to the bottom edge of the
  canvas, so that it spans the width under the two panels. Drag the divider above it down to give the plots more room.

Each change is one undo step, so ⌘Z takes back the last one. The status bar reads `3 panels`, `2 sources` and
`unsaved`. Press ⌘S **Save** to save the dashboard as the applet `training-run` in the workspace. If you start a new
dashboard, open another one or close the window while a dashboard has unsaved changes, Caliper asks
**Save changes to Training run?** first.

## 3. Change a panel with ⌘K, accept it and undo it

Open the widget agent's settings with ⌘, (or **⌘, Agent** on the landing screen). Choose the **Ollama** preset, choose
**List** and then `qwen3.8:27b` under **Model**, and choose **Probe**: Tool calling and Vision say Yes, and Context
length says 16,384 tokens, from the preset. [Set up the widget agent](../../docs/guides/set-up-the-widget-agent.md)
explains each setting. A change of these settings applies at once.

Press ⌘B to leave build mode. Click the series panel to focus it; its frame turns bronze. Press ⌘K: a one-line prompt
opens at the top of the panel, with the model's name at its end. Type "Use a log scale for the y axis" and press
Enter. While the agent works, the prompt row shows its latest step; a turn took 10 to 30 seconds here. When it has
finished, the panel is drawn with the change and **PREVIEW** in its title strip, with the agent's reply and the
changed key, `y_log → true`. Nothing has changed yet: press Enter or choose **⏎ Accept** to apply it as one undo
step. To undo it, press ⌘B for build mode and then ⌘Z, and the panel returns to what it was.

The agent may use up to six requests to the model for one change. If it stops at that limit, the card says **LIMIT**
and that the change is not finished, with the edits made so far: **Accept** applies those, and **Refine** lets the
agent go on.

⌘K also opens on a selection, on a property and on a source. Shift-drag across the series to select a range of steps,
press ⌘K and type "Make a table of these points"; or use the ⌘K beside a property in Properties.

## 4. Ask the side chat for the attention entropy per layer

The side chat builds on the live sources, so the app must still be attached to the stream of step 1. If `train.py`
has finished, start it again and attach to it with F4 as before.

Press ⌘L **Chat**. The side chat opens as the right column of the window. In its agent picker, choose **Custom
command**, name it `Caliper agent`, and give it the absolute path of `chat-agent.sh` in this folder, with the agent's
Python in front, since an app started from the Finder gives its agents only the system's `PATH`:

```text
/usr/bin/env CALIPER_AGENT_PYTHON=<repo>/.venv-agent/bin/python <repo>/examples/build_a_dashboard/chat-agent.sh
```

The script starts `caliper-agent` from the repository on the model that Ollama serves; it uses no account and reads
no key. Then choose **Local model** in the same picker: it marks this agent as running on your machine, so the chat's
header shows **Shared GPU** while a training run is attached.

Type "Plot attention entropy per layer while it trains" and press Enter. The agent reads the catalog's live sources
and makes a dashboard over them: a new spec, one source per layer, and a series panel. The app shows the spec page
for the new spec beside the chat, still attached to the stream, so that you see the dashboard whose edits you review.
Each edit waits on an approval card at the bottom of the window, with its change; choose **Approve** on each. In the
conversation, calls of the same command fold into one row with their count and the approvals' outcome, such as
`source.add ×4 · approved · done`. Once the panel is approved, its lines follow the training run.

A turn took one and a half to three and a half minutes here, most of it the model. The agent reaches Caliper with the
chat session's own token, so it can change nothing that you have not approved. A model this size sometimes makes a
call that fails, such as asking for the open spec before there is one, and then carries on.

## 5. Let an outside agent add a panel through caliper-mcp

An agent outside the app works on the same workspace through `caliper-mcp`, an MCP server on standard input and
output that reads the app's API file, whose token runs commands that you review. With the app running with Caliper
API and the dashboard of step 2 open, add `caliper-mcp` to your agent's MCP configuration
([MCP server](../../docs/references/mcp-server.md) gives the entries), or try it with the example client, which needs
only Python:

```sh
python3 examples/build_a_dashboard/outside_agent.py --mcp cmake-build-release/src/builder/mcp/caliper-mcp
```

It lists the tools, reads the catalog and the open spec, and adds a series panel of the learning rate beside the
first panel. The edit does not change the dashboard at once: it waits on an approval card in the app from **MCP
agent**, with its change, and the client's call waits too. Choose **Approve**, and the panel appears on the dashboard
and the call returns its result. Choose **Deny** instead, and the dashboard stays as it was and the agent is told so.
With `--model qwen3.8:27b`, the model that Ollama serves chooses the calls instead of the script.

## 6. Save, open again and open in a browser

With the dashboard of step 2 open, press ⌘S **Save** if the status bar says `unsaved`. Press Esc to give up a panel's
focus, and Esc again to go Home. The landing screen lists **Training run** under Projects and has it selected. Press
Enter: the dashboard opens again and draws as before.

To open it in a browser, choose **⌘O Open in browser** in the page bar. Caliper starts Caliper API on this machine if
it is not running and opens the saved dashboard in your default browser. The page draws the same panels with the
app's layout and look, from the data the running app sends. Its link carries a token that reads the workspace's saved
dashboards and their data and nothing else, and it stops working when Caliper API stops. The browser shows the saved
dashboard, so save first if you have changed it.

## Dry runs

Each script in `dry_run/` plays one step in the app started with `--hidden`, which draws offscreen and shows nothing
on screen. It keeps the app's data and workspace in a folder of its own, stops every program it started when it
ends, and deletes only temporary folders that it made itself. It prints one line per check and exits with 0 when
every check passed. `run_all.py` runs the six in order and writes every capture into one folder, with a summary of
which step passed, where one stopped and how long each took:

```sh
python3 examples/build_a_dashboard/dry_run/run_all.py --out build/walkthrough --size 1440x900 \
    --train-python .venv/bin/python --agent-python .venv-agent/bin/python
```

`--steps 1,2,6` runs some of them. With `--story` the steps share one workspace, so the dashboard built in step 2 is
the one that ⌘K changes, the outside agent adds to and step 6 opens again. `--stream train` gives the side chat the
training run of step 1 itself; without it step 4 plots a small test stream. Each script also runs alone, with the
same options (`--caliper`, `--out`, `--size`, and `--train-python` or `--agent-python` where it needs one); its
`--help` lists them.

Steps 3 and 4 need Ollama with `qwen3.8:27b`, and the runner stops before starting anything when it is missing.
During those steps it prints the free memory every 5 seconds and stops them if memory runs short (`--memory-low`,
`--memory-critical`). Step 6 needs Node and Playwright from the web package (`cd web && npm ci`), with its headless
Chromium (`npx playwright install chromium-headless-shell` in `web/`). The scripts themselves need only Python's
standard library.

What a dry run does differently from a person:

- A hidden window receives no drop from the Finder, so step 2 hands the two paths to the function that the window's
  drop callback calls. Every other input is a click, a drag, a key or typed text, played through the app's own event
  queue.
- The hidden app opens no browser. In step 6 the link of **⌘O Open in browser** goes to a file, and the script opens
  it in headless Chromium.
- Approvals are given through the hidden app's own test input, in place of clicks on the cards.
