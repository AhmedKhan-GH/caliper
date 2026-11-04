# Live requests and the Qwen inspector

The inspector is a Python process holding one MLX text model. The start screen's
**Projects** section lists running scripts with controls and an **Attach** button.
The usual **F4 → Attach to process** list also works. Attaching opens **Inference**,
**Model**, **Model 3D**, and **Tensor** windows. Other scripts can implement the same requests;
there is no Qwen or MLX code in the app.

## Run

From this checkout, with its Python package installed in `.venv`:

```sh
.venv/bin/python examples/python/qwen_inspector.py
```

In another terminal:

```sh
./cmake-build-notorch/caliper
```

For a separate review executable that leaves the existing app binary in place:

```sh
cmake -S . -B cmake-build-notorch \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$PWD/cmake-build-notorch/review-bin"
cmake --build cmake-build-notorch --target caliper -j8
./cmake-build-notorch/review-bin/caliper
```

Select **Attach** beside `qwen_inspector.py`, or use F4. Enter a prompt and press
**Generate**. **Stop** cancels at the next token or prompt chunk. The global
view pause controls tensor snapshots; it does not pause inference.

Expand a layer in **Model**, or filter by tensor name. Select a tensor to see
its heatmap and controls. Enter **Row** and **Col**, press **Read**,
then use **Set**, **Zero row**, **Zero column**, **Scale row**, **Scale column**,
or **Revert tensor**. Edits are applied to the same model used for the next
Generate. Stop generation before editing. The embedding is shared with the
output layer and is listed once in the tree, with an additional tied-output panel
in Model 3D.

The Tensor window fits each image with square element geometry and empty space
around it. A vector has a plot of the full vector above a shallow strip of
24-pixel square cells; pan the strip to inspect other indices. Its protocol
coordinates remain `(index, 0)`. Elongated tensors have a true-aspect overview
beside a region view (above it for wide tensors). This keeps the full shape and
current location visible while making embedding cells readable. Click the overview
to move, drag the region to pan, and use the wheel to zoom. **Fit tensor** shows
the whole matrix or resets the vector strip; **Cells** zooms around Row/Col. At cell resolution, hover reports
the exact coordinate and value. The colour scale reports the sampled region's
range. Window resizing requests new samples at the displayed resolution.

Model 3D uses a shared element scale and stacks layer groups in order. Linear
attention has a single group frame; full attention has a double frame, with text
labels for both. The final norm and tied output are labelled. Left-drag orbits,
middle-drag or Shift-drag pans, the wheel zooms, and **F** or **Fit** fits the model.
The embedding is much larger than the other tensors; **Focus selected** helps
inspect a small panel at its real scale. Hover gives its name, shape, and sampled
range. Clicking selects it in the tree and Tensor window. Active edits use the
theme's accent outline. **Reset layout** in Model restores the four-pane layout.

A host started before these changes still attaches and supplies legacy overviews.
Restart that host when convenient to enable region sampling and advertised
inspector capabilities. Do not start another model host while one is running.

The default folder is
`~/PycharmProjects/weight-space/models/Qwen3.5-0.8B`. `--model` accepts another
local folder compatible with this host. Model files are only read; edits and
undo patches disappear when the process exits. Ctrl-C stops this host.

For a fresh Python install, use Python 3.12 in a repository venv, with CMake
and Ninja on PATH. The existing wheel build instructions also apply:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python scikit-build-core nanobind
CMAKE_BUILD_PARALLEL_LEVEL=8 uv pip install --python .venv/bin/python \
  --no-build-isolation --config-settings=build.tool-args=-j8 '.[inspector,test]'
cmake --preset release-notorch
cmake --build --preset release-notorch -j8
```

## Limits of this draft

- MLX loads the language model from the multimodal release. Vision and auxiliary
  prediction tensors discarded by the MLX text loader are not shown. The tree
  shows the loaded MLX names, shapes and dtypes. These include the loader's
  convolution layout and normalization conversions.
- Region images sample bin centres at the requested pixel resolution, up to
  2,048 per side and 131,072 values total. These are samples, not averages;
  coarse samples can miss outliers. Exact cells are returned when the region
  fits the sample budget. 3D textures are fetched lazily, at most 128 × 128
  each. Vectors are N × 1 on the wire; higher ranks flatten all dimensions
  before the last into rows. Ranges are sample ranges, not whole-tensor extrema.
- Each generation is one user message with a fresh cache. Thinking uses the
  tokenizer's `enable_thinking` option; `<|im_end|>` ends generation. Generated
  reasoning text is displayed as received. There is no conversation history.
- Prompts are limited to 2,048 tokens including the chat template, output to
  1,024 tokens, and recent edit history to 512 entries. Undo stores only affected
  slices on CPU, capped at 64 MiB and 512 active edits. Revert restores the
  originally loaded tensor even after overlapping edits.
- MLX receives a 3 GiB memory budget guideline and a 128 MiB allocator cache
  limit. These are process-local settings, not a hard cap on total resident memory.
  No model backup, quantization, or weight-file writes are made. An advisory
  per-user lock prevents a second inspector from loading another model. The
  empty `/tmp/caliper-<uid>/qwen-inspector.lock` file remains after exit; its
  operating-system lock is released automatically.
- Stop is cooperative; an already running GPU kernel must finish. Edits and
  generation are serialized. One controller can connect at a time. Detaching
  cancels generation, but keeps edits in the host. Lost requests are never
  replayed automatically. A crashed host can leave its uniquely named socket
  behind; a new default session uses a new token.

## Request channel

Live-session shared memory continues to carry ordinary tensor snapshots.
Optional commands use a Unix socket at `/tmp/caliper-<uid>/<session>.sock`.
The directory is private (0700), and the socket is 0600. A deterministic path
keeps discovery tied to the existing session token without changing its shared
memory ABI. A socket provides reliable framing and backpressure for commands
and token streams; snapshots intentionally allow dropped frames.

RequestServer publishes a private, atomic `<session>.requests.json` sidecar
containing `protocol`, `session`, `pid`, and `capabilities`. The start screen
reads this file and the socket attributes without connecting to the controller
socket. A legacy socket without a sidecar is listed as **Controls**. Busy sessions
remain visible with Attach disabled.

Frames are UTF-8 JSON objects followed by a newline, at most 4 MiB including
the newline. JSON strings escape embedded newlines. Requests have an integer
`id` and string `op`; replies echo the id. One operation may be outstanding,
except `hello` and `stop`, which can be sent during another operation.

```json
{"id":1,"op":"hello"}
{"id":1,"done":true,"result":{"protocol":1,"capabilities":["generate","list_tensors","show_tensor","read_weight","edit_weight","revert_tensor","list_edits","tensor_regions"]}}
{"id":2,"op":"generate","prompt":"Hello","max_new_tokens":32,"temperature":0.7,"top_p":0.9,"top_k":20,"thinking":false}
{"id":2,"done":false,"event":{"text":"Hello!","tokens":1,"tokens_per_second":20.0}}
{"id":2,"done":true,"result":{"tokens":1,"tokens_per_second":20.0,"cancelled":false}}
```

Errors are final replies: `{"id":2,"done":true,"error":"reason"}`.
Malformed or oversized frames disconnect that controller. Disconnect and
`stop` set a cancellation event independently of the model thread. A client
that stops reading is disconnected after a bounded send timeout.

| Operation | Request fields | Result |
| --- | --- | --- |
| `hello` | none | `protocol`, operation names and optional feature markers in `capabilities` |
| `generate` | `prompt`, `max_new_tokens`, `temperature`, `top_p`, `top_k`, `thinking` | streamed `text`, `tokens`, `tokens_per_second`; final metrics and `cancelled` |
| `stop` | none | `stopping`; the generation also receives its own final reply |
| `list_tensors` | none | `model`, `tensors`: `name`, `group`, `shape`, `dtype`, `size` (elements), `bytes`, `rows`, `cols`, `tied`, `role`, `layer_index`, `layer_type`, `edited`, `revision`; optional `output_alias` |
| `show_tensor` | `name`, optional `region`, `target` | `name`, full `rows`/`cols`, `region`, `sample_rows`/`sample_cols`, flat row-major `values`, `row_indices`/`col_indices`, `min`/`max`, `exact`, `edits`, `edited`, `revision` |
| `read_weight` | `name`, `row`, `col` | coordinates and realized dtype's `value` |
| `edit_weight` | `name`, `action`, `row`, `col`, `value` | realized `value`, `edits`; actions: `set`, `zero_row`, `scale_row`, `zero_column`, `scale_column` |
| `revert_tensor` | `name` | `name`, `edits` |
| `list_edits` | none | `edits`, `undo_bytes` |

Edit records are `{id, name, description}`. `show_tensor`, edit, and revert
return history for that tensor; `list_edits` returns recent history for all
of them, including revert records.

`tensor_regions` is a feature marker, not an operation. A region is
`{"row":100,"col":0,"rows":32,"cols":64}` in tensor coordinates, with positive
extents wholly inside the tensor. A target is `{"width":640,"height":480}`;
width samples columns and height samples rows. Both default to 128 and must
be integers from 1 to 2,048. Counts are reduced proportionally to the total
sample budget. Bin-centre indices are returned explicitly, and sampling occurs
before copying to NumPy. `exact` means every cell of the region was returned.
Each edit/revert advances the tensor revision; `edited` describes active undo
patches, not the presence of historical edit records. `output_alias` names the
embedding tensor used by the tied output; it does not duplicate model storage.

A script offers controls with:

```python
from caliper.live import SessionTransport

with SessionTransport.open(script_name="my_model.py") as session:
    with session.request_server(["generate"]) as server:
        server.serve(handler)  # handler(request, emit, cancelled) -> result dict
```

Handlers run serially on the thread calling `serve`. The socket reader never
runs model operations. `emit` streams event dictionaries; the handler's return
value produces the final reply. Poll `cancelled.is_set()` during long work.

## Checks

Run the real-model check with no other inspector holding the model:

```sh
.venv/bin/python examples/python/check_qwen_inspector.py
```

It starts one host, checks metadata, embedding sampling, text generation,
set/read/revert, row and column edits, generation after editing, invalid input,
thinking, and Stop. It shuts down only its own child. To reuse a running host,
pass `--connect SESSION`; detach the app first so the test can be its controller.
The check refuses a host with active edits and restores the tensors it changes.

Transport and backend-double tests do not require Metal:

```sh
.venv/bin/python -m pytest python/tests/test_requests.py \
  python/tests/test_qwen_inspector.py -q -p no:cacheprovider
./cmake-build-notorch/tests/caliper_live_requests_tests
```

The CMake test targets also include `caliper_tensor_geometry_tests`,
`caliper_model3d_geometry_tests`, `caliper_live_inspector_tests`,
`caliper_live_model3d_tests`, and `caliper_request_discovery_tests`. Geometry,
camera projection/picking, region scheduling, thumbnail caching, and discovery
run without a GPU. The 3D geometry fixture includes all 320 Qwen tensor shapes
and the tied output alias. Rendering appearance and interactive GPU performance
still require an app window outside the sandbox.

Backend doubles test edit and cancellation logic; they do not validate MLX
kernels or actual Qwen output. A real-model check needs access to Metal.
