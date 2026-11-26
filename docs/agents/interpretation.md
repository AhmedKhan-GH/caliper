# Interpretation Tools

Caliper MCP queries feature dictionaries, example windows, attribution graphs, activation sweeps, and hosted inference runs.
The model, transcoders, and sparse autoencoders stay inside the Python process.
The app forwards bounded requests over wire protocol v3 and never blocks its render thread on model execution.

## Commands

Each command maps to one MCP tool by replacing the dot with an underscore, such as `trace_run` or `patch_sweep`.
Read commands use the `read` effect.
`trace.run`, `patch.sweep`, and `trace.validate` use the `run` effect and follow the run approval rules.

| Command | Request Fields | Result Data |
|---|---|---|
| `feature.dictionary` | None | Registered dictionary metadata and index identity |
| `feature.show` | `layer` and `feature`, or `node` | Label, count, maximum, promoted tokens, example tokens |
| `feature.examples` | `layer` and `feature`, or `node`, plus optional `limit` | Ranked 33-token windows and `live:examples.*` bindings |
| `trace.run` | `prompt`, plus optional `max_targets`, `max_nodes`, `max_edges`, `max_n_logits`, `desired_logit_prob`, `node_threshold`, `edge_threshold`, `batch_size`, `update_interval` | Trace state, step, duration, counts, completeness |
| `trace.status` | None | Latest trace status, diagnostics, and hosted `inference` state |
| `trace.read` | Optional `node_offset`, `node_limit`, `edge_offset`, `edge_limit` | Paged node and edge records, counts, next offsets |
| `patch.sweep` | `clean_prompt`, `corrupted_prompt`, `metric` (`positive_token`, `negative_token`) | Clean and corrupted metrics, `(layers, positions)` recovery grid, `live:patch.grid` |
| `trace.validate` | None | Zero-ablation effects for up to ten top features, Spearman correlation, sign matches |

Every command returns `request_id`, `pending`, and `result`.
When `pending` is true, `result` is null.
Call the same tool with only `request_id` to poll the retained reply without starting a second operation.

Feature queries accept either a `node` row in the current graph or explicit `layer` and `feature` integers.
Prompts accept 1 to 4,096 UTF-8 bytes and 1 to 32 model tokens, including special tokens.
`trace.run` accepts `max_targets` up to `5,000`, `max_nodes` up to `8,192`, and `max_edges` up to `262,144`.
`trace.read` returns at most 256 nodes and 256 edges per page.

## Run Service

A custom script can register `InterpretationService` before starting its control loop:

```python
from caliper.interp import LiveFeatureIndex, Tracer, load_backend
from caliper.interp._service import register_service, unregister_service

backend = load_backend("tiny-lm")
index = LiveFeatureIndex(backend)
tracer = Tracer(backend.model, backend.tokenizer, backend.dictionary, backend=backend, index=index)
service = register_service(
    model=backend.model,
    tokenizer=backend.tokenizer,
    dictionary=backend.dictionary,
    index=index,
    tracer=tracer,
)
try:
    run_loop()
finally:
    unregister_service(service)
    index.close()
```

`InterpretationService` queues at most 32 pending requests and executes them at `caliper.step()`.
`trace.status` returns cached state immediately without running a model forward pass.
`feature.show` and `feature.examples` suspend active interventions so feature activations reflect the clean baseline.
Missing resources return structured error codes such as `interpret.missing_dictionary`, `interpret.missing_trace`, and `interpret.unsupported`.

## Example Panel Bindings

When a viewer picks a feature node in `attribution_graph@1`, set shared state `selected_node` and call `feature.examples` with `node`.
When `feature.examples` completes, the service publishes five tensors for `token_strip@1`:

| `token_strip@1` Input | Source | Shape |
|---|---|---|
| `tokens` | `live:examples.tokens` | `(16, 25)` |
| `activations` | `live:examples.activations` | `(16, 25)` |
| `example_ids` | `live:examples.example_ids` | `(16,)` |
| `valid_tokens` | `live:examples.valid_tokens` | `(16,)` |
| `label_text` | `live:examples.label_text` | `(400, 16)` |

The strip displays eight rows of 25 tokens per page across pages `0` and `1`.
Packed labels use 16 little-endian `uint32` words per token (at most 63 UTF-8 bytes plus a zero byte).
Set `activation_max` from `feature.show` (`max(1e-6, maximum)`).

`attribution_graph@1` binds `graph_counts`, `node_i`, `node_f`, `edge_i`, `edge_f`, and `label_text` to `live:trace.*`.
For graphs up to 512 nodes, `trace.label_text` has shape `(544, 16)`.
For graphs above 512 nodes (up to 8,192 nodes), `trace.label_text` has shape `(8224, 16)`.

### Graph Density

`attribution_graph@1` fits token columns horizontally and layer bands vertically in an orthographic projection.
Nodes within each `(layer, position)` cell sort in descending influence order.
When the projected node radius falls below two physical pixels, the draw shader renders a single-pixel density marker that preserves the node color and pick ID.
Zooming in restores full node shapes once the radius reaches two pixels.

### Output Bars

In a hosted inference run, bind `logit_bars@1` to the paired live comparison sources:

| Input | Source | Shape |
|---|---|---|
| `probs` | `live:output.probs` | `(2, 10)` |
| `tokens` | `live:output.tokens` | `(10,)` |
| `label_text` | `live:output.label_text` | `(10, 16)` |

Row `0` holds clean baseline probabilities and row `1` holds steered probabilities across a shared union of top tokens.
Standalone M7 traces without `Output` also publish baseline-only `live:trace.probs`, `live:trace.logit_tokens`, and `live:trace.logit_label_text`.

## Inference Runs

`run.start` accepts `kind: "inference"` with `model`, `dictionary`, and `prompt` to launch a managed `HostedInference` process (`caliper.interp._hosted`).
Hosted inference registers parameters `prompt`, `selected_node`, `clamp`, `clamp.status`, `retrace`, `output.baseline`, `output.steered`, and `output.status`.
It also registers actions `apply_clamp`, `reset`, and `generate` (plus `dallas.cell14` through `dallas.cell29` on 26-layer Gemma Scope transcoders).

`run.fork` checkpoints the active prompt, feature selection, and intervention specifications into `.json` metadata and starts a branched inference run without copying model weights.
`trace.status` includes an `inference` dictionary with `prompt_hash`, `interventions_hash`, latest `comparison` top tokens, and `generation` token counts.

## Text Parameters

`caliper.param("prompt", "")` registers a free-text parameter under the `text_params` wire capability.
Values accept up to 4,096 UTF-8 bytes without embedded NUL characters.
Spec controls with `kind = "text"` bind to `param:prompt`, `param:output.baseline`, and `param:output.steered`.
Pressing Return or leaving the field submits edits at the next run boundary.

## Structured Wire Metadata

Interpretation requests travel over wire protocol v3 CONTROL and CONTROL_REPLY frames when both peers announce `interpretation`.
Requests use `op = "interpret"` with `{"version": 1, "method": ..., "params": ...}`, and replies return `queued`, `ok`, or `error` with structured JSON payloads.
Each structured payload has a 256 KiB byte limit, at most eight nesting levels, 64 keys per object, 512 items per list, and 16,384 total values.
