# Model Interpretation

Caliper LIB connects live tensor publication and controls to `transformer-lens`, `circuit-tracer`, and `sae-lens`.
The model, transcoders, and sparse autoencoders stay inside the Python process.
Library pieces draw attribution graphs, feature examples, and next-token probabilities in the native app and the browser viewer.
The [interpretation tools](../agents/interpretation.md) expose the same operations through Caliper MCP.

## Installation

Install the `interp` extra from the repository root to pin compatible library versions:

```sh
pip install -e ".[interp]"
```

The build needs Rust (`cargo`), because it compiles the runtime WGSL compiler into `caliper._producer`, which runs the derive halves of the library pieces.

The `interp` extra installs compatible tested versions:

- `torch>=2.15.0` and `transformers>=5.24.0,<5.25.0`
- `transformer-lens==3.9.0`
- `sae-lens==6.56.3`
- `nnsight==0.8.4`
- `circuit-tracer` at commit `82996f144b3371ae858ea5713590d004c218f0c2`

On Apple Silicon, `caliper.interp` sets `TRANSFORMERLENS_ALLOW_MPS=1` and selects `mps` automatically when available.

## Model Probes

`intervene.points` (`caliper._ablate.points(model)`) lists module, attention-head, and semantic probe points on a PyTorch or `HookedTransformer` model.
Decoder blocks expose `resid_pre`, `attn_out`, `mlp_out`, and `attn_pattern` (plus `mlp_in` when `cfg.use_hook_mlp_in` is true) alongside module and `blocks.L.attn#H` head points.
Any watched module also exposes its latest forward activation to derive programs as `live:<module>.act` without a CPU copy.

`caliper.interp.publish(name, tensor)` exposes a named tensor to derive programs.
For example, `publish("trace.counts", counts)` updates the catalog source `live:trace.counts`.
Published tensors keep their device and storage, and watch removal clears publications owned by that model.

### Publication Lifetime

Each publication belongs to the current `caliper.watch` lifetime, or to the next watch if none is active yet.
Watch removal or model replacement invalidates prior publication leases and clears published tensors.
A trace, example, or sweep request checks its lease under the program lock before it publishes a tensor group.
If the watched model changes during execution, the request raises `RuntimeError` with code `interpret.watch_changed`.
Producers observe either the complete previous tensor group or the complete new tensor group.

## Feature Dictionaries

`caliper.interp.load_backend(model_ref, dictionary_ref=None, *, sae_id=None, device=None, dtype=None)` loads a staged `Backend`.
Pass a Hugging Face model name or local snapshot path as `model_ref`.
When `dictionary_ref` names a transcoder hub alias, a `TranscoderSet`, or a local transcoder directory, `load_backend` builds a staged `TransformerLensReplacementModel`.
When `dictionary_ref` names a `sae-lens` release (with `sae_id`) or a local SAE directory, `load_backend` builds a `HookedTransformer` and stages the `SAE` on CPU before moving it to the target device.

```python
from caliper.interp import Tracer, load_backend

backend = load_backend("/path/to/model", "/path/to/transcoders")
tracer = Tracer(backend.model, backend.tokenizer, backend.dictionary, backend=backend)
graph = tracer.trace("He saw a carrot and had to grab it, his hunger was like a starving")
```

`backend.dictionary` is a `DictionaryView` with uniform `metadata`, `encode(layer, x)`, and lazy `decoder_vectors(layer, ids)`.
Any `HookedTransformer` architecture in `transformer-lens` and any compatible `sae-lens` or `circuit-tracer` dictionary can attach through `load_backend`.
For offline tests without model downloads, pass `model_ref="tiny-lm"` to construct a deterministic 2-layer `ReplacementModel` and `TranscoderSet`.

## Example Index

`FeatureIndex` loads corpus examples and promoted tokens from a memory-mapped index on disk.
When no disk index exists, `LiveFeatureIndex(backend)` resolves feature summaries and top 33-token windows in memory.
When a session has network access and `SAE.cfg.metadata` defines `neuronpedia_id`, `LiveFeatureIndex` queries `get_neuronpedia_feature`.
When a session runs offline or uses unindexed transcoders, `LiveFeatureIndex` scans prompt text with a bounded min-heap.
That scan zeros BOS positions.

`feature.show` returns the heuristic label, activation count, maximum value, and promoted tokens for a feature.
`feature.examples` returns up to 16 ranked 33-token windows and publishes the central 25 slots to `token_strip@1`.

### Index Attachment

`FeatureIndex` reads corpus examples from an index folder on disk.
`FeatureIndex` and `FeatureIndex.resolve(path)` check three locations in order and select the first directory with `index.json`:

1. The index directory directly, such as `<path>/index`.
2. A nested folder inside the dictionary at `<dictionary>/index`.
3. A sibling folder next to the dictionary at `<dictionary>/../index`.

Path resolution checks only local paths for sibling folders.
The sibling check runs only when `<dictionary>` exists on disk.
As a result, non-path hub aliases do not match an unrelated `index` folder in the current directory.
An explicit index path always takes priority over nested and sibling folders.

Pass either the dictionary folder or the index folder to `FeatureIndex`:

```python
from caliper.interp import FeatureIndex, load_backend

backend = load_backend("/path/to/model", "/path/to/dictionary")
index_dir = FeatureIndex.resolve("/path/to/dictionary")
if index_dir is not None:
    index = FeatureIndex(
        index_dir,
        model=backend.model,
        tokenizer=backend.tokenizer,
        dictionary=backend.dictionary,
    )
```

Run `tools/interp/build_index.py` to build a new index inside `<dictionary>/index` from a JSONL corpus:

```sh
python -m tools.interp.build_index --model /path/to/model --dictionary "$DICT_DIR" --text-jsonl "$CORPUS_JSONL"
```

### Index Layout

Each index folder holds `index.json` and four binary files.
`FeatureIndex` maps the binary files in read-only mode and checks the layer and latent counts against `dictionary.metadata`.
In `index.json`, the latent count uses either `"latents"` or `"features"`.
When `"latents"` is absent, `FeatureIndex` reads `"features"` to support legacy feature indexes.
If both keys appear with different values, `FeatureIndex` raises `ValueError`.

| File | Format | Contents |
|---|---|---|
| `index.json` | JSON | `layers`, `latents` or `features`, `index_hash`, and corpus metadata |
| `table.npy` | NumPy `(layers, latents, 16)` | Ranked `activation` (`<f8`) and token `offset` (`<i8`) pairs |
| `counts.npy` | NumPy `(layers, latents)` | Positive activation count (`<u8`) per feature |
| `tokens.bin` | Raw `<u4` stream | Concatenated token IDs |
| `documents.bin` | Fixed 145-byte records | UTF-8 `id` (`S128`), `start` (`<i8`), `length` (`<i8`), and `truncated` flag |

## Attribution Graphs

`Tracer(model, tokenizer, dictionary, backend=backend).trace(prompt)` runs `circuit_tracer.attribute`, transfers the graph to CPU, runs `prune_graph`, and publishes graph tensors.
Prompts accept up to 32 model tokens with BOS and refuse longer inputs without truncation.
Default settings select up to 10 logit targets (`desired_logit_prob=0.95`), up to `5,000` feature targets, `node_threshold=0.7`, and `edge_threshold=0.9`.
Pass `show_more=True` to relax pruning to `node_threshold=0.78` and `edge_threshold=0.93`.

After a standalone `tracer.trace(prompt)` call, invoke `caliper.step()` once so derive programs process the published tensors.
In a hosted inference run, `trace.run` executes and publishes automatically at the next run boundary.
`Tracer` caches up to 16 completed graphs keyed by prompt, active interventions, thresholds, and label index identity.

## Graph Tensors

`Graph.publish()` writes fixed-capacity tensors on the model device.
Graphs within 512 nodes and 8,192 edges use `(512, 8192)` capacities, while larger graphs up to 8,192 nodes and 262,144 edges use `(8192, 262144)` capacities.

| Source | Type | Shape | Fields |
|---|---|---|---|
| `live:trace.counts` | `int32` | `(4,)` | Nodes, edges, tokens, layers |
| `live:trace.node_i` | `int32` | `(N, 4)` | Kind, layer, position, feature or token ID |
| `live:trace.node_f` | `float32` | `(N, 4)` | Activation, influence, top-logit effect, reserved |
| `live:trace.edge_i` | `int32` | `(E, 2)` | Source and target node rows |
| `live:trace.edge_f` | `float32` | `(E, 2)` | Weight and normalized weight |
| `live:trace.label_text` | `uint32` | `(32 + N, 16)` | Packed prompt and node labels |

Node kinds are token `0`, feature `1`, error `2`, and logit `3`.
Token layers use `-1`, logit layers use the layer count, and feature and error layers use zero-based indices.
`trace.read` pages through full JSON node and edge records up to 256 items per page.

## Graph Pieces

`attribution_graph@1` places prompt tokens horizontally and model layers vertically.
Bind `graph_counts` to `live:trace.counts` and bind `node_i`, `node_f`, `edge_i`, `edge_f`, and `label_text` to their `live:trace.*` sources.
Its `selected_node` parameter reads shared state `selected_node`, highlights connected edges, and displays the selected label.

`token_strip@1` binds to the five `live:examples.*` tensors documented in [Example Panel Bindings](../agents/interpretation.md#example-panel-bindings).
`logit_bars@1` binds either to `live:trace.probs` for static baseline logits or to `live:output.probs`, `live:output.tokens`, and `live:output.label_text` for live baseline-versus-steered comparisons.
`heat_grid@1` binds to `live:patch.grid` to render layer-by-position activation recovery.

## Feature Steering

`Backend.intervene_logits` and live `InterventionExecutor` hooks support `clamp`, `scale`, `add`, and `ablate` interventions.
On a `ReplacementModel`, feature `clamp` and `scale` edits call `ReplacementModel.feature_intervention` directly.
On a standalone `HookedTransformer` with an `SAE`, edits run in delta form (`x + (sae.decode(modified) - sae.decode(base))`), which preserves exact logits at identity.

`Output(backend, interventions)` compares clean and steered next-token distributions and publishes `live:output.probs` `(2, 10)`, `live:output.tokens` `(10,)`, and `live:output.label_text` `(10, 16)`.
`Output.generate(prompt)` generates 24 greedy baseline tokens and 24 greedy steered tokens while keeping permanent replacement hooks intact.

## Activation Patching

`caliper.interp.patch_sweep(backend, interventions, clean_prompt=..., corrupted_prompt=..., metric=...)` runs a `hook_resid_pre` activation patching sweep across all `(layer, position)` cells.
Clean and corrupted prompts must tokenize to equal lengths within the 32-token cap.
`metric` specifies distinct `positive_token` and `negative_token` vocabulary IDs.
The sweep runs at most 32 cells per forward pass, publishes normalized recovery as `live:patch.grid` `(layers, positions)`, and returns full summary values.

## Refusals and Limits

`MemoryGuard` enforces a 40 GB process limit (`40,000,000,000` bytes) and an 8 GB host available memory reserve (`8,000,000,000` bytes).
Missing dictionaries, unknown probe points, out-of-range features, and unequal patch prompt lengths return structured refusals.
See [Interpretation Tools](../agents/interpretation.md) for MCP request schemas and paging bounds.

## Reference Calls

The `caliper.interp` package exports the following functions and classes.

[](){#caliper.interp.publish}
### `publish`

::: caliper.interp.publish
    options:
      show_root_heading: false
      show_root_toc_entry: false

[](){#caliper.interp.load_backend}
### `load_backend`

::: caliper.interp.load_backend
    options:
      show_root_heading: false
      show_root_toc_entry: false

[](){#caliper.interp.Tracer}
### `Tracer`

::: caliper.interp.Tracer
    options:
      show_root_heading: false
      show_root_toc_entry: false

[](){#caliper.interp.Graph}
### `Graph`

::: caliper.interp.Graph
    options:
      show_root_heading: false
      show_root_toc_entry: false

[](){#caliper.interp.FeatureIndex}
### `FeatureIndex`

::: caliper.interp.FeatureIndex
    options:
      show_root_heading: false
      show_root_toc_entry: false

[](){#caliper.interp.LiveFeatureIndex}
### `LiveFeatureIndex`

::: caliper.interp.LiveFeatureIndex
    options:
      show_root_heading: false
      show_root_toc_entry: false

[](){#caliper.interp.patch_sweep}
### `patch_sweep`

::: caliper.interp.patch_sweep
    options:
      show_root_heading: false
      show_root_toc_entry: false

[](){#caliper.interp.make_tiny_backend}
### `make_tiny_backend`

::: caliper.interp.make_tiny_backend
    options:
      show_root_heading: false
      show_root_toc_entry: false

[](){#caliper.interp.make_tiny_sae_backend}
### `make_tiny_sae_backend`

::: caliper.interp.make_tiny_sae_backend
    options:
      show_root_heading: false
      show_root_toc_entry: false
