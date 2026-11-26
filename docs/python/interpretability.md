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

The `interp` extra installs exact tested versions:

- `torch>=2.15.0` and `transformers>=5.24.0,<5.25.0`
- `transformer-lens==3.9.0`
- `sae-lens==6.56.3`
- `nnsight==0.8.4`
- `circuit-tracer` at commit `82996f144b3371ae858ea5713590d004c218f0c2`

On Apple Silicon, `caliper.interp` sets `TRANSFORMERLENS_ALLOW_MPS=1` and selects `mps` automatically when available.

## Model Probes

`caliper.interp.points(model)` lists module, attention-head, and semantic probe points on a PyTorch or `HookedTransformer` model.
Decoder blocks expose `resid_pre`, `mlp_in`, `mlp_out`, and `attn_pattern` alongside module and `blocks.L.attn#H` head points.
`caliper.interp.capture(model, points)` records activations on the model device without a CPU copy and removes its hooks on exit.

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
When `dictionary_ref` names a transcoder hub alias (`"gemma"` or `"llama"`), a `TranscoderSet`, or a local transcoder directory, `load_backend` builds a staged `TransformerLensReplacementModel`.
When `dictionary_ref` names a `sae-lens` release (with `sae_id`) or a local SAE directory, `load_backend` builds a `HookedTransformer` and stages the `SAE` on CPU before moving it to the target device.

```python
from caliper.interp import Tracer, load_backend

backend = load_backend("google/gemma-2-2b", "gemma")
tracer = Tracer(backend.model, backend.tokenizer, backend.dictionary, backend=backend)
graph = tracer.trace("Fact: the capital of the state containing Dallas is", node_threshold=0.7, edge_threshold=0.9)
```

`backend.dictionary` is a `DictionaryView` with uniform `metadata`, `encode(layer, x)`, and lazy `decoder_vectors(layer, ids)`.
Any `HookedTransformer` architecture in `transformer-lens` and any compatible `sae-lens` or `circuit-tracer` dictionary can attach through `load_backend`.
For offline tests without model downloads, pass `model_ref="tiny-lm"` to construct a deterministic 2-layer `ReplacementModel` and `TranscoderSet`.

## Example Index

`LiveFeatureIndex(backend)` resolves feature summaries and top activating 33-token windows without requiring an offline index build.
When online and `neuronpedia_id` is present on `SAE.cfg.metadata`, `LiveFeatureIndex` queries `get_neuronpedia_feature`.
When offline or using unindexed transcoders, it scans recent and reference prompts with a bounded min-heap and zeros BOS positions.

For large local corpora, `tools/interp/build_index.py` builds a memory-mapped disk index read by `FeatureIndex`:

```sh
python -m tools.interp.build_index --model google/gemma-2-2b --dictionary "$SCOPE_DIR" --output "$INDEX_DIR"
```

`feature.show` returns the heuristic feature label, activation count, maximum, and promoted tokens.
`feature.examples` returns up to 16 ranked 33-token windows and publishes the central 25 slots to `token_strip@1`.

## Attribution Graphs

`Tracer(model, tokenizer, dictionary, backend=backend).trace(prompt)` runs `circuit_tracer.attribute`, transfers the graph to CPU, runs `prune_graph`, and publishes graph tensors.
Prompts accept up to 32 model tokens, including BOS, and refuse longer inputs without truncation.
Default settings select up to 10 logit targets (`desired_logit_prob=0.95`), up to `5,000` feature targets, `node_threshold=0.8`, and `edge_threshold=0.98`.
For the published Gemma 2 2B Dallas graph, set `node_threshold=0.7` and `edge_threshold=0.9`.

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

## Refusals And Limits

`MemoryGuard` enforces a 40 GB process limit (`40,000,000,000` bytes) and an 8 GB host available memory reserve (`8,000,000,000` bytes).
Missing dictionaries, unknown probe points, out-of-range features, and unequal patch prompt lengths return structured refusals.
See [Interpretation Tools](../agents/interpretation.md) for MCP request schemas and paging bounds.
