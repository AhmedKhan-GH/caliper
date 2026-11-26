# Circuit Tracing

## Purpose

M7 and M8 connect Caliper live instruments to `transformer-lens`, `circuit-tracer`, and `sae-lens`.
Python runs attribution, sparse dictionary encoding, feature steering, and activation patching on the model device.
The native app and the browser viewer render attribution graphs, feature examples, and output probability bars.
Each feature intervention tests the causal effect that the attribution graph predicts.

## Verified Base

Branch `interp-tl` replaces custom M7 math and offline transcoder trainers with pinned upstream libraries.
Caliper retains its GPU pieces, wire v3 codecs, MCP tools, hosted runs, and live controls.

### Model And Dictionary

`caliper.interp.load_backend` loads models through `HookedTransformer` or `TransformerLensReplacementModel`.
Disable weight folding and centering (`fold_ln=False`, `center_writing_weights=False`, `center_unembed=False`) to preserve forward logits.
For local checkpoints, load the Hugging Face weights once, convert them into TransformerLens, and release the first copy.
On Apple Silicon, stage model construction before transcoder attachment so unified memory stays within hardware limits.

`circuit-tracer` loads per-layer transcoders as a `TranscoderSet` (`load_transcoder_from_hub` or `load_transcoder_set`).
`sae-lens` loads public sparse autoencoders and transcoders through `SAE.from_pretrained` and `SAE.load_from_disk`.
On Apple Silicon, stage `safetensors` reads on CPU before moving each layer to `mps`.
The reference benchmark uses `google/gemma-2-2b` (`c5ebcd40`) with 26-layer 16k Gemma Scope transcoders (`50eec2f2`) in `bfloat16`.

### Existing Interfaces

`caliper.interp.points` lists module, head, and semantic probe points (`resid_pre`, `mlp_in`, `mlp_out`, and `attn_pattern`).
`caliper.interp.capture` records probe tensors on the model device without a CPU copy.
`caliper.interp.publish` registers named tensors for derive programs under `live:` catalog sources.
Watch removal clears published tensors and temporary hooks for that model.

The intervention commands `intervene.points`, `intervene.set`, `intervene.clear`, and `intervene.list` use wire CONTROL `intervene`.
They support module and head `ablate`, vector `add`, feature `scale` and `clamp`, and activation `patch`.
Text parameters bind `param:prompt` and `param:output.*` fields to app and browser controls.
`run.start` with `kind = "inference"` launches `caliper.interp._hosted`, and `run.fork` preserves active interventions in `.json` metadata.

### Text And Registry

A piece text table holds at most 4,096 strings, with at most 64 UTF-8 bytes per string.
The host label layout places at most 512 labels per panel.
Thus `token_strip@1` displays eight example rows of 25 tokens per page across two pages.
`feature.examples` returns up to 16 examples with full 33-token windows in structured JSON.
Long display labels truncate at UTF-8 character boundaries while JSON responses keep complete text.

| Feature | Dependencies |
|---|---|
| INT-35 Model Probes | INT-34 |
| INT-36 Feature Dictionaries | INT-35 |
| INT-37 Example Index | INT-36 |
| INT-38 Attribution Graphs | INT-35, INT-36 |
| INT-39 Graph Pieces | INT-38 |
| INT-40 Feature Steering | INT-34, INT-36 |
| INT-41 Activation Patching | INT-35, INT-40 |
| INT-42 Inference Runs | INT-38, INT-39, INT-40, INT-41 |

## Dictionary Contract

`DictionaryView` wraps a `circuit-tracer` `TranscoderSet` or one or more `sae-lens` `SAE` modules behind uniform metadata.
Its `metadata` dictionary records `dictionary_id`, `kind`, `layers`, `covered_layers`, `latents`, `width`, `read_points`, `write_points`, and `neuronpedia_id`.
Lazy decoder row lookup calls `_get_decoder_vectors` so hosted runs never materialize all 26 full decoder matrices at once.
Incompatible layer indices or out-of-range feature indices fail with structured refusals.

`FeatureIndex` reads an optional prebuilt disk index (`tools/interp/build_index.py`) with memory-mapped top-16 feature windows.
When no disk index exists, `LiveFeatureIndex` queries Neuronpedia (`get_neuronpedia_feature`) when online or scans local prompts with a bounded min-heap.
It zeros BOS activations during local ranking so document-start tokens do not dominate feature examples.
`MemoryGuard` enforces a 40 GB process limit (`40,000,000,000` bytes) and an 8 GB host reserve (`8,000,000,000` bytes).

## Replacement Model

Attribution delegates directly to `circuit_tracer.attribute`, `prune_graph`, `compute_graph_scores`, and `compute_node_influence`.
`TransformerLensReplacementModel` freezes attention patterns and normalization scales from the clean prompt pass.
Each MLP becomes its active transcoder features plus an explicit detached error term.
Logit targets use demeaned unembedding vectors for up to `max_n_logits` tokens up to `desired_logit_prob` cumulative probability.

On Apple Silicon, `attribute` runs with `batch_size = 256` and per-batch Objective-C autorelease pool drains.
After `attribute` finishes, Caliper transfers `Graph` tensors, `input_tokens`, and `activation_values` to CPU before `prune_graph`.
`prune_graph` applies cumulative `node_threshold` and `edge_threshold` masks and preserves `replacement_score` and `completeness_score`.
No custom VJP loop or dense Jacobian assembly exists in Caliper.

## Tensor Contract

`Tracer.trace` builds a `Graph` and publishes fixed-capacity tensors on the model device.
Compact traces up to 512 nodes and 8,192 edges use `(512, 8192)` buffers.
Larger traces such as Gemma 2 2B Dallas (`847` nodes and `33,999` edges) use `(8192, 262144)` buffers.
`trace.counts` defines the valid prefix rows, and unused rows contain zeros.

| Tensor | Shape | Type | Fields |
|---|---|---|---|
| `trace.counts` | `(4,)` | int32 | Nodes, edges, tokens, layers |
| `trace.node_i` | `(N, 4)` | int32 | Kind, layer, position, feature or token ID |
| `trace.node_f` | `(N, 4)` | float32 | Activation, influence, top-logit effect, reserved |
| `trace.edge_i` | `(E, 2)` | int32 | Source row, target row |
| `trace.edge_f` | `(E, 2)` | float32 | Weight, weight divided by largest magnitude |

Node kinds are token `0`, feature `1`, error `2`, and logit `3`, where `N` is `512` or `8192` and `E` is `8192` or `262144`.
Token nodes use layer `-1`, logit nodes use the model layer count, and feature and error nodes use zero-based layer indices.
`trace.label_text` has shape `(32 + N, 16)` of packed `uint32` words.
`Tracer` caches up to 16 graphs keyed by prompt, active interventions, thresholds, and label index identity.

## Pieces And Commands

The library pieces are `attribution_graph@1`, `token_strip@1`, `logit_bars@1`, and `heat_grid@1`.
`attribution_graph@1` accepts up to `8,192` nodes and `262,144` edges across seven derive passes (`50 ms` budget) and two draw passes (`16 ms` budget).
Shared state `selected_node` highlights the chosen node and dims unrelated edges in both Metal and WebGPU.
The native app and the browser viewer load identical WGSL piece definitions.

The MCP interpretation commands are `feature.dictionary`, `feature.show`, `feature.examples`, `trace.run`, `trace.status`, `trace.read`, `patch.sweep`, and `trace.validate`.
Read commands use the `read` effect, while `trace.run`, `patch.sweep`, and `trace.validate` use the `run` effect.
Long commands return a `request_id` immediately and execute on the run thread at the next step boundary.

## Intervention Contract

Transcoder single-feature and group edits call `ReplacementModel.feature_intervention` with `(layer, position, feature, value)` tuples.
Cross-prompt donor edits read clean activations with `ReplacementModel.get_activations` on donor prompts and pass scaled targets to `feature_intervention`.
Standalone `HookedTransformer` SAEs apply edits in delta form: `x + (sae.decode(modified) - sae.decode(base))`.
Delta form gives zero reconstruction error when the edit equals the baseline activation.

`Output.compare` publishes baseline row `0` and intervened row `1` as `output.probs` `(2, 10)`, `output.tokens` `(10,)`, and `output.label_text` `(10, 16)`.
Both baseline and intervened argmax token IDs remain in the ten bar slots even when a distribution shifts sharply.
`Output.generate` emits 24 greedy baseline and steered tokens via `feature_intervention_generate` or `HookedTransformer.generate` while preserving permanent hooks.
`patch.sweep` tiles `hook_resid_pre` patches in batches of at most 32 cells and publishes normalized recovery as `patch.grid`.
`trace.validate` zero-ablates up to ten top features and reports Spearman rank correlation and sign agreement.

## Validation And Delivery

Offline unit tests use a deterministic 2-layer `ReplacementModel` (`tiny-lm`) and a 2-layer `StandardSAE` fixture without network downloads.
End-to-end acceptance runs `demos/circuit_trace/run.py` through `caliper-mcp` on `google/gemma-2-2b` (`Fact: the capital of the state containing Dallas is`).
Caliper compares direct `circuit-tracer` outputs against `REFERENCE-RESULTS-12.json` on identical inputs:

- Baseline `' Austin'` (`22605`, probability `0.419921875`, logit `26.125`) and all seven published Dallas intervention top tokens (Cells 14, 16, 18, 20, 24, 26, and 29) match.
- Six of seven intervention probabilities match bitwise, Cell 26 stays within one `bfloat16` ULP (`2^-10`) from Metal `index_add_` order, and Cell 29 has `Say Victoria == 0.0`.
- Attribution graph counts match (`5,307` full nodes, `847` kept nodes, `689` kept features, and `33,999` kept nonzero edges at thresholds `0.7` and `0.9`).
- `attribution_graph@1`, `token_strip@1`, and `logit_bars@1` render in both the native Metal app and the browser WebGPU viewer.
- Load, trace, prune, intervention, generation, and sweep durations are measured and reported without arbitrary time gates.
- Process memory stays within 40 GB and host available memory stays at or above 8 GB.
