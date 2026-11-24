# Model Interpretation

Caliper LIB exposes model probes, transcoder dictionaries, feature examples, and attribution graphs.
The model and its dictionary stay in the Python process.
Library pieces draw the graph, example tokens, and next-token probabilities in the app and browser.
The [interpretation tools](../agents/interpretation.md) expose the same data through Caliper MCP.

## Model Probes

`caliper.interp.points(model)` lists the supported probe points.
Decoder blocks expose `resid_pre`, `mlp_in`, `mlp_out`, and `attn_pattern` beside the existing module and head points.
The fixture uses names such as `layers.0.mlp_in`.
The Qwen text model uses names such as `model.layers.0.mlp_in`.

`caliper.interp.capture(model, points)` captures tensors without a CPU copy.
Use it as a context manager around the model call.
The context removes its hooks when it closes.
Qwen DeltaNet patterns represent signed causal mixing coefficients, which can differ from attention probabilities.

`caliper.interp.publish(name, tensor)` exposes a named tensor to a derive program.
For example, a publication named `trace.counts` becomes the source `live:trace.counts`.
A new publication replaces the previous tensor for that name.
The tensor retains its device and storage.
Watch removal clears the model's publications and capture hooks.

## Feature Dictionaries

A dictionary contains one transcoder for each MLP layer.
Each transcoder maps the MLP input to sparse feature activations, then reconstructs the MLP output.
The dictionary folder contains `dictionary.json` and one `layer_NN.safetensors` file per layer.
The manifest records model identity, probe points, dimensions, activation type, text partitions, and measured quality.

`caliper.interp.load_dictionary(path, model=model)` verifies the dictionary against the model.
An explicit `model_hash` can supply the expected identity instead.
The loader refuses an absent identity or a mismatch.
Its `memory_limit_gb` argument caps total process memory at 12 GB by default and refuses larger limits.
Header checks precede allocation, and memory admission includes the model, dictionary, and conversion buffers.
The returned object exposes `encode(layer, x)`, `decode(layer, a)`, and `error(layer, x, y)`.
Its `metadata` contains the manifest, and its `layers` contain the weights.

Each encoder and decoder has shape `(latents, width)`.
TopK encoding applies ReLU, then retains the largest `k` feature activations.
Decoding multiplies the activations by the decoder and adds the decoder bias.
The error is the original MLP output minus the reconstruction.

## Local Resources

The supported Qwen checkpoint is `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`.
It has 24 decoder layers and width 1,024.
Local preparation requires PyTorch 2.15.0 and Transformers 5.24.0.
Place the complete pinned snapshot in the Hugging Face cache before tool execution.

Model and tokenizer files must exist locally. The tools refuse an incomplete model checkpoint.
Tests never download a model or dataset.

The text source is `HuggingFaceFW/fineweb-edu`, configuration `sample-10BT`, split `train`.
Its pinned revision is `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
Training, quality checks, and the example index use separate document partitions.
The manifest records the source and partition rules.

A pinned local dataset manifest can identify the staged Parquet files.
Without that manifest, the tools use the specifically approved public dataset stream at the fixed revision above.
This dataset stream can require network access. Local model files do not imply a local dataset source.

The preparation commands below use local resource folders.
Each full job requires an external resource wrapper, a heavy slot, and an exclusive model slot.
The wrapper must set `nice -n 10` and enforce exclusive GPU access for training.
These requirements are run policy. Portable tools do not contain machine-specific lock paths.
Set `HEAVY_WRAPPER` to the project wrapper, which accepts a job name, a log path, and a command.

```sh
"$HEAVY_WRAPPER" TRAIN-DICTIONARY "$TRAIN_LOG" env PYTHONPATH=python:. python -m tools.interp.train_transcoders \
  --output "$DICTIONARY" --local-dataset-manifest "$DATASET_MANIFEST"
"$HEAVY_WRAPPER" MODEL-QUALITY "$QUALITY_LOG" env PYTHONPATH=python:. python -m tools.interp.eval_dictionary \
  --dictionary "$DICTIONARY" --local-dataset-manifest "$DATASET_MANIFEST"
"$HEAVY_WRAPPER" MODEL-INDEX "$INDEX_LOG" env PYTHONPATH=python:. python -m tools.interp.build_index \
  --dictionary "$DICTIONARY" --local-dataset-manifest "$DATASET_MANIFEST"
```

The trainer defaults to 16,384 latents per layer, TopK with `k=32`, and 40 million tokens.
The context contains 256 tokens.
The process limit is 24 GB, and the wall-time limit is ten hours.

Checkpoints support `--resume`. The periodic checkpoint interval is 240 seconds after each completed save.
A 270-second work interval forces a clean stop. Checkpoint writes consume additional time within the aggregate cap.
A bounded pilot must establish time and memory requirements before a full run.

Quality checks use 200,000 separate tokens.
Mean unexplained variance must be at most 0.30, and the dead-feature fraction must be at most 0.15.
Replacement of all MLPs must increase cross-entropy by at most 0.5 nats.
These thresholds describe acceptance criteria, not a claim that a local dictionary passed.
The example index uses two million other tokens and a 12 GB process limit.

Gemma 2 2B with Gemma Scope transcoders remains a documented fallback.
It requires a compatible converter because its normalization points and encoder orientation differ.
A Qwen dictionary cannot serve that model.

## Example Index

`FeatureIndex` in `caliper.interp._index` reads the index under the dictionary folder.
The index retains the 16 largest activations for each feature, with document IDs and token positions.
It also records activation counts and maxima.
Queries calculate real window activations and retain up to 256 feature results in a cache.

`feature.show` returns the feature label, activation count, maximum, and promoted tokens.
`feature.examples` returns up to 16 examples in descending activation order.
Each example contains 33 token slots, with null padding at document boundaries.
The display uses the central 25 slots and removes boundary padding from each visible prefix.
Two pages expose eight examples each, with stable IDs zero through 15.

## Attribution Graphs

`caliper.interp.Tracer(model, tokenizer, dictionary).trace(prompt)` returns a `Graph`.
The local replacement model fixes attention routing, normalization denominators, and feature masks from the prompt.
Each MLP becomes its transcoder output plus a constant error vector.
Qwen gates and DeltaNet routing factors also remain fixed.
The graph describes effects through these fixed paths, not the causes of attention choices.

Edges represent direct effects between token, feature, error, and logit nodes.
The tracer accounts for biases separately and measures completeness before display pruning.
It reports feature share, error share, and omitted influence.
The graph uses per-layer transcoders, without cross-layer features, supernodes, or global weight analysis.
Feature labels use promoted output tokens and frequent example tokens, without model-generated descriptions.

A prompt contains at most 32 tokens, including special tokens.
The tracer refuses excess tokens without truncation.
Defaults cap feature targets at 512, displayed nodes at 512, and edges at 8,192.
Logit targets cover 95 percent cumulative probability, with at most ten targets.
An eight-entry cache uses the prompt and intervention state as its key.

## Graph Tensors

The tracer publishes fixed tensor shapes on the model device.
Counts define the valid rows, and unused rows contain zeros.
Graph JSON retains labels, prompt tokens, and completeness diagnostics.
`trace.read` pages at most 256 nodes and 256 edges per request.

| Source | Type | Shape | Fields |
|---|---|---|---|
| `live:trace.counts` | `int32` | `(4,)` | Nodes, edges, tokens, layers |
| `live:trace.node_i` | `int32` | `(512,4)` | Kind, layer, position, feature or token ID |
| `live:trace.node_f` | `float32` | `(512,4)` | Activation, influence, top-logit effect, reserved |
| `live:trace.edge_i` | `int32` | `(8192,2)` | Source and target node rows |
| `live:trace.edge_f` | `float32` | `(8192,2)` | Weight and normalized weight |
| `live:trace.label_text` | `uint32` | `(544,16)` | Packed prompt and node labels |

Node kinds are token zero, feature one, error two, and logit three.
Token layers use minus one, and logit layers use the layer count.
Feature and error layers use zero-based indices.
Edges refer to displayed node rows.

## Graph Pieces

`attribution_graph@1` places token positions horizontally and layers vertically.
Bind its `graph_counts` input to `live:trace.counts` and its other structural inputs to the corresponding tensors above.
Signed edges show direct effects, and node picks identify graph rows.
The `selected_node` parameter reads shared state with the same name.
Prompt tokens and selected or hovered labels appear as host text.

`token_strip@1` uses the five `live:examples.*` sources documented in the [example bindings](../agents/interpretation.md#example-panel-bindings).
Its `page` parameter selects zero or one.
Its `activation_max` parameter sets the color scale from the selected feature maximum.
The API retains full labels when display labels exceed 63 UTF-8 bytes.

M7 `logit_bars@1` reads `live:trace.probs`, `live:trace.logit_tokens`, and `live:trace.logit_label_text`.
Both probability rows describe the traced baseline.
These sources do not describe a steered output distribution.
The app and browser use the same piece files.

## Refusals And Limits

Missing models, dictionaries, indexes, and completed graphs produce structured refusals.
Unsupported probe points and incompatible dictionary identities also fail explicitly.
A queued command is not a completed result.
Use its request ID to retrieve the final reply, and use events to wait for progress.
The [command reference](../agents/interpretation.md) specifies the request and reply limits.

The seeded CPU fixture verifies mathematical identities without a downloaded model.
Its reference forward hash is `2464295694bd3690f156dcc7248882f0ad289f55972caafbbd7bd23f4e3d109e`.
Fixture completeness requires a residual at most `1e-9`.
Qwen requires at most `0.02` per target and separate learned-model acceptance evidence.
