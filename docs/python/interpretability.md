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

### Publication Lifetime

A publication belongs to the current watch, or to the next watch if no watch exists.
Completed publications before the first watch remain available when that watch attaches.
Watch removal or replacement clears all publications.
Attachment of the same model without removal preserves the current lifetime.

A trace or example request captures the watch lifetime before model work or resolution of lazy resources.
The request checks that lifetime under the program lock before it publishes a complete tensor group.
The check also requires the request model to match the watched model.
Removal, replacement, or first attachment during the request invalidates its publication lease.
Removal followed by attachment of the same model also invalidates the lease.
A stale request raises `RuntimeError` with code `interpret.watch_changed`, which the interpretation service returns as an error.

Each graph group contains all structural tensors, labels, and probability bars.
Each example group contains tokens, activations, example IDs, valid lengths, and labels.
The producer sees the complete previous group or the complete new group.
A refused publication changes no tensor in the group.
Model work and service waits never hold the program lock.

Watch removal does not cancel model work or wait for a service request.
A result can complete before removal and reach its caller after removal clears the tensors.
Concurrent model mutation and concurrent traces through one tracer remain unsupported.
An explicit `publish` or standalone `Graph.publish` call assigns supplied tensors to the current lifetime.
These explicit calls do not establish the tensors' model identity.

## Feature Dictionaries

A dictionary contains one transcoder for each MLP layer.
Each transcoder maps the MLP input to sparse feature activations, then reconstructs the MLP output.
The dictionary folder contains `dictionary.json` and one `layer_NN.safetensors` file per layer.
The manifest records model identity, probe points, dimensions, activation type, text partitions, and measured quality.

`caliper.interp.load_dictionary(path, model=model)` verifies the dictionary against the model.
An explicit `model_hash` can supply the expected identity instead.
The loader refuses an absent identity or a mismatch.
Its `memory_limit_gb` argument caps total process memory at 12 GB by default and refuses larger limits.
The loader admits metadata before the read and JSON parse.
Metadata has a 1 MiB byte limit and a maximum depth of 64 containers.

Header checks precede tensor allocation, and memory admission includes the model, dictionary, metadata, and conversion buffers.
The returned object exposes `encode(layer, x)`, `decode(layer, a)`, and `error(layer, x, y)`.
Its `metadata` contains the manifest, and its `layers` contain the weights.

An external HF model requires `caliper.interp.verify_hf_model(model, snapshot)` before admission against a pinned HF dictionary.
The local snapshot must contain the checkpoint tensors and `config.json` for the same repository and commit.
Verification compares weights and forward settings with that trusted snapshot, including normalization epsilon, activation functions, rotary settings, and layer types.
The model must use evaluation mode.
Eager and SDPA attention are supported backend choices and preserve the checkpoint identity.
Later identity checks refuse changes to weights or forward settings, including changes before dictionary admission or tracer construction.

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
Mean unexplained variance must be at most 0.40, and the dead-feature fraction must be at most 0.15.
Replacement of all MLPs must increase cross-entropy by at most 0.5 nats.
These thresholds describe acceptance criteria, not a claim that a local dictionary passed.
The example index uses two million other tokens and a 12 GB process limit.

Gemma 2 2B with Gemma Scope transcoders requires a separate converter.
Its normalization points and encoder orientation differ.
This release contains no Gemma converter or fallback model run.
The demo uses the accepted Qwen dictionary. A Qwen dictionary cannot serve Gemma.

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
The selected label follows this value on each draw frame, even when the run is idle.
Prompt tokens and selected or hovered labels appear as host text.

The graph uses `draw.text.selected` for selection and `draw.text.hover` for the local pointer hit.
Both labels use the shown node map and text resource without a new derive pass or model step.
Pointer hover uses the ID readback and has no graph parameter.

`token_strip@1` uses the five `live:examples.*` sources documented in the [example bindings](../agents/interpretation.md#example-panel-bindings).
Its `page` parameter selects zero or one.
Its `activation_max` parameter sets the color scale from the selected feature maximum.
The API retains full labels when display labels exceed 63 UTF-8 bytes.

M7 `logit_bars@1` reads `live:trace.probs`, `live:trace.logit_tokens`, and `live:trace.logit_label_text`.
Both probability rows describe the traced baseline.
These sources do not describe a steered output distribution.
The app and browser use the same piece files.

## Feature Steering

`caliper.watch(model, optimizer=None)` registers inference controls and intervention points.
The hosted inference run also binds its dictionary and tokenizer to the intervention service.
Caliper MCP applies `intervene.set` requests at run boundaries.
The [intervention schemas](../agents/interpretation.md#m8-command-schemas) define their fields and limits.

An `add` intervention writes a vector times its coefficient at the selected point.
A `scale` intervention adds `(multiplier - 1) * activation * decoder` at the feature's MLP output.
A `clamp` intervention adds `(value - activation) * decoder` at that output.
Feature activations come from the current model input at the selected token positions.
The original module and head ablation operations remain available.

An absent `positions` field selects every token, and an empty list selects none.
Optional `from_step` and `until_step` fields bound the active interval, with an exclusive end.
The runtime refuses incompatible dimensions, unknown features, invalid positions, and nonfinite values.
Hook failures remove the failed hook and expose an error in intervention metadata.

Feature requests cannot target layers excluded by an upper-half dictionary.
The default dictionary identifier binds tensor content, actual TopK values, and metadata except the quality report.
Generic state restore refuses a different dictionary under that identifier.
A dictionary change requires resource reconfiguration before another feature request.
An explicit identifier is an alias that the caller supplies.
The caller must verify its restore compatibility.

Device content checks require a supplied process memory guard with a finite limit.
The hosted runner supplies its guard and retains its lower memory limits.

## Activation Patching

A `patch` intervention replaces selected values with values from a clean source prompt.
`source_position` selects one source token when the destination positions need a fixed mapping.
The runtime preserves the current intervention state around clean source capture.
Generation refuses a patch whose implicit source mapping exceeds the clean prompt.

Patch source reuse checks all model parameters and buffers, including alias writes and nonpersistent buffers.
Inference tensors use their contents because they have no version counters.
Each readback uses bounded chunks and the supplied memory guard.

Temporary replay runs on the model's run thread within `internal_forward()`.
The caller must exclude concurrent model forwards from other contexts during that scope.
Other threads can queue control requests for the next run boundary.

`patch.sweep` compares clean and corrupted prompts with equal token counts.
Its metric specifies distinct `positive_token` and `negative_token` IDs.
The sweep patches residual inputs across layers and positions, with at most 32 cells in each batch.
It publishes `patch.grid` and returns normalized recovery from actual model outputs.
The circuit demo uses three panels, and the sweep has separate tests.

`trace.validate` replays the latest trace's intervention snapshot.
It zero-clamps up to ten influential features and measures changes in the centered top-token logit.
The reply contains sign agreement, Spearman correlation, individual measurements, and explicit failure reasons.
Insufficient features and zero variance do not produce a passing result.

## Inference Runs

`run.start` with `kind = "inference"` starts the installed hosted runner.
It requires a local model, dictionary, and prompt.
The model is either `tiny-lm` or the exact pinned Qwen snapshot folder in the Hugging Face cache.
The dictionary folder must also contain its compatible example index.
The loader uses local safe weights and refuses missing or incompatible resources.

The `prompt` parameter accepts text, and `selected_node` accepts a graph row or minus one for no selection.
Selection alone applies no clamp.
The `clamp` parameter ranges from minus four through four times the selected feature's indexed maximum.
It applies at that feature's traced token position.
Zero removes the selected feature contribution.
The `reset` action removes that feature clamp and preserves other intervention kinds.

The `retrace` Boolean enables a new trace after prompt or intervention changes.
The `generate` action returns 24 greedy tokens for each of the baseline and steered outputs.
The text parameters `output.baseline` and `output.steered` expose these results.
Inference checkpoints store resource identities, parameters, selection, and intervention lifecycle state.
They contain no model weights or executable objects.

M8 output bars use the sources below.
Row zero is the baseline, and row one is the steered distribution.
Both rows share ten token IDs, including both actual top tokens.

| Source | Type | Shape |
|---|---|---|
| `live:output.probs` | `float32` | `(2,10)` |
| `live:output.tokens` | `int32` | `(10,)` |
| `live:output.label_text` | `uint32` | `(10,16)` |

## Hidden Demo

The driver is `demos/circuit_trace/run.py` in the source checkout.
It requires the built app, `caliper-mcp`, and the native Python sender and producer modules.
The fixture comparison also requires `caliper_circuit_parity`, the browser WebAssembly files, and the local Chromium headless shell from the web test setup.
`CALIPER_APP_EXE`, `CALIPER_MCP_EXE`, `CALIPER_CIRCUIT_PARITY_EXE`, and `CALIPER_WASM_DIR` can override the corresponding resource paths.

The selected interpreter needs an installed or editable Caliper package because inference launch ignores `PYTHONPATH`.
After the native module build, install the editable package without another native build:

```sh
uv pip install --python .venv/bin/python scikit-build-core watchdog psutil
uv pip install --python .venv/bin/python --no-deps --no-build-isolation \
  -e . --config-settings=wheel.cmake=false
.venv/bin/python -E -P -c \
  "import importlib.util; assert importlib.util.find_spec('caliper.interp._hosted')"
```

Create fixture resources once in an empty task folder:

```sh
PYTHONPATH=python:. .venv/bin/python -m caliper.interp._hosted \
  --fixture-dir "$FIXTURE_DICTIONARY"
```

Set `HEAVY_WRAPPER` to the task's resource wrapper and `DEMO_OUT` to a new folder within task scratch.
Run the hidden fixture after exclusive graphics access becomes available:

```sh
"$HEAVY_WRAPPER" 'GPU MODEL circuit fixture' "${DEMO_OUT}.log" \
  env CALIPER_AGENT_OFFLINE=1 PYTHONPATH=python:. \
  .venv/bin/python demos/circuit_trace/run.py --model tiny-lm \
  --dictionary "$FIXTURE_DICTIONARY" --hidden --out "$DEMO_OUT"
```

For Qwen, set `MODEL` to the pinned snapshot folder and `DICTIONARY` to its prepared dictionary folder.
Use a separate output folder and a `GPU MODEL` wrapper description:

```sh
"$HEAVY_WRAPPER" 'GPU MODEL Qwen circuit trace' "${DEMO_OUT}.log" \
  env CALIPER_AGENT_OFFLINE=1 PYTHONPATH=python:. \
  .venv/bin/python demos/circuit_trace/run.py --model "$MODEL" \
  --dictionary "$DICTIONARY" --hidden --out "$DEMO_OUT"
```

The fixture prompt is `The red cube is beside a tree.`
The Qwen prompt is `Paris is the capital of`.
`--prompt` supplies another prompt within the token limit.
Every control operation uses Caliper MCP, including feature selection, examples, clamps, validation, and window capture.
The driver waits for file events, run events, and published steps.

For `tiny-lm`, the driver sets `CALIPER_FIXTURE_DISPLAY_MPS=1` in its child environment.
This option copies only 18 fixed display tensors to MPS, with a total limit of 256 KiB.
The copies use the shared 12 GB process guard, and a resource stop remains in effect.
The model, dictionary, attribution math, and reference hash stay on the unchanged CPU fixture.
Ordinary CPU fixture runs leave this option disabled.
The driver removes this option for Qwen.

Only the founder uses `--show` in place of `--hidden`.
The visible form retains its MCP event bridge until its own app closes.
A graph pick updates `selected_node`, requests the feature examples, and fills the example strip.
After a trace changes the selected node, the bridge updates the graph state and examples through MCP.
When the run clears selection, the bridge clears the strip.

Enable Automatic Trace for prompt edits that also update the graph.
The visible result records that hidden pixel evidence was not collected and makes no Qwen acceptance claim.

## Acceptance Evidence

The driver saves `result.json` with the graph, command log, event log, example pages, validation result, generated text, and instrument specification.
Hidden runs also save `before.png`, `after.png`, and the second example page capture under the acceptance folder.
A bounded hidden diagnostic reads actual renderer IDs, color pixels, panel image quads, clips, generations, and font references from those captures.

An explicit founder decision can admit one demo dictionary through `--quality-override FILE`.
The JSON record uses version 1 and authority `founder`, with `dictionary_fingerprint`, `evaluated_tokens`, `fvu`, `dead_fraction`, `cross_entropy_delta`, and `failed_gates`.
The fingerprint and measured values must match the live dictionary evidence exactly.
The driver retains failed quality status and displays the override, measurements, and full fingerprint in the instrument and `result.json`.
The override cannot bypass incomplete resources, stale evidence, or causal, pixel, parity, and budget checks.
The hidden test reads the optional record path from `CALIPER_QWEN_QUALITY_OVERRIDE`.

The pixel oracle applies the window origin, framebuffer scale, linear texture samples, and text alpha composition.
It compares the resulting panel pixels with the window PNG and requires visible geometry for every source item.
The PNG/raw equality check establishes only capture consistency.

An independent text reference uses the actual font atlas and expected glyph quads for the prompt and selected labels.
Each required label needs visible ink and pixels within two channel levels of that reference.
Synthetic captures test failures in this oracle. They do not supply application acceptance evidence.

The fixture comparison uses the actual published graph resources with the existing native and browser parity harness.
It exports those resources, closes the hidden app and run, then starts the native and browser comparisons in sequence.
The process exits with code one when a required check fails.

The driver reads optional inference metadata from `trace.status` before and after Generate.
A completion must advance its identity and supply exactly two rows of 24 integer token IDs.
Prompt and intervention hashes must match the current state, and both output texts must match their completion hashes.
The semantic steering check uses the actual comparison argmax IDs from this metadata.

The demo selects the feature with the largest absolute effect on the top logit before the last layer.
The result also records the original influence-based selection and its zero-ablation measurement from `trace.validate`.
Validation retains the ten most influential active features and reports every prediction and measurement.
A positive zero-ablation effect raises the centered logit. The probability-fall and shorter-bar checks still require a decrease.

Each capture also saves actual `program.status` values and the declared piece limits after the resource barrier.
Derive passes have limits of 50 ms and 64 MiB, while draw frames have limits of 4 ms and 64 MiB.
The result checks these budgets before and after the clamp.
Missing GPU times remain unverified and fail the required check.

Qwen acceptance requires feature share of at least 0.30, between 20 and 512 nodes, and a token path to the top logit.
It also requires 16 positive sorted examples, at least eight correct signs among ten features, and Spearman correlation of at least 0.6.
The top-token probability must fall by at least 0.05, or the top token must change.
Every target residual must remain at most 0.02.
The trace limit is 20 seconds, the steering limit is one second, and the process limit is 12 GB.

macOS process checks use the peak memory footprint from `/usr/bin/time`, which includes unified memory.
Linux checks use maximum resident set size.
Missing high-water measurements fail acceptance, and the native parity process has the same 12 GB limit.

Pixel checks require all visible node IDs and bar heights within two pixels of probability times scale.
The top bar must become shorter after the Qwen clamp.
Fixture parity requires exact interior IDs, at most two color levels of interior difference, and at most one percent differing edge pixels.
App/browser parity applies to the fixture graph. Qwen retains both its semantic steering check and its strict shorter-bar check.

The macOS and Windows fixture workflows also run isolated inference entry checks.
These checks use the production entry helper with a harmless hosted substitute.
They establish argument and environment transfer, including fork metadata and prompt removal from process arguments.
They do not establish production package attachment, checkpoint restoration, shutdown, or native Windows inference.

Fixture results separate mathematical and display checks from learned-model semantics.
The unchanged fixture requires a positive measured clamp effect and records the stricter steering result without a Qwen acceptance claim.
It retains the sign and rank thresholds from validation.
A probability change smaller than one pixel can leave the fixture bar height unchanged.
The selected fixture examples have 13 valid tokens and 12 padding slots in each 25-slot display row.
The two pages retain all 16 example IDs and verify each token slot against the API response.

The normal fixture test is `tests/demos/test_circuit_trace.py::test_hidden_fixture` in Native Build CI.
Missing Python modules or resources fail setup, while verified absence of Metal permits a platform skip.
The opt-in `local_model` test uses the same driver with `CALIPER_QWEN_MODEL` and `CALIPER_QWEN_DICTIONARY`.
That test runs only outside CI, with explicit local paths and no downloads.

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
