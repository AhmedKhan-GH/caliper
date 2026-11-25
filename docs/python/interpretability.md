# Model Interpretation

Caliper LIB exposes model probes, transcoder dictionaries, feature examples, and attribution graphs.
The model and its dictionary stay in the Python process.
Library pieces draw the graph, example tokens, and next-token probabilities in the app and browser.
The [interpretation tools](../agents/interpretation.md) expose the same data through Caliper MCP.

Supported decoder families are the seeded fixture, Gemma 2, and experimental Qwen3.5 text.
Generic Llama models are unsupported.
Gemma 2 2B is the only M8 acceptance model, and M8 remains held.
Its target is: reproduces the published circuit-tracer Gemma 2 2B Dallas example.
Release requires the published comparison, all product checks, and platform CI to pass.

## Model Probes

`caliper.interp.points(model)` lists the supported probe points.
Decoder blocks expose `resid_pre`, `mlp_in`, `mlp_out`, and `attn_pattern` beside the existing module and head points.
The fixture uses names such as `layers.0.mlp_in`.
Qwen and Gemma use names such as `model.layers.0.mlp_in`.

Gemma `mlp_in` is the actual GeGLU input after the pre-feedforward norm and its learned gain.
Gemma `mlp_out` is the post-feedforward norm output at the residual sum.
It differs from the raw `.mlp` output.
The [Gemma Scope reference](../reference/gemma-scope.md) defines these coordinates and their dictionary conversion.

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
Its `memory_limit_gb` argument caps total process memory at 12 GB by default.
Only the exact pinned Gemma identity permits an explicit larger limit, up to 40 GB.
The optional `memory` argument retains the supplied guard, its lower limit, and any prior stop.

The loader admits metadata before the read and JSON parse.
Metadata has a 1 MiB byte limit and a maximum depth of 64 containers.

Header checks precede tensor allocation, and memory admission includes the model, dictionary, metadata, and conversion buffers.
The returned object exposes `encode(layer, x)`, `decode(layer, a)`, and `error(layer, x, y)`.
Its `metadata` contains the manifest, and its `layers` contain the weights.

An external HF model requires `caliper.interp.verify_hf_model(model, snapshot)` before admission against a pinned HF dictionary.
The local snapshot must contain the checkpoint tensors and `config.json` for the same repository and commit.
Verification compares weights and forward settings with that trusted snapshot, including normalization epsilon, activation functions, rotary settings, and layer types.
The local snapshot is trusted input. Verification does not authenticate its files against the remote repository.
The model must use evaluation mode.
Set `memory_limit_gb=40` at initial Gemma verification when the complete run needs that budget.
Later callers cannot raise an earlier verification limit or clear its recorded stop.
Qwen permits eager and SDPA attention with the same checkpoint identity.
Gemma requires eager attention because the supported SDPA backend cannot apply its attention softcap.
Later identity checks refuse changes to weights or forward settings, including changes before dictionary admission or tracer construction.

For default Qwen3.5 text and Gemma 2 RoPE, the verifier reconstructs buffers within a fixed CPU allocation limit.
It also reconstructs the Gemma embedding-scale scalar.
It refuses other nonpersistent buffers and nondefault RoPE modes.

Each encoder and decoder has shape `(latents, width)`.
TopK encoding applies ReLU, then retains the largest `k` feature activations.

JumpReLU encoding keeps a preactivation only when it exceeds its positive learned threshold.
Equality produces zero, and JumpReLU uses `k=None`.
JumpReLU files include `threshold` beside `W_enc`, `b_enc`, `W_dec`, and `b_dec`.
Identity and evaluation evidence include threshold bytes.
Decoding multiplies the activations by the decoder and adds the decoder bias.
The error is the original semantic MLP output minus the reconstruction.

## Local Resources

The supported Qwen checkpoint is `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`.
It has 24 decoder layers and width 1,024.
Local preparation requires PyTorch 2.15.0 and Transformers 5.24.0.
Place the complete pinned snapshot in the Hugging Face cache before tool execution.

Model and tokenizer files must exist locally. The tools refuse an incomplete model checkpoint.
Tests never download a model or dataset.

The text source is `HuggingFaceFW/fineweb-edu`, configuration `sample-10BT`, split `train`.
Its pinned revision is `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
Qwen training, quality checks, and the example index use separate document partitions.
The manifest records the source and partition rules.
Imported Gemma dictionaries bind local evaluation data through a separate `evaluation_source` record.
Their publisher training provenance and overlap remain unknown.

A pinned local dataset manifest can identify the staged Parquet files.
Without that manifest, the tools use the specifically approved public dataset stream at the fixed revision above.
This dataset stream can require network access. Local model files do not imply a local dataset source.

Local Parquet input supports flat UTF-8 text with uncompressed or SNAPPY pages.
Each decode uses one row and checks process and host memory.
Admission requires at least 8 decimal GB of available host memory after the requested decode allocation.
The allocation must also fit the remaining process budget.
Historical peak checks and pressure stop thresholds remain active.
Other codecs and unbounded decode units are refused before data allocation.

The Qwen preparation commands below document supported interfaces with local resource folders.
Further Qwen GPU work and acceptance attempts remain shelved.
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

The Qwen trainer defaults to 16,384 latents per layer, TopK with `k=32`, and 40 million tokens.
The context contains 256 tokens.
The process limit is 24 GB, and the wall-time limit is ten hours.
The memory guard sets the MPS allocation fraction for the whole process.
Use a separate process when another workload requires a different MPS limit.

Checkpoints support `--resume`. The periodic checkpoint interval is 240 seconds after each completed save.
A 270-second work interval forces a clean stop. Checkpoint writes consume additional time within the aggregate cap.
A bounded pilot must establish time and memory requirements before a full run.

Quality checks use 200,000 separate tokens.
The Qwen mean unexplained variance limit is 0.40, and the dead-feature fraction limit is 0.15.
Replacement of all MLPs must increase cross-entropy by at most 0.5 nats.
These thresholds describe acceptance criteria, not a claim that a local dictionary passed.
The Qwen example index uses two million other tokens and a 12 GB process limit.
Index version 2 records the `live_input_v1` encoder policy in its manifest hash.
It also binds model tensor precision, device, effective forward settings, autocast, and float32 matmul precision.
Rebuild the index when these settings change, including a move between CPU and MPS.
The loader rejects missing or different evidence before it opens data maps.
Version 1 remains readable when all model and encoder floating tensors use the same dtype.
Rebuild other version 1 indexes because their stored activations use encoder storage precision.

## Gemma Resources

The Gemma model is `google/gemma-2-2b` at revision `c5ebcd40d208330abc697524c919956e692655cf`.
Its 26 layers have width 2,304.
The official model requires gated access, and its complete pinned snapshot must exist locally before use.
The [Gemma Scope reference](../reference/gemma-scope.md) records every source pin, selected L0 value, and license.

`tools/interp/convert_gemma_scope.py` imports the selected 16,384-feature JumpReLU transcoders without training.
The ordinary `caliper` profile transposes the source encoder and divides input columns by the learned gain.
The ordinary hosted Gemma path retains F32.
The decoder writes after the post-feedforward norm without another normalization.
A Qwen dictionary cannot serve Gemma.

The explicit `circuit_tracer_82996f14` profile uses an original-coordinate BF16 artifact.
`W_enc_original` is the direct source encoder transpose cast to BF16.
Reference hooks capture the normalized input before the learned gain, without division of rounded inputs after the gain.
The runtime casts source F32 norms to BF16, then promotes them to F32 for gain addition and casts the gain to BF16.
Conversion and qualification bind all source bytes and the exact version 2 conversion schema.
The [reference contract](../reference/gemma-scope.md#conversion-and-provenance) defines the artifact and qualification boundaries.

The published index requires exactly 2,000,000 source positions and 256 model positions per context, with BOS.
It uses original coordinates, BF16 reference arithmetic, and the complete artifact fingerprint and execution identity.
Format version 2 requires `encoder_policy="gemma_scope_pre_gain_v1"`.
Changes to source bytes, arithmetic, dtype, device, or execution settings require a new index.
Ordinary indexes retain their compatible policies and limits.

The historical F32 research comparison contains 52,231 source positions and 52,174 metric observations.
It retains `status="partial"`, `source_budget_complete=false`, and stop reason `signal 2` against a 200,000-position resume budget.
Founder Decision 8 and Coordinator Decision 10 define this result as separate comparison evidence.
It cannot qualify the new BF16 artifact or index, and the published profile adds no quality gate.
The generic `research_v1` protocol retains its sample, identity, context, and per-layer loss checks.
Qwen CPU protocol paths remain supported, but further Qwen GPU runs, regrading, and acceptance attempts remain shelved.

Independent arithmetic checks use actual TransformerLens conversion with small CPU models.
They cover F32 and BF16, contexts of 33, 256, and 1,024 positions, and right padding at 256 positions.
These checks do not prove full model parity, device memory, trace time, or action time.
The explicit Gemma process limit remains 40 decimal GB, with an 8 decimal GB host reserve.
The Qwen quality override does not apply to Gemma.

## Example Index

`FeatureIndex` in `caliper.interp._index` reads the index under the dictionary folder.
The model and every dictionary tensor must use the same device at build and load admission.
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
Supply `index=feature_index` to include heuristic feature labels in standalone traces.
The index must use the same model, tokenizer, and dictionary objects.
Without an index, feature labels contain the layer and feature IDs.
After a standalone trace, call `caliper.step()` once to derive and send its published tensors.
The hosted `trace.run` command performs this work at its existing run boundary.

The local replacement model fixes attention routing, normalization denominators, and feature masks from the prompt.
Each MLP becomes its transcoder output plus a constant error vector.
Qwen gates and DeltaNet routing factors also remain fixed.
The graph describes effects through these fixed paths, not the causes of attention choices.

Gemma attention capture preserves its softcap and alternating sliding-window masks before it fixes the probabilities.
The final-logit softcap uses a local affine tangent with the captured value and derivative.
Its centered constant enters completeness accounting.
Live output and interventions retain the actual nonlinear softcap.

`Tracer` defaults to `memory_limit_gb=12`.
An explicit Gemma trace can use at most 40 GB, and hosted Gemma inference supplies that limit.
Other supported families retain the 12 GB ceiling.

The hosted trace and validation paths use inference forwards without autograd or autocast.
Mixed dictionary storage uses the model input precision for feature arithmetic.
Encoder conversion uses bounded blocks during these inference forwards.
This temporary conversion bound does not cover retained autograd tensors in an external training forward.
Exact feature identity at a numerical tie can depend on the device kernel and tensor shape.
The completeness and stale-feature checks retain their existing tolerances.

Edges represent direct effects between token, feature, error, and logit nodes.
The tracer accounts for biases separately and measures completeness before display pruning.
It reports feature share, error share, and omitted influence.
The graph uses per-layer transcoders, without cross-layer features or global weight analysis.
Ordinary feature labels use promoted output tokens and frequent example tokens, without model-generated descriptions.
The published Dallas profile also exposes the exact source groups and labels.

A prompt contains at most 32 tokens, with special tokens.
The tracer refuses excess tokens without truncation.
Ordinary calls cap feature targets at 512, displayed nodes at 512, and edges at 8,192.
Logit targets cover 95 percent cumulative probability, with at most ten targets.
An eight-entry cache binds the prompt, intervention state, profile, trace settings, and feature-label index identity.
A different label index preserves compatible forward activations but requires a new graph identity.

### Reference Profile

`Tracer.trace` accepts the explicit profile `circuit_tracer_82996f14` beside the ordinary `caliper` profile.
The published Dallas request supplies 5,000 feature targets, ten logits, and cumulative probability 0.95.
It sets node threshold 0.7, edge threshold 0.9, batch size 512, and update interval 4.
Node and edge ceilings are 8,192 and 262,144.
An excess causes refusal without truncation of the pruned result.
The [reference graph contract](../reference/gemma-scope.md#reference-graph) distinguishes published parameters from unpublished reconstruction choices.

The reference objective centers unembedding directions before the final softcap derivative.
Its projection excludes BOS feature and error sources before influence normalization.
Separate Caliper diagnostics retain completeness over all sources.
The complete graph remains available through `trace.read`, with at most 256 records of each kind per page.
Ordinary calls retain their lower limits, objective, and default batch size of 64.

## Graph Tensors

The tracer publishes tensor shapes fixed by the profile on the model device.
The table below describes ordinary calls.
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

The reference profile uses 8,192 node rows, 262,144 edge rows, and 8,224 label rows.
Counts and field layouts retain their ordinary form.

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
A `scale` intervention adds `(multiplier - 1) * activation * decoder` at the feature's semantic MLP output.
A `clamp` intervention adds `(value - activation) * decoder` at that output.
Feature activations come from the current model input at the selected token positions.

Gemma reads the actual MLP input after input edits and writes the feature delta after the post-feedforward norm.
The tracer refuses Gemma edits at the raw `.mlp` output.
The [ablation reference](../reference/trace-ablation-support.md) lists the supported points by family.
The original module and head ablation operations remain available.

An absent `positions` field selects every token, and an empty list selects none.
Optional `from_step` and `until_step` fields bound the active interval, with an exclusive end.
The runtime refuses incompatible dimensions, unknown features, invalid positions, and nonfinite values.
Runtime hook failures remove the failed hook and expose an error in intervention metadata.

A refused replacement retains its previous live hook and applied specification.
The record stays `active`, and `error` and `error_code` describe the refused request.
Normal expiration clears these refusal fields. A terminal runtime failure retains its error.
Failure notices run after hook and source-cache cleanup, and each runtime failure produces one notice.

Lifecycle snapshots distinguish `specification`, the requested effect, from `applied_specification`, the saved applied effect.
`load_state_dict()` restores metadata by default. The next boundary admits the requested effect.
A saved applied specification without a live hook does not provide a fallback when that request fails.
`load_state_dict(..., restore_active=True)` performs bounded admission and installs saved applied effects before return.
It preserves queued replacements for a later boundary.

A conflicting scheduled preparation rolls back its new hooks and preserves current requests.
Queued actions defer after rollback and keep their original deadlines.
A failed new hook can end without action deferral when earlier live effects need no rollback.
Interpretation work can read committed effects after cleanup, even when actions defer.
Its snapshot describes applied effects, which can differ from queued requests.

Feature requests cannot target layers excluded by an upper-half dictionary.
The default dictionary identifier binds tensor content, activation type, optional TopK values, and metadata except the quality report.
JumpReLU identity includes every threshold.
Generic state restore refuses a different dictionary under that identifier.
A dictionary change requires resource reconfiguration before another feature request.

An explicit identifier is an alias that the caller supplies.
The caller must verify its restore compatibility.

Device content checks require a supplied process memory guard with a finite limit.
The hosted runner supplies its guard and enforces the model family limit.

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

The generic `trace.validate` diagnostic replays the latest trace's intervention snapshot.
Its sign and rank results do not define published Dallas acceptance.
It zero-clamps up to ten influential features and measures changes in the centered top-token logit.
The reply contains sign agreement, Spearman correlation, individual measurements, and explicit failure reasons.
Insufficient features and zero variance do not produce a passing result.

## Inference Runs

`run.start` with `kind = "inference"` starts the installed hosted runner.
It requires a local model, dictionary, and prompt.
The model is `tiny-lm` or an exact pinned Qwen or Gemma snapshot folder in the Hugging Face cache.
The dictionary folder must also contain its compatible example index.
The loader uses local safe weights and refuses missing or incompatible resources.

### Published Controls

An explicit published inference request pairs `replication_profile="published_dallas_v1"` with `private_output_directory`.
The caller must first create a unique local private directory.
Script requests reject both fields, and inference requests require the pair.
The hosted profile selects the pinned Gemma model, qualified reference arithmetic, BF16 artifact, and compatible original-coordinate index.
It refuses missing or incompatible sources without a profile fallback or download.

The exact prompt is `Fact: the capital of the state containing Dallas is`.
The [published group table](../reference/gemma-scope.md#published-dallas-profile) lists the seven interventions, source labels, and expected top tokens.
Each group action uses `run.action` with `published_dallas_v1.cellN`.
Cell 10 restores the baseline, and each other action replaces the previous complete case atomically.
A failed member restores the prior state and returns one terminal failure.
The controls show actual outputs from fresh model execution.

Preparation captures baseline and donor activations from independent, unchanged prompts before action timing.
It retains baseline attention patterns while actions recompute norms and MLPs.
Absolute clamps use those prepared activations, never index maxima or an earlier case's modified activations.
Full vocabulary evidence stays in the explicit private directory, outside MCP replies and the public repository.
The published profile permits selected rows through 8,191, while ordinary runs retain 511.

### Ordinary Controls

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
`CALIPER_WEB_DIR` selects the WebAssembly files that the app serves to browsers.
The driver uses `CALIPER_WASM_DIR` when `CALIPER_WEB_DIR` is absent.

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

For published Gemma collection, set `MODEL` and `DICTIONARY` to the qualified pinned resources.
Create a unique private directory named by `PRIVATE_OUTPUT` before launch.
Select an empty output folder beneath it:

```sh
"$HEAVY_WRAPPER" 'GEMMA GPU MODEL published Dallas' "$DEMO_LOG" \
  env CALIPER_AGENT_OFFLINE=1 PYTHONPATH=python:. \
  .venv/bin/python demos/circuit_trace/run.py --model "$MODEL" \
  --dictionary "$DICTIONARY" --replication-profile published_dallas_v1 \
  --private-output-directory "$PRIVATE_OUTPUT" \
  --hidden --out "$PRIVATE_OUTPUT/driver"
```

Both profile and private output arguments are required.
The published profile fixes the exact lowercase Dallas prompt and refuses `--quality-override`.
It selects a 40 decimal GB process cap and retains the 8 decimal GB host reserve.
Collection uses Caliper MCP for the full reference-profile trace, baseline, and all seven published actions.
A fresh baseline precedes each intervention.
The driver waits for terminal events and complete output publication.

The fixture prompt remains `The red cube is beside a tree.`
Ordinary model demos retain `--prompt` and the default Dallas prompt within the 32-token limit.
Their generic diagnostics do not establish published replication.
The published profile permits no replacement prompt or invented confidence threshold.

For `tiny-lm`, the driver sets `CALIPER_FIXTURE_DISPLAY_MPS=1` in its child environment.
This option copies only 18 fixed display tensors to MPS, with a total limit of 256 KiB.
The copies use the shared 12 GB process guard, and a resource stop remains in effect.
The model, dictionary, attribution math, and reference hash stay on the unchanged CPU fixture.
Ordinary CPU fixture runs leave this option disabled.
The driver removes this option for Qwen and Gemma.

Only the founder uses `--show` in place of `--hidden`.
This form prepares the baseline instrument without the acceptance batch.
`--hidden --demo` uses the same interactive path without a visible window.
The interactive path checks the browser asset routes before it creates the graph.
Hidden startup proof verifies the served assets, program views, retained attachment, and visible error retention.
These results establish launch and display reliability, not published replication or product acceptance.
It records `demo_ready` and makes no acceptance claim.
The visible form retains its MCP event bridge until its own app closes.
A setup error stays visible while the app remains open, including errors before the bridge exists.
A graph pick updates `selected_node`, requests the feature examples, and fills the example strip.
After a trace changes the selected node, the bridge updates the graph state and examples through MCP.
When the run clears selection, the bridge clears the strip.

Enable Automatic Trace for prompt edits that also update the graph.
The visible result records that hidden pixel evidence was not collected and makes no learned-model acceptance claim.

## Acceptance Evidence

Published evidence collection and independent acceptance review are separate steps.
The hidden collection first measures a bounded preparation and baseline, then collects the complete graph and all seven actions.
Live `--demo` and founder `--show` modes prepare the interactive instrument without that acceptance batch.
A live `demo_ready` result establishes no acceptance claim.

### Published Comparison

The actual pinned circuit-tracer run completed all seven cases and canonical CPU pruning.
All seven top tokens match the published outcomes, but most rounded probability lists differ.
Cell 26 matches its rounded list.
The [reference results](../reference/gemma-scope.md#reference-results) state the graph counts and distinguish source-library scores from Caliper completeness.
The comparison must use these actual full reference outputs and graph, not rounded publication labels alone.

Feature-node Jaccard overlap must reach 0.9.
Every union edge must satisfy `abs(caliper - reference) <= 1e-6 + 1e-3 * abs(reference)`.
A missing edge counts as zero.
Baseline and intervention probabilities must agree across the full vocabulary within 0.001 absolute, with exact top token IDs.
Both implementations must reproduce every published semantic outcome.
The ineffective British Columbia intervention remains part of that requirement.

The published path requires completeness residuals at most 0.02 per target and trace time at most 20 seconds.
Each complete intervention action must take at most one second.
This duration includes model execution, validation, export, publication, and terminal checks.
Preparation time remains separate, and cached outputs cannot satisfy this limit.
The process must remain within 40 decimal GB and preserve 8 decimal GB of available host memory.
Historical five-point clamps, 8/10 signs, Spearman 0.6, and largest-effect selection do not decide published Gemma acceptance.

### Collection And Review

The driver saves `result.json`, `collection.json`, complete graph pages, action records, example pages, captures, and resource identities beneath the private root.
Private action files retain full vocabulary logits and probabilities as exact F32 casts of native values.
Metadata retains native dtype and exact source and case identities.
Collection checks fresh output sequences and hashes every required file.
An exit of zero means collection completed, while `accepted_replication` remains false.
Measured failures remain in the collection and cannot become optional checks.

After independent comparison and capture inspection, the lead supplies a review file beneath the same private root.
The review binds the collection hash to the numerical, browser, capture, parity, and memory evidence.
Use a new output file for final verification:

```sh
.venv/bin/python demos/circuit_trace/_replication_review.py \
  --private-output-directory "$PRIVATE_OUTPUT" \
  --collection driver/collection.json --review review.json \
  --out "$PRIVATE_OUTPUT/verified-result.json"
```

The verifier rechecks every file hash, case identity, required check, and external review metric.
Only a successful review sets `accepted_replication=true` and `acceptance_complete=true`.
Resource qualification and synthetic review fixtures cannot establish real acceptance.
M8 remains held until all checks, independent reviews, and platform CI pass.

### Visual Evidence

Native and browser captures must show the graph, published labels, group controls, examples, and actual output bars.
Every visible graph node needs its expected ID, and bar heights must match current probabilities within two pixels.
Both example pages retain eight rows and 16 distinct examples in total.
The pixel oracle uses actual image quads, clips, framebuffer scale, texture samples, and text alpha composition.
PNG/raw equality establishes capture consistency only.

An independent text reference uses the actual font atlas and expected glyph quads for prompt and selected labels.
Every required label needs visible ink and pixels within two channel levels of that reference.
App/browser parity requires exact interior IDs, at most two color levels of interior difference, and at most one percent unmatched edge pixels.
The published path requires independent parity evidence for its actual resources.
Synthetic captures test oracle failures but do not establish product acceptance.

Each capture records actual `program.status` values and declared piece limits after the resource barrier.
Derive limits remain 50 ms and 64 MiB, while draw limits remain 4 ms and 64 MiB.
Missing GPU times leave required checks unverified.
Browser proof requires available asset routes, visible program views, and attachment before and after the capture.
The verified startup repair does not replace these checks for the published collection.

macOS process checks use peak memory footprint from `/usr/bin/time`, with unified memory.
Linux checks use maximum resident set size.
Missing high-water or host-reserve evidence fails published acceptance.
Local measurements remain acceptance evidence and do not establish public product performance before approval.

### Generic Diagnostics

Ordinary demos retain feature selection, zero clamps, `trace.validate`, generation, and fixture checks.
The generic selector uses the largest absolute top-logit effect before the last layer.
Its historical causal check uses 8/10 signs and Spearman 0.6.
Its directional check requires a probability change of 0.05 or a changed top token, with the predicted direction.
These ordinary diagnostics do not replace any published group action or reference comparison.

Generic learned-model checks retain 20 to 512 nodes, feature share at least 0.30, and a token-to-logit path.
They require 16 positive sorted examples and completeness residuals at most 0.02.
Ordinary Gemma diagnostics require baseline confidence above 0.5 for a custom prompt, but the default Dallas prompt has no confidence threshold.
These generic requirements remain separate from published acceptance.

The generic Generate check requires a fresh completion with two rows of 24 integer token IDs.
Prompt, intervention, and text hashes must match the current state.
The actual comparison argmax IDs determine semantic outcomes, independently of bar order.
Ordinary controls and their CPU tests remain supported.

The fixture comparison exports actual graph resources and runs the native and browser parity harnesses in sequence.
It retains the 12 GB limit and requires a positive measured clamp effect without a learned-model acceptance claim.
Its sign and rank thresholds remain unchanged.
A probability change smaller than one pixel can leave its bar height unchanged.
The selected fixture examples retain 13 valid tokens and 12 padding slots in each 25-slot display row.

The normal fixture test is `tests/demos/test_circuit_trace.py::test_hidden_fixture` in Native Build CI.
Missing modules or resources fail setup, while verified absence of Metal permits a platform skip.
Isolated inference entry checks also cover argument and environment transfer on macOS and Windows.
Those checks do not establish production attachment, checkpoint restoration, shutdown, or native Windows inference.

Supported local-model test entry points remain available, but Qwen acceptance attempts remain shelved.
The Qwen entry uses `CALIPER_QWEN_MODEL` and `CALIPER_QWEN_DICTIONARY`, with optional `CALIPER_QWEN_QUALITY_OVERRIDE`.
The generic Gemma entry uses `CALIPER_GEMMA_MODEL` and `CALIPER_GEMMA_DICTIONARY`.
These opt-in checks run outside CI with explicit local resources and no downloads.

## Experimental Qwen Limits

Qwen3.5-0.8B remains experimental and does not decide M8 acceptance.
Its CPU and fixture tests remain in CI.
Further Qwen GPU runs, research regrading, and acceptance attempts remain shelved.
Its historical dictionary and index retain their measured limitations.

The historical Qwen dictionary contains 8,192 features per layer across 24 layers and records 40 million training tokens.
Its held-out FVU is 0.329109366111, dead fraction is 0.261149088542, and cross-entropy rise is 1.18043922644 nats.
FVU passes the Qwen-specific 0.40 limit but exceeds the original 0.30 limit.
Dead fraction and cross-entropy fail their unchanged limits of 0.15 and 0.5 nats.
Its historical index contains two million tokens.

The explicit founder override admits only this exact Qwen dictionary and retains those failed measurements.
The generic driver accepts it through `--quality-override FILE`, with the full fingerprint and identical measured values.
Its version 1 record uses authority `founder` and cannot admit Gemma or bypass missing resources and failed nonquality checks.
The instrument and result retain the override and the failed quality status.

The historical Qwen graph result failed at 6/10 signs and Spearman correlation 0.539394.
Its historical requirements were 8/10 signs and 0.6.
The quality override did not change those requirements.
These Qwen results supply no Gemma quality or acceptance evidence.

The historical influence-based selection, L3 F5742, supplies a secondary comparison.
It predicted -0.361462 and measured +0.003521.
The historical primary selection used the largest absolute predicted effect on the top logit outside the last layer.
Validation used the ten most influential active features.
Historical Qwen evidence passed 151 of 152 required checks, with graph validation below its thresholds.

## Refusals And Limits

Missing models, dictionaries, indexes, and completed graphs produce structured refusals.
Unsupported probe points and incompatible dictionary identities also fail explicitly.
A queued command is not a completed result.
Use its request ID to retrieve the final reply, and use events to wait for progress.
The [command reference](../agents/interpretation.md) specifies the request and reply limits.

The seeded CPU fixture verifies mathematical identities without a downloaded model.
Its reference forward hash is `2464295694bd3690f156dcc7248882f0ad289f55972caafbbd7bd23f4e3d109e`.
Fixture completeness requires a residual at most `1e-9`.
Qwen and Gemma require at most `0.02` per target and separate learned-model acceptance evidence.
