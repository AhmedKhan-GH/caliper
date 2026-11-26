# Gemma Scope Integration

Caliper LIB supports Gemma 2 semantic probes, frozen attribution paths, and JumpReLU dictionaries.
The converter imports the pinned Gemma Scope selection for `google/gemma-2-2b`.
Other external decoder families are unsupported.
The `published_dallas_v1` profile targets replication of the published circuit-tracer Gemma 2 2B Dallas example.
Gemma 2 2B is the only supported model for this profile.
The [milestone roadmap](../roadmap/milestones.md) records acceptance status.

## Verification Scope

Small random Gemma 2 configurations supply CPU evidence without gated weights.
The tests cover semantic probes, replacement forward agreement, independent-write linearity, completeness, centered logit derivatives, and feature interventions.
Synthetic arrays test encoder conversion, JumpReLU boundaries, identity refusal, and bounded file admission.
These checks do not establish results for the pretrained model.

Independent CPU comparisons use actual TransformerLens conversion with nontrivial gains.
They cover F32 and BF16 at 33, 256, and 1,024 positions, plus batches with right padding at 256 positions.
These small models do not prove full model parity, device parity, index correctness, or product acceptance.
The published comparison requires separate evidence from the actual pinned circuit-tracer run.
No local timing or model memory result establishes public product performance before acceptance.

## Pinned Sources

The model is [google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b/tree/c5ebcd40d208330abc697524c919956e692655cf), revision `c5ebcd40d208330abc697524c919956e692655cf`.
It has 26 layers, residual width 2,304, and vocabulary size 256,000.
The model uses the Gemma license and requires gated access.
Model and tokenizer loads use only local files from the exact official snapshot.

The dictionary source is [google/gemma-scope-2b-pt-transcoders](https://huggingface.co/google/gemma-scope-2b-pt-transcoders/tree/50eec2f25c60545a9a74c1c3a26a0afdd0b4b872), revision `50eec2f25c60545a9a74c1c3a26a0afdd0b4b872`.
It declares the [CC-BY-4.0 license](https://creativecommons.org/licenses/by/4.0/).
Each selected layer has 16,384 features.

The selection comes from `config.yaml` in [mntss/gemma-scope-transcoders](https://huggingface.co/mntss/gemma-scope-transcoders/tree/9250a2d4860ce5ed5c96c14d5882b7d8162809a3), revision `9250a2d4860ce5ed5c96c14d5882b7d8162809a3`.
The reference is [decoderesearch/circuit-tracer](https://github.com/decoderesearch/circuit-tracer/tree/82996f144b3371ae858ea5713590d004c218f0c2), revision `82996f144b3371ae858ea5713590d004c218f0c2`, under MIT.
The former `safety-research/circuit-tracer` repository resolves to that repository.

The tutorial's `gemma` alias uses the [historical alias config](https://huggingface.co/mwhanna/gemma-scope-transcoders/resolve/721fe56903edd07c42f71a87427faee6611d53ce/config.yaml).
Revision `721fe56903edd07c42f71a87427faee6611d53ce` matches the selected 26 original Google NPZ URIs.
The notebook omits model, transcoder, and alias revisions.
These fixed snapshots and historical alias resolution are explicit reconstruction choices, not published notebook pins.

## Published Dallas Profile

The hosted profile `published_dallas_v1` explicitly selects BF16 and the arithmetic profile `circuit_tracer_82996f14`.
An absent hosted profile retains ordinary F32 Gemma behavior.
The [pinned tutorial](https://github.com/decoderesearch/circuit-tracer/blob/82996f144b3371ae858ea5713590d004c218f0c2/demos/circuit_tracing_tutorial.ipynb), cell 8, uses exactly `Fact: the capital of the state containing Dallas is`.
Its baseline top token is ` Austin`.
The lowercase `the` is part of the input bytes.
The [published graph JSON](https://neuronpedia-attrib.s3.us-east-1.amazonaws.com/user-graphs/cmao6p1xd0000ypedw0qhlhob/gemma-fact-dallas-austin.json) uses uppercase `The` and cannot substitute for this executable example.

The seven interventions use exact published feature groups and positions.
Position IDs count the BOS prefix.
The runtime catalog retains each source label and every layer, feature, and position ID.
Cell numbers below are zero-based notebook cells.
Each `run.action` name is `published_dallas_v1.cellN`, where `N` is the cell number.
Cell 10 restores the baseline.

| Source Cell | Group Clamp | Published Top Token |
|---|---|---|
| [14][dallas-tutorial] | `capital cities / say a capital city`: -2 × Dallas activation | ` Texas` |
| [16][dallas-tutorial] | `capital`: -2 × Dallas activation | ` Texas` |
| [18][dallas-tutorial] | `Texas`: -2 × Dallas activation | ` the` |
| [20][dallas-tutorial] | `state`: -2 × Dallas activation | ` Austin` |
| [24][dallas-tutorial] | `Texas`: -2 × Dallas activation, `California`: 2 × Oakland activation | ` Sacramento` |
| [26][dallas-tutorial] | `Texas`: -2 × Dallas activation, `China`: 2 × Shanghai activation | ` Beijing` |
| [29][dallas-tutorial] | `Texas`: -2 × Dallas activation, `British Columbia`: 2 × Vancouver activation | ` the` |

[dallas-tutorial]: https://github.com/decoderesearch/circuit-tracer/blob/82996f144b3371ae858ea5713590d004c218f0c2/demos/circuit_tracing_tutorial.ipynb

Donor prompts are `Fact: the capital of the state containing Oakland is` and `Fact: the capital of the country containing Shanghai is`.
The third donor prompt is `Fact: the capital of the territory containing Vancouver is`.
Preparation reads activations from independent, unchanged prompts in the current Caliper model.
Each clamp sets an absolute value from that source activation, without multiplication of a previous intervention result.
Every complete action replaces the previous case atomically, with rollback on failure.

The source labels also include `preposition followed by place name`, `Say Austin`, `Say Sacramento`, and `Say Victoria`.
The published actions do not directly clamp those observed nodes.
The British Columbia intervention must retain the published ineffective outcome, with no `Say Victoria` activation.
Actions freeze baseline attention patterns but recompute norms and MLP outputs, as the [pinned implementation](https://github.com/decoderesearch/circuit-tracer/blob/82996f144b3371ae858ea5713590d004c218f0c2/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L737) specifies.
Preparation binds prompt bytes, token IDs, source resources, arithmetic, and dtype before action timing.

## Reference Graph

Graph reconstruction uses ten logits, cumulative probability 0.95, and at most 5,000 feature targets.
The cited graph JSON supplies node threshold 0.7 and edge threshold 0.9.
Its display thresholds and feature-density filter do not replace these generation parameters.
Reconstruction applies these settings to the lowercase prompt and compares the actual computed reference graph.

The Dallas materials omit batch size, update interval, and offload settings.
Reconstruction uses the [pinned API defaults](https://github.com/decoderesearch/circuit-tracer/blob/82996f144b3371ae858ea5713590d004c218f0c2/circuit_tracer/attribution/attribute_transformerlens.py#L42): batch size 512, update interval 4, and `offload=None`.
These defaults are reconstruction choices, not verified original Dallas settings.
Caliper records logical batches and physical chunks when memory admission requires smaller chunks.

The explicit trace request uses `profile="circuit_tracer_82996f14"`, `max_targets=5000`, `max_nodes=8192`, and `max_edges=262144`.
It also supplies `max_n_logits=10`, `desired_logit_prob=0.95`, `node_threshold=0.7`, `edge_threshold=0.9`, `batch_size=512`, and `update_interval=4`.
Node and edge limits are resource ceilings.
An excess causes refusal without silent truncation of the pruned graph.
Every setting enters cache identity and result metadata, and `trace.read` exposes the complete result in bounded pages.

The reference objective centers unembedding directions before the final softcap derivative.
The reference projection excludes BOS feature and error sources before influence normalization.
Caliper retains separate diagnostics for completeness over all sources.
The ordinary `caliper` profile retains its original objective, 512 feature targets, 512 nodes, and 8,192 edges.

## Reference Results

Reference evidence must include the baseline, all seven interventions, and the graph after canonical CPU pruning.
Record each top token and full vocabulary distribution.
Compare actual reference outputs with the published outcomes, and record any difference.
Use the actual reference graph for node and edge comparisons.

Reference replacement and completeness scores are source-library metrics.
They do not replace Caliper completeness residuals or the required product checks.

## Layer Selection

Each source path is `layer_N/width_16k/average_l0_L0/params.npz`, with the layer and listed L0 from this table.
Listed L0 values identify the selected artifacts.
They are not measured Caliper activations or TopK limits.

| Layer | Features | Listed L0 |
|---|---|---|
| 0 | 16384 | 76 |
| 1 | 16384 | 65 |
| 2 | 16384 | 49 |
| 3 | 16384 | 54 |
| 4 | 16384 | 88 |
| 5 | 16384 | 87 |
| 6 | 16384 | 95 |
| 7 | 16384 | 70 |
| 8 | 16384 | 52 |
| 9 | 16384 | 72 |
| 10 | 16384 | 88 |
| 11 | 16384 | 5 |
| 12 | 16384 | 6 |
| 13 | 16384 | 8 |
| 14 | 16384 | 8 |
| 15 | 16384 | 8 |
| 16 | 16384 | 10 |
| 17 | 16384 | 12 |
| 18 | 16384 | 13 |
| 19 | 16384 | 12 |
| 20 | 16384 | 11 |
| 21 | 16384 | 13 |
| 22 | 16384 | 15 |
| 23 | 16384 | 25 |
| 24 | 16384 | 37 |
| 25 | 16384 | 41 |

## Semantic Coordinates

`model.layers.N.resid_pre` is the block input.
`model.layers.N.mlp_in` is the actual GeGLU input, after `pre_feedforward_layernorm` and its learned gain.
`model.layers.N.mlp_out` is the output of `post_feedforward_layernorm`, which enters the residual sum.
The raw `.mlp` output precedes that norm and is a different point.

The source encoder reads the RMS-normalized input before the learned gain.
Let `u` denote that input and let `g = 1 + pre_feedforward_layernorm.weight`.
Caliper reads `x = u * g`.
The ordinary `caliper` converter transposes the source encoder and divides each input column by `g`:

```text
W_enc_caliper = W_enc_source.T / g[None, :]
preactivation = x @ W_enc_caliper.T + b_enc
activation = preactivation * (preactivation > threshold)
reconstruction = activation @ W_dec + b_dec
```

The source encoder has shape `(2304, 16384)`, and the converted encoder has shape `(16384, 2304)`.
The converter preserves decoder rows, encoder bias, decoder bias, and thresholds.
It refuses zero or nonfinite gains and nonpositive or nonfinite thresholds.
Equality with a threshold produces zero activation.
JumpReLU uses `activation="jumprelu"` and `k=None`.

The decoder already predicts the residual write after the post-feedforward norm.
Reconstruction and feature deltas enter there without another application of that norm.
Ordinary scale and clamp read the actual MLP input after input interventions, then add their decoder delta at this semantic output.
Nested clean-source forwards preserve the outer read, and block exit releases it after success or failure.
The tracer refuses raw `.mlp` edits for Gemma.

### Original Coordinates

The explicit reference artifact stores `W_enc_original` as the direct source encoder transpose cast to BF16.
Reference hooks capture the normalized BF16 input before the learned gain.
They never divide rounded inputs after the gain to recover this coordinate.
The reference encoder applies `W_enc_original` and the encoder bias to that captured input, then applies JumpReLU.

The runtime first casts each source F32 norm weight to BF16.
It then promotes that runtime norm to F32, adds one, and casts the gain to BF16:

```text
runtime_norm = source_norm.to(bfloat16)
gain = (runtime_norm.to(float32) + 1).to(bfloat16)
W_enc_original = W_enc_source.T.to(bfloat16)
W_enc = (W_enc_source.T / gain).to(bfloat16)
```

Compatibility `W_enc` division uses at least F32 precision and the rounded runtime gain.
It does not define the original coordinate.
The reference artifact also casts decoder rows, both biases, and thresholds directly to BF16.
Addition before the source norm cast can produce different gain bytes and fails reference qualification.

## Frozen Path

The live forward computes GeGLU, all four block norms, and the embedding scale.
The frozen path preserves their actual locations and captures the scaled embedding.
It fixes the normalization denominators and actual attention probabilities from the prompt.
Attention capture uses eager attention with the attention logit softcap and alternating sliding-window masks intact.
The capture restores the configured attention backend when it closes.

The live final-logit softcap also remains active.
For `f(z) = c * tanh(z / c)`, the replacement uses a local affine tangent at the captured raw logit `z0`:

```text
s = 1 - tanh(z0 / c)^2
b = f(z0) - s * z0
local_logit(z) = s * z + b
```

This map preserves the captured value and derivative.
It does not reproduce the nonlinear model for arbitrary perturbations.
Ordinary centered targets use the scaled token row minus the mean of all scaled unembedding rows.
The explicit offset and any head bias receive the same vocabulary centering and enter completeness accounting.
The implementation uses bounded row blocks without a second full unembedding matrix.

Every nonzero JumpReLU feature enters the source set.
Target and display limits do not silently impose TopK on those sources.
An active set that exceeds memory admission receives a refusal.

## Conversion And Provenance

`tools/interp/convert_gemma_scope.py` converts the exact local Scope snapshot against a verified pinned Gemma model.
Its API is `convert_gemma_scope(source, output, *, model, memory_limit_gb=40, evaluation_source=None, profile="caliper", model_snapshot=None)`.
The CLI accepts `--source`, `--output`, `--model-snapshot`, `--memory-limit-gb`, `--evaluation-source`, and `--profile`.
Select `--profile circuit_tracer_82996f14` explicitly for the original-coordinate BF16 artifact.
Reference conversion requires the pinned model snapshot for both the CLI and Python API.
The model and source paths must identify their exact revisions in the standard Hugging Face cache layout.
The converter performs no download or training.

The API and CLI default to 40 GB because this converter accepts only the pinned Gemma model.
A supplied lower limit remains active and can refuse the model load before allocation.

The converter checks bounded NPZ headers before array allocation and uses `allow_pickle=False`.
It releases each layer before the next layer and publishes the complete directory through one rename.
An existing output directory causes refusal.
`import_source` records the source pins, selection, license, source and output hashes, gain hashes, and normalization conversion.
These hashes record trusted local bytes and do not authenticate them against a remote service.

Reference conversion and qualification require the exact version 2 conversion schema.
Qualification checks every source NPZ, original F32 norm, runtime BF16 norm, effective gain, output file, and tensor byte.
It independently checks source conversion results and binds the pinned model and dictionary sources.
The reference declaration names `gemma_scope_pre_gain`, `W_enc_original`, and `bfloat16`.
Version 1 reference evidence, an inverse conversion, or a profile name alone cannot qualify these resources.
Ordinary conversion retains its original version 1 evidence.

The manifest records `training.origin="imported"`, `local_training=false`, zero local training tokens, and empty training partitions.
Publisher training fields remain unknown, and `publisher_tokens` remains null.
The separate `evaluation_source` record contains `dataset_id`, `dataset_config`, `dataset_revision`, `split`, `source_id`, and `partitions`.
The converter can accept this record through the API or `--evaluation-source` JSON file.
Acceptance requires it even though conversion permits its omission.

Evaluation and index preparation retain FineWeb-Edu `sample-10BT`, split `train`, at revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
They retain the `sha256-utf8-text-mod10000-v1` partition rule and exclusive upper bounds.
Quality uses buckets `[9400, 9500)`, and the index uses `[9500, 10000)`.
The `source_id` must match the admitted stream, including a local Parquet manifest when used.
These partitions separate local evaluation data from local index data.
They establish no disjointness from publisher training data, and both result records retain `publisher_training_overlap="unknown"`.

The evaluation and index tools use this source contract with `--model google/gemma-2-2b`.
Both tools select a 40 GB limit by default for that explicit model choice.
Evaluation must replace the semantic residual write and apply the actual final softcap during cross-entropy computation.
The local affine tangent belongs only to attribution.

## Legacy Context Policy

The pinned Gemma tokenizer adds BOS by default, and the pinned reference prepends a special token before attribution.
The legacy Gemma quality comparison uses an FVU threshold of 0.30.

Legacy quality evaluation and the feature index use one shared context adapter for Gemma.
The source stream still excludes tokenizer special tokens and appends EOS to each document.
The adapter retains those source IDs, document partitions, and content token budgets.
It prepends BOS ID 2 to each context and reserves one model position for that prefix.
Thus, a model context of 256 contains at most 255 source tokens.
Contexts end at document boundaries, the context limit, or the content token budget.

Quality batches retain document IDs and source positions across stream reads.
The batch token limit includes synthetic prefixes and right padding, and each row has its own attention mask.
FVU, feature activity, and evaluated token counts include only source positions.
During replacement, BOS and padding retain their actual MLP outputs at each layer.
Original and replacement cross-entropy use identical targets and the actual nonlinear final-logit softcap.
Cross-entropy includes BOS-to-first-content prediction and excludes pairs across document or context boundaries and targets in padding.

Evaluation settings and quality metadata record `context_policy`.
The cursor records exact `[document_id, source_position, content_length]` rows in `evaluation/state.json` under `content_segments`.
Quality metadata records their count and SHA256 under `content_segmentation`.
The digest covers rows in compact ASCII JSON, each with a final newline, in evaluation order.
Resume refuses an absent or changed policy before statistics load and retains the tensor, model, and source checks.

Index files retain source tokens and offsets without synthetic prefixes.
The manifest records `context_policy` and `example_context_policy`, and both records enter the index identity.
Document lengths and `content_context` determine the indexed source spans.
Example windows retain 33 content slots and use a separate BOS position, for at most 34 model positions.
Their fresh context can produce activations that differ from the original indexed context.
The index excludes prefix and padding positions from feature counts, offsets, and returned example activations.

An older Gemma index without these policies requires a rebuild.
Generic research qualification requires its research report and both corrected index policies.
Published qualification uses the separate BF16 resource contract below.
The live response also checks both index policies against the current model and tokenizer.
Example admission includes the additional prefix position and retains a lower memory limit when supplied.
The public Gemma tokenizer and index windows both place BOS before content.

## Reference Index

The published profile requires a new index over exactly 2,000,000 source positions from the pinned index partition.
Use `--profile circuit_tracer_82996f14` with `tools.interp.build_index` and `--model google/gemma-2-2b`.
Construction uses BF16 reference arithmetic and captures original coordinates through the reference hooks.
Qualify the artifact against both pinned snapshots before index construction.
Index compatibility checks do not replace source qualification.

Each index context has 256 model positions: one BOS prefix and at most 255 source positions.
It retains 16 examples per feature and 33 source slots per example, with an additional BOS position.
Prefixes and padding do not enter source counts or example offsets.
Stable tie order and document partition rules remain unchanged.

Format version 2 requires `encoder_policy="gemma_scope_pre_gain_v1"` and the exact reference declaration.
Its threshold policy remains `live_input_v1`.
The index binds the complete dictionary fingerprint, every artifact tensor, file hashes, tokenizer identity, and model execution identity.
Execution identity includes the arithmetic operations, package versions, dtype, device, autocast, forward settings, and float32 matmul precision.
Reference execution requires disabled autocast and highest float32 matmul precision.
A changed resource or execution identity requires a new index.

The `feature.dictionary` response uses `index_transport="operation_hash_pairs_v1"` for published index evidence.
It carries operation names and hashes as ordered pairs because operation names can exceed metadata key limits.
The receiver restores the complete operation map before it checks the manifest hash.
The stored index and its identity remain unchanged.

## Research Evaluation Protocol

The explicit `research_v1` protocol retains CPU-tested aggregation and context code for Gemma and synthetic fixtures.
M8 comparison evidence uses Gemma only.
Use `--protocol research_v1` with `tools.interp.eval_dictionary`, or pass `protocol="research_v1"` to `evaluate`.
This mode defaults to 1,024 model positions and a batch limit of 1,024 positions.
An explicit context can contain 2 through 1,024 model positions.
The batch limit must include all prefixes and right padding.
A smaller context remains an explicit, recorded protocol setting.

The supplied [Gemma Scope report](https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf), Section 4.1, pages 5–6, specifies 2,048 sequences of length 1,024.
It excludes BOS, EOS, and padding from evaluation.
Caliper retains the held-out FineWeb-Edu source, its pinned revision, and the configured resume budget of 200,000 source positions.
Source EOS counts toward that budget, but it does not count as a metric observation or CE target.
Contexts also end at document boundaries and the source budget, and thus can contain fewer than 1,024 positions.
The dataset and sample differ from the paper, and these results do not reproduce its exact evaluation.

Each Gemma context starts with genuine BOS ID 2.
EOS ID 1, BOS ID 2, and padding ID 0 do not enter FVU or activity statistics.
A full Gemma context has one prefix position and at most 1,023 source positions.

The metric mask selects source positions whose IDs are not genuine BOS, EOS, or padding.
The CE mask selects an attended predictor whose next position is a nonspecial source target in the same context row.
Thus, Gemma can use BOS to predict its first nonspecial source token.
A special predictor remains eligible when its next target satisfies the mask.
No loss pair crosses a document or context boundary.

For each layer, FVU equals total squared reconstruction error divided by total squared deviation from the dataset mean vector.
The denominator uses one mean vector over all metric observations in that layer.
It does not average context FVU ratios or normalize by the uncentered activation norm.
CPU float64 statistics merge the mean shifts between chunks and batches.
Each layer also reports mean L0 and the fraction of features without a positive activation.

The baseline forward supplies the reconstruction observations once per batch attempt.
Separate forwards replace each active MLP alone, then all active MLPs together.
`per_layer_cross_entropy` and `per_layer_cross_entropy_delta` report the isolated interventions.
`replaced_cross_entropy` and `cross_entropy_delta` retain the all-active result.
Every condition uses the same source sample, CE mask, and target denominator.
Special positions retain their actual MLP outputs, and Gemma retains its nonlinear final-logit softcap.

Research reports reside in `evaluation-research-v1/quality.json` and use `protocol="research_v1"` with `protocol_version=1`.
The separate cursor is `evaluation-research-v1/state.json`, with state schema version 2.
`source_tokens` and its `evaluated_tokens` alias count committed source positions.
`metric_observations` counts nonspecial source positions, and `cross_entropy_tokens` counts CE targets.
`source_budget_complete` becomes true only at 200,000 committed source positions.
The numeric quality thresholds classify comparison results, and a failed result remains failed.

The comparison sample can end at the first published checkpoint with at least 50,000 source positions.
The evaluator accepts SIGINT or SIGTERM for a clean stop at a checkpoint.
Its partial report retains `status="partial"`, `source_budget_complete=false`, and the exact stop reason.
The report records the exact committed sample, observation count, loss-target count, and final cursor SHA256.
The configured budget remains 200,000, and the cursor permits a later resume toward that budget.

Quality measurements supply comparison evidence, not a numerical release gate.
F32 results cannot qualify BF16 artifact bytes, reference arithmetic, or an original-coordinate index.
Published qualification records whether quality evidence exists without a new quality gate.

Research mode leaves the legacy `dictionary.json:quality` report and `evaluation/` cursor intact.
The default `legacy` selector retains the historical contexts, masks, and resume behavior.
Research settings bind the protocol, source, model, dictionary, context policy, masks, dimensions, and aggregation.
Resume rejects a changed setting before it loads statistics.
The report includes the committed cursor digest for consumers that require a consistent report and cursor.

Both protocols reject malformed counts, source coordinates, token IDs, and tensor headers before they load statistics.
Policy comparisons preserve JSON types, and thus a boolean cannot substitute for a token ID or version.
The evaluator checks mask type, shape, and device before it applies a replacement.

A batch commits only after the baseline, isolated interventions, and all-active intervention succeed.
Cooperative checkpoints run every 120 seconds, including during a layer sweep, and retain the prior complete statistics plus pending source coordinates.
Resume replays an incomplete batch without a second contribution from any partial observation.
A native operation that blocks Python must return before a checkpoint can occur.
All temporary hooks are removed on success or exception.

Prospective admission includes baseline captures, attention, MLP workspace, feature chunks, head chunks, and statistics.
The process limit, supplied lower guard, historical peak limit, and 8 GB host reserve remain active.
Features and head projections use bounded token chunks, and each model pass releases its intermediates within an autorelease scope.
No full feature matrix or sequence-wide vocabulary logits remain between passes.
The 1,024-position setting requires a measured preflight under 40 GB before a full Gemma run.
CPU tests establish protocol behavior, not pretrained quality or measured device memory.

## Resource And Acceptance Limits

The explicit Gemma process budget is at most 40 GB in decimal units.
Standalone verification, dictionary loads, and traces default to 12 GB unless the caller supplies a larger permitted limit.
The Gemma converter defaults to 40 GB.
Verification and dictionary admission permit more than 12 GB only for the exact pinned Gemma identity.

The trace family permits an explicit limit up to 40 GB for Gemma 2.
Hosted Gemma inference and the Gemma acceptance profile select 40 GB explicitly.
Fixture paths retain 12 GB.

Shared Parquet admission leaves at least 8 decimal GB of available host memory after each requested allocation.
The same allocation must fit the remaining process budget.
Historical peak checks and pressure stop thresholds remain active.

Standalone `register_service`, `Tracer`, `FeatureIndex`, `build_index`, and `validate_trace` accept `memory_limit_gb=40` for Gemma.
The service, tracer, and index APIs also accept a process guard through `memory`.
Validation accepts that guard through `guard` and otherwise reuses a matching intervention executor guard.
A supplied lower guard takes precedence over a larger requested budget, including subsequent limit changes and stop requests.
Without an explicit budget or guard, existing defaults remain unchanged.
The service also uses its index guard for a default tracer when no service guard is supplied.

CPU model load, device transfer, dictionary load, and identity snapshots have separate allocation admission checks.
Fresh current-memory measurements govern those reserves, and the process guard also enforces the historical peak.
Pending dictionary and snapshot copies retain allocation floors until their phase completes.
An earlier released peak below the limit does not consume the next phase reserve.
Any historical breach still requires a clean stop.

Dictionary identity covers all tensor content, including thresholds, and content checks require memory admission.
Gemma MPS dictionaries can retain a private MPS snapshot when the complete process reserve fits.
Each identity check compares every byte on the device in blocks of at most 16 MiB.
A batch contains at most 64 blocks and fits the current process and host reserves.
The device returns comparison flags to the CPU after each batch.
The snapshot preserves source device identity and never supplies model weights.
If the process budget cannot fit the snapshot, checks hash all live bytes instead.
A host-reserve failure stops the operation.

Example queries admit up to 16 windows in batches that fit current memory, including any live snapshot.
Each batch retains the workspace reserve and releases temporary tensors within an autorelease scope before the next admission.
Queries preserve example order and cache the complete result only after all batches succeed.
If one window cannot fit, the query refuses before its tensor allocation.
The budget is a limit, not a measured memory result or a speed guarantee.

Generic research qualification accepts complete measurements from a sample of 50,000 through 199,999 source positions.
That sample requires `status="partial"` and `source_budget_complete=false`.
Its stop reason must be `signal 2` or `signal 15`.
Qualification also accepts a full 200,000-position result with consistent completion status and its original quality status.
Both forms require all per-layer measurements, exact masks, valid aggregates, and the current source and resource identities.
The service binds the report to the exact cursor bytes.
The acceptance result and quality disclosure retain the actual sample count and original status.

Quality comparisons retain thresholds of 0.30 mean FVU, 0.15 dead fraction, and 0.5 nats all-layer loss rise.
These historical quality thresholds do not decide published Dallas acceptance.
Quality evidence binds the evaluated model, dictionary tensor bytes, forward settings, and evaluation device.
An evaluation device change requires new comparison evidence because kernel arithmetic can differ.
It does not require a new quality gate for the published profile.

### Published Acceptance

Feature-node Jaccard overlap with the actual reference must reach 0.9.
Every edge must satisfy `abs(caliper - reference) <= 1e-6 + 1e-3 * abs(reference)`.
The comparison uses the union of both edge sets and counts missing edges as zero.
Every baseline and intervention comparison covers the full vocabulary, with absolute probability error at most 0.015 and exact top token IDs.
The approved bound equals the largest reference deviation from the published top percentages.
State suppression sets this bound: `abs(0.375 - 0.36) = 0.015`, or 1.5 percentage points.
This reference envelope supersedes the earlier probability limit in the immutable source manifest.
Graph tolerances remain unchanged.
Both implementations must reproduce every published semantic outcome.
The ineffective British Columbia intervention remains part of that requirement.

Caliper completeness residuals must remain at most 0.02 per target.
Trace time must remain at most 20 seconds, and each complete intervention action must remain at most one second.
Preparation time remains separate, and cached outputs cannot satisfy action timing.
Visual, pixel, app/browser parity, the 40 decimal GB process cap, and the 8 decimal GB host reserve remain required.
The five-point clamp, 8/10 signs, Spearman 0.6, and largest-effect selection do not define published Gemma acceptance.

Startup evidence covers browser assets, program views, and error retention.
These checks do not establish published Dallas replication or product acceptance.
The [demo procedure](../python/interpretability.md#hidden-demo) separates live use, evidence collection, and independent review.
