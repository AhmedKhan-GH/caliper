# Gemma Scope Integration

Caliper LIB supports Gemma 2 semantic probes, frozen attribution paths, and JumpReLU dictionaries.
The converter imports the pinned Gemma Scope selection for `google/gemma-2-2b`.
Qwen3.5 support remains available, and generic Llama support remains deferred.
M8 remains held until Gemma passes every quality and acceptance check and platform CI passes.

## Verification Scope

Small random Gemma 2 configurations supply CPU evidence without gated weights.
The tests cover semantic probes, replacement forward agreement, independent-write linearity, completeness, centered logit derivatives, and feature interventions.
Synthetic arrays test encoder conversion, JumpReLU boundaries, identity refusal, and bounded file admission.
These checks do not establish results for the pretrained model.

The recorded source checks also cover the official Scope files and a conversion preflight with synthetic unit gain.
That preflight does not verify conversion with the actual model gains.
No completed model-bound conversion, 200,000-token quality evaluation, two-million-token Gemma index, or real Gemma acceptance result exists in this evidence.
Peak memory with the actual Gemma model, trace time, and steering time remain unmeasured.

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
The converter transposes the source encoder and divides each input column by `g`:

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
Scale and clamp read the actual MLP input after input interventions, then add their decoder delta at this semantic output.
Nested clean-source forwards preserve the outer read, and block exit releases it after success or failure.
The tracer refuses raw `.mlp` edits for Gemma.

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
Centered targets use the scaled token row minus the mean of all scaled unembedding rows.
The explicit offset and any head bias receive the same vocabulary centering and enter completeness accounting.
The implementation uses bounded row blocks without a second full unembedding matrix.

Every nonzero JumpReLU feature enters the source set.
Target and display limits do not silently impose TopK on those sources.
An active set that exceeds memory admission receives a refusal.

## Conversion And Provenance

`tools/interp/convert_gemma_scope.py` converts the exact local Scope snapshot against a verified pinned Gemma model.
Its API is `convert_gemma_scope(source, output, *, model, memory_limit_gb=40, evaluation_source=None)`.
The CLI accepts `--source`, `--output`, `--model-snapshot`, `--memory-limit-gb`, and `--evaluation-source`.
The model and source paths must identify their exact revisions in the standard Hugging Face cache layout.
The converter performs no download or training.

The API and CLI default to 40 GB because this converter accepts only the pinned Gemma model.
A supplied lower limit remains active and can refuse the model load before allocation.

The converter checks bounded NPZ headers before array allocation and uses `allow_pickle=False`.
It releases each layer before the next layer and publishes the complete directory through one rename.
An existing output directory causes refusal.
`import_source` records the source pins, selection, license, source and output hashes, gain hashes, and normalization conversion.
These hashes record trusted local bytes and do not authenticate them against a remote service.

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
The Gemma FVU gate remains 0.30.

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
Release qualification requires the corrected policy in both quality and index evidence.
The live response also checks both index policies against the current model and tokenizer.
Example admission includes the additional prefix position and retains a lower memory limit when supplied.
The public Gemma tokenizer and index windows both place BOS before content.
Legacy Qwen evaluation retains its existing context preparation, replacement behavior, and metric targets.
The Qwen index format remains unchanged.

## Research Evaluation Protocol

The explicit `research_v1` protocol applies the same aggregation and context limits to Gemma and Qwen.
Use `--protocol research_v1` with `tools.interp.eval_dictionary`, or pass `protocol="research_v1"` to `evaluate`.
This mode defaults to 1,024 model positions and a batch limit of 1,024 positions.
An explicit context can contain 2 through 1,024 model positions.
The batch limit must include all prefixes and right padding.
A smaller context remains an explicit, recorded protocol setting.

The supplied [Gemma Scope report](https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf), Section 4.1, pages 5–6, specifies 2,048 sequences of length 1,024.
It excludes BOS, EOS, and padding from evaluation.
Caliper retains the held-out FineWeb-Edu source, its pinned revision, and the 200,000-source-token budget.
Source EOS counts toward that budget, but it does not count as a metric observation or CE target.
Contexts also end at document boundaries and the source budget, and thus can contain fewer than 1,024 positions.
The dataset and sample differ from the paper, and these results do not reproduce its exact evaluation.

Each Gemma context starts with genuine BOS ID 2.
EOS ID 1, BOS ID 2, and padding ID 0 do not enter FVU or activity statistics.
A full Gemma context has one prefix position and at most 1,023 source positions.
The pinned Qwen tokenizer declares no BOS, and its text configuration declares no BOS ID.
Qwen records `bos_policy="native_no_bos"` and uses no prefix or substitute token.
A full Qwen context can contain 1,024 source positions.

The metric mask selects source positions whose IDs are not genuine BOS, EOS, or padding.
The CE mask selects an attended predictor whose next position is a nonspecial source target in the same context row.
Thus, Gemma can use BOS to predict its first nonspecial source token.
Qwen has no first-position target because that position has no predictor.
A special predictor remains eligible when its next target satisfies the mask.
No loss pair crosses a document or context boundary.

For each layer, FVU equals total squared reconstruction error divided by total squared deviation from the dataset mean vector.
The denominator uses one mean vector over all metric observations in that layer.
It does not average context FVU ratios or normalize by the uncentered activation norm.
CPU float64 statistics merge the mean shifts between chunks and batches.
Each layer also reports mean L0 and the fraction of features without a positive activation.
Inactive fallback layers retain null metric entries.

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
The existing numeric thresholds remain historical measurements, and a failed result remains failed.

Research mode leaves the legacy `dictionary.json:quality` report and `evaluation/` cursor intact.
The default `legacy` selector retains the historical contexts, masks, and resume behavior.
Research settings bind the protocol, source, model, dictionary, context policy, masks, dimensions, and aggregation.
Resume rejects a changed setting before it loads statistics.
The report includes the committed cursor digest for consumers that require a consistent report and cursor.

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
Qwen and fixture paths retain 12 GB, and the Qwen trainer retains 24 GB.

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
Gemma MPS dictionaries can retain a private CPU snapshot when the complete process reserve fits.
Each identity check reads bounded live byte blocks and compares every byte with that snapshot.
Readbacks have a 16 MiB cap, and macOS CPU comparisons use native `memcmp`.
The snapshot preserves source device identity and never supplies model weights.
If the reserve does not fit, checks hash all live bytes instead.

Example queries admit up to 16 windows in batches that fit current memory, including any live snapshot.
Each batch retains the workspace reserve and releases temporary tensors within an autorelease scope before the next admission.
Queries preserve example order and cache the complete result only after all batches succeed.
If one window cannot fit, the query refuses before its tensor allocation.
The budget is a limit, not a measured memory result or a speed guarantee.

Gemma quality requires exactly 200,000 tokens, mean FVU at most 0.30, dead fraction at most 0.15, and cross-entropy rise at most 0.5 nats.
Quality evidence binds the evaluated model, dictionary tensor bytes, forward settings, and evaluation device.
Use the acceptance device for evaluation. A device change requires a new evaluation because kernel arithmetic can differ.
The index requires exactly 2,000,000 tokens.
Graph validation still requires at least eight correct signs among ten features and Spearman correlation of at least 0.6.
The existing feature-share, completeness, visual, parity, and directional steering checks also apply.
Trace and steering limits remain 20 seconds and one second.
The Qwen quality override and upper-half fallback do not apply to Gemma.
