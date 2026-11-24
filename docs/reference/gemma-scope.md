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
Gemma peak memory, trace time, and steering time remain unmeasured.

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
Its API is `convert_gemma_scope(source, output, *, model, memory_limit_gb=12, evaluation_source=None)`.
The CLI accepts `--source`, `--output`, `--model-snapshot`, `--memory-limit-gb`, and `--evaluation-source`.
The model and source paths must identify their exact revisions in the standard Hugging Face cache layout.
The converter performs no download or training.

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

The separate runtime tool update supplies Gemma evaluation and index preparation with this source contract.
This documentation base does not yet include that tool update.
Evaluation must replace the semantic residual write and apply the actual final softcap during cross-entropy computation.
The local affine tangent belongs only to attribution.

## Resource And Acceptance Limits

The explicit Gemma process budget is at most 28 GB in decimal units.
Standalone verification, dictionary conversion, dictionary loads, and traces default to 12 GB unless the caller supplies a larger permitted limit.
Verification and dictionary admission permit more than 12 GB only for the exact pinned Gemma identity.
The trace family permits an explicit limit up to 28 GB for Gemma 2.
Hosted Gemma inference and the Gemma acceptance profile select 28 GB explicitly.
Qwen and fixture paths retain 12 GB, and the Qwen trainer retains 24 GB.

CPU model load and device transfer have separate allocation admission checks.
Fresh current-memory measurements govern those reserves, and the process guard also enforces the historical peak.
Dictionary identity covers all tensor content, including thresholds, and content checks require memory admission.
The budget is a limit, not a measured memory result or a speed guarantee.

Gemma quality requires exactly 200,000 tokens, mean FVU at most 0.30, dead fraction at most 0.15, and cross-entropy rise at most 0.5 nats.
The index requires exactly 2,000,000 tokens.
Graph validation still requires at least eight correct signs among ten features and Spearman correlation of at least 0.6.
The existing feature-share, completeness, visual, parity, and directional steering checks also apply.
Trace and steering limits remain 20 seconds and one second.
The Qwen quality override and upper-half fallback do not apply to Gemma.
