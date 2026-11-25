# Circuit Tracing

## Purpose

M7 adds model probes, transcoder dictionaries, feature examples, and attribution graphs.
M8 adds interventions and a hosted inference run.
The only acceptance model is Gemma 2 2B with the selected Gemma Scope transcoders.

Qwen3.5-0.8B remains experimental, and generic Llama support remains deferred.
Qwen CPU and fixture checks remain required, but further Qwen GPU runs, regrading, and acceptance attempts are outside this run.
Its dictionary and index remain unchanged.

The instrument shows one graph, its feature examples, and next-token probabilities.
Published group actions test the Dallas example against the actual circuit-tracer reference.
M8 remains held, with this target: reproduces the published circuit-tracer Gemma 2 2B Dallas example.
Founder Decision 12 replaces the former causal bars and feature selection procedure.

## Verified Base

The original source base was `origin/main` at `2a9c3b73`, with M6 merged.
This section records the historical checks before track implementation.
The Dictionary Contract and Replacement Model sections describe the current Gemma work.

### Model And Dictionary

The model is `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`.
Its [config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json) has 24 decoder layers and a residual width of 1,024.
The vocabulary has 248,320 entries. Six layers use full attention. The other 18 layers use Gated DeltaNet.
Full attention also has an output gate. A text-only load uses `Qwen3_5ForCausalLM` with the checkpoint text config.
The local environment uses PyTorch 2.15.0 and Transformers 5.24.0.

The bounded dictionary search used Hugging Face queries for the exact model with `transcoder`, `clt`, and `sae`.
The search also checked the [circuit-tracer model options](https://github.com/decoderesearch/circuit-tracer).
No complete per-layer MLP transcoder set for this checkpoint was found.
The [available Qwen SAEs](https://huggingface.co/decoderesearch/qwen-3.5-saes) cover selected residual layers.
The [readout dictionaries](https://huggingface.co/hematteo/sparse-readout-prism) factor the unembedding matrix.
Neither artifact supplies the required MLP input-to-output path. T2 trained the Qwen dictionary locally.

The text source is [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), config `sample-10BT`, split `train`.
The dataset revision is `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
Document partitions for training, quality checks, and examples must be disjoint and reproducible.
The manifest records those partitions. The trainer streams text and keeps no activation files.

The fallback repository is [Gemma Scope transcoders](https://huggingface.co/google/gemma-scope-2b-pt-transcoders), revision `50eec2f25c60545a9a74c1c3a26a0afdd0b4b872`.
Its file list covers all 26 Gemma 2 2B layers. A representative layer loads with `numpy.load(..., allow_pickle=False)`.
Its encoder shape is `(2304, 16384)`. Its decoder shape is `(16384, 2304)`.
It also holds encoder bias, decoder bias, and a JumpReLU threshold. A two-row encode and decode gives finite outputs.
This verifies the artifact format, not a complete fallback model run.

The source reads the RMS-normalized input before the learned pre-feedforward gain and predicts the output after the post-feedforward norm.
The ordinary converter transposes the encoder and divides its input columns by that learned gain.
The published profile requires a separate original-coordinate BF16 artifact.
Historical F32 conversion does not qualify those new bytes, and Gemma acceptance remains pending.

The founder accepted the existing Qwen dictionary for its recorded visual results and prohibited further dictionary training.
The M8 acceptance target is Gemma 2 2B with the public Gemma Scope transcoders.
M8 remains held until the actual reference comparison and every product check pass.
The former 8/10 sign and Spearman 0.6 bars do not define published Gemma acceptance.
The Qwen quality override does not apply to Gemma.

### Existing Interfaces

`TorchAdapter.points` delegates to `_ablate.points`. It lists modules and supported attention heads.
Semantic points require no CPU copy. Decoder block names come from the actual module tree.
The fixture blocks use `layers.N`. The text-only Qwen wrapper uses `model.layers.N`.
The probe suffixes are `resid_pre`, `mlp_in`, `mlp_out`, and `attn_pattern`.
The Qwen DeltaNet pattern is a signed causal mixing operator, not a softmax probability matrix.
Semantic probes retain the existing wire kind `module`. Their `probe` field identifies the semantic form.

`_programs._resolve` reads parameters, buffers, gradients, and module outputs through `.act`.
The named-tensor registry also connects model publications to this resolver.
Published tensors retain their device and storage. A new publication replaces the prior binding.
Watch removal must clear model-owned bindings and capture hooks.

The command table defines `intervene.points`, `intervene.set`, `intervene.clear`, and `intervene.list`.
Their transport uses CONTROL op `intervene`.
The request supports ablation, addition, scale, clamp, and patch operations through the `kind` field.
The runtime normalizes that field to the internal operation and preserves ordinary ablation requests.
The existing approval policy remains in force. No second intervention command family is added.

The app's `text` control supports free-text script parameters through `param:` bindings.
`run.start` supports scripts and hosted inference.
Fork checkpoints retain intervention state and the explicit published profile and private output directory.
An optimizer-free watch supplies producer attachment, program capabilities, and intervention lifecycle support.

### Text And Registry

`design/programmable-graphics.md` does not exist on the verified base.
The implemented text contract is in `docs/agents/pieces.md`, Text, and `src/builder/programs/piece_checks.cpp`.
A text table holds 4,096 strings, with at most 64 UTF-8 bytes per string.
The renderer places at most 512 labels per panel. Thus 16 rows of 33 token labels do not fit.
The examples panel uses the specified fallback of eight rows and 25 displayed tokens.
The API retains 16 examples with 33-token windows. Two pages expose all examples.

The acceptance check counts eight visible row IDs on each page and verifies 16 distinct example IDs across both pages.
This replaces the incompatible requirement for 16 simultaneous visible rows.
Long token labels truncate only at UTF-8 boundaries. Their full text stays in the query response.

The roadmap schema comes from `sternbild_docs/schemas/roadmap.schema.json` in the pinned docs package.
`planned`, `core`, and `issue: null` are valid. Every feature also needs `depends_on`.
Current dependency IDs `INT-34`, `AGT-31`, and `STR-55` support the dependency list below.
Registry status describes actual delivery and remains separate from M8 acceptance.

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

The family map has explicit fixture, Qwen3.5 text, and Gemma 2 entries.
Gemma uses `google/gemma-2-2b` at revision `c5ebcd40d208330abc697524c919956e692655cf`.
The model has 26 layers, residual width 2,304, and vocabulary size 256,000.
It requires the official gated snapshot under the Gemma license.
CPU tests use small random configurations without those weights.
Generic Llama decoders remain follow-up work.

The converter selects the official Scope snapshot at revision `50eec2f25c60545a9a74c1c3a26a0afdd0b4b872`.
Its selection follows `mntss/gemma-scope-transcoders` at `9250a2d4860ce5ed5c96c14d5882b7d8162809a3`.
The reference implementation is `decoderesearch/circuit-tracer` at `82996f144b3371ae858ea5713590d004c218f0c2`.
All 26 selected layers have 16,384 features.
The Scope files use CC-BY-4.0, and the reference code uses MIT.
The complete per-layer L0 table and source paths are in `docs/reference/gemma-scope.md`.

Gemma `resid_pre` is the block input, and `mlp_in` is the actual GeGLU input after the pre-feedforward norm and learned gain.
The semantic `mlp_out` is the post-feedforward norm output at the residual sum.
The raw `.mlp` output is a different point, which the tracer refuses for interventions.
The actual forward retains GeGLU, all four block norms, and the embedding scale.

Let `g = 1 + pre_feedforward_layernorm.weight`.
The converter changes source `W_enc` from `(2304, 16384)` to `W_enc.T / g[None, :]`, with shape `(16384, 2304)`.
This compatibility conversion belongs to the ordinary `caliper` profile.
It does not establish original-coordinate BF16 parity.
Zero or nonfinite gains cause refusal.
Decoder rows, both biases, and JumpReLU thresholds retain their source values.
The decoder writes after the post-feedforward norm without another normalization.

The explicit `circuit_tracer_82996f14` artifact adds `W_enc_original`, the direct source encoder transpose cast to BF16.
Reference hooks capture the normalized input before learned gain and never divide rounded inputs after gain.
Source F32 norm weights first cast to runtime BF16.
Gain preparation promotes those runtime weights to F32, adds one, and casts the gain to BF16.
Compatibility `W_enc` divides by that rounded gain before its BF16 cast, but reference operations use `W_enc_original`.

Conversion and qualification bind every source NPZ, checkpoint norm, runtime norm, gain, artifact file, and tensor byte.
They require the exact version 2 conversion schema and explicit pinned source and model snapshots.
An inverse conversion or version 1 reference evidence cannot qualify the published artifact.
The original `caliper` conversion retains its version 1 evidence and ordinary behavior.

The converter checks NPZ headers before allocation, disables pickle, and releases each layer before the next.
It requires the exact verified model and source revisions and publishes only a complete new output directory.
The `import_source` record includes all source pins, selected paths, L0 values, hashes, norm conversion, and license.
The hashes identify trusted local bytes without remote authentication.

Imported training metadata records no local training, zero local tokens, and unknown publisher data and token counts.
The separate `evaluation_source` record binds the pinned FineWeb source, source identity, and unchanged quality and index partitions.
The converter accepts this record, and acceptance requires it.
Local partition separation does not establish disjointness from publisher training data.
Quality and index records must retain `publisher_training_overlap="unknown"`.
The evaluation and index tools implement this source contract for the explicit Gemma model selection.

Founder Decision 8 and Coordinator Decision 10 make research quality separate comparison evidence.
The historical F32 sample contains 52,231 source positions and 52,174 metric observations against a 200,000-position resume budget.
It retains partial status and stop reason `signal 2`.
Its identity and quality values cannot qualify new BF16 bytes, arithmetic, or index resources.
The published profile discloses absent or supplied historical quality evidence and adds no quality gate.
The generic research protocol retains its identity, sample, context, and per-layer loss checks.

The published index requires exactly 2,000,000 source positions, with 256 model positions per context with BOS.
It uses original-coordinate BF16 captures and format version 2 with encoder policy `gemma_scope_pre_gain_v1`.
The complete artifact fingerprint and model execution identity bind every file, tensor, arithmetic operation, package version, dtype, device, and forward setting.
The index retains 16 examples, 33 source slots per example, a separate BOS position, and stable tie order.
Artifact qualification must precede index construction, and index compatibility does not replace source qualification.
No new dictionary training or Qwen regrading is authorized.

The explicit Gemma process budget is at most 40 decimal GB.
Standalone verification, dictionary loads, and traces retain a 12 GB default.
The Gemma converter defaults to 40 GB because the model and complete dictionary exceed 12 GB.
A supplied lower converter limit remains active.

Only the pinned Gemma identity permits a larger verification or dictionary limit, and only the Gemma family permits a larger trace limit.
Hosted Gemma inference and acceptance select 40 GB explicitly.
Qwen and fixture limits remain 12 GB, and Qwen training retains 24 GB.

Shared Parquet admission requires at least 8 decimal GB of available host memory after the requested allocation.
It also requires sufficient process headroom and retains historical peak checks.
The pressure stop thresholds remain unchanged.

The loader admits CPU load and device transfer separately, verifies checkpoint identity on the CPU, then checks that identity after transfer.
The identity includes tied weights, effective settings, default RoPE buffers, and the nonpersistent embedding scale.
Random Gemma CPU tests cover these paths and semantic interventions.
Independent CPU checks cover actual TransformerLens conversion at 33, 256, and 1,024 positions, plus right padding at 256 positions.
They cover F32 and BF16, but small CPU models do not prove full model or device parity.
The actual reference completed all published cases and CPU pruning.

Full Caliper comparison, index qualification, timing, memory, and product acceptance remain separate lead checks.

`load_dictionary(path, *, model=None, model_hash=None, device=None, dtype=None)` validates the model identity before use.
The loader requires the supplied model, an explicit expected hash, or the current watched model.
It refuses a mismatch and refuses an absent identity. The manifest alone cannot establish compatibility.
The return object exposes `encode(layer, x)`, `decode(layer, a)`, and `error(layer, x, y)`.
Its `metadata` holds the parsed manifest. Its `layers` hold the per-layer weights.

The folder holds `dictionary.json` and `layer_NN.safetensors` for each layer.
The manifest records model ID, model hash, revision, layers, width, latents, activation, k, and probe names.
It also records training text, partitions, token count, elapsed time, peak memory, and measured quality.

Each encoder has shape `(latents, width)`. Each decoder has the same shape.
The TopK encoder computes `relu(x @ W_enc.T + b_enc)`, then keeps the largest k activations.
The decoder computes `a @ W_dec + b_dec`. The error is the original semantic MLP output minus the decoder output.

The JumpReLU encoder computes `z = x @ W_enc.T + b_enc`, then `z * (z > threshold)`.
JumpReLU requires positive finite thresholds and `k=None`.
Threshold equality produces zero, and every nonzero feature enters the attribution source set.
Threshold bytes enter dictionary identity and evaluation evidence.

The historical Qwen training plan used 16,384 latents per layer and k = 32.
For 24 layers, its 805,724,160 parameters occupy about 3.223 GB in float32 or 1.611 GB in float16.
The trainer uses 40 million tokens with context 256. It updates all layers after each model forward pass.
It processes layer losses in sequence to bound temporary memory. Training does not allocate full-vocabulary logits.
A ten-minute pilot measures throughput and peak memory with `/usr/bin/time -l` on a quiet machine.

The historical fallback plan reduced latents to 8,192, then tokens to 20 million, if the pilot predicted more than ten hours.
The hard limits remain ten hours and 24 GB.
The trainer uses a 240-second checkpoint interval after each completed save and stops if the work interval reaches 270 seconds.
Checkpoint writes consume additional time within the aggregate cap.
The founder checkpoint rule overrides the earlier 30-minute interval.

The external resource wrapper runs training at `nice -n 10`, with one heavy slot and exclusive GPU access.
These requirements are run policy. Portable tools do not contain machine-specific lock paths.
The process checks memory pressure and stops cleanly at the specified thresholds.
Startup stops return a bounded structured result and a nonzero exit without a traceback.
A stop before complete initialization or restore preserves existing checkpoints and creates no new checkpoint.

Model and tokenizer files must exist locally before a tool starts.
The dataset can use a pinned local Parquet manifest or the specifically approved public FineWeb-Edu stream.
The stream uses the fixed revision and document partitions in the training metadata.

Qwen quality uses 200,000 held-out tokens. Its revised mean unexplained variance limit is 0.40.
Dead latents must not exceed 15 percent. Full MLP replacement must increase cross-entropy by at most 0.5 nats.
Evaluation admission compares runtime tensor bytes with the artifact after the loader dtype conversion.
The cursor binds those bytes, the dictionary fingerprint, and evaluation settings. Resume refuses a cursor without that tensor evidence.
A stop before admission preserves the prior quality report and cursor. Stream creation alone does not admit evaluation.
After admission, an incomplete evaluation records `partial` and the current dictionary fingerprint, even before two tokens complete.
It emits `quality_end` and replaces any earlier passed report only when the final artifact verification succeeds.
If memory admission prevents that verification, the evaluator preserves the prior published evidence.

The 23 November decision accepts the 8,192-feature dictionary and supersedes the earlier quality fallback plan.
That run completed 40 million tokens, quality evaluation, the example index, and the hidden Qwen captures.
No new dictionary run or upper-half retraining is authorized. Failed causal checks remain failures and retain their captures.
Those Qwen results retain their historical feature-share, graph-validation, parity, and budget checks.
They do not define the later published Gemma replication procedure.

The library retains upper-half dictionary support for existing artifacts.
Such a dictionary represents the lower half through error nodes.
For Qwen, layers 12 through 23 train, and layers 0 through 11 retain zero transcoders.
The original lower MLPs remain in the quality evaluation.
The fallback records the admitted source metadata, exact fingerprint, and completed evaluation cursor in `fallback_source`.

Fallback quality has scope `upper_half` and status `fallback_passed`, `fallback_failed`, or `fallback_partial`.
Upper-half acceptance does not establish all-layer acceptance or receive the all-layer `passed` status.
Lower layers have no feature examples or feature intervention targets because their MLP contributions enter error nodes.
This reduced coverage can increase error share and reduce feature share. Generic fallback checks retain a feature share of at least 0.30.
The acceptance report must record both measured shares and the fallback scope.

The original and fallback training jobs share the aggregate ten-hour cap, which includes model load, checkpoints, and export.
The fallback adds no time beyond that cap. Its token budget and quality scope are separate from the aggregate time allowance.
The detailed contract is in `docs/reference/upper-half-dictionaries.md`.

The ordinary example index uses two million other held-out tokens and keeps the largest 16 activations per feature.
Its memory-mapped table records document, token position, activation count, and maximum.
Version 2 binds the `live_input_v1` encoder policy in its manifest hash.
It also binds tensor precision, device, effective forward settings, autocast, and float32 matmul precision.
The builder checks this evidence again before completion, and the loader checks it before data maps open.
Missing or different evidence requires a rebuild.
The model and dictionary must use one shared device at build and load admission.
Version 1 remains readable only when model and encoder floating tensors use one dtype.
Other legacy indexes require a rebuild because their encoder used storage precision.
Window queries run at most 16 windows together and use an LRU cache of 256 features.

## Replacement Model

The method follows [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) and the open circuit-tracer implementation.
Each MLP becomes its transcoder output plus a detached error vector.
RMSNorm denominators and attention mixing coefficients are constants from the prompt forward pass.
Active feature masks are fixed for the local path. Encoder and decoder biases remain explicit terms.
Without a final softcap, logit targets use the unembedding row minus the mean unembedding row.

Qwen requires additional frozen factors to make this path linear.
For full attention, freeze the softmax pattern and the sigmoid output gate.
For DeltaNet, freeze queries, keys, decay, beta, the value SiLU gate, the output gate, and normalization denominators.
Keep the value projection, causal convolution, recurrent value transport, and output projection linear.
The value SiLU becomes its input times the recorded sigmoid factor.
This extends the same frozen-attention definition to the actual hybrid architecture. It does not attribute attention routing decisions.

Gemma captures actual eager attention probabilities after the attention softcap and alternating sliding-window masks.
Capture restores the configured attention backend on exit.
The frozen value path includes the post-attention norm at its actual location.
Independent MLP writes enter after the post-feedforward norm, and the captured embedding includes its scale.

Gemma keeps the nonlinear final softcap in the live model.
For `f(z) = c * tanh(z / c)`, the replacement uses `f(z0) + f'(z0) * (z - z0)` at the captured raw logit.
Its derivative is `1 - tanh(z0 / c)^2`.
The affine tangent preserves the captured value and derivative but does not reproduce arbitrary nonlinear perturbations.
Ordinary centered targets subtract the mean scaled unembedding row from the scaled token row.
The explicit offset and head bias receive the same vocabulary centering for completeness.

CPU tests check replacement agreement, independent-write linearity, completeness, and centered derivatives against actual random Gemma forwards.
They also cover nonunit gains, short sliding windows, attention biases, small softcaps, and JumpReLU threshold boundaries.
These tests do not establish learned-model quality or acceptance.

The replacement output must equal the recorded output before any edge calculation.
A test must also prove linearity under source perturbations. Ordinary nonlinear-model gradients do not satisfy this contract.
The tracer never constructs a dense residual Jacobian.
Input interventions at `resid_pre` and `mlp_in` also form part of this affine path.
An input patch or ablation removes the upstream derivative at its selected positions.
An input addition retains that derivative. Recorded constant changes have an explicit intervention bias term.
For each target, it injects its encoder direction at the read point and backpropagates through the frozen path.
A source edge is its activation times the dot product of its decoder and the gradient at its write point.
Token embeddings and error vectors act as direct vector sources.

The ordinary `caliper` profile traces at most ten logits with 95 percent cumulative probability.
It visits at most 512 feature targets in descending estimated logit influence.
It normalizes absolute incoming edges and computes influence with 32 power steps and logit probability weights.
Ordinary pruning retains 80 percent node influence and 98 percent edge influence, with caps of 512 nodes and 8,192 edges.
Completeness is measured before display pruning. The report also records omitted influence and error share.
The unpruned accounting preserves bias terms separately from displayed source nodes.

### Published Reconstruction

The explicit graph profile is `circuit_tracer_82996f14`.
It uses centered unembedding directions before the final softcap derivative and excludes BOS feature and error sources from the reference projection.
Caliper retains separate completeness diagnostics over all sources.
The published graph JSON supplies ten logits, probability mass 0.95, 5,000 feature targets, and pruning thresholds 0.7 and 0.9.
Reconstruction uses the exact lowercase tutorial prompt, not the uppercase prompt stored in that JSON.

The Dallas sources omit batch size, update interval, and offload settings.
Reconstruction selects the pinned API defaults of 512, 4, and `None` respectively and identifies these as unpublished reconstruction choices.
The request explicitly supplies all settings and uses ceilings of 8,192 nodes and 262,144 edges.
An excess causes refusal without silent truncation of the pruned result.
All settings and physical batch schedules enter the result, while cache identity binds the requested settings.
The complete kept graph remains available through pages of at most 256 nodes and 256 edges.

## Tensor Contract

`Tracer(model, tokenizer, dictionary).trace(prompt, max_targets=512, max_nodes=512, max_edges=8192)` returns a `Graph`.
Ordinary calls publish the five tensors below on the model device. Padding is zero and counts define valid rows.
Shapes depend on the profile and never change with pruning. A prompt has at most 32 tokens, with special tokens.
Each trace is one run step.
The eight-entry cache binds prompt, intervention state, profile, trace settings, and label index identity.

| Tensor | Shape | Type | Fields |
|---|---|---|---|
| `trace.counts` | `(4,)` | int32 | Nodes, edges, tokens, layers |
| `trace.node_i` | `(512, 4)` | int32 | Kind, layer, position, feature or token ID |
| `trace.node_f` | `(512, 4)` | float32 | Activation, influence, top-logit effect, reserved |
| `trace.edge_i` | `(8192, 2)` | int32 | Source, target |
| `trace.edge_f` | `(8192, 2)` | float32 | Weight, weight divided by largest magnitude |

The reference profile expands node rows to 8,192 and edge rows to 262,144 without a field-layout change.
Label storage also expands to 8,224 rows, with prompt slots.
Ordinary calls retain their lower capacities.

Node kinds are token = 0, feature = 1, error = 2, and logit = 3.
Token nodes use layer -1. Logit nodes use layer count. Feature and error layers are zero-based.
Edges reference the published node rows. Graph JSON retains token text, feature labels, status, and completeness diagnostics.
The read API supports paging. A trace request queues work at the next step boundary.

## Pieces And Commands

The library pieces are `attribution_graph@1`, `token_strip@1`, `logit_bars@1`, and `heat_grid@1`.
Graph layout uses seven derive passes: cell assignment, influence order, four offset relaxations, and cubic edge curves.
The draw uses signed additive edges, node sprites, and a node ID buffer.
Shared state `selected_node` selects the feature and dims unrelated edges.
The text table shows prompt tokens and the hovered or selected label.
The app and browser use the same piece files.

The draw budget is 4 ms and 64 MB. The derive budget is eight passes, 50 ms, and 64 MB.
The probe and dictionary implementations stay in Python. Graph layout and draw stay in WGSL.
The app holds no model weights or dictionary weights.
All commands enter `src/builder/commands/command_table.cpp`.
The Python control and intervention modules retain their current transport and lifecycle boundaries.

M7 commands are `feature.dictionary`, `feature.show`, `feature.examples`, `trace.run`, `trace.status`, and `trace.read`.
M8 extends the existing intervention commands and adds `patch.sweep` and `trace.validate`.
Read commands do not mutate the model. Change commands use the existing approval and step-boundary paths.
A missing dictionary gives a structured refusal.

The outline reference `tests/scene/draw_reference.h` is absent on the base.
The implemented cross-backend limits are in `web/test/draw_parity.spec.ts`.

Interior pixels differ by at most two channel levels.
The existing edge-side exemption accepts an edge pixel that matches an adjacent reference color.
After this exemption, unmatched edge pixels occupy at most one percent of the image.
Interior IDs must match exactly. The new piece checks retain these limits.

## Intervention Contract

`ablate` retains its behavior. `add` writes a coefficient times a vector at selected positions.
`scale` writes `(multiplier - 1) * activation * decoder` at the semantic MLP output.
`clamp` writes `(value - activation) * decoder`. Both use the live MLP input and the real model.
`patch` substitutes a cached clean-prompt activation. Hook failure ends the intervention and reports its error.
Fork checkpoints carry the full intervention specification.

Gemma feature hooks read the actual MLP input after input edits and write their delta after the post-feedforward norm.
They preserve an outer read across nested clean-source forwards and release it after block success or failure.

Ordinary output comparison uses one batch of two: baseline and intervention. It publishes `output.probs` with shape `(2, 10)`.
`output.tokens` has shape `(10,)`. The bars share token IDs and retain a baseline ghost bar.
The comparison reserves slots for both baseline and intervention argmax token IDs, even when either leaves the other row's top ten.
The generate action emits 24 greedy tokens for each case as bound text.
The feature slider spans negative four to positive four times the indexed maximum, with zero as ablation.
A reset removes the intervention. A retrace toggle controls whether the graph also updates.

The patch sweep uses equal-length clean and corrupted prompts and a two-token logit difference.
It processes at most 32 layer-position cells per batch and publishes normalized recovery as `patch.grid`.
Generic validation ablates the ten most influential features and reports Spearman correlation and sign agreement.
That diagnostic does not define published Gemma acceptance.

## Published Actions

The hosted profile `published_dallas_v1` explicitly selects reference arithmetic, the BF16 artifact, and the original-coordinate index.
`run.start` requires `replication_profile` and `private_output_directory` together for inference and rejects them for scripts.
The caller creates the private directory before launch.
The CLI preserves the pair as `--replication-profile` and `--private-output-directory`.
Missing or incompatible source resources cause refusal without a profile fallback.

The prompt is exactly `Fact: the capital of the state containing Dallas is`.
The pinned tutorial defines seven group actions in cells 14, 16, 18, 20, 24, 26, and 29.
Cell 10 supplies the baseline.
The source-linked lookup and full pins are in `docs/reference/gemma-scope.md`.
The runtime catalog retains source labels and exact layer, feature, and position IDs.

The first four actions suppress `capital cities / say a capital city`, `capital`, `Texas`, and `state` with -2 times their Dallas activations.
The other three actions combine Texas suppression with twice the donor activation for `California`, `China`, or `British Columbia`.
Donors use independent Oakland, Shanghai, and Vancouver prompts from the source.
Labels `Say Austin`, `Say Sacramento`, and `Say Victoria` identify observed nodes, not additional direct clamps.
The British Columbia case retains the published ineffective outcome.

Preparation binds exact prompt bytes, token IDs, model, dictionary, arithmetic, dtype, and unmodified donor activations.
Each `published_dallas_v1.cellN` callback replaces the previous complete case atomically.
Failed admission or execution restores the prior complete state and returns one terminal failure.
Actions freeze baseline attention patterns and recompute norms and MLP outputs.
They use original-coordinate captures, never inverted rounded inputs, graph selection, index maxima, or a previous case's modified activation.

The runtime publishes a fresh actual output for every action and retains full vocabulary evidence beneath the explicit private root.
Candidate files alone do not establish a committed output.
The group lifecycle preserves ordinary clear, restore, checkpoint, fork, and rollback behavior.
One control applies each published case, and the instrument shows source group labels and actual probability bars.

## Validation And Delivery

M8 remains held until Caliper reproduces the actual pinned reference and every product check passes.
The reference completed all seven cases and canonical CPU pruning.
All seven top tokens match published outcomes, but most rounded probability lists differ, while cell 26 matches.
The numerical comparison uses the actual full outputs and kept graph, not rounded percentages alone.
Source-library completeness scores do not replace Caliper completeness residuals.

Feature-node Jaccard overlap must reach 0.9.
Every union edge must satisfy `abs(caliper - reference) <= 1e-6 + 1e-3 * abs(reference)`, with missing edges equal to zero.
Every baseline and intervention probability must agree across the vocabulary within 0.001 absolute, with exact top token IDs.
Both implementations must reproduce all published semantic outcomes.
The ineffective British Columbia intervention remains part of that requirement.
A failed reference outcome remains a failure and cannot establish a Caliper pass.

Caliper completeness residuals must remain at most 0.02 per target.
The trace limit remains 20 seconds, and each complete intervention action must finish within one second.
Action timing includes actual model work, output validation, export, publication, and terminal checks.
Preparation time remains separate, and cached outputs cannot satisfy the action limit.
The process cap remains 40 decimal GB, with an 8 decimal GB host reserve.
Visual, pixel, and app/browser parity checks remain mandatory.

### Live And Collection Paths

The live path creates a startup instrument before model initialization and retains its MCP event bridge.
Only the founder uses `--show`, while `--hidden --demo` exercises the same interactive path without a visible window.
A driver error leaves the app open with the error displayed, also for failures before bridge creation.
Browser asset routes and program views now have verified hidden startup proof, with retained attachment and error display.
This proof establishes launch reliability, not published replication or product acceptance.

Hidden collection uses the explicit profile and paired private output arguments.
It first measures a separate bounded preparation and baseline, then collects the full graph and all seven actions through Caliper MCP.
A fresh baseline precedes each intervention, and collection waits for each terminal event and output publication.
Complete collection has exit zero but retains `accepted_replication=false`.
The result and collection manifest retain failures, resource identities, complete pages, private case exports, captures, and file hashes.
The live path skips this acceptance batch and records only demo readiness.

Independent review compares the actual Caliper and reference resources and inspects the captures.
It also supplies browser, parity, and memory receipts bound to the collection hash.
The separate verifier rechecks all file hashes, case identities, and required metrics before it can set acceptance complete.
Resource qualification, cached flags, synthetic replies, and historical quality evidence cannot replace that review.
The public procedure is in `docs/python/interpretability.md`.

### Visual And Resource Checks

The pixel oracle maps graph, example, and probability resources through actual image quads, clips, and framebuffer scale.
Every visible node needs its expected ID, and each bar height must match its current probability within two pixels.
The two example pages retain eight visible rows each and 16 distinct example IDs in total.
An independent text reference uses the actual font atlas and expected prompt and selected-label glyph quads.
Each required label needs visible ink and pixels within two channel levels of the reference.
Synthetic failures test these oracles but do not establish application acceptance.

App/browser parity uses the actual published resources.
Interior IDs must match exactly, interior colors differ by at most two channel levels, and unmatched edge pixels occupy at most one percent.
Browser evidence must show program views and attachment before and after capture.
Derive limits remain 50 ms and 64 MiB, and draw limits remain 4 ms and 64 MiB.
Missing times, peak memory, host reserve, or capture evidence fails the corresponding required check.
Local measurements do not establish public product performance before acceptance.

### Ordinary Checks

The seeded fixture retains its four layers, width 64, four heads, and exact TopK MLPs.
Its float64 CPU forward has a saved hash and requires no model download or GPU.
Fixture completeness remains at most 1e-9, and finite-difference edges and power iteration retain their 1e-6 tolerances.
Ordinary calls retain the 12 GB process limit, generic feature selection, clamps, generation, and diagnostic validation.
Their CPU and fixture tests remain supported.
Generic learned-model checks retain 20 to 512 nodes, feature share at least 0.30, and a token-to-logit path.

The former largest-effect selection, five-point probability change, 8/10 signs, and Spearman 0.6 remain generic or historical diagnostics.
They do not define published Gemma acceptance.
The Qwen quality override cannot admit Gemma.
M8 requires independent review, public-content checks, and green macOS and Windows CI after all acceptance checks pass.

### Historical Qwen Results

Qwen3.5-0.8B remains experimental and shelved for further GPU runs, research regrading, and acceptance attempts.
Its CPU paths, fixture tests, dictionary, index, and upper-half support remain available.
The historical dictionary has 8,192 features per layer across 24 layers and records 40 million training tokens.
Held-out FVU is 0.329109, dead fraction is 0.261149, and cross-entropy rise is 1.180439 nats.
The explicit founder quality override admits that exact dictionary and preserves its failed quality measurements.
Its historical index contains two million tokens.

The historical graph result failed at 6/10 signs and Spearman correlation 0.539394.
The historical requirements were 8/10 and 0.6, and the quality override did not change them.
These results supply no Gemma acceptance evidence.
The supported generic Qwen driver retains explicit fingerprint and measurement checks for its override.
No new Qwen acceptance run follows from retention of these CPU paths.
