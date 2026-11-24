# Circuit Tracing

## Purpose

M7 adds model probes, transcoder dictionaries, feature examples, and attribution graphs.
M8 adds interventions and a hosted inference run. The acceptance model is Qwen3.5-0.8B.
The instrument shows one graph, its feature examples, and next-token probabilities.
A feature clamp tests the effect that the graph predicts.

## Verified Base

The source base is `origin/main` at `2a9c3b73`, with M6 merged.
The checks below resolve the outline assumptions before track implementation.

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
Neither artifact supplies the required MLP input-to-output path. T2 trains the dictionary locally.

The text source is [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), config `sample-10BT`, split `train`.
The dataset revision is `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
Document partitions for training, quality checks, and examples must be disjoint and reproducible.
The manifest records those partitions. The trainer streams text and keeps no activation files.

The fallback repository is [Gemma Scope transcoders](https://huggingface.co/google/gemma-scope-2b-pt-transcoders), revision `50eec2f25c60545a9a74c1c3a26a0afdd0b4b872`.
Its file list covers all 26 Gemma 2 2B layers. A representative layer loads with `numpy.load(..., allow_pickle=False)`.
Its encoder shape is `(2304, 16384)`. Its decoder shape is `(16384, 2304)`.
It also holds encoder bias, decoder bias, and a JumpReLU threshold. A two-row encode and decode gives finite outputs.
This verifies the artifact format, not a complete fallback model run.

Gemma Scope folds pre-MLP normalization gains and predicts the output after the post-MLP normalization.
A fallback converter must preserve those points and transpose the encoder into the Caliper format.
Gemma remains a documented fallback. Qwen remains the acceptance target.

### Existing Interfaces

`TorchAdapter.points` delegates to `_ablate.points`. It lists modules and supported attention heads.
T1 adds semantic points without a CPU copy. Decoder block names come from the actual module tree.
The fixture blocks use `layers.N`. The text-only Qwen wrapper uses `model.layers.N`.
The probe suffixes are `resid_pre`, `mlp_in`, `mlp_out`, and `attn_pattern`.
The Qwen DeltaNet pattern is a signed causal mixing operator, not a softmax probability matrix.
Semantic probes retain the existing wire kind `module`. Their `probe` field identifies the semantic form.

`_programs._resolve` reads parameters, buffers, gradients, and module outputs through `.act`.
It has no arbitrary named-tensor registry. T1 adds that registry and connects it to this resolver.
Published tensors retain their device and storage. A new publication replaces the prior binding.
Watch removal must clear model-owned bindings and capture hooks.

The command table already defines `intervene.points`, `intervene.set`, `intervene.clear`, and `intervene.list`.
Their transport uses CONTROL op `intervene`. The nested request uses `op`, with `ablate` as the implemented value.
The values `scale`, `steer`, and `patch` are reserved. T6 and T7 extend these commands and preserve old ablation requests.
They accept the new `kind` field as the intervention discriminator and normalize it to the existing internal operation.
The existing approval policy remains in force. No second intervention command family is added.

The app already has a `text` control, but it binds only local values and shared state.
`_controls._param_spec` refuses a string without choices. T6 adds a free-text script parameter and `param:` text binding.
Hosted runs currently start scripts through `run.start`. There is no `inference` run kind.
T8 extends that runner contract and carries intervention state through its existing fork checkpoint.
An optimizer-free watch also needs producer attachment, program capabilities, and intervention registration.
T1 supplies producer attachment. T7 supplies the remaining capability and intervention lifecycle.

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
Existing dependency IDs `INT-34`, `AGT-31`, and `STR-55` exist. The following new dependency list is frozen.
Registry status stays `planned` until implementation, then uses the schema status for the actual delivery state.

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

`load_dictionary(path, *, model=None, model_hash=None, device=None, dtype=None)` validates the model identity before use.
The loader requires the supplied model, an explicit expected hash, or the current watched model.
It refuses a mismatch and refuses an absent identity. The manifest alone cannot establish compatibility.
The return object exposes `encode(layer, x)`, `decode(layer, a)`, and `error(layer, x, y)`.
Its `metadata` holds the parsed manifest. Its `layers` hold the per-layer weights.

The folder holds `dictionary.json` and `layer_NN.safetensors` for each layer.
The manifest records model ID, model hash, revision, layers, width, latents, activation, k, and probe names.
It also records training text, partitions, token count, elapsed time, peak memory, and measured quality.
Each encoder has shape `(latents, width)`. Each decoder has the same shape.
The encoder computes `relu(x @ W_enc.T + b_enc)`, then keeps the largest k activations.
The decoder computes `a @ W_dec + b_dec`. The error is the original MLP output minus the decoder output.

The initial dictionary uses 16,384 latents per layer and k = 32.
For 24 layers, its 805,724,160 parameters occupy about 3.223 GB in float32 or 1.611 GB in float16.
The trainer uses 40 million tokens with context 256. It updates all layers after each model forward pass.
It processes layer losses in sequence to bound temporary memory. Training does not allocate full-vocabulary logits.
A ten-minute pilot measures throughput and peak memory with `/usr/bin/time -l` on a quiet machine.

If the pilot predicts more than ten hours, reduce latents to 8,192, then tokens to 20 million.
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

Quality uses 200,000 held-out tokens. Mean unexplained variance must not exceed 0.40.
Dead latents must not exceed 15 percent. Full MLP replacement must increase cross-entropy by at most 0.5 nats.
Evaluation admission compares runtime tensor bytes with the artifact after the loader dtype conversion.
The cursor binds those bytes, the dictionary fingerprint, and evaluation settings. Resume refuses a cursor without that tensor evidence.
A stop before admission preserves the prior quality report and cursor. Stream creation alone does not admit evaluation.
After admission, an incomplete evaluation records `partial` and the current dictionary fingerprint, even before two tokens complete.
It emits `quality_end` and replaces any earlier passed report only when the final artifact verification succeeds.
If memory admission prevents that verification, the evaluator preserves the prior published evidence.

The 23 November decision accepts the 8,192-feature dictionary and supersedes the earlier quality fallback plan.
This run completes 40 million tokens and then evaluates quality, builds the example index, and captures the hidden Qwen demo.
No new dictionary run or upper-half retraining is authorized. Failed causal checks remain failures and retain their captures.
The feature-share, graph-validation, steering, pixel, parity, and budget checks remain unchanged.

The library retains upper-half dictionary support for existing artifacts.
Such a dictionary represents the lower half through error nodes.
For Qwen, layers 12 through 23 train, and layers 0 through 11 retain zero transcoders.
The original lower MLPs remain in the quality evaluation.
The fallback records the admitted source metadata, exact fingerprint, and completed evaluation cursor in `fallback_source`.

Fallback quality has scope `upper_half` and status `fallback_passed`, `fallback_failed`, or `fallback_partial`.
Upper-half acceptance does not establish all-layer acceptance or receive the all-layer `passed` status.
Lower layers have no feature examples or feature intervention targets because their MLP contributions enter error nodes.
This reduced coverage can increase error share and reduce feature share. Acceptance still requires a feature share of at least 0.30.
The acceptance report must record both measured shares and the fallback scope.

The original and fallback training jobs share the aggregate ten-hour cap, which includes model load, checkpoints, and export.
The fallback adds no time beyond that cap. Its token budget and quality scope are separate from the aggregate time allowance.
The detailed contract is in `docs/reference/upper-half-dictionaries.md`.

The example index uses two million other held-out tokens and keeps the largest 16 activations per feature.
Its memory-mapped table records document, token position, activation count, and maximum.
Window queries run at most 16 windows together and use an LRU cache of 256 features.

## Replacement Model

The method follows [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) and the open circuit-tracer implementation.
Each MLP becomes its transcoder output plus a detached error vector.
RMSNorm denominators and attention mixing coefficients are constants from the prompt forward pass.
Active feature masks are fixed for the local path. Encoder and decoder biases remain explicit terms.
Logit targets use the unembedding row minus the mean unembedding row.

Qwen requires additional frozen factors to make this path linear.
For full attention, freeze the softmax pattern and the sigmoid output gate.
For DeltaNet, freeze queries, keys, decay, beta, the value SiLU gate, the output gate, and normalization denominators.
Keep the value projection, causal convolution, recurrent value transport, and output projection linear.
The value SiLU becomes its input times the recorded sigmoid factor.
This extends the same frozen-attention definition to the actual hybrid architecture. It does not attribute attention routing decisions.

The replacement output must equal the recorded output before any edge calculation.
A test must also prove linearity under source perturbations. Ordinary nonlinear-model gradients do not satisfy this contract.
The tracer never constructs a dense residual Jacobian.
Input interventions at `resid_pre` and `mlp_in` also form part of this affine path.
An input patch or ablation removes the upstream derivative at its selected positions.
An input addition retains that derivative. Recorded constant changes have an explicit intervention bias term.
For each target, it injects its encoder direction at the read point and backpropagates through the frozen path.
A source edge is its activation times the dot product of its decoder and the gradient at its write point.
Token embeddings and error vectors act as direct vector sources.

Trace the top logits with 95 percent cumulative probability, at most ten tokens.
Then visit feature targets in descending estimated logit influence, at most 512 targets.
Normalize absolute incoming edges for each target. Compute influence with 32 power steps and logit probability weights.
Prune to 80 percent node influence and 98 percent edge influence, with caps of 512 nodes and 8,192 edges.
Completeness is measured before display pruning. The report also records omitted influence and error share.
The unpruned accounting preserves bias terms separately from displayed source nodes.

## Tensor Contract

`Tracer(model, tokenizer, dictionary).trace(prompt, max_targets=512, max_nodes=512, max_edges=8192)` returns a `Graph`.
It publishes the five tensors below on the model device. Padding is zero and counts define valid rows.
The shapes never change with pruning. A prompt has at most 32 tokens, including special tokens.
Each trace is one run step. A cache of eight entries uses both prompt and intervention state as its key.

| Tensor | Shape | Type | Fields |
|---|---|---|---|
| `trace.counts` | `(4,)` | int32 | Nodes, edges, tokens, layers |
| `trace.node_i` | `(512, 4)` | int32 | Kind, layer, position, feature or token ID |
| `trace.node_f` | `(512, 4)` | float32 | Activation, influence, top-logit effect, reserved |
| `trace.edge_i` | `(8192, 2)` | int32 | Source, target |
| `trace.edge_f` | `(8192, 2)` | float32 | Weight, weight divided by largest magnitude |

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
All commands enter `src/builder/commands/command_table.cpp`; T6 owns that file across both milestones.
T7 owns intervention execution in `_controls.py`; T6 owns text parameters and interpretation request transport there.

M7 commands are `feature.dictionary`, `feature.show`, `feature.examples`, `trace.run`, `trace.status`, and `trace.read`.
M8 extends the existing intervention commands and adds `patch.sweep` and `trace.validate`.
Read commands do not mutate the model. Change commands use the existing approval and step-boundary paths.
A missing dictionary gives a structured refusal.

The outline reference `tests/scene/draw_reference.h` is absent on the base.
The implemented cross-backend limits are in `web/test/draw_parity.spec.ts`.

Interior pixels differ by at most two channel levels.
The existing edge-side exemption accepts an edge pixel that matches an adjacent reference color.
After this exemption, differing edge pixels occupy at most one percent of the image.
Interior IDs must match exactly. The new piece checks retain these limits.

## Intervention Contract

`ablate` retains its behavior. `add` writes a coefficient times a vector at selected positions.
`scale` writes `(multiplier - 1) * activation * decoder` at the MLP output.
`clamp` writes `(value - activation) * decoder`. Both use the live MLP input and the real model.
`patch` substitutes a cached clean-prompt activation. Hook failure ends the intervention and reports its error.
Fork checkpoints carry the full intervention specification.

Output comparison uses one batch of two: baseline and intervention. It publishes `output.probs` with shape `(2, 10)`.
`output.tokens` has shape `(10,)`. The bars share token IDs and retain a baseline ghost bar.
The generate action emits 24 greedy tokens for each case as bound text.
The feature slider spans negative four to positive four times the indexed maximum, with zero as ablation.
A reset removes the intervention. A retrace toggle controls whether the graph also updates.

The patch sweep uses equal-length clean and corrupted prompts and a two-token logit difference.
It processes at most 32 layer-position cells per batch and publishes normalized recovery as `patch.grid`.
Validation ablates the ten most influential features and reports Spearman correlation and sign agreement.

## Validation And Delivery

The fixture has four layers, width 64, four heads, vocabulary 512, and exact TopK MLPs with 256 latents and k = 8.
Its float64 CPU forward has a saved hash. Fixture tests require no model download and no GPU.
Target completeness must reach 1e-9, and finite-difference edges must agree within 1e-6.
Power iteration must agree with the matrix inverse within 1e-6.
Qwen checks use the `local_model` label and never run in CI.

The hidden acceptance driver uses only `caliper-mcp` to build and operate the instrument.
It waits for events or published steps, captures the graph before and after a feature clamp, and writes `result.json`.
The graph must have 20 to 512 nodes and a token-to-logit path.
Feature influence share must reach 0.30, and completeness residual must not exceed 0.02.
Steering must reduce the top probability by 0.05 or change the top token.
Validation needs at least eight correct signs and a Spearman correlation of at least 0.6.

The trace budget is 20 seconds, the steering budget is one second, and the demo memory limit is 12 GB.
The driver checks node IDs and window bar heights within two pixels. The Qwen top bar must also become shorter after the clamp.
App/browser parity applies to the fixture graph, with the existing draw tolerances.

The 24 November founder decision permits this demo dictionary despite its recorded dead-feature and cross-entropy failures.
A quality override must identify the exact dictionary fingerprint, complete evaluation count, measured values, and failed gates.
The driver displays that decision and retains the failed measurements in its result.
This exception does not change causal, pixel, parity, or budget checks, and it does not block either code milestone merge.

The driver ties graph, example, and bar targets to the window through the actual image quads, clips, and framebuffer scale.
An independent CPU reference uses the actual font atlas and expected glyph quads to check prompt and selected labels.
The reference permits two channel levels of interior difference and at most one percent differing edge pixels.
Each required label also needs visible ink and glyph pixels within two channel levels.
Synthetic failure injections check these oracles. They do not establish application acceptance.

Hosted `trace.status` supplies optional inference evidence with current prompt and intervention hashes.
Its comparison evidence supplies both actual argmax token IDs. The driver does not infer them from the ten bars.
Before Generate, the driver records the prior completion identity.
A fresh completion must bind two rows of 24 token IDs and both text hashes to the current prompt and interventions.
Both the semantic steering gate and the shorter-bar gate remain required for Qwen.

The demo selects the feature with the largest absolute effect on the top logit before the last layer.
The result also records the original influence-based selection and its measured zero-ablation effect from the validation table.
The validation still uses the ten most influential active features and the same sign and rank thresholds.
Positive zero-ablation effects raise the centered logit. The steering and shorter-bar checks retain their direction when such a feature is selected.

Only the founder uses `--show`. Every automated run remains hidden.
The demo states the per-layer dictionary, frozen routing, feature-label heuristic, prompt cap, and lack of global weight analysis.
Fallbacks follow the outline, and every fallback records its effect on coverage and acceptance.

T1 and T5 start after this base. T2 follows T1 and delivers its loader before T3, T4, and T7 start.
T6 follows the T3 and T4 interfaces. T8 follows T5, T6, and T7. T9 follows all tracks.
Integration branches are `m7` and `m8`. Reviews use fresh Codex sessions and Gemini when quota permits.
Each merge requires green macOS and Windows CI, the pre-push checks, and a public-content diff check.

Git cannot store `m7` beside `m7/base`, or `m8` beside `m8/interventions`.
Integration therefore uses `m7-integration` and `m8-integration` while track branches exist.
After the workers finish, archive their branch names under `archive/` and rename each integration branch to its milestone name.
The PR branches remain `m7` and `m8`. No worker changes its assigned branch while its task is active.
