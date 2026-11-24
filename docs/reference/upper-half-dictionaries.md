# Upper-Half Dictionaries

The upper-half fallback trains only the upper MLP transcoders of an existing dictionary.
The lower layers retain zero transcoders with the original dimensions and layer indices.
For the 24-layer Qwen model, layers 12 through 23 train, and layers 0 through 11 become error-only layers.
For other layer counts, the active range starts at `layers // 2`.
An odd layer count gives the upper group one additional layer.
The source must contain at least two layers.

Section 7 of the approved M7 and M8 outline authorizes this fallback after complete source training and failed full quality evaluation.

## Eligibility

The source must complete its full token budget and its separate quality evaluation on 200,000 tokens.
The quality report must record a failure under the existing thresholds.
A partial evaluation or a diagnostic from a training batch cannot establish eligibility.
The source and fallback must use the same model identity, dimensions, activation, `k`, and text source.
The source files remain unchanged.

The completed `evaluation/state.json` must identify the current source dictionary bytes and metadata.
It must record 200,000 tokens, no pending batch, and the same text source and quality partition.
New quality reports also include `dictionary_hash`, which must agree with that identity.
An older report without this field can qualify through its completed evaluation state.
A source weight change invalidates that evidence and causes refusal before model allocation.

The trainer requires `--upper-half-from PATH` and `--prior-wall-seconds NUMBER` together.
The `--output` folder must be separate from the source folder.
A new fallback requires an empty output folder.
An existing fallback requires `--resume`.
The `--tokens` value defines the full token budget for the fallback itself.
The original defaults remain in effect when both fallback options are absent.

## Initialization

The trainer copies upper parameters from the source one parameter at a time.
Each active layer starts with fresh Adam state and fresh feature activity counters.
The fallback token counter starts at zero.
Upper decoder biases come from the source and do not require initialization from a batch.
Lower parameters remain frozen at zero and have no optimizer.
A direct update request for a lower layer fails.

The model forward retains its original MLPs.
The trainer captures inputs and outputs only for active layers, then updates those layers in sequence.
Each exported dictionary still contains every layer file.
Caliper LIB checks both layer partitions and refuses any nonzero lower encoder, decoder, or bias tensor before type conversion.
The example index retains its existing dictionary interface.

## Metadata

The manifest retains format version 1 and its original model hash.
The optional `training.fallback` object identifies the fallback contract.
Its absence identifies the original dictionary contract for all layers.
The object contains these fields:

| Field | Value |
|---|---|
| `version` | `1` |
| `mode` | `"upper_half"` |
| `active_layers` | Ordered upper layer indices |
| `zero_layers` | Ordered lower layer indices |
| `source_dictionary_hash` | SHA-256 over source metadata and every layer file |
| `source_checkpoint_generation` | Final source checkpoint generation |
| `source_training_tokens` | Completed source token count |
| `source_elapsed_seconds` | Largest recorded source elapsed time |
| `prior_wall_seconds` | Prior aggregate time supplied by the operator |
| `aggregate_wall_limit_seconds` | Aggregate cap, at most 36,000 seconds |
| `token_budget` | Full fallback token budget |
| `warm_start` | `"upper_weights_fresh_adam"` |
| `source_id` | Exact text source identity |
| `lower_behavior` | `"zero_dictionary_original_mlp"` |

The source hash includes source training metadata and tensor file bytes.
It excludes the mutable quality report.
The source generation, source hash, and text source identity remain separate because they describe different facts.
The existing `training.tokens` and `training.elapsed_seconds` fields describe only the fallback run.

Each new fallback also contains a top-level `fallback_source` record.
The trainer creates it after all source admission checks pass.
Its `metadata` field contains a deep copy of the complete admitted source manifest, including the failed quality report.
Its `identity` field contains the source `model_hash` and exact `dictionary_fingerprint`.
The source must use all layers and must not contain another `fallback_source` record.

The `evaluation` field contains the verified source cursor projection:

| Field | Value |
|---|---|
| `version` | `1` |
| `count` | `200000` |
| `pending_ids` | `null` |
| `identity` | Exact source fingerprint |
| `settings.tokens` | `200000` |
| `settings.scope` | `"all_layers"` |
| `settings.active_layers` | All source layer indices in order |
| `stream.partition` | `"quality"` |
| `stream.source_id` | Source training text identity |
| `stream.partitions` | Deep copy of source partition rules |

This record preserves evidence verified at admission. It does not claim a second source evaluation after admission.
The current dictionary fingerprint includes the complete record, and the source folder stays unchanged.
The record must fit the service response bounds with its metadata wrapper.
Older diagnostic dictionaries without this record can still load, but each new fallback run must produce it.

## Resume

The first fallback checkpoint contains the warm upper weights and fresh optimizer state.
Later resume restores weights from that checkpoint.
It does not initialize upper weights from the source again.
The trainer checks the unchanged source manifest and fingerprint before model allocation.
Checkpoint settings also bind the resolved source folder, model hash, text source identity, fallback object, and optimizer dimensions.
The resolved source folder stays in checkpoint settings and does not enter the exported manifest.

Resume also compares the complete admitted `fallback_source` record before model allocation and before training.
A missing or changed source record causes refusal. Checkpoints and exports retain the admitted record without substitution.

A changed active range, source, token budget, warm-start rule, or aggregate cap causes refusal before model allocation.
The `pending_ids` field retains an unfinished model batch.
The `next_layer` field uses a global layer index and advances only after a complete active update.
Resume skips completed active layers, and the batch token count advances only after all active layers finish.
Checkpoints retain lower weight and activity files, but only active layers have Adam state.

## Aggregate Time

The operator supplies the aggregate time for all earlier training jobs through `--prior-wall-seconds`.
That value must be at least the largest elapsed value from the source manifest, checkpoint, and runtime record.
The available fallback allowance equals `max_wall_seconds - prior_wall_seconds`.
Resume also subtracts elapsed fallback time.
The cap includes model load, checkpoints, and dictionary export.
The trainer refuses a new invocation when its remaining allowance cannot cover the model-load and checkpoint reserve.

The fallback grants no additional time beyond the aggregate ten-hour cap.
The final training record uses `fallback_complete` only when the full fallback token budget completes.
An earlier clean stop records `fallback_incomplete` and retains resumable state.
These training states do not establish dictionary quality.

## Quality Scope

The evaluator computes FVU and dead-feature fraction only across active upper layers.
It retains the global layer indices in its arrays and uses null metrics for lower layers.
The `per_layer_status` array labels those lower layers `error_only`.
The report includes `scope: "upper_half"` and `active_layers`.
The FVU, dead-feature, and cross-entropy thresholds remain 0.30, 0.15, and 0.5 nats.

For cross-entropy, the evaluator replaces only upper MLPs and retains the original lower MLP computation.
The `replaced_mlps` field records the actual replacement count.
Every lower block precedes every active block, which preserves the intended error-only lower path.
The evaluator restores every replaced module after success or an exception.

A complete evaluation reports `fallback_passed` or `fallback_failed`.
After artifact admission, a shorter or interrupted evaluation reports `fallback_partial`, including an interruption before the first batch.
A fallback never receives the original `passed` status.
Its acceptance scope remains `upper_half`, separate from full MLP acceptance.

Evaluation admission compares each runtime tensor with bounded artifact slices after conversion to the runtime dtype.
Resume binds the tensor digest, dictionary fingerprint, active layer indices, scope, and evaluation dimensions.
A legacy cursor without the tensor digest cannot resume. The public dictionary fingerprint format remains unchanged.
The caller must keep exclusive ownership of the dictionary throughout evaluation.

A stop before artifact admission preserves the prior quality report and cursor. Stream creation alone does not admit evaluation.
After admission, an all-layer evaluation that stops before two tokens records `partial`, null metrics, and the current dictionary fingerprint.
Both scopes emit `quality_end` and replace an earlier report only when final artifact verification succeeds.
If memory admission prevents final verification, the evaluator preserves the last published evidence.
All-layer thresholds remain unchanged for dictionaries without fallback metadata.

## Graph Coverage

Lower-layer MLP contributions enter error nodes because their transcoders remain zero.
The lower layers have no feature examples or feature intervention targets.
This reduced coverage can increase `error_share` and reduce `feature_share`.
Acceptance still requires a feature share of at least 0.30.
The acceptance report must state the fallback scope and its measured feature share and error share.

The CPU tests use tiny synthetic tensors and artificial status records.
Those checks establish implementation behavior and provide no quality evidence for a trained dictionary.
An actual fallback launch requires the completed source quality report and the operator's decision within the remaining aggregate allowance.
