# Interpretation Tools

Caliper MCP queries feature dictionaries, example indexes, and attribution graphs through the attached run.
The model, dictionary, and index stay in the Python process.
The app sends bounded requests and retains their replies.
The app does not wait for a model operation on its frame thread.

## Commands

Each command has one MCP tool. Replace the dot with an underscore, as in `feature_examples`.
Read commands keep the `read` effect. `trace.run` has the `run` effect and uses the existing approval policy.
A read-only stream accepts interpretation reads. It refuses trace changes.

| Command | Request Fields | Result Data |
|---|---|---|
| `feature.dictionary` | None | Registered dictionary metadata |
| `feature.show` | `layer`, `feature`, or `node` | Label, count, maximum, promoted tokens, example tokens |
| `feature.examples` | `layer`, `feature`, or `node`; optional `limit` | Real example windows and tensor bindings |
| `trace.run` | `prompt`; optional `max_targets`, `max_nodes`, `max_edges` | Trace state, step, duration, counts, diagnostics |
| `trace.status` | None | Last trace state and diagnostics |
| `trace.read` | Optional node and edge offsets and limits | Graph records, counts, diagnostics, next offsets |

Every command returns `request_id`, `pending`, and `result`.
When `pending` is true, `result` is null.
Call the same command with only `request_id` to read its retained reply.
This call does not start another operation.
For a long trace, `trace.status` also reports `pending`, `running`, `done`, or `error` in its result data.
A successful queue response does not imply successful execution.

Feature queries accept either a graph row in `node`, or both `layer` and `feature`.
The node must identify a feature in the current completed graph.
Examples retain their indexed order and stable `example_id` values.
Each of at most sixteen examples has 33 token slots, with null values at document boundaries.
The response keeps full token text and real per-token activations.

A prompt has at most 4,096 UTF-8 bytes and 32 model tokens, including special tokens.
The tracer refuses excess tokens without truncation.
The maximum values are 512 feature targets, 512 nodes, and 8,192 edges.
A graph page has at most 256 nodes and 256 edges.
The default page has 128 nodes and 256 edges.

## Run Service

A script registers the service before it starts its stream.
The service accepts the actual dictionary, `FeatureIndex`, and `Tracer`, or lazy factories for those objects.
A factory takes no arguments and runs once on the run thread.
The public `caliper.interp.Tracer` import remains lazy.

```python
from caliper.interp._service import register_service, unregister_service

service = register_service(
    model=model,
    tokenizer=tokenizer,
    dictionary=dictionary,
    index_factory=lambda: FeatureIndex(dictionary_path, model=model,
                                       tokenizer=tokenizer, dictionary=dictionary),
)
try:
    # The script calls caliper.step() at each run boundary.
    run_loop()
finally:
    unregister_service(service)
```

`register_service` returns an `InterpretationService`.
An explicit `controls` argument selects a control owner for an embedded run.
The service queues at most 32 requests.
`Controls` applies pending parameters and interventions before it calls `service.step(step)`.
The service rejects recursive execution during internal model forwards.
An inference runner must also suppress its own forward checkpoint callback during those internal forwards.

`trace.status` reads a cached result without a model forward.
Other requests execute at the next boundary and send a deferred reply.
Feature queries suspend active interventions when the run supplies an intervention service.
Thus example activations refer to the indexed baseline.
With legacy ablation hooks, feature queries return `interpret.baseline_required` before index access.
Clear those ablations at a step boundary before the query.
Unregistration refuses queued requests and waits for active work and its final reply.
Service replacement also waits before it changes the registered owner.
Call these lifecycle operations from the script, outside an active service request or an app event loop.
An active request cannot re-register, replace, or close its current service and receives `interpret.busy` if it tries.
An idle service can register again with the same owner.
Resource ownership remains with the script. Close the index after unregistration returns.

Missing resources produce structured errors, including `interpret.missing_dictionary`, `interpret.missing_index`, and `interpret.missing_model`.
An absent service or capability produces `interpret.unsupported`.
An absent graph produces `interpret.missing_trace`.
No missing resource produces a successful result with invented data.

## Example Panel Bindings

A graph pick exposes its display row as `index`.
Use the existing pick action to set the shared integer state `selected_node` to that value.
Bind the graph parameter `selected_node` to the same state.
After the pick event, read that state and call `feature.examples` with `node`.
When its reply completes, the run publishes the tensor sources below.

| `token_strip@1` Input | Source | Shape |
|---|---|---|
| `tokens` | `live:examples.tokens` | `(16, 25)` |
| `activations` | `live:examples.activations` | `(16, 25)` |
| `example_ids` | `live:examples.example_ids` | `(16,)` |
| `valid_tokens` | `live:examples.valid_tokens` | `(16,)` |
| `label_text` | `live:examples.label_text` | `(400, 16)` |

The strip displays eight rows and 25 columns.
Its `page` parameter accepts zero or one, which exposes both sets of eight examples.
The service selects the central 25 slots and removes document-edge padding from each visible prefix.
It preserves the original 33-slot windows in the command result.
Missing example rows have ID minus one and no valid tokens.

Packed labels use sixteen little-endian `u32` words per token.
Each label has at most 63 UTF-8 bytes and a zero terminator.
The service cuts a label only at a UTF-8 boundary.
Each visible page has 200 labels, below the renderer limit of 512.
Set `activation_max` from the selected feature maximum, with a positive minimum of `1e-6`.

Use `graph_counts`, `node_i`, `node_f`, `edge_i`, and `edge_f` for the graph inputs.
Their sources are `live:trace.counts`, `live:trace.node_i`, `live:trace.node_f`, `live:trace.edge_i`, and `live:trace.edge_f`.
The graph and example strip use the existing pieces and shared state tools.
The optional graph `label_text` input uses `live:trace.label_text`, with shape `(544, 16)`.
Rows zero through 31 contain prompt tokens. Row `32 + index` contains the node label.
The graph exposes at most 34 labels, and the token strip exposes 200 per page.

Use `logit_bars@1` for the traced baseline distribution:

| Input | Source | Shape |
|---|---|---|
| `probs` | `live:trace.probs` | `(2, 10)` |
| `tokens` | `live:trace.logit_tokens` | `(10,)` |
| `label_text` | `live:trace.logit_label_text` | `(10, 16)` |

Both probability rows contain the same traced values and share token order.
Unused slots have zero probability, token ID minus one, and an empty label.
These tensors come from graph logit records without another model forward.
The graph, strip, and bars use at most 244 visible labels together.

## Text Parameters

`caliper.param("prompt", "")` registers a free-text parameter.
A string with `choices` retains the existing choice behavior.
A free-text value accepts at most 1,024 UTF-8 bytes, including spaces and an empty value.
Invalid UTF-8 and embedded zero characters are refused.
A text parameter does not accept numeric bounds or a logarithmic scale.

An app control with `kind = "text"` accepts `bind = "param:prompt"`.
It shows the script value or its pending value.
Return or loss of focus submits the edit at the next run boundary.
The control is disabled when the parameter is absent, has another type, or the connection cannot change the run.
The existing `value` and `state:` text bindings retain their behavior.
The text control limit is 1,024 bytes, while a direct `trace.run` prompt accepts 4,096 bytes.

## Structured Wire Metadata

The extension uses the existing CONTROL and CONTROL_REPLY frame types.
Both peers announce the `interpretation` capability.
A request uses `op = "interpret"` and the object below as `interpretation`.

```json
{"version": 1, "method": "feature.examples", "params": {"layer": 0, "feature": 7}}
```

A reply retains the request ID and uses `queued`, `ok`, or `error`.
A final successful reply carries a structured object in `result`.
An error carries its code and message in `error`.
The extension never stores JSON in a text scalar.
Older peers retain their original controls and cannot send interpretation requests.

Each structured object has a 256 KiB byte limit.
It permits at most eight nested levels, 64 fields per object, 512 entries per list, and 16,384 total values.
Strings have at most 4,096 UTF-8 bytes.
Numbers must be finite. Integer magnitudes cannot exceed `2**53 - 1`.
Method-specific limits apply before the run queues work.
Both native and Python codecs enforce these limits.

The separate `text_params` capability enables the text parameter type.
A producer or hub omits text parameters from CONTROLS for an older peer.
Numeric, Boolean, and choice parameters retain their original types and behavior.

## M8 Command Schemas

The command table also describes the M8 operations.
`intervene.set` and each `run.fork` intervention accept the same specification.
Supply `point` and either `op` or `kind`. When both discriminators are present, their values must match.
The `steer` value stays reserved and produces `intervene.op_reserved`.

| Kind | Required Fields | Optional Fields |
|---|---|---|
| `ablate` | None | Step range |
| `add` | `vector`, `coefficient` | `positions`, step range |
| `scale` | `feature`, `multiplier` | `dictionary`, `positions`, step range |
| `clamp` | `feature`, `value` | `dictionary`, `positions`, step range |
| `patch` | `source_prompt` | `source_position`, `positions`, step range |

A step range uses `from_step` and `until_step`. The end step is exclusive and must exceed the start step.
Null or absent steps use the next boundary and an indefinite end.
A vector has one through 16,384 finite values, and its length must match the point width.
Positions contains at most 32 unique integers from zero through 31.
Its omission selects all positions. An empty list selects none.
A patch prompt has at most 4,096 UTF-8 bytes and 32 model tokens.
A dictionary identifier has at most 255 UTF-8 bytes.
Each specification and each fork specification list has a 256 KiB limit.
Integer magnitudes cannot exceed `2**53 - 1`. The executor also checks model dimensions and numeric range.

`patch.sweep` accepts `clean_prompt`, `corrupted_prompt`, and `metric`.
The metric has distinct integer `positive_token` and `negative_token` IDs from zero through `2**31 - 1`.
Both prompts have at most 4,096 UTF-8 bytes. The executor checks their token counts.
`trace.validate` accepts no operation fields and uses the latest trace snapshot.
Both commands have the `run` effect and use the same pending request protocol as `trace.run`.
A reply query contains only `request_id`.

The Python service dispatches these two commands through its `handlers` mapping.
A handler receives the exact request fields as keyword arguments and returns a bounded JSON object.
The service calls it on the run thread at a step boundary.
Without a registered handler, the service returns `interpret.unsupported`.
The transport does not calculate a sweep or invent validation results.

An inference start has the form below:

```json
{"kind": "inference", "model": "local-model", "dictionary": "local-dictionary", "prompt": "AB"}
```

The optional `id` sets the managed run ID.
Model and dictionary resource names have at most 1,024 UTF-8 bytes. The prompt has at most 4,096 UTF-8 bytes.
These fields require the hosted inference runner.
The existing script form retains `script`, `args`, `params`, and `id`, with an optional `kind` value of `script`.
Script starts refuse `model`, `dictionary`, and `prompt` fields. Inference starts refuse `script`, `args`, and `params` fields.
Both forms require run approval.
The inference adapter accepts a local model folder or the `tiny-lm` fixture identifier, and a local dictionary folder.
Relative resource paths start at the workspace. The adapter refuses missing folders before it requests approval.
The hosted runner owns resource checks, model loading, inference output, and checkpoint restoration.
