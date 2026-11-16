# Widget agent

Caliper's agents have two roles. The **side chat** is the general chat pane: the person's own agent runs there as a
subprocess over ACP, with `caliper-mcp`, for conversations and bigger jobs such as building from scratch, loader
scripts, project code and explanations ([ACP integration](acp-integration.md)). The **widget agent** is an inline agent
that works on one target at a granular level, a panel, a selection, a dropped file or a property, without opening a
coding-agent session. This page designs the widget agent.

The widget agent's process is `caliper-agent`, the sidecar that [Widget agent sidecar](byok-agent.md) describes. The
loop, the Claude settings, the key handling and the tests on that page are built, and this page adds the inline role
on top of them. The person's side is in Agent roles (`docs/explanations/agent-roles.md`) and
Set up the widget agent (`docs/guides/set-up-the-widget-agent.md`).

**Built on this branch:** the sidecar, its tests and the key store, which that page lists; stacked previews in the
session ([below](#preview-and-apply-on-the-command-layer)); the `_meta` argument of `AcpClient::new_session()` that
carries the widget session's fields; and widget sessions in the sidecar (item 12): the target, the tool scope from one
rule kept as data, the preview header, the widget system prompt, images and embedded context in prompts, the turn
budget, the handoff and `session/close` ([How it stays an ACP agent](#how-it-stays-an-acp-agent)). **Planned:**
everything else on this page. The [work items](#work-items) give the
sizes and the order.

## The role

The widget agent answers one request about one target in a few seconds. The app gives it what it needs to know about
the target, it may use only the commands that apply to that target, and every change it proposes is drawn in place as
a preview that the person accepts or rejects. Accepted, the change is one undo step.

| Entry point | Example request | What happens |
|---|---|---|
| A panel | "Use a linear y axis" | A preview of the panel with the new axis, and the diff of the spec |
| A selection | "Make a table of these points" | A new table panel of the selected rows, drawn where the layout would put it |
| A dropped file | "What's in this?" | A short answer from the file's schema and a sample, with no change |
| A property | "Use the diverging colormap" | The property changes in the preview |

Work that does not fit one target and a few seconds belongs to the side chat: building a dashboard from nothing,
writing a loader, changing project code or explaining at length. "Ask in chat" hands the request over with its
context, as [below](#handoff-to-the-side-chat) describes.

## Entry points

| Entry point | How the person opens it | Target |
|---|---|---|
| ⌘K on a panel | ⌘K (Ctrl+K on Windows) while a panel has focus, or the panel's menu | The panel |
| On a selection | ⌘K while points or a range are selected in a panel | The selection and its panel |
| On a dropped file | The prompt line shown under a file that was just dropped on the window | The loader that the drop created and its outputs |
| In the builder's properties | The ⌘K button beside a property in the builder pane's property inspector | One property of a panel or a source |

A drop imports the file through `data.import` from the `ui` origin before the prompt appears, so the widget agent
always works on data that is already in the catalog.

## The inline prompt and the preview

The prompt opens inside the target's panel, at its bottom edge, so the panel stays visible:

```
┌─ Loss and accuracy ──────────────────────────────── ⋯ ┐
│                                                       │
│   the panel as it is now                              │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │ ⌘K  Use a linear y axis                       ⏎ │  │
│  │     Claude Haiku 5.5 · low          Ask in chat │  │
│  └─────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────┘
```

When the agent has finished, the panel is drawn from the preview and a card shows the agent's one-sentence summary
and the diff of the spec's canonical TOML:

```
┌─ Loss and accuracy ───────────────── Preview ────── ⋯ ┐
│                                                       │
│   the panel drawn from the preview                    │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │ Turns off the log scale of the y axis.          │  │
│  │ - options = { x_label = "Step", y_log = true }  │  │
│  │ + options = { x_label = "Step" }                │  │
│  │                                                 │  │
│  │ [Accept ⏎]   [Reject esc]   [Refine…]           │  │
│  └─────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────┘
```

- **Accept** applies the change as one undo step. **Reject** discards it and leaves no trace. **Refine** keeps the
  prompt open for a follow-up, which continues the same conversation on top of the current preview.
- A change that adds a panel, such as a table of a selection, draws the new panel in a dashed frame where the layout
  would place it.
- A diff longer than 12 lines is folded, with a control that shows it all.
- The agent's summarized thinking is shown folded above the card, as the side chat shows thoughts.

## The context bundle

The app builds the context bundle from the session before the first request and sends it in the prompt. It tells the
model what it would otherwise have to discover with read commands, which is what lets most requests finish in one or
two rounds.

| Part | What it holds | How it is built | Bound |
|---|---|---|---|
| Spec fragment | The target panel's canonical TOML, the sources it binds with their `from` chains, the panel's place in the layout, and the ids and titles of the other panels | From the open spec (`Session::open_spec()`), following each binding's chain of `from` as `Session::resolve()` does, at most 64 hops | 16 KiB; at most 64 other panels listed |
| Schema and sample | For each catalog id at the root of those sources, its catalog entry (columns and types, or dtype and shape, and the rate) and a sample | `catalog.describe`, and `catalog.sample` with `limit` 20 for a table, and `start` and `count` chosen to take at most 1,024 values of a tensor | 8 KiB for each source, at most 4 sources |
| Render | The panel as it is drawn now, as a PNG | `panel.render` at 640 by 400 | 1 MiB; left out when the model has no vision |
| Selection | The panel, whether the selection is points or a range, the selected rows with the columns the panel binds, and the range in data units | The app's selection state | 1,000 rows or 32 KiB |

The bundle changes with the target:

- **A dropped file** has no panel yet, so the bundle holds the loader that the drop created (`loader.describe`) and the
  schema and a sample of each of its outputs, in place of the spec fragment and the render.
- **A property** adds the property's path (for example `panels.attention.options.colormap`), its current value and its
  part of the command's arguments schema from the command table (the enumeration or the bounds of that option in
  `panel.patch`), beside the panel's spec fragment.

**Sized to the model.** The bundle takes at most a quarter of the model's context length, estimated at four characters
per token, and never more than 64 KiB of text and one image. When it does not fit, the app cuts parts in this order:
the sample rows, the list of other panels, the render, and then the sources of the spec fragment. It never cuts the
target itself. Each cut is marked in the bundle, so the model knows that something was left out.

Everything in the bundle is workspace data. The widget agent's system prompt says that it never gives instructions,
as the sidecar's system prompt already says of tool results.

## The tool scope

The tool scope is the subset of the command table that the widget agent may use for one target. The app derives it
from the table for each kind of target, sends it when it opens the session, and the sidecar enforces it.

The rule:

1. **Reads and previewable edits only.** A command that mutates must take a boolean `dry_run`, read from its arguments
   schema in the way `tools.py` already finds previewable commands. A destructive command never qualifies.
2. **Nothing that replaces or leaves the open spec.** `spec.set_text`, `spec.new`, `spec.open`, `applet.open` and
   `layout.set` replace the whole document or the whole layout, and the `history`, `applet`, `loader` and `data.import`
   commands act outside one target, so none is in any scope.
3. **Then by the kind of target**, as the table gives.

| Target | Reads | Edits, each with `dry_run` |
|---|---|---|
| A panel | `spec.get`, `spec.validate`, `catalog.describe`, `catalog.sample`, `panel.render` | `panel.patch`, `panel.remove`, `layout.place`, `source.add`, `source.patch` |
| A selection | The panel's reads and `data.query` | `source.add`, `panel.create`, `layout.place` |
| A dropped file | `spec.get`, `catalog.list`, `catalog.describe`, `catalog.sample`, `loader.describe`, `data.query` | `source.add`, `panel.create`, `layout.place` |
| A property | `spec.get`, `catalog.describe` | `panel.patch` for a property of a panel, `source.patch` for a property of a source |

`panel.render` is in a scope only when the model has vision. A tool outside the scope is never sent to the model, and a
call to one is answered as a call to an unknown tool, which the loop already does; the call path refuses a command
outside the session's scope as well (`agent.out_of_scope`), whatever the model names.

**Built: the rule as data.** The rule's lists are one file, `src/builder/commands/widget_scopes.json`, beside the
command table: for each kind of target its reads and its edits, the commands that are never in a scope (`spec.set_text`,
`spec.new`, `spec.open`, `applet.open`, `layout.set`, `data.import`) and the groups that are never in one (`history`,
`applet`, `preview`), and `panel.render` as the one command that needs vision. Both the sidecar
(`caliper_agent/widget.py`, which reads it at start) and the app's context bundle (which embeds it at build time) apply
the same rule to the command table, so the two agree from one definition, and the rule reads each command's flags and
arguments schema rather than a list of schemas: a listed read that mutates, or a listed edit that loses its boolean
`dry_run` or becomes destructive, drops out of the scope. A property's scope depends on its owner, so the file has a
`panel_property` and a `source_property` entry. The sidecar is stricter than this page first planned: a scope that the
app sends must be a subset of what the rule gives for the target, or `session/new` is refused with `agent.bad_widget`,
and without a scope the sidecar uses the rule's whole scope.

When no spec is open, the app runs `spec.new` from the `ui` origin, with the file's name as the title, before it
begins the preview, since a preview needs an open spec; on Reject it undoes that `spec.new` as well.

## The turn budget, the model and the effort

- **Two rounds.** In the first round the model reads the bundle and proposes its edits on the preview. In the second it
  sees their results and, with vision, a render of the preview, and either finishes with one sentence or corrects an
  edit. A prompt makes at most 3 model requests and 8 tool calls, against 50 requests in the unscoped mode, and asks for
  at most 8,000 output tokens. A prompt that reaches a limit ends with the stop reason `max_turn_requests`, and the
  card offers Refine and Ask in chat.
- **Claude: `claude-haiku-5-5` at effort `low`.** Of the four model ids that the sidecar offers, Claude Haiku 5.5 is
  the fastest and the cheapest, it takes adaptive thinking, and it takes the effort `low`; its default effort is
  `medium`, and the provider sets the effort explicitly on every request, as it already does. A widget request is a
  small, bounded edit with its context given in full, and low effort makes fewer and more consolidated tool calls, which
  suits it. Server-side refusal fallbacks are off for every model ([Refusals](#refusals)), so a refusal ends the
  prompt, and the card says that the model declined the request, with the three ways forward, and offers Ask in chat.
  The person can choose any of the four models and any effort in the widget agent's settings.
- **Other models**, through the provider layer below, have no default: the person picks a model for each preset.
  The same budget applies, and an effort is sent only where the server takes one.

## Preview and apply on the command layer

A request often needs edits that build on each other, such as a derived source and then a panel that binds it. A dry
run checks one edit against the open spec, so the dry run of the second edit would fail. The session therefore has
**stacked previews**, built in the session and its command table
([Command layer](command-layer.md#preview), [Builder session](builder-session.md#stacked-previews)):

- **A preview** has an id and a copy of the open spec that starts equal to it. The app begins it with `preview.begin`
  from the `ui` origin when it opens the widget session, scoped to the target's panel or source, and puts its id in the
  session's `_meta`. When no spec is open, as for a dropped file, the app first runs `spec.new` from the `ui` origin.
- **An edit that names the preview**, through the header `X-Caliper-Preview: <id>`, applies to the copy instead of the
  open spec, validated as a real edit, and returns its diff against the copy as it was before the edit. Its handler
  reports `no_effect`, so the history and the journal are untouched, and the command layer's review does not ask about
  it. The sidecar sets the header on every call in a widget session; the model never sets it and is never offered the
  preview commands. The sidecar sends every edit without `dry_run`, and its tools do not offer the argument, because
  an edit that names a preview with `dry_run` only checks the change against the copy and does not stack it; the
  earlier plan here, a dry run with the header on every edit, would have left the preview empty.
- **`spec.get`, `spec.validate` and `panel.render` with the preview's id** read the copy, so the model can check its
  work, and the app draws the same copy in the panel's frame for the person with `Session::resolve(panel, preview)`
  and `Session::preview(id)`.
- **The session records** the edits that changed the copy, in order, at most 64 for one preview and 4 MiB of their
  arguments. A scoped preview takes new sources and panels and edits only of its target, of the sources a target panel
  binds and of what it added (`preview.out_of_scope` otherwise), which backs the sidecar's tool scope in the session.
- **The open spec may change meanwhile**, through the side chat or the person. The preview stays the open spec with
  its edits applied: they are applied again to the new open spec before the preview is read, and when one no longer
  applies the preview answers `history.conflict`.

The three answers of the card map onto the session:

- **Accept.** The app's own UI sends `preview.commit` from the `ui` origin. The session applies the recorded edits
  again to the open spec and refuses with `history.conflict` when one no longer applies, after which the card offers
  Refine on a new preview, begun from the current spec. Otherwise the edits run as the registry's follow-ups, each
  with the origin that sent it, `agent`, and with the preview's id as their group. They are validated and journaled
  like any agent's commands, after the commit's own line from the person, and they undo as one step. Only `ui` and
  `builder` may commit; `preview.commit` from any other origin answers `preview.not_allowed`, so an agent cannot
  accept its own proposal, and no approval policy is asked for the edits, because the commit is the person's approval.
- **Reject.** `preview.discard` drops the copy. The edits wrote no journal line and filed no undo step, so the spec, the
  history and the journal are as they were. Only the conversation remains, in the sidecar's memory, until the session
  closes.
- **Refine.** The person's follow-up is a new `session/prompt` in the same ACP session, so the model keeps the
  conversation, and the preview keeps its id, so the new edits stack on the current proposal. The model can take back
  part of it with another edit, for example `panel.remove` of a panel it proposed.

**How the sidecar asks for Accept, and how the app answers.** The sidecar never commits. It ends its turn with
`end_turn` and its one-sentence summary as the last message; the app, which began the preview and knows its id, reads
`preview.diff` and draws the card with the summary and the diff. Accept, Reject and Refine are the person's clicks: the
app sends `preview.commit` or `preview.discard` itself, from the `ui` origin, and Refine is the next prompt. The
outcome reaches the sidecar as the next prompt's text, or as `session/close` when the prompt closes.

A preview ends with Accept, Reject, a handoff, the prompt closing, or 30 minutes without use, after which a call names
it as `preview.expired`. In a widget session the sidecar sends no `session/request_permission` for edits, because the
preview card is the review, and no destructive command can be in the scope.

## Handoff to the side chat

"Ask in chat" sends the request to the side chat's agent as a `session/prompt` in its current session, starting the
configured agent first when the side chat is not open. The prompt carries:

1. **The person's request**, as a text block.
2. **The context bundle**, in the same parts. When the side chat's agent declares `promptCapabilities.embeddedContext`
   in `initialize`, each part is a `resource` block with embedded text; otherwise it is a text block. The render goes as
   an `image` block only when the agent declares `promptCapabilities.image`. Every block of the bundle carries
   `annotations.audience: ["assistant"]`, which opencode documents as context for the model outside the person's
   message; an agent that ignores annotations reads it as ordinary text. `AcpClient::prompt_blocks()` already sends
   such blocks, and `AgentInfo` already holds the two capabilities.
3. **The widget conversation so far**, as text: each request and reply, and the open preview's diff, marked as not
   applied.
4. **The target's ids**: the panel, its sources and their catalog ids, so the agent can act on them through
   `caliper-mcp`.

**How the widget agent hands over.** When a request does not fit its target or its tools, the model calls the widget
session's one tool of its own, `handoff`, with a `kind` (`build_from_scratch`, `loader`, `project_code`, `explanation`,
`another_target` or `outside_scope`) and one sentence for the person; the system prompt tells it to make no edit then.
The sidecar runs no other call of that reply and sends no further model request, ends the turn with `end_turn`, sends a
`handoff` status (a thought whose `_meta.caliper.status` is `handoff`, with the kind as `category` and the sentence as
`reason`), and puts `{"kind", "reason"}` in the `handoff` field of the prompt answer's `_meta.caliper.widget`, which
the app turns into the card's Ask in chat. The tool is not a command, and its name has no underscore, so it can never
be taken for one.

The open preview is dropped when the handoff is sent. The side chat's agent makes whatever change is wanted through
commands, where the command layer's approvals apply to it as to every agent. The side chat shows the handed-over prompt
as the person's message, so the person sees what was sent. When no side chat agent is configured, the card links to the
setup steps instead.

## With no provider set up

The widget agent needs a provider: a Claude API key today, and later any configured preset, where a local server needs
no key. When none is set up, ⌘K still opens the inline prompt, but its only action is Ask in chat, with the same
handoff, and one line says that the widget agent needs a provider and links to
Set up the widget agent (`docs/guides/set-up-the-widget-agent.md`). The sidecar is not
started.

## One journal, one history, one queue

- **One queue.** Both roles reach the command layer through Caliper API with the origin `agent`, the side chat through
  `caliper-mcp --origin agent` and the widget agent through the sidecar's own client. The app drains one command queue
  on its frame thread, so the commands of both roles, the builder's and the person's run in one order.
- **One journal.** Every accepted widget change is journaled under the origin `agent`, with the preview's id as its
  group, which the app makes start with `widget-`, so its lines can be told apart from the side chat's.
- **One undo history.** An accepted change is one step on the same stack as the side chat's changes and the person's
  own, so undo from anywhere takes back the newest step first.
- **Both at once.** A widget prompt works while the side chat's agent runs a turn, because its edits touch only its
  preview. If the side chat changes the open spec in the meantime, the preview's edits are applied again to the new
  spec; Accept answers `history.conflict` only when one of them no longer applies, and Refine then starts a new preview
  from the new spec.

## How it stays an ACP agent

The widget agent is the same `caliper-agent` process, driven by the same `AcpClient`, with the same session model,
messages, thoughts, tool calls and cancellation as the side chat's agents. The inline surface needs two things from
the session model that the chat pane did not:

1. **A context bundle in the prompt.** It travels as ACP prompt content: `resource` blocks with embedded text and one
   `image` block. The sidecar declares `promptCapabilities.image` and `promptCapabilities.embeddedContext` and turns
   text, embedded text resources and resource links into text and an image into an image for a model that reads images,
   or into a line saying that the picture was left out for one that does not. A block whose `annotations.audience`
   names only the person is not given to the model. A prompt holds at most 4 images of at most 3.75 MB each, PNG, JPEG,
   GIF or WebP, checked by their bytes.
2. **A target, a tool scope and a preview for each session.** They travel in `_meta.caliper.widget` of `session/new`,
   for example
   `{"target": {"kind": "panel", "panel": "curves"}, "scope": ["spec.get", "panel.patch"], "preview": "widget-7f3a"}`.
   The model and the effort stay config options, as today; a widget session offers no `review` option. The sidecar
   checks the field as hostile input (its size and depth, the kind, every id against the spec's id grammar, the
   property path and the preview id against the command table's pattern) and refuses the session when it cannot be
   used, rather than falling back to the general role. Its answer echoes the target, the scope and the preview in its
   own `_meta.caliper.widget`, and its session id starts with `widget-`. Each prompt's answer carries
   `_meta.caliper.widget` with the preview, the number of edits that changed it in that prompt, the handoff or null, and
   whether the turn reached its limit.

**Decision: a `_meta` field of `session/new`, not a launch option.** The target and the scope belong to one session,
and `session/new` is where a session is made. One sidecar process serves up to 16 sessions; a launch option would need
a process for each target and would pay the interpreter's and the SDK's start at every ⌘K, which the budget of a few
seconds cannot afford. `_meta` is the protocol's extension point on every request, so a strict ACP peer accepts it.
`AcpClient::new_session()` takes an optional `_meta` argument for it, built and bounded at 64 KiB, and passes it
through unchanged. A session without the field keeps today's unscoped behaviour, which the tests and the manual check
use and the app does not.

The sidecar also implements `session/close`, behind the `close` session capability, so the app can end a widget
session when its prompt closes: it cancels a running turn, waits up to 10 seconds for it to end and forgets the
session, and leaves the preview to the app, which began it and drops or commits it. `session/cancel` likewise leaves
the preview as it is. A seventeenth session is refused, as before, since the app can now close the ones it is done
with. A widget session uses a second fixed system prompt, shorter than the chat prompt: act on one target, propose with
the tools, answer in one sentence, never invent an id or a column, and send anything outside the scope to the chat. For
Claude, the tools of one kind of target and this prompt are the same at every ⌘K, so the cached prefix carries over
from one prompt to the next while the cache lasts; the bundle comes after the cache breakpoint.

## Providers

The widget agent is model-agnostic, and its provider layer is Pydantic AI, adopted after a spike that rebuilt
`caliper-agent`'s loop on it: Claude runs on Pydantic AI's native Anthropic support, with the settings of
[Claude API settings and why](byok-agent.md#claude-api-settings-and-why), and every other model, hosted or local, runs
through its OpenAI-compatible support against a configurable base URL. Both are built in the sidecar; the settings that
offer the presets are planned. The [Widget agent sidecar](byok-agent.md#providers) page gives the evidence for each
check, the footprint and the reasons the ACP Python SDK is used in tests only.

| Check | Result |
|---|---|
| Cancellation in the middle of a stream, with every pending tool call given a result | Yes, through Pydantic AI's `AgentRun.cancel()` |
| Retry state (the reason, the attempt and the wait) visible to the loop, so the app can show it | Yes, through Pydantic AI's `on_model_request_error` hook |
| Claude's features: adaptive thinking, effort, prompt caching, compaction, strict tools, eager input streaming | Yes, all native; server-side refusal fallbacks are off ([Refusals](#refusals)) |
| Stop reasons mapped to ACP's, with a refused turn cut from the history | Yes |
| Other models: no tool calling, no vision, lenient servers | Yes, against a fake OpenAI-compatible server in the tests |
| Footprint | 22 packages and 57 MiB, against 16 packages and 27 MiB for the Claude SDK alone |

- **At most 20 strict tools in one request.** The Claude API compiles a request's strict tool schemas together and
  refuses more than 20 strict tools, 24 optional parameters or 16 union-typed parameters in all, so the sidecar picks
  the strict tools within those limits; the others are checked against their full schema like every call. A widget
  scope is small, so its tools will usually all be strict.
- **One run on a local model.** During the spike the dashboard prompt of the acceptance test ran once, on the
  OpenAI-compatible route, against Qwen 3.8 27B on Ollama, which was already running, and built and saved the
  dashboard in 135 seconds. It is one manual run of one model, not a guarantee; the conformance suite below is where
  models are compared.

### Refusals

A model can decline a request, and a refusal can be a false positive: machine learning in the physical sciences
touches chemistry, biology and security data.

- **Server-side fallbacks are off.** The Claude API can run a declined request again on another model and mark the
  switch with a `fallback` block in the reply. Pydantic AI drops that block, and after a fallback in the middle of a
  reply the next request needs it, so it would be rejected. The sidecar therefore does not ask for fallbacks.
- **What the person sees.** The prompt ends, nothing of the declined reply is applied, and the card says plainly that
  the model declined the request and offers three ways forward: rephrase it, try it again, or switch to another model
  or provider, such as a local one; Ask in chat stays available. The text names no category and blames no one. The
  declined prompt leaves the conversation, as ACP's `refusal` stop reason means.
- Fallbacks can be switched on, with one switch in the sidecar's Claude provider, once Pydantic AI keeps the block.

**Claude runs on its native API, so that its features are kept; other models run through an OpenAI-compatible API.**
Anthropic's page on its
[OpenAI SDK compatibility](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk), read on
15 November 2026, says that prompt caching is not supported through that layer, that adaptive thinking
(`{"type": "adaptive"}`) returns a 400 and the response does not include the thinking, that `strict` on tools and
`reasoning_effort` are ignored, and that the layer is meant for testing and comparison rather than as a long-term or
production solution. The page does not mention compaction or server-side refusal fallbacks, so whether either works
through it could not be confirmed.

### Presets

The widget agent's settings offer presets, each of which fills the base URL; the person then picks the model and,
where needed, enters a key. The sidecar reaches every preset today, from its command line; the settings that offer
them are planned.

| Preset | Base URL | Key |
|---|---|---|
| Claude | The native Claude API, `https://api.anthropic.com`, fixed in code as today | A Claude API key |
| OpenAI | `https://api.openai.com/v1` | An API key |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | An API key |
| OpenRouter | `https://openrouter.ai/api/v1` | An API key |
| Ollama | `http://localhost:11434/v1` | None |
| LM Studio | `http://localhost:1234/v1` | None |
| Custom | Any base URL the person types | Optional |

Every base URL was read from the service's or the project's own documentation on 15 November 2026. The second group are
services and servers that the Custom preset reaches; none was run for this page.

| Service or server | Base URL | Preset | Source |
|---|---|---|---|
| OpenAI | `https://api.openai.com/v1` | OpenAI | [openai-python README](https://github.com/openai/openai-python/blob/main/README.md) |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | Gemini | [OpenAI compatibility](https://aistudio.google.com/docs/openai) of the Gemini API |
| OpenRouter | `https://openrouter.ai/api/v1` | OpenRouter | [Quickstart](https://openrouter.ai/docs/quickstart) |
| Ollama | `http://localhost:11434/v1` | Ollama | [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility) |
| LM Studio | `http://localhost:1234/v1` | LM Studio | [OpenAI compatibility endpoints](https://lmstudio.ai/docs/developer/openai-compat) |
| Mistral | `https://api.mistral.ai/v1` | Custom | [Migration guides](https://docs.mistral.ai/resources/migration-guides) |
| DeepSeek | `https://api.deepseek.com` | Custom | [API documentation](https://api-docs.deepseek.com/) |
| Groq | `https://api.groq.com/openai/v1` | Custom | [OpenAI compatibility](https://console.groq.com/docs/openai) |
| llama.cpp server | `http://127.0.0.1:9931/v1`, its default host and port | Custom | [Server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) |
| vLLM | `http://localhost:8000/v1` | Custom | [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/) |
| `mlx_lm.server` | `http://localhost:8080/v1` | Custom | [HTTP model server](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md) |

- The llama.cpp server's README gives 9931 as its default port; older guides that say 8080 are out of date.
- The `mlx_lm.server` page says that the server is not recommended for production because it implements only basic
  security checks, and it does not say plainly whether the server supports tool calling; the capability probe decides.
- The llama.cpp server documents tool calling with its `--jinja` flag, which is on by default, and parallel tool calls
  only for some models.

**Keys.** Each preset that needs a key has its own entry in the operating system's credential store: on macOS a
Keychain item of the service `caliper.agent-keys` with the preset's id as the account, as the Claude key is stored
today. A Custom entry's account is `custom-` followed by the first 12 hexadecimal digits of the SHA-256 of its base
URL. `KeyStore` already takes a provider id and knows only `anthropic`, so it gains the presets' ids. Each key reaches
the sidecar only in its environment at spawn, the variables that could send a request elsewhere or add a credential are
removed at spawn and again at start, and the scrubber covers every key, as for Claude today. Local servers need no key,
so none is stored or sent.

### Capabilities and degradations

When the person picks a model, the settings list the server's models where the server offers `/models`, and the
sidecar then probes the chosen model with two short test calls: one offers a single tool that the model must call with
fixed arguments, which tests tool calling, and one sends a small PNG of one solid colour that the model must name,
which tests vision. The result is cached in the app's settings for each base URL and model, with its date; the person
can override either answer, and can run the probe again. The context length comes from the server where it reports
one, else from the person's setting, else 8,192 tokens are assumed. The Claude preset needs no probe, since every model
it offers takes tools and images.

The probe is planned. What the sidecar does today is learn the same from the server: a request refused with a 400 that
names tools marks the model as without tool calling, and one that names images marks it as without vision, after which
the request is sent again with each image replaced by a line of text. The sidecar's options `--model-tools` and
`--model-vision` set either answer in advance, which is how the probe's result and the person's override will reach
it.

| Capability | When the model lacks it |
|---|---|
| Tool calling | The model cannot drive Caliper. The settings say so when it is chosen, and ⌘K offers only Ask in chat. |
| Vision | The bundle has no render, `panel.render` leaves the scope so the render check is skipped, and a PNG in a tool result becomes a note that the image was left out. |
| Context length | The bundle shrinks in the order given [above](#the-context-bundle). Below 8,192 tokens the settings warn that requests may not fit. |
| Strict tool schemas | A server may ignore `strict`. The loop checks every call against the command's full schema before it runs, so a malformed call gets an error result and the model can correct it. |
| Parallel tool calls | A server may return one call per reply. The loop runs the calls of a reply one at a time in any case. |
| Streamed tool arguments | Servers stream arguments differently. The provider assembles each call whole before the loop checks it. |
| Reasoning output | Shown as thoughts where the server returns it, whichever field carries it; otherwise nothing is shown. |
| Prompt caching | Only Claude caches the fixed prefix in its way. Usage and cost come from what the server returns; where it returns no cost, the card shows token counts only. |

## The provider conformance suite

The conformance suite runs the same widget tasks against each preset and against popular models, checks them against
pass criteria, and publishes the results, so that a person can see which models do the widget agent's work well.

- **Stubs by default.** Automated tests run the suite against scripted transports, one passing and one failing answer
  per task, to test the suite itself. No automated test calls a hosted API or needs a model server.
- **Real runs are manual or opt-in.** A person runs the suite on demand with their own key or their own server, since a
  run costs money or needs hardware. It never runs in automated tests against a hosted API.
- **The fixture.** A workspace with a Parquet table `run42_metrics` (columns `step`, `loss` and `accuracy`, 200 rows),
  a safetensors tensor `attn_weights` of shape [4, 8, 16, 16] (layers, heads, tokens, tokens), and the spec fixture
  `dashboard.spec.toml` open. Every task starts from this state.

| # | Target and request | Expected commands or result | Passes when |
|---|---|---|---|
| 1 | Panel `curves`: "Call this panel Training curves" | `panel.patch` of `title` | The preview differs from the open spec only in the title of `curves`, which is "Training curves" |
| 2 | Property `attention.options.colormap`: "Use the diverging colormap" | `panel.patch` setting `colormap` to `rdbu` | Only that option changed, and it is `rdbu` |
| 3 | Panel `curves`: "Use a linear y axis" | `panel.patch` that removes `y_log` or sets it to false | `curves` has no log y axis, and nothing else changed |
| 4 | Panel `curves`: "Show only steps 50 to 150" | `panel.patch` setting `x_range` to [50, 150] | `x_range` is [50, 150], and nothing else changed |
| 5 | A selection of 12 points on `curves`: "Make a table of these points" | `source.add` of a source restricted to the selection, `panel.create` of a `table` bound to it, `layout.place` | Resolving the new panel in the preview gives exactly the 12 selected rows, and no existing panel changed |
| 6 | The dropped Parquet file: "What's in this?" | No edit; a short answer | The preview is unchanged, and the reply names the three columns and the row count |
| 7 | The dropped Parquet file: "Plot accuracy against step" | `panel.create` of a `series`, with a `source.add` when no source binds the table | Resolving the new panel gives `step` as x and `accuracy` as y |
| 8 | The dropped safetensors file: "Show layer 2 averaged over heads as a heatmap" | `source.add` with a slice of axis 0 at index 2 and a mean over axis 0, then `panel.create` of a `heatmap` | The resolved data has shape [16, 16] and equals the suite's reference within 1e-6 |
| 9 | Panel `rows`: "Remove this panel" | `panel.remove` | `rows` is gone, the layout holds the other two panels, and nothing else changed |
| 10 | Panel `curves`: "Delete the loader behind this data" | No command, since no loader command is in the scope | The preview is unchanged, no call left the scope, and the reply points to the chat |

- **Runs.** Each task runs three times for each model and passes when at least two runs pass, each within the turn
  budget. The time to the answer is recorded but not graded, since it depends on the hardware.
- **A model's row** says "passes" with 9 or 10 tasks, "partial" with 6 to 8 and the failed tasks named, "fails" with
  fewer, and "cannot drive Caliper" when the probe found no tool calling. The row records the preset, the model id as
  the server names it, the server and its version, the quantization when the server reports it, the hardware class
  for a local server (the chip and the memory), the versions of Caliper and the sidecar, the date, the tasks passed and
  the median time.
- **Publishing.** The rows go on a reference page of the documentation, which comes with the suite. A row from a
  sidecar more than two versions old is marked as stale.

## GPU contention

On a Mac, a local model server shares the GPU and the unified memory with everything else, including a training run
that streams into Caliper. Both slow down, and a model that does not fit beside the run can push the machine into
swapping. On a machine with a discrete GPU, the model's weights take video memory that the run may need.

- **The app shows it.** While a live session streams into the app, the inline prompt shows a short note when the
  widget agent's base URL is a loopback address, and the side chat's header shows the same note when the person has
  marked the side chat's agent as running on a local model. The app cannot tell where an outside agent's model runs,
  so the side chat relies on that setting.
- **The advice**, in Set up the widget agent (`docs/guides/set-up-the-widget-agent.md`):
  during a run, use a smaller or more quantized model; keep its context length
  modest, since the context takes memory too; unload the model when done; use a hosted preset while a long run trains;
  and watch the system's memory pressure. With Ollama, `ollama ps` shows whether a model sits on the GPU or the CPU and
  with which context, `ollama stop` unloads it, and a model stays loaded for five minutes after its last request
  unless `keep_alive` says otherwise ([Ollama FAQ](https://docs.ollama.com/faq), read on 15 November 2026).

## Security

The rules of the sidecar hold, as [Widget agent sidecar](byok-agent.md#security-notes-for-the-later-security-pass)
describes them:

- **Commands only.** The widget agent has no shell, no file tools and no web access. It acts only through the commands
  of its scope, and every edit is a dry run until the person accepts it.
- **Workspace data is data.** The bundle and every tool result are workspace data, and the system prompt says that
  only the person's messages give instructions.
- **Keys.** A key lives only in the operating system's credential store, one entry per provider. It reaches the sidecar
  in its environment at spawn, never in argv, is removed from the environment at start, and is scrubbed from every
  output.
- **No self-approval.** No command and no endpoint commits a preview or approves a command, so an agent cannot accept
  its own proposal.
- **One host.** The provider sends requests only to its configured base URL's host, and a test checks that the
  transport saw no other host. Caliper API stays on loopback.

For a local server:

- **Loopback by default.** The Ollama and LM Studio presets use loopback addresses.
- **A warning for a remote address.** A Custom base URL on another host gets a warning that prompts and workspace data
  (samples, spec text, renders) leave the machine, and a plain `http` URL on another host gets a second warning that
  they travel unencrypted.
- **No key.** A local server takes no key, so none is stored or sent.

## Limits

- One target for each prompt, 3 model requests and 8 tool calls for each prompt, and 8,000 output tokens per request.
- The bundle's bounds above, 64 recorded invocations per preview, 16 sessions per sidecar, and 30 minutes before an
  unused preview ends.
- The widget agent cannot create or run loaders, save applets, undo, or replace the spec or its layout; those belong to
  the side chat and to the app.
- How well a small local model does this work varies widely; the conformance suite says which models pass.

## Decisions

- **Two roles, not two interchangeable backends.** The side chat is for conversations and bigger jobs, the widget
  agent for one target in seconds, and "Ask in chat" joins them.
- **The app builds the context bundle.** The app has the spec, the catalog, the render and the selection at hand, and
  giving them to the model up front saves the read calls that would cost a round each.
- **The tool scope is derived from the command table**, with reads and previewable edits only, so it follows the table
  as commands are added and can never hold a destructive command.
- **Stacked previews in the session**, so that edits that build on each other can be previewed together, and Reject
  leaves no trace because no edit of a preview reaches the journal or the history.
- **Accept is `preview.commit`, which only the `ui` and `builder` origins may send**, so an agent cannot accept its own
  proposal, and the commit goes through the queue, the journal and the history like every change; the committed edits
  carry the origin `agent` and one group, so they are journaled as the agent's work and undo as one step.
- **A preview follows the open spec.** Its edits are applied again when the open spec changes, so a commit applies
  what the card shows or refuses with `history.conflict`, in place of the `preview.stale` planned earlier, which
  refused any change of the open spec.
- **A `_meta` field of `session/new`** for the target, the scope and the preview, for the reasons given above.
- **The scope rule as data beside the command table**, read by the sidecar and embedded in the app, so both derive the
  scope from one definition; an app's scope wider than the rule refuses the session.
- **Edits with the preview header and without `dry_run`**, since a dry run that names a preview only checks the edit.
- **A `handoff` tool for "Ask in chat"**, so that the model says in a structured way that a request belongs in the side
  chat, and the turn ends there.
- **`session/close` and `session/cancel` leave the preview to the app**, which began it.
- **Claude Haiku 5.5 at effort `low`** as the Claude default, for speed on small bounded edits.
- **The preview card replaces per-edit review** in a widget session.
- **A dropped file with no open spec** previews on an empty spec, and Accept creates the spec from the `ui` origin
  first.
- **Ctrl+K on Windows** in place of ⌘K.
- **Claude on its native API** and other models through an OpenAI-compatible API, so that Claude's features are kept.
- **Pydantic AI as the provider layer**, after a spike that checked cancellation, retries, Claude's features, stop
  reasons, other models and the footprint.
- **Server-side refusal fallbacks off**, since Pydantic AI drops the reply's `fallback` block, with a neutral refusal
  text that offers three ways forward.
- **One credential entry per provider**, and none for a local server.
- **Real conformance runs only on demand**, by a person with their own key or server.

## Work items

The items come after sprint 2 merges. The numbers are their places in the integration order of the
[builder overview](builder-overview.md#the-integration-order-after-sprint-2), which also lists the items that come
before them, such as the session in the app and the command-layer enforcement. S is one to two days, M three to five
and L six to ten; each figure is rough.

| Order | Item | Size |
|---|---|---|
| 9 | Done: stacked previews in the session, with `preview.begin`, `preview.diff`, `preview.commit` and `preview.discard`, edits and reads that name a preview, and `Session::resolve()` of a preview's panel | None left |
| 10 | Done: an optional `_meta` argument on `AcpClient::new_session()` | None left |
| 11 | Done: the evaluation of the ACP Python SDK for the sidecar's agent side. The hand-written agent side stays, and the SDK validates every message in the tests ([Widget agent sidecar](byok-agent.md#the-acp-python-sdk)) | None left |
| 12 | Done: widget sessions in the sidecar: `_meta.caliper.widget`, the scoped tools and their check, the widget system prompt, images and embedded context in prompts, `X-Caliper-Preview` on every call, the turn budget, the handoff and `session/close` | None left |
| 13 | The context bundle, built in the app from the session, with its bounds and cuts | M, 3 days |
| 14 | The inline surface: the four entry points, the inline prompt, the preview drawn in place, Accept, Reject and Refine, and Ask in chat; with the packaging of `caliper-agent` and a Python runtime | L, 8 days |
| 15 | Adopted: Pydantic AI as the provider layer, with the Claude and OpenAI-compatible providers built in the sidecar. What remains is passing the chosen provider, base URL and key from the app when it starts the sidecar; widget sessions (12) and the probe (16) build on it | S, 1 day |
| 16 | The capability probe and its cache, with the override | S, 2 days |
| 17 | The presets and one key per provider in the settings, with `KeyStore` knowing the presets' ids | M, 3 days |
| 18 | The provider conformance suite, its stubs and its reference page | M, 4 days |
| 19 | The GPU contention note | S, 1 day |

The Windows Credential Manager store, with one entry per provider, is on the [Windows parity](windows-parity.md) page.
