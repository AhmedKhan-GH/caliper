# Widget agent

The widget agent is `caliper-agent`, Caliper's own ACP agent, in the role in which it works on one target: a panel, a
selection, a dropped file or a property. This page gives the interface between Caliper and that role: how a widget
session is opened, which commands it may use, what a prompt may hold, and what each answer carries.
[Agent roles](../explanations/agent-roles.md) explains the two roles,
[Set up the widget agent](../guides/set-up-the-widget-agent.md) shows the person's side, and
[Widget agent conformance](widget-agent-conformance.md) tests a provider and a model against the widget tasks.

!!! note "Status"
    The agent and its widget sessions are built and tested in `src/builder/agent/`. The inline surface that opens them
    on a panel, a selection, a dropped file and a property is built in `src/host/builder/` and works on the app's spec
    page ([The inline surface](#the-inline-surface)). `caliper-agent` is not yet packaged with a Python runtime, so
    the app finds it through `CALIPER_AGENT_PYTHON` ([Environment variables](environment.md)).

## Starting the agent

Caliper starts the agent as a child process and speaks ACP version 1 on its standard input and output:

```sh
<python> -I -m caliper_agent [--api-file <path>] [--command-table <path>] [--widget-scopes <path>] [options]
```

| Option | Meaning |
|---|---|
| `--api-file <path>` | Caliper API's file, which gives the loopback address and a token; the app gives its sidecar a file of its own, whose token has the origin `agent`. A session that `session/new` hands Caliper's MCP server uses that server's file instead ([In the side chat](#in-the-side-chat)) |
| `--command-table <path>` | The command table as JSON |
| `--widget-scopes <path>` | The scope rule below; by default `widget_scopes.json` beside the command table. Without it, widget sessions are refused |
| `--preset <id>` | A provider preset, which fills the provider and the base URL (below) |
| `--provider claude\|openai-compatible`, `--base-url <url>`, `--model <id>`, `--effort <level>` | Where model requests go, and the defaults of a session in the general role |
| `--cache-dir <path>` | The folder of the capability cache that `caliper-agent probe` writes; the agent reads its answer for the model at start |
| `--model-tools yes\|no`, `--model-vision yes\|no`, `--context-length <tokens>` | What the model can do, when Caliper knows it; each wins over the cache |
| `--widget-requests <n>` | The model requests a widget prompt may make, from 1 to 20; 6 by default (see Limits below) |

The Claude API key reaches the agent in `ANTHROPIC_API_KEY`, and a key for an OpenAI-compatible server in
`CALIPER_MODEL_API_KEY`, never in its arguments. `-I` keeps the workspace folder off Python's import path.

### Presets

The presets are one table, `presets.json` in the agent's package, which Caliper's key store reads too:

| Id | Preset | Base URL | Key |
|---|---|---|---|
| `anthropic` | Claude | `https://api.anthropic.com`, fixed | Required, in `ANTHROPIC_API_KEY` |
| `openai` | OpenAI | `https://api.openai.com/v1` | Required, in `CALIPER_MODEL_API_KEY` |
| `gemini` | Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | Required, in `CALIPER_MODEL_API_KEY` |
| `openrouter` | OpenRouter | `https://openrouter.ai/api/v1` | Required, in `CALIPER_MODEL_API_KEY` |
| `ollama` | Ollama | `http://localhost:11434/v1`, or another host with `--base-url` | None |
| `lmstudio` | LM Studio | `http://localhost:1234/v1`, or another host with `--base-url` | None |
| `custom` | Custom | `--base-url`, required | Optional, in `CALIPER_MODEL_API_KEY` |

The model is always the person's choice. The Claude, OpenAI, Gemini and OpenRouter presets take no other base URL, so
a preset's key only goes to that preset's host. A base URL must be an http or https address without a user name,
password, query or fragment; anything else makes the agent exit with 2. The answer to `initialize` names the choice,
with what the pane should warn about:

```json
{"_meta": {"caliper": {"provider": {"preset": "lmstudio", "api": "openai-compatible",
  "base_url": "http://192.0.2.10:1234/v1", "host": "192.0.2.10", "loopback": false,
  "warnings": [{"code": "remote_host", "message": "..."}, {"code": "unencrypted", "message": "..."}]}}}}
```

`remote_host` is given for a local preset or a Custom entry on a host that is not loopback, since prompts and
workspace data then leave the machine, and `unencrypted` for plain http to such a host.

Besides the key itself, the agent scrubs from everything it writes any text shaped like a Claude API key (`sk-ant-`),
an OpenRouter key (`sk-or-v1-` and 64 hexadecimal digits) or a Google API key (`AIza` and 35 characters).

With `CALIPER_AGENT_OFFLINE=1`, which every test of the repository sets, the agent reaches no model server off
loopback: it answers `session/new` and `session/prompt` with an error whose `data.code` is `agent.offline` unless it runs
a scripted model, a stubbed transport or a server on a loopback address.

## Checking a model

`caliper-agent probe` checks what a model can do and prints one JSON object on standard output. Caliper's settings run
it when the person picks a model:

```sh
<python> -I -m caliper_agent probe --preset ollama --model <id> --cache-dir <settings folder>
```

For an OpenAI-compatible server it lists the server's models, makes one streamed request that asks the model to call a
tool with fixed arguments and one that asks it to name the colour of a small PNG, each with at most 1,024 output
tokens and 60 seconds, and reads the context length where the server reports it. For a server on this machine it also
asks Ollama's `POST /api/show` for the model's metadata, which loads nothing, and keeps the bytes of context cache one
token takes as the quirk `kv_bytes_per_token`; a server that is not Ollama answers 404 and nothing is kept. The
metadata's context length is the most the model takes, not what the server runs it with, so it is not used: after the
test requests have loaded the model, the probe reads Ollama's `GET /api/ps`, which states the context in force, and
otherwise leaves the context length unknown. For the
Ollama preset the two test requests go to Ollama's own chat API with the preset's context length, as the agent's do
(see [Ollama's own chat API](#ollamas-own-chat-api)), and that context length is the answer, with the source
`preset`. For Claude it reads the Models API and makes no request that costs tokens. The key comes from the
environment, as for the agent.

| Option | Meaning |
|---|---|
| `--preset <id>`, `--provider`, `--base-url` | The provider, as for the agent |
| `--cache-dir <path>` | Keep the answer in `model-capabilities.json` in this folder, and answer from it while it is fresh (30 days) |
| `--refresh` | Probe again even when the cache has a fresh answer |
| `--list-models` | Only list the server's models, with no test request |
| `--no-vision-probe` | Leave the image request out |
| `--set tools=yes\|no\|auto`, `--set vision=yes\|no\|auto`, `--set context_length=<tokens>\|auto` | The person's override, kept in the cache until set to `auto`; no request is made |

| Field | Meaning |
|---|---|
| `models` | The server's models, or `null` when it lists none or the answer came from the cache |
| `capabilities` | `tools`, `vision` and `context_length`, each `{"value", "source", "detail"}`: what the agent will assume, with `source` one of `probe`, `server`, `known`, `preset`, `override`, `default` and `unknown` |
| `can_drive_caliper` | `false` when the model cannot call tools |
| `probe`, `overrides`, `quirks` | The cached probe answers, the person's overrides, and what the server needed (`strict`, `max_tokens_field`) or showed (`reasoning_field`, `streamed_arguments`, and `kv_bytes_per_token` from Ollama's metadata) |
| `probed_at`, `from_cache`, `stale` | When the probe ran, and whether this answer came from the cache |
| `errors`, `warnings` | Why an answer is unknown (such as `probe.unreachable`, `probe.key_rejected`, `probe.timeout`, or `probe.cut_off` for a reply that reached the output limit first), and notes such as a context length below 8,192 tokens |

The command exits with 0 when it printed its answer and with 2 for bad usage. A base URL must be an http or https
address without a user name, password, query or fragment.

## What the agent assumes of a model

At start the agent takes, for each capability, an option on its command line, else the person's override in the cache,
else a fresh probe answer, else its default: tool calling and vision until the server refuses them, and a context
length of 8,192 tokens for an OpenAI-compatible model (unknown for Claude). A model without tool calling cannot drive
Caliper, so `session/new` is refused with error -32602, `data.code` `agent.no_tools`, the model and where the answer
came from, and a message the pane can show. Every other `session/new` answer carries the model's facts, which Caliper
gives to `build_context_bundle()` as the context length and whether to send a render:

```json
{"_meta": {"caliper": {"model": {"id": "qwen3.8:27b", "tools": true, "vision": false, "context_length": 32768,
                                 "sources": {"tools": "probe", "vision": "probe", "context_length": "override"}}}}}
```

A model without vision gets no image and no `panel.render`. When the context length is known, a request asks for at
most half of it as output.

## Ollama's own chat API

Ollama's OpenAI-compatible endpoint takes no context length: it ignores `options.num_ctx`, so Ollama loads the model at
the context length it chooses itself, which for a model of 262,144 tokens of context can take more than 10 GB of
memory for the context cache alone. The Ollama preset therefore sends its chat requests to Ollama's own `/api/chat`,
each with `options.num_ctx` and a `keep_alive` of 60 seconds, so the model is loaded with that context length and
unloaded a minute after its last request. The list of models still comes from `/v1/models`. Every other preset, LM
Studio and Custom included, stays on the OpenAI-compatible endpoint.

| Role | Context length (`num_ctx`) | The person's setting |
|---|---|---|
| The widget agent (⌘K) | 16,384 tokens | Context length in the widget agent's settings (an override in the cache), or `--context-length` |
| The general role (the side chat's `caliper-agent`) | 32,768 tokens | `--context-length` on its command line |

The defaults are what one request of each role costs: the widget agent's first request, with its ten tools, the
context bundle and the render, takes about 7,000 tokens before the model's reply, and the general role's, with every
command of the table as a tool, about 11,000, growing with each tool result. Where the agent chose the context length,
it never lets the server cut a request: before each model request it estimates the request from the prompt tokens the
server counted for the last one, and when the request would leave less than an eighth of the context (at most 4,096
tokens) for the reply, it sends nothing and the turn ends with the stop reason `max_tokens`, which Caliper shows as
LIMIT, and a message, with the status `context_full`, that the conversation no longer fits the model's context and that
the person can start again or raise the context length.
For a server whose context length the agent did not choose, nothing is checked.

## Opening a widget session

Before it opens the session, Caliper begins a stacked preview of the open spec with `preview.begin`, from the `ui`
origin and scoped to the target, as the [Caliper API](api.md) page describes for `X-Caliper-Preview`. It then sends
`session/new` with the preview's id in `_meta.caliper.widget`:

```json
{"cwd": "/path/to/workspace", "mcpServers": [],
 "_meta": {"caliper": {"widget": {
   "target": {"kind": "panel", "panel": "curves"},
   "preview": "widget-curves",
   "scope": ["spec.get", "panel.patch"]}}}}
```

| Target | Fields |
|---|---|
| A panel | `{"kind": "panel", "panel": <panel id>}` |
| A selection | `{"kind": "selection", "panel": <panel id>}` |
| A dropped file | `{"kind": "dropped_file", "loader": <loader id>, "catalog": [<catalog ids>]}` |
| A property | `{"kind": "property", "panel": <panel id>, "path": "options.colormap"}`, or `"source"` in place of `"panel"` |

- `preview` is required, and must be the id of a preview Caliper began.
- `scope` is optional; without it, the agent uses the whole scope of the target's kind.
- The agent checks the field: at most 64 KiB and 16 levels, panel and source ids of the spec's form (a lowercase letter,
  then at most 63 lowercase letters, digits, `_` and `-`), a property path of at most 8 dotted names, at most 16
  catalog ids and 64 scope names. A field it cannot use, or a scope wider than the rule allows, refuses the session
  with error -32602 and `data.code` `agent.bad_widget`. A `session/new` without the field opens a session in the
  general role.

The answer's session id starts with `widget-`, its `_meta.caliper.widget` repeats the target, the scope and the preview,
and its config options are `model` and `effort` (on the Claude API), with `claude-haiku-5-5` and `low` as their values.
There is no review option, because every change is a preview that the person accepts or rejects.

## The tool scope

The commands a widget session may use come from one rule, kept as data in `src/builder/commands/widget_scopes.json`
beside the command table, which both the agent and Caliper read:

| Target | Reads | Edits |
|---|---|---|
| A panel | `spec.get`, `spec.validate`, `catalog.describe`, `catalog.sample`, `panel.render` | `panel.patch`, `panel.remove`, `layout.place`, `source.add`, `source.patch` |
| A selection | the panel's reads and `data.query` | `source.add`, `panel.create`, `layout.place` |
| A dropped file | `spec.get`, `catalog.list`, `catalog.describe`, `catalog.sample`, `loader.describe`, `data.query` | `source.add`, `panel.create`, `layout.place` |
| A property of a panel | `spec.get`, `catalog.describe` | `panel.patch` |
| A property of a source | `spec.get`, `catalog.describe` | `source.patch` |

The rule is applied to the command table, not to a fixed list: a read must not change anything, and an edit must take
a boolean `dry_run` and must not delete or overwrite saved data. `spec.set_text`, `spec.new`, `spec.open`,
`applet.open`, `layout.set`, `data.import` and the `history`, `applet` and `preview` commands are never in a scope.
`panel.render` leaves the scope when the model cannot read images. A call to any other command is answered as a call to
an unknown tool and never reaches Caliper.

A command whose descriptor in the command table says `"agents": false` is the person's own: today `history.undo` and
`history.redo`. The agent offers such a command as a tool in no session, general or widget. A call the model makes
anyway is answered as a call to an unknown tool, and the agent's client of Caliper API refuses it with
`command.not_allowed`, the command layer's code for the same refusal, before anything is sent. A table without the
`agents` field opens every command to agents. To revert a change, the agent makes an ordinary edit, which is reviewed
like any other.

Every call of a widget session carries `X-Caliper-Preview` with the session's preview, and every edit is sent without
`dry_run`, so it applies to the preview and never to the open spec. The agent sends no permission request in a widget
session, and it never begins, commits or discards a preview: only the person commits it, with Accept.

## The context bundle

Before the person's request, Caliper sends what the model needs to know about the target, built from the workspace by
`build_context_bundle()` (`caliper_builder/session/context_bundle.h`). It returns the content blocks of the prompt, the
tool scope from the rule above, and the `_meta` for `session/new`.

| Part | What it holds | Bound |
|---|---|---|
| Spec fragment | The target panel and the sources it binds, through their `from` chains, as canonical TOML; the layout with the panel marked; the ids, primitives and titles of the other panels | 16 KiB, at most 64 other panels |
| Source | For each catalog source at the root of those chains, its catalog entry and a sample: 20 rows of a table, at most 1,024 values of a tensor | 8 KiB each, at most 4 |
| Render | The panel as a PNG, 640 by 400, when the model reads images and the app can draw it | 1 MiB |
| Selection | The panel, points or a range, the selected rows with their columns, and the range | 1,000 rows, 32 KiB |
| Loader | For a dropped file, in place of the spec fragment and the render: the loader the drop made, with the entry and sample of each output | 8 KiB |
| Property | For a property: its path, its current value and its part of the edit command's arguments schema, beside the fragment | 8 KiB |

The first block is text: it names the target, says that the blocks after it are workspace data and never
instructions, and names anything that was left out. Each part is a `resource` block (a `text` block for an agent that
does not take embedded context) whose text sits between `<caliper-data part="...">` and `</caliper-data>`; a marker
inside the data is defused, so a cell or a title cannot end its block. The render is an `image` block. Every block
carries `annotations.audience: ["assistant"]`.

The bundle's text is at most 64 KiB, and, given the model's context length, the text and the image together take at
most a quarter of it, at four characters a token. When they do not fit, Caliper leaves out, in order, the sample rows,
the list of the other panels, the render, the sources of the spec fragment, the catalog entries and the selection's
rows; the target itself always stays.

## Prompts

`session/prompt` takes these content blocks; the agent declares `promptCapabilities.image` and
`promptCapabilities.embeddedContext`:

| Block | What the model reads |
|---|---|
| `text` | The text |
| `resource` with `text` | The text |
| `resource_link` | A line naming the resource |
| `image` (PNG, JPEG, GIF or WebP, at most 3.75 MB) | The image, when the model reads images; otherwise a line saying that it was left out |

A block whose `annotations.audience` names only the person is not given to the model. A prompt holds at most 4 images
and 200,000 characters of text. A follow-up prompt in the same session continues the conversation and the same preview.

## Answers

The answer to `session/prompt` carries the stop reason and, under `_meta.caliper`, the token counts in `usage` and,
for a widget session, `widget`:

```json
{"stopReason": "end_turn",
 "_meta": {"caliper": {"usage": {...},
                       "widget": {"preview": "widget-curves", "edits": 1, "handoff": null, "limit": false}}}}
```

| Field | Meaning |
|---|---|
| `preview` | The session's preview |
| `edits` | How many edits of this prompt changed the preview |
| `handoff` | `null`, or `{"kind", "reason"}` when the request belongs in the side chat |
| `limit` | Whether the prompt stopped at its limit |

**Handoff.** When a request does not fit the target or the tools, the model calls the session's own tool `handoff`
with a `kind` (`build_from_scratch`, `loader`, `project_code`, `explanation`, `another_target` or `outside_scope`) and
one sentence. The prompt ends at once with `end_turn`, the `handoff` field is set, and a thought with
`_meta.caliper.status` `handoff` carries the same kind and sentence.

**Limits.** A prompt makes at most 6 model requests and 8 tool calls, and each request asks for at most 8,000 output
tokens; `--widget-requests <n>` sets another number of requests, from 1 to 20. A prompt that reaches a limit ends with
`max_turn_requests`, a thought whose `_meta.caliper.status` is `limit` (with `edits`, the commands of the edits made,
and `next`, the command of the first call that did not run, or `null`), and a reply that says the change is not
finished, which edits the preview holds and that Accept applies them or Refine lets the agent finish, such as "The
change is not finished: the agent stopped at its limit of 6 model requests after 2 edits (source.add, panel.create).
Accept applies the edits shown, or Refine lets it finish." Caliper shows the reply in full on the card, under LIMIT and
the count of edits.

## In the side chat

`caliper-agent` can be the [side chat](side-chat.md)'s custom agent, in the general role, for example on a local model
through the Ollama preset, with no account or key. The side chat sends `session/new` with one MCP server, `caliper`:
`caliper-mcp` with `--api-file <the session's own API file>`, whose token has the origin `agent`. The agent starts no
MCP server; it calls Caliper API itself with that file, so its commands run as the origin `agent` and its edits of the
open spec wait on the approval cards, which the chat mirrors as rows. Caliper's server is any server named `caliper`,
in any form, or whose program is `caliper-mcp`. A session whose Caliper server names no usable `--api-file` (an
absolute path), such as one in the HTTP form, which has no arguments, is refused with -32602 and `data.code`
`agent.no_session_api_file`, rather than run with `api.json`, whose token is the origin `mcp`. Without a Caliper
server, a session uses `--api-file`.

Each call of a Caliper command carries the command in `_meta.caliper.command` of its `tool_call` update, such as
`{"caliper": {"command": "panel.create"}}`, so that the side chat shows it as that command and folds its approval into
its row.

## Closing

`session/cancel` stops a running prompt, which answers `cancelled`; the preview stays as it is. `session/close`, which
the agent declares as `sessionCapabilities.close`, cancels a running prompt, waits up to 10 seconds for it to end,
forgets the session and answers `{}`. A sidecar keeps at most 16 sessions.

## The inline surface

The inline surface is the app's side of a widget session on a panel (`src/host/builder/widget_surface.h`). On the
spec page, a click on a panel focuses it, and its frame turns bronze. ⌘K (Ctrl+K on Windows), or the page bar's ⌘K
Command, opens a one-line prompt at the top of the focused panel, under its title strip; the panel's content is drawn
below the prompt, so the prompt covers nothing. With no panel focused, ⌘K opens the page's command input in the page
bar instead: typing a panel's title focuses it, and Enter opens its prompt. Escape leaves the prompt's innermost state
first, then the command input, then the panel's focus, and last goes Home.

**The other targets.**

- **A selection.** A shift-drag across a series panel's plot selects a range of x, such as steps, and across a 2-D
  points panel a rectangle of points; the selection is drawn in bronze, a shift-click clears it, and so does Escape.
  ⌘K on that panel opens the prompt for the selection: its context bundle carries the selected rows (the drawn steps
  or points, at most 1,000) and the range. A change that adds a panel, such as a table of the selection, is drawn as
  the preview of the whole spec: the new panel stands where the layout puts it, in a dashed bronze frame with PREVIEW
  in its strip.
- **A dropped file.** After files are dropped on the spec page, each source the import made shows ⌘K at the end of its
  row in the builder's Sources palette (and ⌘K with that row selected and no panel focused does the same). The prompt
  opens under the row, for the file's loader and its catalog ids. An answer without a change is shown in the card,
  over up to four lines, with Text for the whole of a longer one; a proposed panel is previewed as above. With no spec
  open, the preview first runs `spec.new` from the `ui` origin. A discard, or the surface going, undoes that step again
  while it is still the top of the undo stack, known by the step's id from `history.list`; when another step was
  committed after it, the new spec stays. Accept leaves two undo steps: the new spec, then the change.
- **A property.** In build mode, ⌘K shows beside the focused panel's title and each of its options in Properties while
  the row is under the pointer. It opens the prompt under that row for that one property, such as
  `options.colormap`, and the edit is previewed in the panel.
- **A property of a source.** With a source of the open spec selected in the builder's Sources palette, ⌘K shows in
  Properties beside what it reads (its catalog id, the source it derives `from`, or its SQL) and beside its
  transforms, which count as one property, `transform`. The prompt opens under that row; the context bundle carries
  the source with its chain and catalog entry, the property's value and its part of `source.patch`'s schema, and the
  session may read with `spec.get` and `catalog.describe` and edit with `source.patch` only. The change shows as its
  compact diff under the row (such as the source's changed transform); the panels that read the source show it once
  you accept.

The prompt row shows the model and effort the session reports; a click on them lists the models the session offers,
and the choice goes to the session with `session/set_config_option` and to every later session. While the agent
works, the row shows its latest step: the command it calls, or the last line of its summarized thinking.

**The session.** The surface opens one widget session for each prompt that it opens:

1. When the first request is submitted, it sends `preview.begin` from the `ui` origin, scoped to the panel, with an id
   that starts with `widget-`.
2. It builds the [context bundle](#the-context-bundle) on that preview and opens the session with `session/new`,
   whose `_meta.caliper.widget` holds the target, the preview and the bundle's scope.
3. It sends the request as `session/prompt`, with the bundle's blocks before the request on the session's first
   prompt only.
4. When the turn ends, it reads `preview.diff` and draws the panel from the preview.
5. Accept sends `preview.commit` from the `ui` origin, which files one undo step. Reject, Escape, a cancel, a handoff
   or a failure sends `preview.discard`. A follow-up (Refine) is another `session/prompt` on the same preview.
6. Closing the prompt ends the session with `session/close`.

One `caliper-agent` process serves every ⌘K. It is started at the first ⌘K with the provider's key placed only in its
environment, and it is kept running until the settings change: a change ends it at once, with its API file and token,
and the next ⌘K starts it again from the new settings.

**States and actions.** Each state shows a tag and its actions:

| State | What the panel shows | Actions |
|---|---|---|
| Prompt | The input, the model and effort the session reports | Enter, Ask in chat, Escape |
| Running | The request, the agent's latest step | esc Cancel (discards the preview, keeps the request) |
| Preview | The panel drawn from the preview, PREVIEW in its title strip, the agent's reply on one line, and the changed keys with old and new values | ⏎ Accept, esc Reject, Refine, Diff (the full diff and the whole reply), Ask in chat |
| DECLINED | The model that declined | Rephrase, Retry, Model, Ask in chat |
| A handoff | The handoff's kind as a tag (LOADER, PROJECT CODE, EXPLANATION, NEW DASHBOARD, ANOTHER TARGET, OUTSIDE SCOPE) and the agent's reason | Ask in chat, Rephrase, esc Close |
| CONFLICT | The open spec changed so that an edit no longer applies | Refine (a new preview from the current spec), esc Reject, Diff |
| STOPPED, FAILED | The error code | Retry, esc Close |
| NO PROVIDER | The input; no agent starts | Setup, Ask in chat (Enter) |

Ask in chat hands the request, the target, the conversation so far and the preview's diff, marked as not applied, to
the side chat, and closes the prompt.

**Layout.** The prompt row, the diff and the action row are rectangles inside the panel. As a panel shrinks, the reply
line goes first, then the diff lines past the first (folded into "+N"), then the model label, then the buttons' keys.
Below the minimum height, which is the rows plus 48 pixels of content at text scale 1, the surface takes the whole
panel.

**caliper-widget-shot** is a development tool, not installed, that drives the surface on one panel of a saved spec
applet through a scripted scenario, with a scripted agent and no model, and writes a PNG of the offscreen canvas at
each state:

```
caliper-widget-shot --workspace <dir> --applet <id> --panel <id> --prompt <text> --out-dir <dir>
                    [--size WxH] [--prefix NAME] [--frames N] [--edit JSON] [--reply TEXT]
                    [--scenario accept|declined|handoff|unavailable|refine]
```

It opens the applet as the workspace's open spec, so give it a copy of a workspace. It prints a JSON summary: the
files and their states, the panel's rectangle, the undo steps before and after Accept, and whether the open spec
changed before Accept.

## The settings in the app

The widget agent's settings are a popup over any page (`src/host/builder/agent_settings.h`). It opens from the
landing screen's **⌘, Agent**, from the key ⌘, (Ctrl+, on Windows) on any page, and from the inline surface's
**Setup** (no provider) and **Model** (after a refusal). Its width is 62 percent of the work area, its top is 12 percent
of the work area's height below the work area's top, and its height is its content's; Escape or esc Close closes it.

| Row | What it holds |
|---|---|
| Provider | The presets of `presets.json`, one chosen |
| Base URL | Fixed text for Claude, OpenAI, Gemini and OpenRouter; a field for Ollama, LM Studio and Custom, with the tags Remote address, Unencrypted and Invalid address |
| Model | A field, and as choices the Claude preset's four models or the server's list after **List** (`caliper-agent probe --list-models`) |
| Effort | Claude only: low, medium, high, xhigh and max, which each of its four models takes |
| Key | Set or Not set; a password field with Save, or Replace and Remove; no row for Ollama and LM Studio, nor for Custom before its address is valid |
| Capabilities | Tool calling, vision and the context length with where each came from (probe, server, known, preset, override, assumed), Yes, No and Auto for the person's override (`caliper-agent probe --set`), **Probe** (`--refresh`), and the tag Context cache about N GB when a server on this machine would hold a context cache of 4 GB or more |
| Note | Shared GPU, while a live run is attached and the base URL is a loopback address |

The settings never show a key or read one back for themselves. They call `KeyStore::set()`, `has()` and `remove()`,
and wipe the field's text when it is saved or cancelled. A hidden app's key store is an in-memory one, so it never
touches the Keychain. The one place a key leaves the store is the probe's start: **Probe**, **List** and the
overrides run `caliper-agent probe` as a child process, and `place_key_in_environment()` copies the chosen provider's
key into that child's environment, as it does for the sidecar, in `ANTHROPIC_API_KEY` or `CALIPER_MODEL_API_KEY`. It
also removes from the child's environment every other key variable and every variable that could redirect it, such as
`ANTHROPIC_BASE_URL` and `OPENAI_BASE_URL`, so no key or endpoint of the app's own environment reaches the provider,
and passes `CALIPER_AGENT_OFFLINE` through when the app has it. When the store cannot place the key, or the provider
requires one and there is no store, the probe does not run, and its tag says why: No key, Malformed key or Unreadable
key. A provider that needs no key is probed with no key variable at all. When the app quits or the settings go while a
probe runs, its child is ended at once (SIGKILL to its process group) and reaped on the probe's own thread, so the
frame never waits for it; an answer that comes after the model or the address changed is dropped.

**`widget-agent.json`**, in the app-data folder beside `side-chat.json`, holds the choice and never a key:

```json
{"version": 1, "preset": "ollama",
 "providers": {"ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen3.8:27b"},
               "anthropic": {"model": "claude-haiku-5-5", "effort": "low"}}}
```

A file that does not parse counts as no choice, and an unknown preset is dropped. The probe's answers and the
person's overrides are kept apart, in `model-capabilities.json` in the same folder, which the agent reads with
`--cache-dir`.

**The agent's start.** The inline surface's agent is made from the settings at the first ⌘K after they change. A
change (a preset, an address, a model, an effort, a key saved or removed, or a probe's answer for the current model)
applies at once, on any page: an open prompt closes as Close closes it, so a running turn is cancelled and its preview
discarded, and the agent goes with its sidecar, whose API file and token are revoked; a removed key so leaves the
sidecar's environment at once. It starts `caliper-agent` with `--preset`, `--base-url` (Ollama, LM Studio and Custom), `--model`, `--effort`
(Claude) and `--cache-dir`, and `place_key_in_environment()` puts the provider's key in its environment: in
`ANTHROPIC_API_KEY` for Claude, in `CALIPER_MODEL_API_KEY` for the others, and none for a local server. ⌘K has no
provider, and offers Setup and Ask in chat, when no preset is chosen, the address is not valid, an OpenAI-compatible
preset has no model, a required key is not set, or the model is known to lack tool calling (the person's override, or
a probe answer younger than 30 days). The settings name that state in a tag: No provider, Invalid address, No model,
No key or No tool calling. While the GPU note applies, the prompt row shows it after the model the session reports.

**caliper-agent-settings-shot** is a development tool, not installed, that draws the settings offscreen in a named
state, with an in-memory key store and, for the local scenarios, a fake model server on loopback that it starts with
the given interpreter:

```
caliper-agent-settings-shot --scenario claude|key-set|local|no-tools|gpu-note --out <file.png> --data-dir <dir>
                            [--size WxH] [--python <interpreter>] [--agent-dir <src/builder/agent>] [--frames N]
```

It prints a JSON summary: the state, whether a key is set, the GPU note, the model choices and every text shown.
