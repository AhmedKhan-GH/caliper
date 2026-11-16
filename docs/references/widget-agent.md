# Widget agent

The widget agent is `caliper-agent`, Caliper's own ACP agent, in the role in which it works on one target: a panel, a
selection, a dropped file or a property. This page gives the interface between Caliper and that role: how a widget
session is opened, which commands it may use, what a prompt may hold, and what each answer carries.
[Agent roles](../explanations/agent-roles.md) explains the two roles, and
[Set up the widget agent](../guides/set-up-the-widget-agent.md) shows the person's side.

!!! note "Status"
    The agent and its widget sessions are built and tested in `src/builder/agent/`; the inline prompt in the app that
    opens them is not built yet.

## Starting the agent

Caliper starts the agent as a child process and speaks ACP version 1 on its standard input and output:

```sh
<python> -I -m caliper_agent [--api-file <path>] [--command-table <path>] [--widget-scopes <path>] [options]
```

| Option | Meaning |
|---|---|
| `--api-file <path>` | Caliper API's file, which gives the loopback address and the session token |
| `--command-table <path>` | The command table as JSON |
| `--widget-scopes <path>` | The scope rule below; by default `widget_scopes.json` beside the command table. Without it, widget sessions are refused |
| `--provider claude\|openai-compatible`, `--base-url <url>`, `--model <id>`, `--effort <level>` | Where model requests go, and the defaults of a session in the general role |

The Claude API key reaches the agent in `ANTHROPIC_API_KEY`, and a key for an OpenAI-compatible server in
`CALIPER_MODEL_API_KEY`, never in its arguments. `-I` keeps the workspace folder off Python's import path.

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

**Limits.** A prompt makes at most 3 model requests and 8 tool calls, and each request asks for at most 8,000 output
tokens. A prompt that reaches a limit ends with `max_turn_requests` and a thought whose `_meta.caliper.status` is
`limit`.

## Closing

`session/cancel` stops a running prompt, which answers `cancelled`; the preview stays as it is. `session/close`, which
the agent declares as `sessionCapabilities.close`, cancels a running prompt, waits up to 10 seconds for it to end,
forgets the session and answers `{}`. A sidecar keeps at most 16 sessions.
