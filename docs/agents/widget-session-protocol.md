---
kind: reference
---

# Widget Session Protocol

This page defines how Caliper opens a widget session, what a prompt holds and what each answer carries.

## Widget Session Startup

Before it opens the session, Caliper begins a stacked preview of the open spec with `preview.begin`. The call comes from
the `ui` origin and has the target as its scope, as the [Caliper API](http-api.md) page describes for
`X-Caliper-Preview`. Caliper then sends `session/new` with the preview's id in `_meta.caliper.widget`:

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

- `preview` is required and must be the id of a preview that Caliper began.
- `scope` is optional. Without it, the agent uses the whole scope of the target's kind.

The agent checks the field against these bounds: at most 64 KiB and 16 levels, panel and source ids of the spec's form (a
lowercase letter, then at most 63 lowercase letters, digits, `_` and `-`), a property path of at most 8 dotted names, at
most 16 catalog ids and at most 64 scope names. If it cannot use a field, or a scope is wider than the rule allows, it
refuses the session with error -32602 and `data.code` `agent.bad_widget`. A `session/new` without the field opens a
session in the general role.

The answer's session id starts with `widget-`. Its `_meta.caliper.widget` repeats the target, the scope and the preview.
Its config options are `model` and `effort` (on the Claude API), with `claude-haiku-5-5` and `low` as their values. There
is no review option, because every change is a preview that the person accepts or rejects.

## Context Bundle

Before the person's request, Caliper sends what the model needs to know about the target. `build_context_bundle()` builds
it from the workspace. It returns the content blocks of the prompt, the tool scope from the [rule](widget-agent.md#tool-scope) and the `_meta`
for `session/new`.

| Part | What it holds | Bound |
|---|---|---|
| Spec fragment | The target panel and the sources it binds, through their `from` chains, as canonical TOML. The layout with the panel marked. The ids, primitives and titles of the other panels | 16 KiB, at most 64 other panels |
| Source | For each catalog source at the root of those chains, its catalog entry and a sample: 20 rows of a table, at most 1,024 values of a tensor | 8 KiB each, at most 4 |
| Render | The panel as a PNG, 640 by 400, when the model reads images and the app can draw it | 1 MiB |
| Selection | The panel, points or a range, the selected rows with their columns, and the range | 1,000 rows, 32 KiB |
| Loader | For a dropped file, in place of the spec fragment and the render: the loader that the drop made, with the entry and sample of each output | 8 KiB |
| Property | For a property: its path, its current value and its part of the edit command's arguments schema, beside the fragment | 8 KiB |

The first block is text. It names the target, says that the blocks after it are workspace data and never instructions,
and names anything left out. Each part is a `resource` block, or a `text` block for an agent that does not take embedded
context. Its text sits between `<caliper-data part="...">` and `</caliper-data>`. Caliper defuses a marker inside the
data, so a cell or a title cannot end its block. The render is an `image` block, and every block carries
`annotations.audience: ["assistant"]`.

The bundle's text is at most 64 KiB. Given the model's context length, the text and the image together take at most a
quarter of it, at four characters a token. When they do not fit, Caliper leaves out these parts in order: the sample
rows, the list of the other panels, the render, the sources of the spec fragment, the catalog entries and the selection's
rows. The target itself always stays.

## Prompts

`session/prompt` takes these content blocks. The agent declares `promptCapabilities.image` and
`promptCapabilities.embeddedContext`:

| Block | What the model reads |
|---|---|
| `text` | The text |
| `resource` with `text` | The text |
| `resource_link` | A line naming the resource |
| `image` (PNG, JPEG, GIF or WebP, at most 3.75 MB) | The image, when the model reads images. Otherwise, a line saying that it was left out |

The agent does not give the model a block whose `annotations.audience` names only the person. A prompt holds at most 4
images and 200,000 characters of text. A follow-up prompt in the same session continues the conversation and the same
preview.

## Answers

The answer to `session/prompt` carries the stop reason. Under `_meta.caliper`, it carries the token counts in `usage`
and, for a widget session, `widget`:

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

**Handoff.** When a request does not fit the target or the tools, the model calls the session's own tool `handoff` with a
`kind` (`build_from_scratch`, `loader`, `project_code`, `explanation`, `another_target` or `outside_scope`) and one
sentence. The prompt ends at once with `end_turn`, and the `handoff` field is set. A thought with
`_meta.caliper.status` `handoff` carries the same kind and sentence.

**Limits.** A prompt makes at most 6 model requests and 8 tool calls. Each request asks for at most 8,000 output tokens.
The option `--widget-requests <n>` sets another number of requests, from 1 to 20. A prompt that reaches a limit ends with
`max_turn_requests`. It also sends a thought whose `_meta.caliper.status` is `limit`, with `edits` (the commands of the
edits made) and `next` (the command of the first call that did not run, or `null`). The reply says that the change is not
finished, which edits the preview holds, and that Accept applies them or Refine lets the agent finish. An example is "The
change is not finished: the agent stopped at its limit of 6 model requests after 2 edits (source.add, panel.create).
Accept applies the edits shown, or Refine lets it finish." Caliper shows the reply in full on the card, under LIMIT and
the count of edits.

## Session Closing

`session/cancel` stops a running prompt, which answers `cancelled`. The preview stays as it is. `session/close`, which the
agent declares as `sessionCapabilities.close`, cancels a running prompt and waits up to 10 seconds for it to end. It then
forgets the session and answers `{}`. A sidecar keeps at most 16 sessions.
