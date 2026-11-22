---
kind: reference
---

# Inline Prompt

This page describes the ⌘K prompt on the spec page and the settings popup that sets the widget agent.

## Inline Surface

The inline surface is the app's side of a widget session on a panel. On the spec page, a click on a panel focuses it, and
its frame turns bronze. ⌘K (Ctrl+K on Windows), or the page bar's ⌘K Command, opens a one-line prompt at the top of the
focused panel, under its title strip. The panel's content is drawn below the prompt, so the prompt covers nothing.

With no panel focused, ⌘K opens the page's command input in the page bar instead. Typing a panel's title focuses it, and
Enter opens its prompt. Escape leaves the prompt's innermost state first, then the command input, then the panel's
focus, and last goes Home.

**The other targets.**

- **A selection.** A shift-drag across a series panel's plot selects a range of x, such as steps. Across a 2-D points
  panel, it selects a rectangle of points. The selection is drawn in bronze, and a shift-click or Escape clears it. ⌘K on
  that panel opens the prompt for the selection. Its context bundle carries the selected rows (the drawn steps or
  points, at most 1,000) and the range. A change that adds a panel, such as a table of the selection, is drawn as the
  preview of the whole spec. The new panel stands where the layout puts it, in a dashed bronze frame with PREVIEW in its
  strip.
- **A dropped file.** After files are dropped on the spec page, each source that the import made shows ⌘K at the end of
  its row in the builder's Sources palette. ⌘K with that row selected and no panel focused does the same. The prompt opens
  under the row, for the file's loader and its catalog ids. The card shows an answer without a change over up to four
  lines, with Text for the whole of a longer one. A proposed panel is previewed as above. With no spec open, the preview
  first runs `spec.new` from the `ui` origin. A discard, or the surface going away, undoes that step again while it is
  still the top of the undo stack, known by the step's id from `history.list`. If another step was committed after it,
  the new spec stays. Accept leaves two undo steps: the new spec, then the change.
- **A property.** In build mode, ⌘K shows beside the focused panel's title and beside each of its options in Properties
  while the row is under the pointer. It opens the prompt under that row for that one property, such as
  `options.colormap`, and the panel previews the edit.
- **A property of a source.** With a source of the open spec selected in the builder's Sources palette, ⌘K shows in
  Properties beside what it reads (its catalog id, the source it derives `from`, or its SQL). It also shows beside its
  transforms, which count as one property, `transform`. The prompt opens under that row. The context bundle carries the
  source with its chain and catalog entry, the property's value and its part of the `source.patch` schema. The session
  may read with `spec.get` and `catalog.describe` and edit with `source.patch` only. The change shows as its compact
  diff under the row, such as the source's changed transform. The panels that read the source show it after the person
  accepts.

The prompt row shows the model and effort that the session reports. A click on them lists the models that the session
offers, and the choice goes to the session with `session/set_config_option` and to every later session. While the agent
works, the row shows its latest step: the command it calls, or the last line of its summarized thinking.

**The session.** The surface opens one widget session for each prompt that it opens:

1. When the person submits the first request, it sends `preview.begin` from the `ui` origin, scoped to the panel, with an
   id that starts with `widget-`.
2. It builds the [context bundle](widget-session-protocol.md#context-bundle) on that preview.
3. It opens the session with `session/new`, whose `_meta.caliper.widget` holds the target, the preview and the bundle's
   scope.
4. It sends the request as `session/prompt`. On the first prompt only, the bundle's blocks come before the request.
5. When the turn ends, it reads `preview.diff` and draws the panel from the preview.
6. Accept sends `preview.commit` from the `ui` origin, which files one undo step.
7. Reject, Escape, a cancel, a handoff or a failure sends `preview.discard`.
8. A follow-up (Refine) is another `session/prompt` on the same preview.
9. Closing the prompt ends the session with `session/close`.

One `caliper-agent` process serves every ⌘K. It starts at the first ⌘K, with the provider's key placed only in its
environment. It keeps running until the settings change. A change ends it at once, with its API file and token, and the
next ⌘K starts it again from the new settings.

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
| NO PROVIDER | The input. No agent starts | Setup, Ask in chat (Enter) |

Ask in chat hands the request, the target, the conversation so far and the preview's diff to the side chat, and it closes
the prompt. The diff is marked as not applied.

**Layout.** The prompt row, the diff and the action row sit inside the panel. As a panel shrinks, the surface drops the
reply line first, then the diff lines past the first (folded into "+N"), then the model label, then the keys of the
buttons. Below a minimum height, the surface takes the whole panel.

## App Settings

The widget agent's settings are a popup over any page. It opens from the landing screen's **⌘, Agent**, from the key ⌘,
(Ctrl+, on Windows) on any page, and from the inline surface's **Setup** (no provider) and **Model** (after a refusal).
Escape or esc Close closes it.

| Row | What it holds |
|---|---|
| Provider | The presets, one chosen |
| Base URL | Fixed text for Claude, OpenAI, Gemini and OpenRouter. A field for Ollama, LM Studio and Custom, with the tags Remote address, Unencrypted and Invalid address |
| Model | A field, and as choices the Claude preset's four models or the server's list after **List** (`caliper-agent probe --list-models`) |
| Effort | Claude only: low, medium, high, xhigh and max, which each of its four models takes |
| Key | Set or Not set. A password field with Save, or Replace and Remove. No row for Ollama and LM Studio, nor for Custom before its address is valid |
| Capabilities | Tool calling, vision and the context length, with where each came from (probe, server, known, preset, override, assumed). Yes, No and Auto for the person's override (`caliper-agent probe --set`). **Probe** (`--refresh`). The tag Context cache about N GB when a server on this machine would hold a context cache of 4 GB or more |
| Note | Shared GPU, while a live run is attached and the base URL is a loopback address |

The settings never show a key or read one back. They save and remove keys through the key store, and they wipe the
field's text when the person saves or cancels. A hidden app's key store is an in-memory one, so it never touches the
Keychain.

A key leaves the store at one point only: the start of a probe. **Probe**, **List** and the overrides run
`caliper-agent probe` as a child process. The app copies the chosen provider's key into that child's environment, as it
does for the sidecar, in `ANTHROPIC_API_KEY` or `CALIPER_MODEL_API_KEY`. It also removes every other key variable and
every variable that could redirect the child, such as `ANTHROPIC_BASE_URL` and `OPENAI_BASE_URL`. As a result, no key or
endpoint of the app's own environment reaches the provider. The app passes `CALIPER_AGENT_OFFLINE` through when it has
it. A provider that needs no key is probed with no key variable at all.

If the store cannot place the key, or the provider requires one and there is no store, the probe does not run. Its tag
says why: No key, Malformed key or Unreadable key. When the app quits or the settings close while a probe runs, the app
ends the probe's child at once, and the frame never waits for it. The settings drop an answer that comes after the model
or the address changed.

**`widget-agent.json`**, in the app-data folder beside `side-chat.json`, holds the choice and never a key:

```json
{"version": 1, "preset": "ollama",
 "providers": {"ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen3.8:27b"},
               "anthropic": {"model": "claude-haiku-5-5", "effort": "low"}}}
```

A file that does not parse counts as no choice, and the app drops an unknown preset. The probe's answers and the
person's overrides are kept apart, in `model-capabilities.json` in the same folder, which the agent reads with
`--cache-dir`.

**The agent's start.** The app makes the inline surface's agent from the settings at the first ⌘K after they change. A
change applies at once, on any page. These changes count: a preset, an address, a model, an effort, a key saved or
removed, or a probe's answer for the current model. An open prompt closes as Close closes it, so a running turn is
cancelled and its preview is discarded. The agent goes with its sidecar, and the app revokes the sidecar's API file and
token. A removed key thus leaves the sidecar's environment at once.

The app starts `caliper-agent` with `--preset`, `--base-url` (Ollama, LM Studio and Custom), `--model`, `--effort`
(Claude) and `--cache-dir`. It puts the provider's key in the agent's environment: in `ANTHROPIC_API_KEY` for Claude, in
`CALIPER_MODEL_API_KEY` for the others, and none for a local server.

⌘K has no provider, and it offers Setup and Ask in chat, in these cases:

- No preset is chosen.
- The address is not valid.
- An OpenAI-compatible preset has no model.
- A required key is not set.
- The model is known to lack tool calling (the person's override, or a probe answer younger than 30 days).

The settings name that state in a tag: No provider, Invalid address, No model, No key or No tool calling. While the GPU
note applies, the prompt row shows it after the model that the session reports.

## Related Pages

- [Widget Agent](widget-agent.md)
- [Widget Agent Startup](widget-agent-startup.md)
- [Widget Session Protocol](widget-session-protocol.md)
- [Set up the widget agent](widget-agent-setup.md)
