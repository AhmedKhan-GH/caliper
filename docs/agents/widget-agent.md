# Widget Agent

The widget agent is `caliper-agent`, Caliper's own ACP agent, in the role in which it works on one target: a panel, a
selection, a dropped file or a property. This page defines the interface between Caliper and that role. It covers how a
widget session opens, which commands the session may use, what a prompt may hold and what each answer carries.
[Agent roles](agent-roles.md) explains the two roles, [Set up the widget agent](widget-agent-setup.md) shows the
person's side, and [Widget agent conformance](widget-agent-conformance.md) tests a provider and a model against the
widget tasks.

Three pages hold the rest of the reference: [Widget Agent Startup](widget-agent-startup.md) covers the agent's options, presets, model checks and context limits. [Widget Session Protocol](widget-session-protocol.md) covers session startup, the context bundle, prompts and answers. [Inline Prompt](inline-prompt.md) covers the ⌘K surface and its settings.

## Tool Scope

The commands that a widget session may use come from one rule. Both the agent and Caliper read it from
`widget_scopes.json`, beside the command table:

| Target | Reads | Edits |
|---|---|---|
| A panel | `spec.get`, `spec.validate`, `catalog.describe`, `catalog.sample`, `panel.render` | `panel.patch`, `panel.remove`, `layout.place`, `source.add`, `source.patch` |
| A selection | the panel's reads and `data.query` | `source.add`, `panel.create`, `layout.place` |
| A dropped file | `spec.get`, `catalog.list`, `catalog.describe`, `catalog.sample`, `loader.describe`, `data.query` | `source.add`, `panel.create`, `layout.place` |
| A property of a panel | `spec.get`, `catalog.describe` | `panel.patch` |
| A property of a source | `spec.get`, `catalog.describe` | `source.patch` |

The agent applies the rule to the command table, not to a fixed list. A read must not change anything. An edit must take
a boolean `dry_run` and must not delete or overwrite saved data. The commands `spec.set_text`, `spec.new`, `spec.open`,
`applet.open`, `layout.set`, `data.import` and the `history`, `applet` and `preview` commands are never in a scope. The
command `panel.render` leaves the scope when the model cannot read images. The agent answers a call to any other command
as a call to an unknown tool, and the call never reaches Caliper.

A command whose descriptor in the command table says `"agents": false` belongs to the person. These commands are
`history.undo` and `history.redo`. The agent offers such a command as a tool in no session, general or widget. If the
model calls one anyway, the agent answers as for an unknown tool. The agent's client of Caliper API also refuses the call
with `command.not_allowed`, before it sends anything. That code is the command layer's code for the same refusal. A table
without the `agents` field opens every command to agents. To revert a change, the agent makes an ordinary edit, which the
person reviews like any other.

Every call of a widget session carries `X-Caliper-Preview` with the session's preview. The agent sends every edit without
`dry_run`, so it applies to the preview and never to the open spec. The agent sends no permission request in a widget
session, and it never begins, commits or discards a preview. Only the person commits it, with Accept.

## Side Chat Use

`caliper-agent` can be the [side chat](side-chat.md)'s custom agent in the general role. An example is a local model
through the Ollama preset, with no account or key. The side chat sends `session/new` with one MCP server, `caliper`. That
server is `caliper-mcp` with `--api-file <the session's own API file>`, and its token has the origin `agent`.

The agent starts no MCP server. It calls Caliper API itself with that file, so its commands run as the origin `agent`.
Its edits of the open spec wait on the approval cards, which the chat mirrors as rows. Caliper's server is any server
named `caliper`, in any form, or any server whose program is `caliper-mcp`. Without a Caliper server, a session uses
`--api-file`.

A session whose Caliper server names no usable `--api-file` (an absolute path) is refused with -32602 and `data.code`
`agent.no_session_api_file`. An example is a server in the HTTP form, which has no arguments. The agent does not fall
back to `api.json`, because its token is the origin `mcp`.

Each call of a Caliper command carries the command in `_meta.caliper.command` of its `tool_call` update, such as
`{"caliper": {"command": "panel.create"}}`. As a result, the side chat shows the call as that command and folds its
approval into its row.
