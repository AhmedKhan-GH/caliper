# MCP Server

`caliper-mcp` is the Model Context Protocol (MCP) server that lets an agent outside the app drive the running Caliper
app. The agent can be Claude Code, Codex or a program of your own. The agent starts `caliper-mcp` as a stdio server.
The server gives the agent one tool per command of the command layer, plus read-only resources. It runs each tool call
through [Caliper API](http-api.md) on the loopback address. The app's [side chat](side-chat.md) hands the same
executable to its agent.

## Requirements

1. The app with Caliper API on: `caliper --api`, or the app with its side chat's agent started, which starts the API
   itself ([Started by the app](http-api.md#app-startup)).
2. `caliper-mcp`, which ships beside the `caliper` executable.

While the API is on, the app writes the **API file**, `api.json` in the app-data folder ([Folders and environment
variables](../reference/folders-and-environment.md#app-data-folder)), with mode 0600. An example path is
`~/Library/Application Support/Caliper/api.json` on macOS. The app removes the file when it exits.

The file holds the address and the token of outside agents. That token runs commands as the origin `mcp`. Each change
an outside agent makes to the open spec waits on an approval card in the app until you approve or deny it. The same
holds for each command that deletes or overwrites saved data ([Who sends a command](http-api.md#command-origin)). Your
own session token runs commands without review. The app prints it on its start line, it is in no file, and
`caliper-mcp` never needs it.

`caliper-mcp` reads the API file at every call, so its configuration needs no token or key. It also follows an app that
restarts on another port. It finds the file as the app does: `--api-file` when given, otherwise
`$CALIPER_DATA_DIR/api.json`, otherwise `api.json` in the app-data folder of the OS. Thus an agent configured once
works with every run of the app that uses the same app-data folder.

## Command Line

```text
caliper-mcp [--api-file <path>] [--docs <dir>] [--sdk <dir>] [--origin mcp|agent]
            [--command-timeout <seconds>]
caliper-mcp --version | --help
```

| Option | Meaning |
|---|---|
| `--api-file <path>` | The API file to read, in place of `api.json` in the app-data folder |
| `--docs <dir>` | A folder of documentation pages, served as `caliper://docs/...` resources, such as the repository's `docs` folder |
| `--sdk <dir>` | A folder of Caliper SDK headers, served as `caliper://sdk/...` resources |
| `--origin mcp` or `agent` | The origin the calls state. The app takes the origin from the token, so this only repeats it. The side chat passes `agent` |
| `--command-timeout <seconds>` | How long a command call may take, 1 to 3600 seconds. The default is 130 because a call may wait up to 120 seconds for your approval |

Standard output carries MCP messages only. Diagnostics go to standard error, and each line starts with `caliper-mcp:`.
A usage error, or a `--docs` or `--sdk` folder that does not exist, exits with 2.

## Agent Configuration

Every MCP client needs the same things for a stdio server: the absolute path of `caliper-mcp` and its arguments.
When the app uses another app-data folder, the client also needs `CALIPER_DATA_DIR` in its environment.

**Claude Code** keeps the servers of a project in `.mcp.json` at the root of the project:

```json
{
  "mcpServers": {
    "caliper": {
      "command": "/absolute/path/to/caliper-mcp",
      "args": ["--docs", "/absolute/path/to/caliper/docs"],
      "env": {}
    }
  }
}
```

The same entry can be added from a terminal with
`claude mcp add --transport stdio --scope project caliper -- /absolute/path/to/caliper-mcp --docs /absolute/path/to/caliper/docs`.
The time limit of Claude Code for a tool call is far longer than 130 seconds, so a call that waits for your approval
needs no setting.

**Codex** keeps them in `~/.codex/config.toml`, or in `.codex/config.toml` of a trusted project:

```toml
[mcp_servers.caliper]
command = "/absolute/path/to/caliper-mcp"
args = ["--docs", "/absolute/path/to/caliper/docs"]
tool_timeout_sec = 140

[mcp_servers.caliper.env]
# CALIPER_DATA_DIR = "/path/to/the/app-data/folder"
```

`tool_timeout_sec` matters. Codex gives up on a tool call after 60 seconds by default, before the 120 seconds a call
may wait for your approval.

**Your own program** starts `caliper-mcp` as a child and writes one JSON-RPC message per line to its standard input.
The messages are `initialize` (revision 2025-11-25 or 2025-06-18), `notifications/initialized`, then `tools/list` and
`tools/call`. The server also serves revision 2026-07-28, which has no handshake.
`examples/build_a_dashboard/outside_agent.py` does this in Python with the standard library only:

```sh
python3 examples/build_a_dashboard/outside_agent.py --mcp /path/to/caliper-mcp
```

The script lists the tools, lists the catalog, reads the open spec and adds a series panel of the learning rate beside
the first panel. The call answers once you approve the card in the app. With `--model <name>`, a model that Ollama
serves on this machine chooses the calls instead, through the chat API of Ollama with a capped context.

## Tools

Each command of the [command table](http-api.md#get-v0commands) that an agent may send becomes one tool. The tool
name is the command name with `_` for its first dot (`panel_create` for `panel.create`), and its title is the command
name. The input schema is the command's arguments, the output schema is the command's result, and the description is
the command's summary.

Eight tools came with spec version 1. The tools `control_create`, `control_patch` and `control_remove` edit the controls of the open spec, as [Spec Version 1](../builder/spec-version-1.md) describes. The tools `run_state`, `run_set_param`, `run_action` and `run_pause` act on the script that the app is attached to. The tool `events_wait` reads the events of the app, and it holds the call until an event comes or its `timeout_s` ends, at most 60 seconds. `caliper-mcp` waits for it up to `timeout_s` plus the margin of a call of Caliper API. The tool `panel_create` also takes the primitive `scene`. The run tools ask for your card in the modes Ask and Accept edits. [Run Control and Events](run-control-and-events.md) describes the arguments, the results, the errors and the events. Without the app, the run tools fail with `command.not_available`.

`history.undo` and `history.redo` are yours and are not offered. An agent takes back a change with an edit that you
review. The list comes from the table of the running app, or from the table built into `caliper-mcp` when no app
answers. Thus `tools/list` always answers.

`caliper-mcp` checks a call's arguments against the command's schema before it sends anything. It reads a null on an
optional argument as the argument left out, as the command layer does. A fault comes back as a tool error that the
model can correct. A successful call returns the command's result as structured content and as text. A PNG
(`capture_frame`, `panel_render`) returns as image content.

Every failure is a tool error (`isError: true`) with one text block that holds `{"error": {"code", "message", "where"}}`.
The codes are of two kinds:

- Command layer codes, such as `command.denied` when you denied the card and `command.approval_timeout` when nobody
  decided within 120 seconds.
- `caliper-mcp` codes, such as `mcp.no_api_file` when the app is not running with the API on and `mcp.app_unreachable`
  when nothing answers on the port of the file.

What waits for you depends on the mode you give the clients of `caliper-mcp` in the app's settings (⌘,, under
Permissions):

- Ask, the default, holds spec edits, run commands and destructive commands for your card.
- Accept edits applies an edit of the open spec at once. The edit is still journaled and undoable. A run command still asks for your card.
- Read only lets reads, dry runs and edits of a preview through. It refuses every other change with `command.denied`.

A card can also allow its command for the rest of the app's run or always, until you revoke it in the settings. The
mode belongs to the app, and the command layer applies it to the token of the API file. Nothing a client sends can
choose or change it. The annotations of a tool stay as the flags of the table say.

## Resources

| URI | What |
|---|---|
| `caliper://docs/<path>` | A Markdown page under the `--docs` folder |
| `caliper://sdk/<path>` | A header under the `--sdk` folder |
| `caliper://schema/caliper-spec-v0.schema.json` | The JSON Schema of a spec's canonical JSON, version 0 |
| `caliper://schema/caliper-spec-v1.schema.json` | The JSON Schema of version 1, with the scene primitive and the controls |
| `caliper://commands/command_table.json` | The command table the tools come from |

The folders are confined. A path is a list of plain segments, and the real path of a file, with its links resolved,
must lie inside the folder.
