# MCP server

`caliper-mcp` is the Model Context Protocol (MCP) server through which an agent outside the app, such as Claude Code,
Codex or a program of your own, drives the running Caliper app. The agent starts it as a stdio server, and it gives
the agent one tool per command of the command layer, with read-only resources beside them, and carries out each tool
call through [Caliper API](api.md) on the loopback address. The app's [side chat](side-chat.md) hands the same
executable to its agent.

!!! note "Status"
    Built on macOS and Linux (`src/builder/mcp/`), and tested against the real app with the tests
    `builder_mcp_app` and `caliper_approval_cards_app`. `examples/build_a_dashboard/outside_agent.py` is a small client
    that speaks MCP to it the way an agent does. The configuration of Claude Code and Codex below follows their public
    documentation; neither agent was run for this page.

## What you need running

1. The app with Caliper API on: `caliper --api`, or the app with its side chat's agent started, which starts the API
   itself ([Started by the app](api.md#started-by-the-app)).
2. `caliper-mcp`, which ships beside the `caliper` executable; a development build puts it at `src/builder/mcp/caliper-mcp` in the build folder.

While the API is on, the app writes the **API file**, `api.json` in the app-data folder ([Folders and environment
variables](environment.md#the-app-data-folder)), for example `~/Library/Application Support/Caliper/api.json` on macOS,
with mode 0600, and removes it when it exits. The file holds the address and the token of outside agents. That token
runs commands as the origin `mcp`: each change an outside agent makes to the open spec waits on an approval card in
the app until you approve or deny it, and so does each command that deletes or overwrites saved data
([Who sends a command](api.md#who-sends-a-command)). Your own session token, which the app prints on its start line
and which runs commands without review, is in no file, and `caliper-mcp` never needs it.

`caliper-mcp` reads the API file at every call, so it needs no token or key in its configuration, and it follows an
app that restarts on another port. It finds the file as the app does: `--api-file` when given, otherwise
`$CALIPER_DATA_DIR/api.json`, otherwise `api.json` in the app-data folder of the OS. An agent configured once
therefore works with every run of the app that uses the same app-data folder.

## Command line

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
| `--origin mcp` or `agent` | The origin the calls state; the app takes the origin from the token, so this only repeats it. The side chat passes `agent` |
| `--command-timeout <seconds>` | How long a command call may take, 1 to 3600 seconds; 130 by default, since a call may wait up to 120 seconds for your approval |

Standard output carries MCP messages only; diagnostics go to standard error, each line starting with `caliper-mcp:`.
A usage error, or a `--docs` or `--sdk` folder that does not exist, exits with 2.

## Configuring an agent

Every MCP client needs the same three things for a stdio server: the absolute path of `caliper-mcp`, its arguments,
and, when the app uses another app-data folder, `CALIPER_DATA_DIR` in its environment.

**Claude Code** keeps a project's servers in `.mcp.json` at the root of the project:

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
Claude Code's time limit for a tool call is far longer than 130 seconds, so a call that waits for your approval needs
no setting.

**Codex** keeps them in `~/.codex/config.toml`, or in `.codex/config.toml` of a trusted project:

```toml
[mcp_servers.caliper]
command = "/absolute/path/to/caliper-mcp"
args = ["--docs", "/absolute/path/to/caliper/docs"]
tool_timeout_sec = 140

[mcp_servers.caliper.env]
# CALIPER_DATA_DIR = "/path/to/the/app-data/folder"
```

`tool_timeout_sec` matters: Codex gives up on a tool call after 60 seconds by default, before the 120 seconds a call
may wait for your approval.

**Your own program** starts `caliper-mcp` as a child and writes one JSON-RPC message per line to its standard input:
`initialize` (revision 2025-11-25 or 2025-06-18), `notifications/initialized`, then `tools/list` and `tools/call`.
Revision 2026-07-28, which has no handshake, is served too. `examples/build_a_dashboard/outside_agent.py` does this in
Python with the standard library only:

```sh
python3 examples/build_a_dashboard/outside_agent.py --mcp /path/to/caliper-mcp
```

It lists the tools, lists the catalog, reads the open spec and adds a series panel of the learning rate beside the
first panel; the call answers once you have approved the card in the app. With `--model <name>`, a model that Ollama
serves on this machine chooses the calls instead, through Ollama's own chat API with a capped context.

## Tools

Each command of the [command table](api.md#get-v0commands) that an agent may send becomes one tool, named after the
command with `_` for its first dot (`panel_create` for `panel.create`) and titled with the command's name. Its input
schema is the command's arguments, its output schema the command's result, and its description the command's summary.
`history.undo` and `history.redo` are yours and are not offered; an agent takes back a change with an edit that you
review. The list comes from the running app's table, or from the table built into `caliper-mcp` when no app answers,
so `tools/list` always answers.

A call's arguments are checked against the command's schema before anything is sent, with a null on an optional argument read as the argument left out, as the command layer reads it, and a fault comes back as a tool
error the model can correct. A successful call returns the command's result as structured content and as text, and a
PNG (`capture_frame`, `panel_render`) as image content. Every failure is a tool error (`isError: true`) with one text
block that holds `{"error": {"code", "message", "where"}}`: the command layer's own codes, such as `command.denied`
when you denied the card and `command.approval_timeout` when nobody decided within 120 seconds, and codes of
`caliper-mcp`'s own, such as `mcp.no_api_file` when the app is not running with the API on and
`mcp.app_unreachable` when nothing answers on the file's port.

What waits for you depends on the mode you give `caliper-mcp`'s clients in the app's settings (⌘,, under
Permissions): Ask, the default, holds spec edits and destructive commands for your card; Accept edits applies an edit
of the open spec at once, still journaled and undoable; Read only lets reads, dry runs and edits of a preview through
and refuses every other change with `command.denied`. A card can also allow its command for the rest of the app's run
or always (until you revoke it in the settings). The mode is the app's, applied by the command layer to the API file's
token; nothing a client sends can choose or change it. A tool's annotations stay as the table's flags say.

## Resources

| URI | What |
|---|---|
| `caliper://docs/<path>` | A Markdown page under the `--docs` folder |
| `caliper://sdk/<path>` | A header under the `--sdk` folder |
| `caliper://schema/caliper-spec-v0.schema.json` | The JSON Schema of a spec's canonical JSON |
| `caliper://commands/command_table.json` | The command table the tools come from |

The folders are confined: a path is a list of plain segments, and a file's real path, with its links resolved, must lie
inside the folder.
