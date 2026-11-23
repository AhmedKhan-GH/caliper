# MCP server

`caliper-mcp` is a small executable that an outside agent, such as Claude Code or Codex, starts as a Model Context
Protocol (MCP) server on its standard input and output. It gives the agent one tool per command of the
[command layer](command-layer.md) and a set of read-only resources (the documentation, the SDK headers, the JSON
Schema of the [spec format](spec-format.md) and the command table), and it carries out each tool call by calling
Caliper API on the loopback address of the running app. The side chat of the app hands the same executable to the
agent it runs over ACP ([ACP integration](acp-integration.md)). This page elaborates sections 7, 8, 9 and 11 to 17 of
the sprint 3 contracts and records the decisions taken while building it.

## What is built and what is planned

Built on this branch, in `src/builder/mcp/` with headers in `src/builder/include/caliper_builder/mcp/`:

- the executable `caliper-mcp` and the library `caliper_builder_mcp` behind it;
- the protocol engine, which serves MCP revision 2026-07-28 and the earlier revisions 2025-11-25 and 2025-06-18 on
  one connection;
- the tools, derived at run time from the command table, and the conversion of results and errors into tool results;
- the resources under the `caliper://` scheme, with the docs and SDK folders confined;
- a strict HTTP/1.1 client for 127.0.0.1 only, and the reader of the API file;
- unit tests, tests that drive `caliper-mcp` as a child process against a fake of Caliper API, and an end-to-end test
  against the real app;
- the command layer's endpoints in the app's Caliper API, so every tool acts on the running app: the walk-through's
  example client, `examples/build_a_dashboard/outside_agent.py`, lists the tools, reads the catalog and the open spec
  and adds a panel, which waits on an approval card ("MCP agent") until the person approves it. The reference page is
  `docs/agents/mcp-server.md`.

Planned:

- Progress notifications while a call waits for approval, a Windows version of the HTTP client, and the other items
  under [Open questions](#open-questions).

## How it runs

```
agent (Claude Code, Codex, the app's side chat)
   │ starts, then speaks MCP on stdin and stdout
   ▼
caliper-mcp ── reads api.json ──► port and the mcp token
   │ HTTP/1.1 on 127.0.0.1, one request per call
   ▼
Caliper API in the running app ──► command queue ──► command layer
```

The agent starts one `caliper-mcp` per session. Standard output carries MCP messages only, one JSON object per line;
diagnostics go to standard error, each line starting with `caliper-mcp:`. When standard input ends, `caliper-mcp`
lets the calls still running finish for up to 2 seconds, since a client may close its input and still read the
answers, then cancels the rest (they answer with the tool error `mcp.cancelled`), ends its subscriptions and exits
with 0. It keeps no state that matters between
requests: the command table it fetched from the app is a cache, and the API file is read again at every call.

### Command line

```text
caliper-mcp [--api-file <path>] [--docs <dir>] [--sdk <dir>] [--origin mcp|agent]
            [--command-timeout <seconds>]
caliper-mcp --version | --help
```

| Option | Meaning |
|---|---|
| `--api-file <path>` | The API file to read. Without it, `api.json` in the app-data folder, which `CALIPER_DATA_DIR` moves. |
| `--docs <dir>` | A folder of documentation pages, served as `caliper://docs/...` resources. Usually the `docs` folder of the repository or of an installed copy. |
| `--sdk <dir>` | A folder that holds the Caliper SDK headers, served as `caliper://sdk/...` resources. Usually the installed SDK prefix. |
| `--origin mcp` or `agent` | The value of the `X-Caliper-Origin` header of command calls; `mcp` by default. The app passes `agent` when its own side chat runs `caliper-mcp` (see [Changes to the contracts](#changes-to-the-contracts)). The origin of a call is the token's, not the header's: the app's Caliper API ignores the header, and the stand-in host refuses one that names another origin than its token's, so the header only states what the token already says. |
| `--command-timeout <seconds>` | How long a command call may take, 1 to 3600 seconds; 130 by default, since a call may wait up to 120 seconds for the person's approval. |

A folder given with `--docs` or `--sdk` must exist; otherwise `caliper-mcp` says so on standard error and exits
with 2, so the agent shows a server that failed to start rather than one that silently lacks its resources. A usage
error also exits with 2.

## Configuring an agent

Every MCP client needs the same three things for a stdio server: the absolute path of the program, its arguments, and
optionally variables for its environment. `caliper-mcp` needs no variable when the app uses the default app-data
folder, and `CALIPER_DATA_DIR` (or `--api-file`) when it does not. It never needs a token or a key in its
configuration: it reads a token from the API file at run time. The token in `api.json` is not the person's session
token: it runs commands as the origin `mcp`, so every spec edit an outside agent makes through `caliper-mcp` waits for
the person's review in the app, and every destructive command for their approval
(Caliper API, `docs/agents/http-api.md`, under Who sends a command).

The configuration below follows the public documentation of each agent as it was on 15 November 2026. Neither agent
was run for this page.

### Claude Code

A project-scoped server lives in `.mcp.json` at the root of the project:

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

An entry without a `type` is a stdio server. The same server can be added from a terminal with
`claude mcp add --transport stdio --scope project caliper -- /absolute/path/to/caliper-mcp --docs /absolute/path/to/caliper/docs`;
the default scope, `local`, keeps it in the user's own configuration instead. Claude Code's default time limit for a
tool call is far longer than 130 seconds, so no setting is needed for approvals; a `timeout` field in milliseconds on
the entry sets one for this server only. A resource is referred to in a prompt as
`@caliper:caliper://docs/reference/cli.md`.

Claude Code asks a stdio server for revision 2026-07-28 first and falls back to the `initialize` handshake when the
server does not support it; `caliper-mcp` supports both.

### Codex

Codex keeps stdio servers in `~/.codex/config.toml`, or in `.codex/config.toml` of a trusted project. Its format is
TOML, with the same three fields:

```toml
[mcp_servers.caliper]
command = "/absolute/path/to/caliper-mcp"
args = ["--docs", "/absolute/path/to/caliper/docs"]
tool_timeout_sec = 140

[mcp_servers.caliper.env]
# CALIPER_DATA_DIR = "/path/to/the/app-data/folder"
```

`tool_timeout_sec` matters: its default is 60 seconds, shorter than the 120 seconds a command may wait for approval,
so without it Codex gives up on a call that the person is about to approve. `codex mcp add caliper --
/absolute/path/to/caliper-mcp --docs /absolute/path/to/caliper/docs` writes the entry without the time limit.

### The app's side chat

When the app runs an agent over ACP, it sends `caliper-mcp` in `session/new` as a stdio server with an absolute
`command`, `args` such as `["--api-file", "<this session's API file>", "--origin", "agent", "--docs", "<docs>"]` and
an empty `env`. The API file is the agent session's own, `api-agent-<pid>-<n>.json`, whose token has the origin
`agent` and is revoked when the session ends; the agent never gets `api.json` or the person's token. The app always names the `caliper-mcp` that ships beside it, never one taken from a workspace.

## The protocol

### Revisions

The current revision of the MCP specification on 15 November 2026 is **2026-07-28**. It removed the `initialize`
handshake and made MCP stateless: every request carries its protocol version and the client's capabilities in
`params._meta`, and a server must offer `server/discover`. Clients written for the earlier, handshake-based
revisions are still common, so `caliper-mcp` is what the specification calls a dual-era server:

| Revision | How a request selects it | What changes in the answers |
|---|---|---|
| 2026-07-28 | The request carries `io.modelcontextprotocol/protocolVersion` with this value, and `io.modelcontextprotocol/clientCapabilities`, in `params._meta`. | Every result has `resultType: "complete"` and the server's identity in `_meta["io.modelcontextprotocol/serverInfo"]`; the list and read results and `server/discover` also carry `ttlMs` and `cacheScope`. |
| 2025-11-25, 2025-06-18 | An `initialize` request names the revision; the answer confirms it, and the process serves later requests that carry no revision in `_meta` under it. | Results have the shape of that revision, without `resultType` or caching hints. |

The rules, in order:

- A request whose `_meta` names a protocol version is served under that version on its own. A version other than
  2026-07-28 is answered with error -32022 (`UnsupportedProtocolVersion`), whose `data` lists the supported versions
  (`["2026-07-28", "2025-11-25", "2025-06-18"]`) and the requested one. A missing or malformed
  `clientCapabilities` is error -32602.
- `initialize` without such a `_meta` starts the legacy era. When it asks for 2025-11-25 or 2025-06-18, the answer
  names the same revision; for anything else, including 2024-11-05 and 2026-07-28, it names 2025-11-25, which is what
  the legacy negotiation rule asks (the client then decides whether it can continue). Another `initialize` later is
  answered the same way.
- `ping` is answered with an empty result in either era, also before `initialize`. Revision 2026-07-28 removed
  `ping`; answering it costs nothing and keeps older clients working.
- Any other request that names no version, before an `initialize`, is error -32602, with the supported versions in
  `data`, so a client learns what to send.

| Method | 2026-07-28 | Legacy | Notes |
|---|---|---|---|
| `server/discover` | yes | no (-32601) | `supportedVersions`, `capabilities`, `instructions` |
| `initialize`, `notifications/initialized` | no (-32601) | yes | |
| `ping` | yes | yes | |
| `tools/list`, `tools/call` | yes | yes | |
| `resources/list`, `resources/read`, `resources/templates/list` | yes | yes | |
| `subscriptions/listen` | yes | no (-32601) | Only `toolsListChanged` is honoured |
| `notifications/cancelled` | yes | yes | |
| anything else | -32601 | -32601 | No prompts, completions, logging or resource subscriptions |

The capabilities are `{"tools": {"listChanged": true}, "resources": {}}` in both eras.

### JSON-RPC handling

Each line on standard input is one message. `caliper-mcp` treats every line as hostile:

| Line | Answer |
|---|---|
| Not JSON, or not valid UTF-8 | -32700, without an id |
| Nested deeper than 128 levels (checked before parsing) | -32600, without an id |
| Longer than 8 MiB (dropped without being kept) | -32600, without an id |
| A batch (a JSON array) | -32600, without an id; MCP has had no batches since 2025-06-18 |
| Not an object, no `"jsonrpc": "2.0"`, no method, a method that is not a string | -32600, with the id when it can be read |
| An id that is not a string or an integer (null, a fraction, an object) | -32600, without an id |
| `params` that is not an object | -32602 |
| An id that a running request or an open subscription already uses | -32600 |
| A response (the server sends no requests) or an unknown notification | Nothing |
| An empty line | Nothing |

`tools/list`, `tools/call`, `resources/list` and `resources/read` may wait on the app or the file system, so each runs
on a thread of its own, at most 32 at once; a 33rd is answered with -32603. Everything else is answered at once, so a
call held for approval never delays a `ping` or a cancellation. `notifications/cancelled` for a running call stops its
HTTP request within 100 ms, and no answer is sent for it, as the specification asks; for an open subscription it ends
the subscription without an answer.

### Subscriptions and list changes

The tool list changes only when the command table changes, which happens when the app is replaced by another version
or when it gains or loses its command layer. `caliper-mcp` notices this when it reads the table for a call. A legacy
client that sent `notifications/initialized` then receives `notifications/tools/list_changed`. A modern client that
opened `subscriptions/listen` with `toolsListChanged: true` receives the acknowledgment with the filter it asked for,
reduced to `toolsListChanged`, and then the same notification tagged with its subscription id. When `caliper-mcp`
stops, it answers each open subscription with its closing result.

## Tools

### From the command table to tools

Each command of the table that an agent may run becomes one tool. The table is read generically at run time, so a
command added to the table, or a schema changed in it, reaches the agent without a change to `caliper-mcp`. A command
whose descriptor says `"agents": false`, today `history.undo` and `history.redo`, the person's own tools, is left out
without a report, and a forced call of its tool name is refused as an unknown tool; the command layer refuses it to
the `agent` and `mcp` origins anyway, with `command.not_allowed`. An app whose table carries no `agents` field is
read as allowing every command to agents. An agent takes back a change with an edit that reverts it.

| Tool field | Comes from |
|---|---|
| `name` | The command name with `_` for the dot: `panel_create` for `panel.create`. A group has no underscore, so the first underscore marks the dot and the mapping is reversible. The names always pass MCP's rule (1 to 128 characters of `A-Z a-z 0-9 _ - .`), and a test checks that they are unique. |
| `title` | The command name, `panel.create`. |
| `description` | The command's summary, which is written for an agent that has never seen Caliper. |
| `inputSchema` | The command's `args` schema, unchanged: always a closed object schema. |
| `outputSchema` | The command's `result` schema. When it has a `png_base64` property, that property is removed (and from `required`), since the PNG travels as image content. |
| `annotations.readOnlyHint` | True when the command does not mutate, except `app.launch` (below). |
| `annotations.destructiveHint` | The command's `destructive` flag, stated for every tool, since MCP's default is true. |
| `annotations.openWorldHint` | False: every tool acts on the local app only. |

Output schemas and structured content came with revision 2025-06-18, so every revision `caliper-mcp` serves has them
and every tool states its output schema. A client that asks for an older revision is answered with 2025-11-25.

`app.launch` does not mutate the workspace in the command layer's sense, but it tears down the open applet and
changes the page the person sees (an attached live session stays attached), so it is not read-only in MCP's sense ("does not modify its environment") and its
`readOnlyHint` is false. A client may approve read-only tools without asking, and launching an applet should not be one
of them.

For example, `app.launch` becomes:

```json
{
  "name": "app_launch",
  "title": "app.launch",
  "description": "Opens the page of one applet, as a click on its row on the app's landing screen does. ...",
  "inputSchema": {"type": "object", "additionalProperties": false, "required": ["id"],
                  "properties": {"id": {"type": "string", "pattern": "^[A-Za-z0-9._-]{1,128}$", "...": "..."}}},
  "outputSchema": {"type": "object", "additionalProperties": false,
                   "required": ["already_open", "id", "status", "version"], "properties": {"...": "..."}},
  "annotations": {"readOnlyHint": false, "destructiveHint": false, "openWorldHint": false}
}
```

### Which table

The app is the truth. When the API file names a running app and `GET /v0/commands` answers with a sound table,
`caliper-mcp` uses the app's table; otherwise it uses the table compiled into it
(`<caliper_builder/commands/command_table.h>`). The choice is made again whenever the API file names another port
or token, and the answer is kept until then.

- A 404 from `GET /v0/commands` means an app without a command layer: the compiled-in table is used, and that is kept
  for this app.
- A table whose envelope is wrong (not `{"version": 0, "commands": [...]}`, more than 1024 commands, over 8 MiB) is
  refused as a whole, and the compiled-in table is used.
- A command of the app's table that lacks a field, repeats a name or uses a schema keyword outside the command layer's
  subset is left out alone, with a line on standard error. Its arguments could not be checked before sending, and an
  agent is better served by a missing tool than by one whose calls fail in ways it cannot predict.
- When the app is not running, the compiled-in table is used, so `tools/list` always answers.

`tools/list` returns every tool in the table's order, which is deterministic, on one page; the command-table
resource is the same table without the commands left out. Its `ttlMs` is 0, since
the list depends on the running app.

## Calling a tool

### Before anything is sent

`tools/call` finds the tool in the current table; an unknown name is error -32602, and `arguments` that are not an
object are error -32602, as the specification asks for malformed requests. Absent or null arguments mean `{}`. The
arguments are then checked against the command's `args` schema with the commands library's validator
(`validate_command_value`), and a fault is a tool error with the code `command.invalid_args` and `where` naming the
value, before the API file is even read. A model can correct its call from that. Arguments over 1 MiB as JSON text,
the limit of the endpoint, are refused the same way.

### Where a call goes

| Tool | Request | Notes |
|---|---|---|
| `app_info` | `GET /v0/info` | |
| `app_applets` | `GET /v0/applets` | |
| `app_launch` | `POST /v0/applets/{id}/launch` | The id is checked again against the API's grammar before it enters the path. |
| `capture_frame` | `GET /v0/capture.png` | The PNG becomes image content. |
| every other tool | `POST /v0/commands/{name}` | The body is the arguments; `Content-Type: application/json` and `X-Caliper-Origin: mcp` (or `agent`). |

The four tools of API v0 use their v0 endpoints also when the app has a command layer, since those endpoints exist in
every version of the API and need no handler to be bound. Every request carries `Authorization: Bearer <token>` and
`Host: 127.0.0.1:<port>`, as the API's checks require.

### Results

A successful call becomes a `CallToolResult` with:

- `structuredContent`: the command's result object (for the v0 routes, the JSON of the endpoint, which has the same
  fields as the command's result);
- a text block with the same JSON, as the specification recommends for clients that do not read structured content;
- for a result with `png_base64` (`capture.frame`, `panel.render`): an image block first, `{"type": "image", "data":
  <base64>, "mimeType": "image/png"}`, while `width` and `height` stay in the structured content. The image is
  checked to start with a PNG signature and header. For `capture_frame`, `caliper-mcp` encodes the PNG bytes of the
  endpoint itself and reads the size from the PNG header.

`app_launch` relays the launch's answer as it is. When the app answers `status` `PENDING_APPROVAL` (the dashboard
would replace the open spec's unsaved changes, so its `applet.open` waits on an approval card) or `PENDING`, the call
succeeds, and the text block an agent reads holds that status and the `note`, which says the dashboard is not open
yet and that `spec_get` names the open spec once it is; so an agent checks rather than believes that it opened.

The structured content is checked against the tool's output schema before it is returned, since a client may check
it too. A result that does not match, which means the app and the table disagree, becomes the tool error
`mcp.invalid_result`, whose message still carries the result, so the agent loses nothing.

### Errors

Every failure of a call is a tool error (`isError: true`), so the model sees it, with one text block that holds a JSON
object:

```json
{"error": {"code": "command.denied", "message": "The person denied the invocation.", "where": ""}}
```

An error envelope from the command layer passes through unchanged, with `approval` when the code is
`command.needs_approval`. The faults of the transport have codes of their own:

| Code | When |
|---|---|
| `command.invalid_args` | The arguments fail the schema, or exceed 1 MiB (also a 400 from a launch). |
| `mcp.no_api_file` | There is no API file: the app is not running with `--api`, or another app-data folder is in use. |
| `mcp.bad_api_file` | The API file is not one Caliper writes (see [The API file](#the-api-file)). |
| `mcp.app_unreachable` | Nothing answers on the file's port: the app has exited and left its file, or is starting. |
| `mcp.unauthorized` | A 401 even after the API file was read again. |
| `mcp.forbidden` | A 403 from the `Host` or `Origin` checks. |
| `mcp.timeout` | No answer within the time limit of the call. |
| `mcp.app_busy`, `mcp.app_timeout` | A 503 or a 504 from the app; the call may be retried. |
| `mcp.no_command_layer` | `POST /v0/commands/{name}` answered 404: "this version of the app has no command layer yet". |
| `mcp.http_error` | Any other status, with the API's own message. |
| `mcp.bad_response`, `mcp.response_too_large` | An answer that is malformed or larger than 64 MiB. |
| `mcp.invalid_result` | A result that does not match the output schema, or a `png_base64` that is not a PNG. |
| `capture.unavailable` | `GET /v0/capture.png` answered 501: the renderer cannot read the window back (GL and Vulkan; Metal can). |
| `app.applet_not_found`, `app.launch_refused` | A launch answered 404 or 409. |

### Approval

A destructive command sent with the origin `mcp` waits in the app while the person decides, for up to 120 seconds
(contracts section 15), and so does an edit of the open spec while the person reviews the agent's changes, which is
on by default for `mcp` and `agent`. `caliper-mcp` therefore allows 130 seconds for a command call by default, both as the silence
it accepts while waiting for the first byte and as the whole exchange. Approved, the call returns the command's
result; denied, the tool error `command.denied`; after the app's limit, `command.approval_timeout`. When the agent
cancels the call, `caliper-mcp` closes the connection. There is no tool and no endpoint that approves, so an agent
cannot approve its own request.

What waits depends on the mode the person gives `mcp` clients in the app (Ask by default, Accept edits, Read only),
and on the rules a card's answer adds ([Command layer](command-layer.md), Agent permissions). The command layer
applies them to the origin `mcp`; `caliper-mcp` needs no change for them, sends no `_meta` or header that could
select one, and offers no tool that reads or sets them. A call that Read only refuses is the tool error
`command.denied` with the command layer's message. The table's `effect` (`read`, `edit`, `change`) is read with the
other flags when `caliper-mcp` parses the app's table; its annotations stay `readOnlyHint` from `mutates` and
`destructiveHint` from `destructive`. Under Allow all, which the person saves for `mcp` clients in the settings, an
`mcp` client's calls run without cards except a data-loss card ([Command layer](command-layer.md), Agent permissions).

## Resources

| URI | What | MIME type |
|---|---|---|
| `caliper://docs/<path>` | A Markdown page (`.md`) under the `--docs` folder | `text/markdown` |
| `caliper://sdk/<path>` | A header (`.h`, `.hpp`) under the `--sdk` folder | `text/x-c`, `text/x-c++hdr` |
| `caliper://schema/caliper-spec-v0.schema.json` | The JSON Schema of a spec's canonical JSON, compiled into `caliper-mcp` from `src/builder/spec/schema/` | `application/schema+json` |
| `caliper://commands/command_table.json` | The command table the tools come from, in the form of `GET /v0/commands` | `application/json` |

`resources/list` lists them sorted by URI, 200 to a page, with an opaque `nextCursor`. `resources/templates/list`
gives `caliper://docs/{+path}` and `caliper://sdk/{+path}` for the folders that were given. The caching hints are 60
seconds for the folders, an hour for the schema and templates, and 0 for the command table. A resource that does not
exist, or a path that is refused, is error -32602 with the URI in `data` under 2026-07-28, and -32002 under the legacy
revisions, which used that code.

The folders are confined. The path part of a URI is percent-decoded once and must then be segments of
`A-Z a-z 0-9 . _ -` that are not empty and do not start with a dot; so `..`, `.` and hidden files, a backslash, a
colon, a space and a NUL are refused before the file system is asked. The file's real path, with every symbolic link
resolved, must lie inside the folder's real path, be a regular file with an allowed extension (also after the links)
and be at most 4 MiB. Listing follows the same rules, does not descend into hidden folders or follow linked folders,
goes at most 16 levels deep and lists at most 5000 files per folder. `caliper-mcp` only reads these files.

## Reaching the app

### The API file

The app writes `api.json` in its app-data folder while it serves the API, with mode 0600, and removes it when it
exits. Its `token` is the token of outside agents, with the origin `mcp`; the person's own token, which runs commands
unreviewed, is printed on the app's start line and is in no file. `caliper-mcp` finds it as the app does: `--api-file` when given, otherwise `$CALIPER_DATA_DIR/api.json` when
the variable is set and not empty, otherwise `api.json` in the app-data folder of the OS
(Folders and environment variables (`docs/reference/folders-and-environment.md`)). It reads the file strictly: at most 64 KiB,
`version` 1, `port` 1 to 65535, `url` exactly `http://127.0.0.1:<port>/v0`, and `token` 16 to 256 characters of
`A-Z a-z 0-9 _ -`, so that it cannot carry anything into a header. Unknown keys are ignored. A file that does not
parse is read once more after 100 ms, in case the app was writing it. `pid` and `stream_url` are optional: the
stand-in host of the [builder session](builder-session.md) serves no stream and writes no `stream_url`. `caliper-mcp`
takes the stream's address from the file's `stream_url` when it needs one and never builds it; it offers no tool for
the stream in this phase.

The file is read at every call. When a request is refused with 401 or its connection is refused, the file is read
once more and the request is sent again if the file now names another port or token. Neither fault means that the
request ran, so the retry cannot run a command twice. This is how `caliper-mcp` follows an app that restarts on another
port with another token, without being restarted itself.

### The HTTP client

The client is written for Caliper API and nothing else. It connects only to 127.0.0.1, refusing any other host name
or address before it makes a socket. It sends one request per connection with `Connection: close`, speaks no TLS,
follows no redirect (a 3xx is the error `http.redirect`), and refuses a response with a transfer coding. A response
head is at most 16 KiB with at most 100 fields, a folded or malformed header line is refused, a `Content-Length` must
be one decimal number that matches the body exactly, and a body without one runs to the close, at most 64 MiB. It
checks the request before sending it, so no method, target or header value can carry a line break into the head. It
never puts a header value into an error message.

| Limit | Value |
|---|---|
| Connect | 2 seconds |
| API v0 endpoints (info, applets, launch, capture) | 15 seconds, as silence and as the whole exchange |
| `GET /v0/commands` | 10 seconds; the table at most 8 MiB |
| `POST /v0/commands/{name}` | 130 seconds by default (`--command-timeout`) |
| A response body | 64 MiB, which a `panel.render` of the largest size could exceed |
| Cancellation | checked at least every 100 ms |

## What works against API v0

API v0, as it is on `sprint2`, has `GET /v0/info`, `GET /v0/applets`, `POST /v0/applets/{id}/launch`,
`GET /v0/capture.png` and the stream, and no command layer. Against it:

- `app_info`, `app_applets`, `app_launch` and `capture_frame` work. `capture_frame` works in the app's window on Metal,
  hidden or not, and in `caliper shot --api`; on the GL and Vulkan renderers it is the tool error
  `capture.unavailable`.
- `tools/list` offers the compiled-in table, since `GET /v0/commands` answers 404.
- Every other tool passes its argument check and is then the tool error `mcp.no_command_layer`, because
  `POST /v0/commands/{name}` answers 404 ("no such endpoint in Caliper API v0"). The v0 server routes a request before
  it checks the token or the body, so that 404 arrives even for a request with a body.

When the app gains the command layer, every tool works without a change to `caliper-mcp`, and the tool list becomes
the app's own.

## Security

- **The token** is read from the API file and goes only into the `Authorization` header of requests to
  127.0.0.1. It is never printed, logged or put into a tool result, a resource or an error message. As a second guard,
  every line `caliper-mcp` writes, to standard output or standard error, is scrubbed of every token it has read. The
  process tests check that no token ever appears in either stream.
- **Loopback only.** The client refuses every host but 127.0.0.1, and the API file is refused when its `url` names
  another host or port.
- **Read-only, confined folders.** The docs and SDK folders are only read, and only through the checked paths above.
  No URI can reach a file outside them, including through a symbolic link.
- **Checked arguments.** Every tool call is checked against its schema before it is sent, and the app checks it again.
  Nothing an agent sends is run as code; `caliper-mcp` starts no process and uses no shell.
- **Hostile input** on standard input and from the app is bounded: line length, nesting, the number of calls in
  flight and of subscriptions, the size of every response, and every time limit.
- **Approval** stays in the app. `caliper-mcp` cannot approve or deny, and the origin it sends must be its token's,
  which selects the policy that applies: it asks for a destructive command and, with review on, for an edit of the
  open spec from both `mcp` and `agent`, and it refuses `history.undo` and `history.redo` to both.

## Limits

| Limit | Value | Beyond it |
|---|---|---|
| One message on standard input | 8 MiB | -32600, the line is dropped |
| Nesting of a message | 128 levels | -32600 |
| Requests running at once | 32 | -32603 |
| Open subscriptions | 16 | -32603 |
| Arguments of a call | 1 MiB as JSON text | `command.invalid_args` |
| A command call | 130 seconds | `mcp.timeout` |
| Commands in the app's table | 1024 | the compiled-in table is used |
| Resource files per folder, depth, size | 5000, 16 levels, 4 MiB | not listed, not read |

## Tests

The tests are headless and run with the builder project's ctest.

- `builder_mcp` holds the unit tests: the codecs; the API file and its hostile forms; the tool derivation, which checks
  that every command of the table becomes one valid tool with a unique name; the result and error conversions; the
  resources and their confinement against `..`, encoded dots, hidden files and symbolic links that leave the folder;
  the protocol engine in both eras with every kind of malformed line; and the HTTP client against an in-process server
  that is slow, trickles, sends too much, sends garbage, a wrong `Content-Length`, a transfer coding or a redirect.
  It also starts `caliper-mcp` as a child process against an in-process fake of Caliper API that implements API v0
  and the two command endpoints, and drives whole sessions over its standard input and output: both eras, every v0
  tool, a command call, a call held for approval and then approved, denied or cancelled, a timeout, a 501 capture, an
  app without a command layer, a 401, the app restarting on another port with another token, the tool list changing,
  and the check that no token appears on standard output or standard error.
- `builder_mcp_acceptance` is the acceptance test of the MCP route, headless. It starts `caliper-builder-host`, the
  stand-in for the app's command endpoints ([Builder session](builder-session.md)), on a fresh workspace that holds a
  Parquet file and a safetensors file, runs `caliper-mcp` against the API file the host writes, and through MCP
  messages alone does what an outside agent would: it lists the tools (the host's table), reads a docs page, the
  command table and the spec schema as resources, imports the two files, runs their loaders while the host holds each
  call until the test allows it through the host's approval lines, reads the catalog, queries the table, builds the
  three-panel dashboard of the spec fixture `dashboard.spec.toml` command by command, saves the applet and reads the
  spec back. The saved spec on disk must equal the fixture. A second `caliper-mcp` with `--origin agent` then sends a
  destructive command, which the host announces with the origin `agent` and the test denies. The build's arguments
  come from the fixture through the session's conversions, so the test knows no command's schema. It runs whenever
  the session module is built, that is when DuckDB is given.
- `builder_mcp_app` is the end-to-end test against the real app, after contracts sections 14 and 17. When the CMake
  variable `CALIPER_BUILDER_TEST_APP` names a `caliper` executable, it starts it with
  `--hidden --api --api-port 0` and its own `CALIPER_DATA_DIR`, runs `caliper-mcp` against the API file it writes,
  calls `app_info`, `app_applets` and `capture_frame` (which must give a PNG that is not blank, or the documented
  `capture.unavailable`), calls a command tool and expects `mcp.no_command_layer`, and stops the app with SIGTERM.
  Without the variable, ctest reports it as skipped. To run it, configure the builder project with
  `-DCALIPER_BUILDER_TEST_APP=<path to a caliper executable>`; the app must be able to start hidden on that machine.

### Results on this branch

The end-to-end test was run on 15 November 2026 with a copy of the `caliper` executable built from `sprint2`, on
macOS with the Metal renderer. It passed: `app_info` gave the app's version and services, `app_applets` an empty list
(the copy had no applets beside it), `capture_frame` a PNG of 2560 by 1600 pixels that is not blank, and
`catalog_list` the tool error `mcp.no_command_layer`. The app ended with exit code 0 after SIGTERM and removed its API
file, and the token did not appear in the output of `caliper-mcp`.

As an extra check, the CLI of the MCP Inspector (`@modelcontextprotocol/inspector` 2.15.0 from npm, installed in a
scratch folder and run headless with its browser opening turned off) was pointed at `caliper-mcp` and the same hidden
app. It connected with the `initialize` handshake of 2025-11-25, listed the tools, called `app_info`,
`capture_frame` (an image block and the size), `catalog_list` and `app_launch` (each tool error as expected), and
listed and read resources and templates. Its `--strict` report on the tool schemas found no errors and 71 warnings,
all for `type` given as a list (such as `["string", "null"]`) in the command table's schemas, which is valid JSON
Schema but does not map onto a dialect with a single type per value, such as the OpenAPI subset of Gemini's function
declarations. `caliper-mcp` passes the schemas through unchanged; see [Open questions](#open-questions).

## Changes to the contracts

- **The origin `agent`.** Contracts section 8 names only the origins `api` and `mcp` for the command endpoints, but
  the [ACP integration](acp-integration.md) page says that commands from the app's own agent arrive with the origin
  `agent`, and that agent reaches the app through `caliper-mcp`. The smallest fix is the option `--origin agent`
  (the default is `mcp`), which sets `X-Caliper-Origin: agent`, so the side chat hands an ACP agent the same server
  with that origin. The stand-in `caliper-builder-host` accepts `agent` already, and the app's API server should
  accept it as well as `mcp` when the command endpoints are wired. The default approval policy treats both alike, and
  a request can never claim `ui` or `builder`; `caliper-mcp` refuses any other value of `--origin` at start, with
  exit code 2.
- **The MCP revision.** The contracts name none. `caliper-mcp` serves 2026-07-28 and, through `initialize`,
  2025-11-25 and 2025-06-18, since the agents in use speak one or the other.
- **The 501 of a capture** becomes the command layer's own code for `capture.frame`, `capture.unavailable`, so a
  client sees the same code from API v0 and from the command layer.

## Decisions

- **Dual-era.** Revision 2026-07-28 is current, but a server that only spoke it would fail with every client that
  still opens with `initialize`, and Claude Code itself falls back to the handshake. Serving both on one connection
  costs one branch per request.
- **`ping` in both eras.** It was removed in 2026-07-28, but answering it is harmless and older clients use it.
- **Tool errors for everything a model can act on**, including argument faults, an app that is not running and a
  missing command layer; JSON-RPC errors only for a malformed request, an unknown tool and an unknown method, as the
  specification divides them.
- **The error text is JSON** (`{"error": {"code", "message", "where"}}`), so an agent can match the code, and the
  message is a plain sentence for the person.
- **The four v0 tools always use their v0 endpoints**, so they work against every version of the app.
- **The app's table wins**, and a bad command in it is left out alone, so one fault does not take every tool away.
- **The output schema is checked before a result is returned**; a mismatch is a tool error that keeps the result.
- **`app.launch` is not read-only**, because it changes what the person sees and tears down the open applet.
- **The API file is read at every call.** It is small, and it makes a restarted app work at the next call with no
  cache to invalidate. The retry after a 401 or a refused connection covers the moment between two reads.
- **No half-close.** The client keeps its writing side open until the answer arrives, because a server may take the end
  of input for a client that has gone and drop a request that waits for the frame thread.
- **A missing docs or SDK folder stops the server at start**, so a misconfiguration shows in the agent at once.
- **The spec's JSON Schema is compiled in**, so that resource needs no file and cannot drift from the build.
- **No new dependency.** The HTTP client is a few hundred lines of POSIX sockets; a library would bring TLS,
  redirects and proxies that must all be turned off.

## Open questions

- Progress notifications while a call waits for approval, so a client with a short time limit could extend it; the
  specification lets a client reset its limit on progress, but Codex's `tool_timeout_sec` is a fixed limit.
- Grouping an agent's edits into one undo step: the endpoint takes `X-Caliper-Group`, but MCP has no notion of a
  turn that `caliper-mcp` could map to it.
- A Windows version of the HTTP client; on Windows the client answers `http.unsupported` for now.
- Whether to expose the stream to agents, which would need the token in the stream's address.
- Whether the command table should write nullable values with `anyOf` instead of a list of types, for clients that
  map tool schemas onto a single-type dialect (the Inspector's portability warnings above). It is a question for the
  table; `caliper-mcp` does not rewrite schemas, since the command layer checks arguments against them as they are.
