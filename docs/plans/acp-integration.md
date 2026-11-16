# ACP integration

Caliper will run the user's own coding agent (Claude Code, Codex or Gemini CLI) as a child process over the
[Agent Client Protocol](https://agentclientprotocol.com) (ACP), hand it Caliper's MCP server, and show its messages,
tool calls and permission requests in an agent pane. The agent uses its own login, and Caliper never handles an API
key. This page records what the protocol and the three agents offered on 15 November 2026, the design of the pane,
and what is built on this branch.

**Built on this branch:** the client library `caliper_builder_acp` (a subprocess helper, a JSON-RPC peer and
`AcpClient`), a stub agent for tests, the headless tool `caliper-acp-harness`, and their tests. **Planned:** the agent
pane itself, its wiring to the command layer, and everything listed under [what waits](#what-is-built-and-what-waits).
The earlier proposal is the "Agent integration" section of the [applet builder](applet-builder.md) page.

## The protocol on 15 November 2026

Every fact in this section comes from the protocol's documentation and schema, read on 15 November 2026:
[initialization](https://agentclientprotocol.com/protocol/initialization.md),
[session setup](https://agentclientprotocol.com/protocol/session-setup.md),
[prompt turn](https://agentclientprotocol.com/protocol/prompt-turn.md),
[tool calls](https://agentclientprotocol.com/protocol/tool-calls.md),
[file system](https://agentclientprotocol.com/protocol/file-system.md),
[terminals](https://agentclientprotocol.com/protocol/terminals.md),
[authentication](https://agentclientprotocol.com/protocol/authentication.md),
[transports](https://agentclientprotocol.com/protocol/transports.md),
[cancellation](https://agentclientprotocol.com/protocol/v1/cancellation),
[session modes](https://agentclientprotocol.com/protocol/session-modes.md),
[agent plan](https://agentclientprotocol.com/protocol/agent-plan.md),
[slash commands](https://agentclientprotocol.com/protocol/slash-commands.md),
the [v2 migration guide](https://agentclientprotocol.com/protocol/v2/migration.md), and the schema files
`schema/v1/schema.json`, `schema/v1/meta.json` and `schema/v2/meta.json` with their changelogs in the
[protocol repository](https://github.com/agentclientprotocol/agent-client-protocol).

### Versions

The protocol version is one integer that changes only for a breaking change. The client sends the latest version it
supports in `initialize`; the agent answers with the same number if it supports it, and otherwise with the latest
version it does support. A client that does not support the answer should close the connection and tell the user.

Version 1 is the stable protocol. Its schema is released about once a week: schema v1.32.0 and the Rust crate v1.18.0
came out on 14 November 2026, and thirteen schema releases came out between 17 September and 14 November. These releases
add optional methods and fields behind capabilities without changing the version number.

Version 2 is a draft. The migration guide says that "v2 as a whole is still labeled draft", and its schema was at
v2.0.0-alpha.16 on 14 November 2026. Version 2 redesigns the turn: `session/prompt` only acknowledges the message, and
progress and the stop reason arrive as `state_update` notifications. It also removes `tool_call` in favour of
`tool_call_update`, replaces modes with config options, renames `authenticate` to `auth/login`, changes the diff
content to a list of changes, removes the client's file-system and terminal methods altogether, and adds the error
-32022 for a version mismatch. The guide asks implementations to "treat v2 support as additive" and to keep serving
version 1 peers.

**Decision: Caliper speaks version 1 first.** All three agents answer version 1, and the Claude adapter serves the v2
draft only to a client that asks for it. Version 2 is added beside version 1 once it is stable, chosen per connection,
as the guide recommends.

### The methods a client needs

| Method | Direction | What it does | What Caliper does |
|---|---|---|---|
| `initialize` | request to the agent | Negotiates the version and exchanges capabilities and `clientInfo` / `agentInfo` | Offers version 1 and declines the optional client capabilities |
| `authenticate` | request to the agent | Runs an advertised login method of type `agent`; terminal-type methods are run by the client instead | Never sends a key; the pane may call it for an `agent`-type method |
| `logout` | request to the agent | Ends the login, when `auth.logout` is advertised | Not used |
| `session/new` | request to the agent | Creates a session with an absolute `cwd` and a list of `mcpServers`; may fail with -32000 (auth required) | Sends the workspace and `caliper-mcp` |
| `session/load` | request to the agent | Replays a saved session as updates, when `loadSession` is advertised | Not used yet |
| `session/resume`, `session/close`, `session/list`, `session/delete` | requests to the agent | Session management, each behind a session capability (listed as stable methods in schema v1.32.0) | Not used yet |
| `session/set_mode`, `session/set_config_option` | requests to the agent | Switch a mode or a config option; modes are to be replaced by config options | Not used yet |
| `session/prompt` | request to the agent | Runs one turn; the answer carries the stop reason | Used |
| `session/cancel` | notification to the agent | Cancels the running turn | Used |
| `session/update` | notification to the client | Streams messages, thoughts, tool calls, plans and more | Turned into typed events |
| `session/request_permission` | request to the client | Asks the user before a tool runs | Shown as a permission prompt |
| `fs/read_text_file`, `fs/write_text_file` | requests to the client | Read and write files through the client, behind `fs.readTextFile` and `fs.writeTextFile` | Declined |
| `terminal/create`, `terminal/output`, `terminal/wait_for_exit`, `terminal/kill`, `terminal/release` | requests to the client | Run commands through the client, behind `terminal` | Declined |
| `elicitation/create`, `elicitation/complete` | client side | Structured questions to the user, behind `elicitation` | Declined |
| `$/cancel_request` | notification, either way | Cancels one request; the receiver must still answer it | Honoured for permission requests |

A client declines a capability by leaving it false or absent in `initialize`: the file-system and terminal pages say
that the agent "MUST NOT attempt to call" a method whose capability is false. The protocol says nothing about what an
agent does instead. Caliper also answers a declined method with -32601 (method not found) if an agent calls it anyway.

The stop reasons of version 1 are `end_turn`, `max_tokens`, `max_turn_requests`, `refusal` and `cancelled`.

### Session updates

The `sessionUpdate` field names the kind. Schema v1.32.0 has fourteen kinds; the last three need client capabilities
that Caliper does not declare.

| Kind | Carries | Event in `AcpClient` |
|---|---|---|
| `user_message_chunk`, `agent_message_chunk`, `agent_thought_chunk` | One content block (text, image, audio, resource link or resource) | A chunk event with its text |
| `tool_call` | `toolCallId`, `title`, `kind`, `status`, `content`, `locations`, `rawInput`, `rawOutput` | `tool_call` |
| `tool_call_update` | The fields that changed | `tool_call_update` |
| `plan` | The whole plan as entries with `content`, `priority` and `status`; each update replaces the plan | `plan` |
| `available_commands_update` | Slash commands with a name, a description and an input hint | `available_commands` |
| `current_mode_update` | The new mode id | `current_mode` |
| `config_option_update`, `session_info_update`, `usage_update` | Config options, a title, context use and cost | `other_update`, kept raw |
| `notice`, `compaction_update`, `compaction_summary_chunk` | Behind `session.notices` and `session.compaction` | `other_update`, kept raw |

A kind that the client does not know is kept raw as `other_update`, so a newer agent does not break an older Caliper.

### MCP servers in session/new

All agents must support stdio servers, given as `name`, `command` (an absolute path), `args` and `env` (a list of
`{name, value}`). The documentation calls `env` optional, but schema v1.32.0 lists it as required, so Caliper always
sends it, empty when there is nothing to pass. HTTP servers (`"type": "http"` with `url` and `headers`) need the
agent's `mcpCapabilities.http`, and SSE servers need `mcpCapabilities.sse`; the documentation marks SSE as deprecated
by MCP. The session-setup page asks agents to start the command without a shell, and asks clients not to send a stdio
server taken from workspace files or agent output until the user has approved its command, arguments and variable
names.

### Tool calls, diffs and permission

A tool call's content is a list of items of type `content` (a content block), `diff` or `terminal`. A diff has the
absolute `path`, `oldText` (null for a new file) and `newText`. An agent that wants approval sends
`session/request_permission` with the tool call (as a `ToolCallUpdate`, which may carry the diff) and a list of options,
each with an `optionId`, a `name` and a `kind` of `allow_once`, `allow_always`, `reject_once` or `reject_always`. The
client answers `{"outcome": {"outcome": "selected", "optionId": ...}}`, or `{"outcome": {"outcome": "cancelled"}}`
when the turn is cancelled. So a client shows a proposed change by drawing the diff of the tool call in the request,
and the change is applied only after the user picks an allow option.

On `session/cancel` the client must answer every open permission request as cancelled, should mark unfinished tool
calls as cancelled, and should keep accepting updates. The agent must stop and answer the prompt with the `cancelled`
stop reason, and must catch the errors that aborting causes rather than answer with an error.

### Transport

On stdio each message is one line of UTF-8 JSON without embedded newlines. The agent writes only ACP messages to its
stdout and may log to stderr; the client writes only ACP messages to the agent's stdin. To shut down, the client closes
the agent's stdin and terminates the process. A streamable HTTP transport is being specified in a separate RFD.

### MCP over ACP

The [MCP-over-ACP RFD](https://agentclientprotocol.com/rfds/mcp-over-acp.md) would let a client serve its MCP tools over
the ACP connection itself, as an MCP server entry of `"type": "acp"` with a `serverId`, carried by one method,
`mcp/message`, and advertised by `mcpCapabilities.acp`. On 15 November 2026 it was a draft that targets MCP
2026-07-28 only, with a Rust reference implementation on unstable schema types. The page says the work does not yet
establish "readiness to publish packages", and no SDK release or agent ships it; the Codex adapter advertises
`acp: false`. **Decision:** Caliper hands its agent the stdio `caliper-mcp` and revisits the RFD when it is stable,
since it would remove the bridge process.

## The agents on 15 November 2026

The facts below come from each agent's repository, its package manifest and the npm registry, read on 15 November
2026. Where the documentation is silent, the behaviour was read from the source code, and this page says so.

| Agent | How it speaks ACP | Launch command | Latest release | Stable releases since 15 August | Licence |
|---|---|---|---|---|---|
| Claude Code | The adapter [claude-agent-acp](https://github.com/agentclientprotocol/claude-agent-acp) on the Claude Agent SDK | `npx -y @agentclientprotocol/claude-agent-acp` | 0.107.0, 13 November 2026 | 47 | Apache-2.0 in its package manifest |
| Codex | The adapter [codex-acp](https://github.com/agentclientprotocol/codex-acp), which starts the Codex App Server | `npx -y @agentclientprotocol/codex-acp` | 2.6.0, 8 November 2026 | 21 | Apache-2.0 |
| Gemini CLI | Natively, in [gemini-cli](https://github.com/google-gemini/gemini-cli) | `gemini --acp` | 0.68.0, 10 November 2026 | 13 | Apache-2.0 |

The protocol keeps an [agent registry](https://agentclientprotocol.com/get-started/registry.md) as JSON
(`https://cdn.agentclientprotocol.com/registry/v1/latest/registry.json`, 46 agents on 15 November 2026), with each
agent's version, licence and distribution (an `npx` package with arguments, or per-platform binaries). Its entries for
the three agents match the launch commands above, pinned to the versions in the table. Its entry for the Claude
adapter gives the licence as "proprietary", which disagrees with the adapter's own manifest; which one is meant could
not be confirmed.

### Claude Code

- **Launch.** Claude Code has no native ACP mode; the adapter runs it. The old package `@zed-industries/claude-code-acp`
  stopped at 0.16.2 in February 2026 and is marked on npm as renamed to `@agentclientprotocol/claude-agent-acp`. Its
  GitHub releases also offer standalone binaries that bundle the `claude` CLI, and `CLAUDE_CODE_EXECUTABLE` points it
  at the user's own `claude`.
- **Maturity.** Very fast-moving: 47 stable releases in three months, with a `preview` tag. Open issues that matter
  to a client include a prompt that can take the result of an earlier cycle after `session/load` or `session/resume`
  ([#1621](https://github.com/agentclientprotocol/claude-agent-acp/issues/1621)), a background task that never ends
  ([#1620](https://github.com/agentclientprotocol/claude-agent-acp/issues/1620)) and problems with leaving plan mode
  ([#1077](https://github.com/agentclientprotocol/claude-agent-acp/issues/1077)).
- **Version.** It answers version 1 and serves the v2 draft only to a client that asks for version 2;
  `CLAUDE_AGENT_ACP_PROTOCOL` can pin one (README). It depends on `@agentclientprotocol/sdk` 2.6.0.
- **MCP servers.** It advertises `http` and `sse` and maps stdio, HTTP and SSE entries (`src/mcp-servers.ts`). The
  session's servers are added to the user's own, and the session's server wins on a name clash (source reading).
- **Login.** Its login methods are all of type `terminal` (`src/auth-methods.ts`) and are advertised only to a client
  that declares `auth.terminal`; Caliper does not, so it receives an empty list, and the user logs in with `claude`
  in their own terminal. No key passes through the client.
- **Declined capabilities.** No code path gates its tools on the client's `fs` or `terminal` capabilities: the Claude
  Agent SDK uses Claude's own file and shell tools (source reading; the documentation is silent).
- **Diffs, permissions and modes.** Edits arrive as `diff` content (`src/tool-calls/reporters/file-edit.ts`).
  Permission options are `allow_once`, `allow_always` and `reject_once`. The modes are `default`, `acceptEdits`,
  `plan`, `auto` and `bypassPermissions`, and slash commands are advertised.

### Codex

- **Launch.** No native ACP mode was found in `openai/codex`; this was checked by searching its CLI source and could
  not be confirmed beyond that. The adapter is now TypeScript (it was a Rust crate under `@zed-industries/codex-acp`,
  which stopped at 0.16.0 in June 2026). It bundles `@openai/codex` and accepts `CODEX_PATH` to use the user's own.
  It ships only through npm; an issue asks for binaries
  ([#746](https://github.com/agentclientprotocol/codex-acp/issues/746)).
- **Maturity.** 21 stable releases in three months, with `preview` and `beta` tags. Open issues include
  `session/prompt` never resolving when a turn fails without completing
  ([#754](https://github.com/agentclientprotocol/codex-acp/issues/754),
  [#747](https://github.com/agentclientprotocol/codex-acp/issues/747)), subagents that cannot call the MCP servers of
  `session/new` ([#761](https://github.com/agentclientprotocol/codex-acp/issues/761)), and a starting mode that ignores
  the sandbox settings ([#702](https://github.com/agentclientprotocol/codex-acp/issues/702)).
- **Version.** It answers version 1 by default and has a separate v2 path. It depends on `@agentclientprotocol/sdk`
  ~2.4.0.
- **MCP servers.** It advertises `http: true`, `sse: false` and `acp: false`. An `sse` entry fails the whole
  `session/new` with invalid params (`src/McpServerConfig.ts`). A requested server whose name the user's Codex
  configuration already defines is dropped, so the user's server wins (source reading).
- **Login.** It advertises `chat-gpt`, `chat-gpt-device-code`, `api-key` and `gateway`. The ChatGPT login is Codex's
  own, and the `api-key` method reads `CODEX_API_KEY` or `OPENAI_API_KEY` from the agent's environment, so the client
  never passes a key.
- **Declined capabilities.** No gate on the client's capabilities was found; Codex runs its own sandboxed shell and
  file changes (source reading).
- **Diffs, permissions and modes.** File changes arrive as `diff` content. It handles all four option kinds; which it
  sends in each case could not be confirmed. The modes are `read-only`, `workspace-write`, `agent`,
  `agent-full-access` and permission profiles.

### Gemini CLI

- **Launch.** Gemini CLI speaks ACP itself with `--acp`; `--experimental-acp` remains as a deprecated alias
  (`packages/cli/src/config/config.ts`). The registry starts it as `npx @google/gemini-cli --acp`.
- **Maturity.** Still 0.x, with 13 stable releases in three months plus previews and nightlies. Open ACP issues
  include `session/close` answering method not found
  ([#30231](https://github.com/google-gemini/gemini-cli/issues/30231)), stderr held back until the process exits
  ([#30128](https://github.com/google-gemini/gemini-cli/issues/30128)) and tool calls that run one at a time
  ([#30235](https://github.com/google-gemini/gemini-cli/issues/30235)).
- **Version.** It answers version 1 and depends on `@agentclientprotocol/sdk` 0.16.1, far older than the adapters.
- **MCP servers.** It advertises `http` and `sse`. The session's servers are added to the user's settings and win on a
  name clash; an entry of an unknown kind is ignored (source reading).
- **Login.** It advertises `oauth-personal` (Google login), `gemini-api-key`, `vertex-ai` and `gateway`. Its
  `authenticate` also accepts a key in `_meta` (source reading); Caliper never uses that, and a user who has logged in needs nothing
  from the client.
- **Declined capabilities.** The code documents the fallback: `acpFileSystemService.ts` reads and writes the local file
  system when the client does not declare `readTextFile` or `writeTextFile`, and nothing uses client terminals.
- **Diffs, permissions and modes.** Edits arrive as `diff` content, with `allow_once`, `allow_always` and `reject_once`
  options. The modes are `default`, `autoEdit`, `yolo` and, when enabled, `plan`.

### What the findings mean for Caliper

All three agents answer version 1, accept stdio MCP servers, emit diffs for edits, ask permission in their default
modes, and log in on their own. None of them needs the client's file-system or terminal methods. The differences that
matter are the handling of a server name that the user has already configured, Codex's refusal of SSE entries, the
release pace of the Claude adapter, and the open Codex issue of a prompt that never resolves, which the client's
stall and cancel handling covers.

## The design of the agent pane

### ACP for both backends

ACP is the session model of the agent pane for both of its backends. The ACP backend runs the user's own agent; the
BYOK backend runs a Caliper-built ACP agent, a Python sidecar named `caliper-agent` that runs its own loop against the
Claude API with the user's key. The app drives both with the same `AcpClient`, so messages, tool calls, spec diffs,
permission requests and cancellation behave the same. The sidecar is described on its own plan page, "BYOK agent".
`AcpClient` assumes nothing about the agent it runs. The key reaches the sidecar in its environment at spawn
(`AcpClientOptions::agent_env`), never in its arguments, and no layer logs an environment value. The stub agent and
the harness are the sidecar's test bed: `caliper-acp-harness --env NAME` copies a variable from the harness's own
environment, so a key never appears on a command line.

### Processes and threads

```
frame thread ── poll() ──► AcpClient event queue ◄── reader thread ◄── agent stdout
     │                                                writer thread ──► agent stdin
     │ prompt(), cancel(), respond_permission()       stderr thread ◄── agent stderr (bounded log)
     ▼
agent pane                    agent ── starts ──► caliper-mcp ── Caliper API ──► command layer
```

- One `AcpClient` serves one pane session. It starts the agent with an argument vector and no shell, in a process group
  of its own, so stopping it also stops what it started (an `npx` launcher, for example).
- Three threads per client do the I/O: a reader that parses lines and answers requests, a writer that drains an
  outgoing queue, and a reader of the agent's stderr. The frame thread never blocks on a pipe. `start()` and
  `new_session()` wait for the agent's answer and run on a worker thread; every other call returns at once.
- Chat events reach the UI on the frame thread, through the client's own event queue, which the pane polls in each
  frame at the same point where the app calls the command queue's `drain()`. Everything that changes the workspace
  travels through the command queue: the agent calls `caliper-mcp`, which calls Caliper API, which submits commands
  with the origin `agent`.
- **Decision:** chat events do not go through the command queue themselves, although the applet builder page proposed
  that. They change no workspace state, so journaling them as commands would fill the journal with chat. The pane reads
  them on the same thread and at the same point, which is what the proposal needed.

### Turns, permission prompts and spec diffs

There are two kinds of approval, and the pane shows both as the same card: a title, a diff when there is one, and one
button per option.

1. **ACP permission requests** come from the agent's own tools, such as a file edit. The card shows the tool call's
   diff and the agent's options, and the pane answers with `respond_permission()`. An `allow_always` option is offered
   only when the user's settings allow it.
2. **Command approvals** come from the command layer (contracts section 7). While a pane session is open, the pane
   installs the approval policy for the origin `agent`: it asks for every destructive command, and, when the user has
   "review changes" on, for every command that changes the open spec (`spec.set_text`, `source.*`, `panel.*`,
   `layout.*`, `applet.save`). A command that needs approval gets an approval id; the pane shows a card and calls
   `approve(id)` or `deny(id)`.

A spec change is shown as a diff before it is applied. For a command, the command layer renders the canonical TOML of
the spec before and after the change, and the card shows the difference with the validator's result. For an agent's
own edit of a `*.spec.toml` file, the diff arrives in the permission request, and the pane runs the spec validator on
the new text before the user answers. After an allowed file edit, the app reloads the spec from disk, because the text
file is the source of truth.

Commands are the preferred path, since they are validated, journaled and undoable, and the session's opening prompt
asks the agent to change specs through the Caliper tools. A file edit by the agent's own tools bypasses the journal
and undo, so it stays possible but is never preferred.

**Considered and not chosen now:** declaring `fs.writeTextFile` for the workspace alone, so that every file write
became a command. Contracts section 10 declines the capability, and the Claude and Codex adapters write with their own
tools whatever the client declares.

### Cancellation

The Stop button calls `cancel()`. The client sends `session/cancel`, answers every open permission request as
cancelled (each card disappears through a `permission_withdrawn` event), and waits for the turn to end. An agent that
does not end the turn within 10 seconds gets a synthetic end with `acp.cancel_unanswered`, and the pane offers to stop
the agent. A cancelled turn that the agent answers with an error is shown as cancelled, as the protocol intends.

### Crashes and hangs

- **A crash** produces `turn_ended` with `acp.agent_exited` and then `agent_exited` with the exit status. The pane shows
  the status and the tail of the agent's stderr, and offers to restart, which starts a new process and a new session;
  `session/load` can restore history later for agents that support it.
- **A hang** is a turn with no message for two minutes, not counting the time a permission card waits for the user. The
  client reports it once as `stalled`, and the pane offers Stop and Stop the agent. There is no fixed time limit for a
  turn, since long turns are normal.
- **Bad messages** (malformed lines, oversized messages, unknown response ids, wrong nesting) are dropped and reported
  as `protocol_fault` events, which the pane lists under diagnostics, not in the chat.
- **Limits:** 8 MiB per message, nesting depth 128, 64 MiB of unsent output, 1024 calls in flight, 20,000 queued
  events (chunks are dropped first, and the drop is reported) and 64 KiB of stderr.
- **Shutdown** closes the agent's input, waits up to a second for it to exit, then sends SIGTERM to its process group
  and SIGKILL after the grace period.

### The working folder and the declined capabilities

The agent starts in the workspace folder, and `session/new` names the workspace as its absolute `cwd`. The agents
bound their file tools by this folder to different degrees, so the folder is a scope and not a sandbox.

Caliper declines `fs.readTextFile`, `fs.writeTextFile`, `terminal`, `auth.terminal` and `elicitation`, for these
reasons:

- Caliper is not an editor. It has no unsaved buffers to serve, and the files on disk are the truth.
- Serving terminals would make Caliper run arbitrary commands on the agent's behalf.
- All three agents fall back to their own file and shell tools, which ask permission in their default modes.
- Caliper does not run login flows. The user logs in with the agent's own command, once, in their own terminal.

### Configuring an agent

| Id | Agent | Command | Login, once, in the user's terminal |
|---|---|---|---|
| `claude` | Claude Code | `npx -y @agentclientprotocol/claude-agent-acp` | `claude`, then `/login` |
| `codex` | Codex | `npx -y @agentclientprotocol/codex-acp` | `codex login` |
| `gemini` | Gemini CLI | `gemini --acp` | `gemini`, then choose a sign-in method |

The same table is `known_agents()` in `caliper_builder/acp/agents.h`. A custom agent is one command line, split into
words as a POSIX shell would split them but run without a shell, plus optional variable names to pass. The agent
command and its variables live in the app's own settings, never in a workspace, because a workspace from elsewhere
must not be able to choose a program to run. The MCP server handed to the agent is always the `caliper-mcp` that
ships with the app, given by absolute path, never one named by workspace files.

**Decision:** the presets name the adapters without a version, so the user runs what the agents' makers ship, and the
pane shows the version from `agentInfo`. A custom command can pin a version.

## The manual check with a real agent

This check is for the user, with their own agent and login. It was not run on this branch, since running an agent
would use a subscription. Each step uses `caliper-acp-harness`, which opens no window and prints one JSON object per
line.

1. Build the builder project and note the path of `caliper-acp-harness` in its `acp` folder.
2. Log in to the agent in your own terminal as you normally do (see the table above).
3. Make an empty folder for the check, for example `~/acp-check`.
4. Run `caliper-acp-harness --preset claude --cwd ~/acp-check --prompt "Reply with the word ready." --timeout 300 > claude.jsonl`.
   The first run of an `npx` preset downloads the adapter.
5. Check that the command exits with 0, that `claude.jsonl` has an `initialized` line with `"protocol_version": 1`,
   a `session` line, `agent_message_chunk` lines, and a last `result` line with `"ok": true`.
6. Run it again with `--prompt "Create a file named note.txt that contains hello."` and the default `--permission deny`.
   Check that a `permission_request` line carries a `diff` for `note.txt`, that a `permission_answer` line follows, and
   that `note.txt` was not created.
7. Once `caliper-mcp` exists and the app is running, add `--mcp caliper=<path to caliper-mcp>` and the prompt
   "List the tools of the caliper MCP server." Check that a `tool_call` line names a Caliper tool.
8. Repeat steps 4 to 7 with `--preset codex` and `--preset gemini`.

Exit codes: 0 when the turn ended with a stop reason, 2 for bad usage, 3 when the agent could not be started,
initialised or given a session, 4 when the turn failed, 5 when the time limit passed and 130 when interrupted.

## What is built and what waits

Built on this branch, in `src/builder/acp/` and `src/builder/include/caliper_builder/acp/`:

- `subprocess.h`: `Process` (spawn from an argument vector with an environment and a working folder, three pipes, reads
  and writes with timeouts, no SIGPIPE, a process group, terminate with a grace period) and `run_process()`, which
  feeds and drains a child without deadlock. It is general, so the Python loader can use it too. Windows gets a stub
  that returns `process.unsupported`.
- `jsonrpc.h`: `JsonRpcPeer`, JSON-RPC 2.0 over newline-delimited JSON, with requests, timeouts, notifications,
  handlers that may answer later from any thread, and the limits listed above.
- `acp_client.h`: `AcpClient`, version 1 with the capabilities declined, typed events in a thread-safe queue, permission
  requests, cancellation, stall detection and shutdown.
- `agents.h`: the presets and the command-line splitter.
- `caliper-acp-harness` and, for tests only, `caliper_acp_stub_agent` (every script and misbehaviour of the client
  tests) and `caliper_acp_test_child`.
- Tests: the peer over in-process pipes, the subprocess helper, the client against every stub script, and the harness
  end to end (ctest `builder_acp` and `builder_acp_harness`).

Waiting for later work:

- The pane itself, its cards and diagnostics, after sprint 2 merges.
- The approval policy for the origin `agent`, the spec diff cards and the opening prompt, with the commands stream.
- `session/load`, `session/resume`, modes and config options in the pane, images in prompts, and version 2.
- The Windows implementation of `Process`.
- The environment of a GUI app: an app started from the Finder does not inherit the user's shell `PATH`, so `npx` or
  `gemini` may not be found. One answer is to read the login shell's environment once; Caliper will decide when it
  builds the pane.
- Reading the agent registry at runtime instead of the fixed presets.

## Decisions

- **Version 1 first,** for the reasons under [versions](#versions).
- **Declined capabilities,** as listed above, with their reasons.
- **Chat events through the client's queue,** read on the frame thread; workspace changes through commands.
- **Lenient reading.** An unknown update kind is kept raw. A malformed message is dropped and reported, and is not
  answered, because JSON-RPC would answer it with a null id, which v2 forbids. A request for an unknown method is
  answered with -32601.
- **Always send `env`** for a stdio MCP server, since the v1 schema requires it, and resolve a bare MCP command to an
  absolute path on `PATH`, since the protocol asks for one.
- **Permission requests left open when a turn ends** are answered as cancelled and withdrawn from the pane.
- **A cancelled turn that ends in an error** counts as cancelled.
- **Unpinned presets,** with the version shown in the pane.
- **Default harness policy `deny`,** so a check never changes files unless asked.
- **One process group per agent;** stdin is closed first, then SIGTERM, then SIGKILL.
- **A stall report, not a turn time limit.**
- **`--env NAME` copies a variable** from the harness's environment, so a secret never appears in argv.

## Risks

- **Pace of change.** The Claude adapter published 47 releases in three months. Its behaviour can change between two
  runs of `npx -y`, which also fetches code from npm at each start. Harness transcripts are the regression check.
- **Edits outside Caliper's control.** An agent in a mode that skips permission (`bypassPermissions`, `yolo`,
  `agent-full-access`) edits files without asking. The pane must show the current mode and warn about such modes.
- **The folder is not a sandbox.** Agents can read and write outside the workspace with their own tools.
- **Server name clashes.** Codex drops a requested server whose name the user's configuration already uses, so a user
  who has configured a server named `caliper` for Codex would get theirs instead of Caliper's.
- **Known adapter issues,** listed per agent above, such as a Codex prompt that never resolves.
- **Version 2,** which changes the turn and the diff shape; the client keeps version 1 until v2 is stable.

## Notes on the contracts

- Contracts section 7 says that `ask` returns `command.needs_approval` with an approval id, but not whether an agent's
  MCP call waits for the answer. For the pane, the agent's tool call should wait until the user answers in the pane,
  with a time limit, rather than fail at once. This is for the commands and MCP streams to settle.
- Contracts section 10 names "Claude Code through its ACP adapter, Codex, Gemini CLI". Codex also speaks ACP through an
  adapter; only Gemini CLI is native.
