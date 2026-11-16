# ACP integration

Caliper will run the person's own agent (Claude Code, Codex, Gemini CLI, opencode or any other agent that speaks the
protocol) as a child process over the [Agent Client Protocol](https://agentclientprotocol.com) (ACP), hand it
Caliper's MCP server, and show its messages, tool calls and permission requests in the side chat. The side chat is one
of the two roles of Caliper's agents: the general chat pane, for conversations and bigger jobs such as building from
scratch, loader scripts, project code and explanations. The other role, the widget agent, works inline on one target
and is designed on the [Widget agent](widget-agent.md) page. The side chat's agent uses its own login and runs on
whatever model it supports, local models included, and Caliper never handles its key. This page records what the
protocol and the agents offered on 15 November 2026, the design of the side chat, and what is built on this branch.

**Built on this branch:** the client library `caliper_builder_acp` (a subprocess helper, a JSON-RPC peer and
`AcpClient`), a stub agent for tests, the headless tool `caliper-acp-harness`, and their tests; and the side chat as a
host component (`src/host/builder/side_chat.h`) with its capture tool `caliper-chat-shot`, on the app's spec page and
live page (`src/host/builder/chat_pane.h`, [The side chat in the app](#the-side-chat-in-the-app)). **Planned:**
everything listed under [what waits](#what-is-built-and-what-waits).
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

**Decision: Caliper speaks version 1 first.** All four agents answer version 1, and the Claude adapter serves the v2
draft only to a client that asks for it. Version 2 is added beside version 1 once it is stable, chosen per connection,
as the guide recommends.

### The methods a client needs

| Method | Direction | What it does | What Caliper does |
|---|---|---|---|
| `initialize` | request to the agent | Negotiates the version and exchanges capabilities and `clientInfo` / `agentInfo` | Offers version 1 and declines the optional client capabilities |
| `authenticate` | request to the agent | Runs an advertised login method of type `agent`; terminal-type methods are run by the client instead | Never sends a key; the side chat may call it for an `agent`-type method |
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

`session/new` may also carry `_meta`, the protocol's extension point. `AcpClient::new_session()` takes an optional
`_meta` object and sends it unchanged, refused before sending when it is not an object, is larger than 64 KiB, nests
deeper than 32 levels or holds more than 16,384 values; the answer's `_meta` comes back in `SessionInfo::meta`. The
widget session puts its target, tool scope and preview there, under `caliper.widget`
([Widget agent](widget-agent.md#how-it-stays-an-acp-agent)); the side chat sends none.

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
2026; for opencode they come from its documentation and the protocol's agent registry. Where the documentation is
silent, the behaviour was read from the source code, and this page says so.

| Agent | How it speaks ACP | Launch command | Latest release | Stable releases since 15 August | Licence |
|---|---|---|---|---|---|
| Claude Code | The adapter [claude-agent-acp](https://github.com/agentclientprotocol/claude-agent-acp) on the Claude Agent SDK | `npx -y @agentclientprotocol/claude-agent-acp` | 0.107.0, 13 November 2026 | 47 | Apache-2.0 in its package manifest |
| Codex | The adapter [codex-acp](https://github.com/agentclientprotocol/codex-acp), which starts the Codex App Server | `npx -y @agentclientprotocol/codex-acp` | 2.6.0, 8 November 2026 | 21 | Apache-2.0 |
| Gemini CLI | Natively, in [gemini-cli](https://github.com/google-gemini/gemini-cli) | `gemini --acp` | 0.68.0, 10 November 2026 | 13 | Apache-2.0 |
| opencode | Natively, in [opencode](https://github.com/anomalyco/opencode) | `opencode acp` | 2.3.0 in the agent registry | Not checked | MIT in the agent registry |

The protocol keeps an [agent registry](https://agentclientprotocol.com/get-started/registry.md) as JSON
(`https://cdn.agentclientprotocol.com/registry/v1/latest/registry.json`, 46 agents on 15 November 2026), with each
agent's version, licence and distribution (an `npx` package with arguments, or per-platform binaries). Its entries for
the first three agents match the launch commands above, pinned to the versions in the table, and its entry for opencode
gives per-platform binaries, for macOS, Linux and Windows, started with the single argument `acp`. Its entry for the Claude
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

### opencode

Every fact here comes from opencode's [ACP page](https://opencode.ai/docs/cli/acp/), its
[permissions](https://opencode.ai/docs/permissions/) and [config](https://opencode.ai/docs/config/) pages and its
[migration guide from V1](https://opencode.ai/docs/migrate-v1/), which are the V2 documentation, read on 15 November
2026. opencode was not run.

- **Launch.** opencode speaks ACP itself with `opencode acp`, over stdio with protocol version 1. It starts a private
  opencode server for its own use and opens no network port for ACP. One process serves many sessions, and it exits
  when the client closes its input. `--print-logs` before `acp` sends its logs to stderr; stdout carries only ACP.
- **Maturity.** The documentation is that of opencode V2, and its install page says that some older install commands
  still install opencode 1. Its release pace was not checked.
- **MCP servers.** It advertises `http: true` and `sse: false`, and does not support MCP over ACP. The stdio and HTTP
  servers of `session/new` are added to the session's folder for the life of the process, so other sessions in that
  folder can use them. A server whose name the folder's configuration already uses is replaced by the client's, and
  opencode's own policy rules (`integration.use`) still apply.
- **Login.** It offers one login method, `opencode-login`, of type `agent` unless the client declares `auth.terminal`,
  which Caliper does not. That method collects no credentials; the person signs in once with `opencode auth login` in
  their own terminal, and a prompt that needs a login fails with ACP's authentication error. A local model through
  Ollama needs no account.
- **Declined capabilities.** The client capabilities it reads are `auth.terminal`, `elicitation.form`,
  `session.compaction` and `_meta["terminal-auth"]`; none concerns files or terminals, so it uses its own tools
  (documentation reading).
- **Sessions.** It uses the configuration of the session's folder, which for Caliper is the workspace: its models,
  agents, commands, skills, instructions, plugins and MCP servers. A session runs one prompt at a time.
- **Prompts.** It takes text, images, embedded resources and resource links. A text block whose `annotations.audience`
  is `["assistant"]` only reaches the model as context outside the person's message.
- **Diffs, permissions and modes.** Permission requests offer `once` (`allow_once`), `always` (`allow_always`) and
  `reject` (`reject_once`), with `diff` content when opencode can build a preview of a file change. Its config options
  are `model` (as `provider/model`), `effort` and `mode` (its primary agents). `session/cancel` ends the prompt with
  `cancelled` and rejects pending permission requests.
- **Its own tools.** opencode is a coding agent with shell, file and web tools of its own, configured by its
  `permissions` setting: an ordered list of rules with `action`, `resource` and `effect` (`allow`, `deny` or `ask`),
  where the last matching rule wins and no match means `ask`. The V2 names are `permissions`, `shell` and `subagent` in
  place of V1's `permission`, `bash` and `task`, and `write` and `patch` became `edit`; the V1 format is still
  accepted. In V2, `external_directory` is allowed by default, where V1 asked, so a person who wants to be asked before
  opencode touches files outside the workspace adds the rule
  `{ "action": "external_directory", "resource": "*", "effect": "ask" }`. Its page warns that `shell` runs with the
  user's own file, process and network authority.

### What the findings mean for Caliper

All four agents answer version 1, accept stdio MCP servers, emit diffs for edits, ask permission by default, and log
in on their own. None of them needs the client's file-system or terminal methods. The differences that matter are the
handling of a server name that the user has already configured, Codex's refusal of SSE entries, the release pace of
the Claude adapter, the open Codex issue of a prompt that never resolves, which the client's stall and cancel handling
covers, and opencode's reading of the workspace's own configuration, which [Risks](#risks) covers.

## The design of the side chat

### One client for both roles

ACP is the session model of both roles. The side chat runs the person's own agent. The widget agent runs a
Caliper-built ACP agent, the Python sidecar `caliper-agent`, which [Widget agent sidecar](byok-agent.md) describes.
The app drives both with the same `AcpClient`, so messages, tool calls, spec diffs, permission requests and
cancellation behave the same. `AcpClient` assumes nothing about the agent it runs. A key for the sidecar reaches it in
its environment at spawn (`AcpClientOptions::agent_env`), never in its arguments, and no layer logs an environment
value. The stub agent and the harness are the sidecar's test bed: `caliper-acp-harness --env NAME` copies a variable
from the harness's own environment, so a key never appears on a command line.

### Processes and threads

```
frame thread ── poll() ──► AcpClient event queue ◄── reader thread ◄── agent stdout
     │                                                writer thread ──► agent stdin
     │ prompt(), cancel(), respond_permission()       stderr thread ◄── agent stderr (bounded log)
     ▼
side chat                    agent ── starts ──► caliper-mcp ── Caliper API ──► command layer
```

- One `AcpClient` serves one side chat session. It starts the agent with an argument vector and no shell, in a process group
  of its own, so stopping it also stops what it started (an `npx` launcher, for example).
- Three threads per client do the I/O: a reader that parses lines and answers requests, a writer that drains an
  outgoing queue, and a reader of the agent's stderr. The frame thread never blocks on a pipe. `start()` and
  `new_session()` wait for the agent's answer and run on a worker thread; every other call returns at once.
- Chat events reach the UI on the frame thread, through the client's own event queue, which the side chat polls in each
  frame at the same point where the app calls the command queue's `drain()`. Everything that changes the workspace
  travels through the command queue: the agent calls `caliper-mcp`, which calls Caliper API, which submits commands
  with the origin `agent`.
- **Decision:** chat events do not go through the command queue themselves, although the applet builder page proposed
  that. They change no workspace state, so journaling them as commands would fill the journal with chat. The side chat reads
  them on the same thread and at the same point, which is what the proposal needed.

### Turns, permission prompts and spec diffs

There are two kinds of approval, and both show a change the same way: a title, the compact diff of
`src/host/builder/diff_card.h` when there is one, and one button per option. The agent's own requests are cards in the
side chat; command approvals are cards on the approval cards' strip, which the side chat points to (see
[One card for one change](#approvals-and-diffs-for-every-agent)).

1. **ACP permission requests** come from the agent's own tools, such as a file edit. The card shows the tool call's
   diff and the agent's options, and the side chat answers with `respond_permission()`. An `allow_always` option is offered
   only when the user's settings allow it.
2. **Command approvals** come from the command layer (contracts section 7). The app installs the approval policy for the
   origins `agent` and `mcp`, which covers both roles: it asks for every destructive command, and, when the user has
   "review changes" on, for every command that changes the open spec (`spec.set_text`, `source.*`, `panel.*`,
   `layout.*`, `applet.save`). A command that needs approval gets an approval id; the side chat shows a card and calls
   `approve(id)` or `deny(id)`.

A spec change is shown as a diff before it is applied. For a command, the command layer renders the canonical TOML of
the spec before and after the change, and the card shows the difference with the validator's result. For an agent's
own edit of a `*.spec.toml` file, the diff arrives in the permission request, and the side chat runs the spec validator on
the new text before the user answers. After an allowed file edit, the app reloads the spec from disk, because the text
file is the source of truth.

Commands are the preferred path, since they are validated, journaled and undoable, and the session's opening prompt
asks the agent to change specs through the Caliper tools. A file edit by the agent's own tools bypasses the journal
and undo, so it stays possible but is never preferred.

**Considered and not chosen now:** declaring `fs.writeTextFile` for the workspace alone, so that every file write
became a command. Contracts section 10 declines the capability, and the Claude and Codex adapters write with their own
tools whatever the client declares.

### Approvals and diffs for every agent

Approvals and diff previews are enforced in the command layer, so they hold for every agent, in either role, whether
or not the agent asks on its own. Three parts are built:

- **Approval of destructive commands.** The default policy asks for a destructive command from the origins `agent`,
  `mcp` and `api`; Caliper API holds a call from `agent` or `mcp` for up to 120 seconds while the person decides; no
  command and no endpoint approves, so an agent cannot approve itself; undo and redo ask about the destructive commands
  of the steps they touch; and a request can never claim the `ui` or `builder` origin
  ([Command layer](command-layer.md)).
- **The origin from the token.** Caliper API takes a command's origin from the token that authenticated it, never from
  `X-Caliper-Origin`, which it ignores (the viewer token aside). The person's session token (`api`) is printed on the
  start line and written to no file; `api.json` holds the token of outside agents (`mcp`); the side chat's agent and
  the widget agent's sidecar each get an `agent` token of their own in an API file of their own,
  `api-agent-<pid>-<n>.json` (mode 0600), which the app revokes and removes when that agent's session ends. An agent
  that reads a file and calls the API with curl therefore runs as `mcp` or `agent`, and its spec edits wait for review
  like those that go through `caliper-mcp` (Caliper API, `docs/references/api.md`, under Who sends a command). On the
  stream, the `mcp` and `agent` tokens watch, and can neither control a run nor publish a session: the hub decides
  per connection, by the token that opened it (`HubOptions::stream_access`), so its WELCOME sets `control` false, a
  CONTROL is answered `forbidden`, and a producer's HELLO is refused with close code 4002 as `caliper.serve()` refuses
  producers, with no change to the protocol.
- **Diff previews of spec edits.** Every edit of the open spec takes `dry_run` and returns its change as a diff of the
  canonical TOML without any effect ([Builder session](builder-session.md)).
- **Review of spec edits.** With review on for an origin, a session setting that is on by default for `agent` and
  `mcp`, an edit of the open spec that would change it does not run at once: the session computes the dry run itself,
  parks the invocation with its review (the unified diff of the canonical TOML before and after, and the validator's
  warnings), and the call is held or answered `command.needs_approval` as a destructive command's is. Approve applies
  it, deny answers `command.denied`, and an edit with an empty diff or a failing validation needs no approval. An edit
  approved after the open spec changed runs only when its change is still the one shown, and otherwise answers
  `history.conflict`. Each edit of a turn is reviewed on its own; a stacked review of several edits is a preview
  ([Command layer](command-layer.md#approval)).

**One card for one change.** `caliper-agent` used to run the dry run and send `session/request_permission` before each
spec edit and each destructive command, which was the agent being polite and, against an app that also holds the call,
a second prompt for the same change. It now reads `review.status` once per turn and asks through ACP only for what
Caliper does not ask itself: with the app reviewing its spec edits and holding its destructive commands, it sends no
permission request and no dry run for them, and the call waits in Caliper until the person decides there. Against an
app without `review.status`, it asks as before. The side chat follows the same rule: command approvals come from the
command layer and permission requests from the agent's own tools. As built, the two are not one kind of card in the
pane: the agent's own permission request is a card in the conversation, drawn with the compact diff that the inline
surface and the approval cards use, and a command approval is decided on the approval cards' strip, where the
conversation shows one row that points to it (the command and "waiting"), so that no change has two cards. An agent
that asks before a Caliper command it is about to send would make two cards for one change, so the side chat's opening
context asks it not to (`chat_opening_block()`).

The app's API server serves the command endpoints and holds a call from `agent` or `mcp` while the person decides,
and the app shows each pending approval as a card, so an outside agent's change over MCP waits in the app for the
person and lands when they approve it.

**The approval cards, as built** (`src/host/builder/approval_cards.h`). One card for each pending approval, whatever
the origin, says:

- who asks, the origin as a noun: MCP agent, Side chat (the `agent` origin), Widget agent (the `agent` origin with a
  stacked preview, which only the widget agent sends) or API;
- what, the command and its target: "panel.patch · Loss and accuracy", with a panel's title, a source's, loader's or
  applet's id, "layout", a file's name or a number of steps;
- for a spec edit, its review: the compact diff of the inline surface, one line per changed key with the old value
  struck through and the new one, folding into "+N" past four lines, a new panel's keys with its primitive and title
  first, the validator's warnings on a line of their own, and the full unified diff behind Diff, in a popup above
  the card;
- for a destructive command, what it loses, as a noun phrase: "Applet run42-dashboard overwritten", "Loader rows and
  its tables deleted", "Tables of loader rows replaced", "2 steps undone";
- the time left before the held call times out, from the queue (`CommandQueue::approval_time_left()`); an `api` call
  waits without a limit and shows none;
- Approve and Deny, which go through the session's queue, `approve(id)` and `deny(id)`, whose decision runs at the
  next drain on the frame thread. They have no keys, so that a keystroke meant for another pane never decides.

The cards sit in a strip at the window's bottom edge, above the jobs tray, on every page. It is a viewport side bar,
like the tray, so it takes its own space and covers no panel, and it waits as the tray does before it takes its space
(0.5 seconds) and before it gives it back (2 seconds), so a request decided at once moves nothing; while it is open,
its height changes only after the same delays. The cards stand side by side, oldest first, since the oldest times out
first, between 340 and 620 points wide each at text scale 1, with "Approvals" and the count of waiting requests at the
left; those that do not fit fold into "+N" at the right end, which lists them all with their own Approve and Deny.
The strip takes at most 40 % of the window's height; as it shrinks, the compact lines past the first fold, then the
warnings line goes. A decided card stays two seconds with its outcome: APPROVED, DENIED, TIMED OUT, CONFLICT (approved,
and the change no longer applied) or FAILED. On the spec page, a panel that a waiting edit changes carries a bronze
REVIEW tag after its title, with the time left, and a bronze rule under its strip, drawn by the page's overlay like a
preview's tag. Each request writes one Activity line when it starts waiting and one when it is decided, times out or
is withdrawn, with its origin ("mcp approval", "panel.patch · Loss and accuracy", "waiting"), beside the line the
journal gives the command itself.

The cards read the session's pending list on the frame thread at most every 100 ms and never while a long command
holds the registry, build a card once for each new approval (its compact diff from the edit made again on the open
spec, or from the review's unified diff when it no longer applies), and read the time left each frame. The side chat
will draw the same cards in its pane: the logic (`ApprovalCards`) and the layout (`layout_approval_strip()`, plain
rectangles tested at two window sizes with one, three and eight cards) do not depend on a page.

Decided while building them:

- **No Approve all for one agent's turn.** The command layer has no safe notion of a turn: a group is the agent's own
  label, which an agent could keep for ever, and a held call answers before the agent sends its next edit, so one turn
  never has two cards at once. A turn approved as a whole is a stacked preview, which the widget agent already uses.
- **The edit is not drawn in place on the spec page in this slice.** The runtime's `set_preview()` would show it, but
  only `panel.patch` has a panel to draw it in (a new panel's dashed frame is not built, and layout and source edits
  touch several), the inline surface owns the same seam on the same page, and the card's compact diff already names
  every changed key. The REVIEW mark points at the panel instead.
- **A strip at the bottom, above the jobs tray, not a column.** The tray is where the host shows what is waiting; a
  strip of cards side by side shows several requests at once without narrowing every panel, and the side chat's own
  column will show its cards in place.
- **A test decides through a file only a hidden app reads.** With `--hidden`, `CALIPER_TEST_APPROVALS` names a file of
  lines such as `approve panel.create mcp` or `deny applet.save`, applied to the oldest waiting card that matches
  through the same path as the buttons. The app ignores the variable when it is not hidden, and nothing over Caliper
  API or MCP decides, so an agent still cannot approve its own request.

What remains: the side chat's own cards for the permission requests of an agent's tools, beside these, and the review
setting per origin in its settings.

The side chat mirrors its agent's pending approvals as rows in the conversation that point to these cards, so that no change has two cards.

What the command layer cannot enforce is an agent's own file and shell tools, which act outside Caliper. The agent's
own permission settings govern them (Claude Code's and Codex's modes, Gemini CLI's modes, opencode's `permissions`
rules), and the side chat shows the agent's mode and warns about modes that skip permission, as [Risks](#risks) says.

### Cancellation

Cancel, which takes the place of Send in the input row while a turn runs, calls `cancel()`. The client sends `session/cancel`, answers every open permission request as
cancelled (each card disappears through a `permission_withdrawn` event), and waits for the turn to end. An agent that
does not end the turn within 10 seconds gets a synthetic end with `acp.cancel_unanswered`, and the side chat offers
Restart in its header, which stops the agent and starts a new one. A cancelled turn that the agent answers with an error is shown as cancelled, as the protocol intends.

### Crashes and hangs

- **A crash** produces `turn_ended` with `acp.agent_exited` and then `agent_exited` with the exit status. The side chat shows
  the status and the last line of the agent's stderr, elided to one line, and offers Restart in its header, which starts a new process and a new session;
  `session/load` can restore history later for agents that support it.
- **A hang** is a turn with no message for two minutes, not counting the time a permission card waits for the user. The
  client reports it once as `stalled`, and the side chat shows STALLED with Cancel in the input row and Restart in the
  header. There is no fixed time limit for a
  turn, since long turns are normal.
- **Bad messages** (malformed lines, oversized messages, unknown response ids, wrong nesting) are dropped and reported
  as `protocol_fault` events, which the side chat counts in a FAULTS chip in its header and lists behind it, not in the
  conversation.
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
- All four agents fall back to their own file and shell tools, which ask permission by default.
- Caliper does not run login flows. The user logs in with the agent's own command, once, in their own terminal.

### Configuring an agent

| Id | Agent | Command | Login, once, in the user's terminal |
|---|---|---|---|
| `claude` | Claude Code | `npx -y @agentclientprotocol/claude-agent-acp` | `claude`, then `/login` |
| `codex` | Codex | `npx -y @agentclientprotocol/codex-acp` | `codex login` |
| `gemini` | Gemini CLI | `gemini --acp` | `gemini`, then choose a sign-in method |
| `opencode` | opencode | `opencode acp` | `opencode auth login`; none for a local model |

These are the agents that Caliper detects, the four rows of `known_agents()` in `caliper_builder/acp/agents.h`. The side
chat finds one by looking for its commands on PATH, without running anything: Claude Code needs `npx` and `claude`,
Codex `npx` and `codex`, Gemini CLI `gemini` and opencode `opencode`. **Decision:** an adapter's preset also needs the
agent's own command, since `npx` alone says nothing about the agent being installed or logged in. Any other ACP agent is
added by its command in the settings, as a custom agent. A custom agent is one command line, split into
words as a POSIX shell would split them but run without a shell; the optional variable names to pass are not built,
since the agent inherits the app's environment. The agent
command and its variables live in the app's own settings, never in a workspace, because a workspace from elsewhere
must not be able to choose a program to run. The MCP server handed to the agent is always the `caliper-mcp` that
ships with the app, given by absolute path, never one named by workspace files.

**Decision:** the presets name the adapters without a version, so the user runs what the agents' makers ship, and the
side chat shows the version from `agentInfo`. A custom command can pin a version.

### Local models in the side chat

The side chat runs any ACP agent on whatever model that agent supports, local models included, and Caliper adds no
code for it: `AcpClient` starts the agent's command as it starts any other. Whether the agent's model runs on a hosted
API or on the person's own machine is the agent's configuration, not Caliper's.

The worked example is opencode on Ollama, with the facts read on 15 November 2026 from opencode's
[providers](https://opencode.ai/docs/providers/) and [models](https://opencode.ai/docs/models/) pages and from
Ollama's [opencode](https://docs.ollama.com/integrations/opencode), [context length](https://docs.ollama.com/context-length)
and [FAQ](https://docs.ollama.com/faq) pages:

1. Install Ollama and pull a model that supports tool calling, for example `ollama pull qwen3:8b`, the model that
   opencode's providers page uses. Ollama binds 127.0.0.1, port 11434, by default.
2. Give the model a context of at least 64,000 tokens. Ollama's opencode page says that opencode needs 64k or more.
   Ollama's context length page gives a default that depends on the video memory (4k below 24 GiB, 32k from 24 to
   48 GiB and 256k from 48 GiB), while its FAQ still says 4,096; set it with the slider in the Ollama app or with
   `OLLAMA_CONTEXT_LENGTH=64000 ollama serve`, and check it in the `CONTEXT` column of `ollama ps`.
3. Install opencode as its [install page](https://opencode.ai/docs/) describes.
4. Choose the model. opencode probes `http://127.0.0.1:11434`, adds the models whose Ollama metadata includes the
   `completion` capability without an account, and reads their context, vision and tool-use capabilities from Ollama.
   Select one with `"model": "ollama/<model>"` in the person's global `~/.config/opencode/opencode.jsonc` rather than in
   the workspace, for the reason under [Risks](#risks). For an Ollama server elsewhere, opencode takes its
   OpenAI-compatible URL, with `/v1`, in `providers.ollama.settings.baseURL`.
5. In Caliper, choose the `opencode` preset (or add `opencode acp` as a custom agent), and mark the agent
   as running on a local model with Local model in the agent picker, so that the side chat shows a note while a
   training run is live ([Widget agent](widget-agent.md#gpu-contention) gives the reasons and the advice).
6. Review opencode's `permissions`, and add the `external_directory` rule above to be asked before it leaves the
   workspace.

What is verified and what is not:

- **Verified from the documentation:** every command, address, setting and default above, as the pages gave them on
  15 November 2026.
- **Where the sources disagree:** Ollama's opencode page configures opencode in the V1 shape (a `provider.ollama` entry
  with `npm`, `options.baseURL` and `models`, in `opencode.json`) and installs it on Windows with
  `npm install -g opencode-ai`, while opencode's V2 pages find a local Ollama without configuration and install from
  the `@opencode/cli` package. V2 still accepts the V1 configuration. Ollama's FAQ and its context length page give
  different default context lengths.
- **Not verified:** nothing was run. Whether opencode passes the `env` of a stdio MCP server as Caliper sends it,
  whether a local model of a given size uses `caliper-mcp`'s 35 tools well (the conformance suite measures the widget
  agent's tasks, not the side chat's), and the whole path on Windows, where Ollama runs as a native application and the
  agent registry lists opencode binaries.

### The handoff from the widget agent

When the person chooses Ask in chat in the widget agent's inline prompt, the side chat receives a `session/prompt` in
its current session, which the app starts first when the side chat is not open. Seen from the side chat, the prompt
holds:

- the person's request, as a text block;
- the widget agent's context bundle: the target's spec fragment, its sources' schema and sample, and the selection, as
  `resource` blocks with embedded text when the agent declared `promptCapabilities.embeddedContext`, and as text
  blocks otherwise, and the render of the panel as an `image` block only when it declared `promptCapabilities.image`;
- `annotations.audience: ["assistant"]` on every block of the bundle, which opencode documents as context for the model
  outside the person's message; the other three agents were not checked, so they may read it as part of the message;
- the widget conversation so far and the diff of its open preview, as text marked as not applied;
- the target's ids, so the agent can act on them through `caliper-mcp`.

Nothing of the preview was applied, and the widget agent's preview is dropped when the handoff is sent. The side
chat's agent makes the change through commands, under the approvals above, and the side chat shows the whole prompt
as the person's message. `AcpClient::prompt_blocks()` sends such blocks today, and `AgentInfo` holds the agent's
prompt capabilities from `initialize`.

As built, the handoff is not sent by itself: `chat_handoff_draft()` makes a draft of the request and these blocks, and
`SideChat::open_handoff()` puts it into the input, with one chip per block (context, spec, source, render,
conversation, preview diff, ids), for the person to send or clear. The blocks are shaped for the agent when the message
is sent, after the agent has started and declared its capabilities, and the request goes last, since the bundle's first
block says that only the request says what to do. **Decision:** a draft rather than a sent prompt, so that the person
sees and may edit what goes to their own agent.

### The side chat as built

`src/host/builder/side_chat.h` holds the pane as a component in the pattern of the inline surface: a state machine
(`SideChat`), the layout as plain rectangles (`layout_chat_column()` and `layout_chat_pane()`), and its ImGui side
(`draw()`). The app owns it through `ChatPane` ([below](#the-side-chat-in-the-app)), which calls `update()` each frame,
claims the right column with `layout_chat_column()`, and calls `draw()` with that rectangle.

- **States.** No agent, idle, starting, ready, running (with a stalled flag), cancelling, needs login, exited and
  failed. The header shows the state as a tag (STARTING, RUNNING, STALLED, STOPPING, LOGIN, EXITED, FAILED, NO AGENT),
  the agent's name, its version from `agentInfo` and its mode, the mode in bronze when it skips permission
  (`bypassPermissions`, `yolo`, `agent-full-access`).
- **Starting.** **Decision:** the agent starts when the person sends the first message, not when the pane opens, so
  the pane opens quietly; the message waits for the session. `start()` and `session/new` run on a worker thread, and
  shutting a client down runs on another, so the frame thread never waits for an agent.
- **The conversation.** The person's messages with the chips of a draft; the agent's text, streamed into one message;
  its thoughts as one line, which a click opens; its tool calls as rows with a status mark, the command's name for a
  Caliper tool (recognised from the names the agents give MCP tools, such as `mcp__caliper__panel_create`), the title
  and the status in a word; its plan as rows; its permission requests as cards with the file, the compact diff for a
  `*.spec.toml` (a new panel's primitive and title first, then its other keys and every other change) or the unified
  diff for any other file, Diff for the full diff, and one button per option, `allow_always` only when the setting
  allows it; a row for each command approval of the turn while it waits; and the stop reason in a word (DONE,
  STOPPED, DECLINED, LIMIT, FAILED). Consecutive calls of one command, with their decided approvals, fold into one
  row that says the count, the approvals' outcome and the status ("source.add ×9 · approved · done") and opens on a
  click; a group with a failure or a refused approval is open from the start (`chat_rows()`). **Decision:** folded
  after a review of the live capture, in which about thirty rows of one request hid the person's message and the plan.
- **Login.** When `session/new` answers that the agent needs a login, the pane shows LOGIN with the agent's own login
  methods and the commands of the preset's hint, and Restart; it never asks for a key and runs no login flow.
- **No agent.** The four agents by name, each marked NOT FOUND, and one action, Custom command.
- **The input.** One line, more with Shift or Command and Enter, up to six; Enter sends; the agent picker beside it
  lists the agents by name, NOT FOUND for those not on PATH, Custom command, and Local model, a mark for the chosen
  agent that says its model runs on this machine.
- **The local model setting.** Local model in the picker marks the chosen agent as running on a local model, or clears
  the mark; it is the person's word, since Caliper cannot tell where an agent's model runs. While the app's live
  session is attached and the chosen agent is marked, the header shows the note "Shared GPU" after the agent's name,
  framed in bronze as in the widget agent's settings (`chat_gpu_note()`); it gives way after the version and before the
  name as the header narrows. **Decision:** the mark is kept per agent, by its id, so that switching from opencode on
  Ollama to a hosted agent drops the note; the custom agent's mark stays with the id `custom` when its command changes.
  **Decision:** the row is in the picker, beside the choice of the agent it marks, since the chat has no settings
  panel of its own.
- **The layout.** The right column is 28 % of the window's width with a minimum of 300 points that grows with the text
  scale; with Selection or Properties open above it, the chat takes 55 % of the column's height with a minimum of 220
  points, and the upper part gives way first. In the pane, the mode, then the version, then the agent's name in the
  header, then the picker's name, then the buttons' keys collapse as it narrows, and the FAULTS chip and then the state's
  tag give way to the header's buttons.
- **The opening context.** Before a session's first message the chat sends `chat_opening_block()`, a block for the
  assistant that asks the agent to change specs through the caliper tools and not to ask before a Caliper command.
- **The environment.** **Decision:** detection and the launch use a PATH that the app gives
  (`ChatEnvironment::path`), the app's own when it gives none; reading the login shell's environment for an app
  started from the Finder is not built.
- **Captures.** `caliper-chat-shot` draws a dashboard with the spec runtime beside the pane on the offscreen canvas,
  runs a session, Caliper API's server with the session's command gateway and an API file, and drives the chat with
  `caliper_chat_stub_agent`, a test agent named "Stub agent" that calls Caliper through the `caliper-mcp` it is handed.

### The side chat in the app

`src/host/builder/chat_pane.h` is the app's part: one `ChatPane`, which the app owns and gives to the spec page.

- **Where.** On the spec page and the live page. **Decision:** not on the landing page, whose command bar and
  Inspector column are its own and which shows no workspace yet; the live page has it because the acceptance case
  starts there, with a stream attached and nothing built. The chat stays open across the two, so a spec that the
  agent opens from the live page shows beside the same conversation.
- **The toggle.** ⌘L Chat (Ctrl+L elsewhere) in the page bar, to the left of ⌘K Command at the right end, outlined in
  bronze while open; the key works on the two pages. It leaves the place after the title to the builder pane's items
  (Build, Text, Undo, Redo, Save), whose room ends before ⌘L Chat.
- **The column.** A viewport side bar at the right, claimed after the page bar, a stream's run bar, the status bar, the
  jobs tray and the approval cards' strip, so those strips keep their places at the window's edges and every page fills
  the work area that is left (`layout_chat_page()` gives the rectangles, with the builder's left column as a
  parameter). In build mode the builder pane's right column holds Properties above the chat; the page then draws the
  chat with `ChatPane::draw_in()` in the pane's `chat_rect` instead of claiming a column.
- **The settings.** `side-chat.json` in the app-data folder: the chosen agent, the custom agent's name and command
  line, whether "always allow" options are offered, the agents marked as running on a local model
  (`local_model_agents`, a list of ids), and whether the chat is open. Never a key, and no address.
- **Caliper API.** `caliper-mcp` needs the API. **Decision, the user's:** when the app was started without `--api`, the
  chat starts the API when its agent starts, the one case in which the API starts without `--api`: on 127.0.0.1 at a
  free port, with new tokens and the API file, as `--api` does, and no setting can change the address. Each agent
  session gets an API file of its own with an `agent` token (above, under approvals), so the agent is given neither
  `api.json` nor the person's token. It is
  visible in three places: a line in the conversation ("Caliper API started on 127.0.0.1:<port> for this chat"), an
  Activity line with the address and never the token, and every page's status bar, which shows the API's address
  whenever it runs. It is tied to the chat: when the chat's agent ends (it exits, cannot start, or another agent is
  chosen), the app stops the API that the chat started and removes the API file; a restart replaces the agent without
  stopping it. An API started with `--api` is left alone in both directions.
- **Approvals.** The chat reads the session's pending approvals each frame while no long command holds the registry,
  and a click on an approval's row outlines its card on the strip in bronze for two seconds (`ApprovalTray::point_to()`).
- **Ask in chat.** The inline surface's Ask in chat opens the chat with the handoff draft. The bundle is built in the
  frame, so it leaves the panel's render out.
- **Tests.** ctest `caliper_chat_pane`: the settings, the toggle and ⌘L, the column the canvas leaves, Ask in chat, the
  approval row, the API line and the end hook over a real `AppSession` and a headless ImGui context; ctest
  `caliper_side_chat_app`: the hidden app's spec page (a turn whose `panel.create` waits on the cards and is approved
  through the test-only decisions file, Ask in chat, the API started on loopback whatever the settings file says and
  stopped when the agent ends) and the acceptance case on `caliper_attention_producer`'s stream with a scripted agent
  (the app started with `--api`, which the chat leaves alone). The hidden app's input driver
  (`CALIPER_TEST_SPEC_UI`) gained steps for the chat.

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
8. Repeat steps 4 to 7 with `--preset codex` and `--preset gemini`, and with `--agent "opencode acp"` for opencode,
   once on a hosted model and once on a local model through Ollama.

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
- `caliper-acp-harness`, with `--meta` for the `_meta` of `session/new`, and, for tests only,
  `caliper_acp_stub_agent` (every script and misbehaviour of the client tests; it records the `_meta` of
  `session/new`, reports it and echoes it in its answer) and `caliper_acp_test_child`.
- The optional `_meta` of `AcpClient::new_session()`, bounded and passed through unchanged.
- In the command layer, the review of spec edits for the origins `agent` and `mcp`, with the diff on each pending
  approval, and `review.status`, which `caliper-agent` reads so that it does not ask twice.
- In the app, the approval cards of every pending approval, with their tests: the logic over a real session with
  review on, the layout, and an outside agent's `panel.create` through the real `caliper-mcp` held in the hidden app,
  approved and landed (ctest `caliper_approval_cards` and `caliper_approval_cards_app`). See
  [Approvals and diffs for every agent](#approvals-and-diffs-for-every-agent).
- Tests: the peer over in-process pipes, the subprocess helper, the client against every stub script, and the harness
  end to end (ctest `builder_acp` and `builder_acp_harness`).

In `src/host/builder/`: the side chat (`side_chat.h`) as [built](#the-side-chat-as-built), with `opencode` in
`known_agents()`, `caliper-chat-shot`, `caliper_chat_stub_agent` and their tests (ctest `caliper_side_chat`, headless,
and `caliper_side_chat_gpu`, the capture scenario as a process).

Waiting for later work:

- The side chat on the landing page, and a Selection tab above the chat on the live page. In the builder pane's
  right column it is built: under Properties, drawn with `ChatPane::draw_in()`.
- Open items of the side chat's Caliper API, found in review and left for a later round:
  - its life cycle at the edges: a launch that fails after `retire(ending=false)` leaves the API that the chat started
    running; the ⌘K widget agent's sidecar loses the API when the chat stops the API it started, since the sidecar's
    token goes with the server; and what closing the pane does to the agent and to the API it started;
  - a second app on the same app-data folder overwrites `api.json`, and the first app's exit then removes the second
    app's file;
  - what Accept on the inline surface expects of `preview.commit` when the preview's edits came from another origin;
  - timing in tests: a 100 ms frame in `tests/test_app_session.cpp` (about line 327), waits in
    `src/builder/mcp/tests/test_process.cpp` (about lines 370 and 387), and the crash-report folder that
    `src/host/builder/tests/test_widget_app.cpp` reads;
  - revoking an agent's token refuses its next requests and upgrades, but a stream connection that the token opened
    before stays open until it closes.
- The review setting per origin in the side chat's settings, and running the spec validator on an agent's own edit of a
  spec file before the person answers.
- The warning when a workspace holds an opencode configuration.
- The packaging of `caliper-mcp` beside the app's executable; a development build finds it in its build tree.
- `session/load`, `session/resume`, modes and config options in the side chat, images in prompts, and version 2.
- The Windows implementation of `Process`.
- The environment of a GUI app: an app started from the Finder does not inherit the user's shell `PATH`, so `npx` or
  `gemini` may not be found. The side chat takes the PATH the app gives it; reading the login shell's environment once
  is one answer, not built.
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
- **Permission requests left open when a turn ends** are answered as cancelled and withdrawn from the side chat.
- **A cancelled turn that ends in an error** counts as cancelled.
- **Unpinned presets,** with the version shown in the side chat.
- **Default harness policy `deny`,** so a check never changes files unless asked.
- **One process group per agent;** stdin is closed first, then SIGTERM, then SIGKILL.
- **A stall report, not a turn time limit.**
- **`--env NAME` copies a variable** from the harness's environment, so a secret never appears in argv.
- **Any ACP agent on any model.** The side chat runs whatever agent the person chooses, on whatever model it supports,
  local models included, with no Caliper code for any of them; opencode on Ollama is the worked example.
- **Four detected agents and a custom command.** Claude Code, Codex, Gemini CLI and opencode have presets; any other
  agent is added by its command.
- **Detection by the agent's own command,** without running anything: an adapter's preset needs `npx` and the agent.
- **The agent starts with the first message,** so the pane opens quietly.
- **The handoff is a draft** the person sends, never a prompt sent by itself.
- **One card for one change in the pane:** the agent's own permission requests are cards in the conversation, command
  approvals are decided on the approval cards' strip and shown as one row each.
- **Restart in one place,** the header, for a stalled, exited, failed or logged-out agent and for a cancel that was
  never answered.
- **The chat on the spec page and the live page,** with ⌘L Chat beside ⌘K Command and the column as a viewport side
  bar after the host's strips.
- **The chat starts Caliper API when the app runs without it,** on loopback, visibly, and stops it when its agent ends.
- **The origin is the token's.** Each token has a fixed origin, `X-Caliper-Origin` is ignored rather than refused, so a
  client written before the change keeps working with the origin of its token, and only the person's token, printed on
  the start line, runs commands unreviewed.
- **A command script only where no one watches.** `CALIPER_COMMAND_SCRIPT`, which may claim the `ui` origin and
  approve, runs only in the hidden app and in `caliper shot`, as `CALIPER_TEST_APPROVALS` does; a visible app refuses
  it whole and says so on standard error and in Activity.
- **Review of spec edits in the command layer,** with the diff attached to the pending approval, so that no agent can
  skip it.
- **One kind of approval card for every agent,** in a strip above the jobs tray on every page, with the REVIEW mark on
  the panel an edit changes; no Approve all for a turn ([Approvals and diffs for every
  agent](#approvals-and-diffs-for-every-agent)).
- **No second prompt for one change.** An agent that knows Caliper reviews a call, as `caliper-agent` learns from
  `review.status`, does not ask the person again; the command layer's card is the one that counts.
- **`_meta` passed through, not interpreted.** The client bounds it and sends it as given, so the widget session's
  fields can change with the sidecar without a change to the client.
- **A warning, not a refusal, when a workspace configures opencode,** since the configuration may be the person's own.
- **A local-model mark per agent, set in the picker,** because the app cannot tell where an agent's model runs; with a
  live run attached, the marked agent's header shows Shared GPU.

## Risks

- **Pace of change.** The Claude adapter published 47 releases in three months. Its behaviour can change between two
  runs of `npx -y`, which also fetches code from npm at each start. Harness transcripts are the regression check.
- **Edits outside Caliper's control.** An agent in a mode that skips permission (`bypassPermissions`, `yolo`,
  `agent-full-access`) edits files without asking. The side chat must show the current mode and warn about such modes.
- **The folder is not a sandbox.** Agents can read and write outside the workspace with their own tools.
- **Server name clashes.** Codex drops a requested server whose name the user's configuration already uses, so a user
  who has configured a server named `caliper` for Codex would get theirs instead of Caliper's.
- **Known adapter issues,** listed per agent above, such as a Codex prompt that never resolves.
- **A workspace configures opencode.** opencode reads the configuration of the session's folder, which is the
  workspace: models, agents, commands, skills, instructions, plugins and MCP servers. A workspace from elsewhere could
  therefore choose what the agent runs, which Caliper's own settings forbid for Caliper itself. The person keeps their
  opencode settings in the global configuration, and the side chat warns when the workspace holds an `opencode.json`,
  an `opencode.jsonc` or an `.opencode` folder.
- **Local models vary.** A small local model may misuse tools or lose track of a long session; that is the agent's and
  the model's matter, and the side chat shows what happens as it does for any agent.
- **Version 2,** which changes the turn and the diff shape; the client keeps version 1 until v2 is stable.

## Notes on the contracts

- Contracts section 7 says that `ask` returns `command.needs_approval` with an approval id, but not whether an agent's
  MCP call waits for the answer. For the side chat, the agent's tool call should wait until the user answers in the side chat,
  with a time limit, rather than fail at once. This is for the commands and MCP streams to settle.
- Contracts section 10 names "Claude Code through its ACP adapter, Codex, Gemini CLI". Codex also speaks ACP through an
  adapter; only Gemini CLI is native.
