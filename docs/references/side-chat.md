# Side chat

The side chat is the pane where your own agent works with you over the Agent Client Protocol (ACP): Claude Code through
its adapter, Codex through its adapter, Gemini CLI, opencode, or any other ACP agent that you add by its command. This
page gives what Caliper starts, what it hands the agent, and what the pane shows. [Agent roles](../explanations/agent-roles.md)
explains when to use the side chat and when the widget agent.

!!! note "Status"
    The side chat is on the app's spec page and live page (`src/host/builder/chat_pane.h`, over the component in
    `src/host/builder/side_chat.h`), and is tested in the hidden app with stub agents. It is not on the landing page,
    and it does not take the builder pane's right column yet.

## Opening it

⌘L Chat in the page bar (Ctrl+L Chat elsewhere), to the left of ⌘K Command, opens and closes the side chat on the spec
page and on the live page; it is outlined in bronze while the chat is open. The chat is the right column of the window,
28 % of its width and at least 300 points, a minimum that grows with the text scale. The page bar, a stream's run bar,
the status bar, the jobs tray and the approval cards keep their places at the window's edges, and the page's panels or
views fill what the column leaves. The chat stays open when the page changes, so a spec that the agent opens from the
live page shows beside the same conversation.

The chat's settings are kept in `side-chat.json` in the app-data folder ([Folders and environment
variables](environment.md)): the chosen agent, the custom agent's name and command line, whether an agent's "always
allow" options are offered, and whether the chat is open. They never hold a key.

## The agents

Caliper detects four agents by looking for their commands on `PATH`. It runs nothing to find them.

| Agent | Launch command | Looked for on `PATH` |
|---|---|---|
| Claude Code | `npx -y @agentclientprotocol/claude-agent-acp` | `npx` and `claude` |
| Codex | `npx -y @agentclientprotocol/codex-acp` | `npx` and `codex` |
| Gemini CLI | `gemini --acp` | `gemini` |
| opencode | `opencode acp` | `opencode` |

A custom agent is a name and one command line. The line is split into words as a POSIX shell splits them, with single
quotes, double quotes and backslashes and nothing else (no variables, globbing or redirection), and the program is run
without a shell. The chosen agent and the custom command belong to the app's own settings, never to a workspace.

The agent uses its own login. When it answers that it needs one, the pane shows LOGIN with the login methods the agent
names and the command its maker documents; Caliper never asks for a key and runs no login flow. Log in in your own
terminal and choose Restart.

## What Caliper starts

The agent starts when you send your first message. Caliper starts its command with the workspace folder as the working
folder, and sends:

- `initialize` with protocol version 1, declining the file-system and terminal capabilities, so the agent uses its own
  file and shell tools under its own permission settings;
- `session/new` with the workspace as `cwd` and one MCP server, named `caliper`: this build's `caliper-mcp` by
  absolute path, with the arguments `--origin agent --api-file <the app's API file>`. `caliper-mcp` needs Caliper API:
  when the app was started without `--api`, the side chat starts it then, on 127.0.0.1 at a free port, says so in the
  conversation ("Caliper API started on 127.0.0.1:<port> for this chat"), in Activity and in the status bar, and stops
  it when the agent ends ([Started by the side chat](api.md#started-by-the-side-chat)); Commands the agent sends through it
  reach the command layer with the origin `agent`, so they are validated, journaled, undoable and reviewed;
- before the session's first message, a block for the assistant (`annotations.audience: ["assistant"]`) that asks the
  agent to change specs through the `caliper` tools and not to ask before calling one, since Caliper asks the person
  itself.

## The pane

The pane is the lower part of the right column, under Selection and Properties. Its header shows CHAT, the agent's
name, its version and its mode, the state as a tag, a FAULTS chip when the agent sent messages Caliper dropped, and
Restart when the agent must start again.

| Tag | State | Action |
|---|---|---|
| STARTING | The agent is starting and opening its session | |
| RUNNING | A turn runs | Cancel in the input row |
| STALLED | The agent has sent nothing for two minutes in a turn | Cancel, Restart |
| STOPPING | A cancel waits for the agent to end the turn | Restart |
| LOGIN | The agent needs a login | Restart |
| EXITED | The agent's process ended; the exit status and its last line of standard error | Restart |
| FAILED | The agent could not be started, such as one that speaks another protocol version | Restart |
| NO AGENT | No agent was found and no custom command is set | Custom command |

The conversation holds your messages, the agent's text, its thoughts as one line, its tool calls as rows (the
command's name for a Caliper tool, the title, and the status: pending, running, done, failed or stopped), its plan, and
the end of each turn in a word: DONE, STOPPED, DECLINED, LIMIT or FAILED.

**The agent's permission requests** are cards: the file, the change, and one button per option the agent offers.
For a `*.spec.toml` the change is the compact diff that the inline surface and the approval cards show (a new panel's
primitive and title first, then its other keys and every other change), with Diff for the full diff; for any other
file it is the unified diff. An agent's "always allow" option is offered only when the setting allows it.

**Command approvals.** A command of the agent that needs your approval, such as an edit of the open spec while review
is on, is decided on the approval cards at the bottom of the window. While it waits, the conversation shows one row
for it, APPROVAL with the command and "waiting"; a click on the row outlines its card in bronze for a moment. Once it
is decided, the row folds into its call's group.

**Groups.** Consecutive calls of the same command fold into one row with the count, the approvals' outcome and the
status, such as "source.add ×9 · approved · done"; a click opens it to show each call and its approval. A group with a
failed call or an approval that was denied, timed out or failed says so in bronze and is open from the start. A lone
call without an approval is a plain row.

**The input** is one line that grows to six with Shift+Enter or Command+Enter; Enter sends. The agent picker beside it
lists the agents by name, with NOT FOUND for those not on `PATH`, and Custom command.

## Ask in chat

Ask in chat in the widget agent's prompt on the spec page opens the side chat with a draft: your request in the input, and above it one
chip for each block of context that goes with it (the target's spec fragment, its sources, the panel's render, the
widget conversation, the preview's diff marked as not applied, and the target's ids). Nothing is sent until you send
it; Clear drops the context. The blocks are shaped for the agent when you send: as embedded resources when it accepts
them and as text otherwise. The context of a handoff in the app leaves the panel's render out.

## caliper-chat-shot

`caliper-chat-shot` is a development tool, not installed, that draws a dashboard with the spec runtime beside the side
chat on the offscreen canvas and writes a PNG at each state of a scripted conversation with a stub agent:

```sh
caliper-chat-shot --workspace <dir> --agent <stub agent> --out-dir <dir> [--size WxH] [--prefix NAME]
                  [--frames N] [--review on|off] [--mcp PATH] [--inputs DIR]
```

It builds the dashboard in an empty workspace from the files of `--inputs` (`run42_metrics.parquet` with the columns
`step`, `loss`, `accuracy` and `lr`, and `attn.safetensors`), or from small files it writes itself, starts Caliper API on 127.0.0.1 with the session's command layer and an
API file, and runs the agent as a custom command named "Stub agent". The states are `1-empty`, `2-turn`,
`3-permission` (with `--review on`, `3-approval`), `4-done`, `5-exited` and `6-no-agent`, each written as
`<prefix>-<state>-<W>x<H>.png`. It prints one JSON object with the files, the conversation's items, the open spec's
panels and whether the journal recorded the agent's `panel.create`.
