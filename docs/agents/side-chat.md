# Side Chat

The side chat is the pane where your own agent works with you over the Agent Client Protocol (ACP): Claude Code through
its adapter, Codex through its adapter, Gemini CLI, opencode, or any other ACP agent that you add by its command. This
page gives what Caliper starts, what it hands the agent, and what the pane shows. [Agent roles](agent-roles.md)
explains when to use the side chat and when the widget agent.

!!! note "Status"
    The side chat is on the app's spec page and live page (`src/host/builder/chat_pane.h`, over the component in
    `src/host/builder/side_chat.h`), and is tested in the hidden app with stub agents, and with `caliper-agent` on a local
    model served by Ollama in the dry run `examples/build_a_dashboard/dry_run/step4_side_chat.py`. It is not on the
    landing page.
    In the spec page's build mode it is the lower part of the builder pane's right column, under Properties
    ([Builder pane](../builder/builder-pane.md)).

## Opening it

⌘L Chat in the page bar (Ctrl+L Chat elsewhere), to the left of ⌘K Command, opens and closes the side chat on the spec
page and on the live page; it is outlined in bronze while the chat is open. The chat is the right column of the window,
28 % of its width and at least 300 points, a minimum that grows with the text scale. The page bar, a stream's run bar,
the status bar, the jobs tray and the approval cards keep their places at the window's edges, and the page's panels or
views fill what the column leaves. The chat stays open when the page changes, so a spec that the agent makes or opens
from the live page shows beside the same conversation: the app shows the spec page for each `spec.new`, `spec.open` and
`applet.open` of the agent, so that you see the spec whose edits you review, except over a compiled applet's page,
which an agent never ends: there the spec opens in the session, the page stays, and Activity says how to get to it
([Caliper CLI](../reference/cli.md#the-spec-page)).

The chat's settings are kept in `side-chat.json` in the app-data folder ([Folders and environment
variables](../reference/folders-and-environment.md)): the chosen agent, the custom agent's name and command line, whether an agent's "always
allow" options are offered, the agents you marked as running on a local model (`local_model_agents`, a list of agent
ids), and whether the chat is open. They never hold a key.

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

`caliper-agent`, the agent behind ⌘K ([Widget agent](widget-agent.md#in-the-side-chat)), can be the custom agent too,
which runs the chat on a local model with no account. It runs from the repository's source with a Python that has its
dependencies (`src/builder/agent/pyproject.toml`), and with Ollama serving a model that calls tools the command line
is:

```text
/usr/bin/env PYTHONPATH=/path/to/caliper/src/builder/agent /path/to/python -P -m caliper_agent --preset ollama --base-url http://127.0.0.1:11434/v1 --model <model>
```

The script `examples/build_a_dashboard/chat-agent.sh` wraps this line: give its absolute path as the custom command,
and it finds the source from its own place, takes the interpreter from `CALIPER_AGENT_PYTHON` (the repository's
`.venv` by default) and the model from `CALIPER_CHAT_MODEL`. An app started from the Finder gives its agents only the
system's `PATH`, so set such a variable in the line itself, as `/usr/bin/env CALIPER_AGENT_PYTHON=... /path/to/chat-agent.sh`.
It reaches Caliper with the session's own API file, so its edits wait on the approval cards like any other agent's.

The agent uses its own login. When it answers that it needs one, the pane shows LOGIN with the login methods the agent
names and the command its maker documents; Caliper never asks for a key and runs no login flow. Log in in your own
terminal and choose Restart.

## What Caliper starts

The agent starts when you send your first message. Caliper starts its command with the workspace folder as the working
folder, and sends:

- `initialize` with protocol version 1, declining the file-system and terminal capabilities, so the agent uses its own
  file and shell tools under its own permission settings;
- `session/new` with the workspace as `cwd` and one MCP server, named `caliper`: this build's `caliper-mcp` by
  absolute path, with the arguments `--origin agent --api-file <this session's API file>`. `caliper-mcp` needs Caliper
  API: when the app was started without `--api`, the side chat starts it then, on 127.0.0.1 at a free port, says so in
  the conversation ("Caliper API started on 127.0.0.1:<port> for this chat"), in Activity and in the status bar, and
  stops it when the agent ends, unless Open in browser has given out a link of it since
  ([Started by the app](http-api.md#started-by-the-app)). The API file is the
  session's own, `api-agent-<pid>-<n>.json` in the app-data folder, never `api.json`: its token is the agent's, with the
  origin `agent`, and the app revokes it and removes the file when the agent's session ends. The origin comes from that
  token, not from `--origin`, so every command the agent sends through Caliper API, by `caliper-mcp` or by any other
  client that reads the file, is validated, journaled, undoable and reviewed, and your session token from the start
  line is in nothing the agent is given;
- before the session's first message, a block for the assistant (`annotations.audience: ["assistant"]`) that asks the
  agent to change specs through the `caliper` tools and not to ask before calling one, since Caliper asks the person
  itself.

## The pane

The pane is the lower part of the right column, under Selection and Properties. Its header shows CHAT, Caliper's
mode for the agent (see [Modes](#modes)), the agent's name, Shared GPU when it applies (see
[Local models](#local-models)), its version and its own mode, the state as a tag, a FAULTS chip when the agent sent
messages Caliper dropped, and Restart when the agent must start again.

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

**The agent's text** shows a small part of Markdown: **bold**, `inline code` in the mono face, fenced code blocks on a
plate, bulleted and numbered lists, and paragraphs. Everything else shows as plain text with its marks taken out:
headings, emphasis, a link's or an image's text (its address is dropped, and nothing in the text can be clicked or
loads anything), HTML tags, quotes and table rows. A mark that is not closed yet, as the text streams in, is left out
until its pair arrives. Only the first 64 KiB of a message are read as Markdown, and a line longer than 2,000 bytes or
a paragraph longer than 8,000 shows as plain text, so a long or malformed message costs no more to draw than its
length. The conversation keeps the text as the agent sent it.

**The agent's permission requests** are cards: the file, the change, and one button per option the agent offers.
For a `*.spec.toml` the change is the compact diff that the inline surface and the approval cards show (a new panel's
primitive and title first, then its other keys and every other change), with Diff for the full diff; for any other
file it is the unified diff. An agent's "always allow" option is offered only when the setting allows it.

**Command approvals.** A command of the agent that needs your approval, such as an edit of the open spec while review
is on, is decided on the approval cards at the bottom of the window. While it waits, the conversation shows one row
for it, APPROVAL with the command and "waiting"; a click on the row outlines its card in bronze for a moment. Once it
is decided, the row folds into its call's group. When what a card showed changes while it waits, such as the open spec
that an overwrite would write, your approval no longer holds: the card is replaced by a new one for the new content,
and the row keeps waiting, on the new card.

**Groups.** Consecutive calls of the same command fold into one row with the count, the approvals' outcome and the
status, such as "source.add ×9 · approved · done"; a click opens it to show each call and its approval. A group with a
failed call or an approval that was denied, timed out or failed says so in bronze and is open from the start. A lone
call without an approval is a plain row.

**The input** is one line that grows to six with Shift+Enter or Command+Enter; Enter sends; Shift+Tab changes
Caliper's mode. The agent picker beside it
lists the agents by name, with NOT FOUND for those not on `PATH`, Custom command, and Local model.

## Modes

Caliper decides which of the agent's commands wait for your card, whatever the agent itself asks. The mode is the
framed word after CHAT in the header; a click on it opens its menu, and Shift+Tab, while the input has the keyboard,
moves through the first three:

| Mode | What the agent's commands do |
|---|---|
| Ask | Edits of the open spec and commands that delete or overwrite saved data or run a Python loader wait for your card. This is the default. |
| Accept edits | Edits of the open spec apply at once; everything else that asked still asks. |
| Read only | The agent can read and propose (dry runs, edits of a preview); every other change is refused, and the agent is told why. |
| Allow all | Only from the menu's last item, Allow all for this session: nothing waits for a card, except an overwrite of a saved applet or the replacement of a spec with unsaved changes. A banner, ALLOW ALL with End, stays under the header while it is on. It is never saved and ends when the agent's session ends. |

A command that runs without a card because of the mode or a rule shows as an APPLIED row with what allowed it, and
Undo, which takes it back while it is the newest change. An approval card of the chat's agent offers Allow once,
Allow always for this session, Always allow `<command>` and Deny; Always allow is saved for this agent. The settings
(⌘,) set the saved mode of the side chat and of `caliper-mcp`'s clients and list the saved rules under Permissions,
each with Revoke; they are kept in `agent-permissions.json` in the app-data folder, which only you can read and write.
That file is no protection against an agent that has its own access to your files, which can change it as it can
change any of them. The widget agent (⌘K) is not affected: its changes always wait in its preview for Accept.

## Local models

Your agent may run its model on your own machine, such as opencode with a model served by Ollama. Caliper cannot tell
where an agent's model runs, so you say so: choose Local model in the agent picker, which marks the chosen agent, or
clears the mark when it is set. The mark belongs to that agent, kept by its id, so choosing another agent leaves it
behind and choosing the agent again brings it back.

While a training run streams into Caliper (the app is attached to a live session) and the chosen agent is marked, the
chat's header shows **Shared GPU** after the agent's name, framed in bronze, as the widget agent's settings and its ⌘K
prompt do for a model server on this machine. The model and the run then share the same GPU and memory; Set up the
widget agent ([Share the GPU with a training run](widget-agent-setup.md#share-the-gpu-with-a-training-run))
gives the advice. As the header narrows, the note gives way after the agent's version and before its name.

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
