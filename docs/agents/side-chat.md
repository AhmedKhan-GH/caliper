# Side Chat

The side chat is the pane where your own agent works with you over the Agent Client Protocol (ACP). It supports Claude
Code and Codex through their adapters, Gemini CLI, opencode, and any other ACP agent that you add by its command.
[Agent roles](agent-roles.md) explains when to use the side chat and when to use the widget agent.

## Side Chat Access

⌘L Chat in the page bar (Ctrl+L Chat elsewhere) opens and closes the side chat on the spec page and on the live page.
The button has a bronze outline while the chat is open. The chat is the right column of the window. It is 28 % of the
window width and at least 300 points wide, and the minimum grows with the text scale. The page bar, the run bar, the
status bar, the jobs tray and the approval cards keep their places, and the page fills the space that the column leaves.

The chat stays open when the page changes. For each `spec.new`, `spec.open` and `instrument.open` of the agent, the app
shows the spec page, so you see the spec whose edits you review beside the same conversation. A library instrument's page
is the exception, because an agent never ends it. There the spec opens in the session, the page stays, and Activity says
how to get to it ([Caliper CLI](../reference/cli.md)).

The app keeps the chat settings in `side-chat.json` in the app-data folder ([Folders and environment
variables](../reference/folders-and-environment.md)). The file holds the chosen agent, the name and command line of the
custom agent, and whether the "always allow" options of an agent are offered. It also holds the agents that you marked
as running on a local model (`local_model_agents`, a list of agent ids) and whether the chat is open. It never holds a
key.

## Agents

Caliper detects four agents by a search for their commands on `PATH`. It runs nothing to find them.

| Agent | Launch command | Looked for on `PATH` |
|---|---|---|
| Claude Code | `npx -y @agentclientprotocol/claude-agent-acp` | `npx` and `claude` |
| Codex | `npx -y @agentclientprotocol/codex-acp` | `npx` and `codex` |
| Gemini CLI | `gemini --acp` | `gemini` |
| opencode | `opencode acp` | `opencode` |

A custom agent is a name and one command line. Caliper splits the line into words as a POSIX shell does, with single
quotes, double quotes and backslashes only. It supports no variables, globbing or redirection, and it runs the program
without a shell. The chosen agent and the custom command belong to the settings of the app, never to a workspace.

`caliper-agent`, the agent behind ⌘K ([Widget agent](widget-agent.md#side-chat-use)), can be the custom agent. It then
runs the chat on a local model with no account. It needs a Python with its dependencies and an Ollama model that calls
tools. The command line is:

```text
/usr/bin/env PYTHONPATH=/path/to/caliper/src/builder/agent /path/to/python -P -m caliper_agent --preset ollama --base-url http://127.0.0.1:11434/v1 --model <model>
```

You can wrap this line in a small shell script and give its absolute path as the custom command. An app that starts
from the Finder gives its agents only the system `PATH`. Thus set environment variables in the command line itself
with `/usr/bin/env`. This agent reaches Caliper with the API file of its session. Thus its edits wait on the approval
cards like those of any other agent.

The agent uses its own login. When it answers that it needs a login, the pane shows LOGIN with the login methods that
the agent names. Caliper never asks for a key and runs no login flow. Log in in your own terminal and choose Restart.

## Agent Startup

The agent starts when you send your first message. Caliper starts its command with the workspace folder as the working
folder and sends these items:

- `initialize` with protocol version 1. It declines the file-system and terminal capabilities, so the agent uses its
  own file and shell tools under its own permission settings.
- `session/new` with the workspace as `cwd` and one MCP server named `caliper`. The server is the `caliper-mcp` of this
  build with the arguments `--origin agent --api-file <this session's API file>`.
- Before the first message, a block for the assistant (`annotations.audience: ["assistant"]`). The block asks the agent
  to change specs through the `caliper` tools and not to ask before it calls one, because Caliper asks the person.

`caliper-mcp` needs Caliper API. If the app started without `--api`, the side chat starts the API on 127.0.0.1 at a free
port and says so in the conversation, in Activity and in the status bar. It stops the API when the agent ends, unless
Open in browser gave out a link of it ([Started by the app](http-api.md#app-startup)).

The API file belongs to the session and is never `api.json`. Its token belongs to the agent and has the origin `agent`.
The app revokes the token and removes the file when the session ends. The origin comes from the token and not from
`--origin`. Thus Caliper validates, journals and reviews every command that the agent sends through Caliper API, and you
can undo it. Your session token is in nothing that the agent receives.

## Pane

The pane is the lower part of the right column, under Selection and Properties. In the build mode of the spec page it
sits under Properties ([Builder pane](../builder/builder-pane.md)). The header shows CHAT, the Caliper mode for the
agent (see [Modes](#modes)), the name of the agent, and Shared GPU when it applies (see
[Local models](#local-models)). It also shows the version and mode of the agent and the state as a tag. A FAULTS chip
marks dropped agent messages, and Restart appears when the agent must start again.

| Tag | State | Action |
|---|---|---|
| STARTING | The agent is starting and opening its session | |
| RUNNING | A turn runs | Cancel in the input row |
| STALLED | The agent has sent nothing for two minutes in a turn | Cancel, Restart |
| STOPPING | A cancel waits for the agent to end the turn | Restart |
| LOGIN | The agent needs a login | Restart |
| EXITED | The process of the agent ended. The tag shows the exit status and the last line of standard error | Restart |
| FAILED | The agent could not start, for example because it speaks another protocol version | Restart |
| NO AGENT | No agent was found and no custom command is set | Custom command |

The conversation holds your messages, the text of the agent, its thoughts as one line, and its plan. Each tool call is a
row with its command name or title and its status (pending, running, done, failed or stopped). Each turn ends with one word: DONE, STOPPED, DECLINED, LIMIT or FAILED.

**The agent's text** shows a small part of Markdown: **bold**, `inline code`, fenced code blocks, bulleted and numbered
lists, and paragraphs. Everything else shows as plain text with its marks removed, including headings, emphasis, link
and image text, HTML tags, quotes and table rows. Nothing in the text can be clicked or loads anything. Caliper reads
only the first 64 KiB of a message as Markdown, and a line longer than 2,000 bytes or a paragraph longer than 8,000
bytes shows as plain text. The conversation keeps the text as the agent sent it.

**The agent's permission requests** are cards with the file, the change, and one button for each option that the agent
offers. For a `*.spec.toml` file the change is the compact diff of the approval cards, and Diff shows the full diff. For
any other file the change is the unified diff. The "always allow" option of an agent appears only when the setting
allows it.

**Command approvals.** A command that needs your approval, such as an edit of the open spec while review is on, waits on
the approval cards at the bottom of the window. The conversation shows one row for it, APPROVAL with the command and
"waiting". When you decide, the row folds into the group of its call. If
the content that a card showed changes while it waits, your approval no longer holds. Then a new card replaces the old
one, and the row keeps waiting.

**Groups.** Consecutive calls of the same command fold into one row with the count, the outcome of the approvals and the
status, such as "source.add ×9 · approved · done". A click opens the row. A group with a failed call, or with an
approval that was denied, timed out or failed, says so in bronze and starts open. A lone call without an approval is a
plain row.

**The input** is one line that grows to six lines with Shift+Enter or Command+Enter. Enter sends the message, and
Shift+Tab changes the Caliper mode. The agent picker beside it lists the agents by name, with NOT FOUND for those not on
`PATH`, Custom command, and Local model.

## Modes

Caliper decides which commands of the agent wait for your card, whatever the agent itself asks. The mode is the framed
word after CHAT in the header. A click on it opens its menu. Shift+Tab moves through the first three modes while the
input has the keyboard.

| Mode | What the agent's commands do |
|---|---|
| Ask | Edits of the open spec and commands that delete or overwrite saved data or run a Python loader wait for your card. This is the default. |
| Accept edits | Edits of the open spec apply at once. Everything else that asked still asks. |
| Read only | The agent can read and propose (dry runs, edits of a preview). Every other change is refused, and the agent is told why. |
| Allow all | Nothing waits for a card, except an overwrite of a saved instrument or the replacement of a spec with unsaved changes. Run commands, managed runs and Python loaders run without a card. You choose it from the last item of the menu or in the settings. Caliper saves it, and it stays on after a restart until you choose another mode. While it is on, a banner, ALLOW ALL with End, stays under the header, and the status bar says "allow all for side chat". End sets Ask. If Caliper cannot save the change, the banner stays with "Still saved" and the settings say why. |

A command that runs without a card because of the mode or a rule shows as an APPLIED row with what allowed it. The row
has Undo, which takes the command back while it is the newest change. An approval card of the chat agent offers Allow
once, Allow always for this session, Always allow `<command>` and Deny. Caliper saves Always allow for this agent.

The settings (⌘,) set the saved mode of the side chat and of the clients of `caliper-mcp`, Allow all included. They
list the saved rules under Permissions, each with Revoke. The modes and the rules are in `agent-permissions.json` in the
app-data folder, which only you can read and write. Caliper reads Allow all only from a file without errors. No command,
endpoint, MCP tool, environment variable, command-line flag or spec sets a mode.

An agent with its own access to your files can change this file, so it gives no protection against such an agent. A
Python loader or a managed run is your own code, and it can also change this file. Under Allow all, such code runs
without a card. The widget agent (⌘K) is not affected, because its changes always wait in its preview for Accept.

## Local Models

Your agent may run its model on your own machine, such as opencode with a model served by Ollama. Caliper cannot tell
where the model runs, so you say so. Choose Local model in the agent picker to mark the chosen agent, or to clear the
mark. The mark belongs to that agent by its id, so it returns when you choose the same agent again.

While a training run streams into Caliper (the app is attached to a live session) and the chosen agent is marked, the
header of the chat shows **Shared GPU** after the name of the agent, framed in bronze. The model and the run then share
the same GPU and memory. [Share the GPU with a training run](widget-agent-setup.md#gpu-sharing) gives the advice.

## Ask in Chat Action

Ask in chat in the prompt of the widget agent on the spec page opens the side chat with a draft. The draft has your
request in the input and, above it, one chip for each block of context that goes with it. The blocks are the spec
fragment of the target, its sources, the render of the panel, the widget conversation, the diff of the preview marked as
not applied, and the ids of the target. Caliper sends nothing until you send the draft, and Clear drops the context.
Caliper sends the blocks as embedded resources if the agent accepts them and as text if not. A handoff in the app leaves
out the render of the panel.
