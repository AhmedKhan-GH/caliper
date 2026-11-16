# Agent pane backends

The agent pane is where you ask an agent to build visuals from your data inside Caliper. It has two backends behind
one chat. With the **ACP backend**, Caliper runs your own coding agent (Claude Code, Codex or Gemini CLI) on that
agent's own login. With the **BYOK backend** (bring your own key), Caliper runs its own agent against the Claude API
with an API key that you enter once. This page explains how the two differ in cost, login, privacy and control, and
when to use each.

!!! note "Status"
    The agent pane is not built yet. What exists today are the parts underneath it: the ACP client that runs an agent,
    the BYOK agent `caliper-agent`, the store that keeps your key, and the command layer they act through. They are
    tested without the pane, and the pane comes once they are wired into the app. The designs are in
    [ACP integration](../plans/acp-integration.md) and [BYOK agent](../plans/byok-agent.md).

## What the two have in common

Both backends speak the same protocol to the pane, the Agent Client Protocol (ACP). The pane therefore shows the same
things for both: the agent's messages and its reasoning, each tool call as it runs, a card with the change as a diff
when the agent proposes to change a spec, a permission card before anything that deletes or overwrites saved data, and
a Stop button that cancels the turn.

Both act on Caliper through its command layer. An agent's change to a spec is a command, so it is checked against the
spec format, recorded in the workspace's journal and undoable from the app. Commands that delete or overwrite saved data, such as saving an applet or running a loader, need your approval
in the app whichever backend sent them. What an agent builds is data (specs and loaders), never code that runs in
Caliper's process; the [trust model](trust-model.md) explains why that matters.

## How they differ

| | ACP backend | BYOK backend |
|---|---|---|
| The agent | Your own agent: Claude Code (through its ACP adapter), Codex (through its adapter) or Gemini CLI | `caliper-agent`, Caliper's own agent, which runs out of Caliper's process |
| Model | Whatever your agent uses | Claude Opus 5.5 by default; Claude Sonnet 5.5, Claude Haiku 5.5 or Claude Fable 5.2 from the pane |
| Login | The agent's own, done once in your terminal; Caliper never sees a credential | A Claude API key that you paste into the pane once; Caliper keeps it in the system's credential store |
| Cost | Your agent's plan or account, as it bills you | Your Claude API account, billed per token for each request |
| What it can do | Caliper's tools, plus its own: reading and editing files, running commands, searching the web, as its settings allow | Caliper's tools only: no files, no shell, no web |
| Where your data goes | To your agent's provider, under that provider's terms, including files the agent reads itself | To the Claude API, under your API account's terms |
| Settings | The agent's own modes and settings | Model, effort and review in the pane |

### Cost

With the ACP backend you pay for your agent the way you already do, often through a subscription that includes it.
Caliper adds nothing to that bill, but a long session in Caliper counts against your plan's limits like any other.

With the BYOK backend each request is billed to your Claude API account by the tokens it uses. Larger models and
higher effort cost more per turn; Claude Haiku 5.5 is the cheapest and Claude Fable 5.2 the most expensive. Caliper
keeps the tools and its instructions at the start of every request so the API can cache them, which makes the
requests after the first in a session much cheaper. A capture of the window costs a few thousand tokens each time the
agent looks at one.

### Login

The ACP backend uses the agent's own login. You sign in with the agent's command in your terminal, once, and the pane
uses that session; Caliper never handles a password, a token or a key.

The BYOK backend needs a Claude API key, which you create in the Claude Console and paste into the pane. Caliper
stores it in the macOS Keychain, shows only whether a key is set, and has a control that removes it. The key reaches
the agent only in its environment when Caliper starts it, and never appears in a file, a log, the journal or a command
line. [Set up BYOK](../guides/set-up-byok.md) gives the steps.

### Privacy

Both backends send your prompts, and what the agent reads through Caliper's tools, to a model provider: catalog
entries, samples of your data, spec text and captures of the window.

With the ACP backend, the agent also has its own tools and may read files in the workspace folder, or elsewhere, and
send their contents to its provider. What it may do depends on the agent's mode, which the pane shows.

With the BYOK backend, the agent can read only what Caliper's commands give it, and it talks only to the Claude API and
to Caliper on your machine. Your data is handled under the terms of your Claude API account.

### Control

The ACP backend gives you an agent that can do more than Caliper's tools: write a loader script, fix a file, look
something up. You keep control through its permission prompts, which Caliper shows, and through its modes. An agent in
a mode that skips permission edits files without asking, and such an edit does not pass through Caliper's journal or
undo.

The BYOK backend can only use Caliper's commands, so every change it makes is checked, journaled and undoable, and
the changes of one prompt undo as one step. By default it asks before each change to the open spec, with the change shown as a diff, and always before anything that
deletes or overwrites saved data. You choose the model and how much it thinks (its effort) in the pane, and you can let
it change the spec without asking for the rest of a session.

## When to use each

Use the **ACP backend** when you already use one of these agents and pay for it, when the work goes beyond building
visuals (writing or fixing a loader script, changing files of a project), or when you want the agent's own habits and
settings inside Caliper.

Use the **BYOK backend** when you do not use a coding agent, when you want the agent confined to Caliper's commands so
that every change is undoable and nothing outside Caliper is touched, when you want to pick the Claude model and its
effort per session, or when your organisation provides Claude API keys rather than agent subscriptions.

You can switch between them in the pane. A session belongs to one backend, so switching starts a new conversation;
the spec and everything else in the workspace stay as they are.
