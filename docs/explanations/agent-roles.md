# Agent roles

Caliper's agents have two roles. The **side chat** is a chat pane where your own agent, such as Claude Code, Codex,
Gemini CLI or opencode, works with you on conversations and bigger jobs. The **widget agent** works inline on one thing
at a time, a panel, a selection, a dropped file or a property, and shows its change in place for you to accept or
reject. This page explains what each role is for, what each costs, how each can run on your own machine, where your
data goes, what each may do, and when the widget agent hands a request to the chat.

!!! note "Status"
    The widget agent's inline prompt works on the app's spec page ([Widget agent](../references/widget-agent.md)), and
    the side chat on the spec page and the live page ([Side chat](../references/side-chat.md)); both are tested with
    test agents, never with a person's agent or key. The designs are in ACP integration (`design/acp-integration.md`),
    Widget agent (`design/widget-agent.md`) and Widget agent sidecar (`design/byok-agent.md`).

## What the two have in common

Both roles speak the same protocol to Caliper, the Agent Client Protocol (ACP), so Caliper shows their messages, their
reasoning and each tool call in the same way, and a Stop control cancels either one.

Both act on Caliper through its command layer. A change to a spec is a command, so it is checked against the spec
format, recorded in the workspace's journal and undoable from the app. Commands that delete or overwrite saved data,
such as saving an applet or running a loader, need your approval in the app, whichever role sent them. A change that an
agent makes to the open spec waits for your review, with its diff, unless you turn review off; the command layer asks,
so it holds for every agent, including one that would not ask on its own, an agent outside the app over MCP too.

You review a waiting change on its card, in a strip that opens at the bottom of the window, above the jobs tray, on
whatever page you are on. The card says who asks (MCP agent, Side chat, Widget agent or API), the command and what it
acts on, such as "panel.patch · Loss and accuracy", and the change: for a spec, each changed key with its old value
struck through and its new one, the validator's warnings, and the whole diff behind Diff; for saved data, what would be
lost, such as "Applet run42-dashboard overwritten". It shows how long the agent still waits, two minutes at most, and
the panel the change touches carries a REVIEW mark on the spec page. Approve applies the change as the card showed it,
or refuses it if the spec changed meanwhile so that it would mean something else; Deny tells the agent no; and a
request nobody decides in time is withdrawn. Activity lists each request when it starts waiting and when it ends.

The two roles share one journal and one undo history, and their commands run one at a time in one order. What an agent builds is
data (specs and loaders), never code that runs in Caliper's process; the [trust model](trust-model.md) explains why
that matters.

## The side chat

The side chat is for conversations and for jobs that take more than one step: building a dashboard from nothing,
writing a loader script, changing your project's code, or explaining what a panel shows. For example, you might ask
"Load the CSV files in runs/, make a loader that joins them on step, and build a dashboard of loss and accuracy for
each run." The agent looks at the files, writes the loader, runs it once you approve, builds the panels and tells you
what it did.

The agent is your own, started by Caliper as a separate program: Claude Code (through its ACP adapter), Codex (through
its adapter), Gemini CLI or opencode, which Caliper detects, or any other ACP agent that you add by its command. It
uses Caliper's tools through Caliper's MCP server and also its own tools, such as reading and editing files, running
commands and searching the web, as its settings allow.

## The widget agent

The widget agent is for one small change or question about one target, answered in a few seconds. For example, you
select a dozen points on a loss curve, press ⌘K (Ctrl+K on Windows) and type "Make a table of these points". A table
of those rows appears where the layout would put it, marked as a preview, with the change to the spec beside it. Accept
applies it, Reject discards it, and Refine lets you say more.

The widget agent is `caliper-agent`, Caliper's own agent, which runs out of Caliper's process. Caliper gives it what
it needs to know about the target: the target's part of the spec, the schema and a sample of its data, a picture of
the panel and your selection. It may use only the commands that apply to that target.

## How they differ

| | Side chat | Widget agent |
|---|---|---|
| What it is for | Conversations and bigger jobs: building from nothing, loader scripts, project code, explanations | One panel, selection, dropped file or property, in a few seconds |
| The agent | Your own ACP agent: Claude Code, Codex, Gemini CLI, opencode, or any other you add by its command | `caliper-agent`, Caliper's own agent |
| Model | Whatever your agent uses, hosted or on your machine | Claude Haiku 5.5 at low effort by default, or another Claude model; other models, hosted or on your machine, are planned |
| Login | The agent's own, done once in your terminal; Caliper never sees a credential | A Claude API key that you enter once, kept in the system's credential store; a local model server will need none |
| Cost | Your agent's plan or account, as it bills you | Your API account, billed per token; a local model has no bill |
| What it may do | Caliper's tools, plus its own: files, commands, the web, as its settings allow | Only the commands that apply to the target, with every change shown as a preview first |
| How a change lands | As the agent makes it, with approval cards where they apply | As one undo step, when you accept the preview |
| Where your data goes | To your agent's model provider, or nowhere when its model runs on your machine | To the provider you chose, or nowhere when the model runs on your machine |

## Cost and login

With the side chat you pay for your agent the way you already do, often through a subscription that includes it.
Caliper adds nothing to that bill, but a long session in Caliper counts against your plan's limits like any other. You
sign in with the agent's own command in your terminal, once, and Caliper never handles a password, a token or a key.

The widget agent bills each request to your Claude API account by the tokens it uses. It defaults to Claude Haiku 5.5,
the fastest and cheapest Claude model it offers, at low effort, and each request is small, because Caliper gives it the
context it needs up front and it finishes in one or two rounds. Caliper keeps the tools and its instructions at the
start of every request, so the API can cache them. The key comes from the Claude Console; Caliper stores it in the
macOS Keychain, shows only whether a key is set, and has a control that removes it. The key reaches the agent only in
its environment when Caliper starts it, and never appears in a file, a log, the journal or a command line. Once other
providers are added, each one that needs a key gets an entry of its own. [Set up the widget
agent](../guides/set-up-the-widget-agent.md) gives the steps.

## Local and free operation

Both roles can run on a model on your own machine, which costs nothing per request; your computer does the work
instead.

- **The side chat** runs any ACP agent on whatever model that agent supports, local models included, and Caliper needs
  nothing extra for it. The worked example is opencode with a model served by Ollama, which
  ACP integration (`design/acp-integration.md`) sets up step by step.
- **The widget agent** will offer presets for local model servers, Ollama and LM Studio among them, with no key. A
  model must be able to call tools to drive Caliper; Caliper checks that, and also whether the model can look at
  images, when you choose it. Caliper will publish which models do the widget agent's work well, from a set of test
  tasks.

A local model shares your computer with everything else. On a Mac it uses the same GPU and the same memory as a
training run that streams into Caliper, so both slow down, and Caliper shows a note while a run is live. The setup
guide gives advice for that case.

## Privacy

Both roles send your prompts, and what the agent reads through Caliper, to a model provider: catalog entries, samples
of your data, spec text and pictures of panels. When the model runs on your machine, none of this leaves it; Ollama's
FAQ, for example, says that Ollama does not see your prompts or data when you run it locally.

With the side chat, the agent also has its own tools and may read files in the workspace folder, or elsewhere, and send
their contents to its provider. What it may do depends on its settings and its mode, which the side chat shows.

The widget agent reads only what Caliper gives it and what its commands return, and it talks only to its model
provider and to Caliper on your machine. When you point it at a model server on another computer, Caliper warns you
that your data will leave this one.

## What each may do

The side chat's agent can do more than Caliper's tools: write a loader script, fix a file, look something up. Its
changes through Caliper's tools are commands, checked, journaled and undoable; each change to the open spec waits for
your review with its diff, and anything that deletes or overwrites saved data waits for your approval. Its own tools act outside Caliper: an edit it makes to a file with them does not
pass through Caliper's journal or undo, and an agent in a mode that skips permission edits files without asking. You
keep control through its permission prompts, which Caliper shows, and through its settings and modes.

The widget agent can use only the commands that apply to its target, and only those that can show their change before
they make it. It cannot delete or overwrite saved data, run a loader, change files or run commands. Every change it
proposes is a preview until you accept it; accepted, the change is one undo step, and rejected, it leaves nothing
behind.

## When ⌘K hands off to the chat

The widget agent hands a request to the side chat when:

- you choose **Ask in chat** in its prompt;
- the request goes beyond one target, such as building a dashboard, writing a loader, changing project code or a long
  explanation; the widget agent says so and offers Ask in chat;
- no provider is set up, or the chosen model cannot call tools; then ⌘K offers only Ask in chat.

The side chat receives your request together with what the widget agent knew: the target's part of the spec, the
schema and a sample of its data, your selection, a picture of the panel when your agent accepts images, and the
conversation so far, including the change that the widget agent proposed, marked as not applied. You see the whole
message in the chat, and the agent goes on from there.
