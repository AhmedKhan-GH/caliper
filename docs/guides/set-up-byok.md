# Set up BYOK

The BYOK backend of the agent pane (bring your own key) runs Caliper's own agent against the Claude API with your API
key. This guide shows how to get a key, enter it in the pane, pick a model and remove the key again.
[Agent pane backends](../explanations/agent-pane-backends.md) explains how BYOK differs from running your own agent.

!!! note "Status"
    The agent pane is not built yet, so the steps that use the pane describe how it is planned to work, and the names
    of its controls may change. What exists today is the agent itself (`caliper-agent`) and the store that keeps the
    key; [BYOK agent](../plans/byok-agent.md) describes them, and its manual check shows how to try the agent with your
    key from the terminal.

## Get a key

1. Sign in to the Claude Console with the account that should pay for the requests, or create one. An organisation
   may give you a key instead; then skip to the next section.
2. Make sure the account has billing set up or credits, since every request of the agent is billed to it by the
   tokens it uses.
3. Open the API keys page, create a key, and give it a name that says where it is used, such as "Caliper on my
   laptop". Copy the key: the Console shows it only once.

Keep the key to yourself. Anyone who has it can make requests that your account pays for.

## Enter the key in the pane

1. Open the agent pane and choose the Claude API backend.
2. Open the pane's settings, paste the key into the key field and save it.
3. The pane now says that a key is set. It never shows the key again, not even in part.

Caliper keeps the key in the macOS Keychain, as an item of the service `caliper.agent-keys` that can be read only while
your Mac is unlocked and is never synchronised to your other devices. It does not write the key to a file, a log, the
workspace or the journal. When the pane starts the agent, it gives the key to the agent in its environment, and the
agent sends it only to the Claude API.

## Pick a model

The pane offers four Claude models for each session.

| Model | Use it for |
|---|---|
| Claude Opus 5.5 | The default: building and checking visuals from your data |
| Claude Sonnet 5.5 | Everyday edits, faster and cheaper |
| Claude Haiku 5.5 | Small changes, the fastest and cheapest |
| Claude Fable 5.2 | The hardest work, at the highest cost |

Next to the model you can set the **effort**, how much the model thinks before it acts: low, medium, high (the
default), extra high or maximum. Higher effort is slower and costs more; lower it for quick edits.

The **review** setting decides when the agent asks before acting. By default it asks before each change to the open
spec and shows the change as a diff, and it always asks before anything that deletes or overwrites saved data, such as
saving an applet or running a loader. Choose "Ask only before deleting or overwriting" to let it change the spec
freely; every change can still be undone from the app.

A new model or effort applies from the agent's next request. Changing the model in the middle of a session works, but
the first request after the change costs more, because the cache of the conversation belongs to the previous model.

## Use it

Type what you want in the pane, for example "Plot the training loss of run 42 next to a heatmap of the attention
entropy". The agent looks at the catalog, says what it is about to do, and makes the changes one command at a time. A
card appears when it needs your approval; Allow runs the command, Deny tells the agent to stop and ask you how to go
on. Stop cancels the turn at any point.

If something goes wrong, the pane says so in plain words:

- **No Claude API key is set.** Enter a key as above.
- **The Claude API did not accept the API key.** The key was revoked or mistyped; create a new one and enter it.
- **The Claude API is rate limited.** The agent waits and tries again on its own, and the pane shows how long it
  waits. If it gives up, try again later or raise your account's limits in the Console.
- **The model is not available to this key.** Pick another model, or ask the account's owner for access.
- **The model declined this request.** Rephrase the request; the declined prompt is left out of the conversation.

## Remove the key

Open the pane's settings and remove the key. The pane then says that no key is set, and the agent refuses new prompts
until you enter one again.

The key also appears in the Keychain Access app as an item of the service `caliper.agent-keys`, and deleting it there
has the same effect. If you think someone else has seen the key, also revoke it on the API keys page of the Claude
Console, so it stops working everywhere.
