# Set up the widget agent

The widget agent works inline on one panel, selection, dropped file or property, and shows each change in place for
you to accept or reject. It needs a model provider: today a Claude API key, and later one of several presets, where a
model server on your own machine needs no key. This guide shows how to get and enter a key, pick a model, use the
widget agent, run it on a local model once that is possible, share the GPU with a training run, and remove the key
again. [Agent roles](../explanations/agent-roles.md) explains how the widget agent differs from the side chat.

!!! note "Status"
    The widget agent's prompt and its settings are not built yet, so the steps that use them describe how they are
    planned to work, and the names of their controls may change. What exists today is the agent itself
    (`caliper-agent`) with Claude as its one provider, and the store that keeps the key;
    [Widget agent sidecar](../design/byok-agent.md) describes them, and its manual check shows how to try the agent with
    your key from the terminal.

## Choose a provider

The widget agent's settings offer presets, each of which fills in the address of the model's API. Only Claude works
today; the others are planned.

| Preset | Key | Status |
|---|---|---|
| Claude | A Claude API key | Works today |
| OpenAI, Gemini, OpenRouter | An API key from that service | Planned |
| Ollama, LM Studio | None: the model runs on your machine | Planned |
| Custom | Optional, for any other server with an OpenAI-compatible API | Planned |

Claude always runs on Anthropic's own API, so that its features, such as caching, stay available.

## Get a Claude API key

1. Sign in to the Claude Console with the account that should pay for the requests, or create one. An organisation
   may give you a key instead; then skip to the next section.
2. Make sure the account has billing set up or credits, since every request of the agent is billed to it by the
   tokens it uses.
3. Open the API keys page, create a key, and give it a name that says where it is used, such as "Caliper on my
   laptop". Copy the key: the Console shows it only once.

Keep the key to yourself. Anyone who has it can make requests that your account pays for.

## Enter the key

1. Open the widget agent's settings and choose the Claude preset.
2. Paste the key into the key field and save it.
3. The settings now say that a key is set. They never show the key again, not even in part.

Caliper keeps the key in the macOS Keychain, as an item of the service `caliper.agent-keys` that can be read only while
your Mac is unlocked and is never synchronised to your other devices. Each provider that needs a key gets an item of
its own. Caliper does not write the key to a file, a log, the workspace or the journal. When it starts the agent, it
gives the key to the agent in its environment, and the agent sends it only to the provider's API.

## Pick a model

The widget agent defaults to Claude Haiku 5.5 at low effort, because it answers fastest and costs least, and a widget
request is a small change with its context given in full. You can choose any of the four Claude models.

| Model | Use it for |
|---|---|
| Claude Haiku 5.5 | The default: quick changes, the fastest and cheapest |
| Claude Sonnet 5.5 | Changes that need more judgement, still fast |
| Claude Opus 5.5 | Harder requests, slower and dearer |
| Claude Fable 5.2 | The hardest requests, at the highest cost |

Next to the model you can set the **effort**, how much the model thinks before it acts: low (the default for the
widget agent), medium, high, extra high or maximum. Higher effort is slower and costs more.

There is no review setting for the widget agent: every change it proposes is a preview that you accept or reject.

## Use it

1. Press ⌘K (Ctrl+K on Windows) on a panel, on a selection in a panel, under a file you just dropped on the window, or
   on the ⌘K button beside a property in the builder.
2. Type what you want, for example "Use a linear y axis" or "Make a table of these points", and press Return.
3. The panel shows the change as a preview, with the change to the spec beside it. **Accept** applies it as one undo
   step, **Reject** discards it and leaves nothing behind, and **Refine** lets you add to the request.
4. When the request is bigger than one target, choose **Ask in chat**, which sends it, with what the widget agent knew,
   to the side chat.

If something goes wrong, the prompt says so in plain words:

- **No provider is set up.** ⌘K offers only Ask in chat. Enter a key as above.
- **The Claude API did not accept the API key.** The key was revoked or mistyped; create a new one and enter it.
- **The Claude API is rate limited.** The agent waits and tries again on its own, and says how long it waits. If it
  gives up, try again later or raise your account's limits in the Console.
- **The model is not available to this key.** Pick another model, or ask the account's owner for access.
- **The model declined this request.** Rephrase it, or choose Ask in chat.
- **The preview is out of date.** The spec changed while the preview was open, for example through the side chat.
  Choose Refine to start again from the current spec.

## Use a local model

This part is planned. A local model costs nothing per request and keeps your data on your machine, but it uses your
computer's GPU and memory, and small models do this work less reliably than large ones.

1. Install a model server, such as Ollama or LM Studio, and download a model that supports tool calling.
2. In the widget agent's settings, choose the Ollama preset, which uses `http://localhost:11434/v1`, or the LM Studio
   preset, which uses `http://localhost:1234/v1`. Neither needs a key.
3. Pick the model from the list that the server offers.
4. Caliper checks the model with two short test requests. If the model cannot call tools, it cannot drive Caliper, and
   the settings say so. If it cannot look at images, the widget agent works without a picture of the panel. You can
   override either answer.
5. Give the model a context length of at least 8,192 tokens; below that, the settings warn that requests may not fit.

Caliper will publish which models pass the widget agent's test tasks, so you can choose one that works. A server on
another computer works too, through the Custom preset, but then your data leaves your machine, and Caliper warns you.

## Share the GPU with a training run

On a Mac, a local model uses the same GPU and the same memory as a training run that streams into Caliper. Both slow
down, and if the model and the run do not fit in memory together, the whole machine slows as it starts swapping.
Caliper shows a note in the widget agent's prompt, and in the side chat when you have marked its agent as running on a
local model, while a run is live.

- Use a smaller or more compressed (quantized) model while a run trains.
- Keep the model's context length modest; the context takes memory too.
- Unload the model when you are done. With Ollama, `ollama stop <model>` unloads it at once, and otherwise a model
  stays loaded for five minutes after its last request.
- Check where the model runs. With Ollama, `ollama ps` shows whether a model is on the GPU or the CPU and with which
  context length.
- Watch the memory pressure in Activity Monitor, and switch to a hosted preset for the length of a long run if it stays
  high.

On a computer with a separate graphics card, the model's weights take video memory that the run may need, and the
same advice holds.

## Local models in the side chat

The side chat runs your own agent, so it uses whatever model your agent is set up for, local models included.
[ACP integration](../design/acp-integration.md#local-models-in-the-side-chat) sets up opencode with a model served by
Ollama as a worked example.

## Remove the key

Open the widget agent's settings and remove the key. The settings then say that no key is set, and ⌘K offers only Ask
in chat until you enter one again.

The key also appears in the Keychain Access app as an item of the service `caliper.agent-keys`, and deleting it there
has the same effect. If you think someone else has seen the key, also revoke it on the API keys page of the Claude
Console, so it stops working everywhere.
