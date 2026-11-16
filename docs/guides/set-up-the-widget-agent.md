# Set up the widget agent

The widget agent works inline on one panel and shows each change in place for you to accept or reject. It needs a
model provider: Claude, OpenAI, Gemini or OpenRouter with an API key, a model server on your own machine such as
Ollama or LM Studio, which needs no key, or any other server with an OpenAI-compatible API. This guide shows how to
open the widget agent's settings, choose a provider, enter a key, pick a model, check what the model can do, use the
widget agent, share the GPU with a training run, and remove the key again.
[Agent roles](../explanations/agent-roles.md) explains how the widget agent differs from the side chat.

!!! note "Status"
    The settings and ⌘K on a panel, on a selection, on a dropped file and on a property are built.
    Until Caliper ships the agent with its own Python runtime, Caliper finds the agent through the environment
    variable `CALIPER_AGENT_PYTHON`, an interpreter that has `caliper-agent`'s dependencies, and the agent reaches
    your workspace through Caliper API, so start Caliper with `--api`.

## Open the settings

The settings open over the page you are on, from any of three places:

- the landing screen's **⌘, Agent** button, at the right of the command bar;
- the key **⌘,** (Ctrl+, on Windows) on any page;
- **Setup** in the ⌘K prompt, when no provider is set up, and **Model** after the model declined a request.

Escape or **esc Close** closes them. What you choose is kept in `widget-agent.json` in Caliper's app-data folder, beside
the side chat's settings. The file holds the provider, the address, the model and the effort, never a key.

## Choose a provider

Choose one of the presets under **Provider**. Each fills in the address of the model's API.

| Preset | Base URL | Key |
|---|---|---|
| Claude | `https://api.anthropic.com`, fixed | A Claude API key |
| OpenAI | `https://api.openai.com/v1`, fixed | An OpenAI API key |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/`, fixed | A Gemini API key |
| OpenRouter | `https://openrouter.ai/api/v1`, fixed | An OpenRouter API key |
| Ollama | `http://localhost:11434/v1`, which you can change | None |
| LM Studio | `http://localhost:1234/v1`, which you can change | None |
| Custom | The address you type | Optional |

Claude always runs on Anthropic's own API, so that its features, such as caching, stay available. The hosted presets
keep their address, so that a preset's key only ever goes to that service; for any other server, use Custom.

Under the address, a tag warns you about where your data goes:

- **Remote address**: an Ollama, LM Studio or Custom address on another computer. Your prompts and your workspace's
  data, such as samples, spec text and pictures of panels, leave this machine.
- **Unencrypted**: such an address that starts with `http://`, so the data also travel unencrypted.
- **Invalid address**: an address Caliper does not take. It must start with `http://` or `https://`, and must not hold a
  user name, a password, a query or a fragment.

## Get a key

For Claude:

1. Sign in to the Claude Console with the account that should pay for the requests, or create one. An organisation
   may give you a key instead; then skip to the next section.
2. Make sure the account has billing set up or credits, since every request of the agent is billed to it by the
   tokens it uses.
3. Open the API keys page, create a key, and give it a name that says where it is used, such as "Caliper on my
   laptop". Copy the key: the Console shows it only once.

For OpenAI, Gemini and OpenRouter, create a key on that service's own API keys page in the same way. Keep every key
to yourself: anyone who has it can make requests that your account pays for.

## Enter the key

1. In the settings, choose the preset.
2. The **Key** row says **Not set**. Paste the key into its field and choose **Save**, or press Return.
3. The row now says **Set**, with **Replace** and **Remove**. The settings never show the key again, not even in part.

Each provider has a key of its own, so entering an OpenAI key leaves your Claude key as it was. Ollama and LM Studio
have no key row. A Custom server's key is optional, and is kept per address.

Caliper keeps the key in the macOS Keychain, as an item of the service `caliper.agent-keys` that can be read only while
your Mac is unlocked and is never synchronised to your other devices. The item is named after the preset (`anthropic`
for Claude, `openai`, `gemini`, `openrouter`), and a Custom server's after its address. Caliper does not write the key
to a file, a log, the workspace or the journal. When it starts the agent, it gives the key to the agent in its
environment, and the agent sends it only to the provider's API.

## Pick a model

Type the model's id under **Model**, or choose it from the choices under the field.

- **Claude** offers its four models. The widget agent defaults to Claude Haiku 5.5 at low effort, because it answers
  fastest and costs least, and a widget request is a small change with its context given in full.

  | Model | Use it for |
  |---|---|
  | Claude Haiku 5.5 | The default: quick changes, the fastest and cheapest |
  | Claude Sonnet 5.5 | Changes that need more judgement, still fast |
  | Claude Opus 5.5 | Harder requests, slower and dearer |
  | Claude Fable 5.2 | The hardest requests, at the highest cost |

  **Effort** sets how much the model thinks before it acts: low (the default), medium, high, xhigh or max. Higher
  effort is slower and costs more.
- **The other presets** have no default model. **List** asks the server for its models and shows them as choices,
  where the server offers a list. A hosted service needs its key for that.

There is no review setting for the widget agent: every change it proposes is a preview that you accept or reject.

## Check what the model can do

Beside the model, **Capabilities** shows three answers, each with where it came from:

| Row | What it means |
|---|---|
| Tool calling | Whether the model calls tools. A model without it cannot drive Caliper |
| Vision | Whether the model reads images. Without it, the widget agent works without a picture of the panel |
| Context length | How many tokens the model takes. Caliper sizes what it sends to the model to a quarter of it |

**Probe** checks the model with two short test requests: one asks it to call a tool, and one asks it to name the
colour of a small picture. For Claude it reads Anthropic's model list instead, and spends no tokens. The context length
comes from the server where it reports one, else Caliper assumes 8,192 tokens; for Ollama it is the preset's own,
16,384 tokens (see [Use a local model](#use-a-local-model)). Caliper keeps the answers for 30 days.

Each row has **Yes**, **No** and **Auto**, and the context length a field: an answer you set wins over the probe's
until you choose Auto again. A short tag says when the probe could not answer, such as **Server unreachable**,
**Timeout** or **Key rejected**.

When the model cannot call tools, the settings show **No tool calling**, and ⌘K offers only Ask in chat. Other tags
say what is still missing: **No provider**, **No key**, **No model** or **Invalid address**. For a server on your own
machine, **Context cache about 18 GB** warns that the model's context length would take that much memory (4 GB or
more), on top of the model itself.

## Use it

1. Focus a panel and press ⌘K (Ctrl+K on Windows).
2. Type what you want, for example "Use a linear y axis", and press Return. The prompt row shows the model, as the
   agent reports it.
3. The panel shows the change as a preview, with the change to the spec beside it. **Accept** applies it as one undo
   step, **Reject** discards it and leaves nothing behind, and **Refine** lets you add to the request.
4. When the request is bigger than one panel, choose **Ask in chat**, which sends it, with what the widget agent knew,
   to the side chat.

A change in the settings takes effect at once: an open ⌘K prompt closes, its preview is discarded and a request still
running is cancelled, and the next ⌘K uses the new settings. If something goes wrong, the prompt shows it with a short
tag, such as DECLINED or NO PROVIDER, and the actions that help:

- **No provider is set up**, or the model cannot call tools. ⌘K offers Setup and Ask in chat; Setup opens the settings.
- **The API did not accept the key.** The key was revoked or mistyped; create a new one and enter it with Replace.
- **The API is rate limited.** The agent waits and tries again on its own, and says how long it waits. If it gives
  up, try again later or raise your account's limits.
- **The model is not available to this key.** Pick another model, or ask the account's owner for access.
- **The model declined this request.** The request ends, nothing is changed, and the prompt shows DECLINED with the
  model's name. This can happen with ordinary work too, for example with chemistry or biology data, so you can
  rephrase the request (Rephrase), try it again (Retry), or switch to another model or provider, such as a local model
  (Model). Automatic fallback to another model is off for now.
- **The preview is out of date.** The spec changed while the preview was open, for example through the side chat,
  so that the change no longer applies, and the prompt shows CONFLICT. Choose Refine to start again from the current
  spec.

## Use a local model

A local model costs nothing per request and keeps your data on your machine, but it uses your computer's GPU and
memory, and small models do this work less reliably than large ones.

1. Install a model server, such as Ollama or LM Studio, and download a model that supports tool calling.
2. In the settings, choose the Ollama or LM Studio preset. Change the address if your server listens elsewhere.
3. Choose **List**, then the model.
4. Choose **Probe**. If the model cannot call tools, choose another one.
5. For LM Studio, type the context length you load the model with into the Context length field. Below 8,192 tokens
   requests may not fit.

**The context length with Ollama.** Ollama loads a model with room for its whole context length unless asked for
less, and that room, the context cache, takes memory on top of the model: for `qwen3.8:27b`, whose context length is
262,144 tokens, it is about 18 GB. Caliper therefore asks Ollama for 16,384 tokens for ⌘K and 32,768 tokens for the
side chat's `caliper-agent`, about 1 GB and 2 GB of cache for that model, and Ollama unloads the model a
minute after its last request. To change it, type another length into the Context length field (for ⌘K) or give
`--context-length` to `caliper-agent` (for the side chat); the tag **Context cache about N GB** shows what a large one
costs. When a conversation no longer fits, Caliper does not send it, and says so. If you reach Ollama through the
Custom preset instead, Caliper cannot set the length, and you set it yourself: start Ollama with
`OLLAMA_CONTEXT_LENGTH=16384 ollama serve`, which applies to every model, or make a copy of the model with a
Modelfile that holds `FROM <model>` and `PARAMETER num_ctx 16384` (`ollama create <name> -f Modelfile`).

[Widget agent conformance](../references/widget-agent-conformance.md) lists how models did on the widget agent's
test tasks, and shows how to run the tasks against your own model. A server on another computer works too, through
the address of Ollama, LM Studio or Custom, but then your data leaves your machine, and the settings show Remote
address.

## Share the GPU with a training run

On a Mac, a local model uses the same GPU and the same memory as a training run that streams into Caliper. Both slow
down, and if the model and the run do not fit in memory together, the whole machine slows as it starts swapping. While
a live run is attached and the widget agent's provider is a server on this machine, Caliper shows **Shared GPU**:
under the presets in the settings, and after the model's name in the ⌘K prompt row. It means that the model and the
run take turns on the same GPU and memory:

- Use a smaller or more compressed (quantized) model while a run trains.
- Keep the model's context length modest; the context takes memory too.
- Unload the model when you are done. With Ollama, `ollama stop <model>` unloads it at once, and otherwise a model
  stays loaded for five minutes after its last request.
- Check where the model runs. With Ollama, `ollama ps` shows whether a model is on the GPU or the CPU and with which
  context length.
- Watch the memory pressure in Activity Monitor, and switch to a hosted preset for the length of a long run if it stays
  high.

On a computer with a separate graphics card, the model's weights take video memory that the run may need, and the
same advice holds. The side chat runs your own agent, so it shows no such note.

## Local models in the side chat

The side chat runs your own agent, so it uses whatever model your agent is set up for, local models included.
ACP integration (`design/acp-integration.md`) sets up opencode with a model served by Ollama as a worked example.

## Remove the key

In the settings, choose the preset and then **Remove**. The Key row then says Not set, the agent that was started with
the key stops at once, and ⌘K offers only Setup and Ask in chat until you enter a key again.

The key also appears in the Keychain Access app as an item of the service `caliper.agent-keys`, and deleting it there
has the same effect. If you think someone else has seen the key, also revoke it on the service's API keys page, so it
stops working everywhere.
