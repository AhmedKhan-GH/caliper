---
kind: reference
---

# Widget Agent Startup

This page lists the options, presets, checks and context limits that Caliper uses when it starts `caliper-agent`.

## Agent Startup

Caliper starts the agent as a child process and speaks ACP version 1 on its standard input and output. The app finds the
Python runtime through `CALIPER_AGENT_PYTHON` ([Environment variables](../reference/folders-and-environment.md)).

```sh
<python> -I -m caliper_agent [--api-file <path>] [--command-table <path>] [--widget-scopes <path>] [options]
```

| Option | Meaning |
|---|---|
| `--api-file <path>` | Caliper API's file, which gives the loopback address and a token. The app gives its sidecar a file of its own, and that token has the origin `agent`. A session whose `session/new` hands over Caliper's MCP server uses that server's file instead ([In the side chat](widget-agent.md#side-chat-use)) |
| `--command-table <path>` | The command table as JSON |
| `--widget-scopes <path>` | The scope rule (see [Tool Scope](widget-agent.md#tool-scope)). By default, `widget_scopes.json` beside the command table. Without it, the agent refuses widget sessions |
| `--preset <id>` | A provider preset, which fills the provider and the base URL ([Presets](#presets)) |
| `--provider claude\|openai-compatible`, `--base-url <url>`, `--model <id>`, `--effort <level>` | Where model requests go, and the defaults of a session in the general role |
| `--cache-dir <path>` | The folder of the capability cache that `caliper-agent probe` writes. The agent reads its answer for the model at start |
| `--model-tools yes\|no`, `--model-vision yes\|no`, `--context-length <tokens>` | What the model can do, when Caliper knows it. Each option wins over the cache |
| `--widget-requests <n>` | The model requests that a widget prompt may make, from 1 to 20. The default is 6 (see Limits in [Answers](widget-session-protocol.md#answers)) |

The Claude API key reaches the agent in `ANTHROPIC_API_KEY`. The key for an OpenAI-compatible server reaches it in
`CALIPER_MODEL_API_KEY`. A key never appears in the arguments. The option `-I` keeps the workspace folder off Python's
import path.

### Presets

Caliper's agent and key store share one preset table:

| Id | Preset | Base URL | Key |
|---|---|---|---|
| `anthropic` | Claude | `https://api.anthropic.com`, fixed | Required, in `ANTHROPIC_API_KEY` |
| `openai` | OpenAI | `https://api.openai.com/v1` | Required, in `CALIPER_MODEL_API_KEY` |
| `gemini` | Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | Required, in `CALIPER_MODEL_API_KEY` |
| `openrouter` | OpenRouter | `https://openrouter.ai/api/v1` | Required, in `CALIPER_MODEL_API_KEY` |
| `ollama` | Ollama | `http://localhost:11434/v1`, or another host with `--base-url` | None |
| `lmstudio` | LM Studio | `http://localhost:1234/v1`, or another host with `--base-url` | None |
| `custom` | Custom | `--base-url`, required | Optional, in `CALIPER_MODEL_API_KEY` |

The person always chooses the model. The Claude, OpenAI, Gemini and OpenRouter presets take no other base URL, so a
preset's key goes only to that preset's host. A base URL must be an http or https address without a user name, password,
query or fragment. Any other address makes the agent exit with 2. The answer to `initialize` names the choice, with the
warnings that the pane should show:

```json
{"_meta": {"caliper": {"provider": {"preset": "lmstudio", "api": "openai-compatible",
  "base_url": "http://192.0.2.10:1234/v1", "host": "192.0.2.10", "loopback": false,
  "warnings": [{"code": "remote_host", "message": "..."}, {"code": "unencrypted", "message": "..."}]}}}}
```

The agent gives `remote_host` for a local preset or a Custom entry on a host that is not loopback, because prompts and
workspace data then leave the machine. It gives `unencrypted` for plain http to such a host.

The agent scrubs the key itself from everything it writes. It also scrubs any text shaped like a Claude API key
(`sk-ant-`), an OpenRouter key (`sk-or-v1-` and 64 hexadecimal digits) or a Google API key (`AIza` and 35 characters).

If `CALIPER_AGENT_OFFLINE=1` is set, the agent reaches no model server off loopback. It answers `session/new` and
`session/prompt` with an error whose `data.code` is `agent.offline`. The exceptions are a scripted model, a stubbed
transport and a server on a loopback address.

## Model Checks

`caliper-agent probe` checks what a model can do and prints one JSON object on standard output. Caliper's settings run it
when the person picks a model:

```sh
<python> -I -m caliper_agent probe --preset ollama --model <id> --cache-dir <settings folder>
```

For an OpenAI-compatible server, the probe lists the server's models. It then makes two streamed requests. One asks the
model to call a tool with fixed arguments, and one asks it to name the color of a small PNG. Each request has at most
1,024 output tokens and 60 seconds. The probe reads the context length where the server reports it.

For a server on this machine, the probe also asks Ollama's `POST /api/show` for the model's metadata, which loads
nothing. It keeps the bytes of context cache that one token takes as the quirk `kv_bytes_per_token`. A server that is not
Ollama answers 404, and the probe keeps nothing. The metadata's context length is the most the model takes, not what the
server runs it with, so the probe does not use it. After the test requests load the model, the probe reads Ollama's
`GET /api/ps`, which states the context in force. If that fails, the context length stays unknown.

For the Ollama preset, the two test requests go to Ollama's own chat API with the preset's context length, as the agent's
requests do (see [Ollama's own chat API](#ollama-chat-api)). That context length is the answer, with the source
`preset`. For Claude, the probe reads the Models API and makes no request that costs tokens. The key comes from the
environment, as for the agent.

| Option | Meaning |
|---|---|
| `--preset <id>`, `--provider`, `--base-url` | The provider, as for the agent |
| `--cache-dir <path>` | Keep the answer in `model-capabilities.json` in this folder, and answer from it while it is fresh (30 days) |
| `--refresh` | Probe again even when the cache has a fresh answer |
| `--list-models` | Only list the server's models, with no test request |
| `--no-vision-probe` | Leave the image request out |
| `--set tools=yes\|no\|auto`, `--set vision=yes\|no\|auto`, `--set context_length=<tokens>\|auto` | The person's override, kept in the cache until set to `auto`. No request is made |

| Field | Meaning |
|---|---|
| `models` | The server's models, or `null` when it lists none or the answer came from the cache |
| `capabilities` | `tools`, `vision` and `context_length`, each `{"value", "source", "detail"}`: what the agent will assume, with `source` one of `probe`, `server`, `known`, `preset`, `override`, `default` and `unknown` |
| `can_drive_caliper` | `false` when the model cannot call tools |
| `probe`, `overrides`, `quirks` | The cached probe answers, the person's overrides, and what the server needed (`strict`, `max_tokens_field`) or showed (`reasoning_field`, `streamed_arguments`, and `kv_bytes_per_token` from Ollama's metadata) |
| `probed_at`, `from_cache`, `stale` | When the probe ran, and whether this answer came from the cache |
| `errors`, `warnings` | Why an answer is unknown (such as `probe.unreachable`, `probe.key_rejected`, `probe.timeout`, or `probe.cut_off` for a reply that reached the output limit first), and notes such as a context length below 8,192 tokens |

The command exits with 0 when it prints its answer and with 2 for bad usage. The base URL rule is the same as for the
agent.

## Model Assumptions

For each capability, the agent takes the first of these at start: an option on its command line, the person's override in
the cache, a fresh probe answer, or its default. The default assumes tool calling and vision until the server refuses
them. For an OpenAI-compatible model, it assumes a context length of 8,192 tokens. For Claude, the context length is
unknown.

A model without tool calling cannot drive Caliper. For such a model, `session/new` fails with error -32602 and
`data.code` `agent.no_tools`. The error names the model and the source of the answer, and it has a message that the pane
can show. Every other `session/new` answer carries the model's facts. Caliper gives them to `build_context_bundle()` as
the context length and whether to send a render:

```json
{"_meta": {"caliper": {"model": {"id": "example-model", "tools": true, "vision": false, "context_length": 32768,
                                  "sources": {"tools": "probe", "vision": "probe", "context_length": "override"}}}}}
```

A model without vision gets no image and no `panel.render`. When the context length is known, a request asks for at most
half of it as output.

## Ollama Chat API

Ollama's OpenAI-compatible endpoint takes no context length. It ignores `options.num_ctx`, so Ollama loads the model at
the context length that it chooses itself. For a model with a very long context, the context cache alone can need a
large amount of memory. The Ollama preset therefore sends its chat requests to Ollama's own `/api/chat`. Each request
has `options.num_ctx` and a `keep_alive` of 60 seconds. As a result, the model loads with that context length and
unloads a minute after its last request. The list of models still comes from `/v1/models`. Every other preset, LM Studio
and Custom included, stays on the OpenAI-compatible endpoint.

| Role | Context length (`num_ctx`) | The person's setting |
|---|---|---|
| The widget agent (⌘K) | 16,384 tokens | Context length in the widget agent's settings (an override in the cache), or `--context-length` |
| The general role (the side chat's `caliper-agent`) | 32,768 tokens | `--context-length` on its command line |

The defaults cover the first request of each role. The general role needs more room because it offers every command of
the table as a tool, and its context grows with each tool result.

Where the agent chose the context length, it never lets the server cut a request. Each request asks Ollama to refuse a
prompt that is too long (`truncate: false`) and not to drop the front of the context during a reply (`shift: false`).
Before each model request, the agent estimates the prompt tokens. It starts from the count that the server gave for the
last request and adds what changed since, so the estimate is at or above the server's count. If the request would leave
less than an eighth of the context (at most 4,096 tokens) for the reply, the agent sends nothing. Otherwise, it asks for
no more output than the context leaves.

A request that does not fit, by the estimate or by the server's refusal, ends the turn with the stop reason
`max_tokens`, which Caliper shows as LIMIT. The turn also carries a message with the status `context_full`. The message
says that the conversation no longer fits the model's context and that the person can start again or raise the context
length. A reply that stops at the end of the context also ends the turn with `max_tokens`, and none of its tool calls
runs. For a server whose context length the agent did not choose, the agent checks nothing.
