"""The `caliper-agent` command: starts the sidecar on stdio.

    caliper-agent [--api-file PATH] [--command-table PATH] [--widget-scopes PATH]
                  [--tools LIST]
                  [--preset ID] [--provider claude|openai-compatible] [--base-url URL]
                  [--model ID] [--effort LEVEL] [--model-tools yes|no]
                  [--model-vision yes|no] [--context-length TOKENS]
                  [--cache-dir PATH] [--no-compaction]
                  [--usage-log PATH] [--version]
    caliper-agent probe --model ID [--preset ID] [--provider ...] [--base-url URL] [--cache-dir PATH]
                  [--refresh] [--list-models] [--no-vision-probe] [--set NAME=VALUE ...]

`caliper-agent probe` checks what a model can do and prints one JSON object
(probe.py, capabilities.py); the app's settings run it when the person picks a
model. The sidecar reads its cached answers from `--cache-dir` at start.

The app starts it as a child process with the key in the environment
(ANTHROPIC_API_KEY for the Claude API, CALIPER_MODEL_API_KEY for an
OpenAI-compatible server that needs one) and talks ACP on its stdin and stdout.
What happens at start, in this order:

1. The protocol keeps the real stdout; file descriptor 1 is pointed at stderr,
   so a stray print from any library cannot corrupt the protocol stream.
2. Uncaught exceptions and log records are routed through the scrubber.
3. The keys are read and removed from the environment, with every variable
   that could redirect an SDK (before the SDKs are imported, since they read
   some of them on import).
4. The command table is read and the tools derived from it, with the widget
   scope rule (widget_scopes.json beside the table) for widget sessions.
5. ACP is served until the client closes stdin.

With CALIPER_AGENT_OFFLINE=1 (offline.py), which every test and tool of the
repository sets, a socket guard refuses every connection off loopback from
the start, and a provider that would reach another host is never built: the
sidecar still serves ACP, and answers `session/new` and `session/prompt` with
an error whose `data.code` is `agent.offline`.

Exit codes: 0 after the client closed the connection; 2 for bad usage or an
unreadable command table.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from . import __version__
from .keyguard import KEY_ENV, MODEL_KEY_ENV, SCRUBBER, KeyStatus, install_guards, take_key_from_environment

EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _default_command_table() -> Path | None:
    here = Path(__file__).resolve().parent
    for candidate in (here / "command_table.json", here.parent.parent / "commands" / "command_table.json"):
        if candidate.is_file():
            return candidate
    return None


def _protocol_streams():
    """The protocol's stdin and stdout; fd 1 is then pointed at stderr."""
    reader = sys.stdin.buffer
    out_fd = os.dup(sys.stdout.fileno())
    writer = os.fdopen(out_fd, "wb", buffering=0)
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    return reader, writer


def _context_length(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("a context length is a whole number of tokens") from None
    if not 0 < value <= 100_000_000:
        raise argparse.ArgumentTypeError("a context length is between 1 and 100,000,000 tokens")
    return value


def _yes_no(value: str | None) -> bool | None:
    return None if value is None else value == "yes"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "probe":
        return probe_main(argv[1:])
    parser = argparse.ArgumentParser(prog="caliper-agent",
                                     description="The widget agent of Caliper, an ACP agent on stdio.")
    parser.add_argument("--api-file", help="Caliper API's file (default: api.json in the app-data folder)")
    parser.add_argument("--command-table", help="the command descriptor table as JSON")
    parser.add_argument("--widget-scopes", help="the widget agent's scope rule (default: widget_scopes.json beside "
                                                "the command table)")
    parser.add_argument("--tools", help="only these commands, as a comma-separated list of names and groups")
    parser.add_argument("--preset", help="a provider preset of presets.json: anthropic (Claude), openai, gemini, "
                                         "openrouter, ollama, lmstudio or custom")
    parser.add_argument("--provider", choices=("claude", "openai-compatible"),
                        help="where model requests go (default: the preset's, else claude, the Claude API)")
    parser.add_argument("--base-url", help="the OpenAI-compatible server, such as http://127.0.0.1:11434/v1")
    parser.add_argument("--model", help="the model of new sessions (default: claude-opus-5-5 on the Claude API)")
    parser.add_argument("--effort", choices=EFFORTS, help="the effort of new sessions (default: high)")
    parser.add_argument("--model-tools", choices=("yes", "no"),
                        help="whether the OpenAI-compatible model calls tools (default: the cached probe, else yes "
                             "until the server says no)")
    parser.add_argument("--model-vision", choices=("yes", "no"),
                        help="whether the OpenAI-compatible model reads images (default: the cached probe, else yes "
                             "until the server says no)")
    parser.add_argument("--context-length", type=_context_length,
                        help="the model's context length in tokens, the person's setting (default: the cached probe, "
                             "else 8192 for an OpenAI-compatible model)")
    parser.add_argument("--cache-dir", help="the folder of the capability cache (model-capabilities.json)")
    parser.add_argument("--no-compaction", action="store_true", help="turn server-side compaction off")
    parser.add_argument("--usage-log", help="append the token counts of each request to this file (JSON Lines)")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    args = parser.parse_args(argv)
    if args.version:
        print(f"caliper-agent {__version__}")
        return 0

    reader, writer = _protocol_streams()
    install_guards()
    from .offline import OfflineError, check_provider, enter_offline
    enter_offline()   # with the switch: no proxy variables, and the socket guard
    claude_key, claude_status = take_key_from_environment(name=KEY_ENV)
    model_key, model_status = take_key_from_environment(name=MODEL_KEY_ENV)
    SCRUBBER.register(claude_key)
    SCRUBBER.register(model_key)

    # Imported after the environment is cleaned: the SDKs read some variables on import.
    from .acp import Connection
    from .agent import Agent
    from .caliper_api import CaliperApi
    from .capabilities import ProbeCache, claude_facts, resolve
    from .claude import API_BASE_URL, DEFAULT_EFFORT, DEFAULT_MODEL, MODELS, ClaudeProvider
    from .loop import build_agent
    from .ollama_native import native_context
    from .openai_compat import BaseUrlError, OpenAICompatProvider
    from .presets import PresetError, choose
    from .prompt import SYSTEM_PROMPT
    from .scripted import SCRIPT_ENV, ScriptedProvider, load_script
    from .stub_transport import StubTransport
    from .tools import TableError, derive_tools, load_command_table, name_filter, not_for_agents
    from .widget import ScopeRule, WidgetError, default_scope_file

    table = Path(args.command_table) if args.command_table else _default_command_table()
    if table is None:
        print("caliper-agent: no command table; pass --command-table.", file=sys.stderr)
        return 2
    try:
        commands = load_command_table(table)
        tools = derive_tools(commands, name_filter(args.tools) if args.tools else None)
    except TableError as exc:
        print(f"caliper-agent: {exc}", file=sys.stderr)
        return 2
    scope_file = Path(args.widget_scopes) if args.widget_scopes else default_scope_file(table)
    scope_rule = None
    if scope_file is not None:
        try:
            scope_rule = ScopeRule.load(scope_file)
        except WidgetError as exc:
            print(f"caliper-agent: {exc}", file=sys.stderr)
            return 2
    else:
        print("caliper-agent: no widget scope file; widget sessions are refused.", file=sys.stderr)

    script = None
    script_path = os.environ.get(SCRIPT_ENV, "")
    if script_path:
        try:
            script = load_script(script_path)
        except (OSError, ValueError) as exc:
            print(f"caliper-agent: the test script cannot be read: {exc}", file=sys.stderr)
            return 2
    backoff = float(script.get("backoff_scale", 1.0)) if script is not None else 1.0

    cache = ProbeCache(args.cache_dir) if args.cache_dir else None
    for warning in (cache.warnings if cache is not None else []):
        print(f"caliper-agent: {warning}", file=sys.stderr)

    try:
        choice = choose(args.preset, args.provider, args.base_url)
    except (PresetError, BaseUrlError) as exc:
        print(f"caliper-agent: {exc}", file=sys.stderr)
        return 2
    for warning in choice.warnings():
        print(f"caliper-agent: {warning['message']}", file=sys.stderr)

    provider = None
    refusal: OfflineError | None = None
    if choice.api == "claude":
        model_ids = tuple(m.id for m in MODELS)
        if args.model is not None and args.model not in model_ids:
            print(f"caliper-agent: unknown model {args.model}; choose one of {', '.join(model_ids)}.",
                  file=sys.stderr)
            return 2
        key_status, key_required, default_model = claude_status, True, args.model or DEFAULT_MODEL
        if claude_status == KeyStatus.PRESENT:
            if script is not None and script.get("route", "function") == "function":
                provider = ScriptedProvider(script, compaction=not args.no_compaction)
            else:
                try:
                    # ClaudeProvider refuses the Claude API itself when offline (offline.check_provider).
                    provider = ClaudeProvider(claude_key, compaction=not args.no_compaction, backoff_scale=backoff,
                                              transport=StubTransport(script) if script is not None else None,
                                              facts=claude_facts(cache, API_BASE_URL, args.context_length))
                except OfflineError as exc:
                    refusal = exc
    else:
        if not args.model:
            print("caliper-agent: an OpenAI-compatible server needs --model.", file=sys.stderr)
            return 2
        key_status, key_required, default_model = model_status, False, args.model
        base_url = choice.base_url
        try:
            check_provider("openai-compatible", base_url=base_url)
            entry = cache.get(base_url, args.model) if cache is not None else None
            known = resolve(entry, options={"tools": _yes_no(args.model_tools), "vision": _yes_no(args.model_vision),
                                            "context_length": args.context_length})
            context, context_source = known.context_length.value, known.context_length.source
            num_ctx = native_context(choice.preset, context, context_source)
            if num_ctx is not None:
                # Ollama loads the model with this context length, so it is the model's.
                context = num_ctx
                if context_source not in ("override", "option"):
                    context_source = "preset"
            provider = OpenAICompatProvider(base_url, args.model,
                                            model_key if model_status == KeyStatus.PRESENT else None,
                                            tools=bool(known.tools.value), vision=bool(known.vision.value),
                                            backoff_scale=backoff, context_length=context,
                                            sources={"tools": known.tools.source, "vision": known.vision.source,
                                                     "context_length": context_source},
                                            quirks=known.quirks,
                                            native={"num_ctx": num_ctx} if num_ctx is not None else None)
        except OfflineError as exc:
            refusal = exc
    if refusal is not None:
        print(f"caliper-agent: {refusal}", file=sys.stderr)

    graph = build_agent(tools, SYSTEM_PROMPT)

    async def serve() -> None:
        conn = Connection(reader, writer)
        api = CaliperApi(Path(args.api_file) if args.api_file else None, not_for_agents=not_for_agents(commands))
        agent = Agent(conn, provider, key_status, tools, graph, api, default_model=default_model,
                      default_effort=args.effort or DEFAULT_EFFORT, usage_log=args.usage_log,
                      key_required=key_required, commands=commands, scope_rule=scope_rule, refusal=refusal,
                      provider_meta=choice.as_meta(), widget_model=args.model, widget_effort=args.effort)
        try:
            await conn.serve(agent.on_request, agent.on_notification)
        finally:
            await agent.aclose()

    asyncio.run(serve())
    return 0


# ---- caliper-agent probe ------------------------------------------------------------------------------


def _parse_override(text: str) -> tuple[str, object]:
    name, sep, value = text.partition("=")
    name, value = name.strip(), value.strip().lower()
    if not sep or name not in ("tools", "vision", "context_length"):
        raise ValueError(f"--set takes tools=..., vision=... or context_length=..., not {text[:40]}.")
    if value == "auto":
        return name, None
    if name == "context_length":
        try:
            return name, _context_length(value)
        except argparse.ArgumentTypeError as exc:
            raise ValueError(f"context_length: {exc}.") from None
    if value not in ("yes", "no"):
        raise ValueError(f"{name} is yes, no or auto.")
    return name, value == "yes"


def probe_main(argv: list[str]) -> int:
    """`caliper-agent probe`: prints one JSON object on stdout; 2 for bad usage."""
    import json

    parser = argparse.ArgumentParser(prog="caliper-agent probe",
                                     description="Checks what a model can do (tool calling, vision, context length) "
                                                 "and prints one JSON object.")
    parser.add_argument("--preset", help="a provider preset of presets.json")
    parser.add_argument("--provider", choices=("claude", "openai-compatible"))
    parser.add_argument("--base-url", help="the OpenAI-compatible server")
    parser.add_argument("--model", required=True, help="the model to check")
    parser.add_argument("--cache-dir", help="the folder of the capability cache")
    parser.add_argument("--refresh", action="store_true", help="probe again even when the cache has a fresh answer")
    parser.add_argument("--list-models", action="store_true", help="only list the server's models, with no test call")
    parser.add_argument("--no-vision-probe", action="store_true", help="leave the vision test call out")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="the person's override, kept in the cache: tools=yes|no|auto, vision=yes|no|auto, "
                             "context_length=<tokens>|auto")
    args = parser.parse_args(argv)

    install_guards()
    from .offline import enter_offline
    enter_offline()   # with the switch: no proxy variables, and the socket guard
    claude_key, claude_status = take_key_from_environment(name=KEY_ENV)
    model_key, model_status = take_key_from_environment(name=MODEL_KEY_ENV)
    SCRUBBER.register(claude_key)
    SCRUBBER.register(model_key)

    from .capabilities import DEFAULT_CONTEXT_LENGTH, CacheError, Entry, ProbeCache, resolve
    from .claude import API_BASE_URL
    from .presets import BaseUrlError, PresetError, choose
    from .ollama_native import native_context
    from .probe import now_iso, probe_claude, probe_openai_compatible
    from .scripted import SCRIPT_ENV, load_script
    from .stub_transport import StubTransport

    def fail(message: str) -> int:
        print(SCRUBBER.scrub(f"caliper-agent probe: {message}"), file=sys.stderr)
        return 2

    try:
        choice = choose(args.preset, args.provider, args.base_url)
    except (PresetError, BaseUrlError) as exc:
        return fail(str(exc))
    kind = choice.api
    base_url = choice.base_url or API_BASE_URL
    try:
        overrides = [_parse_override(item) for item in args.set]
    except ValueError as exc:
        return fail(str(exc))
    if overrides and not args.cache_dir:
        return fail("--set keeps the override in the cache, so it needs --cache-dir.")

    cache = ProbeCache(args.cache_dir) if args.cache_dir else None
    warnings: list[str] = list(cache.warnings) if cache is not None else []
    try:
        for name, value in overrides:
            cache.set_override(base_url, args.model, name, value)
    except CacheError as exc:
        return fail(str(exc))

    transport = None
    script_path = os.environ.get(SCRIPT_ENV, "")
    if script_path and kind == "claude":
        try:
            script = load_script(script_path)
        except (OSError, ValueError) as exc:
            return fail(f"the test script cannot be read: {exc}")
        transport = StubTransport(script)

    # The Ollama preset's test calls load the model with the context length the
    # agent will use: the person's override, else the preset's.
    over = cache.get(base_url, args.model) if cache is not None else None
    over_context = over.overrides.get("context_length") if over is not None else None
    num_ctx = native_context(choice.preset, over_context, "override" if over_context is not None else "default")
    native = {"num_ctx": num_ctx} if num_ctx is not None else None

    async def run_probe(list_only: bool):
        if kind == "claude":
            return await probe_claude(args.model, claude_key if claude_status == KeyStatus.PRESENT else None,
                                      transport=transport, list_only=list_only)
        return await probe_openai_compatible(base_url, args.model,
                                             model_key if model_status == KeyStatus.PRESENT else None,
                                             preset=choice.preset.id if choice.preset is not None else None,
                                             vision=not args.no_vision_probe, list_only=list_only, native=native)

    out: dict = {"version": 1, "preset": choice.preset.id if choice.preset is not None else None,
                 "provider": kind, "base_url": base_url, "model": args.model}
    warnings += [w["message"] for w in choice.warnings()]
    if args.list_models:
        result = asyncio.run(run_probe(True))
        out.update({"models": result.models, "errors": result.errors, "warnings": warnings})
        print(SCRUBBER.scrub(json.dumps(out)))
        return 0

    entry = cache.get(base_url, args.model) if cache is not None else None
    result = None
    if overrides and not args.refresh:
        pass  # only the overrides change; no request
    elif entry is not None and not entry.stale() and not args.refresh:
        pass  # the cached answer is fresh
    else:
        result = asyncio.run(run_probe(False))
        entry = cache.put_result(result) if cache is not None else Entry.from_result(result)
    if cache is not None and (result is not None or overrides):
        cache.save()
        warnings += [w for w in cache.warnings if w not in warnings]
    known = resolve(entry, default_context=DEFAULT_CONTEXT_LENGTH if kind == "openai-compatible" else None)
    if kind == "claude":
        known.tools.value, known.tools.source = True, "known"
    context = known.context_length.value
    if isinstance(context, int) and context < DEFAULT_CONTEXT_LENGTH:
        warnings.append("The model's context length is below 8,192 tokens, so requests may not fit.")
    probed = None
    if entry is not None and entry.probed_at is not None:
        probed = {name: entry.result[name].as_dict() for name in ("tools", "vision", "context_length")
                  if name in entry.result}
    out.update({
        "models": result.models if result is not None else None,
        "probed_at": now_iso(entry.probed_at) if entry is not None and entry.probed_at is not None else None,
        "from_cache": result is None and entry is not None and entry.probed_at is not None,
        "stale": entry.stale() if entry is not None else True,
        "capabilities": known.as_dict(),
        "can_drive_caliper": known.tools.value is not False,
        "probe": probed,
        "overrides": dict(entry.overrides) if entry is not None else {},
        "quirks": dict(entry.result.get("quirks", {})) if entry is not None else {},
        "errors": result.errors if result is not None else [],
        "warnings": warnings,
    })
    print(SCRUBBER.scrub(json.dumps(out)))
    return 0
