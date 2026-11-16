"""The `caliper-agent` command: starts the sidecar on stdio.

    caliper-agent [--api-file PATH] [--command-table PATH] [--widget-scopes PATH]
                  [--tools LIST]
                  [--provider claude|openai-compatible] [--base-url URL]
                  [--model ID] [--effort LEVEL] [--model-tools yes|no]
                  [--model-vision yes|no] [--no-compaction]
                  [--usage-log PATH] [--version]

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="caliper-agent",
                                     description="The widget agent of Caliper, an ACP agent on stdio.")
    parser.add_argument("--api-file", help="Caliper API's file (default: api.json in the app-data folder)")
    parser.add_argument("--command-table", help="the command descriptor table as JSON")
    parser.add_argument("--widget-scopes", help="the widget agent's scope rule (default: widget_scopes.json beside "
                                                "the command table)")
    parser.add_argument("--tools", help="only these commands, as a comma-separated list of names and groups")
    parser.add_argument("--provider", choices=("claude", "openai-compatible"), default="claude",
                        help="where model requests go (default: claude, the Claude API)")
    parser.add_argument("--base-url", help="the OpenAI-compatible server, such as http://127.0.0.1:11434/v1")
    parser.add_argument("--model", help="the model of new sessions (default: claude-opus-5-5 on the Claude API)")
    parser.add_argument("--effort", choices=EFFORTS, help="the effort of new sessions (default: high)")
    parser.add_argument("--model-tools", choices=("yes", "no"), default="yes",
                        help="whether the OpenAI-compatible model calls tools (default: yes, until the server says no)")
    parser.add_argument("--model-vision", choices=("yes", "no"), default="yes",
                        help="whether the OpenAI-compatible model reads images (default: yes, until the server says no)")
    parser.add_argument("--no-compaction", action="store_true", help="turn server-side compaction off")
    parser.add_argument("--usage-log", help="append the token counts of each request to this file (JSON Lines)")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    args = parser.parse_args(argv)
    if args.version:
        print(f"caliper-agent {__version__}")
        return 0

    reader, writer = _protocol_streams()
    install_guards()
    from .offline import OfflineError, check_provider, install_socket_guard, offline
    if offline():
        install_socket_guard()
    claude_key, claude_status = take_key_from_environment(name=KEY_ENV)
    model_key, model_status = take_key_from_environment(name=MODEL_KEY_ENV)
    SCRUBBER.register(claude_key)
    SCRUBBER.register(model_key)

    # Imported after the environment is cleaned: the SDKs read some variables on import.
    from .acp import Connection
    from .agent import Agent
    from .caliper_api import CaliperApi
    from .claude import DEFAULT_EFFORT, DEFAULT_MODEL, MODELS, ClaudeProvider
    from .loop import build_agent
    from .openai_compat import BaseUrlError, OpenAICompatProvider
    from .prompt import SYSTEM_PROMPT
    from .scripted import SCRIPT_ENV, ScriptedProvider, load_script
    from .stub_transport import StubTransport
    from .tools import TableError, derive_tools, load_command_table, name_filter
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

    provider = None
    refusal: OfflineError | None = None
    if args.provider == "claude":
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
                    check_provider("claude", stubbed=script is not None)
                    provider = ClaudeProvider(claude_key, compaction=not args.no_compaction, backoff_scale=backoff,
                                              transport=StubTransport(script) if script is not None else None)
                except OfflineError as exc:
                    refusal = exc
    else:
        if not args.base_url or not args.model:
            print("caliper-agent: --provider openai-compatible needs --base-url and --model.", file=sys.stderr)
            return 2
        key_status, key_required, default_model = model_status, False, args.model
        try:
            check_provider("openai-compatible", base_url=args.base_url)
            provider = OpenAICompatProvider(args.base_url, args.model,
                                            model_key if model_status == KeyStatus.PRESENT else None,
                                            tools=args.model_tools == "yes", vision=args.model_vision == "yes",
                                            backoff_scale=backoff)
        except BaseUrlError as exc:
            print(f"caliper-agent: {exc}", file=sys.stderr)
            return 2
        except OfflineError as exc:
            refusal = exc
    if refusal is not None:
        print(f"caliper-agent: {refusal}", file=sys.stderr)

    graph = build_agent(tools, SYSTEM_PROMPT)

    async def serve() -> None:
        conn = Connection(reader, writer)
        api = CaliperApi(Path(args.api_file) if args.api_file else None)
        agent = Agent(conn, provider, key_status, tools, graph, api, default_model=default_model,
                      default_effort=args.effort or DEFAULT_EFFORT, usage_log=args.usage_log,
                      key_required=key_required, commands=commands, scope_rule=scope_rule, refusal=refusal)
        try:
            await conn.serve(agent.on_request, agent.on_notification)
        finally:
            await agent.aclose()

    asyncio.run(serve())
    return 0
