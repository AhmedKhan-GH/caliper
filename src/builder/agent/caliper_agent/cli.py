"""The `caliper-agent` command: starts the sidecar on stdio.

    caliper-agent [--api-file PATH] [--command-table PATH] [--model ID]
                  [--effort LEVEL] [--no-compaction] [--usage-log PATH] [--version]

The app starts it as a child process with the key in ANTHROPIC_API_KEY and
talks ACP on its stdin and stdout. What happens at start, in this order:

1. The protocol keeps the real stdout; file descriptor 1 is pointed at stderr,
   so a stray print from any library cannot corrupt the protocol stream.
2. Uncaught exceptions and log records are routed through the scrubber.
3. The key is read and removed from the environment, with every variable that
   could redirect the SDK (before the SDK is imported, since it reads some of
   them on import).
4. The command table is read and the tools derived from it.
5. ACP is served until the client closes stdin.

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
from .keyguard import SCRUBBER, KeyStatus, install_guards, take_key_from_environment

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
                                     description="The BYOK agent of Caliper's agent pane, an ACP agent on stdio.")
    parser.add_argument("--api-file", help="Caliper API's file (default: api.json in the app-data folder)")
    parser.add_argument("--command-table", help="the command descriptor table as JSON")
    parser.add_argument("--model", help="the model of new sessions (default: claude-opus-5-5)")
    parser.add_argument("--effort", choices=EFFORTS, help="the effort of new sessions (default: high)")
    parser.add_argument("--no-compaction", action="store_true", help="turn server-side compaction off")
    parser.add_argument("--usage-log", help="append the token counts of each request to this file (JSON Lines)")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    args = parser.parse_args(argv)
    if args.version:
        print(f"caliper-agent {__version__}")
        return 0

    reader, writer = _protocol_streams()
    install_guards()
    key, key_status = take_key_from_environment()
    SCRUBBER.register(key)

    # Imported after the environment is cleaned: the SDK reads some variables on import.
    from .acp import Connection
    from .agent import Agent
    from .caliper_api import CaliperApi
    from .claude import DEFAULT_EFFORT, DEFAULT_MODEL, MODELS, ClaudeProvider
    from .prompt import SYSTEM_PROMPT
    from .stub_transport import transport_from_environment
    from .tools import TableError, derive_tools, load_command_table

    model_ids = tuple(m.id for m in MODELS)
    if args.model is not None and args.model not in model_ids:
        print(f"caliper-agent: unknown model {args.model}; choose one of {', '.join(model_ids)}.", file=sys.stderr)
        return 2
    table = Path(args.command_table) if args.command_table else _default_command_table()
    if table is None:
        print("caliper-agent: no command table; pass --command-table.", file=sys.stderr)
        return 2
    try:
        tools = derive_tools(load_command_table(table))
    except TableError as exc:
        print(f"caliper-agent: {exc}", file=sys.stderr)
        return 2

    try:
        stub = transport_from_environment()
    except (OSError, ValueError) as exc:
        print(f"caliper-agent: the test script cannot be read: {exc}", file=sys.stderr)
        return 2

    def provider_factory():
        if key is None:
            return None
        return ClaudeProvider(key, tools, SYSTEM_PROMPT, transport=stub, compaction=not args.no_compaction,
                              backoff_scale=stub.backoff_scale if stub is not None else 1.0)

    async def serve() -> None:
        conn = Connection(reader, writer)
        api = CaliperApi(Path(args.api_file) if args.api_file else None)
        agent = Agent(conn, provider_factory if key_status == KeyStatus.PRESENT else None, key_status, tools, api,
                      model_ids=model_ids, model_names={m.id: m.name for m in MODELS},
                      default_model=args.model or DEFAULT_MODEL, default_effort=args.effort or DEFAULT_EFFORT,
                      usage_log=args.usage_log)
        try:
            await conn.serve(agent.on_request, agent.on_notification)
        finally:
            await agent.aclose()

    asyncio.run(serve())
    return 0
