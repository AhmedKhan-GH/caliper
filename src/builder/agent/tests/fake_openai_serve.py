"""The fake OpenAI-compatible server of fake_openai.py as a process, for the app's tests and captures.

The C++ tests and caliper-agent-settings-shot cannot import fake_openai.py, so
they start this script with the test interpreter: it serves on a free port of
127.0.0.1, prints its base URL on one line, and stops when its standard input
closes. Nothing leaves the machine and no key is used.

    python -I fake_openai_serve.py [--models a,b] [--context N] [--refuse-tools] [--blind] [--record FILE]

With --record, each request appends one JSON line to FILE: its method, its
path and its Authorization header (null when it has none), so that a test can
check which credential, if any, reached the server.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fake_openai import FakeOpenAI  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="", help="the model ids GET /v1/models lists, comma-separated")
    parser.add_argument("--context", type=int, help="the context_length of each listed model")
    parser.add_argument("--refuse-tools", action="store_true", help="refuse every request with tools")
    parser.add_argument("--blind", action="store_true", help="accept images but never name their colour")
    parser.add_argument("--record", help="a file that gets one JSON line per request")
    args = parser.parse_args()
    models = [{"id": m, **({"context_length": args.context} if args.context else {})}
              for m in args.models.split(",") if m] or None
    server = FakeOpenAI(models=models, refuse_tools=args.refuse_tools, vision_mode="blind" if args.blind else "see",
                        record=args.record)
    print(server.base_url, flush=True)
    try:
        sys.stdin.read()
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
