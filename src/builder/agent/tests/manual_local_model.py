"""A manual, opt-in check: the dashboard prompt against a local model server.

Not part of ctest or pytest (the name does not start with test_). It needs a
model server that is already running with a model that calls tools, such as
Ollama (`http://127.0.0.1:11434/v1`) or LM Studio (`http://127.0.0.1:1234/v1`);
it never starts a server or pulls a model. It makes a fresh workspace in
`--workdir` with the two data files of the dashboard test, starts
`caliper-builder-host` on it, runs `caliper-acp-harness` with the sidecar on the
OpenAI-compatible route and the dashboard prompt (allowing every permission
request), and prints what happened. The transcript is kept in the workdir.

    python -I tests/manual_local_model.py --host-exe <build>/session/caliper-builder-host \
        --harness <build>/acp/caliper-acp-harness --base-url http://127.0.0.1:11434/v1 \
        --model <model> --workdir <empty folder>
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import struct
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent
COMMAND_TABLE = AGENT_DIR.parent / "commands" / "command_table.json"

PROMPT = (
    "Import the files data/run42_metrics.parquet and data/attn.safetensors and run their loaders. "
    "Then make a new spec with the id run42-dashboard and the title Run 42. Give it a source metrics from the "
    "table that the parquet loader made, and a source attn from the attention tensor, with three transforms: "
    "a slice of axis 0 at index 0, a mean over axis 0, and a minmax normalise. Add three panels: a series panel "
    "curves with x metrics.step and y metrics.loss and metrics.accuracy; a heatmap panel attention bound to attn "
    "with the magma colormap; and a table panel rows bound to metrics that shows the columns step and loss. Lay "
    "them out with curves and attention side by side above rows. Then save the applet."
)


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=3) as response:  # loopback only
        return json.loads(response.read())


def check_server(base_url: str, model: str) -> str:
    models = [m.get("id") for m in _get(base_url.rstrip("/") + "/models").get("data", [])]
    if model not in models:
        raise SystemExit(f"the server does not list the model {model}; it lists {models}")
    if ":11434" in base_url:
        root = base_url.rsplit("/v1", 1)[0]
        tags = {m["name"]: m for m in _get(root + "/api/tags").get("models", [])}
        caps = tags.get(model, {}).get("capabilities", [])
        if "tools" not in caps:
            raise SystemExit(f"Ollama says {model} has no tool support ({caps})")
        return f"Ollama, capabilities {caps}"
    return "an OpenAI-compatible server"


def write_files(data: Path) -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    data.mkdir(parents=True, exist_ok=True)
    steps = list(range(10))
    pq.write_table(pa.table({"step": steps, "loss": [1.0 / (s + 1) for s in steps],
                             "accuracy": [s / 10 for s in steps]}), data / "run42_metrics.parquet")
    values = np.arange(96, dtype="<f4")
    header = json.dumps({"weights": {"dtype": "F32", "shape": [2, 3, 4, 4], "data_offsets": [0, values.nbytes]}})
    header += " " * (-len(header) % 8)
    (data / "attn.safetensors").write_bytes(struct.pack("<Q", len(header)) + header.encode() + values.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-exe", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    print("server:", check_server(args.base_url, args.model))
    root = Path(args.workdir).resolve()
    ws = root / "workspace"
    write_files(ws / "data")
    api_dir = root / "api"
    host = subprocess.Popen([args.host_exe, "--workspace", str(ws), "--api-dir", str(api_dir), "--origin", "agent",
                             "--approve", "allow"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(200):
            if (api_dir / "api.json").exists():
                break
            time.sleep(0.05)
        agent = " ".join(shlex.quote(a) for a in [
            sys.executable, "-P", "-m", "caliper_agent", "--provider", "openai-compatible",
            "--base-url", args.base_url, "--model", args.model, "--command-table", str(COMMAND_TABLE),
            "--api-file", str(api_dir / "api.json")])
        env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "OPENAI_", "CALIPER_"))}
        env["PYTHONPATH"] = str(AGENT_DIR)
        # The local server is on loopback; the switch keeps every other host out (offline.py).
        env["CALIPER_AGENT_OFFLINE"] = "1"
        started = time.monotonic()
        proc = subprocess.run([args.harness, "--agent", agent, "--cwd", str(ws), "--env", "PYTHONPATH",
                               "--env", "CALIPER_AGENT_OFFLINE",
                               "--prompt", PROMPT, "--permission", "allow", "--timeout", str(args.timeout),
                               "--stall", str(args.timeout)],
                              capture_output=True, text=True, env=env, timeout=args.timeout + 60)
        elapsed = time.monotonic() - started
    finally:
        host.terminate()
        host.wait(timeout=10)

    (root / "transcript.jsonl").write_text(proc.stdout)
    (root / "harness-stderr.txt").write_text(proc.stderr)
    lines = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    kinds = Counter(line["type"] for line in lines)
    calls: dict[str, str] = {}
    outcome: dict[str, str] = {}
    for line in lines:
        if line["type"] == "tool_call":
            calls[line["tool_call"]["id"]] = line["tool_call"].get("title", "")
        if line["type"] == "tool_call_update" and line["tool_call"].get("status") in ("completed", "failed"):
            outcome[line["tool_call"]["id"]] = line["tool_call"]["status"]
    said = "".join(line.get("text", "") for line in lines if line["type"] == "agent_message_chunk")
    print(f"harness exit {proc.returncode} after {elapsed:.0f} s; messages by type: {dict(kinds)}")
    print("tool calls:", [(calls[c], outcome.get(c, "open")) for c in calls])
    print("result:", [line for line in lines if line["type"] == "result"])
    print("reply:", said[:600])
    saved = ws / "applets" / "run42-dashboard" / "run42-dashboard.spec.toml"
    print("saved spec:", saved.read_text() if saved.exists() else "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
