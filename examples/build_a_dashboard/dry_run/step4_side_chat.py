"""Step 4 of the walk-through in the hidden app: the side chat with a real agent on a local model.

The side chat runs caliper-agent, Caliper's own ACP agent, on a model served
by Ollama on this machine, so that no account or key is used: its custom command
is `chat-agent.sh` beside this folder, the script a person gives the chat. The app is attached to a live stream, the person's message is "Plot
attention entropy per layer while it trains", every edit the agent sends waits
on an approval card and is approved, and the dashboard the agent builds then
draws the stream live on the spec page beside the chat.

    python3 step4_side_chat.py --build <build folder> --out <folder> [--agent-python <python>] \
        [--work <folder>] [--model qwen3.8:27b] [--base-url http://127.0.0.1:11434/v1] \
        [--context-length 32768] [--runs 1] [--minutes 10] [--size 1440x900]
        [--stream producer|train --train-python <python>]

- `--build` is a build folder of the repository with `caliper`,
  `caliper_attention_producer` and the applets in it.
- `--agent-python` is a Python with caliper-agent's dependencies
  (`src/builder/agent/pyproject.toml`), which chat-agent.sh takes from
  CALIPER_AGENT_PYTHON; the agent runs from this repository's source, so
  nothing is installed. The app gets the PATH of an app started from the
  Finder, the system's folders only, which its agent inherits.
- The stream is `caliper_attention_producer` by default, a test tool that
  stands in for the training script with the same views and no torch; with
  `--stream train` it is `train.py` beside this folder, run by `--train-python`
  (a Python with torch and Caliper LIB).

What the script plays is the person's part, through the hidden app's input
script (CALIPER_TEST_SPEC_UI): it opens the chat, types the message and presses
Enter, waits until the approval strip shows the first card and captures it,
waits for the turn's
end, waits for the spec page, which the app shows for the agent's new spec, and
captures the dashboard twice a few seconds apart. Each approval card is
decided through the hidden app's test-only decisions file
(CALIPER_TEST_APPROVALS), which applies the decision through the same path as
the card's Approve button: a thread here reads the app's `[approval] ... waiting`
lines and approves each card once the first has been captured.

Captures are written to `--out` as step4-<what>-<W>x<H>.png (request, card,
live-a, live-b), from the first run that got there. Every run's files are kept
under `--work`, and the script prints one JSON object per run: how it ended,
the seconds to the first card and to the end of the turn, the approvals, and
the commands the agent's turn sent, from the workspace's journal. It exits with
0 when at least one run built a dashboard with panels.

CALIPER_AGENT_OFFLINE=1 is set for everything it starts, so caliper-agent
reaches no host off this machine. The app never starts a detected agent: the
chat's settings name the custom command.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REQUEST = "Plot attention entropy per layer while it trains"
APPROVAL = re.compile(r"^\[approval\] (\S+) (\S+) \((\w+)\) (.*)$")


def loopback(url: str) -> bool:
    return (urlsplit(url).hostname or "") in ("127.0.0.1", "localhost", "::1")


def check_model(base_url: str, model: str) -> str:
    """'' when the server lists the model, else why not. A list, not a model call."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(base_url.rstrip("/") + "/models", timeout=5) as r:
            ids = [m.get("id") for m in json.loads(r.read()).get("data", [])]
    except OSError as exc:
        return f"the model server at {base_url} does not answer ({exc})"
    return "" if model in ids else f"the model server does not list {model}; it lists {', '.join(map(str, ids))}"


class Log:
    """The app's standard error, written to a file and read line by line, with
    the time each line came."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines: list[tuple[float, str]] = []
        self.lock = threading.Lock()

    def follow(self, stream, start: float) -> None:
        with self.path.open("w") as out:
            for raw in stream:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                out.write(line + "\n")
                out.flush()
                with self.lock:
                    self.lines.append((time.monotonic() - start, line))

    def snapshot(self) -> list[tuple[float, str]]:
        with self.lock:
            return list(self.lines)

    def first(self, pattern: str) -> float | None:
        for t, line in self.snapshot():
            if pattern in line:
                return t
        return None


def start_stream(args, folder: Path) -> tuple[subprocess.Popen, str]:
    """The live stream: its process and the link the app attaches to."""
    if args.stream == "train":
        argv = [args.train_python, str(HERE.parent / "train.py"), "--seconds", str(args.minutes * 60 + 120)]
        want = re.compile(r"(http://127\.0\.0\.1:\S+|ws://127\.0\.0\.1:\S+)")
    else:
        argv = [str(Path(args.build) / "caliper_attention_producer"), "--layers", "6", "--heads", "8",
                "--tokens", "64", "--hz", "8"]
        want = re.compile(r"\[producer\] hub (\S+)")
    log = (folder / "stream.log").open("w")
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env_for_children())
    link = ""
    deadline = time.monotonic() + 120
    while not link and time.monotonic() < deadline:
        raw = proc.stdout.readline()
        if not raw:
            break
        line = raw.decode("utf-8", "replace")
        log.write(line)
        log.flush()
        m = want.search(line)
        if m:
            link = m.group(1)
    threading.Thread(target=lambda: [log.write(r.decode("utf-8", "replace")) or log.flush() for r in proc.stdout],
                     daemon=True).start()
    return proc, link


def env_for_children() -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ANTHROPIC_", "OPENAI_", "CALIPER_MODEL_API_KEY", "CALIPER_TEST_", "CALIPER_API"))}
    env["CALIPER_AGENT_OFFLINE"] = "1"
    return env


def agent_command(usage_log: Path | None = None) -> str:
    """The side chat's custom command line: chat-agent.sh beside this folder, the
    script a person gives the chat, with the token counts of each model request
    in `usage_log` (an argument it passes on to caliper-agent)."""
    words = [str(HERE.parent / "chat-agent.sh")]
    if usage_log is not None:
        words += ["--usage-log", str(usage_log)]
    return " ".join(shlex.quote(w) for w in words)


def agent_environment(args) -> dict:
    """What chat-agent.sh reads, through the app's environment, which its agent inherits."""
    env = {"CALIPER_CHAT_MODEL": args.model, "CALIPER_CHAT_BASE_URL": args.base_url}
    if args.agent_python:
        env["CALIPER_AGENT_PYTHON"] = args.agent_python
    if args.context_length:
        env["CALIPER_CHAT_CONTEXT"] = str(args.context_length)
    return env


def ui_script(args, gate: Path) -> str:
    patience = args.minutes * 60
    return "; ".join([
        "wait 120", "chat open", "wait 30", "chat focus", "wait 10", f"type {REQUEST}", "wait 3", "key enter",
        f"patience {patience}",
        "chat item tool - completed", "wait 60", "shot request",
        "chat item approval - waiting", "cards 1", "shot card", f"touch {gate}",
        "chat until ready", "patience 60", "page spec", "wait 240", "shot live-a", "wait 180", "shot live-b",
        "quit",
    ])


def approve_cards(log: Log, decisions: Path, gate: Path, done: threading.Event, record: list) -> None:
    """Approves each card the app reports waiting, once the first has been captured."""
    seen: set[str] = set()
    while not done.is_set():
        for t, line in log.snapshot():
            m = APPROVAL.match(line)
            if not m or m.group(4) != "waiting" or m.group(1) in seen:
                continue
            seen.add(m.group(1))
            while not gate.exists() and not done.is_set():
                time.sleep(0.2)
            time.sleep(1.0)
            with decisions.open("a") as f:
                f.write(f"approve {m.group(2)} {m.group(3)}\n")
            record.append({"id": m.group(1), "command": m.group(2), "waiting_at": round(t, 1)})
        time.sleep(0.2)


def journal(ws: Path) -> list[dict]:
    path = ws / "journal" / "commands.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def copy_shots(raw: Path, out: Path, size: str) -> list[str]:
    """The run's frames as step4-<what>-<W>x<H>.png, scaled to the window's size in points."""
    w, h = size.split("x")
    written = []
    for name in ("request", "card", "live-a", "live-b"):
        src = raw / f"{name}.png"
        if not src.exists():
            continue
        dst = out / f"step4-{name}-{size}.png"
        if shutil.which("sips"):
            subprocess.run(["sips", "-z", h, w, str(src), "--out", str(dst)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False)
        else:
            shutil.copyfile(src, dst)
        written.append(str(dst))
    return written


def one_run(args, n: int, work: Path) -> dict:
    folder = work / f"run-{n}"
    shutil.rmtree(folder, ignore_errors=True)
    ws, data, apps, raw = folder / "ws", folder / "data", folder / "apps", folder / "raw"
    for d in (ws, data, apps, raw):
        d.mkdir(parents=True)
    decisions, gate = folder / "decisions.txt", folder / "first-card-captured"
    decisions.touch()
    (data / "side-chat.json").write_text(json.dumps({
        "agent": "custom", "custom_name": "Caliper agent", "custom_command": agent_command(folder / "usage.jsonl"),
        "allow_always": False, "open": False}))
    report: dict = {"run": n, "model": args.model, "stream": args.stream}
    stream, link = start_stream(args, folder)
    if not link:
        stream.terminate()
        report["outcome"] = "the stream printed no link"
        return report
    build = Path(args.build)
    env = env_for_children()
    env.update(agent_environment(args))
    # The PATH an app started from the Finder has, which its agents inherit.
    env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
    env.update({"CALIPER_DATA_DIR": str(data), "CALIPER_USER_APPLETS_DIR": str(apps), "CALIPER_WINDOW_SIZE": args.size,
                "CALIPER_TEST_APPROVALS": str(decisions), "CALIPER_TEST_SPEC_UI": ui_script(args, gate),
                "CALIPER_TEST_SPEC_UI_SHOTS": str(raw), "CALIPER_EXIT_AFTER": str(args.minutes * 60 + 600)})
    start = time.monotonic()
    app = subprocess.Popen([str(build / "caliper"), "--hidden", "--applets", str(build / "applets"), "--url", link,
                            "--workspace", str(ws)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
    log = Log(folder / "app.log")
    reader = threading.Thread(target=log.follow, args=(app.stderr, start), daemon=True)
    reader.start()
    done, approvals = threading.Event(), []
    approver = threading.Thread(target=approve_cards, args=(log, decisions, gate, done, approvals), daemon=True)
    approver.start()
    try:
        app.wait(timeout=args.minutes * 60 + 660)
    except subprocess.TimeoutExpired:
        app.terminate()
        app.wait(timeout=30)
    done.set()
    reader.join(timeout=5)
    stream.terminate()
    try:
        stream.wait(timeout=10)
    except subprocess.TimeoutExpired:
        stream.kill()

    sent = log.first("[ui] pressed enter")
    first_card = next((t for t, line in log.snapshot() if APPROVAL.match(line) and line.endswith(" waiting")), None)
    ended = log.first("[ui] the chat is ready")
    stopped = next((line for _, line in log.snapshot() if line.startswith("[ui] stopped")), "")
    agent = [e for e in journal(ws) if e.get("origin") == "agent"]
    counts: dict[str, int] = {}
    for e in agent:
        key = e["name"] if e.get("ok") else f"{e['name']} failed ({e.get('error', {}).get('code', '?')})"
        counts[key] = counts.get(key, 0) + 1
    panels = sum(1 for e in agent if e.get("ok") and e["name"] == "panel.create")
    decided = [line for _, line in log.snapshot() if APPROVAL.match(line) and not line.endswith(" waiting")]
    usage = folder / "usage.jsonl"
    requests = len(usage.read_text().splitlines()) if usage.exists() else 0
    a, b = raw / "live-a.png", raw / "live-b.png"
    live = a.exists() and b.exists() and a.read_bytes() != b.read_bytes()
    report.update({
        "seconds_to_first_card": round(first_card - sent, 1) if first_card is not None and sent is not None else None,
        "seconds_to_end_of_turn": round(ended - sent, 1) if ended is not None and sent is not None else None,
        "model_requests": requests, "approvals": len(approvals),
        "approved": [APPROVAL.match(line).group(4) for line in decided].count("approved"),
        "agent_commands": counts, "panels": panels, "spec_page": log.first("[ui] page spec") is not None,
        "live_frames_differ": live, "script": stopped or "done", "shots": sorted(p.name for p in raw.glob("*.png")),
        "app_exit": app.returncode, "folder": str(folder)})
    report["outcome"] = ("built a dashboard" if panels and ended is not None and report["spec_page"] else
                         "no edit was made" if not agent else
                         "edits but no panel" if not panels else
                         "the turn did not end" if ended is None else "the spec page did not show")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--build", required=True)
    p.add_argument("--agent-python", help="the Python chat-agent.sh runs caliper-agent with (CALIPER_AGENT_PYTHON); "
                                          "by default the repository's .venv/bin/python, else python3")
    p.add_argument("--out", required=True)
    p.add_argument("--work", help="where each run's files go (default: a new temporary folder)")
    p.add_argument("--model", default="qwen3.8:27b")
    p.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    p.add_argument("--context-length", type=int,
                   help="the context the model is loaded with, in tokens (caliper-agent's --context-length)")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--minutes", type=int, default=10, help="how long the agent's turn may take")
    p.add_argument("--size", default="1440x900")
    p.add_argument("--stream", choices=("producer", "train"), default="producer")
    p.add_argument("--train-python", help="with --stream train: a Python with torch and Caliper LIB")
    args = p.parse_args()
    if not loopback(args.base_url):
        print("step4: --base-url must be a server on this machine", file=sys.stderr)
        return 2
    if args.stream == "train" and not args.train_python:
        print("step4: --stream train needs --train-python", file=sys.stderr)
        return 2
    why = check_model(args.base_url, args.model)
    if why:
        print(f"step4: {why}", file=sys.stderr)
        return 2
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="caliper-step4-"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["CALIPER_AGENT_OFFLINE"] = "1"
    captured = False
    for n in range(1, args.runs + 1):
        report = one_run(args, n, work)
        if not captured and report.get("outcome") == "built a dashboard":
            report["captures"] = copy_shots(work / f"run-{n}" / "raw", out, args.size)
            captured = True
        print(json.dumps(report), flush=True)
    return 0 if captured else 1


if __name__ == "__main__":
    sys.exit(main())
