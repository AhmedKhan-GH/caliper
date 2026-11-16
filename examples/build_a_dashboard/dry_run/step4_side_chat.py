"""Step 4 of the walk-through in the hidden app: the side chat with a real agent on a local model.

The side chat runs caliper-agent, Caliper's own ACP agent, on a model served
by Ollama on this machine, so that no account or key is used: its custom
command is `chat-agent.sh` beside this folder, the script a person gives the
chat. The app is attached to a live stream, the person's message is "Plot
attention entropy per layer while it trains", every edit the agent sends waits
on an approval card and is approved, and the dashboard the agent builds then
draws the stream live on the spec page beside the chat.

    python3 step4_side_chat.py --agent-python <python> [--caliper <app>] [--out <folder>] [--work <folder>]
        [--size 1440x900] [--url <stream link> | --stream producer|train --train-python <python>]
        [--workspace <folder> --app-data <folder>] [--model qwen3.8:27b]
        [--base-url http://127.0.0.1:11434/v1] [--context-length 32768] [--runs 1] [--minutes 10]

The options are those of hidden_app.py.
- `--agent-python` is a Python with caliper-agent's dependencies
  (`src/builder/agent/pyproject.toml`), which chat-agent.sh takes from
  CALIPER_AGENT_PYTHON; the agent runs from this repository's source, so
  nothing is installed. The app gets the PATH of an app started from the
  Finder, the system's folders only, which its agent inherits.
- The stream: with `--url`, a stream that is already running, such as the
  one run_all.py starts. Otherwise the script starts its own:
  `caliper_attention_producer` beside the caliper executable by default, a
  test tool that stands in for the training script with the same views and
  no torch, or with `--stream train` `train.py` beside this folder, run by
  `--train-python` (a Python with torch and Caliper LIB).
- With `--workspace` and `--app-data`, as in run_all.py --story, the app
  works in the story's workspace, and the agent's dashboard is made there
  beside the one of steps 2 and 3.

What the script plays is the person's part, through the hidden app's input
script (CALIPER_TEST_SPEC_UI): it opens the chat, types the message and presses
Enter, waits until the approval strip shows the first card and captures it,
waits for the turn's end, waits for the spec page, which the app shows for the
agent's new spec, and captures the dashboard twice a few seconds apart. Each
approval card is decided through the hidden app's test-only decisions file
(CALIPER_TEST_APPROVALS), which applies the decision through the same path as
the card's Approve button: a thread here reads the app's `[approval] ...
waiting` lines and approves each card once the first has been captured.

Captures are written to `--out` as step4-<what>-<W>x<H>.png (request, card,
live-a, live-b), W x H the window's size in points, from the first run that got
there. Every run's files go into a new subfolder of `--work`, or into a
temporary folder that is removed at the end, and the script prints one JSON
object per run: how it ended, the seconds to the first card and to the end of
the turn, the approvals, and the commands the agent's turn sent, from the
workspace's journal. It exits with 0 when at least one run built a dashboard
with panels, 1 when none did, 2 when something it needs is missing, and 130
when interrupted. Every process it starts is stopped by its pid when it ends,
fails or is interrupted.

CALIPER_AGENT_OFFLINE=1 is set for everything it starts, so caliper-agent
reaches no host off this machine. The app never starts a detected agent: the
chat's settings name the custom command.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from hidden_app import (FAILED, INTERRUPTED, MISSING, OK, Children, Stop, add_options, app_env, caliper_problem,
                        clean_env, fresh_folder, work_folder)

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent / "train.py"
PRODUCER = "caliper_attention_producer"
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
    except (OSError, ValueError) as exc:
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


def start_stream(children: Children, args, folder: Path, data: Path) -> tuple[subprocess.Popen, str]:
    """The script's own live stream: its process and the link the app attaches to. train.py gets the
    app's data folder, where its stream file goes, never the person's own."""
    if args.stream == "train":
        argv = [args.train_python, "-u", str(TRAIN), "--seconds", str(args.minutes * 60 + 120)]
        want = re.compile(r"(http://127\.0\.0\.1:\S+|ws://127\.0\.0\.1:\S+)")
        name, sig = "train.py", signal.SIGINT
    else:
        argv = [str(Path(args.caliper).parent / PRODUCER), "--layers", "6", "--heads", "8", "--tokens", "64",
                "--hz", "8"]
        want = re.compile(r"\[producer\] hub (\S+)")
        name, sig = PRODUCER, signal.SIGTERM
    proc = children.start(name, argv, stop_signal=sig, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          stdin=subprocess.DEVNULL, env=dict(clean_env(), CALIPER_DATA_DIR=str(data)))
    # A thread copies the stream's output to its log, so that the wait for the link below ends at its
    # deadline even when the stream prints nothing.
    lines: list[str] = []

    def copy() -> None:
        with (folder / "stream.log").open("w") as log:
            for raw in proc.stdout:
                line = raw.decode("utf-8", "replace")
                log.write(line)
                log.flush()
                lines.append(line)

    threading.Thread(target=copy, daemon=True).start()
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and proc.poll() is None:
        m = next((m for m in map(want.search, list(lines)) if m), None)
        if m:
            return proc, m.group(1)
        time.sleep(0.2)
    m = next((m for m in map(want.search, list(lines)) if m), None)
    return proc, m.group(1) if m else ""


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
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def copy_shots(raw: Path, out: Path, size: str) -> list[str]:
    """The run's frames as step4-<what>-<W>x<H>.png, as captured."""
    written = []
    for name in ("request", "card", "live-a", "live-b"):
        src = raw / f"{name}.png"
        if src.exists():
            dst = out / f"step4-{name}-{size}.png"
            shutil.copyfile(src, dst)
            written.append(str(dst))
    return written


def one_run(children: Children, args, n: int, work: Path) -> dict:
    folder = fresh_folder(work, f"run-{n}")
    story = bool(args.workspace)
    ws = Path(args.workspace).resolve() if story else folder / "ws"
    data = Path(args.app_data).resolve() if story else folder / "data"
    apps, raw = folder / "apps", folder / "raw"
    for d in (data, apps, raw):
        d.mkdir(parents=True, exist_ok=True)
    before = len(journal(ws))                     # in the story, the workspace's journal already holds earlier steps
    decisions, gate = folder / "decisions.txt", folder / "first-card-captured"
    decisions.touch()
    # The chat's agent, chosen as a person chooses it in the picker.
    (data / "side-chat.json").write_text(json.dumps({
        "agent": "custom", "custom_name": "Caliper agent", "custom_command": agent_command(folder / "usage.jsonl"),
        "allow_always": False, "open": False}))
    report: dict = {"run": n, "model": args.model, "stream": "given" if args.url else args.stream,
                    "story": story}
    if args.url:
        stream, link = None, args.url
    else:
        stream, link = start_stream(children, args, folder, data)
    if not link:
        report["outcome"] = "the stream printed no link"
        return report
    build = Path(args.caliper).parent
    extra = dict(agent_environment(args), CALIPER_TEST_APPROVALS=str(decisions))
    env = app_env(data, apps, args.size, ui_script(args, gate), raw, exit_after=args.minutes * 60 + 600, extra=extra)
    argv = [str(args.caliper), "--hidden", "--url", link, "--workspace", str(ws)]
    if (build / "applets").is_dir():
        argv[2:2] = ["--applets", str(build / "applets")]
    start = time.monotonic()
    app = children.start("the app", argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                         stdin=subprocess.DEVNULL, env=env)
    log = Log(folder / "app.log")
    reader = threading.Thread(target=log.follow, args=(app.stderr, start), daemon=True)
    reader.start()
    done, approvals = threading.Event(), []
    approver = threading.Thread(target=approve_cards, args=(log, decisions, gate, done, approvals), daemon=True)
    approver.start()
    try:
        app.wait(timeout=args.minutes * 60 + 660)
    except subprocess.TimeoutExpired:
        children.stop(app)
    finally:
        done.set()
    reader.join(timeout=5)
    if stream is not None:
        children.stop(stream)

    sent = log.first("[ui] pressed enter")
    first_card = next((t for t, line in log.snapshot() if APPROVAL.match(line) and line.endswith(" waiting")), None)
    ended = log.first("[ui] the chat is ready")
    stopped = next((line for _, line in log.snapshot() if line.startswith("[ui] stopped")), "")
    agent = [e for e in journal(ws)[before:] if e.get("origin") == "agent"]
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
    add_options(p, train=True, agent=True, model=True, url=True)
    p.add_argument("--base-url", default="http://127.0.0.1:11434/v1",
                   help="the model server's OpenAI-compatible address, on this machine")
    p.add_argument("--context-length", type=int,
                   help="the context the model is loaded with, in tokens (caliper-agent's --context-length)")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--minutes", type=int, default=10, help="how long the agent's turn may take")
    p.add_argument("--stream", choices=("producer", "train"), default="producer",
                   help="without --url, the stream the script starts itself")
    args = p.parse_args()
    problems = [caliper_problem(args.caliper)]
    if not loopback(args.base_url):
        problems.append("--base-url must be a server on this machine")
    if args.url and not loopback(args.url):
        problems.append("--url must be a stream on this machine")
    if not args.url and args.stream == "train" and not args.train_python:
        problems.append("--stream train needs --train-python")
    if not args.url and args.stream == "producer" and not (Path(args.caliper).parent / PRODUCER).is_file():
        problems.append(f"no {PRODUCER} beside {args.caliper}; build it, or give --url or --stream train")
    if bool(args.workspace) != bool(args.app_data):
        problems.append("--workspace and --app-data go together")
    for problem in filter(None, problems):
        print(f"step4: {problem}", file=sys.stderr)
        return MISSING
    why = check_model(args.base_url, args.model)
    if why:
        print(f"step4: {why}", file=sys.stderr)
        return MISSING
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    captured = False
    with Children() as children:
        try:
            work = work_folder(args, 4)
            for n in range(1, args.runs + 1):
                report = one_run(children, args, n, work)
                if not captured and report.get("outcome") == "built a dashboard":
                    report["captures"] = copy_shots(Path(report["folder"]) / "raw", out, args.size)
                    captured = True
                print(json.dumps(report), flush=True)
        except Stop as why:
            print(f"step4: interrupted by {why}", flush=True)
            return INTERRUPTED
    return OK if captured else FAILED


if __name__ == "__main__":
    sys.exit(main())
