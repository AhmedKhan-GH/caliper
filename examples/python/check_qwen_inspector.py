#!/usr/bin/env python3
"""Exercise the inspector through the same request channel as the app.

Default: start one host, check it, and stop only that child process. Use
--connect SESSION to check an already running inspector without loading a
second model. Outputs go to the terminal; no log files are written.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

from caliper.requests import MAX_MESSAGE, encode, socket_path


class Controller:
    def __init__(self, session, timeout=120):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        self.socket.connect(socket_path(session))
        self.file = self.socket.makefile("rb")
        self.serial = 0

    def send(self, op, **arguments):
        self.serial += 1
        self.socket.sendall(encode(dict(id=self.serial, op=op, **arguments)))
        return self.serial

    def receive(self):
        line = self.file.readline(MAX_MESSAGE + 1)
        if not line:
            raise RuntimeError("host disconnected")
        if len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
            raise RuntimeError("invalid response frame")
        message = json.loads(line)
        if "error" in message:
            raise RuntimeError(message["error"])
        return message

    def request(self, op, **arguments):
        rid = self.send(op, **arguments)
        events = []
        while True:
            message = self.receive()
            assert message["id"] == rid, message
            if message["done"]:
                return message["result"], events
            events.append(message["event"])

    def close(self):
        self.file.close()
        self.socket.close()


def check(session):
    controller = Controller(session)
    try:
        hello, _ = controller.request("hello")
        assert hello["protocol"] == 1
        assert "generate" in hello["capabilities"]
        edits, _ = controller.request("list_edits")
        if edits["undo_bytes"]:
            raise RuntimeError("checks require a host without active edits; use a fresh host or revert them first")
        manifest, _ = controller.request("list_tensors")
        tensors = manifest["tensors"]
        assert tensors and all(t["shape"] and t["dtype"] and t["bytes"] > 0 for t in tensors)
        embedding = next(t for t in tensors if t["name"].endswith("embed_tokens.weight"))
        assert embedding["shape"] == [248320, 1024]
        assert embedding["tied"]
        name = embedding["name"]
        overview, _ = controller.request("show_tensor", name=name)
        assert overview["sample_rows"] <= 128 and overview["sample_cols"] <= 128
        assert len(overview["values"]) == overview["sample_rows"] * overview["sample_cols"]
        assert len(overview["row_indices"]) == overview["sample_rows"]
        print(f"PASS: {len(tensors)} parameter tensors; embedding overview "
              f"{overview['sample_rows']} x {overview['sample_cols']}", flush=True)

        settings = dict(prompt="Say hello in one short sentence.", max_new_tokens=16,
                        temperature=0.0, top_p=1.0, top_k=0, thinking=False)
        result, events = controller.request("generate", **settings)
        text = "".join(event.get("text", "") for event in events)
        assert text.strip(), (result, events)
        assert result["tokens"] > 0 and result["tokens_per_second"] > 0
        assert not result["cancelled"]
        print(f"PASS: generated {text!r} ({result['tokens_per_second']:.1f} tokens/s)", flush=True)

        before, _ = controller.request("read_weight", name=name, row=0, col=0)
        replacement = 0.25 if before["value"] != 0.25 else -0.25
        changed = False
        try:
            controller.request("edit_weight", name=name, action="set", row=0, col=0, value=replacement)
            changed = True
            after, _ = controller.request("read_weight", name=name, row=0, col=0)
            assert after["value"] == replacement
            # The host uses the same edited model for its very next generation.
            result, events = controller.request("generate", **dict(settings, max_new_tokens=4))
            assert result["tokens"] > 0 and any(e.get("text") for e in events)
            print(f"PASS: tied embedding set/read {before['value']:g} -> {after['value']:g}; "
                  "generation after edit completed", flush=True)
        finally:
            if changed:
                controller.request("revert_tensor", name=name)
        restored, _ = controller.request("read_weight", name=name, row=0, col=0)
        assert restored["value"] == before["value"]
        print("PASS: embedding reverted exactly", flush=True)

        # Exercise intersecting edits on a small matrix; reverse undo must
        # recover all four cells exactly, even where operations overlap.
        small = min((t for t in tensors if t["rows"] > 1 and t["cols"] > 1),
                    key=lambda t: t["size"])
        small_name = small["name"]
        def read(row, col):
            reply, _ = controller.request("read_weight", name=small_name, row=row, col=col)
            return reply["value"]
        original = {(r, c): read(r, c) for r in (0, 1) for c in (0, 1)}
        edited = False
        try:
            controller.request("edit_weight", name=small_name, action="scale_row", row=0, col=0, value=2.0)
            edited = True
            assert read(0, 0) == original[0, 0] * 2.0
            controller.request("edit_weight", name=small_name, action="scale_column", row=0, col=0, value=0.5)
            assert read(0, 0) == original[0, 0]
            assert read(1, 0) == original[1, 0] * 0.5
            controller.request("edit_weight", name=small_name, action="zero_row", row=0, col=0)
            assert read(0, 0) == 0.0 and read(0, 1) == 0.0
            controller.request("edit_weight", name=small_name, action="zero_column", row=0, col=0)
            assert read(1, 0) == 0.0
        finally:
            if edited:
                controller.request("revert_tensor", name=small_name)
        assert {(r, c): read(r, c) for r in (0, 1) for c in (0, 1)} == original
        print("PASS: row and column scale/zero; overlapping edits reverted exactly", flush=True)
        try:
            controller.request("read_weight", name=small_name, row=small["rows"], col=0)
            raise AssertionError("out-of-bounds read succeeded")
        except RuntimeError as exc:
            assert "row" in str(exc), str(exc)
        assert read(0, 0) == original[0, 0]
        print("PASS: invalid coordinate rejected; next request healthy", flush=True)

        # Stop is sent after receiving a streamed event while the long request
        # is still active, and gets a separate acknowledgement.
        rid = controller.send("generate", **dict(settings, prompt="Count upward from one. Keep counting.",
                                                  max_new_tokens=512, thinking=True))
        first = controller.receive()
        assert first["id"] == rid and not first["done"], first
        stop_id = controller.send("stop")
        completed = {}
        while len(completed) < 2:
            message = controller.receive()
            if message["done"]:
                completed[message["id"]] = message["result"]
        assert completed[stop_id]["stopping"]
        assert completed[rid]["cancelled"]
        print("PASS: thinking enabled; streamed generation stopped", flush=True)
        print(f"Peak MLX allocation: {completed[rid].get('peak_memory_bytes', 0) / 2**30:.2f} GiB", flush=True)
    finally:
        controller.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", metavar="SESSION", help="use an existing host; do not start another model")
    parser.add_argument("--model", type=Path, help="local model folder passed to a new host")
    parser.add_argument("--startup-timeout", type=float, default=180)
    args = parser.parse_args()
    if args.connect:
        check(args.connect)
        return

    root = Path(__file__).resolve().parents[2]
    session = f"check-{os.getpid()}"
    with tempfile.TemporaryDirectory(prefix="caliper-inspector-") as data:
        command = [sys.executable, str(Path(__file__).with_name("qwen_inspector.py")),
                   "--session", session, "--data-dir", data]
        if args.model:
            command += ["--model", str(args.model.expanduser())]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "python") + os.pathsep + env.get("PYTHONPATH", "")
        # A local path is required by the host; prevent accidental hub access.
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        child = subprocess.Popen(command, env=env, cwd=root, start_new_session=True)
        try:
            deadline = time.monotonic() + args.startup_timeout
            while not Path(socket_path(session)).exists():
                if child.poll() is not None:
                    raise RuntimeError(f"host exited during startup with status {child.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("host startup timed out")
                time.sleep(0.1)
            check(session)
        finally:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()  # only the exact child started by this script
                    child.wait()
            assert not Path(socket_path(session)).exists(), "host left its socket behind"


if __name__ == "__main__":
    main()
