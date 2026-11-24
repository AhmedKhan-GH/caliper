"""Hosted prompt bytes with a synthetic tokenizer and the CPU fixture."""
from contextlib import contextmanager
import json
from pathlib import Path

import pytest

from caliper import _controls, _managed, _wire as wire
from caliper.interp._hosted import HostedInference, TinyTokenizer, fixture_resources, load_resources
from caliper.interp._output import prompt_inputs
from .test_service import Transport


class WordTokenizer(TinyTokenizer):
    """Encode each whitespace-separated word, with one additional BOS token."""
    def encode(self, text, add_special_tokens=True):
        return ([self.bos_token_id] if add_special_tokens else []) + [
            sum(word.encode("utf-8")) % 256 for word in text.split()]

    def __call__(self, text, **kwargs):
        assert kwargs.get("truncation", False) is False
        return super().__call__(text, **kwargs)


TOKEN_OVERFLOW = " ".join(["token"] * 32)  # BOS plus 32 words gives 33 tokens.


@pytest.fixture
def prompt_run(tmp_path, monkeypatch):
    import caliper
    from caliper import _net
    folder = fixture_resources(tmp_path / "dictionary")
    serial = 0

    @contextmanager
    def start(prompt, *, resume=None, params=None, interventions=None):
        nonlocal serial
        serial += 1
        model, _, dictionary, index, guard = load_resources("tiny-lm", folder)
        tokenizer = WordTokenizer()
        index.tokenizer = tokenizer
        controls = _controls.Controls()
        monkeypatch.setattr(caliper, "_CTL", controls)
        monkeypatch.setattr(_controls, "CONTROLS", controls)
        managed_folder = tmp_path / f"run-{serial}"
        managed_folder.mkdir()
        managed = _managed.ManagedRun("prompt", str(managed_folder), str(managed_folder / "link"),
                                      resume=resume, params=params, interventions=interventions)
        managed.kept.clock = lambda: 0.
        monkeypatch.setattr(_managed, "current", lambda: managed)
        monkeypatch.setattr(_managed, "watch_parent", lambda: None)
        run = HostedInference("tiny-lm", folder, prompt,
                              resources=(model, tokenizer, dictionary, index, guard))
        route = controls.attach(Transport())
        try:
            run.start()
            yield run, route
        finally:
            run.close()
            route.detach()
            server = _net.current_server()
            if server is not None:
                server.close()
    return start


def assert_prompt(run, expected):
    assert run.prompt.value.encode() == expected.encode()
    assert run.output.prompt.encode() == expected.encode()
    metadata = wire.Controls({"state": "running"}, [run.prompt.entry()])
    assert wire.decode(wire.encode(metadata)).params[0]["value"].encode() == expected.encode()


@pytest.mark.parametrize("prompt", ["river" * 205 + " x", "é" * 2048])
def test_hosted_startup_preserves_long_prompt(prompt_run, prompt):
    with prompt_run(prompt) as (run, _):
        assert len(prompt.encode()) > 1024
        assert prompt_inputs(run.model, run.tokenizer, prompt)["input_ids"].shape == (1, len(prompt.split()) + 1)
        assert_prompt(run, prompt)


def test_trace_control_and_checkpoint_preserve_exact_prompt(prompt_run, tmp_path):
    traced = "r" * 4094 + " x"
    edited = "é" * 2048
    assert len(traced.encode()) == len(edited.encode()) == 4096
    with prompt_run("start") as (run, route):
        message = wire.Control(1, "interpret", interpretation={"version": 1, "method": "trace.run",
            "params": {"prompt": traced, "max_targets": 2, "max_nodes": 16, "max_edges": 32}})
        message = wire.decode(wire.encode(message))
        assert route.on_control(None, message.to_meta())["status"] == "queued"
        run.boundary()
        assert route.transport.replies[-1].status == "ok"
        assert run.traced_prompt == traced
        assert_prompt(run, traced)
        message = wire.decode(wire.encode(wire.Control(2, "set", name="prompt", value=edited)))
        assert route.on_control(None, message.to_meta())["status"] == "ok"
        assert run.prompt.pending == edited
        run.boundary()
        assert_prompt(run, edited)
        checkpoint = tmp_path / "prompt.json"
        checkpoint.write_bytes(run.managed._state(run.model, None, run.controls.n))
        record = json.loads(checkpoint.read_bytes())
        assert record["params"]["prompt"].encode() == edited.encode()
        for bad in (edited + "x", "bad\0prompt", "\ud800"):
            record["params"]["prompt"] = bad
            with pytest.raises(ValueError):
                run.load_checkpoint(record)
            assert run.restored is None
            assert_prompt(run, edited)
    with prompt_run("initial", resume=str(checkpoint)) as (restored, _):
        assert_prompt(restored, edited)


def control(route, request_id, op, **kwargs):
    message = wire.decode(wire.encode(wire.Control(request_id, op, **kwargs)))
    return route.on_control(None, message.to_meta())


def save_boundary(run, route, kind):
    """Write a real checkpoint after pending text reaches the hosted boundary."""
    from caliper.interp._memory import CleanStop

    if kind == "explicit":
        assert control(route, 90, "run", name="caliper.checkpoint")["status"] == "queued"
    if kind == "clean-stop":
        def stop():
            raise CleanStop("prompt checkpoint test stop")
        action = run.controls.action("test_stop", stop)
        try:
            assert control(route, 90, "run", name="test_stop")["status"] == "queued"
            with pytest.raises(CleanStop, match="prompt checkpoint test stop"):
                run.boundary()
        finally:
            action.remove()
    else:
        run.boundary()
    if kind == "kept":
        step, path = run.managed.kept.kept[-1]
        assert step == run.controls.n
    elif kind == "explicit":
        reply, = [reply for reply in route.transport.replies if reply.request_id == 90]
        assert reply.status == "ok"
        path = reply.message
    else:
        path = Path(run.managed.dir) / f"checkpoint-{run.controls.n}.json"
    path = Path(path)
    record = json.loads(path.read_bytes())
    assert record["step"] == run.controls.n
    return path, record


@pytest.mark.parametrize("rejected", ["", TOKEN_OVERFLOW], ids=["empty", "33-tokens"])
@pytest.mark.parametrize("kind", ["kept", "explicit", "clean-stop"])
def test_rejected_boundary_prompt_keeps_valid_state_and_forks(prompt_run, rejected, kind):
    accepted = "current prompt"
    with prompt_run("initial") as (run, route):
        assert control(route, 1, "set", name="prompt", value=accepted)["status"] == "ok"
        run.boundary()
        assert_prompt(run, accepted)
        assert len(run.tokenizer.encode(TOKEN_OVERFLOW)) == 33
        assert control(route, 2, "set", name="prompt", value=rejected)["status"] == "ok"
        assert run.prompt.pending == rejected and run.prompt.value == accepted
        checkpoint, record = save_boundary(run, route, kind)
        # The rejected editor text and error stay visible, but do not replace durable state.
        assert run.prompt.value == rejected and run.prompt.pending is None
        assert "prompt must contain" in run.status.value
        assert run.output.prompt == accepted
        assert record["params"]["prompt"] == accepted
    with prompt_run("initial", resume=str(checkpoint)) as (fork, _):
        assert_prompt(fork, accepted)
        assert fork.controls.n == record["step"]
        assert json.loads(Path(fork.managed.kept.kept[-1][1]).read_bytes())["params"]["prompt"] == accepted
    override = "é" * 2048
    with prompt_run("initial", resume=str(checkpoint), params={"prompt": override}) as (fork, _):
        assert_prompt(fork, override)
        assert len(override.encode()) == 4096
        assert json.loads(Path(fork.managed.checkpoint(fork.model, None)).read_bytes())["params"]["prompt"] == override
    for invalid in ("", TOKEN_OVERFLOW, "x" * 4097, "bad\0prompt"):
        with pytest.raises(ValueError, match="prompt must contain"):
            with prompt_run("initial", resume=str(checkpoint), params={"prompt": invalid}):
                pytest.fail("invalid fork prompt started")


@pytest.mark.parametrize("rejected", ["", TOKEN_OVERFLOW], ids=["empty", "33-tokens"])
def test_legacy_rejected_checkpoint_requires_valid_prompt_override(prompt_run, rejected, tmp_path):
    with prompt_run("accepted") as (run, route):
        assert control(route, 1, "set", name="prompt", value=rejected)["status"] == "ok"
        checkpoint, record = save_boundary(run, route, "explicit")
        # Reproduce the old serializer after the actual rejected boundary.
        record["params"]["prompt"] = run.prompt.value
        checkpoint = tmp_path / "legacy.json"
        checkpoint.write_text(json.dumps(record), encoding="utf-8")
    before = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="prompt must contain"):
        with prompt_run("initial", resume=str(checkpoint)):
            pytest.fail("invalid saved prompt started without an override")
    for invalid in ("", TOKEN_OVERFLOW):
        with pytest.raises(ValueError, match="prompt must contain"):
            with prompt_run("initial", resume=str(checkpoint), params={"prompt": invalid}):
                pytest.fail("invalid override started")
    with prompt_run("initial", resume=str(checkpoint), params={"prompt": "valid override"}) as (fork, _):
        assert_prompt(fork, "valid override")
        saved = Path(fork.managed.checkpoint(fork.model, None))
        assert json.loads(saved.read_bytes())["params"]["prompt"] == "valid override"
    with prompt_run("initial", resume=str(saved)) as (fork, _):
        assert_prompt(fork, "valid override")
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("prompt", ["", "x" * 4097, "é" * 2048 + "x", "bad\0prompt", "\ud800"])
def test_prompt_validation_refuses_bytes_before_tokenization(prompt):
    class UnusedTokenizer:
        def __call__(self, *args, **kwargs):
            pytest.fail("invalid prompt reached the tokenizer")
    with pytest.raises(ValueError, match="UTF-8"):
        prompt_inputs(None, UnusedTokenizer(), prompt)


def test_prompt_token_limit_is_independent_of_byte_limit():
    import torch
    from .fixture import TinyLM
    model = TinyLM()
    class Tokenizer:
        count = 32
        def __call__(self, text, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": torch.ones((1, self.count), dtype=torch.long)}
    tokenizer = Tokenizer()
    assert prompt_inputs(model, tokenizer, "x" * 4096)["input_ids"].shape[1] == 32
    tokenizer.count = 33
    with pytest.raises(ValueError, match="32 integer tokens"):
        prompt_inputs(model, tokenizer, "x" * 4096)


def test_local_mcp_transport_preserves_prompt_bytes(prompt_run, tmp_path):
    """Use the MCP executable and a local API adapter for the real Controls route."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import os
    from pathlib import Path
    import selectors
    import subprocess
    import threading

    executable = os.environ.get("CALIPER_PROMPT_MCP")
    if not executable:
        pytest.skip("CALIPER_PROMPT_MCP must name the local MCP executable")
    root = Path(__file__).resolve().parents[2]
    table = json.loads((root / "src/builder/commands/command_table.json").read_text())
    token = "prompt-contract-local-test-token-0001"
    expected = "é" * 2048
    received = []
    errors = []
    with prompt_run("start") as (run, route):
        class Api(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def answer(self, body):
                data = json.dumps(body, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.answer(table)

            def do_POST(self):
                try:
                    assert self.path == "/v0/commands/run.set_param"
                    assert self.headers["Authorization"] == "Bearer " + token
                    assert self.headers["X-Caliper-Origin"] == "mcp"
                    args = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    received.append(args)
                    message = wire.decode(wire.encode(wire.Control(9, "set", name=args["name"], value=args["value"])))
                    reply = route.on_control(None, message.to_meta())
                    assert reply["status"] == "ok"
                    self.answer({"id": "prompt-1", "ok": True,
                                 "result": {"name": args["name"], "value": reply["value"], "pending": True}})
                except BaseException as exc:
                    errors.append(exc)
                    self.answer({"id": "prompt-1", "ok": False, "error": {"code": "test.failed", "message": str(exc)}})

        server = HTTPServer(("127.0.0.1", 0), Api)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        api_file = tmp_path / "api.json"
        port = server.server_address[1]
        api_file.write_text(json.dumps({"version": 1, "pid": os.getpid(), "port": port, "token": token,
            "url": f"http://127.0.0.1:{port}/v0", "stream_url": f"ws://127.0.0.1:{port}/v3/stream"}))
        process = None
        try:
            process = subprocess.Popen([executable, "--api-file", str(api_file)], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            metadata = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
                        "io.modelcontextprotocol/clientCapabilities": {},
                        "io.modelcontextprotocol/clientInfo": {"name": "prompt-contract", "version": "1"}}
            request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": "run_set_param", "arguments": {"name": "prompt", "value": expected}, "_meta": metadata}}
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                assert selector.select(15), "MCP response deadline expired"
                response = json.loads(process.stdout.readline())
            assert not errors, errors
            result = response["result"]
            assert result["isError"] is False, result
            assert result["structuredContent"]["value"].encode() == expected.encode()
            assert received == [{"name": "prompt", "value": expected}]
            assert run.prompt.pending == expected
            run.boundary()
            assert_prompt(run, expected)
        finally:
            if process is not None:
                process.stdin.close()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=5)
                process.stdout.close()
                process.stderr.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
