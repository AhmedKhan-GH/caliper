"""Actual offline MCP commands through the native run connection and fixture service."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import selectors
import subprocess
import threading
import time

import pytest
import caliper
from caliper import _controls
from caliper.interp._service import register_service, unregister_service
from .test_service import actual


@pytest.mark.parametrize("actual", [8.0], indirect=True)
def test_offline_command_sidecar(actual, tmp_path, monkeypatch):
    host = os.environ.get("CALIPER_INTERPRET_HOST")
    mcp = os.environ.get("CALIPER_MCP_EXE")
    if not host or not mcp:
        pytest.skip("Set CALIPER_INTERPRET_HOST and CALIPER_MCP_EXE to built targets.")
    assert os.environ.get("CALIPER_AGENT_OFFLINE") == "1"
    monkeypatch.setenv("CALIPER_SESSION", "0")
    monkeypatch.setenv("CALIPER_NETIO", "1")
    monkeypatch.setenv("CALIPER_DATA_DIR", str(tmp_path / "data"))
    controls = _controls.Controls()
    monkeypatch.setattr(_controls, "CONTROLS", controls)
    monkeypatch.setattr(caliper, "_CTL", controls)
    done = threading.Event()
    failures = []

    def boundary():
        try:
            while not done.is_set():
                with controls.cond:
                    controls.cond.wait_for(lambda: actual.pending or done.is_set(), timeout=5)
                if actual.pending:
                    controls.step()
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=boundary, name="interpret-fixture-boundary")

    def stop_worker():
        done.set()
        with controls.cond:
            controls.cond.notify_all()
        if worker.ident is not None:
            worker.join(10)
            assert not worker.is_alive(), "The fixture boundary thread did not stop."

    def stop_process(process):
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    from caliper._net import current_server
    assert current_server() is None, "The fixture requires an isolated server."

    def close_server():
        # serve() can create the server before a later stream-file failure.
        server = current_server()
        if server is not None:
            server.close()

    with ExitStack() as cleanup:
        register_service(actual, controls=controls)
        cleanup.callback(unregister_service, actual, controls=controls)
        cleanup.callback(stop_worker)
        cleanup.callback(close_server)
        server = caliper.serve(port=0, control=True)
        worker.start()
        log = cleanup.enter_context((tmp_path / "host.log").open("w"))
        host_process = subprocess.Popen([host, server.stream_url, str(tmp_path / "workspace"), str(tmp_path / "api")],
                                        stdout=subprocess.PIPE, stderr=log, text=True)
        cleanup.callback(stop_process, host_process)
        def line_from(process):
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                assert selector.select(30), (tmp_path / "host.log").read_text()
            return process.stdout.readline()

        assert line_from(host_process).strip() == "ready"
        sidecar = subprocess.Popen([mcp, "--api-file", str(tmp_path / "api" / "api.json")],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True)
        cleanup.callback(stop_process, sidecar)
        counter = 0

        def tool(name, params):
            nonlocal counter
            counter += 1
            meta = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
                    "io.modelcontextprotocol/clientCapabilities": {},
                    "io.modelcontextprotocol/clientInfo": {"name": "fixture-test", "version": "1"}}
            sidecar.stdin.write(json.dumps({"jsonrpc": "2.0", "id": counter, "method": "tools/call",
                                           "params": {"_meta": meta, "name": name.replace(".", "_"), "arguments": params}}) + "\n")
            sidecar.stdin.flush()
            while True:
                line = line_from(sidecar)
                assert line, (tmp_path / "host.log").read_text()
                message = json.loads(line)
                if message.get("id") == counter:
                    assert "result" in message, message
                    return message["result"]

        def complete(name, params):
            reply = tool(name, params)
            deadline = time.monotonic() + 30
            while not reply.get("isError") and reply["structuredContent"]["pending"]:
                assert time.monotonic() < deadline
                reply = tool(name, {"request_id": reply["structuredContent"]["request_id"]})
            assert not reply.get("isError"), reply
            return reply["structuredContent"]["result"]

        assert complete("feature.dictionary", {})["metadata"]["layers"] == 4
        feature = int(actual.index.counts[0].argmax())
        assert complete("feature.show", {"layer": 0, "feature": feature})["feature"] == feature
        examples = complete("feature.examples", {"layer": 0, "feature": feature})
        assert len(examples["examples"]) == 16
        assert len(examples["examples"][0]["tokens"]) == 33
        assert complete("trace.run", {"prompt": "AB", "max_targets": 32, "max_nodes": 64, "max_edges": 128})["state"] == "done"
        assert complete("trace.status", {})["state"] == "done"
        graph = complete("trace.read", {"node_limit": 256, "edge_limit": 256})
        feature_node = next(i for i, node in enumerate(graph["nodes"]) if node["kind"] == "feature")
        selected = complete("feature.examples", {"node": feature_node})
        assert selected["feature"] == graph["nodes"][feature_node]["feature"]
        from caliper.interp._publish import _resolve
        assert _resolve("examples.label_text").shape == (400, 16)
        assert _resolve("trace.node_i").shape == (512, 4)
        assert tool("trace.read", {"node_limit": 257})["isError"]
        for name, params in (("patch.sweep", {"clean_prompt": "AB", "corrupted_prompt": "AC",
                                              "metric": {"positive_token": 1, "negative_token": 2}}),
                             ("trace.validate", {})):
            reply = tool(name, params)
            deadline = time.monotonic() + 30
            while not reply.get("isError"):
                assert time.monotonic() < deadline and reply["structuredContent"]["pending"]
                reply = tool(name, {"request_id": reply["structuredContent"]["request_id"]})
            assert json.loads(reply["content"][0]["text"])["error"]["code"] == "interpret.unsupported"
        actual.dictionary = None
        reply = tool("feature.dictionary", {})
        deadline = time.monotonic() + 30
        while not reply.get("isError"):
            assert time.monotonic() < deadline
            assert reply["structuredContent"]["pending"]
            reply = tool("feature.dictionary", {"request_id": reply["structuredContent"]["request_id"]})
        error = json.loads(reply["content"][0]["text"])["error"]
        assert error["code"] == "interpret.missing_dictionary"
        assert not failures


@pytest.mark.parametrize("stage", ["serve", "process"])
def test_sidecar_start_failure_releases_resources(actual, tmp_path, monkeypatch, stage):
    from caliper._net import current_server
    if stage == "serve":
        original = caliper.serve
        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("fixture startup failure")
        monkeypatch.setattr(caliper, "serve", fail)
    else:
        def fail(*args, **kwargs):
            raise RuntimeError("fixture startup failure")
        monkeypatch.setattr(subprocess, "Popen", fail)
    with pytest.raises(RuntimeError, match="fixture startup failure"):
        test_offline_command_sidecar(actual, tmp_path, monkeypatch)
    assert current_server() is None
    assert not any(t.name == "interpret-fixture-boundary" and t.is_alive() for t in threading.enumerate())
    assert actual.closed
