"""Installed fixture identity, live hosted controls, and safe inference forks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
import torch

from caliper import _controls, _managed, _wire as wire
from caliper.interp._hosted import HostedInference, TinyLM, TinyTokenizer, fixture_resources, load_resources
from caliper.interp._output import internal_forward
from .fixture import TinyLM as ReferenceLM, TinyTokenizer as ReferenceTokenizer, forward_hash


@pytest.fixture(scope="module")
def resource_folder(tmp_path_factory):
    return fixture_resources(tmp_path_factory.mktemp("hosted-dictionary"))


def test_installed_fixture_has_the_reference_identity():
    from caliper.interp._dictionary import model_identity
    assert model_identity(TinyLM()) == model_identity(ReferenceLM())
    assert forward_hash(TinyLM()) == Path(__file__).with_name("fixture.sha256").read_text().strip()
    assert TinyTokenizer().encode("AB CD") == ReferenceTokenizer().encode("AB CD")


def test_fixture_dictionary_and_index_load_locally(resource_folder):
    model, tokenizer, dictionary, index, guard = load_resources("tiny-lm", resource_folder)
    try:
        assert next(model.parameters()).device.type == "cpu" and guard is None
        assert len(dictionary.layers) == 4
        layer, feature = 0, int(index.counts[0].argmax())
        assert index.show(layer, feature)["maximum"] > 0
        examples = index.examples(layer, feature)
        assert len(examples) == 16
        assert all(row["activation"] > 0 for row in examples)
        assert len({row["document_id"] for row in examples}) == 16
        assert index.examples(layer, feature) == examples
        with pytest.raises(ValueError, match="empty"):
            fixture_resources(resource_folder)
    finally:
        index.close()


def test_remote_or_unpinned_model_resources_are_refused(resource_folder):
    for model in ("https://example.invalid/model", str(resource_folder)):
        with pytest.raises((ValueError, OSError)):
            load_resources(model, resource_folder)


def test_managed_callback_skips_each_internal_forward_scope(tmp_path):
    controls = _controls.Controls()
    steps = []
    managed = _managed.ManagedRun("fixture", str(tmp_path))
    managed._pkg = SimpleNamespace(_CTL=controls, step=lambda: steps.append(1))
    model = TinyLM()
    service = controls.interventions_on(model)
    try:
        managed._forward(model, (), None)
        with internal_forward():
            managed._forward(model, (), None)
        controls.interpretation_service = SimpleNamespace(running=True)
        managed._forward(model, (), None)
        controls.interpretation_service = None
        service.executor.preparing_source = True
        managed._forward(model, (), None)
        service.executor.preparing_source = False
        assert steps == [1]
    finally:
        service.remove()


def test_safe_checkpoint_restores_bindings_and_counter(resource_folder, tmp_path):
    run = HostedInference("tiny-lm", resource_folder, "AB")
    controls = _controls.Controls()
    service = controls.interventions_on(run.model)
    service.configure(dictionary=run.dictionary, tokenizer=run.tokenizer)
    run.interventions = service
    run.selection = {"layer": 0, "feature": 1, "position": 1}
    service.set({"kind": "clamp", "point": "layers.0.mlp_out", "feature": 1, "value": 0., "positions": [1]})
    controls.step(7)
    managed = _managed.ManagedRun("parent", str(tmp_path))
    managed._pkg = SimpleNamespace(_CTL=controls)
    managed.inference = run
    # This initial check needs no text-control dependency.
    controls.params["prompt"] = SimpleNamespace(value="AB")
    try:
        path = Path(managed.checkpoint(run.model, None))
        record = json.loads(path.read_bytes())
        assert path.suffix == ".json" and record["step"] == 7
        assert "model" not in record and "optimizer" not in record
        assert record["interventions"]["interventions"][0]["state"] == "active"
        fork = _managed.ManagedRun("fork", str(tmp_path), resume=str(path))
        fork.inference = run
        other = _controls.Controls()
        fork._load(SimpleNamespace(_CTL=other), run.model, None)
        assert other.n == 6 and fork.step == 7
        assert fork.values["prompt"] == "AB"
        assert run.restored["bindings"] == run._bindings()
        record["bindings"]["model_hash"] = "wrong"
        with pytest.raises(ValueError, match="bindings"):
            run.load_checkpoint(record)
    finally:
        controls.params.clear()
        service.remove()
        run.close()


class HostedChild:
    def __init__(self, root, dictionary, name, *, resume=None):
        self.folder = root / name
        self.folder.mkdir()
        self.ready = threading.Event()
        self.lines = []
        env = {k: v for k, v in os.environ.items() if not k.startswith("CALIPER_RUN_")}
        env.update(CALIPER_RUN_ID=name, CALIPER_RUN_DIR=str(self.folder), CALIPER_RUN_LINK_FILE=str(self.folder / "link"),
                   CALIPER_RUN_APP_PID=str(os.getpid()), CALIPER_AGENT_OFFLINE="1", CALIPER_TEST_RUN="1",
                   CALIPER_NETIO="0", CALIPER_SESSION="0", PYTHONUNBUFFERED="1")
        if resume is not None:
            env.update(CALIPER_RUN_RESUME=str(resume), CALIPER_RUN_PARENT="parent")
        self.process = subprocess.Popen([sys.executable, "-m", "caliper.interp._hosted", "--model=tiny-lm",
                                         "--dictionary=" + str(dictionary), "--prompt=AB"],
                                        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        def read_output():
            with (self.folder / "output.log").open("w") as log:
                for line in self.process.stdout:
                    log.write(line)
                    log.flush()
                    self.lines.append(line)
                    if line.startswith("caliper: inference ready"):
                        self.ready.set()
            self.ready.set()
        self.reader = threading.Thread(target=read_output, daemon=True)
        self.reader.start()

    def connect(self):
        from websockets.sync.client import connect
        assert self.ready.wait(30), "hosted startup did not complete"
        assert any(s.startswith("caliper: inference ready") for s in self.lines), "".join(self.lines)
        link = urlsplit((self.folder / "link").read_text())
        token = parse_qs(link.query)["token"][0]
        self.socket = connect(f"ws://{link.netloc}{wire.PATH}?{wire.TOKEN_PARAM}={token}",
                              compression=None, open_timeout=10, close_timeout=1)
        self.socket.send(wire.encode(wire.Hello("viewer", [wire.CAP_CONTROLS, wire.CAP_INTERVENTIONS, wire.CAP_INTERPRETATION])))
        self.expect(wire.Welcome)
        self.request_id = 0
        return self.expect(wire.Controls, lambda message: message.run["state"] == "running")

    def expect(self, kind, predicate=lambda m: True):
        end = time.monotonic() + 20
        while time.monotonic() < end:
            message = wire.decode(self.socket.recv(timeout=max(.01, end - time.monotonic())))
            if isinstance(message, kind) and predicate(message):
                return message
        raise TimeoutError("hosted response did not arrive")

    def control(self, op, **kwargs):
        self.request_id += 1
        self.socket.send(wire.encode(wire.Control(self.request_id, op, **kwargs)))
        return self.expect(wire.ControlReply, lambda m: m.request_id == self.request_id and m.status != "queued")

    def checkpoint(self):
        reply = self.control("run", name="caliper.checkpoint")
        assert reply.status == "ok", reply.error
        path = Path(reply.message)
        return path, json.loads(path.read_bytes())

    def close(self):
        if hasattr(self, "socket"):
            self.socket.close()
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(10)
        self.reader.join(10)
        self.process.stdout.close()


def test_real_hosted_fork_carries_active_feature_and_text(resource_folder, tmp_path):
    parent = HostedChild(tmp_path, resource_folder, "parent")
    child = None
    try:
        first = parent.connect()
        assert first.run["step"] == 1
        assert "prompt" in {p["name"] for p in first.params}
        traced = parent.control("interpret", interpretation={"version": 1, "method": "trace.run",
                                "params": {"prompt": "AB", "max_targets": 16}})
        assert traced.status == "ok", traced.error
        from .test_intervene import explicit_forward
        _, activations = explicit_forward(ReferenceLM(), ReferenceTokenizer()("AB")["input_ids"])
        feature = int(activations[-1][0, -1].argmax())
        request = {"set": {"kind": "clamp", "point": "layers.3.mlp_out", "feature": feature,
                           "positions": [2], "value": 0.}}
        assert parent.control("intervene", intervene=request).status == "ok"
        validated = parent.control("interpret", interpretation={"version": 1, "method": "trace.validate", "params": {}})
        assert validated.status == "ok", validated.error
        assert validated.result["required"] == 10
        swept = parent.control("interpret", interpretation={"version": 1, "method": "patch.sweep", "params": {
            "clean_prompt": "AB", "corrupted_prompt": "AC", "metric": {"positive_token": 65, "negative_token": 66}}})
        assert swept.status == "ok", swept.error
        assert swept.result["layers"] == 4 and swept.result["positions"] == 3
        checkpoint, saved = parent.checkpoint()
        assert saved["interventions"]["interventions"][0]["state"] == "active"
        child = HostedChild(tmp_path, resource_folder, "fork", resume=checkpoint)
        restored = child.connect()
        assert restored.run["step"] == saved["step"]
        _, forked = child.checkpoint()
        assert forked["interventions"] == saved["interventions"]
        assert forked["bindings"] == saved["bindings"]
        assert forked["params"]["prompt"] == "AB"
        assert child.control("run", name="generate").status == "ok"
        _, generated = child.checkpoint()
        assert set(generated["params"]) >= {"output.baseline", "output.steered"}
        assert generated["step"] == forked["step"] + 2
        assert child.control("run", name="caliper.stop").status == "ok"
        assert child.process.wait(10) == 0
        assert parent.control("run", name="caliper.stop").status == "ok"
        assert parent.process.wait(10) == 0
    finally:
        if child is not None:
            child.close()
        parent.close()


def test_script_checkpoint_keeps_the_existing_tensor_format(tmp_path):
    model = torch.nn.Linear(3, 2)
    controls = _controls.Controls()
    controls.n = 5
    managed = _managed.ManagedRun("script", str(tmp_path))
    managed._pkg = SimpleNamespace(_CTL=controls)
    path = Path(managed.checkpoint(model, None))
    assert path.suffix == ".pt"
    record = torch.load(path, weights_only=True)
    assert record["format"] == _managed.FORMAT
    restored = torch.nn.Linear(3, 2)
    fork = _managed.ManagedRun("fork", str(tmp_path), resume=str(path))
    fork._load(SimpleNamespace(_CTL=_controls.Controls()), restored, None)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    assert fork.step == 5


def test_local_checkpoint_supports_vectors_without_executable_objects():
    from caliper.interp._hosted import checkpoint_metadata
    assert checkpoint_metadata({"vector": [0.] * 1024})
    assert not checkpoint_metadata({"model": TinyLM()})
    assert not checkpoint_metadata({"vector": [float("nan")]})
    assert not checkpoint_metadata({"prompt": "A" * 4097})


def test_resource_stop_remains_latched_after_a_caught_action_error():
    from caliper.interp._hosted import _RunGuard
    from caliper.interp._memory import CleanStop
    class Limit:
        def check(self, *a, **kw):
            raise CleanStop("fixture memory limit")
    guard = _RunGuard(Limit())
    with pytest.raises(CleanStop):
        guard.check("forward")
    with pytest.raises(CleanStop):
        guard.check("boundary", force=False)


def test_real_hosted_controls_update_bars_and_retrace_only_on_request(tmp_path, monkeypatch):
    import caliper
    from caliper import _net
    from caliper.interp import load_dictionary
    from caliper.interp._index import FeatureIndex, build_index
    from caliper.interp._publish import _resolve
    from .test_dictionary import write_fixture
    from .test_service import Transport

    # The explicit injection seam supplies an exact dictionary with visible graph features.
    model = ReferenceLM()
    with torch.no_grad():
        for block in model.layers:
            block.mlp.W_dec.mul_(8)
    folder = tmp_path / "dictionary"
    write_fixture(folder, model)
    tokenizer = ReferenceTokenizer()
    dictionary = load_dictionary(folder, model=model)
    build_index(folder, [("fixture", tokenizer.encode("AB CD AB CD"))], model=model, tokenizer=tokenizer,
                dictionary=dictionary, provenance={"partition": "fixture"}, context_length=32)
    index = FeatureIndex(folder, model=model, tokenizer=tokenizer, dictionary=dictionary)
    controls = _controls.Controls()
    monkeypatch.setattr(caliper, "_CTL", controls)
    monkeypatch.setattr(_controls, "CONTROLS", controls)
    monkeypatch.setenv("CALIPER_NETIO", "0")
    run_folder = tmp_path / "managed"
    run_folder.mkdir()
    managed = _managed.ManagedRun("controls", str(run_folder), str(run_folder / "link"))
    monkeypatch.setattr(_managed, "current", lambda: managed)
    monkeypatch.setattr(_managed, "watch_parent", lambda: None)
    run = HostedInference("tiny-lm", folder, "AB", resources=(model, tokenizer, dictionary, index, None))
    transport = Transport()
    route = controls.attach(transport)
    try:
        run.start()
        trace_calls = []
        tracer = run.service._resource("tracer")
        trace = tracer.trace
        def traced(*a, **kw):
            trace_calls.append(1)
            return trace(*a, **kw)
        monkeypatch.setattr(tracer, "trace", traced)
        replies = []
        run.service.submit({"version": 1, "method": "trace.run", "params": {"prompt": "AB", "max_targets": 32}},
                           lambda *args: replies.append(args))
        run.boundary()
        assert replies[-1][0] == "ok", replies
        node = next(i for i, n in enumerate(run.service.graph.nodes) if n["kind"] == "feature")
        baseline = _resolve("output.probs").clone()
        assert route.on_control(None, wire.Control(1, "set", name="selected_node", value=node).to_meta())["status"] == "ok"
        run.boundary()
        assert not run.interventions.specifications()
        assert route.on_control(None, wire.Control(2, "set", name="clamp", value=0.).to_meta())["status"] == "ok"
        run.boundary()
        assert len(trace_calls) == 1
        assert run.interventions.specifications()[0]["value"] == 0.
        assert not torch.equal(baseline, _resolve("output.probs"))
        saved = run.interventions.state_dict()
        result = run.validate()
        assert result["trace_id"] == run.service.graph.trace_id
        assert run.interventions.state_dict() == saved
        assert len(trace_calls) == 1
        run.retrace.set(True)
        run.boundary()
        assert len(trace_calls) == 2
        run.retrace.set(False)
        run.reset()
        assert not run.interventions.specifications()
        run.boundary()
        assert len(trace_calls) == 2
        assert torch.equal(baseline, _resolve("output.probs"))
        run.clamp.set(2.)
        run.boundary()
        spec = run.interventions.specifications()[0]
        assert spec["value"] == 2 * index.show(run.selection["layer"], run.selection["feature"])["maximum"]
        assert spec["positions"] == [run.selection["position"]]
        assert run.controls.n == 7
        run.prompt.set("AC")
        run.boundary()
        assert len(trace_calls) == 2
        assert run.output.prompt == "AC"
    finally:
        run.close()
        route.detach()
        server = _net.current_server()
        if server is not None:
            server.close()
