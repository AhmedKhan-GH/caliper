"""Installed fixture identity, live hosted controls, and safe inference forks."""
from contextlib import ExitStack
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
from .fixture import TinyLM as ReferenceLM, TinyTokenizer as ReferenceTokenizer, forward_logits


@pytest.fixture(scope="module")
def resource_folder(tmp_path_factory):
    return fixture_resources(tmp_path_factory.mktemp("hosted-dictionary"))


def test_installed_fixture_has_the_reference_identity():
    from caliper.interp._dictionary import model_identity
    assert model_identity(TinyLM()) == model_identity(ReferenceLM())
    expected = torch.tensor(json.loads(Path(__file__).with_name("fixture_logits.json").read_text()),
                            dtype=torch.float64) / 1e8
    # The saved reference permits the same CPU kernel tolerance as test_fixture.
    torch.testing.assert_close(forward_logits(TinyLM()), expected, atol=1e-6, rtol=1e-6)
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


def test_trace_snapshot_uses_applied_effects_during_a_queued_replacement(resource_folder, monkeypatch):
    from caliper.interp._hosted import HostedService
    from caliper.interp._service import InterpretationService
    run = HostedInference("tiny-lm", resource_folder, "AB")
    controls = _controls.Controls()
    interventions = controls.interventions_on(run.model)
    interventions.configure(dictionary=run.dictionary, tokenizer=run.tokenizer)
    original = {"kind": "clamp", "point": "layers.0.mlp_out", "feature": 1, "value": 0.}
    interventions.set(original)
    controls.step(7)
    run.interventions = interventions
    run.prompt = SimpleNamespace(set=lambda value: None)
    run.selected = controls.param("selected_node", -1, min=-1, max=511)
    service = HostedService(run)

    def finish_trace(*args):
        # A request arrives after the trace forward, before the service records its snapshot.
        interventions.set(dict(original, value=2.))
        return {"state": "done"}

    monkeypatch.setattr(InterpretationService, "_execute", finish_trace)
    try:
        service._execute("trace.run", {"prompt": "AB"}, 7)
        assert interventions.state_dict()["interventions"][0]["specification"]["value"] == 2.
        assert run.traced_state[0]["value"] == 0.
        assert run.traced_state == interventions.specifications()
        assert controls.n == 7
    finally:
        service.close()
        interventions.remove()
        run.close()


def test_hosted_service_close_waits_for_output_and_refuses_recursive_close(resource_folder, monkeypatch):
    from caliper.interp._hosted import HostedService
    from caliper.interp._output import Output
    from caliper.interp._service import Refused
    run = HostedInference("tiny-lm", resource_folder, "AB")
    controls = _controls.Controls()
    interventions = controls.interventions_on(run.model)
    output = Output(run.model, run.tokenizer, interventions)
    service = HostedService(run)
    entered, release, waiting, finished = (threading.Event() for _ in range(4))
    errors, results = [], []
    wait = service.idle.wait

    def wait_for_boundary(*args):
        waiting.set()
        return wait(*args)

    def update_controls():
        with pytest.raises(Refused, match="hosted boundary"):
            service.close()

    def update_output():
        entered.set()
        assert release.wait(5), "output gate did not open"
        results.append(output.compare("AB"))

    def invoke(fn):
        try:
            fn()
        except BaseException as exc:
            errors.append(exc)

    def close():
        service.close()
        finished.set()

    monkeypatch.setattr(service.idle, "wait", wait_for_boundary)
    monkeypatch.setattr(run, "update_controls", update_controls)
    monkeypatch.setattr(run, "update_output", update_output)
    worker = threading.Thread(target=invoke, args=(service.step,))
    closer = threading.Thread(target=invoke, args=(close,))
    try:
        worker.start()
        assert entered.wait(5), errors
        closer.start()
        assert waiting.wait(5), errors
        assert not finished.is_set()
        release.set()
        worker.join(5)
        closer.join(5)
        assert not worker.is_alive() and not closer.is_alive()
        assert not errors and len(results) == 1
        assert finished.is_set() and service.closed
        assert controls.n == 0
    finally:
        release.set()
        worker.join(5)
        if closer.ident is not None:
            closer.join(5)
        interventions.remove()
        run.close()


class HostedChild:
    def __init__(self, root, dictionary, name, *, resume=None):
        self.folder = root / name
        self.folder.mkdir()
        self.ready = threading.Event()
        self.lines = []
        self.connections = ExitStack()
        env = {k: v for k, v in os.environ.items() if not k.startswith("CALIPER_RUN_")}
        env.update(CALIPER_RUN_ID=name, CALIPER_RUN_DIR=str(self.folder), CALIPER_RUN_LINK_FILE=str(self.folder / "link"),
                   CALIPER_RUN_APP_PID=str(os.getpid()), CALIPER_AGENT_OFFLINE="1", CALIPER_TEST_RUN="1",
                   CALIPER_DATA_DIR=str(self.folder / "data"),
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
        try:
            self.reader.start()
        except BaseException:
            self.close()
            raise

    def connect(self):
        from websockets.sync.client import connect
        assert self.ready.wait(30), "hosted startup did not complete"
        assert any(s.startswith("caliper: inference ready") for s in self.lines), "".join(self.lines)
        link = urlsplit((self.folder / "link").read_text())
        token = parse_qs(link.query)["token"][0]
        self.socket = self.connections.enter_context(connect(
            f"ws://{link.netloc}{wire.PATH}?{wire.TOKEN_PARAM}={token}",
            compression=None, open_timeout=10, close_timeout=1))
        self.socket.send(wire.encode(wire.Hello("viewer", [wire.CAP_CONTROLS, wire.CAP_TEXT_PARAMS,
                                                           wire.CAP_INTERVENTIONS, wire.CAP_INTERPRETATION])))
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
        try:
            self.connections.close()
        finally:
            try:
                try:
                    if self.process.poll() is None:
                        try:
                            self.process.terminate()
                        except ProcessLookupError:
                            pass
                finally:
                    try:
                        self.process.wait(10)
                    except subprocess.TimeoutExpired:
                        try:
                            self.process.kill()
                        except ProcessLookupError:
                            pass
                        self.process.wait()
            finally:
                try:
                    if self.reader.ident is not None:
                        self.reader.join(10)
                        assert not self.reader.is_alive(), "hosted output reader did not stop"
                finally:
                    self.process.stdout.close()


@pytest.mark.parametrize("socket_failure", [False, True])
def test_hosted_cleanup_reaps_exact_children_after_shutdown_timeout(monkeypatch, socket_failure):
    children = []

    def child():
        # This process waits for EOF. The test forces the terminate-timeout branch.
        owned = HostedChild.__new__(HostedChild)
        owned.connections = ExitStack()
        owned.process = subprocess.Popen(
            [sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.read()"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        children.append(owned)
        owned.reader = threading.Thread(target=lambda: owned.process.stdout.read(), daemon=True)
        assert owned.process.stdout.readline() == "ready\n"
        owned.reader.start()
        return owned

    def fail_socket():
        raise RuntimeError("fixture socket cleanup failure")

    try:
        with pytest.raises(RuntimeError, match="fixture socket cleanup failure") if socket_failure else ExitStack():
            with ExitStack() as cleanup:
                parent = child()
                cleanup.callback(parent.process.stdin.close)
                cleanup.callback(parent.close)
                fork = child()
                cleanup.callback(fork.process.stdin.close)
                cleanup.callback(fork.close)
                if socket_failure:
                    fork.connections.callback(fail_socket)
                wait = fork.process.wait
                calls = []

                def timeout_once(timeout=None):
                    calls.append(timeout)
                    if len(calls) == 1:
                        raise subprocess.TimeoutExpired(fork.process.args, timeout)
                    return wait(timeout)

                monkeypatch.setattr(fork.process, "wait", timeout_once)
                monkeypatch.setattr(fork.process, "terminate", lambda: None)
        assert calls == [10, None]
        assert all(owned.process.returncode is not None for owned in children)
    finally:
        # Independent emergency cleanup also protects assertions inside this test.
        with ExitStack() as cleanup:
            for owned in children:
                cleanup.callback(owned.process.stdin.close)
                cleanup.callback(owned.close)
    for owned in children:
        assert owned.process.returncode is not None
        assert not owned.reader.is_alive()
        assert owned.process.stdout.closed and owned.process.stdin.closed


def test_real_hosted_fork_carries_active_feature_and_text(resource_folder, tmp_path):
    with ExitStack() as cleanup:
        parent = HostedChild(tmp_path, resource_folder, "parent")
        cleanup.callback(parent.close)
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
        ablation = {"kind": "ablate", "point": "layers.3.mlp_out"}
        assert parent.control("intervene", intervene={"set": ablation}).status == "ok"
        assert parent.control("intervene", intervene={"set": {"kind": "add", "point": "layers.3.mlp_out",
                              "vector": ReferenceLM().lm_head.weight[65].detach().tolist(), "coefficient": 2.}}).status == "ok"
        assert parent.control("intervene", intervene={"set": dict(ablation, until_step=100)}).status == "ok"
        assert parent.control("run", name="generate").status == "ok"
        checkpoint, saved = parent.checkpoint()
        assert saved["interventions"]["interventions"][0]["state"] == "active"
        child = HostedChild(tmp_path, resource_folder, "fork", resume=checkpoint)
        cleanup.callback(child.close)
        restored = child.connect()
        assert restored.run["step"] == saved["step"]
        _, forked = child.checkpoint()
        assert forked["interventions"] == saved["interventions"]
        assert forked["bindings"] == saved["bindings"]
        assert forked["params"]["prompt"] == "AB"
        assert child.control("run", name="generate").status == "ok"
        _, generated = child.checkpoint()
        assert set(generated["params"]) >= {"output.baseline", "output.steered"}
        for name in ("output.baseline", "output.steered"):
            assert 0 < len(generated["params"][name].encode("utf-8")) <= 1024
            assert generated["params"][name] == saved["params"][name]
        assert generated["params"]["output.baseline"] != generated["params"]["output.steered"]
        assert generated["step"] == forked["step"] + 2
        assert child.control("run", name="caliper.stop").status == "ok"
        assert child.process.wait(10) == 0
        assert parent.control("run", name="caliper.stop").status == "ok"
        assert parent.process.wait(10) == 0


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


@pytest.mark.parametrize("bind_late,explicit_override", [(False, False), (True, False), (False, True)])
def test_script_fork_restores_outputs_and_defers_queued_lifecycle(tmp_path, monkeypatch, bind_late, explicit_override):
    import caliper
    from caliper import _net
    from caliper.interp._output import is_internal_forward
    from .test_intervene import dictionary_for

    parent_model, tokenizer = TinyLM(), TinyTokenizer()
    parent_controls = _controls.Controls()
    parent_service = parent_controls.interventions_on(parent_model)
    parent_service.configure(dictionary=dictionary_for(parent_model), tokenizer=tokenizer)
    parent = _managed.ManagedRun("script-parent", str(tmp_path))
    parent._pkg = SimpleNamespace(_CTL=parent_controls)
    inputs = tokenizer("AB")
    add = {"kind": "add", "point": "layers.3.mlp_out", "vector": [1.] * 64, "coefficient": .25}
    override = {"kind": "add", "point": "layers.2.mlp_out", "vector": [1.] * 64, "coefficient": .125}
    try:
        parent_service.set({"kind": "ablate", "point": "layers.3.mlp_out"})
        parent_service.set(add)
        parent_service.set({"kind": "clamp", "point": "layers.0.mlp_out", "feature": 0, "value": 2.})
        parent_service.set({"kind": "scale", "point": "layers.1.mlp_out", "feature": 1,
                            "multiplier": 2., "from_step": 7})
        parent_service.set({"kind": "ablate", "point": "layers.2.mlp_out", "until_step": 6})
        parent_service.set({"kind": "patch", "point": "layers.1.resid_pre", "source_prompt": "AC", "positions": [1]})
        parent_controls.step(5)
        parent_service.set(dict(add, coefficient=.75, from_step=8))
        parent_service.clear_feature("layers.0.mlp_out", 0)
        checkpoint = parent.checkpoint(parent_model, None)
        record = torch.load(checkpoint, weights_only=True)
        assert Path(checkpoint).suffix == ".pt" and record["format"] == _managed.FORMAT
        assert record["interventions"] == parent_service.state_dict()
        assert {r["state"] for r in record["interventions"]["interventions"]} == {"active", "pending", "ended"}
        expected, states = [], []
        for step in (None, 5, 6, 7):
            if step is not None:
                if step == 5 and explicit_override:
                    parent_service.set(override)
                parent_controls.step(step)
            states.append(parent_service.state_dict())
            # Apply only the explicit override with an independent tensor oracle.
            hook = parent_model.layers[2].mlp.register_forward_hook(lambda m, a, y: y + .125) \
                if step is None and explicit_override else None
            try:
                with internal_forward(), torch.no_grad():
                    expected.append(parent_model(**inputs).logits.clone())
            finally:
                if hook is not None:
                    hook.remove()
        assert not torch.equal(expected[0], expected[1])
        assert not torch.equal(expected[2], expected[3])
    finally:
        parent_service.remove()

    controls = _controls.Controls()
    model = TinyLM()
    folder = tmp_path / "script-fork"
    folder.mkdir()
    fork = _managed.ManagedRun("script-fork", str(folder), str(folder / "link"), resume=checkpoint,
                               interventions=[override] if explicit_override else None)
    monkeypatch.setattr(caliper, "_CTL", controls)
    monkeypatch.setattr(_controls, "CONTROLS", controls)
    monkeypatch.setattr(_managed, "current", lambda: fork)
    monkeypatch.setattr(_managed, "watch_parent", lambda: None)
    monkeypatch.setenv("CALIPER_NETIO", "0")
    handle = None
    try:
        handle = caliper.watch(model, activations=False)
        service = controls.intervention_service
        assert service.state_dict()["interventions"] == []
        if bind_late:
            with pytest.raises(ValueError, match="dictionary"):
                model(**inputs)
            with pytest.raises(ValueError, match="dictionary"):
                model(**inputs)
            assert fork._first is not None and controls.n == 4
        # Scripts bind resources after watch and before their first forward.
        service.configure(dictionary=dictionary_for(model), tokenizer=tokenizer)
        seen = []
        def before_forward(*args):
            if not is_internal_forward() and not service.executor.preparing_source:
                seen.append((controls.n, service.state_dict()))
        handle._handles.append(model.register_forward_pre_hook(before_forward))
        with torch.no_grad():
            actual = [model(**inputs).logits.clone() for _ in expected]
        for wanted, received in zip(expected, actual):
            torch.testing.assert_close(received, wanted, rtol=0, atol=0)
        assert [step for step, _ in seen] == [4, 5, 6, 7]
        def inherited(state):
            return [record for record in state["interventions"] if
                    (record["specification"]["point"], record["specification"]["kind"]) != (override["point"], override["kind"])]
        if explicit_override:
            assert [inherited(state) for _, state in seen] == [inherited(state) for state in states]
            assert any(record["applied_specification"] is not None
                       and record["applied_specification"].get("coefficient") == .125
                       for record in seen[0][1]["interventions"])
        else:
            assert [state for _, state in seen] == states
        assert controls.n == 8 and fork._first is None
    finally:
        if handle is not None:
            handle.remove()
        server = _net.current_server()
        if server is not None:
            server.close()


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


def test_hosted_service_exists_before_watch_starts_stream(monkeypatch, request):
    import caliper
    original = caliper.watch
    seen = []

    def watch(model, *args, **kwargs):
        service = caliper._CTL.interpretation_service
        assert service is not None and service.owner.model is model
        seen.append(service)
        return original(model, *args, **kwargs)

    monkeypatch.setattr(caliper, "watch", watch)
    run, _ = request.getfixturevalue("hosted_run")
    assert seen == [run.service]


@pytest.fixture
def hosted_run(tmp_path, monkeypatch, request):
    import caliper
    from caliper import _net
    from caliper.interp import load_dictionary
    from caliper.interp._index import FeatureIndex, build_index
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
    guard = SimpleNamespace(check=lambda *args, **kwargs: None) if getattr(request, "param", False) else None
    run = HostedInference("tiny-lm", folder, "AB", resources=(model, tokenizer, dictionary, index, guard))
    transport = Transport()
    route = controls.attach(transport)
    try:
        run.start()
        yield run, route
    finally:
        run.close()
        route.detach()
        server = _net.current_server()
        if server is not None:
            server.close()


def test_real_hosted_controls_update_bars_and_retrace_only_on_request(hosted_run, monkeypatch):
    from caliper.interp._publish import _resolve
    run, route = hosted_run
    index = run.index
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
    run.reset()
    assert not run.interventions.specifications()
    run.boundary()
    assert len(trace_calls) == 1
    assert torch.equal(baseline, _resolve("output.probs"))
    run.clamp.set(2.)
    run.boundary()
    spec = run.interventions.specifications()[0]
    assert spec["value"] == 2 * index.show(run.selection["layer"], run.selection["feature"])["maximum"]
    assert spec["positions"] == [run.selection["position"]]
    run.retrace.set(True)
    run.boundary()
    assert len(trace_calls) == 2
    assert run.controls.n == 7
    run.retrace.set(False)
    run.prompt.set("AC")
    run.boundary()
    assert len(trace_calls) == 2
    assert run.output.prompt == "AC"


def _trace_at_boundary(run, **params):
    replies = []
    run.service.submit({"version": 1, "method": "trace.run", "params": params}, lambda *args: replies.append(args))
    run.boundary()
    assert replies[-1][0] == "ok", replies
    return run.service.graph


@pytest.mark.parametrize("new_params,retained", [({"prompt": "AC", "max_targets": 32}, False),
    ({"prompt": "AB", "max_targets": 32, "max_nodes": 20}, False),
    ({"prompt": "AB", "max_targets": 32, "max_nodes": 20}, True)])
def test_new_trace_reconciles_semantic_selection_before_clamp_or_reset(hosted_run, new_params, retained):
    run, _ = hosted_run
    original = _trace_at_boundary(run, prompt="AB", max_targets=32)
    changed = _trace_at_boundary(run, **new_params)
    fields = ("layer", "feature", "position")
    changed_ids = {node["id"] for node in changed.nodes}
    candidates = [(i, node) for i, node in enumerate(original.nodes) if node["kind"] == "feature"
                  and (node["id"] in changed_ids) == retained
                  and (i >= len(changed.nodes) or changed.nodes[i]["id"] != node["id"])]
    assert candidates, "fixture must change a selected graph row"
    row, feature = candidates[0]
    _trace_at_boundary(run, prompt="AB", max_targets=32)
    run.selected.set(row)
    run.update_controls()
    selected = {key: feature[key] for key in fields}
    assert run.selection == selected
    _trace_at_boundary(run, **new_params)
    expected = next((i for i, node in enumerate(run.service.graph.nodes) if node["kind"] == "feature"
                     and all(node[key] == value for key, value in selected.items())), -1)
    assert run.selected.value == expected
    assert run.selection == (selected if expected >= 0 else None)
    assert (expected >= 0) == retained
    run.clamp.set(2.)
    if expected == -1:
        with pytest.raises(ValueError, match="selected feature"):
            run.update_controls()
        run.reset()
        assert not run.interventions.specifications()
    else:
        run.update_controls()
        spec, = run.interventions.specifications()
        assert spec["feature"] == selected["feature"] and spec["positions"] == [selected["position"]]
        run.reset()
        assert not run.interventions.specifications()


def test_reset_removes_owned_clamp_after_feature_disappears(hosted_run):
    run, _ = hosted_run
    graph = _trace_at_boundary(run, prompt="AB", max_targets=32)
    run.selected.set(next(i for i, node in enumerate(graph.nodes) if node["kind"] == "feature"))
    run.clamp.set(0.)
    run.boundary()
    saved = run.interventions.state_dict()
    _trace_at_boundary(run, prompt="AC", max_targets=32)
    assert run.selection is None and run.selected.value == -1
    assert saved["interventions"]
    run.reset()
    assert not run.interventions.specifications()
    assert run.state_dict()["owned_clamp"] is None


def test_reset_with_retrace_has_one_public_boundary(hosted_run, monkeypatch):
    run, route = hosted_run
    graph = _trace_at_boundary(run, prompt="AB", max_targets=32)
    row = next(i for i, node in enumerate(graph.nodes) if node["kind"] == "feature")
    run.selected.set(row)
    run.clamp.set(2.)
    run.boundary()
    run.retrace.set(True)
    run.boundary()
    assert run.selection is not None
    calls = []
    tracer = run.service._resource("tracer")
    trace = tracer.trace
    def traced(*args, **kwargs):
        calls.append(run.controls.n)
        return trace(*args, **kwargs)
    monkeypatch.setattr(tracer, "trace", traced)
    before = run.controls.n
    assert route.on_control(None, wire.Control(3, "run", name="reset").to_meta())["status"] == "queued"
    run.boundary()
    assert run.controls.n == before + 1
    assert calls == [before + 1]
    assert not run.interventions.specifications()
    assert not run.service.running and not run.service.boundary_running


@pytest.mark.parametrize("hosted_run", [True], indirect=True)
def test_hosted_resource_stop_blocks_patch_installation_and_replay(hosted_run, monkeypatch):
    from caliper.interp._memory import CleanStop
    run, _ = hosted_run
    def stop(*args, **kwargs):
        raise CleanStop("fixture hosted stop")
    monkeypatch.setattr(run.guard.guard, "check", stop)
    with pytest.raises(CleanStop, match="fixture hosted stop"):
        run.guard.check("test stop")
    calls = []
    hook = run.model.register_forward_pre_hook(lambda *args: calls.append(1))
    spec = {"kind": "patch", "point": "layers.1.resid_pre", "source_prompt": "AC"}
    try:
        from caliper.interp._intervene import InterventionError
        before = run.interventions.state_dict()
        with pytest.raises(InterventionError) as refused:
            run.interventions.set(spec)
        assert refused.value.code == "interpret.stopped"
        assert not calls and run.interventions.state_dict() == before
        assert not run.interventions.specifications()
        with pytest.raises(CleanStop, match="fixture hosted stop"):
            with run.interventions.temporary_state([spec]):
                pytest.fail("a stopped patch replay entered its body")
        assert not calls
    finally:
        hook.remove()
