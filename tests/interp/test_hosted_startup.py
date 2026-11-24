"""Event-controlled status requests before hosted prompt setup."""
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from caliper import _controls, _managed, _wire
from caliper.interp._hosted import HostedInference
from caliper.interp._intervene import InterventionService
from .fixture import make_fixture
from .test_dictionary import fixture_manifest
from .test_intervene import dictionary_for
from .test_service import Transport, call


@pytest.mark.parametrize("phase", ["watch", "dictionary", "prompt"])
def test_status_is_bounded_before_prompt_setup_and_atomic_at_readiness(monkeypatch, tmp_path, phase):
    import caliper
    model, tokenizer = make_fixture()
    dictionary = dictionary_for(model)
    dictionary.metadata = fixture_manifest(model)
    index = SimpleNamespace(metadata={"dictionary_hash": "fixture", "tokenizer_hash": "fixture"},
                            close=lambda: None)
    controls = _controls.Controls()
    monkeypatch.setattr(caliper, "_CTL", controls)
    monkeypatch.setattr(_managed, "current", lambda: None)
    run = HostedInference("tiny-lm", tmp_path, "AB", resources=(model, tokenizer, dictionary, index, None))
    entered, release, answered = (threading.Event() for _ in range(3))
    errors, replies = [], []

    def gate(at):
        if at == phase:
            entered.set()
            assert release.wait(5), "startup gate did not open"

    def watch(*args, **kwargs):
        assert controls.interpretation_service is run.service
        gate("watch")
        interventions = controls.interventions_on(model)
        return SimpleNamespace(remove=interventions.remove)

    prepare = InterventionService._prepare_dictionary_identity
    def identity(executor):
        gate("dictionary")
        prepare(executor)

    def param(name, value, **kwargs):
        if name == "selected_node":
            gate("prompt")
        return controls.param(name, value, **kwargs)

    monkeypatch.setattr(caliper, "watch", watch)
    monkeypatch.setattr(caliper, "param", param)
    monkeypatch.setattr(caliper, "action", controls.action)
    monkeypatch.setattr(InterventionService, "_prepare_dictionary_identity", identity)
    monkeypatch.setattr(run, "boundary", lambda: None)
    route = controls.attach(Transport())

    def invoke(function):
        try:
            function()
        except BaseException as error:
            errors.append(error)

    def status():
        try:
            replies.append(call(route, "trace.status"))
        finally:
            answered.set()

    starter = threading.Thread(target=lambda: invoke(run.start))
    reader = threading.Thread(target=lambda: invoke(status))
    try:
        starter.start()
        assert entered.wait(5), "startup did not reach the selected gate"
        reader.start()
        assert answered.wait(5), "status waited for startup completion"
        assert not errors
        assert replies[0]["status"] == "ok"
        frame = _wire.decode(_wire.encode_frame(_wire.MsgType.CONTROL_REPLY, replies[0]))
        early = frame.result["inference"]
        assert early["state"] == "starting"
        assert early["prompt_hash"] is early["interventions_hash"] is None
        assert early["comparison"] is early["dictionary_validation"] is None
        assert "AB" not in json.dumps(early) and len(json.dumps(early)) < 1024
        early["generation"]["token_ids"][0].append(123)
        assert run.inference_status()["generation"]["token_ids"] == [[], []]
        release.set()
        starter.join(5)
        assert not starter.is_alive() and not errors
        ready = call(route, "trace.status")["result"]["inference"]
        assert ready["state"] == "ready"
        assert ready["prompt_hash"] == hashlib.sha256(b"AB").hexdigest()
        assert ready["dictionary_validation"]["mode"] == "snapshot"
        assert ready["generation"]["token_ids"] == [[], []]
        assert not controls.queue
    finally:
        release.set()
        if starter.ident is not None:
            starter.join(5)
        if reader.ident is not None:
            reader.join(5)
        run.close()
        route.detach()
        assert not starter.is_alive() and not reader.is_alive()
