"""CPU checks for hosted generation and resource ownership."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest
import torch

from caliper import _controls
from caliper.interp._intervene import InterventionError
from caliper.interp._output import bounded_text
from caliper.interp._publish import _resolve
from tests.interp.dictionary_check_fixture import dictionary_checks
from .concurrency_isolation import isolated_test
from .fixture import TinyLM, TinyTokenizer
from .test_controls_concurrency import other_thread
from .test_hosted import hosted_run
from .test_hosted_controls import _feature_rows, _queue_action
from .test_hosted_watch_ownership import assert_unchanged, watch_snapshot


@pytest.fixture(autouse=True)
def local_transport(monkeypatch):
    monkeypatch.setattr("caliper.serve", lambda: SimpleNamespace(url="in-process"))
    monkeypatch.setattr(_controls._Timers, "at", lambda *args: None)


class ReversedTokenizer(TinyTokenizer):
    def encode(self, text, add_special_tokens=True):
        return super().encode(text[::-1], add_special_tokens=add_special_tokens)


def select_feature(run):
    run.selected.set(_feature_rows(run)[0])
    run.boundary()
    run.clamp.set(1.)
    run._clamp_version = run.clamp._version


def pending_effects(run, service, *, patch=False):
    spec = ({"kind": "patch", "point": "layers.1.resid_pre", "source_prompt": "AC", "source_position": 1}
            if patch else {"kind": "clamp", "point": "layers.0.mlp_out", "feature": 1, "value": 3.})
    service.set(spec)
    with run.controls.lock:
        run.controls._apply_interventions()
    record = run.controls._ivs[(spec["point"], spec["kind"])]
    assert record.state == "active" and record.hook is not None
    service.set(dict(spec, **({"source_prompt": "AD"} if patch else {"value": 4.})))
    service.set({"kind": "ablate", "point": "layers.0.resid_pre", "from_step": run.controls.n + 20})
    assert record.replace and record.spec != record.applied_spec


def replace_watch(run):
    model = TinyLM()
    model.load_state_dict(run.model.state_dict())
    dictionary = deepcopy(run.dictionary)
    identifier = run.interventions.executor.dictionary_id
    service = run.controls.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=run.tokenizer, dictionary_id=identifier)
    assert model is not run.model and dictionary is not run.dictionary
    pending_effects(run, service)
    return service


def reconfigure(run, resource):
    service = run.interventions
    dictionary, tokenizer = run.dictionary, run.tokenizer
    if resource == "dictionary":
        dictionary = deepcopy(dictionary)
        with torch.no_grad():
            for layer in dictionary.layers:
                layer.W_dec.neg_()
        assert dictionary is not run.dictionary
        assert not torch.equal(dictionary.layers[0].W_dec, run.dictionary.layers[0].W_dec)
    else:
        tokenizer = ReversedTokenizer()
        assert tokenizer.encode("AC") != run.tokenizer.encode("AC")
    identifier = service.executor.dictionary_id
    service.configure(dictionary=dictionary, tokenizer=tokenizer, dictionary_id=identifier)
    assert run.controls._iv_model is service and service._operation_owner is None
    assert service.executor.dictionary_id == identifier
    assert run.output.model is run.service.model is run.model
    assert run.output.tokenizer is run.service.tokenizer is run.tokenizer
    assert run.service.dictionary is run.dictionary and run.service.index is run.index
    pending_effects(run, service, patch=resource == "tokenizer")
    return service


def output_snapshot(run):
    names = ("output.probs", "output.tokens", "output.label_text")
    return {
        "step": run.controls.n,
        "hook_calls": tuple(iv.hook.calls for iv in run.controls._ivs.values() if iv.hook is not None),
        "generation": deepcopy(run._generation), "comparison": deepcopy(run._comparison),
        "texts": (run.baseline_text.value, run.steered_text.value),
        "versions": (run.baseline_text._version, run.steered_text._version),
        "output_key": deepcopy(run._output_key), "action_id": run._output_action_id,
        "tensors": tuple(_resolve(name) for name in names),
        "values": tuple(_resolve(name).clone() for name in names),
    }


def assert_output_unchanged(run, before):
    after = output_snapshot(run)
    tensors, values = after.pop("tensors"), after.pop("values")
    assert all(actual is expected for actual, expected in zip(tensors, before["tensors"]))
    for actual, expected in zip(values, before["values"]):
        assert torch.equal(actual, expected)
    assert after == {key: value for key, value in before.items() if key not in ("tensors", "values")}


@pytest.mark.parametrize("changed_prompt", [False, True])
def test_stale_generation_preserves_replacement_and_output(hosted_run, monkeypatch, changed_prompt):
    run, _ = hosted_run
    run.generate()
    if changed_prompt:
        run.prompt.set("AC")
    replacement = replace_watch(run)
    before, output = watch_snapshot(run, replacement), output_snapshot(run)
    calls = []
    original = run.output.generate

    def generate(prompt):
        calls.append(prompt)
        return original(prompt)

    monkeypatch.setattr(run.output, "generate", generate)
    try:
        with pytest.raises(ValueError, match="no longer attached"):
            run.generate()
        assert not calls
        assert_unchanged(run, replacement, before)
        assert_output_unchanged(run, output)
    finally:
        replacement.remove()


def test_stale_generation_has_one_terminal_action_error(hosted_run, monkeypatch):
    run, route = hosted_run
    run.generate()
    action = run.controls.actions["generate"]
    generate = action.fn
    replacements = []

    def detached():
        replacement = replace_watch(run)
        replacements.append(replacement)
        before, output = watch_snapshot(run, replacement), output_snapshot(run)
        try:
            return generate()
        finally:
            assert_unchanged(run, replacement, before)
            assert_output_unchanged(run, output)

    monkeypatch.setattr(action, "fn", detached)
    _queue_action(route, "generate", 11)
    try:
        with pytest.warns(RuntimeWarning, match="no longer attached"):
            run.boundary()
        reply, = [reply for reply in route.transport.replies if reply.request_id == 11]
        assert reply.status == "error" and reply.error["code"] == "action_failed"
        assert "no longer attached" in reply.error["message"]
        assert not action.queued and not action.running
    finally:
        for replacement in replacements:
            replacement.remove()


@pytest.mark.parametrize("resource", ["dictionary", "tokenizer"])
@pytest.mark.parametrize("operation", ["clamp", "comparison", "generation"])
def test_completed_reconfiguration_refuses_mixed_resources(hosted_run, resource, operation):
    run, _ = hosted_run
    select_feature(run)
    run.generate()
    service = reconfigure(run, resource)
    calls = []
    observer = run.model.register_forward_pre_hook(lambda *args: calls.append(1))
    before, output = watch_snapshot(run, service), output_snapshot(run)
    try:
        invoke = {"clamp": lambda: run._set_clamp(require_commit=True),
                  "comparison": lambda: run.update_output(force=True), "generation": run.generate}[operation]
        with pytest.raises(ValueError, match="resources no longer match"):
            invoke()
        assert not calls
        assert_unchanged(run, service, before)
        assert_output_unchanged(run, output)
        assert service._readers == 0
    finally:
        observer.remove()


@pytest.mark.parametrize("caller_id", [False, True])
def test_same_resources_allow_reconfiguration_without_extra_validation(dictionary_checks, hosted_run, caller_id):
    run, _ = hosted_run
    select_feature(run)
    service = run.interventions
    identifier = "hosted-fixture" if caller_id else None
    service.configure(dictionary=run.dictionary, tokenizer=run.tokenizer, dictionary_id=identifier)
    before = dictionary_checks(service.executor)
    run.apply_clamp()
    assert dictionary_checks(service.executor) - before == (0 if caller_id else 2)
    owned = run._owned_record()
    hook = owned.hook
    before = dictionary_checks(service.executor)
    run.apply_clamp()
    assert dictionary_checks(service.executor) - before == (0 if caller_id else 1)
    assert run._owned_record() is owned and owned.hook is hook and not hook.removed
    before = dictionary_checks(service.executor)
    calls = []
    observer = run.model.register_forward_pre_hook(lambda *args: calls.append(1))
    try:
        run.update_output(force=True)
        assert calls == [1]
        run.generate()
        assert calls == [1] * 25
    finally:
        observer.remove()
    assert dictionary_checks(service.executor) == before
    assert run._generation["counts"] == [24, 24]
    assert all(len(row) == 24 for row in run._generation["token_ids"])
    assert service._readers == 0


@pytest.mark.parametrize("phase", ["work", "baseline", "steered", "metadata"])
@pytest.mark.parametrize("operation", ["watch", "dictionary", "tokenizer"])
@isolated_test("hosted_run", "monkeypatch")
def test_generation_pins_resources_through_publication(hosted_run, monkeypatch, phase, operation):
    run, _ = hosted_run
    service = run.interventions
    model = TinyLM()
    model.load_state_dict(run.model.state_dict())
    dictionary = deepcopy(run.dictionary)
    attempts, calls = [], []
    generation_id = run._generation["id"]

    def attempt():
        def replace():
            assert run.controls.lock.acquire(blocking=False), "generation holds Controls.lock"
            run.controls.lock.release()
            with pytest.raises(InterventionError, match="lifecycle operation is in progress"):
                if operation == "watch":
                    run.controls.interventions_on(model)
                else:
                    service.configure(dictionary=dictionary if operation == "dictionary" else run.dictionary,
                                      tokenizer=ReversedTokenizer() if operation == "tokenizer" else run.tokenizer,
                                      dictionary_id=service.executor.dictionary_id)
            attempts.append(operation)
        other_thread(replace)

    generate, script_set, hashes = run.output.generate, run.controls._script_set, run._input_hashes

    def generated(prompt):
        attempt()
        return generate(prompt)

    def published(param, value):
        result = script_set(param, value)
        if param is (run.baseline_text if phase == "baseline" else run.steered_text):
            assert run._generation["id"] == generation_id
            attempt()
        return result

    def hashed(state):
        assert run._generation["id"] == generation_id
        attempt()
        return hashes(state)

    observer = run.model.register_forward_pre_hook(lambda *args: calls.append(1))
    before = service.executor._dictionary_status()
    try:
        if phase == "work":
            monkeypatch.setattr(run.output, "generate", generated)
        elif phase == "metadata":
            monkeypatch.setattr(run, "_input_hashes", hashed)
        else:
            monkeypatch.setattr(run.controls, "_script_set", published)
        run.generate()
        assert attempts == [operation] and calls == [1] * 24
        assert run._generation["id"] == generation_id + 1
        assert run._generation["counts"] == [24, 24]
        texts = [run.baseline_text.value, run.steered_text.value]
        assert texts == [bounded_text(run.tokenizer.decode(row)) for row in run._generation["token_ids"]]
        assert run._generation["text_hashes"] == [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        assert service.executor._dictionary_status() == before
        assert run.controls._iv_model is service and service._readers == 0
    finally:
        observer.remove()


def test_invalid_model_dictionary_already_preserves_resources(hosted_run):
    run, _ = hosted_run
    service = run.interventions
    pending_effects(run, service)
    before, output = watch_snapshot(run, service), output_snapshot(run)
    dictionary = deepcopy(run.dictionary)
    dictionary.metadata["model_hash"] = "incompatible-model"
    with pytest.raises(InterventionError, match="dictionary model identity mismatch"):
        service.configure(dictionary=dictionary, tokenizer=run.tokenizer,
                          dictionary_id=service.executor.dictionary_id)
    assert service.executor.dictionary is run.dictionary
    assert_unchanged(run, service, before)
    assert_output_unchanged(run, output)
