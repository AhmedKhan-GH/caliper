"""Managed restore order across internal preparation and public inference."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

import caliper
from caliper import _controls, _managed
from caliper.interp import _replication_actions as published
from caliper.interp._hosted import HostedInference
from caliper.interp._output import Output, internal_forward
from .fixture import make_fixture
from .test_dictionary import fixture_manifest
from .test_intervene import dictionary_for, explicit_forward


def add(point, coefficient):
    return {"kind": "add", "point": point, "vector": [1.] * 8, "coefficient": coefficient}


SAVED = [add("layers.0.mlp_out", .25), add("layers.1.mlp_out", .5)]
OVERRIDE = add("layers.0.mlp_out", .75)


@pytest.fixture
def environment(monkeypatch):
    controls = _controls.Controls()
    monkeypatch.setattr(caliper, "_CTL", controls)
    monkeypatch.setattr(_controls, "CONTROLS", controls)
    monkeypatch.setattr(caliper, "serve", lambda: SimpleNamespace(url="in-process"))
    monkeypatch.setattr(_managed, "watch_parent", lambda: None)
    monkeypatch.setenv("CALIPER_NETIO", "0")
    monkeypatch.setenv("CALIPER_SESSION", "0")
    return controls


@pytest.fixture
def hosted_factory(environment, monkeypatch, tmp_path):
    # Only fixture coordinates and prompts change. Preparation and cases stay real.
    monkeypatch.setattr(published, "PROMPTS", dict(dallas="AB", oakland="CD", shanghai="EF", vancouver="GH"))
    monkeypatch.setattr(published, "GROUPS", {name: [(0, 0, 1)] for name in published.GROUPS})
    runs = []

    def create(name, *, resume=None, overrides=(), replication=True):
        model, tokenizer = make_fixture(layers=2, width=8, heads=2, latents=16, k=2)
        dictionary = dictionary_for(model)
        dictionary.metadata = fixture_manifest(model)
        closed = []
        index = SimpleNamespace(metadata={"dictionary_hash": "fixture", "tokenizer_hash": "fixture"},
                                close=lambda: closed.append(True))
        folder = tmp_path / name
        folder.mkdir()
        managed = _managed.ManagedRun(name, str(folder), str(folder / "link"), resume=resume,
                                      interventions=deepcopy(overrides))
        monkeypatch.setattr(_managed, "current", lambda: managed)
        run = HostedInference("tiny-lm", tmp_path / "dictionary", "AB",
                              resources=(model, tokenizer, dictionary, index, None),
                              replication_profile="published_dallas_v1" if replication else None,
                              private_output_directory=folder / "private")
        runs.append(run)
        return run, managed, closed

    yield create
    for run in reversed(runs):
        run.close()


def parent_checkpoint(create, *, replication=True):
    parent, managed, _ = create("parent", replication=replication)
    parent.start()
    for spec in SAVED:
        parent.interventions.set(spec)
    parent.boundary()
    checkpoint = managed.checkpoint(parent.model, None)
    parent.close()
    return checkpoint


def manual_logits(model, tokenizer, specifications):
    handles = []
    try:
        for spec in specifications:
            block = model.layers[int(spec["point"].split(".")[1])]
            if spec["kind"] == "ablate":
                handles.append(block.mlp.register_forward_hook(lambda m, a, out: torch.zeros_like(out)))
            else:
                handles.append(block.mlp.register_forward_hook(
                    lambda m, a, out, value=spec["coefficient"]: out + value))
        with torch.no_grad():
            return model(**tokenizer("AB")).logits[0, -1].clone()
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("overrides", [[], [OVERRIDE], [{"kind": "ablate", "point": "layers.0.mlp"}]])
def test_published_fork_restores_before_first_boundary(hosted_factory, monkeypatch, saved, overrides):
    checkpoint = parent_checkpoint(hosted_factory) if saved else None
    run, managed, _ = hosted_factory("child", resume=checkpoint, overrides=overrides)
    expected_specs = deepcopy(SAVED) if saved else []
    for spec in overrides:
        expected_specs = [old for old in expected_specs if
                          _controls.intervention_key(old) != _controls.intervention_key(spec)]
        expected_specs.append(spec)
    baseline = manual_logits(run.model, run.tokenizer, [])
    if not saved and overrides and overrides[0]["kind"] == "ablate":
        # PublishedCases retains baseline attention for a module-only ablation.
        with run.model.frozen_attention():
            expected = manual_logits(run.model, run.tokenizer, expected_specs)
    else:
        expected = manual_logits(run.model, run.tokenizer, expected_specs)
    source_values = {}
    with torch.no_grad():
        for source, prompt in published.PROMPTS.items():
            _, values = explicit_forward(run.model, run.tokenizer(prompt)["input_ids"])
            source_values[source] = float(values[0][0, 1, 0])
    forwards, boundaries, outputs = [], [], []
    forward = published.forward_logits
    boundary = run.boundary
    publish = Output.publish_logits

    def prepare_forward(*args, **kwargs):
        service = run.controls.intervention_service
        assert service._operation_owner[0] == "preparation"
        assert service.state_dict()["interventions"] == []
        first = managed._first
        assert (first is not None) == bool(overrides)
        result = forward(*args, **kwargs)
        assert managed._first is first
        assert service.state_dict()["interventions"] == []
        forwards.append(True)
        return result

    def first_boundary():
        assert len(forwards) == 4
        assert managed._first is None
        assert run._inference_ready
        assert (run.restored is not None) == saved
        records = run.interventions.state_dict()["interventions"]
        actual = [record["specification"] for record in records]
        assert {json.dumps(s, sort_keys=True) for s in actual} == {
            json.dumps(dict(s, op=s["kind"]), sort_keys=True) for s in expected_specs}
        boundaries.append(run.controls.n)
        return boundary()

    def publish_logits(output, logits):
        assert boundaries and run._inference_ready
        outputs.append(logits.detach().clone())
        return publish(output, logits)

    monkeypatch.setattr(published, "forward_logits", prepare_forward)
    monkeypatch.setattr(run, "boundary", first_boundary)
    monkeypatch.setattr(Output, "publish_logits", publish_logits)
    run.start()
    assert len(boundaries) == len(outputs) == 1
    torch.testing.assert_close(outputs[0][0], baseline, rtol=0, atol=1e-12)
    torch.testing.assert_close(outputs[0][1], expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(run.replication.baseline_logits, baseline, rtol=0, atol=1e-12)
    assert {source: values[(0, 0, 1)] for source, values in run.replication.values.items()} == source_values
    assert run.controls.n == boundaries[0] + 1
    assert managed._restored_interventions is _managed.NONE
    run.close()
    assert managed.inference is None
    assert not run.controls.params and not run.controls.actions
    assert run.controls.intervention_service is None
    assert run.controls.interpretation_service is None
    assert all(not module._forward_pre_hooks and not module._forward_hooks for module in run.model.modules())


def test_unpublished_hosted_fork_keeps_override_precedence(hosted_factory):
    checkpoint = parent_checkpoint(hosted_factory, replication=False)
    run, managed, _ = hosted_factory("child", resume=checkpoint, overrides=[OVERRIDE], replication=False)
    run.start()
    assert managed._first is None and run.replication is None
    assert run.interventions.specifications() == [dict(OVERRIDE, op="add"), dict(SAVED[1], op="add")]


@pytest.mark.parametrize("failure", ["saved", "override"])
def test_hosted_restore_refusal_propagates_and_cleans_up(hosted_factory, monkeypatch, failure):
    checkpoint = parent_checkpoint(hosted_factory)
    if failure == "saved":
        from pathlib import Path
        record = json.loads(Path(checkpoint).read_text())
        record["interventions"]["version"] = 2
        Path(checkpoint).write_text(json.dumps(record))
    overrides = [{"kind": "ablate", "point": "missing"}] if failure == "override" else [OVERRIDE]
    run, managed, closed = hosted_factory("child", resume=checkpoint, overrides=overrides)
    monkeypatch.setattr(run, "boundary", lambda: pytest.fail("a refused fork reached its first boundary"))
    with pytest.raises((RuntimeError, ValueError), match="unsupported intervention state|cannot apply"):
        run.start()
    assert closed and managed.inference is None
    assert managed._first is not None
    assert managed.interventions == overrides
    assert not run._inference_ready
    assert not run.controls.params and not run.controls.actions
    assert run.controls.intervention_service is None
    assert run.controls.interpretation_service is None
    assert not run.replication._handles and not run.replication.patterns
    assert all(not module._forward_pre_hooks and not module._forward_hooks for module in run.model.modules())


@pytest.mark.parametrize("mode", ["saved", "override", "both"])
@pytest.mark.parametrize("explicit", [False, True])
def test_script_internal_forward_retains_restore_until_requested(environment, monkeypatch, tmp_path, mode, explicit):
    model, tokenizer = make_fixture(layers=2, width=8, heads=2, latents=16, k=2)
    controls = environment
    baseline = manual_logits(model, tokenizer, [])
    wanted = SAVED[:1] if mode == "saved" else [OVERRIDE]
    expected = manual_logits(model, tokenizer, wanted)
    checkpoint = None
    if mode != "override":
        service = controls.interventions_on(model)
        service.configure(dictionary=dictionary_for(model), tokenizer=tokenizer)
        service.set(SAVED[0])
        controls.step(5)
        parent = _managed.ManagedRun("parent", str(tmp_path))
        parent._pkg = SimpleNamespace(_CTL=controls)
        checkpoint = parent.checkpoint(model, None)
        service.remove()
    managed = _managed.ManagedRun("child", str(tmp_path), str(tmp_path / "link"), resume=checkpoint,
                                  interventions=[] if mode == "saved" else [OVERRIDE])
    monkeypatch.setattr(_managed, "current", lambda: managed)
    with caliper.watch(model, activations=False) as handle:
        service = controls.intervention_service
        first, pending, step = managed._first, managed._restored_interventions, controls.n
        with internal_forward(), internal_forward(), torch.no_grad():
            torch.testing.assert_close(model(**tokenizer("AB")).logits[0, -1], baseline, rtol=0, atol=1e-12)
        assert managed._first is first and managed._restored_interventions is pending
        assert controls.n == step and not service.state_dict()["interventions"]
        service.configure(dictionary=dictionary_for(model), tokenizer=tokenizer)
        if explicit:
            with internal_forward():
                managed._intervene()
            assert managed._first is None and controls.n == step
        with torch.no_grad():
            actual = model(**tokenizer("AB")).logits[0, -1]
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
        assert managed._first is None and managed._restored_interventions is _managed.NONE
        assert controls.n == step + 1
        assert service.specifications() == [dict(wanted[0], op="add")]
        assert handle.active
    assert controls.intervention_service is None and not controls.actions
    assert all(not module._forward_pre_hooks and not module._forward_hooks for module in model.modules())


@pytest.mark.parametrize("saved", [False, True])
def test_script_refusal_retains_pending_restore_and_explicit_calls(environment, monkeypatch, tmp_path, saved):
    from caliper.interp._intervene import InterventionError
    model, tokenizer = make_fixture(layers=2, width=8, heads=2, latents=16, k=2)
    managed = _managed.ManagedRun("child", str(tmp_path), str(tmp_path / "link"))
    if saved:
        # A feature effect needs the dictionary that the script binds after watch.
        spec = {"kind": "clamp", "point": "layers.0.mlp_out", "feature": 0, "value": 0.}
        managed._restored_interventions = {"version": 1, "interventions": [
            {"specification": spec, "state": "active", "since_step": 1, "clear": False,
             "error": None, "error_code": None}]}
    else:
        managed.interventions = [{"kind": "ablate", "point": "missing"}]
    monkeypatch.setattr(_managed, "current", lambda: managed)
    with caliper.watch(model, activations=False):
        first, pending = managed._first, managed._restored_interventions
        with internal_forward(), torch.no_grad():
            model(**tokenizer("AB"))
            with pytest.raises(RuntimeError) as explicit:
                managed._intervene()
        assert isinstance(explicit.value.__cause__, InterventionError)
        with pytest.raises(RuntimeError):
            model(**tokenizer("AB"))
        assert managed._first is first and managed._restored_interventions is pending
        assert not managed._intervening and environment.n == 0
        assert not environment.intervention_service.state_dict()["interventions"]
        if saved:
            environment.intervention_service.configure(dictionary=dictionary_for(model), tokenizer=tokenizer)
            with torch.no_grad():
                model(**tokenizer("AB"))
            assert managed._first is None and managed._restored_interventions is _managed.NONE
            assert environment.n == 1
    assert all(not module._forward_pre_hooks and not module._forward_hooks for module in model.modules())


def test_optimizer_run_applies_override_before_first_script_forward(environment, monkeypatch, tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    managed = _managed.ManagedRun("training", str(tmp_path), str(tmp_path / "link"),
                                  interventions=[{"point": "0", "op": "ablate"}])
    monkeypatch.setattr(_managed, "current", lambda: managed)
    inputs = torch.ones(1, 2)
    with caliper.watch(model, optimizer, activations=False, scene=False):
        with internal_forward():
            model(inputs)
        assert managed._first is not None and environment.n == 0
        output = model(inputs)
        assert torch.count_nonzero(output) == 0
        assert managed._first is None and environment.n == 0
        output.sum().backward()
        optimizer.step()
        assert environment.n == 1
