"""Published trace failures restore the complete prior CPU state."""
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _hosted, _publish, _replication_export, _trace, _validate
from caliper.interp._memory import CleanStop
from caliper.interp._service import Refused
from .test_hosted_trace_admission import case
from .test_service_memory import RequestGuard
from .test_service_trace_admission import enqueue, run, unchanged_publication


def prepare(case, tmp_path, prior):
    owner = case.owner
    owner.replication_profile = "published_dallas_v1"
    owner.replication = SimpleNamespace(directory=tmp_path)
    owner.service = case.service
    owner.traced_prompt = owner.traced_specifications = owner.traced_state = None
    owner.checkpoint_prompt = owner.prompt.value
    case.service.handlers["trace.validate"] = lambda: _hosted.HostedInference.validate(owner)
    owner.controls.step()
    if prior:
        run(case.service)
    _publish.publish("fixture.unrelated", torch.tensor([17]))
    executor = owner.interventions.executor
    executor._cache["fixture"] = torch.tensor([23])
    executor._cache_identity = ("fixture", ())
    case.index._summaries = OrderedDict([("fixture", "label")])
    return dict(graph=case.service.graph, tensors=dict(_publish._TENSORS),
                versions=dict(_publish._VERSIONS), version=_publish._VERSION,
                cache=None if case.service.tracer is None else case.service.tracer._cache.copy(),
                forward=None if case.service.tracer is None else case.service.tracer._forward_cache,
                prompt=None if case.service.tracer is None else case.service.tracer._prompt,
                guard=None if case.service.tracer is None else case.service.tracer.guard,
                index_memory=case.index.memory,
                donor=executor._cache.copy(), donor_identity=executor._cache_identity,
                labels=case.index._summaries.copy(),
                hosted=(owner.traced_prompt, owner.traced_specifications, owner.traced_state,
                        owner.checkpoint_prompt, owner.prompt.value, owner.prompt._version),
                files={file.name: file.read_bytes() for file in tmp_path.iterdir()})


def restored(case, tmp_path, before):
    owner, tracer = case.owner, case.service.tracer
    assert case.service.graph is before["graph"]
    unchanged_publication(before["tensors"])
    assert _publish._VERSIONS == before["versions"] and _publish._VERSION == before["version"]
    assert list(tracer._cache.items()) == list((before["cache"] or {}).items())
    assert tracer._forward_cache is before["forward"] and tracer._prompt == before["prompt"]
    executor = owner.interventions.executor
    assert executor._cache.keys() == before["donor"].keys()
    assert all(executor._cache[key] is value for key, value in before["donor"].items())
    assert executor._cache_identity == before["donor_identity"]
    assert case.index._summaries == before["labels"]
    assert (owner.traced_prompt, owner.traced_specifications, owner.traced_state,
            owner.checkpoint_prompt, owner.prompt.value, owner.prompt._version) == before["hosted"]
    assert {file.name: file.read_bytes() for file in tmp_path.iterdir()} == before["files"]
    assert not case.service._borrowed and case.service._request_stack is None
    assert case.index.memory is before["index_memory"]
    if before["guard"] is not None:
        assert tracer.guard is before["guard"]
        assert tracer._forward_cache[1].guard is before["guard"]


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("fault", ["write", "serialize", "complete", "hosted"])
def test_failed_published_trace_restores_all_state(case, tmp_path, monkeypatch, prior, fault):
    before = prepare(case, tmp_path, prior)
    labels = case.index.labels

    def changed_labels(features, **kwargs):
        case.index._summaries["candidate"] = "label"
        return labels(features, **kwargs)

    monkeypatch.setattr(case.index, "labels", changed_labels)
    if fault == "write":
        def fail(*args):
            raise OSError("fixture export write")
        monkeypatch.setattr(_replication_export.os, "fsync", fail)
    elif fault == "serialize":
        def fail(*args):
            raise ValueError("fixture export serialization")
        monkeypatch.setattr(_trace.Graph, "to_json", fail)
    elif fault == "complete":
        def fail(stage):
            if stage == "private graph complete":
                raise CleanStop("fixture export resource stop")
        case.owner.guard.at_stage = fail
    else:
        def fail(graph):
            raise ValueError("fixture hosted commit")
        case.owner._reconcile_selection = fail
    replies = enqueue(case.service, prompt="CD")
    if fault == "complete":
        with pytest.raises(CleanStop, match="fixture export resource stop"):
            case.service.step()
        assert replies[0][2]["code"] == "interpret.stopped"
    else:
        assert case.service.step() == 1
    assert replies[0][0] == "error"
    assert case.service.status["state"] == "error"
    restored(case, tmp_path, before)
    if not prior:
        for method in ("trace.read", "trace.validate"):
            with pytest.raises(Refused) as error:
                case.service._execute(method, {}, 1)
            assert error.value.code == "interpret.missing_trace"
    else:
        assert case.service._execute("trace.read", {}, 1)["trace_id"] == before["graph"].trace_id
        seen = []

        def validate(model, tokenizer, dictionary, replay, graph, *, prompt, guard):
            seen.append((graph, prompt, replay.specifications()))
            return {"fixture": True}

        monkeypatch.setattr(_validate, "validate_trace", validate)
        assert case.service._execute("trace.validate", {}, 1) == {"fixture": True}
        assert seen == [(before["graph"], "AB", before["hosted"][2])]


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("source", ["owner", "service"])
def test_lower_export_cap_refuses_before_json(case, tmp_path, monkeypatch, prior, source):
    before = prepare(case, tmp_path, prior)
    memory = case.owner.guard
    if source == "service":
        # A cached graph predates this lower service guard.
        memory = case.service.memory = RequestGuard(40_000_000_000)

    def lower(stage):
        if stage == "private graph evidence":
            memory._guard.limit = 4 * 1024 * 1024

    memory.at_stage = lower
    monkeypatch.setattr(_trace.Graph, "to_json", lambda *args: pytest.fail("lower cap reached JSON"))
    replies = enqueue(case.service, prompt="AB" if prior else "CD")
    with pytest.raises(CleanStop, match="memory limit"):
        case.service.step()
    assert replies[0][2]["code"] == "interpret.stopped"
    restored(case, tmp_path, before)


def test_export_keeps_original_request_deadline(case, tmp_path, monkeypatch):
    before = prepare(case, tmp_path, True)
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(_trace, "time", SimpleNamespace(monotonic=lambda: clock.now))
    fsync, forward = _replication_export.os.fsync, case.model.forward

    def elapsed_forward(*args, **kwargs):
        result = forward(*args, **kwargs)
        clock.now += 12
        return result

    def slow_write(fd):
        fsync(fd)
        clock.now += 9

    monkeypatch.setattr(case.model, "forward", elapsed_forward)
    monkeypatch.setattr(_replication_export.os, "fsync", slow_write)
    replies = enqueue(case.service, prompt="CD")
    assert case.service.step() == 1
    assert replies[0][0] == "error" and replies[0][2]["code"] == "interpret.trace_timeout"
    restored(case, tmp_path, before)


def test_export_failure_preserves_prompt_edit_during_trace(case, tmp_path, monkeypatch):
    prepare(case, tmp_path, True)

    def edit(text):
        if case.service._request_stack is not None:
            case.owner.prompt.set("EF")

    case.tokenizer.callback = edit

    def fail(*args):
        raise OSError("fixture write")

    monkeypatch.setattr(_replication_export.os, "fsync", fail)
    replies = enqueue(case.service, prompt="CD")
    assert case.service.step() == 1 and replies[0][0] == "error"
    assert case.owner.prompt.value == "EF" and case.owner.traced_prompt == "AB"


def test_cached_trace_conflict_preserves_different_evidence(case, tmp_path):
    before = prepare(case, tmp_path, True)
    target = tmp_path / ("trace-" + before["graph"].trace_id + ".json")
    target.write_bytes(b'{"fixture":"different evidence"}')
    before["files"][target.name] = target.read_bytes()
    replies = enqueue(case.service)
    assert case.service.step() == 1 and replies[0][0] == "error"
    assert "changed for the same trace" in replies[0][2]["message"]
    restored(case, tmp_path, before)


def test_late_failure_restores_pending_selection_count(case, tmp_path, monkeypatch):
    before = prepare(case, tmp_path, True)
    owner = case.owner
    owner.selected = owner.controls.param("selected_node", -1, min=-1, max=511)
    fsync = _replication_export.os.fsync

    def pending_selection(fd):
        fsync(fd)
        owner.selected._pending = 2
        owner.controls._npending += 1

    def fail(graph):
        owner.selected.set(0)
        raise OSError("fixture selection publication")

    monkeypatch.setattr(_replication_export.os, "fsync", pending_selection)
    owner._reconcile_selection = fail
    replies = enqueue(case.service, prompt="CD")
    assert case.service.step() == 1 and replies[0][0] == "error"
    restored(case, tmp_path, before)
    assert owner.selected.value == -1 and owner.selected.pending == 2
    assert owner.selected._version == 0 and owner.controls._npending == 1


def test_ordinary_profile_does_not_enter_export_transaction(case, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("ordinary trace entered private export")

    monkeypatch.setattr(_replication_export, "export_graph", fail)
    monkeypatch.setattr(_replication_export, "trace_transaction", fail)
    graph = run(case.service)
    assert case.owner.traced_prompt == "AB" and case.service.graph is graph
