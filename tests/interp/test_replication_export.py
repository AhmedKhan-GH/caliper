"""Private graph evidence preserves complete schedules without replacement."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from caliper.interp._graph import Graph
from caliper.interp import _replication_export as export
from .test_hosted_trace_admission import case
from .test_service_trace_admission import run


def graph_fixture():
    nodes = [{"kind": "feature", "id": f"feature:0:{i}", "label": "α😀", "layer": 0,
              "position": 0, "feature": i, "activation": 1., "influence": .5, "effect": .25}
             for i in range(300)]
    graph = Graph.build(nodes=nodes, edges=[{"source": 0, "target": 1, "weight": .5}],
                        tokens=[{"text": "A", "token_id": 65}], layers=1,
                        completeness={"targets": [{"target": 3}], "target_count": 1},
                        omitted_influence=0., error_share=0., feature_share=1., trace_id="a" * 64,
                        logits=[], settings={"physical_batches": [10, 3, 2],
                                             "logical_queues": [[1, 2, 3], [4, 5]]}, device="cpu")
    return graph


def test_complete_private_graph_export_is_immutable(tmp_path):
    graph = graph_fixture()
    expected = graph.to_json().encode("ascii")
    descriptor = export.export_graph(graph, tmp_path)
    target = tmp_path / descriptor["file"]
    assert target.read_bytes() == expected
    assert hashlib.sha256(expected).hexdigest() == descriptor["sha256"]
    assert len(json.loads(expected)["nodes"]) == 300
    assert export.export_graph(graph, tmp_path) == descriptor
    assert list(tmp_path.iterdir()) == [target]
    graph.settings["physical_batches"].append(1)
    with pytest.raises(ValueError, match="changed for the same trace"):
        export.export_graph(graph, tmp_path)
    assert target.read_bytes() == expected and list(tmp_path.iterdir()) == [target]


def test_graph_export_refuses_before_serialization_and_file_creation(tmp_path, monkeypatch):
    graph = graph_fixture()

    def refuse(stage, amount):
        assert stage == "private graph evidence" and amount > 4 * 1024 * 1024
        raise RuntimeError("fixture reserve refusal")

    graph._memory = SimpleNamespace(reserve=refuse)
    monkeypatch.setattr(graph, "to_json", lambda: pytest.fail("refused export reached serialization"))
    with pytest.raises(RuntimeError, match="reserve refusal"):
        export.export_graph(graph, tmp_path)
    assert not list(tmp_path.iterdir())


def test_workspace_covers_serialization_chunks_and_both_buffers():
    graph = graph_fixture()
    payload = graph.to_json().encode("ascii")
    assert export._workspace(graph, lambda stage: None) > 4 * 1024 * 1024 + 3 * len(payload)


@pytest.mark.parametrize("phase", ["write", "link", "complete"])
def test_failed_export_removes_temporary_and_new_target(tmp_path, monkeypatch, phase):
    graph = graph_fixture()

    def fail(*args):
        raise OSError("fixture export failure")

    if phase == "write":
        monkeypatch.setattr(export.os, "fsync", fail)
    elif phase == "link":
        monkeypatch.setattr(export.os, "link", fail)
    else:
        graph._memory = SimpleNamespace(reserve=lambda *args: None, check=fail)
    with pytest.raises(OSError, match="fixture export failure"):
        export.export_graph(graph, tmp_path)
    assert not list(tmp_path.iterdir())


def test_late_failure_preserves_identical_existing_evidence(tmp_path):
    graph = graph_fixture()
    descriptor = export.export_graph(graph, tmp_path)
    target = tmp_path / descriptor["file"]
    before = target.read_bytes()

    def fail(stage):
        if stage == "private graph complete":
            raise ValueError("fixture late failure")

    with pytest.raises(ValueError, match="fixture late failure"):
        export.export_graph(graph, tmp_path, check=fail)
    assert target.read_bytes() == before and list(tmp_path.iterdir()) == [target]


def test_graph_export_requires_safe_identity(tmp_path):
    graph = graph_fixture()
    graph.trace_id = "../outside"
    with pytest.raises(ValueError, match="trace identity"):
        export.export_graph(graph, tmp_path)
    assert not list(tmp_path.iterdir())


def test_hosted_trace_exports_the_actual_graph(case, tmp_path):
    case.owner.replication_profile = "published_dallas_v1"
    case.owner.replication = SimpleNamespace(directory=tmp_path)
    graph = run(case.service)
    target = tmp_path / ("trace-" + graph.trace_id + ".json")
    assert target.read_bytes() == graph.to_json().encode("ascii")
    assert case.owner.traced_prompt == "AB" and case.owner.traced_state == []
    assert run(case.service) is graph
    assert list(tmp_path.iterdir()) == [target]
