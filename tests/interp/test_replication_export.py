"""Private graph evidence preserves complete schedules without replacement."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from caliper.interp._replication_export import export_graph
from .test_hosted_trace_admission import case
from .test_service_trace_admission import run


def graph_fixture():
    result = {"trace_id": "a" * 64, "settings": {"physical_batches": [10, 3, 2],
              "logical_queues": [[1, 2, 3], [4, 5]]}, "completeness": {"targets": [{"target": 3}]},
              "nodes": [{"label": "\u03b1", "index": 0}], "edges": []}
    return SimpleNamespace(trace_id=result["trace_id"], _memory=None,
                           to_json=lambda: json.dumps(result, separators=(",", ":"))), result


def test_complete_private_graph_export_is_immutable(tmp_path):
    graph, expected = graph_fixture()
    descriptor = export_graph(graph, tmp_path)
    target = tmp_path / descriptor["file"]
    assert json.loads(target.read_text()) == expected
    assert hashlib.sha256(target.read_bytes()).hexdigest() == descriptor["sha256"]
    assert export_graph(graph, tmp_path) == descriptor
    assert list(tmp_path.iterdir()) == [target]
    before = target.read_bytes()
    expected["settings"]["physical_batches"].append(1)
    with pytest.raises(ValueError, match="changed for the same trace"):
        export_graph(graph, tmp_path)
    assert target.read_bytes() == before and list(tmp_path.iterdir()) == [target]


def test_graph_export_refuses_before_byte_copy_and_file_creation(tmp_path):
    graph, _ = graph_fixture()
    def refuse(stage, amount):
        assert stage == "private graph evidence" and amount > 4 * 1024 * 1024
        raise RuntimeError("fixture reserve refusal")
    graph._memory = SimpleNamespace(reserve=refuse)
    with pytest.raises(RuntimeError, match="reserve refusal"):
        export_graph(graph, tmp_path)
    assert not list(tmp_path.iterdir())


def test_graph_export_requires_safe_identity(tmp_path):
    graph, _ = graph_fixture()
    graph.trace_id = "../outside"
    with pytest.raises(ValueError, match="trace identity"):
        export_graph(graph, tmp_path)
    assert not list(tmp_path.iterdir())


def test_hosted_trace_exports_the_actual_graph(case, tmp_path):
    case.owner.replication_profile = "published_dallas_v1"
    case.owner.replication = SimpleNamespace(directory=tmp_path)
    graph = run(case.service)
    target = tmp_path / ("trace-" + graph.trace_id + ".json")
    assert json.loads(target.read_text()) == json.loads(graph.to_json())
    assert run(case.service) is graph
    assert list(tmp_path.iterdir()) == [target]
