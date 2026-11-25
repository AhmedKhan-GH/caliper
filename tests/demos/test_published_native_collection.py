"""Private collection inventories include original runtime tensor evidence."""
from copy import deepcopy
import json

import pytest

from test_circuit_trace import driver
from _mcp import Failure
from _published import descriptor
from _replication_review import write_collection


@pytest.fixture
def collection(tmp_path):
    out = tmp_path / "driver"
    out.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name in ("metadata.json", "preparation.json", "arrays.npz", "native.pt", "observation.pt"):
        (runtime / name).write_bytes(name.encode())
    trace_id = "a" * 64
    graph = runtime / f"trace-{trace_id}.json"
    graph.write_text('{"schedule": []}')
    record = {"evidence": descriptor(tmp_path, runtime / "arrays.npz"),
              "native_evidence": descriptor(tmp_path, runtime / "native.pt"),
              "observations": {"Say Victoria": {
                  "native_evidence": descriptor(tmp_path, runtime / "observation.pt")}}}
    evidence = {"result": record, "metadata": descriptor(tmp_path, runtime / "metadata.json"),
                "preparation_file": descriptor(tmp_path, runtime / "preparation.json")}
    result = {"published_actions": [{"evidence": evidence}], "trace": {"trace_id": trace_id},
              "evidence_collection_complete": True}
    driver.save(out / "result.json", result)
    return tmp_path, out, runtime, result


def test_native_observation_and_full_graph_enter_inventory(collection):
    root, out, runtime, result = collection
    manifest = write_collection(out, root, result)
    expected = {str(path.relative_to(root)) for path in runtime.iterdir()}
    assert {item["file"] for item in manifest["files"]} == expected | {"driver/result.json"}
    assert manifest == json.loads((out / "collection.json").read_text())
    assert manifest["accepted_replication"] is False


@pytest.mark.parametrize("kind", ["native", "observation"])
def test_changed_native_file_cannot_enter_inventory(collection, kind):
    root, out, runtime, result = collection
    (runtime / (kind + ".pt")).write_bytes(b"changed")
    with pytest.raises(Failure, match="hash differs"):
        write_collection(out, root, result)
    assert not (out / "collection.json").exists()


def test_outside_native_file_is_refused(collection):
    root, out, _, result = collection
    result = deepcopy(result)
    result["published_actions"][0]["evidence"]["result"]["native_evidence"]["file"] = str(out / "result.json")
    with pytest.raises(Failure, match="relative path"):
        write_collection(out, root, result)


def test_ordinary_legacy_collection_retains_existing_contract(collection):
    root, out, _, result = collection
    record = result["published_actions"][0]["evidence"]["result"]
    del record["native_evidence"], record["observations"]
    del result["trace"]
    manifest = write_collection(out, root, result)
    assert len(manifest["files"]) == 4
