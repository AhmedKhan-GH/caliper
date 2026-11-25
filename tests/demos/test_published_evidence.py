"""Tamper checks for native case evidence and complete collection coverage."""
import json
import pickle
import zipfile

import pytest
import torch

from test_published_dallas import callback_boundary, evidence_factory, published, reviewed_collection, write_json
from _mcp import Failure
from _replication_review import verify_review, write_collection


def sync_metadata(root, record):
    write_json(root / record["metadata_file"],
               {key: value for key, value in record.items() if key not in ("metadata_file", "action_seconds")})


def replace_native(root, descriptor, tensors):
    target = root / descriptor["file"]
    with target.open("wb") as stream:
        torch.save(tensors, stream)
    descriptor.update(published.descriptor(root, target))
    if "tensor" in descriptor:
        name, = tensors
        descriptor.update(tensor=name, dtype=str(tensors[name].dtype), shape=list(tensors[name].shape))
    else:
        descriptor["tensors"] = {name: {"dtype": str(value.dtype), "shape": list(value.shape)}
                                 for name, value in tensors.items()}


@pytest.mark.parametrize("fault", ["omitted", "tampered", "dtype", "shape", "values", "declaration"])
def test_native_arrays_bind_actual_bytes(evidence_factory, fault):
    root, _, make = evidence_factory
    status = make(14)
    record = status["inference"]["replication"]["result"]
    native = record["native_evidence"]
    if fault == "omitted":
        record.pop("native_evidence")
    elif fault == "tampered":
        (root / native["file"]).write_bytes(b"changed")
    elif fault == "declaration":
        native["tensors"]["logits"]["dtype"] = "torch.float32"
    else:
        values = torch.load(root / native["file"], weights_only=True)
        if fault == "dtype":
            values["logits"] = values["logits"].float()
        elif fault == "shape":
            values["logits"] = values["logits"].reshape(-1)
        else:
            values["logits"][0, 0] = 2
        replace_native(root, native, values)
    sync_metadata(root, record)
    with pytest.raises(Failure):
        published.validate_case(root, status, 14, previous_sequence=0)


@pytest.mark.parametrize("fault,message", [
    ("omitted", "observation identity differs"),
    ("node_layer", "observation identity differs"),
    ("node_feature", "observation identity differs"),
    ("node_position", "observation identity differs"),
    ("native_omitted", "Native observation declaration differs"),
    ("tampered", "Private evidence hash differs"),
    ("hash", "Private evidence hash differs"),
    ("absolute", "Private evidence requires a relative path"),
    ("tensor", "Native observation declaration differs"),
    ("nested", "Native observation declaration differs"),
    ("declared_dtype", "Native tensor declarations differ"),
    ("declared_shape", "Native tensor declarations differ"),
    ("nonzero", "activation is not zero"),
    ("nan", "Native tensor contains nonfinite values"),
    ("dtype", "Native archive member exceeds its evidence limit"),
    ("dtype16", "Unsupported native tensor metadata global"),
    ("shape", "Native tensor shape or storage differs"),
    ("actual_name", "Native tensor coverage differs"),
])
def test_cell29_requires_actual_zero_observation(evidence_factory, fault, message):
    root, _, make = evidence_factory
    status = make(29)
    published.validate_case(root, status, 29, previous_sequence=0)
    record = status["inference"]["replication"]["result"]
    observation = record["observations"]["Say Victoria"]
    native = observation["native_evidence"]
    if fault == "omitted":
        record.pop("observations")
    elif fault.startswith("node_"):
        observation["node"][fault.removeprefix("node_")] += 1
    elif fault == "native_omitted":
        observation.pop("native_evidence")
    elif fault == "tampered":
        (root / native["file"]).write_bytes(b"changed")
    elif fault == "hash":
        native["sha256"] = "0" * 64
    elif fault == "absolute":
        native["file"] = str(root / native["file"])
    elif fault == "tensor":
        native["tensor"] = "other"
    elif fault == "nested":
        native["tensors"] = {native.pop("tensor"): {"dtype": native.pop("dtype"), "shape": native.pop("shape")}}
    elif fault == "declared_dtype":
        native["dtype"] = "torch.float32"
    elif fault == "declared_shape":
        native["shape"] = [1, 1]
    else:
        activation = torch.tensor([1. if fault == "nonzero" else float("nan") if fault == "nan" else 0.],
                                  dtype=torch.float32 if fault == "dtype" else
                                  torch.float16 if fault == "dtype16" else torch.bfloat16)
        if fault == "shape":
            activation = activation.reshape(1, 1)
        replace_native(root, native, {"other" if fault == "actual_name" else "activation": activation})
        # Keep valid declarations to exercise actual archive metadata validation.
        native.update(tensor="activation", dtype="torch.bfloat16", shape=[1])
    sync_metadata(root, record)
    with pytest.raises(Failure, match=message):
        published.validate_case(root, status, 29, previous_sequence=0)


@pytest.mark.parametrize("fault,message", [
    ("pickle_global", "Unsupported native tensor metadata global"),
    ("large_member", "Native archive member exceeds its evidence limit"),
    ("large_file", "Native archive exceeds its evidence limit"),
    ("duplicate", "Duplicate native archive entries"),
])
def test_observation_reader_refuses_unqualified_archives(evidence_factory, fault, message):
    root, _, make = evidence_factory
    status = make(29)
    published.validate_case(root, status, 29, previous_sequence=0)
    record = status["inference"]["replication"]["result"]
    native = record["observations"]["Say Victoria"]["native_evidence"]
    target = root / native["file"]
    with zipfile.ZipFile(target) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    metadata = next(name for name in entries if name.endswith("/data.pkl"))
    if fault == "pickle_global":
        entries[metadata] = pickle.dumps(len, protocol=2)
    elif fault == "large_member":
        entries[metadata] = b"0" * (16384 + 1)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
        if fault == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate"):
                archive.writestr(metadata, entries[metadata])
    if fault == "large_file":
        with target.open("ab") as stream:
            stream.truncate(32 * 1024 * 1024 + 1)
    native.update(published.descriptor(root, target))
    sync_metadata(root, record)
    with pytest.raises(Failure, match=message):
        published.validate_case(root, status, 29, previous_sequence=0)


def rebind_collection(root, review, collection):
    write_json(root / "driver/collection.json", collection)
    review["collection_sha256"] = published.file_hash(root / "driver/collection.json")
    write_json(root / "review.json", review)


@pytest.mark.parametrize("kind", ["result", "metadata", "preparation_file", "evidence", "native_evidence", "observation", "graph"])
def test_complete_inventory_cannot_omit_bound_files(reviewed_collection, kind):
    root, review, result = reviewed_collection
    collection = json.loads((root / "driver/collection.json").read_text())
    evidence = result["published_actions"][-1]["evidence"]
    target = {"result": collection["result"], "metadata": evidence["metadata"],
              "preparation_file": evidence["preparation_file"], "evidence": evidence["result"]["evidence"],
              "native_evidence": evidence["result"]["native_evidence"],
              "observation": evidence["result"]["observations"]["Say Victoria"]["native_evidence"],
              "graph": {"file": f"trace-{result['trace']['trace_id']}.json"}}[kind]
    collection["files"] = [item for item in collection["files"] if item["file"] != target["file"]]
    rebind_collection(root, review, collection)
    with pytest.raises(Failure):
        verify_review(root, "driver/collection.json", "review.json")


@pytest.mark.parametrize("fault", ["trace_id", "counts", "settings", "tokens", "nodes", "edges", "schedule", "diagnostics",
                                  "logical_schedule", "duplicate_index", "duplicate_edge", "endpoint", "boolean_endpoint",
                                  "next_node", "next_edge"])
def test_full_graph_must_match_collection_identity(reviewed_collection, fault):
    root, review, result = reviewed_collection
    target = root / f"trace-{result['trace']['trace_id']}.json"
    graph = json.loads(target.read_text())
    if fault == "trace_id":
        graph["trace_id"] = "f" * 64
    elif fault == "counts":
        graph["counts"]["edges"] += 1
    elif fault == "settings":
        graph["settings"]["batch_size"] = 64
    elif fault == "tokens":
        graph["tokens"][0]["token_id"] += 1
    elif fault == "nodes":
        graph["nodes"].pop()
    elif fault == "edges":
        graph["edges"].pop()
    elif fault == "schedule":
        graph["settings"].pop("physical_batches")
    elif fault == "logical_schedule":
        graph["settings"].pop("logical_queues")
    elif fault == "duplicate_index":
        graph["nodes"][1]["index"] = graph["nodes"][0]["index"]
    elif fault == "duplicate_edge":
        graph["edges"][1] = dict(graph["edges"][0])
    elif fault == "endpoint":
        graph["edges"][0]["target"] = len(graph["nodes"])
    elif fault == "boolean_endpoint":
        graph["edges"][0]["source"] = False
    elif fault == "next_node":
        graph["next_node_offset"] = 1
    elif fault == "next_edge":
        graph["next_edge_offset"] = 1
    else:
        graph["completeness"]["targets"].clear()
    write_json(target, graph)
    collection = json.loads((root / "driver/collection.json").read_text())
    for item in collection["files"]:
        if item["file"] == target.name:
            item.update(published.descriptor(root, target))
    rebind_collection(root, review, collection)
    with pytest.raises(Failure):
        verify_review(root, "driver/collection.json", "review.json")


@pytest.mark.parametrize("field,check", [("peak_process_bytes", "process_memory"), ("bounded_peak_bytes", "bounded_memory")])
def test_review_memory_cannot_understate_driver(reviewed_collection, field, check):
    root, review, result = reviewed_collection
    measured = next(gate["measured"] for gate in result["checks"] if gate["name"] == check)
    review["memory"][field] = measured - 1
    write_json(root / "review.json", review)
    with pytest.raises(Failure):
        verify_review(root, "driver/collection.json", "review.json")


@pytest.mark.parametrize("fault", ["graph", "graph_identity", "native", "observation"])
def test_writer_refuses_incomplete_complete_collection(reviewed_collection, fault):
    root, _, result = reviewed_collection
    if fault == "graph":
        (root / f"trace-{result['trace']['trace_id']}.json").unlink()
    elif fault == "graph_identity":
        write_json(root / f"trace-{result['trace']['trace_id']}.json", {"trace_id": "f" * 64})
    elif fault == "native":
        result["published_actions"][0]["evidence"]["result"].pop("native_evidence")
    else:
        result["published_actions"][-1]["evidence"]["result"].pop("observations")
    with pytest.raises((Failure, OSError)):
        write_collection(root / "driver", root, result)


def test_duplicate_memory_measurements_cannot_hide_a_peak(reviewed_collection):
    root, review, result = reviewed_collection
    gate = next(item for item in result["checks"] if item["name"] == "process_memory")
    result["checks"].append(dict(gate, measured=1))
    review["memory"]["peak_process_bytes"] = 1
    write_json(root / "driver/result.json", result)
    collection = write_collection(root / "driver", root, result)
    rebind_collection(root, review, collection)
    with pytest.raises(Failure):
        verify_review(root, "driver/collection.json", "review.json")


def test_review_memory_may_equal_driver_peaks(reviewed_collection):
    root, review, result = reviewed_collection
    for field, name in (("peak_process_bytes", "process_memory"), ("bounded_peak_bytes", "bounded_memory")):
        review["memory"][field] = next(gate["measured"] for gate in result["checks"] if gate["name"] == name)
    write_json(root / "review.json", review)
    assert verify_review(root, "driver/collection.json", "review.json")["verified"]
