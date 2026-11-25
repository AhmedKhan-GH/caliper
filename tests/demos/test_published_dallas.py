"""Published source and evidence checks with tiny private arrays and MCP boundaries."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
import pytest

from test_circuit_trace import driver, ROOT
from _checks import check, finish
from _mcp import Failure
import _published as published
import _published_ui as ui
from _replication_review import verify_review, write_collection


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def test_exact_published_manifest():
    source = published.manifest()
    assert hashlib.sha256((ROOT / "demos/circuit_trace/published_dallas_v1.json").read_bytes()).hexdigest() == published.MANIFEST_SHA256
    assert source["protocol"] == published.PROFILE
    assert published.PROMPT == "Fact: the capital of the state containing Dallas is"
    assert list(published.CASES) == [10, 14, 16, 18, 20, 24, 26, 29]
    assert [published.action_name(cell) for cell in published.CASES] == [
        "published_dallas_v1.cell10", "published_dallas_v1.cell14", "published_dallas_v1.cell16", "published_dallas_v1.cell18",
        "published_dallas_v1.cell20", "published_dallas_v1.cell24", "published_dallas_v1.cell26", "published_dallas_v1.cell29"]
    assert published.TRACE_REQUEST == {"profile": "circuit_tracer_82996f14", "max_targets": 5000,
        "max_nodes": 8192, "max_edges": 262144, "max_n_logits": 10, "desired_logit_prob": .95,
        "node_threshold": .7, "edge_threshold": .9, "batch_size": 512, "update_interval": 4}
    assert published.CASES[29]["published_output_labels"][0] == " the 8%"
    assert published.SOURCE["groups"]["British Columbia"] == [{"layer": 18, "feature": 1025, "position": 10}]
    assert len([node for op in published.CASES[26]["operations"] for node in source["groups"][op["group"]]]) == 13


@pytest.fixture
def evidence_factory(tmp_path):
    prepared = {"protocol": published.PROFILE, "source_revision": published.SOURCE["source_revision"], "producer": "caliper",
        "model_content_hash": "a" * 64, "model_settings": {"arithmetic": "source"}, "dictionary_identity": "b" * 64,
        "dtype": "torch.bfloat16", "softmax_dtype": "torch.bfloat16", "freeze_attention": True, "freeze_norms": False,
        "freeze_mlps": False, "decoder_accumulation": "sorted_position_feature_index_add",
        "prompts": {key: {"text": prompt, "sha256": published.text_hash(prompt), "token_ids": list(range(11))}
                    for key, prompt in published.SOURCE["prompts"].items()}}
    prepared["id"] = published.digest(prepared)
    prepared["seconds"] = .1
    write_json(tmp_path / "preparation.json", prepared)

    def make(cell=14, sequence=1):
        probabilities = np.array([[.5, .25, .125, .0625, .0625]] * 2, dtype=np.float32)
        if cell != 10:
            probabilities[1] = probabilities[1, [1, 0, 2, 3, 4]]
        archive = tmp_path / f"cell{cell}-{sequence}.npz"
        np.savez(archive, logits=probabilities, probabilities=probabilities)
        top = []
        expected = published.CASES[cell]["published_output_labels"][0].rsplit(" ", 1)[0]
        for row in (0, 1):
            ids = sorted(range(5), key=lambda i: (-probabilities[row, i], i))
            top.append([{"id": i, "text": (" Austin" if row == 0 else expected) if j == 0 else f" token{i}",
                         "probability": float(probabilities[row, i])} for j, i in enumerate(ids)])
        members = []
        for operation in published.CASES[cell]["operations"]:
            for node in published.SOURCE["groups"][operation["group"]]:
                members.append({**node, **operation, "activation": .5, "value": .5 * operation["factor"],
                    "canonical_id": f"feature:{node['layer']}:{node['feature']}:{node['position']}"})
        record = {"case_id": published.action_name(cell), "cell": cell, "label": published.CASES[cell]["label"],
            "producer": "caliper", "preparation_id": prepared["id"], "sequence": sequence, "group_values": members,
            "argmax_token_ids": probabilities.argmax(axis=1).tolist(), "top_token_ids": [row[0]["id"] for row in top],
            "top_five": top, "baseline_top_text": " Austin", "baseline_top_probability": .5,
            "logits_dtype": "torch.bfloat16", "softmax_dtype": "torch.bfloat16", "compute_export_seconds": .01,
            "evidence": {**published.descriptor(tmp_path, archive), "shape": [2, 5], "rows": ["baseline", "case"],
                         "storage_dtype": "float32", "native_values_preserved": True}}
        metadata = tmp_path / f"cell{cell}-{sequence}.json"
        write_json(metadata, record)
        record.update(metadata_file=metadata.name, action_seconds=.02)
        comparison = {"prompt_hash": published.text_hash(published.PROMPT), "interventions_hash": "c" * 64,
                      "case_id": record["case_id"], "argmax_token_ids": record["argmax_token_ids"],
                      "baseline_top_text": " Austin", "baseline_top_probability": .5, "completed_step": sequence}
        return {"state": "done", "inference": {
            "state": "ready", "prompt_hash": comparison["prompt_hash"], "interventions_hash": comparison["interventions_hash"],
            "comparison": comparison, "replication": {"protocol": published.PROFILE, "prepared": True,
                "preparation_id": prepared["id"], "preparation_seconds": prepared["seconds"], "preparation_file": "preparation.json",
                "active_cell": cell, "groups": deepcopy(published.SOURCE["groups"]),
                "cases": [{"cell": c, "action": published.action_name(c), "label": v["label"], "operations": v["operations"]}
                          for c, v in published.CASES.items()],
                "last_attempt": {"case_id": record["case_id"], "sequence": sequence, "status": "ok"}, "result": record}}}
    return tmp_path, prepared, make


@pytest.mark.parametrize("cell", list(published.CASES))
def test_private_case_identity_and_manifest_members(evidence_factory, cell):
    root, prepared, make = evidence_factory
    status = make(cell)
    record = published.validate_case(root, status, cell, previous_sequence=0, preparation_id=prepared["id"])
    assert record["result"]["cell"] == cell
    assert "probabilities" not in json.dumps(status)


@pytest.mark.parametrize("fault", ["sequence", "attempt", "active", "preparation", "prompt", "publication", "publication_hash",
                                  "catalog", "groups", "member", "metadata", "archive", "escape", "missing", "dtype", "duration"])
def test_failed_stale_or_unpublished_case_refuses(evidence_factory, fault):
    root, prepared, make = evidence_factory
    status = make(26)
    inference = status["inference"]
    replication = inference["replication"]
    result = replication["result"]
    if fault == "sequence": result["sequence"] = 0
    if fault == "attempt": replication["last_attempt"]["status"] = "error"
    if fault == "active": replication["active_cell"] = 10
    if fault == "preparation": result["preparation_id"] = "d" * 64
    if fault == "prompt": inference["prompt_hash"] = "d" * 64
    if fault == "publication": inference["comparison"] = None
    if fault == "publication_hash": inference["comparison"]["interventions_hash"] = "d" * 64
    if fault == "catalog": replication["cases"].pop()
    if fault == "groups": replication["groups"]["China"].pop()
    if fault == "member": result["group_values"].pop()
    if fault == "metadata": write_json(root / result["metadata_file"], {})
    if fault == "archive": (root / result["evidence"]["file"]).write_bytes(b"stale")
    if fault == "escape": replication["preparation_file"] = "../preparation.json"
    if fault == "missing": (root / result["metadata_file"]).unlink()
    if fault == "dtype": result["softmax_dtype"] = "torch.float32"
    if fault == "duration": result["action_seconds"] = None
    with pytest.raises(Failure):
        published.validate_case(root, status, 26, previous_sequence=0, preparation_id=prepared["id"])


@pytest.mark.parametrize("path_kind", ["absolute", "symlink"])
def test_private_paths_cannot_leave_root(tmp_path, path_kind):
    inside = tmp_path / "private"
    inside.mkdir()
    outside = tmp_path / "record.json"
    outside.write_text("{}")
    link = inside / "alias.json"
    link.symlink_to(outside)
    with pytest.raises(Failure):
        published.private_path(inside, str(outside) if path_kind == "absolute" else link.name)


class GraphMCP:
    def __init__(self):
        self.calls = []
        self.trace = {"state": "done", "status": "done", "trace_id": "e" * 64, "seconds": .1, "step": 4,
            "settings": dict(published.TRACE_REQUEST), "counts": {"nodes": 300, "edges": 300, "tokens": 11, "layers": 26},
            "completeness": {"max_absolute_residual": .001}}
        self.nodes = [{"index": i, "kind": "feature", "layer": 15, "feature": 4494 + i, "position": 4,
                       "effect": i, "label": str(i)} for i in range(300)]
        self.edges = [{"source": i, "target": (i + 1) % 300, "weight": .1} for i in range(300)]
        self.edit = lambda page: None

    def complete(self, method, **kwargs):
        assert method == "trace.read"
        self.calls.append(kwargs)
        page = {**deepcopy(self.trace), "tokens": [], "nodes": [], "edges": []}
        for kind in ("node", "edge"):
            offset, limit = kwargs.get(kind + "_offset", 0), kwargs[kind + "_limit"]
            values = getattr(self, kind + "s")
            page[kind + "s"] = deepcopy(values[offset:offset + limit])
            end = offset + len(page[kind + "s"])
            page["next_" + kind + "_offset"] = end if limit and end < len(values) else None
        self.edit(page)
        return page


def test_all_graph_pages_and_published_example_selection():
    transport = GraphMCP()
    graph, pages = published.read_graph(SimpleNamespace(events=transport), transport.trace)
    assert len(graph["nodes"]) == len(graph["edges"]) == 300 and len(pages) == 4
    assert all(max(call["node_limit"], call["edge_limit"]) == 256 for call in transport.calls)
    assert published.published_selection(graph)["index"] == 0  # Largest effect is row 299.


@pytest.mark.parametrize("fault", ["id", "counts", "settings", "short", "offset", "status", "duplicate", "endpoint"])
def test_incomplete_graph_and_identity_mismatches_refuse(fault):
    transport = GraphMCP()
    def edit(page):
        if fault == "id": page["trace_id"] = "f" * 64
        if fault == "counts": page["counts"]["nodes"] = 299
        if fault == "settings": page["settings"]["batch_size"] = 64
        if fault == "short" and page["nodes"]: page["nodes"].pop()
        if fault == "offset": page["next_node_offset"] = 0
        if fault == "status": page["status"] = "incomplete"
        if fault == "duplicate" and page["nodes"]: page["nodes"][0]["index"] = 1
        if fault == "endpoint" and page["edges"]: page["edges"][0]["target"] = 300
    transport.edit = edit
    with pytest.raises(Failure):
        published.read_graph(SimpleNamespace(events=transport), transport.trace)


def test_published_controls_use_registered_dsl_and_keep_example_height():
    calls = []
    session = SimpleNamespace(replication_profile=published.PROFILE,
                              mcp=SimpleNamespace(body=lambda method, **args: calls.append((method, args))))
    driver.instrument(session)
    schemas = {entry["name"]: entry["args"] for entry in json.loads(
        (ROOT / "src/builder/commands/command_table.json").read_text())["commands"]}
    for method, args in calls:
        jsonschema.validate(args, schemas[method])
    controls = {args["id"]: args for method, args in calls if method == "control.create"}
    assert [controls[f"cell{cell}"]["bind"] for cell in published.CASES] == [
        "action:" + published.action_name(cell) for cell in published.CASES]
    assert controls["prompt"]["options"]["format"] == published.PROMPT
    assert [item["label"] for item in controls["published_group"]["options"]["choices"]] == list(published.SOURCE["groups"])
    assert all(f"L{n['layer']} F{n['feature']} P{n['position']}" in ui.group_text(i)
               for i, nodes in enumerate(published.SOURCE["groups"].values()) for n in nodes)
    layout = next(args["layout"] for method, args in calls if method == "layout.set")
    nodes = {node["id"]: node for node in layout["nodes"]}
    assert nodes["body"]["ratio"] == .70 and nodes["body"]["children"] == ["plots", "examples"]
    assert nodes["header"]["children"] == ["prompt", "clamp_status"]
    def depth(name):
        return 1 + max((depth(child) for child in nodes[name]["children"]), default=0) if name in nodes else 1
    assert depth(layout["root"]) <= 8
    assert not {"clamp", "generate", "retrace", "apply_clamp"} & controls.keys()


def test_actual_callback_boundary_and_output_publication(evidence_factory):
    root, prepared, make = evidence_factory
    before, after = make(10, 1), make(26, 2)
    state = {"value": before}
    session = driver.Session.__new__(driver.Session)
    session.out, session.private_output_directory, session.run_id = root, root, "test-run"
    pending = []
    def body(method, **args):
        if method == "run.action":
            assert args == {"run": "test-run", "name": published.action_name(26)}
            state["value"] = after
            pending.append({"kind": "run", "change": "action", "name": args["name"], "status": "ok"})
        elif method == "run.state":
            return {"attached": True}
        elif method == "state.set":
            assert "Before:" in args["value"] and "After:" in args["value"]
        else:
            raise AssertionError(method)
    session.mcp = SimpleNamespace(body=body)
    session.events = SimpleNamespace(pending=pending, wait=lambda seconds: None,
                                    until=lambda name, callback: callback(), complete=lambda method: deepcopy(state["value"]))
    entry = driver.published_action(session, 26, prepared["id"])
    assert entry["completed"] and entry["prior_sequence"] == 1
    assert entry["full_action_seconds"] >= entry["callback_seconds"] >= 0
    assert json.loads((root / "action-2.json").read_text())["completed"]


def test_historical_causal_pass_never_approves_published_acceptance():
    result = {"mode": "gemma", "completed": True, "pixel_evidence_collected": True,
              "checks": [check("validation", True, {"correct_signs": 10, "spearman": 1}, True)]}
    assert finish(result) == 1
    assert not result["gemma_acceptance"] and not result["accepted_replication"]
    assert {"published_profile", "external_replication_review"} <= set(result["failed_checks"])


def test_collection_completion_is_not_acceptance(tmp_path):
    out = tmp_path / "driver"
    out.mkdir()
    result = {"mode": "gemma", "replication_profile": published.PROFILE, "evidence_collection_complete": True,
              "completed": False, "checks": [], "published_actions": []}
    write_json(out / "result.json", result)
    collection = write_collection(out, tmp_path, result)
    assert collection["evidence_collection_complete"] and not collection["accepted_replication"]
    assert finish(result) == 1 and not result["accepted_replication"]
    write_json(tmp_path / "review.json", {"schema": "caliper.published-dallas.review.v1", "protocol": published.PROFILE,
               "source_manifest_sha256": published.MANIFEST_SHA256, "collection_sha256": "0" * 64})
    with pytest.raises(Failure, match="another collection"):
        verify_review(tmp_path, "driver/collection.json", "review.json")
