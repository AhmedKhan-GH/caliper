"""Published source and evidence checks with tiny private arrays and MCP boundaries."""
from copy import deepcopy
import hashlib
import json
import os
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
        "model_content_hash": "sha256:" + "a" * 64, "model_settings": {"arithmetic": "source"},
        "dictionary_identity": "sha256:" + "b" * 64,
        "dtype": "torch.bfloat16", "softmax_dtype": "torch.bfloat16", "freeze_attention": True, "freeze_norms": False,
        "freeze_mlps": False, "decoder_accumulation": "sorted_position_feature_index_add",
        "prompts": {key: {"text": prompt, "sha256": published.text_hash(prompt), "token_ids": list(range(11))}
                    for key, prompt in published.SOURCE["prompts"].items()}}
    prepared["id"] = published.digest(prepared)
    prepared["seconds"] = .1
    write_json(tmp_path / "preparation.json", prepared)

    def make(cell=14, sequence=1):
        import torch
        from caliper.interp._replication_native import write_native
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
        record["native_evidence"] = write_native(
            archive.with_suffix(".pt"),
            {name: torch.from_numpy(probabilities).to(torch.bfloat16) for name in ("logits", "probabilities")},
            root=tmp_path, created=[], reserve=lambda amount: None)
        if cell == 29:
            record["observations"] = {"Say Victoria": {
                "node": deepcopy(published.SOURCE["groups"]["Say Victoria"][0]),
                "native_evidence": write_native(
                    archive.with_name(archive.stem + "-observation.pt"), {"activation": torch.zeros(1, dtype=torch.bfloat16)},
                    root=tmp_path, created=[], reserve=lambda amount: None)}}
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


def test_preparation_accepts_producer_content_identity(evidence_factory):
    import torch
    from caliper.interp._intervene import _content_identity
    root, prepared, _ = evidence_factory
    prepared["model_content_hash"] = _content_identity({"weight": torch.tensor([1., 2.])}, None)
    prepared["id"] = published.digest({k: v for k, v in prepared.items() if k not in ("id", "seconds")})
    write_json(root / "preparation.json", prepared)
    assert published.preparation(root, {"preparation_file": "preparation.json",
                                       "preparation_id": prepared["id"]}) == prepared


@pytest.mark.parametrize("key", ["model_content_hash", "dictionary_identity"])
@pytest.mark.parametrize("invalid", [None, "a" * 64, "sha256:" + "g" * 64, "sha256:" + "a" * 63])
def test_preparation_refuses_invalid_content_identity(evidence_factory, key, invalid):
    root, prepared, _ = evidence_factory
    prepared[key] = invalid
    prepared["id"] = published.digest({k: v for k, v in prepared.items() if k not in ("id", "seconds")})
    write_json(root / "preparation.json", prepared)
    with pytest.raises(Failure, match="current resource identity"):
        published.preparation(root, {"preparation_file": "preparation.json", "preparation_id": prepared["id"]})


@pytest.mark.parametrize("cell", list(published.CASES))
def test_private_case_identity_and_manifest_members(evidence_factory, cell):
    root, prepared, make = evidence_factory
    status = make(cell)
    record = published.validate_case(root, status, cell, previous_sequence=0, preparation_id=prepared["id"])
    assert record["result"]["cell"] == cell
    assert "probabilities" not in record["result"]
    assert record["result"]["native_evidence"]["tensors"]["probabilities"] == {
        "dtype": "torch.bfloat16", "shape": [2, 5]}


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
    candidate = str(outside)
    if path_kind == "symlink":
        link = inside / "alias.json"
        try:
            link.symlink_to(outside)
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                pytest.skip("file symlinks require Windows privilege")
            raise
        candidate = link.name
    with pytest.raises(Failure):
        published.private_path(inside, candidate)


class GraphMCP:
    def __init__(self):
        self.calls = []
        self.trace = {"state": "done", "status": "done", "trace_id": "e" * 64, "seconds": .1, "step": 4,
            "settings": dict(published.TRACE_REQUEST, dtype="torch.bfloat16", objective="centered_pre_softcap_logit",
                             source_projection="selected_features_without_bos_features_or_errors"),
            "counts": {"nodes": 300, "edges": 300, "tokens": 11, "layers": 26},
            "completeness": {"max_absolute_residual": .001, "source_scope": "full_sources_before_reference_projection"}}
        self.nodes = [{"index": i, "kind": "feature", "layer": 15, "feature": 4494 + i, "position": 4,
                       "effect": i, "label": str(i)} for i in range(300)]
        self.edges = [{"source": i, "target": (i + 1) % 300, "weight": .1} for i in range(300)]
        self.edit = lambda page: None

    def complete(self, method, **kwargs):
        assert method == "trace.read"
        self.calls.append(kwargs)
        page = {**deepcopy(self.trace), "tokens": [{"position": i, "token_id": i} for i in range(11)], "nodes": [], "edges": []}
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


def test_partial_collection_is_not_acceptance(tmp_path):
    out = tmp_path / "driver"
    out.mkdir()
    result = {"mode": "gemma", "replication_profile": published.PROFILE, "evidence_collection_complete": False,
              "completed": False, "checks": [], "published_actions": []}
    write_json(out / "result.json", result)
    collection = write_collection(out, tmp_path, result)
    assert not collection["evidence_collection_complete"] and not collection["accepted_replication"]
    assert finish(result) == 1 and not result["accepted_replication"]
    write_json(tmp_path / "review.json", {"schema": "caliper.published-dallas.review.v1", "protocol": published.PROFILE,
               "source_manifest_sha256": published.MANIFEST_SHA256, "collection_sha256": "0" * 64})
    with pytest.raises(Failure, match="incomplete"):
        verify_review(tmp_path, "driver/collection.json", "review.json")


@pytest.fixture
def callback_boundary(evidence_factory):
    root, prepared, make = evidence_factory
    state = {"value": make(10, 0), "sequence": 0, "calls": [], "fail_cell": None}
    state["value"]["inference"]["replication"].update(last_attempt=None, result=None)
    session = driver.Session.__new__(driver.Session)
    session.out, session.private_output_directory, session.run_id = root, root, "test-run"
    pending = []
    def body(method, **args):
        if method == "run.action":
            name = args["name"]
            cell = next(cell for cell in published.CASES if published.action_name(cell) == name)
            state["calls"].append(cell)
            state["sequence"] += 1
            candidate = make(cell, state["sequence"])
            failed = cell == state["fail_cell"]
            if failed:
                state["value"]["inference"]["replication"]["last_attempt"] = {
                    "case_id": name, "sequence": state["sequence"], "status": "error"}
            else:
                state["value"] = candidate
            pending.append({"kind": "run", "change": "action", "name": name, "status": "error" if failed else "ok"})
        elif method == "run.state":
            return {"attached": True}
        elif method != "state.set":
            raise AssertionError(method)
    session.mcp = SimpleNamespace(body=body)
    session.events = SimpleNamespace(pending=pending, wait=lambda seconds: None,
                                    until=lambda name, callback: callback(), complete=lambda method: deepcopy(state["value"]))
    return root, prepared, session, state


@pytest.mark.parametrize("failed", [None, 16, 29])
def test_all_cases_execute_real_callback_route_from_fresh_baselines(callback_boundary, failed):
    root, prepared, session, state = callback_boundary
    initial = driver.published_action(session, 10, prepared["id"])
    assert initial["completed"]
    state["fail_cell"] = failed
    records = list(driver.published_cases(session, prepared["id"]))
    assert state["calls"] == [10, 10, 14, 10, 16, 10, 18, 10, 20, 10, 24, 10, 26, 10, 29]
    for cell, baseline, case in records:
        assert baseline["completed"]
        assert case["completed"] is (cell != failed)
        if cell == failed:
            assert "action failed" in case["error"]
    assert len(list(root.glob("action-*.json"))) == 15


@pytest.fixture
def reviewed_collection(callback_boundary):
    from test_published_qualification import published_reply
    root, prepared, session, state = callback_boundary
    actions = [driver.published_action(session, 10, prepared["id"])]
    for cell, before, case in driver.published_cases(session, prepared["id"]):
        actions.extend((before, case))
    names = {"trace_time", "completeness", "complete_graph", "process_memory", "bounded_memory", "dictionary_identity_stable",
             "example_page_1_records", "example_page_2_records"}
    for name in ["cell" + str(cell) for cell in published.CASES] + ["examples-page-2"]:
        names.update(name + "." + suffix for suffix in (
            "capture_generation", "rendered_nodes", "host_labels", "example_page", "bar_height", "graph_composition",
            "examples_composition", "output_composition", "graph_visible_nodes", "example_row_layout", "output_bar_layout",
            "graph_resource_generation", "examples_resource_generation", "output_resource_generation",
            "graph_orthographic", "examples_orthographic", "output_orthographic",
            "graph_label_spacing", "examples_label_spacing", "output_label_spacing",
            "graph_decoded_text", "examples_decoded_text", "output_decoded_text",
            "graph_framing", "graph_panel_share", "compact_disclosures", "output_label_count"))
        for program in ("attribution", "probabilities", "example_tokens"):
            names.update(name + "." + program + "." + metric for metric in (
                "derive_time_ms", "derive_memory_bytes", "draw_time_ms", "draw_memory_bytes"))
    result = {"mode": "gemma", "replication_profile": published.PROFILE, "evidence_collection_complete": True,
              "pixel_evidence_collected": True, "preparation_id": prepared["id"], "published_actions": actions,
              "checks": [check(name, True, 0, "Synthetic external process receipt") for name in sorted(names)],
              "dictionary_evidence": published_reply(),
              "browser_assets": {name: {"status": 200} for name in ("caliper-views-loader.js", "caliper-views.js", "caliper-views.wasm")}}
    transport = GraphMCP()
    transport.nodes = transport.nodes[:3]
    transport.edges = [{"source": i, "target": (i + 1) % 3, "weight": .1} for i in range(3)]
    transport.trace["counts"].update(nodes=3, edges=3)
    result["trace"] = deepcopy(transport.trace)
    graph, _ = published.read_graph(SimpleNamespace(events=transport), transport.trace)
    graph["settings"].update(physical_batches=[3], logical_queues=[[0, 1, 2]])
    result["trace"]["settings"].update(physical_batch_count=1, logical_queue_count=1, schedule_in_full_export=True)
    graph["completeness"]["targets"] = [{"target": 1, "absolute_residual": .001}]
    graph["completeness"]["target_count"] = result["trace"]["completeness"]["target_count"] = 1
    write_json(root / f"trace-{graph['trace_id']}.json", graph)
    for gate in result["checks"]:
        if gate["name"] in ("process_memory", "bounded_memory"):
            gate["measured"] = 20_000_000_000
    out = root / "driver"
    out.mkdir()
    write_json(out / "result.json", result)
    collection = write_collection(out, root, result)
    receipt = root / "external-receipt.json"
    write_json(receipt, {"source": "Synthetic external process boundary"})
    evidence = [published.descriptor(root, receipt)]
    cases = {int(action["case_id"].split("cell")[1]): action["evidence"]["result"] for action in reversed(actions)}
    review = {"schema": "caliper.published-dallas.review.v1", "protocol": published.PROFILE,
        "source_manifest_sha256": published.MANIFEST_SHA256, "collection_sha256": published.file_hash(out / "collection.json"),
        "reference": {"producer": "circuit-tracer", "source_revision": published.SOURCE["source_revision"], "prompt": published.PROMPT,
                      "dtype": "bfloat16", "trace_request": published.TRACE_REQUEST, "evidence": evidence},
        "probability_policy": {"name": published.PROBABILITY_POLICY,
                               "absolute_tolerance": published.PROBABILITY_ABSOLUTE_TOLERANCE},
        "graph": {"caliper_trace_id": result["trace"]["trace_id"], "feature_node_jaccard": .9,
                  "maximum_edge_tolerance_ratio": 1., "edge_absolute_tolerance": 1e-6, "edge_relative_tolerance": .001,
                  "union_edges_compared": 3, "absent_edges_counted_as_zero": True},
        "cases": [{"case_id": published.action_name(cell), "caliper_sequence": cases[cell]["sequence"],
                   "caliper_npz_sha256": cases[cell]["evidence"]["sha256"], "reference_top_token_ids": cases[cell]["top_token_ids"],
                   "reference_top_text": [row[0]["text"] for row in cases[cell]["top_five"]], "probabilities_compared": 10,
                   "maximum_probability_absolute_error": .015, "published_outcome": {"reference": True, "caliper": True}}
                  for cell in published.CASES], "british_columbia_ineffective": True,
        "browser": {"program_views": True, "attached_before": True, "attached_after": True, "evidence": evidence},
        "capture_inspection": {"native": True, "browser": True, "all_case_controls": True, "group_ids": True,
                               "actual_outputs": True, "evidence": evidence},
        "parity": {"interior_over": 0, "interior_id_mismatch": 0, "maximum_interior_difference": 2, "edge_share": .01, "evidence": evidence},
        "memory": {"peak_process_bytes": 40_000_000_000, "bounded_peak_bytes": 40_000_000_000,
                   "minimum_host_available_bytes": 8_000_000_000, "evidence": evidence}}
    write_json(root / "review.json", review)
    return root, review, result


def test_review_cli_separates_complete_collection_from_acceptance(reviewed_collection):
    from _replication_review import main
    root, review, result = reviewed_collection
    assert finish(result) == 1 and not result["accepted_replication"]
    assert main(["--private-output-directory", str(root), "--collection", "driver/collection.json",
                 "--review", "review.json", "--out", str(root / "verified.json")]) == 0
    verified = json.loads((root / "verified.json").read_text())
    assert verified["accepted_replication"] and verified["gemma_acceptance"] and not verified["qwen_acceptance"]
    assert not json.loads((root / "driver/collection.json").read_text())["accepted_replication"]


@pytest.mark.parametrize("fault", ["missing_reference", "missing_browser", "missing_parity", "missing_inspection", "memory_cap",
                                  "reserve", "reference_prompt", "reference_settings", "case_coverage", "case_hash", "case_sequence",
                                  "full_vocabulary", "probability_error", "probability_policy", "top_id", "published_outcome", "british_columbia",
                                  "node_overlap", "edge_tolerance", "edge_coverage", "artifact", "collection_file"])
def test_required_external_evidence_cannot_be_missing_or_stale(reviewed_collection, fault):
    root, review, result = reviewed_collection
    if fault.startswith("missing_"):
        review.pop({"missing_reference": "reference", "missing_browser": "browser", "missing_parity": "parity",
                    "missing_inspection": "capture_inspection"}[fault])
    elif fault == "memory_cap": review["memory"]["peak_process_bytes"] += 1
    elif fault == "reserve": review["memory"]["minimum_host_available_bytes"] -= 1
    elif fault == "reference_prompt": review["reference"]["prompt"] = published.SOURCE["published_graph_prompt"]
    elif fault == "reference_settings": review["reference"]["trace_request"] = {**published.TRACE_REQUEST, "batch_size": 64}
    elif fault == "case_coverage": review["cases"].pop()
    elif fault == "case_hash": review["cases"][1]["caliper_npz_sha256"] = "b" * 64
    elif fault == "case_sequence": review["cases"][1]["caliper_sequence"] += 1
    elif fault == "full_vocabulary": review["cases"][1]["probabilities_compared"] = 5
    elif fault == "probability_error":
        review["cases"][1]["maximum_probability_absolute_error"] = float(np.nextafter(.015, np.inf))
    elif fault == "probability_policy": review["probability_policy"]["absolute_tolerance"] = .001
    elif fault == "top_id": review["cases"][1]["reference_top_token_ids"] = [0, 0]
    elif fault == "published_outcome": review["cases"][1]["published_outcome"]["reference"] = False
    elif fault == "british_columbia": review["cases"][-1]["reference_top_text"][1] = " Victoria"
    elif fault == "node_overlap": review["graph"]["feature_node_jaccard"] = .89
    elif fault == "edge_tolerance": review["graph"]["maximum_edge_tolerance_ratio"] = 1.01
    elif fault == "edge_coverage": review["graph"]["union_edges_compared"] = 2
    elif fault == "artifact": (root / "external-receipt.json").write_text("changed")
    elif fault == "collection_file": (root / "driver/result.json").write_text("{}")
    write_json(root / "review.json", review)
    with pytest.raises(Failure):
        verify_review(root, "driver/collection.json", "review.json")


def test_graph_prompt_identity_is_bound_to_preparation():
    transport = GraphMCP()
    with pytest.raises(Failure, match="prompt token sequence"):
        published.read_graph(SimpleNamespace(events=transport), transport.trace, token_ids=[99] * 11)


@pytest.mark.parametrize("show", [False, True])
@pytest.mark.skipif(os.name == "nt", reason="The owned child uses a POSIX executable.")
def test_missing_native_profile_retains_startup_and_owned_error_path(tmp_path, monkeypatch, show):
    import sys
    child = tmp_path / "owned-child"
    child.write_text(f"#!{sys.executable}\n" + "\n".join([
        "import json, os, pathlib, signal",
        "folder = pathlib.Path(os.environ['CALIPER_DATA_DIR'])",
        "(folder / 'api.json').write_text(json.dumps({'pid': os.getpid()}))",
        "signal.pause()",
    ]) + "\n")
    child.chmod(0o700)
    calls = []
    class CommandBoundary:
        def __init__(self, *args):
            pass
        def body(self, method, **args):
            calls.append((method, args))
            if method == "run.start":
                raise Failure("Native replication_profile dependency is absent.")
            return {}
        def close(self):
            pass
    monkeypatch.setattr(driver, "executable", lambda *args: child)
    monkeypatch.setattr(driver, "MCP", CommandBoundary)
    monkeypatch.setattr(driver, "Events", lambda mcp: SimpleNamespace(history=[]))
    args = SimpleNamespace(model="local-gemma", dictionary=str(tmp_path), prompt=published.PROMPT, family="gemma",
                           replication_profile=published.PROFILE, private_output_directory=tmp_path)
    with pytest.raises(driver.StartupFailure, match="dependency is absent") as raised:
        driver.Session(args, tmp_path / "session", show=show, interactive=not show)
    session = raised.value.session
    try:
        assert session.live()
        methods = [method for method, _ in calls]
        assert methods.index("applet.save") < methods.index("run.start")
        request = next(args for method, args in calls if method == "run.start")
        assert request["replication_profile"] == published.PROFILE
        assert request["private_output_directory"] == str(tmp_path)
        assert request["prompt"] == published.PROMPT
        assert methods.count("run.start") == 1
    finally:
        session.close()
    assert not session.live()


@pytest.mark.parametrize("options", [[], ["--replication-profile", published.PROFILE],
                                     ["--private-output-directory", "evidence"]])
def test_cli_requires_explicit_profile_pair_before_app_launch(tmp_path, options):
    model = tmp_path / "models--google--gemma-2-2b" / "snapshots" / driver.GEMMA_REVISION
    if not options:
        parsed = driver.parser().parse_args(["--model", str(model), "--dictionary", "dict", "--out", str(tmp_path)])
        assert parsed.replication_profile is None and parsed.private_output_directory is None
    else:
        with pytest.raises(Failure, match="must appear together"):
            driver.main(["--model", str(model), "--dictionary", "dict", "--out", str(tmp_path), *options])


def test_published_live_dispatch_has_no_acceptance_loop():
    import ast
    source = ast.parse((ROOT / "demos/circuit_trace/run.py").read_text())
    functions = {node.name: node for node in source.body if isinstance(node, ast.FunctionDef)}
    for name, collect in (("live_demo", False), ("acceptance", True)):
        calls = [node for node in ast.walk(functions[name]) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "published_demo"]
        assert len(calls) == 1
        assert next(arg.value.value for arg in calls[0].keywords if arg.arg == "collect") is collect


def test_bounded_profile_callback_needs_no_instrument(callback_boundary):
    root, prepared, session, state = callback_boundary
    original = session.mcp.body
    def body(method, **args):
        assert method != "state.set"
        return original(method, **args)
    session.mcp.body = body
    result = driver.published_action(session, 10, prepared["id"], display=False)
    assert result["completed"] and state["calls"] == [10]


@pytest.mark.parametrize("sequence", [1, 2])
def test_live_case_requires_a_new_result(evidence_factory, sequence):
    root, prepared, make = evidence_factory
    status = make(14, sequence)
    baseline_hash = published.validate_case(root, status, 14, previous_sequence=sequence - 1)["baseline_array_sha256"]
    calls = []
    session = SimpleNamespace(out=root, replication_profile=published.PROFILE, private_output_directory=root,
                              published_last_sequence=1, published_preparation_id=prepared["id"],
                              published_baseline_hash=baseline_hash, events=SimpleNamespace(complete=lambda method: status),
                              mcp=SimpleNamespace(body=lambda method, **args: calls.append((method, args))))
    event = {"kind": "run", "change": "action", "name": published.action_name(14), "status": "ok"}
    if sequence == 1:
        with pytest.raises(Failure, match="stale"):
            driver.Bridge(session).handle(event)
        assert not calls
    else:
        driver.Bridge(session).handle(event)
        assert session.published_last_sequence == 2 and "Before:" in calls[-1][1]["value"]


@pytest.mark.parametrize("field,value", [("dtype", "torch.float32"), ("objective", "centered_logit"),
                                        ("source_projection", "all_sources")])
def test_reference_trace_refuses_legacy_arithmetic(field, value):
    transport = GraphMCP()
    transport.trace["settings"][field] = value
    with pytest.raises(Failure, match="published objective"):
        published.read_graph(SimpleNamespace(events=transport), transport.trace)
    assert not transport.calls


def test_missing_case_readout_publication_is_not_complete(callback_boundary):
    root, prepared, session, state = callback_boundary
    original = session.mcp.body
    def body(method, **args):
        if method == "state.set":
            raise Failure("The readout publication failed.")
        return original(method, **args)
    session.mcp.body = body
    result = driver.published_action(session, 10, prepared["id"])
    assert not result["completed"] and "readout publication" in result["error"]
    assert not json.loads((root / "action-1.json").read_text())["completed"]


@pytest.mark.parametrize("missing", ["cell29.graph_label_spacing", "cell10.output_resource_generation", "bounded_memory",
                                     "dictionary_identity_stable", "examples-page-2.compact_disclosures"])
def test_absent_native_requirements_cannot_be_accepted(reviewed_collection, missing):
    root, review, result = reviewed_collection
    result["checks"] = [gate for gate in result["checks"] if gate["name"] != missing]
    write_json(root / "driver/result.json", result)
    write_collection(root / "driver", root, result)
    review["collection_sha256"] = published.file_hash(root / "driver/collection.json")
    write_json(root / "review.json", review)
    with pytest.raises(Failure, match="Required native"):
        verify_review(root, "driver/collection.json", "review.json")


@pytest.mark.parametrize("mode", ["--show", "--demo"])
def test_published_live_cli_starts_interactive_session_before_any_preflight(tmp_path, monkeypatch, mode):
    import sys
    launches = []
    def launch(command, **kwargs):
        launches.append(kwargs["env"])
        raise Failure("The external app boundary refused this test launch.")
    monkeypatch.setattr(driver, "executable", lambda *args: Path(sys.executable))
    monkeypatch.setattr(driver.subprocess, "Popen", launch)
    model = tmp_path / "models--google--gemma-2-2b" / "snapshots" / driver.GEMMA_REVISION
    out = tmp_path / "driver"
    code = driver.main(["--model", str(model), "--dictionary", str(tmp_path / "dictionary"),
                        "--replication-profile", published.PROFILE, "--private-output-directory", str(tmp_path),
                        "--out", str(out), mode])
    assert code == 1 and len(launches) == 1
    assert not (out / "bounded").exists() and (out / "acceptance").is_dir()
    assert launches[0]["CALIPER_HIDDEN"] == ("0" if mode == "--show" else "1")
    result = json.loads((out / "result.json").read_text())
    assert not result["accepted_replication"] and "external app boundary" in result["error"]


@pytest.mark.parametrize("fault", ["f32", "missing_import", "stale_fingerprint"])
def test_external_review_cannot_replace_current_bf16_resource_evidence(reviewed_collection, fault):
    """Synthetic external receipts cannot bypass the independent resource gate."""
    from _replication_review import main
    from test_gemma_acceptance import gemma_reply
    root, review, result = reviewed_collection
    if fault == "f32":
        result["dictionary_evidence"] = gemma_reply()
    elif fault == "missing_import":
        result["dictionary_evidence"].pop("import_binding")
    else:
        result["dictionary_evidence"]["index"]["dictionary_fingerprint"] = "0" * 64
    out = root / "driver"
    write_json(out / "result.json", result)
    write_collection(out, root, result)
    review["collection_sha256"] = published.file_hash(out / "collection.json")
    write_json(root / "review.json", review)
    assert main(["--private-output-directory", str(root), "--collection", "driver/collection.json",
                 "--review", "review.json", "--out", str(root / "refused.json")]) == 1
    refused = json.loads((root / "refused.json").read_text())
    assert not refused["accepted_replication"] and not refused["gemma_acceptance"]
    assert "dictionary_qualification" in refused["failed_checks"]
    external = next(gate for gate in refused["checks"] if gate["name"] == "external_replication_review")
    assert external["passed"]
