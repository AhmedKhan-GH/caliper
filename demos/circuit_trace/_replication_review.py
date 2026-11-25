"""Bind independent review evidence to a completed private collection."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from _mcp import Failure
from _published import PROBABILITY_ABSOLUTE_TOLERANCE, PROBABILITY_POLICY
from _published import (CASES, MANIFEST_SHA256, PROFILE, PROMPT, SOURCE, TRACE_REQUEST, action_name,
                        descriptor, digest, file_hash, finite, private_path, read_json, require, validate_case, verify_descriptor)


def _case_files(evidence, *, complete):
    record = evidence["result"]
    files = [evidence["metadata"], evidence["preparation_file"], record["evidence"]]
    require(not complete or "native_evidence" in record, "Complete cases require native evidence.")
    if "native_evidence" in record:
        files.append(record["native_evidence"])
    observations = record.get("observations", {})
    require(not complete or record.get("cell") != 29 or "Say Victoria" in observations,
            "Complete cases require the Say Victoria observation.")
    files.extend(item["native_evidence"] for item in observations.values())
    return files


def _graph_path(root, evidence, trace):
    trace_id = trace.get("trace_id")
    require(isinstance(trace_id, str) and len(trace_id) == 64
            and all(char in "0123456789abcdef" for char in trace_id), "Invalid full graph identity.")
    folder = verify_descriptor(root, evidence["preparation_file"]).parent
    return folder / f"trace-{trace_id}.json"


def _full_graph(path, evidence, trace):
    """Bind the complete export to the paginated trace and prepared prompt."""
    graph = read_json(path)
    require(isinstance(graph, dict) and graph.get("trace_id") == trace["trace_id"] and graph.get("status") == "done"
            and digest(graph.get("counts")) == digest(trace.get("counts")), "Full graph identity or counts differ.")
    settings = graph["settings"]
    summary = {key: value for key, value in settings.items() if key not in ("physical_batches", "logical_queues")}
    if "physical_batches" in settings:
        require(isinstance(settings["physical_batches"], list) and isinstance(settings.get("logical_queues"), list),
                "Full graph schedules are absent.")
        summary.update(physical_batch_count=len(settings["physical_batches"]),
                       logical_queue_count=len(settings["logical_queues"]), schedule_in_full_export=True)
    require(not trace["settings"].get("schedule_in_full_export")
            or {"physical_batches", "logical_queues"} <= settings.keys(), "Full graph schedules are absent.")
    completeness = {key: value for key, value in graph["completeness"].items() if key != "targets"}
    require(digest(summary) == digest(trace["settings"]) and digest(completeness) == digest(trace["completeness"]),
            "Full graph settings or completeness differ.")
    require(type(completeness.get("target_count")) is int and isinstance(graph["completeness"].get("targets"), list)
            and len(graph["completeness"]["targets"]) == completeness["target_count"],
            "Full graph target diagnostics are incomplete.")
    require(all(type(settings.get(key)) is type(value) and settings[key] == value for key, value in TRACE_REQUEST.items())
            and settings.get("dtype") == "torch.bfloat16"
            and settings.get("objective") == "centered_pre_softcap_logit"
            and settings.get("source_projection") == "selected_features_without_bos_features_or_errors"
            and completeness.get("source_scope") == "full_sources_before_reference_projection",
            "Full graph reference profile differs.")
    for kind, maximum in (("nodes", TRACE_REQUEST["max_nodes"]), ("edges", TRACE_REQUEST["max_edges"])):
        count = graph["counts"][kind]
        require(type(count) is int and 0 <= count <= maximum and len(graph[kind]) == count,
                "Full graph coverage differs.")
    tokens = graph["tokens"]
    require(len(tokens) == graph["counts"]["tokens"]
            and all(type(item.get(key)) is int for item in tokens for key in ("position", "token_id"))
            and [item.get("position") for item in tokens] == list(range(len(tokens)))
            and [item.get("token_id") for item in tokens] == evidence["preparation"]["prompts"]["dallas"]["token_ids"],
            "Full graph prompt token identities differ.")
    indices = [node["index"] for node in graph["nodes"]]
    require(all(type(index) is int for index in indices) and indices == list(range(len(indices)))
            and graph.get("next_node_offset") is None and graph.get("next_edge_offset") is None,
            "Full graph node coverage differs.")
    pairs = [(edge["source"], edge["target"]) for edge in graph["edges"]]
    require(all(type(index) is int and 0 <= index < len(indices) for pair in pairs for index in pair)
            and len(set(pairs)) == len(pairs)
            and all(finite(edge.get("weight"), -float("inf")) for edge in graph["edges"]), "Full graph edges differ.")


def write_collection(out, root, result):
    from run import save
    files = {str(path.resolve()) for path in out.rglob("*") if path.is_file() and path.name != "collection.json"}
    complete = result.get("evidence_collection_complete") is True and not result.get("error") and not result.get("cleanup_error")
    graphs = set()
    actions = result.get("published_actions", [])
    require(not complete or actions, "A complete collection requires published actions.")
    for action, required in [(result.get("bounded_evidence", {}), False), *[(item, complete) for item in actions]]:
        evidence = action.get("evidence")
        require(not required or evidence, "A complete collection requires case evidence.")
        if evidence:
            for item in _case_files(evidence, complete=required):
                files.add(str(verify_descriptor(root, item)))
            if required or result.get("trace", {}).get("trace_id"):
                graph = _graph_path(root, evidence, result.get("trace", {}))
                require(not required or graph.is_file(), "A complete collection requires the full graph export.")
                if graph.exists():
                    graph = verify_descriptor(root, descriptor(root, graph))
                    if required and graph not in graphs:
                        exported = read_json(graph)
                        require(isinstance(exported, dict) and exported.get("trace_id") == result["trace"]["trace_id"],
                                "Full graph identity differs.")
                        graphs.add(graph)
                    files.add(str(graph))
    collection = {"schema": "caliper.published-dallas.collection.v1", "protocol": PROFILE,
                  "source_manifest_sha256": MANIFEST_SHA256,
                  "evidence_collection_complete": bool(complete),
                  "accepted_replication": False, "result": descriptor(root, out / "result.json"),
                  "files": [descriptor(root, Path(path)) for path in sorted(files)]}
    save(out / "collection.json", collection)
    return collection


def _artifacts(root, record):
    evidence = record.get("evidence")
    require(isinstance(evidence, list) and evidence, "External review has no source artifacts.")
    for item in evidence:
        verify_descriptor(root, item)


def verify_review(root, collection_path, review_path):
    """Bind local evidence and validate trusted comparator receipts without model code."""
    try:
        collection = read_json(private_path(root, collection_path))
        review = read_json(private_path(root, review_path))
        require(collection.get("schema") == "caliper.published-dallas.collection.v1"
                and collection.get("protocol") == PROFILE and collection.get("source_manifest_sha256") == MANIFEST_SHA256
                and collection.get("evidence_collection_complete") is True, "The published collection is incomplete.")
        require(review.get("schema") == "caliper.published-dallas.review.v1"
                and review.get("protocol") == PROFILE and review.get("source_manifest_sha256") == MANIFEST_SHA256
                and review.get("collection_sha256") == file_hash(private_path(root, collection_path)),
                "External review belongs to another collection or source.")
        files = collection.get("files")
        require(isinstance(files, list) and files, "The collection has no file inventory.")
        inventory = set()
        for item in files:
            key = str(verify_descriptor(root, item)), item["sha256"]
            require(key not in inventory, "The collection repeats an evidence file.")
            inventory.add(key)

        def inventoried(item):
            path = verify_descriptor(root, item)
            require((str(path), item["sha256"]) in inventory, "Required evidence is absent from the collection inventory.")
            return path

        result = read_json(inventoried(collection["result"]))
        require(result.get("replication_profile") == PROFILE and result.get("evidence_collection_complete") is True
                and not result.get("error") and not result.get("cleanup_error"), "The collection driver did not complete.")
        actions = result.get("published_actions", [])
        cells = [10] + [item for cell in CASES if cell != 10 for item in (10, cell)]
        require([item.get("case_id") for item in actions] == [action_name(cell) for cell in cells],
                "The collection lacks fresh baselines or all seven callbacks.")
        sequence, cases, baseline_hash, graphs = 0, {}, None, set()
        bounded = result.get("bounded_evidence", {}).get("evidence")
        if bounded:
            for item in _case_files(bounded, complete=False):
                inventoried(item)
        for cell, action in zip(cells, actions):
            require(action.get("completed") is True and action.get("prior_sequence") == sequence
                    and action.get("event", {}).get("status") == "ok"
                    and action["event"].get("name") == action_name(cell)
                    and all(action.get(key, {}).get("attached") is True for key in ("attachment_before", "attachment_after")),
                    "A published callback failed or detached.")
            evidence = validate_case(root, action["status"], cell, previous_sequence=sequence,
                                     preparation_id=result["preparation_id"])
            require(evidence == action["evidence"], "A collected case differs from its current private evidence.")
            for item in _case_files(evidence, complete=True):
                inventoried(item)
            graph_path = _graph_path(root, evidence, result["trace"])
            if graph_path not in graphs:
                inventoried(descriptor(root, graph_path))
                _full_graph(graph_path, evidence, result["trace"])
                graphs.add(graph_path)
            require(baseline_hash is None or baseline_hash == evidence["baseline_array_sha256"],
                    "The collection changed baseline arrays between cases.")
            baseline_hash = evidence["baseline_array_sha256"]
            prior_step = (action["before_status"]["inference"].get("comparison") or {}).get("completed_step")
            require(prior_step is None or action["status"]["inference"]["comparison"]["completed_step"] > prior_step,
                    "A collected output publication is stale.")
            require(finite(action.get("callback_seconds")) and finite(action.get("full_action_seconds"))
                    and action["full_action_seconds"] >= action["callback_seconds"], "The complete action timer is absent.")
            if cell != 10:
                require(max(action["full_action_seconds"], evidence["result"]["action_seconds"]) <= 1,
                        "A published case exceeds the one-second steering limit.")
            cases.setdefault(cell, evidence["result"])
            sequence += 1
        checks = {item["name"]: item for item in result["checks"]}
        required = {"trace_time", "completeness", "complete_graph", "process_memory", "bounded_memory", "dictionary_identity_stable",
                    "example_page_1_records", "example_page_2_records"}
        for name in ["cell" + str(cell) for cell in CASES] + ["examples-page-2"]:
            required.update(name + "." + suffix for suffix in (
                "capture_generation", "rendered_nodes", "host_labels", "example_page", "bar_height",
                "graph_composition", "examples_composition", "output_composition",
                "graph_resource_generation", "examples_resource_generation", "output_resource_generation",
                "graph_orthographic", "examples_orthographic", "output_orthographic",
                "graph_label_spacing", "examples_label_spacing", "output_label_spacing",
                "graph_decoded_text", "examples_decoded_text", "output_decoded_text",
                "graph_framing", "graph_panel_share", "compact_disclosures", "output_label_count",
                "graph_visible_nodes", "example_row_layout", "output_bar_layout"))
            for program in ("attribution", "probabilities", "example_tokens"):
                required.update(name + "." + program + "." + metric for metric in (
                    "derive_time_ms", "derive_memory_bytes", "draw_time_ms", "draw_memory_bytes"))
        require(required <= checks.keys() and all(checks[key].get("passed") is True
                                                  and checks[key].get("required") is True for key in required),
                "Required native, completeness, timing, or memory checks are absent or failed.")
        reference = review["reference"]
        require(reference.get("producer") == "circuit-tracer" and reference.get("source_revision") == SOURCE["source_revision"]
                and reference.get("prompt") == PROMPT and reference.get("dtype") == "bfloat16"
                and reference.get("trace_request") == TRACE_REQUEST, "Reference identity or parameters differ.")
        _artifacts(root, reference)
        policy = review.get("probability_policy", {})
        require(isinstance(policy, dict) and policy.get("name") == PROBABILITY_POLICY
                and policy.get("absolute_tolerance") == PROBABILITY_ABSOLUTE_TOLERANCE,
                "The probability comparison lacks the approved reference envelope.")
        graph = review["graph"]
        require(graph.get("caliper_trace_id") == result["trace"]["trace_id"]
                and finite(graph.get("feature_node_jaccard")) and .9 <= graph["feature_node_jaccard"] <= 1
                and finite(graph.get("maximum_edge_tolerance_ratio")) and graph["maximum_edge_tolerance_ratio"] <= 1
                and graph.get("edge_absolute_tolerance") == 1e-6 and graph.get("edge_relative_tolerance") == .001
                and type(graph.get("union_edges_compared")) is int
                and graph["union_edges_compared"] >= result["trace"]["counts"]["edges"]
                and graph.get("absent_edges_counted_as_zero") is True, "Independent graph comparison failed or is incomplete.")
        rows = review["cases"]
        require(isinstance(rows, list) and [row.get("case_id") for row in rows] == [action_name(cell) for cell in CASES],
                "External case comparison coverage differs.")
        for cell, row in zip(CASES, rows):
            record = cases[cell]
            expected = CASES[cell]["published_output_labels"][0].rsplit(" ", 1)[0]
            require(row.get("caliper_sequence") == record["sequence"]
                    and row.get("caliper_npz_sha256") == record["evidence"]["sha256"]
                    and row.get("reference_top_token_ids") == record["top_token_ids"]
                    and row.get("reference_top_text") == [record["top_five"][i][0]["text"] for i in (0, 1)]
                    and row["reference_top_text"] == [" Austin", expected]
                    and row.get("probabilities_compared") == 2 * record["evidence"]["shape"][1]
                    and finite(row.get("maximum_probability_absolute_error"))
                    and row["maximum_probability_absolute_error"] <= PROBABILITY_ABSOLUTE_TOLERANCE
                    and row.get("published_outcome") == {"reference": True, "caliper": True},
                    "Published outcomes or full-vocabulary reference agreement failed.")
        require(review["british_columbia_ineffective"] is True, "The published ineffective outcome is absent.")
        browser, inspection, parity, memory = [review[key] for key in ("browser", "capture_inspection", "parity", "memory")]
        for record in (browser, inspection, parity, memory):
            _artifacts(root, record)
        require(all(browser.get(key) is True for key in ("program_views", "attached_before", "attached_after"))
                and all(result["browser_assets"][name]["status"] == 200 for name in (
                    "caliper-views-loader.js", "caliper-views.js", "caliper-views.wasm")), "Browser evidence is incomplete.")
        require(all(inspection.get(key) is True for key in ("native", "browser", "all_case_controls", "group_ids", "actual_outputs")),
                "Independent capture inspection is incomplete.")
        require(parity.get("interior_over") == 0 and parity.get("interior_id_mismatch") == 0
                and finite(parity.get("maximum_interior_difference")) and parity["maximum_interior_difference"] <= 2
                and finite(parity.get("edge_share")) and parity["edge_share"] <= .01, "Native and browser parity failed.")
        require(finite(memory.get("peak_process_bytes"), 1) and memory["peak_process_bytes"] <= 40_000_000_000
                and finite(memory.get("bounded_peak_bytes"), 1) and memory["bounded_peak_bytes"] <= 40_000_000_000
                and finite(memory.get("minimum_host_available_bytes"), 8_000_000_000), "Memory or host reserve evidence failed.")
        for field, name in (("peak_process_bytes", "process_memory"), ("bounded_peak_bytes", "bounded_memory")):
            measurements = [item.get("measured") for item in result["checks"] if item["name"] == name]
            require(all(finite(value) and value <= memory[field] for value in measurements),
                    "External memory evidence understates the driver measurement.")
        return {"verified": True, "collection_sha256": review["collection_sha256"],
                "review_sha256": file_hash(private_path(root, review_path)), "cases": list(CASES)}
    except (KeyError, TypeError, ValueError, OSError, IndexError) as error:
        raise Failure(f"Invalid external replication review: {error}") from error


def main(argv=None):
    from _checks import finish
    from run import save
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-output-directory", type=Path, required=True)
    parser.add_argument("--collection", required=True, help="Collection path relative to the private root")
    parser.add_argument("--review", required=True, help="Review path relative to the private root")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.private_output_directory.resolve(strict=True)
    require(args.out.resolve().is_relative_to(root) and not args.out.exists(), "The review output must be a new private file.")
    collection = read_json(private_path(root, args.collection))
    result = deepcopy(read_json(verify_descriptor(root, collection["result"])))
    result.update(replication_evidence_root=str(root), replication_collection=args.collection,
                  replication_review=args.review, completed=result.get("evidence_collection_complete") is True)
    code = finish(result)
    save(args.out, result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
