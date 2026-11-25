"""Bind independent review evidence to a completed private collection."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from _mcp import Failure
from _published import (CASES, MANIFEST_SHA256, PROFILE, PROMPT, SOURCE, TRACE_REQUEST, action_name,
                        descriptor, file_hash, finite, private_path, read_json, require, validate_case, verify_descriptor)


def write_collection(out, root, result):
    from run import save
    files = {str(path.resolve()) for path in out.rglob("*") if path.is_file() and path.name != "collection.json"}
    for action in [result.get("bounded_evidence", {}), *result.get("published_actions", [])]:
        evidence = action.get("evidence")
        if evidence:
            for item in (evidence["metadata"], evidence["preparation_file"], evidence["result"]["evidence"]):
                files.add(str(verify_descriptor(root, item)))
    collection = {"schema": "caliper.published-dallas.collection.v1", "protocol": PROFILE,
                  "source_manifest_sha256": MANIFEST_SHA256,
                  "evidence_collection_complete": result.get("evidence_collection_complete") is True and not result.get("error"),
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
    """Validate comparator metrics and mandatory review receipts without model code."""
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
        for item in files:
            verify_descriptor(root, item)
        result = read_json(verify_descriptor(root, collection["result"]))
        require(result.get("replication_profile") == PROFILE and result.get("evidence_collection_complete") is True
                and not result.get("error") and not result.get("cleanup_error"), "The collection driver did not complete.")
        actions = result.get("published_actions", [])
        cells = [10] + [item for cell in CASES if cell != 10 for item in (10, cell)]
        require([item.get("case_id") for item in actions] == [action_name(cell) for cell in cells],
                "The collection lacks fresh baselines or all seven callbacks.")
        sequence, cases, baseline_hash = 0, {}, None
        for cell, action in zip(cells, actions):
            require(action.get("completed") is True and action.get("prior_sequence") == sequence
                    and action.get("event", {}).get("status") == "ok"
                    and action["event"].get("name") == action_name(cell)
                    and all(action.get(key, {}).get("attached") is True for key in ("attachment_before", "attachment_after")),
                    "A published callback failed or detached.")
            evidence = validate_case(root, action["status"], cell, previous_sequence=sequence,
                                     preparation_id=result["preparation_id"])
            require(evidence == action["evidence"], "A collected case differs from its current private evidence.")
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
                    and row["maximum_probability_absolute_error"] <= .001
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
