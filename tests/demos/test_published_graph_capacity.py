"""Bounded complete graph evidence and unchanged metadata limits."""
from contextlib import nullcontext
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_published_dallas import (callback_boundary, evidence_factory, published, reviewed_collection, write_json)
from _mcp import Failure


@pytest.mark.parametrize("reader,limit", [
    (published.read_json, published.JSON_EVIDENCE_LIMIT),
    (published.read_full_graph, published.FULL_GRAPH_JSON_LIMIT),
])
def test_oversized_json_refuses_before_open(tmp_path, monkeypatch, reader, limit):
    target = tmp_path / "oversized.json"
    with target.open("wb") as stream:
        stream.truncate(limit + 1)

    def refuse_open(*args, **kwargs):
        pytest.fail("Oversized evidence reached the file read.")

    monkeypatch.setattr(Path, "open", refuse_open)
    with pytest.raises(Failure, match="JSON exceeds the evidence limit"):
        reader(target)
    assert published.JSON_EVIDENCE_LIMIT == 16 * 1024 * 1024


def test_full_graph_rechecks_open_file_size(tmp_path, monkeypatch):
    target = tmp_path / "replaced.json"
    with target.open("wb") as stream:
        stream.truncate(published.FULL_GRAPH_JSON_LIMIT + 1)
    with target.open("rb") as stream:
        def refuse_read(size):
            pytest.fail("The oversized open file reached the payload read.")

        guarded = SimpleNamespace(fileno=stream.fileno, read=refuse_read)
        monkeypatch.setattr(Path, "stat", lambda self: SimpleNamespace(st_size=0))
        monkeypatch.setattr(Path, "open", lambda self, mode: nullcontext(guarded))
        with pytest.raises(Failure, match="Full graph JSON exceeds"):
            published.read_full_graph(target)


@pytest.mark.parametrize("extra", [-1, 0, 1])
def test_full_graph_bounds_read_after_file_growth(monkeypatch, extra):
    limit = 64
    raw = b'{"label":"\\u03b1"}'
    raw += b" " * (limit + extra - len(raw))
    reads = []

    class GrowingFile(io.BytesIO):
        def fileno(self):
            return -1

        def read(self, size=-1):
            reads.append(size)
            assert size == limit + 1
            return super().read(size)

    target = SimpleNamespace(stat=lambda: SimpleNamespace(st_size=0),
                             open=lambda mode: GrowingFile(raw))
    monkeypatch.setattr(published, "FULL_GRAPH_JSON_LIMIT", limit)
    monkeypatch.setattr(published.os, "fstat", lambda fd: SimpleNamespace(st_size=0))
    if extra > 0:
        with pytest.raises(Failure, match="Full graph JSON exceeds"):
            published.read_full_graph(target)
    else:
        assert published.read_full_graph(target) == {"label": "α"}
    assert reads == [limit + 1]


@pytest.mark.parametrize("reader", [published.read_json, published.read_full_graph])
@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_nonfinite_json_refuses_in_nested_fields(tmp_path, reader, number):
    target = tmp_path / "nonfinite.json"
    target.write_text('{"completeness":{"targets":[{"expected":' + number + '}]}}', encoding="utf-8")
    with pytest.raises(Failure, match="Nonfinite JSON evidence"):
        reader(target)


def capacity_graph(token_ids, *, feature_count, edge_count):
    from caliper.interp._graph import Graph

    nodes = [{"index": i, "id": f"feature:{i // 200}:{1 + i % 10}:{i}", "kind": "feature",
              "layer": i // 200, "position": 1 + i % 10, "feature": i, "activation": .5,
              "influence": .0001220703125, "effect": -.00010204315185546875, "label": f"L{i // 200} F{i}"}
             for i in range(feature_count)]
    tokens = [{"position": i, "token_id": token, "text": str(token)} for i, token in enumerate(token_ids)]
    for token in tokens:
        nodes.append({"index": len(nodes), "id": f"token:{token['position']}", "kind": "token", "layer": -1,
                      "position": token["position"], "token_id": token["token_id"], "activation": 1.,
                      "influence": .0001220703125, "effect": .25, "label": token["text"]})
    logits = []
    for token in range(10):
        node = {"index": len(nodes), "id": f"logit:{len(tokens) - 1}:{token}", "kind": "logit", "layer": 26,
                "position": len(tokens) - 1, "token_id": token, "activation": .5, "probability": .1,
                "influence": .1, "effect": .25, "label": str(token)}
        nodes.append(node)
        logits.append(node)
    # Four-digit endpoints and an exact BF16 weight reproduce the reported byte boundary.
    source, target = (1000, 3000) if feature_count == 5000 else (0, feature_count // 2)
    span = min(512, feature_count // 2)
    edges = [{"source": source + i // span, "target": target + i % span, "weight": -.00010204315185546875}
             for i in range(edge_count)]
    targets = logits + nodes[:feature_count]
    diagnostics = [{"id": node["id"], "expected": .5, "incoming": .5,
                    "biases": dict.fromkeys(("decoder", "normalization", "attention", "intervention",
                                             "encoder", "unembedding"), 0.),
                    "absolute_residual": 0., "relative_residual": 0.,
                    "reference_projection": {"selected_source_incoming": .5, "unselected_feature_incoming": 0.,
                                             "completeness_scope": "projection_only"}}
                   for node in targets]
    queues = [[node["id"] for node in logits]]
    queues.extend([node["id"] for node in nodes[start:min(start + 2048, feature_count)]]
                  for start in range(0, feature_count, 2048))
    batches = [min(512, len(queue) - start) for queue in queues for start in range(0, len(queue), 512)]
    return Graph.build(nodes=nodes, edges=edges, tokens=tokens, layers=26, trace_id="c" * 64, device="cpu",
                       omitted_influence=.1, error_share=0., feature_share=.9, logits=logits,
                       completeness={"target_count": len(diagnostics), "targets": diagnostics,
                                     "max_absolute_residual": 0., "max_relative_residual": 0.,
                                     "source_scope": "full_sources_before_reference_projection"},
                       settings=dict(published.TRACE_REQUEST, dtype="torch.bfloat16",
                                     objective="centered_pre_softcap_logit",
                                     source_projection="selected_features_without_bos_features_or_errors",
                                     physical_batches=batches, logical_queues=queues))


def test_small_graph_producer_consumer_boundary(tmp_path):
    from caliper.interp._replication_export import export_graph
    from _replication_review import _full_graph

    graph = capacity_graph(list(range(11)), feature_count=16, edge_count=32)
    record = export_graph(graph, tmp_path)
    target = published.verify_descriptor(tmp_path, record)
    evidence = {"preparation": {"prompts": {"dallas": {"token_ids": list(range(11))}}}}
    _full_graph(target, evidence, graph.read(node_limit=0, edge_limit=0))
    decoded = published.read_full_graph(target)
    assert decoded["settings"] == graph.settings
    assert decoded["completeness"] == graph.completeness
    assert decoded["nodes"] == graph.nodes and decoded["edges"] == graph.edges


def test_capacity_graph_above_metadata_limit_reaches_collection_and_review(reviewed_collection):
    import json
    from caliper.interp._replication_export import export_graph
    from _replication_review import main, verify_review, write_collection

    root, review, result = reviewed_collection
    prepared = result["published_actions"][0]["evidence"]["preparation"]
    graph = capacity_graph(prepared["prompts"]["dallas"]["token_ids"], feature_count=5000,
                           edge_count=published.TRACE_REQUEST["max_edges"])
    record = export_graph(graph, root)
    target = published.verify_descriptor(root, record)
    size = target.stat().st_size
    assert 16 * 1024 * 1024 < size <= published.FULL_GRAPH_JSON_LIMIT
    assert len(graph.nodes) <= published.TRACE_REQUEST["max_nodes"]
    assert len(graph.edges) == published.TRACE_REQUEST["max_edges"]
    assert len(graph.completeness["targets"]) == 5010
    with pytest.raises(Failure, match="Private JSON exceeds"):
        published.read_json(target)
    decoded = published.read_full_graph(target)
    assert decoded["nodes"] == graph.nodes and decoded["edges"] == graph.edges
    assert decoded["settings"] == graph.settings
    assert decoded["completeness"] == graph.completeness
    assert decoded["next_node_offset"] is None and decoded["next_edge_offset"] is None
    del decoded
    result["trace"].update(graph.read(node_limit=0, edge_limit=0))
    write_json(root / "driver/result.json", result)
    collection = write_collection(root / "driver", root, result)
    assert record in collection["files"]
    assert collection["evidence_collection_complete"] and not collection["accepted_replication"]
    review["collection_sha256"] = published.file_hash(root / "driver/collection.json")
    review["graph"].update(caliper_trace_id=graph.trace_id, union_edges_compared=len(graph.edges))
    write_json(root / "review.json", review)
    assert verify_review(root, "driver/collection.json", "review.json")["verified"]
    assert main(["--private-output-directory", str(root), "--collection", "driver/collection.json",
                 "--review", "review.json", "--out", str(root / "verified.json")]) == 0
    assert published.read_json(root / "verified.json")["accepted_replication"]
    print(json.dumps({"graph_bytes": size, "nodes": len(graph.nodes), "edges": len(graph.edges),
                      "diagnostics": len(graph.completeness["targets"]),
                      "physical_batches": len(graph.settings["physical_batches"]),
                      "logical_queues": len(graph.settings["logical_queues"]), "sha256": record["sha256"]}))


@pytest.mark.parametrize("consumer", ["collection", "review"])
@pytest.mark.parametrize("fault", ["oversize", "nonfinite", "overflow"])
def test_graph_consumers_refuse_invalid_json(reviewed_collection, monkeypatch, consumer, fault):
    import json
    from _replication_review import verify_review, write_collection

    root, review, result = reviewed_collection
    target = root / f"trace-{result['trace']['trace_id']}.json"
    if fault == "oversize":
        monkeypatch.setattr(published, "FULL_GRAPH_JSON_LIMIT", target.stat().st_size - 1)
        message = "Full graph JSON exceeds"
    else:
        graph = published.read_full_graph(target)
        graph["completeness"]["targets"][0]["expected"] = float("nan")
        raw = json.dumps(graph)
        target.write_text(raw if fault == "nonfinite" else raw.replace("NaN", "1e999"), encoding="utf-8")
        message = "Nonfinite JSON evidence"
    if consumer == "collection":
        with pytest.raises(Failure, match=message):
            write_collection(root / "driver", root, result)
    else:
        collection = published.read_json(root / "driver/collection.json")
        for item in collection["files"]:
            if item["file"] == target.name:
                item.update(published.descriptor(root, target))
        write_json(root / "driver/collection.json", collection)
        review["collection_sha256"] = published.file_hash(root / "driver/collection.json")
        write_json(root / "review.json", review)
        with pytest.raises(Failure, match=message):
            verify_review(root, "driver/collection.json", "review.json")
