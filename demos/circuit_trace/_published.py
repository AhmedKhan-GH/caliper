"""Exact published cases and private runtime evidence validation."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import pickletools
import re
import struct
import zipfile
from collections import OrderedDict

import numpy as np

from _inference_evidence import bound_completion, inference_state, text_hash
from _mcp import Failure

PROFILE = "published_dallas_v1"
# The approved reference envelope supersedes the pinned source record's earlier limit.
PROBABILITY_POLICY = "reference_publication_envelope_v1"
PROBABILITY_ABSOLUTE_TOLERANCE = .015
MANIFEST_SHA256 = "0988e260b5f0c9cb8bfc6791edba779634b61f95c4f68e73ab6cdbdb390afa8d"


def require(condition, message):
    if not condition:
        raise Failure(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest():
    path = Path(__file__).with_name("published_dallas_v1.json")
    require(file_hash(path) == MANIFEST_SHA256, "The published source manifest differs from the pinned source.")
    return json.loads(path.read_text(encoding="utf-8"))


SOURCE = manifest()
PROMPT = SOURCE["prompts"]["dallas"]
CASES = {10: {"label": "Baseline", "operations": [],
              "published_output_labels": SOURCE["baseline_published_output_labels"]},
         **{case["cell"]: {k: v for k, v in case.items() if k != "cell"} for case in SOURCE["interventions"]}}
TRACE_REQUEST = {"profile": "circuit_tracer_82996f14", "max_targets": SOURCE["graph"]["generation"]["max_feature_nodes"],
                 "max_nodes": 8192, "max_edges": 262144,
                 **{key: SOURCE["graph"]["generation"][key] for key in ("max_n_logits", "desired_logit_prob")},
                 **SOURCE["graph"]["pruning"],
                 **{key: SOURCE["graph"][key] for key in ("batch_size", "update_interval")}}
JSON_EVIDENCE_LIMIT = 16 * 1024 * 1024
# Allow 128 bytes per edge, 4 KiB per node, and 4 KiB per target with its schedule.
# At 262144 edges, 8192 nodes, and 5010 targets, these allowances total under 84 MiB.
# Another 16 MiB covers other fields. A 128 MiB ceiling leaves finite headroom.
FULL_GRAPH_JSON_LIMIT = 128 * 1024 * 1024


def action_name(cell):
    require(type(cell) is int and cell in CASES, "Unknown published case.")
    return f"{PROFILE}.cell{cell}"


def finite(value, minimum=0):
    return type(value) in (float, int) and math.isfinite(value) and value >= minimum


def private_path(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
            "Private evidence requires a relative path.")
    path = (Path(root) / relative).resolve(strict=True)
    require(path.is_relative_to(Path(root).resolve()) and path.is_file(), "Private evidence leaves its configured root.")
    return path


def _json_float(value):
    number = float(value)
    require(math.isfinite(number), "Nonfinite JSON evidence.")
    return number


def _read_json(path, limit, message):
    require(path.stat().st_size <= limit, message)
    with path.open("rb") as stream:
        require(os.fstat(stream.fileno()).st_size <= limit, message)
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, message)
    return json.loads(raw.decode("utf-8"), parse_float=_json_float,
                      parse_constant=lambda value: (_ for _ in ()).throw(Failure("Nonfinite JSON evidence.")))


def read_json(path):
    return _read_json(path, JSON_EVIDENCE_LIMIT, "Private JSON exceeds the evidence limit.")


def read_full_graph(path):
    """Read complete graph evidence within the separate graph byte limit."""
    return _read_json(path, FULL_GRAPH_JSON_LIMIT, "Full graph JSON exceeds the evidence limit.")


def descriptor(root, path):
    path = Path(path).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Evidence leaves its configured root.")
    return {"file": str(path.relative_to(Path(root).resolve())), "sha256": file_hash(path)}


def verify_descriptor(root, record):
    require(isinstance(record, dict), "Evidence descriptor is absent.")
    path = private_path(root, record.get("file"))
    require(file_hash(path) == record.get("sha256"), "Private evidence hash differs.")
    return path


def _native_tensor(storage, offset, shape, stride, requires_grad, hooks):
    """Retain tensor metadata without tensor allocation or model imports."""
    require(offset == 0 and requires_grad is False and not hooks, "Unsupported native tensor view.")
    return storage, shape, stride


class _NativeMetadata(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {("torch", "BFloat16Storage"): "bfloat16",
                   ("torch._utils", "_rebuild_tensor_v2"): _native_tensor,
                   ("collections", "OrderedDict"): OrderedDict}
        require((module, name) in allowed, "Unsupported native tensor metadata global.")
        return allowed[module, name]

    def persistent_load(self, value):
        require(isinstance(value, tuple) and len(value) == 5 and value[:2] == ("storage", "bfloat16")
                and isinstance(value[2], str) and re.fullmatch(r"[0-9]+", value[2])
                and value[3] == "cpu" and type(value[4]) is int and value[4] > 0,
                "Unsupported native tensor storage.")
        return value[2], value[4]


def native_arrays(root, record, shapes):
    """Read only bounded, dense BF16 records from the private native exporter."""
    try:
        path = verify_descriptor(root, record)
        require(record.get("tensors") == {name: {"dtype": "torch.bfloat16", "shape": shape}
                                           for name, shape in shapes.items()}, "Native tensor declarations differ.")
        storage_bytes = sum(math.prod(shape) * 2 for shape in shapes.values())
        require(22 <= path.stat().st_size <= storage_bytes + 65536, "Native archive exceeds its evidence limit.")
        # Bound the directory before ZipFile creates its entry table.
        with path.open("rb") as stream:
            stream.seek(-22, 2)
            signature, disk, directory_disk, disk_count, count, length, _, comment = struct.unpack(
                "<4s4H2LH", stream.read(22))
        require(signature == b"PK\x05\x06" and disk == directory_disk == comment == 0
                and disk_count == count and 1 <= count <= 16 and length <= 8192,
                "Unsupported native archive directory.")
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [item.filename for item in entries]
            require(len(entries) == count and len(set(names)) == count, "Duplicate native archive entries.")
            prefix = names[0].split("/")[0] + "/"
            require(all(name.startswith(prefix) for name in names), "Mixed native archive roots.")
            members = {item.filename[len(prefix):]: item for item in entries}
            auxiliary = {"data.pkl", "byteorder", "version", ".data/serialization_id",
                         ".format_version", ".storage_alignment"}
            require({"data.pkl", "byteorder", "version"} <= members.keys()
                    and all(name in auxiliary or re.fullmatch(r"data/[0-9]+", name) for name in members),
                    "Unsupported native archive members.")
            require(all(item.compress_type == zipfile.ZIP_STORED and not item.flag_bits & 1
                        and item.file_size <= (storage_bytes if name.startswith("data/") else 16384)
                        for name, item in members.items()), "Native archive member exceeds its evidence limit.")
            require(archive.read(members["byteorder"]) == b"little"
                    and archive.read(members["version"]) == b"3\n", "Unsupported native archive format.")
            raw = archive.read(members["data.pkl"])
            operations = {"PROTO", "EMPTY_DICT", "EMPTY_TUPLE", "MARK", "GLOBAL", "BINPUT", "BINGET",
                          "BINUNICODE", "BININT", "BININT1", "BININT2", "TUPLE", "TUPLE1", "TUPLE2", "TUPLE3",
                          "BINPERSID", "NEWFALSE", "REDUCE", "SETITEM", "SETITEMS", "STOP"}
            require(raw.startswith(b"\x80\x02") and all(op.name in operations for op, _, _ in pickletools.genops(raw)),
                    "Unsupported native metadata encoding.")
            metadata = _NativeMetadata(io.BytesIO(raw)).load()
            require(type(metadata) is dict and metadata.keys() == shapes.keys(), "Native tensor coverage differs.")
            arrays, used = {}, set()
            for name, shape in shapes.items():
                (key, size), actual_shape, stride = metadata[name]
                expected_stride = tuple(math.prod(shape[i + 1:]) for i in range(len(shape)))
                require(actual_shape == tuple(shape) and stride == expected_stride and size == math.prod(shape),
                        "Native tensor shape or storage differs.")
                member = members["data/" + key]
                require(member.file_size == size * 2, "Native storage byte count differs.")
                bits = np.frombuffer(archive.read(member), dtype="<u2").astype(np.uint32) << 16
                arrays[name] = bits.view(np.float32).reshape(shape)
                require(np.isfinite(arrays[name]).all(), "Native tensor contains nonfinite values.")
                used.add("data/" + key)
            require({name for name in members if name.startswith("data/")} == used,
                    "Native archive contains unbound storage.")
            return arrays
    except (KeyError, TypeError, ValueError, OSError, IndexError, EOFError, pickle.UnpicklingError,
            zipfile.BadZipFile, struct.error, RecursionError) as error:
        raise Failure(f"Invalid native evidence: {error}") from error


def native_array(root, record, name, shape):
    """Validate a flat observation descriptor with the native BF16 reader."""
    require(isinstance(record, dict) and set(record) == {"file", "sha256", "tensor", "dtype", "shape"}
            and record.get("tensor") == name, "Native observation declaration differs.")
    return native_arrays(root, {"file": record["file"], "sha256": record["sha256"],
                               "tensors": {name: {"dtype": record["dtype"], "shape": record["shape"]}}},
                         {name: shape})[name]


def replication_state(status):
    state = inference_state(status, PROMPT)
    replication = state.get("replication")
    require(state.get("state") == "ready" and isinstance(replication, dict)
            and replication.get("protocol") == PROFILE and replication.get("prepared") is True,
            "The explicit published runtime is absent or unprepared.")
    expected = [{"cell": cell, "action": action_name(cell), "label": case["label"], "operations": case["operations"]}
                for cell, case in CASES.items()]
    require(digest(replication.get("cases")) == digest(expected) and digest(replication.get("groups")) == digest(SOURCE["groups"]),
            "The runtime case catalog differs from the published source.")
    return state, replication


def preparation(root, replication):
    record = read_json(private_path(root, replication.get("preparation_file")))
    identity = {k: v for k, v in record.items() if k not in ("id", "seconds")}
    require(record.get("id") == replication.get("preparation_id") == digest(identity), "Preparation identity differs.")
    require(record.get("protocol") == PROFILE and record.get("source_revision") == SOURCE["source_revision"]
            and record.get("producer") == "caliper" and record.get("dtype") == "torch.bfloat16"
            and record.get("softmax_dtype") == "torch.bfloat16"
            and record.get("freeze_attention") is True and record.get("freeze_norms") is False
            and record.get("freeze_mlps") is False
            and record.get("decoder_accumulation") == "sorted_position_feature_index_add",
            "Preparation does not use the published execution settings.")
    require(finite(record.get("seconds")) and record.get("model_settings")
            and all(isinstance(record.get(key), str) and re.fullmatch(r"sha256:[0-9a-f]{64}", record[key])
                    for key in ("dictionary_identity", "model_content_hash")),
            "Preparation lacks current resource identity.")
    require(set(record.get("prompts", {})) == set(SOURCE["prompts"]), "Preparation prompt coverage differs.")
    for source, prompt in SOURCE["prompts"].items():
        item = record["prompts"][source]
        ids = item.get("token_ids")
        require(item.get("text") == prompt and item.get("sha256") == text_hash(prompt)
                and isinstance(ids, list) and len(ids) > 10 and all(type(i) is int and i >= 0 for i in ids),
                "Preparation prompt bytes or token identities differ.")
    return record


def validate_case(root, status, cell, *, previous_sequence, preparation_id=None):
    """Authorize the next serialized case through its current committed status."""
    return _validate_case(root, status, cell, previous_sequence=previous_sequence,
                          preparation_id=preparation_id, consecutive=True)


def validate_live_case(root, status, cell, *, previous_sequence, preparation_id):
    """Authorize a newer live case after earlier attempts become superseded."""
    require(preparation_id is not None, "The live case has no established preparation identity.")
    return _validate_case(root, status, cell, previous_sequence=previous_sequence,
                          preparation_id=preparation_id, consecutive=False)


def _validate_case(root, status, cell, *, previous_sequence, preparation_id, consecutive):
    """Authorize private arrays only through the current committed status."""
    try:
        state, replication = replication_state(status)
        prepared = preparation(root, replication)
        require(preparation_id is None or prepared["id"] == preparation_id, "The action changed preparation identity.")
        record = replication.get("result")
        require(isinstance(record, dict), "The published action has no committed result.")
        sequence = record.get("sequence")
        require(type(sequence) is int and sequence > previous_sequence
                and (not consecutive or sequence == previous_sequence + 1),
                "The published action result is stale or has a missing sequence.")
        require(replication.get("active_cell") == cell and replication.get("last_attempt") == {
            "case_id": action_name(cell), "sequence": sequence, "status": "ok"}, "The published action failed or is no longer active.")
        require(record.get("case_id") == action_name(cell) and record.get("cell") == cell
                and record.get("label") == CASES[cell]["label"] and record.get("producer") == "caliper"
                and record.get("preparation_id") == prepared["id"], "Committed case identity differs.")
        require(all(isinstance(record.get(key), list) and len(record[key]) == 2
                    and all(type(token) is int and token >= 0 for token in record[key])
                    for key in ("argmax_token_ids", "top_token_ids")), "Case token identities are invalid.")
        comparison = state.get("comparison")
        require(bound_completion(comparison, state) and comparison.get("case_id") == action_name(cell)
                and comparison.get("argmax_token_ids") == record.get("argmax_token_ids"),
                "The action has no current output publication.")
        require(finite(record.get("action_seconds")) and finite(record.get("compute_export_seconds"))
                and record["action_seconds"] >= record["compute_export_seconds"], "Full action duration is absent.")
        require(record.get("logits_dtype") == record.get("softmax_dtype") == prepared["dtype"], "Case dtype differs.")
        expected = [(operation, node) for operation in CASES[cell]["operations"]
                    for node in SOURCE["groups"][operation["group"]]]
        values = record.get("group_values")
        require(isinstance(values, list) and len(values) == len(expected), "Published group members are incomplete.")
        for member, (operation, node) in zip(values, expected):
            require(all(member.get(k) == v for k, v in {**operation, **node}.items())
                    and member.get("canonical_id") == f"feature:{node['layer']}:{node['feature']}:{node['position']}"
                    and finite(member.get("activation"), -math.inf) and finite(member.get("value"), -math.inf)
                    and member["value"] == member["activation"] * operation["factor"], "Published clamp values differ.")
        metadata = read_json(private_path(root, record.get("metadata_file")))
        require(metadata == {k: v for k, v in record.items() if k not in ("metadata_file", "action_seconds")},
                "Private metadata differs from the committed result.")
        evidence = record.get("evidence")
        archive = verify_descriptor(root, evidence)
        require(evidence.get("rows") == ["baseline", "case"] and evidence.get("storage_dtype") == "float32"
                and evidence.get("native_values_preserved") is True, "Private array format differs.")
        shape = evidence.get("shape")
        require(isinstance(shape, list) and len(shape) == 2 and shape[0] == 2 and type(shape[1]) is int
                and 5 <= shape[1] <= 1_000_000 and archive.stat().st_size <= 32 * 1024 * 1024, "Private array shape is invalid.")
        with np.load(archive, allow_pickle=False) as data:
            require(set(data.files) == {"logits", "probabilities"}, "Private vocabulary arrays are incomplete.")
            logits, probabilities = data["logits"], data["probabilities"]
        require(all(a.shape == tuple(shape) and a.dtype == np.float32 and np.isfinite(a).all()
                    for a in (logits, probabilities)), "Private arrays have invalid values or shape.")
        native = native_arrays(root, record.get("native_evidence"), {"logits": shape, "probabilities": shape})
        require(all(native[name].tobytes() == value.tobytes() for name, value in
                    (("logits", logits), ("probabilities", probabilities))), "Native bytes differ from private arrays.")
        if cell == 29:
            observation = record.get("observations", {}).get("Say Victoria", {})
            require(observation.get("node") == SOURCE["groups"]["Say Victoria"][0],
                    "The Say Victoria observation identity differs.")
            observed = native_array(root, observation.get("native_evidence"), "activation", [1])
            require(observed[0] == 0, "The Say Victoria activation is not zero.")
        require(((probabilities >= 0) & (probabilities <= 1)).all()
                and np.allclose(probabilities.sum(axis=1), 1, atol=.01, rtol=0), "Private probabilities are invalid.")
        require(record["argmax_token_ids"] == probabilities.argmax(axis=1).tolist(), "Private argmax identity differs.")
        rows = record.get("top_five")
        require(isinstance(rows, list) and len(rows) == 2, "Actual before and after outputs are absent.")
        for row, items in enumerate(rows):
            require(len(items) == 5 and len({item["id"] for item in items}) == 5, "Case top five is incomplete.")
            for item in items:
                token = item["id"]
                require(type(token) is int and 0 <= token < shape[1] and isinstance(item.get("text"), str)
                        and item.get("probability") == float(probabilities[row, token]), "Case top output differs from private arrays.")
            require([item["probability"] for item in items] == np.sort(np.partition(probabilities[row], -5)[-5:])[::-1].tolist()
                    and items[0]["id"] == record["top_token_ids"][row], "Case top token identity differs.")
        require(record.get("baseline_top_probability") == float(probabilities[0].max())
                and comparison.get("baseline_top_probability") == record["baseline_top_probability"]
                and comparison.get("baseline_top_text") == record.get("baseline_top_text"), "Baseline publication differs.")
        if cell == 10:
            require(np.array_equal(logits[0], logits[1]) and np.array_equal(probabilities[0], probabilities[1]),
                    "The fresh baseline does not restore the original output.")
        return {"status": status, "preparation": prepared, "result": record,
                "baseline_array_sha256": hashlib.sha256(logits[0].tobytes() + probabilities[0].tobytes()).hexdigest(),
                "metadata": descriptor(root, private_path(root, record["metadata_file"])),
                "preparation_file": descriptor(root, private_path(root, replication["preparation_file"]))}
    except (KeyError, TypeError, ValueError, OSError, IndexError) as error:
        raise Failure(f"Invalid private case evidence: {error}") from error


def published_selection(graph):
    """Select the first surviving published node for examples only."""
    for group in SOURCE["groups"].values():
        for member in group:
            for node in graph["nodes"]:
                if node.get("kind") == "feature" and all(node.get(k) == v for k, v in member.items()):
                    return node
    raise Failure("The complete graph has no published feature for examples.")


def read_graph(session, trace, *, token_ids=None):
    """Read every MCP page and refuse incomplete or mixed trace identities."""
    require(trace.get("state") == "done" and trace.get("status") == "done" and trace.get("trace_id"),
            "The published trace did not complete.")
    require(all(type(trace.get("settings", {}).get(k)) is type(v) and trace["settings"][k] == v
                for k, v in TRACE_REQUEST.items()), "The trace did not use the exact reference settings.")
    require(trace["settings"].get("dtype") == "torch.bfloat16"
            and trace["settings"].get("objective") == "centered_pre_softcap_logit"
            and trace["settings"].get("source_projection") == "selected_features_without_bos_features_or_errors"
            and trace.get("completeness", {}).get("source_scope") == "full_sources_before_reference_projection",
            "The trace lacks the published objective and separate full-source diagnostics.")
    pages, graph = [], None
    for kind, ceiling in (("node", TRACE_REQUEST["max_nodes"]), ("edge", TRACE_REQUEST["max_edges"])):
        target, offset = kind + "s", 0
        total = trace["counts"][target]
        require(type(total) is int and 0 <= total <= ceiling, "The complete graph exceeds its resource ceiling.")
        while offset < total or offset == 0:
            page = session.events.complete("trace.read", **{kind + "_offset": offset, kind + "_limit": 256,
                         ("edge" if kind == "node" else "node") + "_limit": 0})
            pages.append(page)
            require(page.get("trace_id") == trace["trace_id"] and page.get("status") == "done"
                    and all(page.get(key) == trace.get(key) for key in ("counts", "settings", "completeness")),
                    "The trace changed during graph pagination.")
            if graph is None:
                graph = {**page, "nodes": [], "edges": []}
            require(page.get("tokens") == graph.get("tokens"), "The graph token identities changed during pagination.")
            rows = page[target]
            require(len(rows) == min(256, total - offset), "The graph page is incomplete.")
            graph[target].extend(rows)
            offset += len(rows)
            expected = offset if offset < total else None
            require(page.get("next_" + kind + "_offset") == expected, "The graph pagination offset is incomplete or stale.")
            if expected is None:
                break
    graph.update(next_node_offset=None, next_edge_offset=None)
    tokens = graph.get("tokens", [])
    require(len(tokens) == trace["counts"]["tokens"] and [item.get("position") for item in tokens] == list(range(len(tokens)))
            and all(type(item.get("token_id")) is int and item["token_id"] >= 0 for item in tokens),
            "The graph token identities are incomplete.")
    if token_ids is not None:
        require([item["token_id"] for item in tokens] == token_ids, "The graph belongs to another prompt token sequence.")
    indices = [node["index"] for node in graph["nodes"]]
    require(indices == list(range(len(indices))), "The complete graph has duplicate or absent node identities.")
    require(all(type(edge[key]) is int and edge[key] in range(len(indices)) for edge in graph["edges"]
                for key in ("source", "target")), "The complete graph has invalid edge endpoints.")
    pairs = [(edge["source"], edge["target"]) for edge in graph["edges"]]
    require(len(pairs) == len(set(pairs)) and all(finite(edge.get("weight"), -math.inf) for edge in graph["edges"]),
            "The complete graph has duplicate or nonfinite edges.")
    return graph, pages
