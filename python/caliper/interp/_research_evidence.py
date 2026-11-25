"""Bounded research reports tied to their cursor and current model resources."""
import hashlib
import json
import os

from ._memory import CleanStop, allocation_bytes


def _stamp(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _read(path, limit, memory):
    info = path.stat()
    if not 0 < info.st_size <= limit:
        raise ValueError("research evidence exceeds its file bound")
    memory.check("research qualification admission")
    if allocation_bytes(memory) + 32 * info.st_size + 8 * 1024 * 1024 > memory.limit:
        raise CleanStop("research qualification exceeds the process memory limit")
    with path.open("rb") as source:
        if _stamp(os.fstat(source.fileno())) != _stamp(info):
            raise ValueError("research evidence changed before its read")
        raw = source.read(info.st_size + 1)
        if len(raw) != info.st_size or _stamp(os.fstat(source.fileno())) != _stamp(info):
            raise ValueError("research evidence changed during its read")
    if _stamp(path.stat()) != _stamp(info):
        raise ValueError("research evidence changed after its read")

    def nonfinite(value):
        raise ValueError("research evidence contains nonfinite JSON")

    value = json.loads(raw, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError("research evidence must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def research_evidence(root, fingerprint, tensor_binding, model_binding, *, memory):
    """Return optional research evidence after cursor and live-resource checks."""
    folder = root / "evaluation-research-v1"
    if not (folder / "quality.json").exists():
        return None
    report, _ = _read(folder / "quality.json", 1024 * 1024, memory)
    cursor, digest = _read(folder / "state.json", 16 * 1024 * 1024, memory)
    if (report.get("checkpoint_sha256") != digest or cursor.get("version") != 2
            or report.get("dictionary_hash") != fingerprint or cursor.get("identity") != fingerprint
            or any(value.get("tensor_binding") != tensor_binding or value.get("model_binding") != model_binding
                   for value in (report, cursor))):
        raise ValueError("research qualification identity differs from the cursor or current resources")
    if (report.get("protocol") != "research_v1" or type(report.get("protocol_version")) is not int
            or report["protocol_version"] != 1 or cursor.get("settings") != report.get("settings")):
        raise ValueError("research qualification protocol differs from its cursor")
    for key, report_key in (("source_tokens", "source_tokens"), ("count", "metric_observations"),
                            ("ce_tokens", "cross_entropy_tokens")):
        if type(cursor.get(key)) is not int or cursor[key] != report.get(report_key):
            raise ValueError("research qualification counts differ from its cursor")
    policy = report.get("context_policy", {})
    capacity = policy.get("content_context")
    rows = cursor.get("content_segments")
    if (type(capacity) is not int or not 1 <= capacity <= 1024 or not isinstance(rows, list)
            or len(rows) > 200_000 or any(not isinstance(row, list) or len(row) != 3
                or any(type(value) is not int or value < 0 for value in row)
                or not 1 <= row[2] <= capacity for row in rows)):
        raise ValueError("research qualification source segmentation is invalid")
    spans = hashlib.sha256()
    for row in rows:
        spans.update((json.dumps(row, separators=(",", ":")) + "\n").encode("ascii"))
    expected = {"contexts": len(rows), "sha256": spans.hexdigest(),
                "encoding": "sha256_ascii_json_rows_with_newlines",
                "fields": ["document_id", "source_position", "content_length"],
                "record": "evaluation-research-v1/state.json:content_segments"}
    if (report.get("content_segmentation") != expected
            or sum(row[2] for row in rows) != report.get("source_tokens")):
        raise ValueError("research qualification source spans differ from its cursor")
    memory.check("research qualification complete")
    return report
