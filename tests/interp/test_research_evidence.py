"""Research qualification rejects stale cursors and unadmitted metadata reads."""
from copy import deepcopy
import hashlib
import json

import pytest

from caliper.interp._memory import CleanStop
from caliper.interp._research_evidence import research_evidence
from .test_memory_phases import process_guard


@pytest.fixture
def evidence(tmp_path):
    folder = tmp_path / "evaluation-research-v1"
    folder.mkdir()
    fingerprint = "a" * 64
    tensors, model = {"tensor_hash": "b" * 64}, {"tensor_hash": "c" * 64}
    rows = [[0, 0, 3], [1, 0, 2]]
    digest = hashlib.sha256(b"[0,0,3]\n[1,0,2]\n").hexdigest()
    cursor = {"version": 2, "identity": fingerprint, "tensor_binding": tensors, "model_binding": model,
              "settings": {"protocol": "research_v1"}, "source_tokens": 5, "count": 4, "ce_tokens": 4,
              "content_segments": rows}
    report = {"protocol": "research_v1", "protocol_version": 1, "dictionary_hash": fingerprint,
              "tensor_binding": tensors, "model_binding": model, "settings": cursor["settings"],
              "source_tokens": 5, "metric_observations": 4, "cross_entropy_tokens": 4,
              "context_policy": {"content_context": 1023},
              "content_segmentation": {"contexts": 2, "sha256": digest,
                "encoding": "sha256_ascii_json_rows_with_newlines",
                "fields": ["document_id", "source_position", "content_length"],
                "record": "evaluation-research-v1/state.json:content_segments"}}

    def save():
        raw = json.dumps(cursor).encode()
        (folder / "state.json").write_bytes(raw)
        report["checkpoint_sha256"] = hashlib.sha256(raw).hexdigest()
        (folder / "quality.json").write_text(json.dumps(report))

    save()
    guard = process_guard(1_000_000_000, {"current": 100, "peak": 100})
    return tmp_path, fingerprint, tensors, model, report, cursor, save, guard


def test_research_report_binds_current_resources_and_cursor_without_mutation(evidence):
    root, identity, tensors, model, report, _, _, guard = evidence
    before = deepcopy(report)
    result = research_evidence(root, identity, tensors, model, memory=guard)
    assert result == before and report == before
    result["context_policy"]["content_context"] = 1
    assert research_evidence(root, identity, tensors, model, memory=guard) == before


@pytest.mark.parametrize("fault", ["cursor_bytes", "count", "spans", "settings", "dictionary", "model", "tensor"])
def test_research_evidence_rejects_mismatched_records(evidence, fault):
    root, identity, tensors, model, report, cursor, save, guard = evidence
    if fault == "cursor_bytes":
        (root / "evaluation-research-v1/state.json").write_text("{}")
    elif fault == "dictionary":
        identity = "d" * 64
    elif fault in ("model", "tensor"):
        if fault == "model":
            model = {"tensor_hash": "d" * 64}
        else:
            tensors = {"tensor_hash": "d" * 64}
    else:
        if fault == "count":
            cursor["ce_tokens"] = 1
        elif fault == "spans":
            cursor["content_segments"][0][1] = 1
        else:
            cursor["settings"] = {"protocol": "different"}
        save()
    with pytest.raises(ValueError, match="research qualification"):
        research_evidence(root, identity, tensors, model, memory=guard)


def test_research_evidence_admits_metadata_before_read(evidence, monkeypatch):
    root, identity, tensors, model, _, _, _, _ = evidence
    guard = process_guard(1000, {"current": 100, "peak": 100})
    monkeypatch.setattr(type(root), "open", lambda *args, **kwargs: pytest.fail("unadmitted metadata read"))
    with pytest.raises(CleanStop, match="research qualification"):
        research_evidence(root, identity, tensors, model, memory=guard)


def test_missing_research_report_preserves_legacy_dictionary_queries(tmp_path):
    guard = process_guard(1000, {"current": 100, "peak": 100})
    assert research_evidence(tmp_path, "a" * 64, {}, {}, memory=guard) is None
