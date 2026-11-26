"""CPU-only command stops and source ownership without model execution."""
import json
from types import SimpleNamespace

import pytest

from caliper._interpret_protocol import bounded
from tools.interp import build_index as indexing, eval_dictionary as evaluation
from tools.interp.memory_limits import CleanStop
from tools.interp.stop_records import clean_stop_record, stop_reason
from tools.interp.text_stream import DATASET_CONFIG, DATASET_ID, DATASET_REVISION, partition_metadata


REASONS = ["normal stop", "nul\0removed", "invalid\ud800unicode", "😀" * 3000,
           "\0\udfff" + "é" * 5000]


def index_metadata(folder):
    metadata = {"model_id": "tiny-lm", "layers": 2, "latents": 8, "training": {
        "dataset_id": DATASET_ID, "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION, "split": "train",
        "partitions": partition_metadata(), "source_id": "fixture-source",
    }}
    (folder / "dictionary.json").write_text(json.dumps(metadata))


@pytest.mark.parametrize("reason", REASONS)
@pytest.mark.parametrize("command", [indexing, evaluation])
def test_command_stops_have_one_bounded_record(tmp_path, monkeypatch, capsys, command, reason):
    index_metadata(tmp_path)
    before = (tmp_path / "dictionary.json").read_bytes()

    def refuse(*args, **kwargs):
        raise CleanStop(reason)

    monkeypatch.setattr(command, "MemoryGuard", refuse)
    monkeypatch.setattr(command, "load_text_model", lambda **k: pytest.fail("model load ran"))
    assert command.main(["--dictionary", str(tmp_path), "--device", "cpu"]) == 2
    captured = capsys.readouterr()
    assert not captured.err
    result = json.loads(captured.out)
    assert result == clean_stop_record(CleanStop(reason))
    assert result["stage"] == "clean_stop" and result["status"] == "interrupted"
    assert "\0" not in result["reason"]
    assert len(result["reason"].encode("utf-8")) <= 1024
    assert bounded({"result": result})
    assert (tmp_path / "dictionary.json").read_bytes() == before


def test_stop_reason_keeps_valid_text_and_complete_codepoints():
    assert stop_reason("abc\0def\ud800ghi") == "abcdef?ghi"
    assert stop_reason("a" * 1023 + "é") == "a" * 1023
    assert stop_reason("😀" * 257) == "😀" * 256


@pytest.mark.parametrize("source_kind", ["local", "dataset", "jsonl"])
@pytest.mark.parametrize("exit_at", ["source_id", "stream", "provenance", "build", "stop", "success"])
def test_index_source_closes_on_every_exit(tmp_path, monkeypatch, capsys, source_kind, exit_at):
    index_metadata(tmp_path)
    events = []

    class Source:
        @property
        def source_id(self):
            if exit_at == "source_id":
                raise ValueError("synthetic source identity failure")
            return "fixture-source"

        def close(self):
            events.append("source closed")

    class Stream:
        def __init__(self, source, tokenizer, partition, **kwargs):
            if exit_at == "stream":
                raise ValueError("synthetic stream failure")
            self.source_id = source.source_id
            if exit_at == "provenance":
                self.source_id = "other-source"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            events.append("stream closed")

    def build(*args, **kwargs):
        if exit_at == "build":
            raise ValueError("synthetic build failure")
        if exit_at == "stop":
            raise CleanStop("synthetic\0stop\ud800" + "é" * 5000)
        return {"tokens": 0, "documents": 0, "index_hash": "fixture"}

    monkeypatch.setattr(indexing, "MemoryGuard", lambda *a: SimpleNamespace(peak=0, check=lambda *a, **k: None))
    monkeypatch.setattr(indexing, "load_text_model", lambda **k: (None, None))
    monkeypatch.setattr(indexing, "load_dictionary", lambda *a, **k: None)
    monkeypatch.setattr(indexing, "fineweb_documents", lambda **k: Source())
    monkeypatch.setattr(indexing, "jsonl_documents", lambda *a: Source())
    monkeypatch.setattr(indexing, "_hash_file", lambda *a: "fixture")
    monkeypatch.setattr(indexing, "TextStream", Stream)
    monkeypatch.setattr(indexing, "build_index", build)
    argv = ["--dictionary", str(tmp_path), "--device", "cpu"]
    if source_kind == "local":
        argv += ["--local-dataset-manifest", "fixture.json"]
    elif source_kind == "jsonl":
        argv += ["--text-jsonl", "fixture.jsonl"]
    if exit_at in ("stop", "success"):
        assert indexing.main(argv) == (2 if exit_at == "stop" else 0)
        result = json.loads(capsys.readouterr().out)
        assert bounded({"result": result})
        assert result["stage"] == ("clean_stop" if exit_at == "stop" else "index_ready")
    else:
        with pytest.raises(ValueError):
            indexing.main(argv)
    assert events[-1] == "source closed"
    assert events.count("source closed") == 1
    if exit_at not in ("source_id", "stream"):
        assert events == ["stream closed", "source closed"]


@pytest.mark.parametrize("exit_at", ["stream", "operation", "success"])
def test_evaluator_closes_source_after_regular_errors(tmp_path, monkeypatch, exit_at):
    command = evaluation
    events = []

    class Source:
        def close(self):
            events.append("source closed")

    class Stream:
        source_id = "fixture-source"

        def __init__(self, *args, **kwargs):
            if exit_at == "stream":
                raise ValueError("synthetic stream error")

        def __enter__(self):
            return self

        def __exit__(self, *_):
            events.append("stream closed")

    def operation(*args, **kwargs):
        if exit_at == "operation":
            raise ValueError("synthetic operation error")
        return {"status": "passed"}

    def load(*, device, memory, mode="inference"):
        assert device == "cpu" and memory is not None
        assert mode == "inference"
        return None, None

    monkeypatch.setattr(command, "MemoryGuard", lambda *a: SimpleNamespace(check=lambda *a, **k: None))
    monkeypatch.setattr(command, "load_text_model", load)
    monkeypatch.setattr(command, "fineweb_documents", lambda **k: Source())
    monkeypatch.setattr(command, "TextStream", Stream)
    monkeypatch.setattr(command, "load_dictionary", lambda *a, **k: None)
    monkeypatch.setattr(command, "evaluate", operation)
    argv = ["--dictionary", str(tmp_path)]
    if exit_at == "success":
        assert command.main([*argv, "--device", "cpu"]) == 0
    else:
        with pytest.raises(ValueError, match="synthetic"):
            command.main([*argv, "--device", "cpu"])
    assert events == (["source closed"] if exit_at == "stream" else ["stream closed", "source closed"])
