"""Bounded local source admission with synthetic stops and artifact checks."""
import copy
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from tools.interp import build_index as indexing, text_stream, train_transcoders as training
from tools.interp.memory_limits import CleanStop, MemoryGuard, StopControl
from .test_trainer_admission import checkpoint, snapshot
from .test_training_stream import local_manifest


def observe_reads(monkeypatch, *, after_read=lambda: None):
    """Use small checksum chunks and record every open checksum handle."""
    monkeypatch.setattr(text_stream, "_CHECKSUM_CHUNK_BYTES", 256)
    original = Path.open
    readers = []

    class Reader:
        def __init__(self, handle):
            self.handle = handle
            self.reads = 0

        def read(self, count=-1):
            if count == -1:
                return self.handle.read()
            assert count == 256
            data = self.handle.read(count)
            self.reads += 1
            after_read()
            return data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

    def open_file(path, mode="r", *args, **kwargs):
        handle = original(path, mode, *args, **kwargs)
        if mode == "rb" and path.suffix == ".parquet":
            reader = Reader(handle)
            readers.append(reader)
            return reader
        return handle

    monkeypatch.setattr(Path, "open", open_file)
    return readers


@pytest.mark.parametrize("boundary", ["chunk", "file"])
def test_local_checksum_stop_closes_handles_before_next_work(tmp_path, monkeypatch, boundary):
    manifest = local_manifest(tmp_path)
    readers = observe_reads(monkeypatch)

    def health():
        if readers and (readers[0].reads == 1 if boundary == "chunk" else readers[0].handle.closed):
            raise CleanStop("synthetic checksum stop")

    with pytest.raises(CleanStop, match="checksum stop"):
        text_stream.fineweb_documents(local_manifest=manifest, health=health)
    assert len(readers) == 1 and readers[0].handle.closed
    if boundary == "chunk":
        assert readers[0].reads == 1


def test_local_checksum_health_covers_each_chunk_and_preserves_identity(tmp_path, monkeypatch):
    manifest = local_manifest(tmp_path)
    expected = text_stream.LocalParquetDocuments(manifest).source_id
    readers = observe_reads(monkeypatch)
    checked_reads = []

    def health():
        checked_reads.append(sum(reader.reads for reader in readers))

    source = text_stream.LocalParquetDocuments(manifest, health=health)
    assert source.source_id == expected
    assert len(readers) == 2 and all(reader.handle.closed for reader in readers)
    assert set(checked_reads) == set(range(sum(reader.reads for reader in readers) + 1))


def health_controls(monkeypatch, command, reason):
    """Inject a signal flag, memory pressure, or clock change after one read."""
    control = StopControl()
    state = SimpleNamespace(time=0.0, pressure=50)
    monkeypatch.setattr(command, "StopControl", lambda: control)
    monkeypatch.setattr(command, "MemoryGuard", lambda limit, device: MemoryGuard(
        limit, device, interval=0, resident=lambda: 0, footprint=lambda: 0, peak=lambda: 0,
        driver=lambda: 0, pressure=lambda: state.pressure, clock=lambda: state.time))

    def after_read():
        if reason == "signal":
            control._signal(signal.SIGTERM, None)
        elif reason == "pressure":
            state.pressure = 14
        else:
            state.time = 20.0

    return state, observe_reads(monkeypatch, after_read=after_read)


@pytest.mark.parametrize("reason", ["signal", "pressure"])
def test_index_checksum_stop_preserves_dictionary_before_model(tmp_path, monkeypatch, capsys, reason):
    manifest = local_manifest(tmp_path)
    metadata = {"training": {"dataset_id": text_stream.DATASET_ID,
                            "dataset_config": text_stream.DATASET_CONFIG,
                            "dataset_revision": text_stream.DATASET_REVISION,
                            "split": "train", "partitions": text_stream.partition_metadata(),
                            "source_id": "synthetic"}, "quality": {"status": "passed"}}
    (tmp_path / "dictionary.json").write_text(json.dumps(metadata))
    before = snapshot(tmp_path)
    _, readers = health_controls(monkeypatch, indexing, reason)
    monkeypatch.setattr(indexing, "load_text_model", lambda **_: pytest.fail("model allocation ran"))
    monkeypatch.setattr(indexing, "build_index", lambda *a, **k: pytest.fail("index work ran"))
    assert indexing.main(["--dictionary", str(tmp_path), "--device", "cpu",
                          "--local-dataset-manifest", str(manifest)]) == 2
    assert len(readers) == 1 and readers[0].reads == 1 and readers[0].handle.closed
    assert snapshot(tmp_path) == before
    event = json.loads(capsys.readouterr().out)
    assert event["stage"] == "clean_stop" and reason in event["reason"]


@pytest.mark.parametrize("mode", ["all_layers", "upper_half"])
@pytest.mark.parametrize("reason", ["signal", "pressure", "deadline"])
def test_trainer_checksum_stop_preserves_prior_artifacts(tmp_path, monkeypatch, capsys, mode, reason):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    manifest = local_manifest(source)
    argv = ["--output", str(output), "--device", "cpu", "--max-wall-seconds", "180",
            "--local-dataset-manifest", str(manifest)]
    if mode == "all_layers":
        _, record, state_file = checkpoint(output)
        record["progress"]["elapsed_seconds"] = 100
        state_file.write_text(json.dumps(record))
        argv.append("--resume")
        monkeypatch.setattr(training, "load_text_model", lambda **_: (object(), None))
        monkeypatch.setattr(training, "make_manifest", lambda *a: copy.deepcopy(record["metadata"]))
        monkeypatch.setattr(training, "Trainer", lambda metadata, **k: SimpleNamespace(
            metadata=metadata, fallback_source=None))
    else:
        argv.extend(["--upper-half-from", str(source), "--prior-wall-seconds", "100"])
        monkeypatch.setattr(training, "prepare_upper_half", lambda *a, **k: pytest.fail("fallback admission ran"))
        monkeypatch.setattr(training, "load_text_model", lambda **_: pytest.fail("model allocation ran"))
    before = snapshot(tmp_path)
    state, readers = health_controls(monkeypatch, training, reason)
    monkeypatch.setattr(training.time, "monotonic", lambda: state.time)
    monkeypatch.setattr(training, "restore_checkpoint", lambda *a: pytest.fail("checkpoint restore ran"))
    monkeypatch.setattr(training, "train_online", lambda *a, **k: pytest.fail("batch loop ran"))
    assert training.main(argv) == 2
    assert len(readers) == 1 and readers[0].reads == 1 and readers[0].handle.closed
    assert snapshot(tmp_path) == before
    assert output.exists() is (mode == "all_layers")
    event = json.loads(capsys.readouterr().out)
    assert event["stage"] == "clean_stop"
    assert ("wall time reserve" if reason == "deadline" else reason) in event["reason"]


@pytest.mark.parametrize("prior,elapsed,run,save,export,expected", [
    (None, 100, None, 0, 0, 840), (200, 100, None, 0, 0, 640),
    (200, 100, 80, 0, 0, 20), (None, 100, None, 20, 25, 810),
])
def test_startup_deadline_accounts_for_each_budget(monkeypatch, prior, elapsed, run, save, export, expected):
    args = SimpleNamespace(max_wall_seconds=1000, prior_wall_seconds=prior, run_seconds=run)
    progress = {"elapsed_seconds": elapsed, "checkpoint_seconds": save, "export_seconds": export}
    monkeypatch.setattr(training.time, "monotonic", lambda: 10)
    assert training._startup_deadline(args, progress, 10) == 10 + expected
    monkeypatch.setattr(training.time, "monotonic", lambda: 10 + expected)
    with pytest.raises(ValueError, match="reserve"):
        training._startup_deadline(args, progress, 10)


@pytest.mark.parametrize("stage", ["start", "before model"])
def test_startup_health_time_cannot_consume_model_reserve(tmp_path, monkeypatch, capsys, stage):
    _, record, state_file = checkpoint(tmp_path)
    record["progress"]["elapsed_seconds"] = 100
    state_file.write_text(json.dumps(record))
    before = snapshot(tmp_path)
    clock = SimpleNamespace(time=0.0)

    def check(name, **kwargs):
        if name == stage:
            clock.time = 20.0

    monkeypatch.setattr(training.time, "monotonic", lambda: clock.time)
    monkeypatch.setattr(training, "MemoryGuard", lambda *a: SimpleNamespace(check=check))
    monkeypatch.setattr(training, "load_text_model", lambda **_: pytest.fail("model allocation ran"))
    assert training.main(["--output", str(tmp_path), "--resume", "--device", "cpu",
                          "--max-wall-seconds", "180"]) == 2
    assert snapshot(tmp_path) == before
    assert json.loads(capsys.readouterr().out)["reason"] == "wall time reserve"
