"""Small metadata and resource limits for Parquet decode admission."""
from contextlib import ExitStack
import json
from types import SimpleNamespace

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from tools.interp import build_index as indexing, eval_dictionary as evaluation
from tools.interp import text_stream
from tools.interp.memory_limits import CleanStop, MemoryGuard
from .fixture import TinyTokenizer
from .test_training_stream import documents, local_manifest


def metadata(*, rows=16, compressed=16, uncompressed=64, **overrides):
    chunk = SimpleNamespace(num_values=rows, total_compressed_size=compressed,
                            total_uncompressed_size=uncompressed, encodings=("PLAIN", "RLE", "RLE_DICTIONARY"),
                            path_in_schema="text", physical_type="BYTE_ARRAY", file_path="", compression="SNAPPY")
    field = SimpleNamespace(path="text", physical_type="BYTE_ARRAY", max_repetition_level=0, max_definition_level=1)
    for key, value in overrides.items():
        setattr(chunk if hasattr(chunk, key) else field, key, value)
    group = SimpleNamespace(num_rows=rows, column=lambda _: chunk)
    return SimpleNamespace(schema_arrow=arrow.schema([("text", arrow.string())]),
                           schema=SimpleNamespace(names=["text"], column=lambda _: field),
                           num_row_groups=1, metadata=SimpleNamespace(num_rows=rows, row_group=lambda _: group))


@pytest.fixture
def resources(monkeypatch):
    state = SimpleNamespace(rss=0, footprint=0, driver=0, available=16_000_000_000, total=64_000_000_000)
    guard = MemoryGuard(1, "cpu", resident=lambda: state.rss, footprint=lambda: state.footprint,
                        driver=lambda: state.driver, peak=lambda: 0, pressure=lambda: 80, clock=lambda: 0)
    monkeypatch.setattr("psutil.virtual_memory", lambda: state)
    return guard, state


def test_bound_includes_dictionary_arrow_and_python_copies(monkeypatch):
    monkeypatch.setattr(text_stream, "_PARQUET_RESERVE_BYTES", 128)
    assert text_stream._parquet_units(metadata(), 16) == [(16, 2 * 16 + 20 * 64 + 64 * 17 + 512 + 128)]
    # Repeated dictionary values can exceed the encoded byte count after decode.
    assert text_stream._parquet_units(metadata(rows=1000), 16)[0][1] > 1000 * 64


@pytest.mark.parametrize("changes", [
    {"uncompressed": 0}, {"uncompressed": -1}, {"uncompressed": 2**31},
    {"compressed": 0}, {"compressed": 101}, {"num_values": 15},
    {"rows": -1}, {"compression": "ZSTD"}, {"encodings": ("DELTA_BYTE_ARRAY",)},
    {"encodings": ("PLAIN", "DELTA_LENGTH_BYTE_ARRAY")}, {"file_path": "other.parquet"},
    {"max_repetition_level": 1}, {"max_definition_level": 2}, {"path": "nested.text"},
])
def test_unsupported_or_impossible_metadata_refuses(changes):
    with pytest.raises(ValueError, match="Parquet"):
        text_stream._parquet_units(metadata(**changes), 100)


@pytest.mark.parametrize("kind", [arrow.large_string(), arrow.binary(), arrow.list_(arrow.string())])
def test_nonflat_string_schemas_refuse(kind):
    source = metadata()
    source.schema_arrow = arrow.schema([("text", kind)])
    with pytest.raises(ValueError, match="flat string"):
        text_stream._parquet_units(source, 100)


@pytest.mark.parametrize("reader", ["rss", "footprint", "driver", "fallback", "host"])
def test_fresh_headroom_refuses_despite_cached_health(resources, reader):
    guard, state = resources
    text_stream._admit_parquet_decode(guard, 100)
    assert guard.last == 0
    if reader == "fallback":
        state.footprint = None
        state.rss, state.driver = guard.limit // 2, guard.limit // 2 - 99
    elif reader == "host":
        state.available = 8_000_000_000 + 99
    else:
        setattr(state, reader, guard.limit - 99)
    with pytest.raises(CleanStop, match="headroom"):
        text_stream._admit_parquet_decode(guard, 100)
    assert guard.last == 0


def test_current_headroom_and_historical_limit_are_separate(resources):
    guard, _ = resources
    text_stream._admit_parquet_decode(guard, 100)
    guard.peak = guard.limit - 1
    text_stream._admit_parquet_decode(guard, 100)
    guard.peak = guard.limit + 1
    with pytest.raises(CleanStop, match="memory limit"):
        text_stream._admit_parquet_decode(guard, 100)


def observed_reader(monkeypatch):
    original = parquet.ParquetFile
    events = []

    class Observed:
        def __init__(self, *args, **kwargs):
            assert kwargs["pre_buffer"] is False
            self.inner = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def iter_batches(self, **kwargs):
            events.append(("decode", kwargs))
            return self.inner.iter_batches(**kwargs)

        def close(self):
            events.append("close")
            self.inner.close()

    monkeypatch.setattr(parquet, "ParquetFile", Observed)
    return events


@pytest.mark.parametrize("failure", ["process", "host", "missing_guard", "unsupported"])
def test_refusal_precedes_decode_and_closes_file(tmp_path, monkeypatch, resources, failure):
    manifest = local_manifest(tmp_path)
    guard, state = resources
    events = observed_reader(monkeypatch)
    source = text_stream.LocalParquetDocuments(manifest, memory=None if failure == "missing_guard" else guard)
    cursor = source.state_dict()
    events.clear()
    if failure == "process":
        state.rss = guard.limit - 1
    elif failure == "host":
        state.available = 8_000_000_000
    elif failure == "unsupported":
        monkeypatch.setattr(text_stream, "_parquet_units", lambda *a: (_ for _ in ()).throw(ValueError("unsupported")))
    with pytest.raises((CleanStop, ValueError)):
        next(iter(source))
    assert events == ["close"]
    assert source.state_dict() == cursor


def test_impossible_cap_refuses_at_source_admission(tmp_path, monkeypatch, resources):
    manifest = local_manifest(tmp_path)
    guard, _ = resources
    guard.limit = 32
    events = observed_reader(monkeypatch)
    with pytest.raises(CleanStop, match="bound exceeds"):
        text_stream.LocalParquetDocuments(manifest, memory=guard)
    assert events == ["close"]


def test_compressed_oversize_metadata_refuses_without_data_allocation(tmp_path, monkeypatch, resources):
    manifest = local_manifest(tmp_path)
    guard, _ = resources
    source = metadata(rows=16, uncompressed=1_600_000_000)
    events = []
    source.close = lambda: events.append("close")
    source.iter_batches = lambda **k: pytest.fail("oversized decode ran")
    monkeypatch.setattr(parquet, "ParquetFile", lambda *a, **k: source)
    with pytest.raises(CleanStop, match="bound exceeds"):
        text_stream.LocalParquetDocuments(manifest, memory=guard)
    assert events == ["close"]


@pytest.mark.parametrize("limit", [12_000_000_000, 24_000_000_000, 40_000_000_000])
def test_approved_layout_fits_caps_with_sufficient_live_headroom(resources, limit):
    guard, state = resources
    # These are footer counts only. No text or Arrow data buffer is allocated.
    source = metadata(rows=1000, compressed=3_478_085, uncompressed=5_957_552)
    required = text_stream._parquet_units(source, 3_478_085)[0][1]
    assert required < 160_000_000
    guard.limit = limit
    state.footprint = limit - 160_000_000
    text_stream._admit_parquet_decode(guard, required)
    state.footprint = limit - required + 1
    with pytest.raises(CleanStop, match="headroom"):
        text_stream._admit_parquet_decode(guard, required)


def test_scalar_conversion_without_batch_python_copy(tmp_path, monkeypatch, resources):
    manifest = local_manifest(tmp_path)
    guard, _ = resources
    source = text_stream.LocalParquetDocuments(manifest, memory=guard)
    original = parquet.ParquetFile
    events = []

    class Column:
        def to_pylist(self):
            pytest.fail("full-batch Python conversion")

        def __getitem__(self, index):
            assert index == 0
            events.append("scalar")
            return SimpleNamespace(as_py=lambda: "small")

    class Reader:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def iter_batches(self, **kwargs):
            assert kwargs == {"batch_size": 1, "row_groups": [0], "columns": ["text"], "use_threads": False}
            try:
                yield SimpleNamespace(column=lambda _: Column())
            finally:
                events.append("batch close")

        def close(self):
            self.inner.close()
            events.append("file close")

    monkeypatch.setattr(parquet, "ParquetFile", Reader)
    iterator = iter(source)
    assert next(iterator) == {"text": "small"}
    iterator.close()
    assert events == ["scalar", "batch close", "file close"]


@pytest.mark.parametrize("reason", ["headroom", "signal"])
def test_decode_stop_preserves_buffer_cursor_and_exact_resume(tmp_path, monkeypatch, resources, reason):
    manifest = local_manifest(tmp_path)
    guard, state = resources
    stopped = False

    def health():
        if stopped and reason == "signal":
            raise CleanStop("synthetic signal")

    source = text_stream.LocalParquetDocuments(manifest, memory=guard, health=health)
    events = observed_reader(monkeypatch)
    with text_stream.TextStream(source, TinyTokenizer(), "train") as stream:
        stream.next_tokens(1)
        stopped = True
        if reason == "headroom":
            state.rss = guard.limit - 1
        with pytest.raises(CleanStop, match=reason):
            stream.next_tokens(100)
        saved = stream.state_dict()
        assert saved["source_state"]["row"] == saved["rows"] == 1
        assert saved["buffer"][0]
        assert events[-1] == "close"
    state.rss, guard.peak = 0, 0
    resumed_source = text_stream.LocalParquetDocuments(manifest, memory=guard)
    with text_stream.TextStream(resumed_source, TinyTokenizer(), "train", state=saved) as resumed:
        expected = text_stream.TextStream(documents(), TinyTokenizer(), "train")
        expected.next_tokens(1)
        assert resumed.next_tokens(100).input_ids.tolist() == expected.next_tokens(100).input_ids.tolist()
        expected.close()


@pytest.mark.parametrize("command", ["index", "evaluator"])
def test_commands_pass_live_memory_guard_to_source(tmp_path, monkeypatch, resources, command):
    guard, _ = resources
    seen = []

    class Source:
        source_id = "synthetic"

        def close(self):
            seen.append("closed")

    def admit(*, memory, health, **kwargs):
        assert memory is guard and callable(health)
        health()
        seen.append("admitted")
        return Source()

    args = SimpleNamespace(dataset_cache=None, local_dataset_manifest=tmp_path / "source.json",
                           text_jsonl=None, partition="index")
    if command == "evaluator":
        monkeypatch.setattr(evaluation, "MemoryGuard", lambda *a: guard)
        monkeypatch.setattr(evaluation, "load_text_model", lambda **k: (object(), None))
        monkeypatch.setattr(evaluation, "load_dictionary", lambda *a, **k: object())
        monkeypatch.setattr(evaluation, "evaluate", lambda *a, **k: {"status": "passed"})
        monkeypatch.setattr(evaluation, "fineweb_documents", admit)
        assert evaluation.main(["--dictionary", str(tmp_path), "--device", "cpu",
                                "--local-dataset-manifest", str(args.local_dataset_manifest)]) == 0
    else:
        monkeypatch.setattr(indexing, "fineweb_documents", admit)
        with ExitStack() as stack:
            indexing._index_stream(args, {"training": {"source_id": "synthetic"}}, {}, stack,
                                   memory=guard, health=lambda: None)
    assert seen == ["admitted", "closed"]


def test_index_source_admission_uses_guard_without_model_work(tmp_path, monkeypatch, resources):
    guard, _ = resources
    module = indexing
    seen = []

    def admit(*, memory, health, **kwargs):
        assert memory is guard
        health()
        seen.append("admitted")
        raise CleanStop("synthetic source stop")

    monkeypatch.setattr(module, "MemoryGuard", lambda *a: guard)
    monkeypatch.setattr(module, "fineweb_documents", admit)
    monkeypatch.setattr(module, "load_text_model", lambda **k: pytest.fail("model load ran"))
    metadata = {"training": {"dataset_id": text_stream.DATASET_ID,
                             "dataset_config": text_stream.DATASET_CONFIG,
                             "dataset_revision": text_stream.DATASET_REVISION,
                             "split": "train", "source_id": "synthetic",
                             "partitions": text_stream.partition_metadata()}}
    (tmp_path / "dictionary.json").write_text(json.dumps(metadata))
    argv = ["--dictionary", str(tmp_path)]
    argv += ["--device", "cpu", "--local-dataset-manifest", str(tmp_path / "source.json")]
    assert module.main(argv) == 2
    assert seen == ["admitted"]
