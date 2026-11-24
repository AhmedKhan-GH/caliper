"""Bounded local source admission with synthetic stops and artifact checks."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest
import torch

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


@contextmanager
def patch_capacity(count):
    """Install short sources on a small decoder at distinct probe points."""
    from caliper._controls import Controls
    from tests.interp.fixture import make_fixture

    model, tokenizer = make_fixture(layers=(count + 2) // 3, width=4, heads=1, latents=8, k=2)
    ctl = Controls()
    service = ctl.interventions_on(model)
    service.configure(tokenizer=tokenizer)
    specs = [{"kind": "patch", "point": f"layers.{index // 3}.{('resid_pre', 'mlp_in', 'mlp_out')[index % 3]}",
              "source_prompt": "a"} for index in range(count)]
    try:
        for spec in specs:
            service.set(spec)
        ctl.step()
        assert len(service.executor._cache) == count
        yield model, tokenizer, ctl, service, specs
    finally:
        service.remove()


def patch_replace(ctl, service, specs, operation):
    from tests.interp.test_intervention_replay import snapshot as intervention_snapshot

    if operation == "boundary":
        service.set(specs[0])
        return ctl.step
    if operation == "restore":
        return lambda: service.load_state_dict(intervention_snapshot(*specs), restore_active=True)
    return lambda: service.apply_overrides([specs[0]])


@pytest.mark.parametrize("operation", ["boundary", "restore", "override"])
def test_patch_replacement_counts_final_sources_at_full_capacity(monkeypatch, operation):
    from caliper.interp import _intervene

    # One real-cap case checks all 64 entries. The other paths use two entries.
    count = 64 if operation == "boundary" else 2
    monkeypatch.setattr(_intervene, "MAX_SOURCE_ENTRIES", count)
    with patch_capacity(count) as (model, tokenizer, ctl, service, specs):
        old = ctl._ivs[(specs[0]["point"], "patch")].hook
        specs[0] = dict(specs[0], source_prompt="b")
        patch_replace(ctl, service, specs, operation)()
        cache = service.executor._cache
        assert len(cache) == count
        assert ("a", specs[0]["point"]) not in cache
        assert ("b", specs[0]["point"]) in cache
        assert sum(value.numel() * value.element_size() for value in cache.values()) <= _intervene.MAX_SOURCE_BYTES
        assert old.removed and len(service.specifications()) == count
        assert service.specifications()[0]["source_prompt"] == "b"
        assert torch.isfinite(model(**tokenizer("c")).logits).all()
        assert service.executor._retiring_sources == frozenset()


@pytest.mark.parametrize("operation", ["boundary", "restore", "override"])
def test_patch_registration_failure_restores_exact_cache_and_hooks(monkeypatch, operation):
    from caliper.interp import _intervene
    from tests.interp.test_intervention_replay import hook_objects

    monkeypatch.setattr(_intervene, "MAX_SOURCE_ENTRIES", 2)
    with patch_capacity(2) as (model, tokenizer, ctl, service, specs):
        before = model(**tokenizer("c")).logits
        hooks = hook_objects(model)
        cache = service.executor._cache
        sources, identity = list(cache.items()), service.executor._cache_identity
        specs[0] = dict(specs[0], source_prompt="b")
        call = patch_replace(ctl, service, specs, operation)
        state = service.state_dict()
        records = list(ctl._ivs.items())
        register = model.layers[0].register_forward_pre_hook
        reached = []

        def fail(hook, **kwargs):
            if isinstance(getattr(hook, "__self__", None), _intervene.Intervention):
                reached.append(True)
                assert ("a", specs[0]["point"]) not in cache
                assert ("b", specs[0]["point"]) in cache
                raise RuntimeError("patch registration refused")
            return register(hook, **kwargs)

        monkeypatch.setattr(model.layers[0], "register_forward_pre_hook", fail)
        if operation == "boundary":
            with pytest.warns(RuntimeWarning, match="registration refused"):
                call()
        else:
            with pytest.raises(RuntimeError, match="registration refused"):
                call()
        assert reached == [True]
        assert service.state_dict() == state and list(ctl._ivs.items()) == records
        assert hook_objects(model) == hooks
        assert service.executor._cache is cache and service.executor._cache_identity is identity
        assert list(cache) == [key for key, _ in sources]
        assert all(cache[key] is tensor for key, tensor in sources)
        assert service.executor._retiring_sources == frozenset()
        torch.testing.assert_close(model(**tokenizer("c")).logits, before, rtol=0, atol=0)


def test_patch_replacement_conflict_restores_sources_and_preserves_queued_request(monkeypatch):
    from caliper.interp import _intervene
    from tests.interp.test_intervention_replay import hook_objects

    monkeypatch.setattr(_intervene, "MAX_SOURCE_ENTRIES", 2)
    with patch_capacity(2) as (model, tokenizer, ctl, service, specs):
        before = model(**tokenizer("c")).logits
        cache, identity = list(service.executor._cache.items()), service.executor._cache_identity
        hooks = hook_objects(model)
        specs[0] = dict(specs[0], source_prompt="b")
        call = patch_replace(ctl, service, specs, "boundary")
        queued = []

        def request(*args):
            assert ctl.lock.acquire(blocking=False), "source callback holds the controls lock"
            ctl.lock.release()
            service.set({"kind": "add", "point": "layers.0.mlp_out", "vector": [1.] * 4, "coefficient": 1})
            queued.append(service.state_dict())

        observer = model.register_forward_pre_hook(request)
        try:
            call()
        finally:
            observer.remove()
        assert len(queued) == 1 and service.state_dict() == queued[0]
        assert hook_objects(model) == hooks
        assert list(service.executor._cache) == [key for key, _ in cache]
        assert all(service.executor._cache[key] is tensor for key, tensor in cache)
        assert service.executor._cache_identity is identity
        torch.testing.assert_close(model(**tokenizer("c")).logits, before, rtol=0, atol=0)
        ctl.step()
        assert service.specifications()[0]["source_prompt"] == "b"
        assert len(service.specifications()) == 3


def test_patch_byte_limit_refuses_final_overflow_without_loss(monkeypatch):
    from caliper.interp import _intervene
    from tests.interp.test_intervention_replay import hook_objects

    with patch_capacity(2) as (model, tokenizer, ctl, service, specs):
        cache = list(service.executor._cache.items())
        monkeypatch.setattr(_intervene, "MAX_SOURCE_BYTES",
                            sum(t.numel() * t.element_size() for _, t in cache))
        before, hooks = model(**tokenizer("c")).logits, hook_objects(model)
        service.set(dict(specs[0], source_prompt="bb"))
        state = service.state_dict()
        with pytest.warns(RuntimeWarning, match="bounded source cache"):
            ctl.step()
        assert service.state_dict() == state and hook_objects(model) == hooks
        assert list(service.executor._cache) == [key for key, _ in cache]
        assert all(service.executor._cache[key] is tensor for key, tensor in cache)
        torch.testing.assert_close(model(**tokenizer("c")).logits, before, rtol=0, atol=0)
        service.set(dict(specs[0], source_prompt="b"))
        ctl.step()
        assert service.specifications()[0]["source_prompt"] == "b"
        assert sum(t.numel() * t.element_size() for t in service.executor._cache.values()) == _intervene.MAX_SOURCE_BYTES
