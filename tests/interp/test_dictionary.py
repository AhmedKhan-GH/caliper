"""Dictionary identity, safe storage, and exact fixture reconstruction."""
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from caliper.interp import load_dictionary
from caliper.interp import _dictionary
from caliper.interp._dictionary import model_identity, weights_hash
from caliper.interp._memory import CleanStop, MemoryGuard
from .fixture import TinyLM


@pytest.fixture(autouse=True)
def memory_readers(monkeypatch):
    values = {"resident": 0, "footprint": 0, "peak": 0, "driver": 0,
              "pressure": 50, "clock": 0}

    def guard(limit, device):
        assert torch.device(device).type == "cpu"
        return MemoryGuard(limit, device, **{
            name: lambda name=name: values[name] for name in values
        })

    monkeypatch.setattr(_dictionary, "MemoryGuard", guard)
    return values


def fixture_manifest(model):
    count = len(model.layers)
    return {
        "format_version": 1, "model_id": "tiny-lm", "model_hash": weights_hash(model),
        "revision": "fixture-v1-seed-7", "layers": count, "width": model.config.hidden_size,
        "latents": model.config.latents, "activation": "topk", "k": model.config.k,
        "read_points": [f"layers.{i}.mlp_in" for i in range(count)],
        "write_points": [f"layers.{i}.mlp_out" for i in range(count)],
        "training": {"dataset_id": "fixture", "dataset_config": "random-inputs",
                     "dataset_revision": "seed-17", "split": "train", "partitions": {},
                     "tokens": 0, "elapsed_seconds": 0, "peak_memory_bytes": 0},
        "quality": {"fvu": None, "dead_fraction": None, "cross_entropy_delta": None,
                    "evaluated_tokens": 0},
    }


def write_fixture(folder, model=None):
    model = model if model is not None else TinyLM()
    folder.mkdir(exist_ok=True, parents=True)
    data = fixture_manifest(model)
    (folder / "dictionary.json").write_text(json.dumps(data))
    for i, block in enumerate(model.layers):
        save_file({name: p.detach().clone() for name, p in block.mlp.named_parameters()},
                  folder / f"layer_{i:02d}.safetensors")
    return model, data


def test_exact_fixture_and_input_gradients(tmp_path):
    model, data = write_fixture(tmp_path)
    dictionary = load_dictionary(tmp_path, model=model)
    assert dictionary.metadata == data
    assert len(dictionary.layers) == 4
    for i, block in enumerate(model.layers):
        x = torch.randn(2, 7, 64, dtype=torch.float64, requires_grad=True)
        a = dictionary.encode(i, x)
        assert torch.equal(a, block.mlp.encode(x))
        assert torch.equal(dictionary.decode(i, a), block.mlp(x))
        assert dictionary.error(i, x, block.mlp(x)).count_nonzero() == 0
        assert (a.count_nonzero(-1) <= 8).all()
        assert dictionary.layers[i].W_dec.device == x.device
        assert not dictionary.layers[i].W_dec.requires_grad
        dictionary.decode(i, a).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


def test_identity_hash_covers_all_weights(tmp_path):
    model, _ = write_fixture(tmp_path)
    same = TinyLM()
    assert weights_hash(model) == weights_hash(same)
    with torch.no_grad():
        same.lm_head.weight[-1, -1] += 1
    assert weights_hash(model) != weights_hash(same)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_dictionary(tmp_path, model=same)


def test_expected_hash_and_watch(tmp_path, monkeypatch):
    from caliper import _programs
    model, data = write_fixture(tmp_path)
    monkeypatch.setitem(_programs._STATE, "model", None)
    with pytest.raises(ValueError, match="supply a model"):
        load_dictionary(tmp_path)
    assert load_dictionary(tmp_path, model_hash=data["model_hash"]).metadata == data
    with pytest.raises(ValueError, match="expected model_hash differ"):
        load_dictionary(tmp_path, model=model, model_hash="sha256:" + "0" * 64)
    monkeypatch.setitem(_programs._STATE, "model", model)
    assert load_dictionary(tmp_path).metadata == data


@pytest.mark.parametrize("field,value", [
    ("activation", "relu"), ("k", 257), ("k", 0), ("latents", 128),
    ("width", 32), ("layers", 3), ("revision", ""), ("training", {}),
    ("quality", {}), ("format_version", 2),
    ("read_points", [f"wrong.{i}.mlp_in" for i in range(4)]),
])
def test_reject_manifest_mismatch(tmp_path, field, value):
    model, data = write_fixture(tmp_path)
    data[field] = value
    (tmp_path / "dictionary.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_dictionary(tmp_path, model=model)


@pytest.mark.parametrize("damage", ["nan", "inf", "shape", "extra", "integer", "overflow"])
def test_reject_unsafe_layer_values(tmp_path, damage):
    model, _ = write_fixture(tmp_path)
    tensors = {n: p.detach().clone() for n, p in model.layers[0].mlp.named_parameters()}
    if damage in ("nan", "inf"):
        tensors["W_enc"][0, 0] = float(damage)
    elif damage == "shape":
        tensors["W_dec"] = tensors["W_dec"].T.contiguous()
    elif damage == "extra":
        tensors["pickle"] = torch.ones(1)
    elif damage == "integer":
        tensors["b_enc"] = tensors["b_enc"].to(torch.int64)
    else:
        tensors["W_enc"][0, 0] = 1e30
    save_file(tensors, tmp_path / "layer_00.safetensors")
    with pytest.raises(ValueError):
        load_dictionary(tmp_path, model=model, dtype=torch.float16 if damage == "overflow" else None)


def test_nonfinite_metadata(tmp_path):
    model, data = write_fixture(tmp_path)
    data["quality"]["fvu"] = float("nan")
    (tmp_path / "dictionary.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="nonfinite"):
        load_dictionary(tmp_path, model=model)


@pytest.mark.parametrize("section,field,value", [
    ("training", "elapsed_seconds", "1e309"),
    ("training", "peak_memory_bytes", "1e309"),
    ("quality", "fvu", "1e309"), ("quality", "dead_fraction", "1e309"),
    ("quality", "cross_entropy_delta", "-1e309"),
    ("training", "elapsed_seconds", "-1"), ("training", "peak_memory_bytes", "-1"),
    ("quality", "fvu", "-0.1"), ("quality", "dead_fraction", "-0.1"),
    ("quality", "dead_fraction", "1.1"),
])
def test_metadata_numeric_ranges(tmp_path, section, field, value):
    model, data = write_fixture(tmp_path)
    data[section][field] = "REPLACE_NUMBER"
    text = json.dumps(data).replace('"REPLACE_NUMBER"', value)
    (tmp_path / "dictionary.json").write_text(text)
    with pytest.raises(ValueError):
        load_dictionary(tmp_path, model=model)


def test_hf_identity_requires_resolved_checkpoint():
    revision = "a" * 40
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5_text",
                            _name_or_path="Qwen/Qwen3.5-0.8B", _commit_hash=revision))
    assert _dictionary._hf_identity(model.config) == f"hf:Qwen/Qwen3.5-0.8B@{revision}"
    with pytest.raises(ValueError, match="configuration does not verify"):
        model_identity(model)
    model.config._commit_hash = "main"
    with pytest.raises(ValueError, match="exact HF"):
        model_identity(model)
    model.config._name_or_path = f"cache/models--Qwen--Qwen3.5-0.8B/snapshots/{revision}"
    model.config._commit_hash = None
    assert _dictionary._hf_identity(model.config).endswith("@" + revision)
    model.config._commit_hash = "b" * 40
    with pytest.raises(ValueError, match="conflicts"):
        model_identity(model)


@pytest.mark.parametrize("separator", ["/", "\\"])
@pytest.mark.parametrize("prefix", ["cache", "C:/cache", "//server/cache"])
def test_hf_snapshot_separators_and_revision_conflict(separator, prefix):
    revision = "a" * 40
    name = f"{prefix}/models--Qwen--Qwen3.5-0.8B/snapshots/{revision}".replace("/", separator)
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5_text",
                            _name_or_path=name, _commit_hash=None))
    assert _dictionary._hf_identity(model.config) == f"hf:Qwen/Qwen3.5-0.8B@{revision}"
    with pytest.raises(ValueError, match="configuration does not verify"):
        model_identity(model)
    model.config._commit_hash = "b" * 40
    with pytest.raises(ValueError, match="conflicts"):
        model_identity(model)


def test_partial_export_refuses_load_and_retry_resets_quality(tmp_path, monkeypatch):
    from copy import deepcopy
    from safetensors.torch import load_file
    from tools.interp import checkpoint_state

    model, data = write_fixture(tmp_path, TinyLM(layers=2, width=8, heads=2, latents=16, k=2))
    data["quality"].update(fvu=0.01, dead_fraction=0, cross_entropy_delta=0,
                           evaluated_tokens=200_000, status="passed")
    (tmp_path / "dictionary.json").write_text(json.dumps(data))
    assert load_dictionary(tmp_path, model=model).metadata["quality"]["status"] == "passed"
    trainer = SimpleNamespace(metadata=deepcopy(data), device="cpu",
                              layers=deepcopy([block.mlp for block in model.layers]))
    trainer.optimizers = [torch.optim.Adam(layer.parameters()) for layer in trainer.layers]
    with torch.no_grad():
        for layer in trainer.layers:
            layer.W_enc.add_(1)
    progress = {"tokens": 32, "elapsed_seconds": 1}
    save = checkpoint_state._save_tensors
    calls = []

    def fail_second(values, path):
        assert not (tmp_path / "dictionary.json").exists()
        calls.append(path)
        if len(calls) == 2:
            raise OSError("fixture export failure")
        save(values, path)

    monkeypatch.setattr(checkpoint_state, "_save_tensors", fail_second)
    with pytest.raises(OSError, match="export failure"):
        checkpoint_state.export_dictionary(tmp_path, trainer, progress)
    assert len(calls) == 2
    assert torch.equal(load_file(tmp_path / "layer_00.safetensors")["W_enc"], trainer.layers[0].W_enc)
    assert torch.equal(load_file(tmp_path / "layer_01.safetensors")["W_enc"], model.layers[1].mlp.W_enc)
    with pytest.raises(FileNotFoundError):
        load_dictionary(tmp_path, model=model)

    monkeypatch.setattr(checkpoint_state, "_save_tensors", save)
    checkpoint_state.export_dictionary(tmp_path, trainer, progress)
    dictionary = load_dictionary(tmp_path, model=model)
    assert dictionary.metadata["training"]["tokens"] == 32
    assert dictionary.metadata["quality"] == {
        "fvu": None, "dead_fraction": None, "cross_entropy_delta": None,
        "evaluated_tokens": 0, "status": "not_evaluated",
    }
    for actual, expected in zip(dictionary.layers, trainer.layers):
        for name, value in expected.named_parameters():
            assert torch.equal(getattr(actual, name), value)


def header_only_dictionary(tmp_path, monkeypatch, *, latents, stored_dtype):
    """Supply Qwen-size headers without weights, a model, or device allocation."""
    import safetensors

    data = fixture_manifest(TinyLM(layers=1, width=8, heads=2, latents=16, k=2))
    revision = "2fc06364715b967f1860aea9cf38778875588b17"
    data.update(model_id="Qwen/Qwen3.5-0.8B", revision=revision,
                model_hash=f"hf:Qwen/Qwen3.5-0.8B@{revision}", layers=24,
                width=1024, latents=latents, k=32,
                read_points=[f"model.layers.{i}.mlp_in" for i in range(24)],
                write_points=[f"model.layers.{i}.mlp_out" for i in range(24)])
    (tmp_path / "dictionary.json").write_text(json.dumps(data))
    shapes = {"W_enc": (latents, 1024), "b_enc": (latents,),
              "W_dec": (latents, 1024), "b_dec": (1024,)}
    calls = []

    class Header:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def keys(self):
            return shapes.keys()

        def get_slice(self, name):
            return SimpleNamespace(get_shape=lambda: shapes[name], get_dtype=lambda: stored_dtype)

        def get_tensor(self, name):
            raise AssertionError("tensor materialization requested")

    def open_header(path, *, framework, device):
        assert framework == "pt" and device == "cpu"
        calls.append(path.name)
        return Header()

    monkeypatch.setattr(safetensors, "safe_open", open_header)
    return data, calls


@pytest.mark.parametrize("latents,stored_dtype,dtype", [
    (131_072, "F32", None), (65_536, "F16", torch.float32),
])
def test_oversized_headers_refuse_before_tensor_or_device_allocation(
    tmp_path, monkeypatch, latents, stored_dtype, dtype,
):
    data, calls = header_only_dictionary(tmp_path, monkeypatch, latents=latents, stored_dtype=stored_dtype)
    guard = _dictionary.MemoryGuard
    monkeypatch.setattr(_dictionary, "MemoryGuard", lambda limit, device:
                        guard(limit, device) if device == "cpu" else pytest.fail("device setup ran"))
    monkeypatch.setattr(torch.Tensor, "to", lambda *_a, **_k: pytest.fail("tensor conversion ran"))
    with pytest.raises(CleanStop, match="memory limit"):
        load_dictionary(tmp_path, model_hash=data["model_hash"], dtype=dtype, device="mps")
    assert len(calls) == 24


@pytest.mark.parametrize("latents,expected_bytes", [(16_384, 3_222_896_640), (8192, 1_611_497_472)])
@pytest.mark.parametrize("stored_dtype", ["F16", "F32"])
def test_qwen_header_envelopes_fit_without_materialization(
    tmp_path, monkeypatch, memory_readers, latents, expected_bytes, stored_dtype,
):
    data, calls = header_only_dictionary(tmp_path, monkeypatch, latents=latents, stored_dtype=stored_dtype)
    memory_readers.update(resident=3_200_000_000, footprint=4_000_000_000, peak=4_000_000_000)
    admitted = []
    admission = _dictionary._admit

    def record(memory, required, stage, **kwargs):
        current = admission(memory, required, stage, **kwargs)
        if stage != "dictionary_metadata_admission":
            admitted.append(required)
        return current

    monkeypatch.setattr(_dictionary, "_admit", record)
    # The header mock stops at the first tensor request after full admission.
    with pytest.raises(AssertionError, match="tensor materialization requested"):
        load_dictionary(tmp_path, model_hash=data["model_hash"], dtype=torch.float32)
    assert len(calls) == 25
    assert expected_bytes < admitted[0] < expected_bytes + 140_000_000
    assert admitted == [admitted[0]] * 3
    assert memory_readers["footprint"] + admitted[0] < 12_000_000_000


@pytest.mark.parametrize("limit", [0, -1, 12.1, float("inf"), float("nan"), True, "12"])
def test_dictionary_memory_limit_validation(limit):
    with pytest.raises(ValueError, match="memory limit"):
        load_dictionary("unused", model_hash="unused", memory_limit_gb=limit)


@pytest.mark.parametrize("reader", ["resident", "footprint", "driver"])
def test_admission_includes_existing_process_and_model_memory(tmp_path, monkeypatch, memory_readers, reader):
    data, _ = header_only_dictionary(tmp_path, monkeypatch, latents=8192, stored_dtype="F32")
    memory_readers[reader] = 11_000_000_000
    with pytest.raises(CleanStop, match="memory limit"):
        load_dictionary(tmp_path, model_hash=data["model_hash"])


def test_admission_includes_fallback_rss_plus_driver(tmp_path, monkeypatch, memory_readers):
    data, _ = header_only_dictionary(tmp_path, monkeypatch, latents=8192, stored_dtype="F32")
    memory_readers.update(footprint=None, resident=6_000_000_000, driver=5_000_000_000)
    with pytest.raises(CleanStop, match="memory limit"):
        load_dictionary(tmp_path, model_hash=data["model_hash"])


def test_explicit_cap_reserves_conversion_copies(tmp_path, memory_readers):
    model, data = write_fixture(tmp_path, TinyLM(layers=2, width=8, heads=2, latents=16, k=2))
    memory_readers.update(resident=10_000, footprint=10_000)
    shapes = {name: tuple(value.shape) for name, value in model.layers[0].mlp.named_parameters()}
    stored = dict.fromkeys(shapes, torch.float64)
    destination, temporary, converted = _dictionary._layer_bytes(stored, shapes, torch.device("cpu"), torch.float32)
    assert converted
    assert destination == 1120
    assert temporary == 2240 + 512 + 2560
    metadata_bytes = _dictionary._metadata_budget((tmp_path / "dictionary.json").stat().st_size)
    with pytest.raises(CleanStop, match="memory limit"):
        load_dictionary(tmp_path, model_hash=data["model_hash"], dtype=torch.float32,
                        memory_limit_gb=(10_000 + 2 * destination) / 1e9)
    dictionary = load_dictionary(
        tmp_path, model_hash=data["model_hash"], dtype=torch.float32,
        memory_limit_gb=(10_000 + metadata_bytes + 2 * destination + temporary + 1) / 1e9,
    )
    for layer, block in zip(dictionary.layers, model.layers):
        for name, value in block.mlp.named_parameters():
            assert torch.equal(getattr(layer, name), value.float())


def test_later_header_failure_precedes_all_tensor_reads(tmp_path, monkeypatch):
    import safetensors

    model, _ = write_fixture(tmp_path, TinyLM(layers=2, width=8, heads=2, latents=16, k=2))
    tensors = {name: value.detach() for name, value in model.layers[1].mlp.named_parameters()}
    tensors["W_enc"] = tensors["W_enc"][:1].contiguous()
    save_file(tensors, tmp_path / "layer_01.safetensors")
    open_source = safetensors.safe_open

    class HeadersOnly:
        def __init__(self, *args, **kwargs):
            self.source = open_source(*args, **kwargs)

        def __enter__(self):
            source = self.source.__enter__()
            return SimpleNamespace(keys=source.keys, get_slice=source.get_slice,
                                   get_tensor=lambda *_: pytest.fail("tensor materialization ran"))

        def __exit__(self, *args):
            self.source.__exit__(*args)

    monkeypatch.setattr(safetensors, "safe_open", HeadersOnly)
    with pytest.raises(ValueError, match="layer 1 has incorrect W_enc shape"):
        load_dictionary(tmp_path, model=model)


def test_conversion_releases_sources_and_rechecks_memory_between_layers(tmp_path, monkeypatch, memory_readers):
    import safetensors
    import weakref

    model, data = write_fixture(tmp_path, TinyLM(layers=2, width=8, heads=2, latents=16, k=2))
    open_source = safetensors.safe_open
    sources, reads = [], []

    class Sources:
        def __init__(self, path, **kwargs):
            self.source = open_source(path, **kwargs)
            self.name = path.name

        def __enter__(self):
            self.source.__enter__()
            return self

        def __exit__(self, *args):
            self.source.__exit__(*args)

        def keys(self):
            return self.source.keys()

        def get_slice(self, name):
            return self.source.get_slice(name)

        def get_tensor(self, name):
            assert all(reference() is None for reference in sources)
            value = self.source.get_tensor(name)
            sources.append(weakref.ref(value))
            reads.append((self.name, name))
            # A new process allocation must prevent the next layer load.
            memory_readers["resident"] = 999_999
            return value

    monkeypatch.setattr(safetensors, "safe_open", Sources)
    with pytest.raises(CleanStop, match="memory limit"):
        load_dictionary(tmp_path, model_hash=data["model_hash"], dtype=torch.float32,
                        memory_limit_gb=0.001)
    assert reads == [("layer_00.safetensors", name) for name in _dictionary._WEIGHTS]
    assert all(reference() is None for reference in sources)


def forbid_metadata_consumers(monkeypatch):
    import safetensors

    monkeypatch.setattr(_dictionary.json, "loads", lambda *_a, **_k: pytest.fail("JSON parser ran"))
    monkeypatch.setattr(safetensors, "safe_open", lambda *_a, **_k: pytest.fail("tensor header access ran"))


def forbid_metadata_read(monkeypatch):
    from contextlib import contextmanager
    from pathlib import Path

    original = Path.open

    @contextmanager
    def open_source(path, *args, **kwargs):
        with original(path, *args, **kwargs) as source:
            yield SimpleNamespace(fileno=source.fileno,
                                  read=lambda *_: pytest.fail("metadata byte allocation ran"))

    monkeypatch.setattr(Path, "open", open_source)


def test_metadata_size_refuses_before_read_parse_or_headers(tmp_path, monkeypatch):
    import stat

    (tmp_path / "dictionary.json").write_text("{}")
    # Report a large size without an oversized file, string, or sparse fixture.
    monkeypatch.setattr(_dictionary.os, "fstat", lambda _: SimpleNamespace(st_mode=stat.S_IFREG, st_size=2**32))
    forbid_metadata_consumers(monkeypatch)
    forbid_metadata_read(monkeypatch)
    with pytest.raises(CleanStop, match="metadata exceeds the byte limit"):
        load_dictionary(tmp_path, model_hash="unused", memory_limit_gb=1)


def test_metadata_reserve_refuses_before_read_parse_or_headers(tmp_path, monkeypatch, memory_readers):
    path = tmp_path / "dictionary.json"
    path.write_text('{"extra":"small"}')
    reserve = _dictionary._metadata_budget(path.stat().st_size)
    memory_readers["footprint"] = 1_000_000_000 - reserve + 1
    forbid_metadata_consumers(monkeypatch)
    forbid_metadata_read(monkeypatch)
    with pytest.raises(CleanStop, match="process memory limit"):
        load_dictionary(tmp_path, model_hash="unused", memory_limit_gb=1)


def test_metadata_nesting_refuses_before_parse_or_headers(tmp_path, monkeypatch):
    (tmp_path / "dictionary.json").write_text('{"extra":' + '[' * 64 + '0' + ']' * 64 + '}')
    forbid_metadata_consumers(monkeypatch)
    with pytest.raises(CleanStop, match="nesting limit"):
        load_dictionary(tmp_path, model_hash="unused")


def test_metadata_growth_refuses_with_bounded_read(tmp_path, monkeypatch):
    import stat
    from contextlib import contextmanager
    from pathlib import Path

    (tmp_path / "dictionary.json").write_text('{"extra":"append"}')
    monkeypatch.setattr(_dictionary.os, "fstat", lambda _: SimpleNamespace(st_mode=stat.S_IFREG, st_size=2))
    original = Path.open
    reads = []

    @contextmanager
    def open_source(path, *args, **kwargs):
        with original(path, *args, **kwargs) as source:
            def read(size):
                reads.append(size)
                return source.read(size)
            yield SimpleNamespace(fileno=source.fileno, read=read)

    monkeypatch.setattr(Path, "open", open_source)
    forbid_metadata_consumers(monkeypatch)
    with pytest.raises(ValueError, match="size changed"):
        load_dictionary(tmp_path, model_hash="unused")
    assert reads == [3]


def test_metadata_budget_remains_in_tensor_admission(tmp_path, memory_readers, monkeypatch):
    import safetensors

    model, data = write_fixture(tmp_path, TinyLM(layers=1, width=8, heads=2, latents=16, k=2))
    data["extra"] = '["escaped brackets\\\\\\"' * 8
    path = tmp_path / "dictionary.json"
    path.write_text(json.dumps(data))
    reserve = _dictionary._metadata_budget(path.stat().st_size)
    # Metadata fits alone, but the retained object and the tensors do not fit together.
    memory_readers.update(footprint=10_000, resident=10_000)
    original = safetensors.safe_open

    class HeadersOnly:
        def __init__(self, *args, **kwargs):
            self.source = original(*args, **kwargs)

        def __enter__(self):
            source = self.source.__enter__()
            return SimpleNamespace(keys=source.keys, get_slice=source.get_slice,
                                   get_tensor=lambda *_: pytest.fail("tensor allocation ran"))

        def __exit__(self, *args):
            self.source.__exit__(*args)

    monkeypatch.setattr(safetensors, "safe_open", HeadersOnly)
    with pytest.raises(CleanStop, match="process memory limit"):
        load_dictionary(tmp_path, model_hash=data["model_hash"], memory_limit_gb=(10_000 + reserve + 1) / 1e9)


def test_metadata_depth_ignores_escaped_string_contents(tmp_path):
    model, data = write_fixture(tmp_path, TinyLM(layers=1, width=8, heads=2, latents=16, k=2))
    data["extra"] = '["\\]' * 100
    (tmp_path / "dictionary.json").write_text(json.dumps(data))
    assert load_dictionary(tmp_path, model_hash=data["model_hash"]).metadata == data
