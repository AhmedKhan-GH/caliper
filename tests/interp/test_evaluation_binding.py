"""CPU regressions for evaluation artifact and cursor evidence."""
import json
import copy
from contextlib import contextmanager

import pytest
import torch
from safetensors.torch import load_file, save_file

from caliper.interp import load_dictionary
from tools.interp import eval_dictionary as evaluation, evaluation_binding as binding_module
from tools.interp.checkpoint_state import dictionary_fingerprint, write_json
from tools.interp.memory_limits import CleanStop
from tools.interp.text_stream import TextStream
from .fixture import TinyLM, TinyTokenizer
from .test_dictionary import memory_readers, write_fixture
from .test_training_stream import documents


def fixture(folder):
    model = TinyLM(layers=2, width=4, heads=2, latents=8, k=2)
    write_fixture(folder, model)
    return model, load_dictionary(folder, model=model, device="cpu")


def evaluate(folder, model, dictionary, **kwargs):
    return evaluation.evaluate(model, dictionary, TextStream(documents(), TinyTokenizer(), "quality"),
                               folder, tokens=8, context=4, batch_tokens=4, chunk_tokens=2, **kwargs)


def publications(folder):
    paths = [folder / "dictionary.json", folder / "evaluation" / "state.json"]
    return [path.read_bytes() if path.exists() else None for path in paths]


def change(folder, dictionary, kind):
    if kind == "alias":
        dictionary.layers[0].W_dec.detach()[0, 0] += 1
    elif kind == "unversioned_alias":
        version = dictionary.layers[0].W_dec._version
        dictionary.layers[0].W_dec.data[0, 0] += 1
        assert dictionary.layers[0].W_dec._version == version
    elif kind == "replacement":
        old = dictionary.layers[0].W_dec
        dictionary.layers[0].W_dec = torch.nn.Parameter(old.detach().clone(), requires_grad=False)
    elif kind == "file":
        path = folder / "layer_00.safetensors"
        values = {k: v.clone() for k, v in load_file(path).items()}
        values["W_dec"][0, 0] += 1
        save_file(values, path.with_suffix(".new"))
        path.with_suffix(".new").replace(path)
    elif kind == "memory_metadata":
        dictionary.metadata["training"]["tokens"] += 1
    elif kind == "disk_metadata":
        path = folder / "dictionary.json"
        data = json.loads(path.read_text())
        data["training"]["tokens"] += 1
        write_json(path, data)
    elif kind == "k":
        dictionary.layers[0].k += 1


@pytest.mark.parametrize("kind", ["alias", "unversioned_alias", "file", "memory_metadata", "disk_metadata", "k"])
def test_admission_refuses_weights_or_metadata_outside_artifact(tmp_path, monkeypatch, kind):
    model, dictionary = fixture(tmp_path)
    change(tmp_path, dictionary, kind)
    before = publications(tmp_path)
    monkeypatch.setattr(evaluation, "evaluate_batch", lambda *a, **k: pytest.fail("batch ran"))
    with pytest.raises(ValueError, match="artifact binding"):
        evaluate(tmp_path, model, dictionary)
    assert publications(tmp_path) == before


def test_admission_refuses_different_compatible_artifact_folder(tmp_path, monkeypatch):
    model, dictionary = fixture(tmp_path / "loaded")
    write_fixture(tmp_path / "other", model)
    change(tmp_path / "other", dictionary, "file")
    before = publications(tmp_path / "other")
    monkeypatch.setattr(evaluation, "evaluate_batch", lambda *a, **k: pytest.fail("batch ran"))
    with pytest.raises(ValueError, match="artifact binding"):
        evaluate(tmp_path / "other", model, dictionary)
    assert publications(tmp_path / "other") == before


@pytest.mark.parametrize("kind", ["alias", "unversioned_alias", "file", "memory_metadata", "disk_metadata",
                                  "cursor", "legacy", "dtype"])
def test_resume_refuses_changed_tensor_evidence_before_restore(tmp_path, monkeypatch, kind):
    model, dictionary = fixture(tmp_path)
    evaluate(tmp_path, model, dictionary)
    state_path = tmp_path / "evaluation" / "state.json"
    state = json.loads(state_path.read_text())
    assert state["identity"] == dictionary_fingerprint(tmp_path, dictionary.metadata)
    assert state["tensor_binding"]["version"] == 1
    if kind == "cursor":
        state["tensor_binding"]["tensor_hash"] = "0" * 64
        write_json(state_path, state)
    elif kind == "legacy":
        del state["tensor_binding"]
        write_json(state_path, state)
    elif kind == "dtype":
        dictionary = load_dictionary(tmp_path, model=model, device="cpu", dtype=torch.float32)
    else:
        change(tmp_path, dictionary, kind)
    before = publications(tmp_path)
    monkeypatch.setattr(evaluation, "load_file", lambda *a, **k: pytest.fail("statistics restore ran"))
    with pytest.raises(ValueError, match="dictionary or settings changed"):
        evaluate(tmp_path, model, dictionary, resume=True)
    assert publications(tmp_path) == before


@pytest.mark.parametrize("kind", ["alias", "unversioned_alias", "replacement", "file", "memory_metadata",
                                  "disk_metadata", "k"])
@pytest.mark.parametrize("boundary", ["batch", "metrics", "stop"])
def test_changes_before_cursor_or_quality_publication_are_refused(tmp_path, monkeypatch, kind, boundary):
    model, dictionary = fixture(tmp_path)
    batch, metrics = evaluation.evaluate_batch, evaluation.QualityStats.metrics
    before = publications(tmp_path)

    def mutate():
        nonlocal before
        change(tmp_path, dictionary, kind)
        before = publications(tmp_path)

    def interrupted_batch(*args, **kwargs):
        result = batch(*args, **kwargs)
        if boundary != "metrics":
            mutate()
        if boundary == "stop":
            raise CleanStop("synthetic stop after a changed batch")
        return result

    def changed_metrics(self):
        result = metrics(self)
        mutate()
        return result

    monkeypatch.setattr(evaluation, "evaluate_batch", interrupted_batch)
    if boundary == "metrics":
        monkeypatch.setattr(evaluation.QualityStats, "metrics", changed_metrics)
    with pytest.raises(ValueError, match="artifact binding"):
        evaluate(tmp_path, model, dictionary)
    assert publications(tmp_path) == before


def test_version_change_is_refused_even_when_values_are_restored(tmp_path, monkeypatch):
    model, dictionary = fixture(tmp_path)
    batch = evaluation.evaluate_batch
    before = publications(tmp_path)

    def changed(*args, **kwargs):
        alias = dictionary.layers[0].W_dec.detach()
        original = alias.clone()
        alias.add_(1)
        result = batch(*args, **kwargs)
        alias.copy_(original)
        return result

    monkeypatch.setattr(evaluation, "evaluate_batch", changed)
    with pytest.raises(ValueError, match="artifact binding"):
        evaluate(tmp_path, model, dictionary)
    assert publications(tmp_path) == before


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_binding_accepts_loader_dtype_conversion_in_bounded_slices(tmp_path, monkeypatch, dtype):
    model, _ = fixture(tmp_path)
    dictionary = load_dictionary(tmp_path, model=model, device="cpu", dtype=dtype)
    identity = dictionary_fingerprint(tmp_path, dictionary.metadata)
    monkeypatch.setattr(binding_module, "_CHUNK_BYTES", 16)
    from caliper.interp import _dictionary
    monkeypatch.setattr(_dictionary, "_HASH_CHUNK_BYTES", 16)
    transfers = []
    original_to = torch.Tensor.to
    original_open = binding_module.safe_open

    @contextmanager
    def sliced_file(*args, **kwargs):
        with original_open(*args, **kwargs) as source:
            class Source:
                keys = source.keys

                def get_slice(self, name):
                    stored = source.get_slice(name)

                    class Slice:
                        get_shape = stored.get_shape

                        def __getitem__(self, selection):
                            value = stored[selection]
                            assert value.numel() <= 2
                            return value

                    return Slice()

                def get_tensor(self, name):
                    pytest.fail("whole tensor read ran")

            yield Source()

    class Guard:
        device = torch.device("cpu")
        peak, limit, checks = 0, 10**9, 0

        def check(self, stage, **kwargs):
            if stage == "evaluation_binding":
                self.checks += 1

    guard = Guard()

    def bounded_to(tensor, *args, **kwargs):
        assert tensor.numel() * tensor.element_size() <= 16
        if kwargs.get("device") == "cpu":
            transfers.append(tensor.numel() * tensor.element_size())
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", bounded_to)
    monkeypatch.setattr(binding_module, "safe_open", sliced_file)
    bound = binding_module.EvaluationBinding(tmp_path, dictionary, identity, model=model, memory=guard)
    bound.verify()
    assert transfers and guard.checks > 0
    assert len(bound.evidence["tensor_hash"]) == 64


def test_binding_memory_refusal_preserves_prior_publications(tmp_path):
    model, dictionary = fixture(tmp_path)
    before = publications(tmp_path)

    class Guard:
        device = torch.device("cpu")
        peak, limit = 0, 10**9

        def check(self, stage, **kwargs):
            if stage == "evaluation_binding":
                raise CleanStop("synthetic readback admission refusal")

    with pytest.raises(CleanStop, match="admission refusal"):
        evaluate(tmp_path, model, dictionary, memory=Guard())
    assert publications(tmp_path) == before


@pytest.mark.parametrize("periodic", [False, True])
def test_unversioned_change_during_statistics_write_refuses_cursor(tmp_path, monkeypatch, periodic):
    model, dictionary = fixture(tmp_path)
    before = publications(tmp_path)
    save = evaluation._save_tensors
    if periodic:
        # A deterministic clock reaches the periodic cursor path after one batch.
        ticks = iter([0, 241, 242])
        monkeypatch.setattr(evaluation.time, "monotonic", lambda: next(ticks))

    def changed(*args, **kwargs):
        save(*args, **kwargs)
        change(tmp_path, dictionary, "unversioned_alias")

    monkeypatch.setattr(evaluation, "_save_tensors", changed)
    with pytest.raises(ValueError, match="artifact binding"):
        evaluate(tmp_path, model, dictionary)
    assert publications(tmp_path) == before


def test_change_after_cursor_preparation_refuses_publication(tmp_path, monkeypatch):
    model, dictionary = fixture(tmp_path)
    before = publications(tmp_path)
    prepare = evaluation.prepare_evaluation

    def changed(*args, **kwargs):
        state = prepare(*args, **kwargs)
        change(tmp_path, dictionary, "unversioned_alias")
        return state

    monkeypatch.setattr(evaluation, "prepare_evaluation", changed)
    with pytest.raises(ValueError, match="artifact binding"):
        evaluate(tmp_path, model, dictionary)
    assert publications(tmp_path) == before
    with pytest.raises(ValueError, match="artifact binding"):
        evaluate(tmp_path, model, dictionary, resume=True)


def test_binding_rechecks_disk_bytes_before_publication(tmp_path, monkeypatch):
    model, dictionary = fixture(tmp_path)
    before = publications(tmp_path)
    batch = evaluation.evaluate_batch

    def changed(*args, **kwargs):
        result = batch(*args, **kwargs)
        change(tmp_path, dictionary, "file")
        return result

    # Stat hints alone are not evidence. Force equal hints across a changed file.
    monkeypatch.setattr(binding_module.EvaluationBinding, "_file_stamps", lambda self: [])
    monkeypatch.setattr(evaluation, "evaluate_batch", changed)
    with pytest.raises(ValueError, match="artifact binding"):
        evaluate(tmp_path, model, dictionary)
    assert publications(tmp_path) == before


@pytest.mark.parametrize("resume", [False, True])
def test_model_mismatch_preserves_evidence_before_batches_or_restore(tmp_path, monkeypatch, resume):
    model, dictionary = fixture(tmp_path)
    if resume:
        evaluate(tmp_path, model, dictionary)
    other = copy.deepcopy(model)
    other.lm_head.weight.data[0, 0] += 1
    before = publications(tmp_path)
    monkeypatch.setattr(evaluation, "evaluate_batch", lambda *a, **k: pytest.fail("batch ran"))
    monkeypatch.setattr(evaluation, "load_file", lambda *a, **k: pytest.fail("statistics restore ran"))
    with pytest.raises(ValueError, match="model identity mismatch"):
        evaluate(tmp_path, other, dictionary, resume=resume)
    assert publications(tmp_path) == before


@pytest.mark.parametrize("change", ["missing", "hash", "version", "runtime", "settings", "legacy_settings"])
def test_cursor_model_evidence_precedes_statistics_restore(tmp_path, monkeypatch, change):
    model, dictionary = fixture(tmp_path)
    evaluate(tmp_path, model, dictionary)
    path = tmp_path / "evaluation" / "state.json"
    state = json.loads(path.read_text())
    if change == "missing":
        del state["model_binding"]
    elif change == "hash":
        state["model_binding"]["model_hash"] = "sha256:" + "0" * 64
    elif change == "version":
        state["model_binding"]["version"] += 1
    elif change == "settings":
        state["model_binding"]["forward_settings_hash"] = "0" * 64
    elif change == "legacy_settings":
        state["model_binding"].pop("forward_settings_hash")
        state["model_binding"]["version"] = 1
    else:
        state["model_binding"]["tensor_hash"] = "0" * 64
    write_json(path, state)
    before = publications(tmp_path)
    monkeypatch.setattr(evaluation, "load_file", lambda *a, **k: pytest.fail("statistics restore ran"))
    with pytest.raises(ValueError, match="cursor binding"):
        evaluate(tmp_path, model, dictionary, resume=True)
    assert publications(tmp_path) == before


def test_same_model_resume_keeps_artifact_fingerprint(tmp_path):
    model, dictionary = fixture(tmp_path)
    identity = dictionary_fingerprint(tmp_path, dictionary.metadata)
    quality = evaluate(tmp_path, model, dictionary)
    resumed = evaluate(tmp_path, copy.deepcopy(model), dictionary, resume=True)
    state = json.loads((tmp_path / "evaluation" / "state.json").read_text())
    assert quality["model_binding"] == resumed["model_binding"] == state["model_binding"]
    assert quality["model_binding"]["model_hash"] == dictionary.metadata["model_hash"]
    assert quality["fvu"] == resumed["fvu"]
    assert identity == dictionary_fingerprint(tmp_path, dictionary.metadata)


@pytest.mark.parametrize("boundary", ["batch", "stop", "metrics", "statistics", "cursor"])
@pytest.mark.parametrize("kind", ["version", "alias", "replacement", "buffer", "forward"])
def test_model_drift_refuses_cursor_and_quality(tmp_path, monkeypatch, boundary, kind):
    model, dictionary = fixture(tmp_path)
    model.register_buffer("evaluation_buffer", torch.zeros(1), persistent=False)
    before = publications(tmp_path)

    def mutate():
        if kind == "version":
            with torch.no_grad():
                model.lm_head.weight.add_(1)
        elif kind == "alias":
            version = model.lm_head.weight._version
            model.lm_head.weight.data[0, 0] += 1
            assert model.lm_head.weight._version == version
        elif kind == "replacement":
            model.lm_head = copy.deepcopy(model.lm_head)
        elif kind == "forward":
            model.norm.eps = 1.0
        else:
            model.evaluation_buffer.data.add_(1)

    attribute = {"batch": "evaluate_batch", "stop": "evaluate_batch", "metrics": "metrics",
                 "statistics": "_save_tensors", "cursor": "prepare_evaluation"}[boundary]
    target = evaluation.QualityStats if boundary == "metrics" else evaluation
    original = getattr(target, attribute)

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        mutate()
        if boundary == "stop":
            raise CleanStop("synthetic stop after model drift")
        return result

    monkeypatch.setattr(target, attribute, changed)
    with pytest.raises(ValueError, match="model identity mismatch"):
        evaluate(tmp_path, model, dictionary)
    assert publications(tmp_path) == before


def test_model_verification_runs_after_mlp_restoration(tmp_path, monkeypatch):
    model, dictionary = fixture(tmp_path)
    originals = tuple(block.mlp for block in model.layers)
    identify = binding_module.model_identity
    checks = []

    def restored(value, **kwargs):
        assert tuple(block.mlp for block in value.layers) == originals
        checks.append(True)
        return identify(value, **kwargs)

    monkeypatch.setattr(binding_module, "model_identity", restored)
    evaluate(tmp_path, model, dictionary)
    assert len(checks) >= 3


def test_stop_after_restore_retains_committed_statistics(tmp_path, monkeypatch):
    model, dictionary = fixture(tmp_path)
    batch = evaluation.evaluate_batch
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise CleanStop("synthetic batch stop")
        return batch(*args, **kwargs)

    monkeypatch.setattr(evaluation, "evaluate_batch", interrupted)
    prior = evaluate(tmp_path, model, dictionary)
    assert prior["evaluated_tokens"] == 4
    state_path = tmp_path / "evaluation" / "state.json"
    before = json.loads(state_path.read_text())

    class Stop:
        calls = 0

        def check(self):
            self.calls += 1
            if self.calls == 2:
                raise CleanStop("synthetic stop after restore")

    monkeypatch.setattr(evaluation, "evaluate_batch", lambda *a, **k: pytest.fail("batch ran"))
    resumed = evaluate(tmp_path, model, dictionary, resume=True, stop=Stop())
    for key in ("evaluated_tokens", "cross_entropy_tokens", "fvu", "dead_fraction", "cross_entropy_delta"):
        assert resumed[key] == prior[key]
    after = json.loads(state_path.read_text())
    for key in ("count", "pending_ids", "stream", "original_ce", "replaced_ce", "ce_tokens"):
        assert after[key] == before[key]
