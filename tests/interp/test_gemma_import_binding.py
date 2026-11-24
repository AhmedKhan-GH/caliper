"""Import evidence checks with small local random Gemma checkpoints."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from caliper.interp import load_dictionary
from caliper.interp import _gemma_import_identity as imported
from caliper.interp._dictionary import _state_hash, dictionary_fingerprint, verify_hf_model
from caliper.interp._index import FeatureIndex, build_index
from caliper.interp._qualification import dictionary_response
from caliper.interp._service import InterpretationService
from demos.circuit_trace._qualification import dictionary_qualification
from tools.interp import convert_gemma_scope, eval_dictionary
from tools.interp.evaluation_binding import EvaluationBinding
from tools.interp.text_stream import TextStream
from .test_gemma_evaluation import imported_metadata
from .test_gemma_runtime import gemma_model, local_gemma, runtime_dictionary
from .test_training_stream import documents


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    # Only the dimensions shrink. Model verification uses actual checkpoint bytes.
    monkeypatch.setattr(imported, "_WIDTH", 16)
    monkeypatch.setattr(imported, "_LATENTS", 20)
    monkeypatch.setattr(imported, "_L0", (76, 65))
    model = gemma_model()
    with torch.no_grad():
        for block in model.model.layers:
            block.pre_feedforward_layernorm.weight.copy_(torch.linspace(-.3, .7, 16))
    snapshot = local_gemma(tmp_path, model)
    verify_hf_model(model, snapshot)
    dictionary = runtime_dictionary(model)
    folder = tmp_path / "dictionary"
    folder.mkdir()
    rows = []
    for i, layer in enumerate(dictionary.layers):
        filename = f"layer_{i:02d}.safetensors"
        save_file(layer.state_dict(), folder / filename)
        gain = 1 + model.model.layers[i].pre_feedforward_layernorm.weight.detach()
        rows.append({"layer": i, "width": 16, "latents": 20, "average_l0": imported._L0[i],
                     "file": filename, "sha256": hashlib.sha256((folder / filename).read_bytes()).hexdigest(),
                     "input_gain_hash": _state_hash({"gain": gain}), "source_sha256": "a" * 64,
                     "source_file": f"layer_{i}/width_16k/average_l0_{imported._L0[i]}/params.npz"})
    metadata = convert_gemma_scope._metadata(rows, imported_metadata()["evaluation_source"])
    metadata.update(dictionary.metadata)
    metadata["training"] = convert_gemma_scope._metadata(rows)["training"]
    (folder / "dictionary.json").write_text(json.dumps(metadata))
    return folder, model, load_dictionary(folder, model=model), snapshot


def test_import_evidence_uses_actual_files_and_one_plus_verified_gain(artifact):
    folder, model, dictionary, _ = artifact
    result = imported.verified_import_binding(folder, dictionary.metadata, model=model)
    assert result == {"version": 1, "model_hash": dictionary.metadata["model_hash"], "layers": [
        {key: row[key] for key in ("layer", "file", "sha256", "input_gain_hash")}
        for row in dictionary.metadata["import_source"]["layers"]]}
    binding = EvaluationBinding(folder, dictionary, dictionary_fingerprint(folder, dictionary.metadata), model=model)
    assert binding.import_evidence == result
    binding.verify()


@pytest.fixture
def random_artifact(tmp_path):
    # No checkpoint verification or pinned import dimension overrides apply here.
    model = gemma_model()
    dictionary = runtime_dictionary(model)
    dictionary.metadata.update(
        format_version=1,
        training={"dataset_id": "fixture", "dataset_config": "random-inputs",
                  "dataset_revision": "seed-51", "split": "train", "partitions": {},
                  "tokens": 0, "elapsed_seconds": 0, "peak_memory_bytes": 0},
        quality={"fvu": None, "dead_fraction": None, "cross_entropy_delta": None, "evaluated_tokens": 0})
    for i, layer in enumerate(dictionary.layers):
        save_file(layer.state_dict(), tmp_path / f"layer_{i:02d}.safetensors")
    (tmp_path / "dictionary.json").write_text(json.dumps(dictionary.metadata))
    return tmp_path, model, load_dictionary(tmp_path, model=model)


@pytest.mark.parametrize("lazy", [False, True])
def test_random_gemma_dictionary_inspection_uses_ordinary_service(random_artifact, lazy):
    folder, model, dictionary = random_artifact
    assert dictionary.metadata["model_hash"].startswith("sha256:")
    assert imported.verified_import_binding(folder, dictionary.metadata, model=model) is None
    tokenizer = SimpleNamespace(eos_token_id=2, convert_ids_to_tokens=lambda value: str(value))
    build_index(folder, [("fixture", [3, 4, 5, 6])], model=model, tokenizer=tokenizer,
                dictionary=dictionary, provenance={"partition": "fixture"}, context_length=4)
    with FeatureIndex(folder, model=model, tokenizer=tokenizer, dictionary=dictionary) as index:
        service = InterpretationService(**({"index_factory": lambda: index} if lazy else {"index": index}))
        replies = []
        try:
            service.submit({"version": 1, "method": "feature.dictionary", "params": {}},
                           lambda *args: replies.append(args))
            assert service.step() == 1
            status, result, error = replies[0]
            assert status == "ok", error
            assert result["identity"]["model_hash"] == dictionary.metadata["model_hash"]
            assert result["identity"]["index_hash"] == index.metadata["index_hash"]
            assert "import_binding" not in result
            assert not dictionary_qualification(result, family="gemma")["qualified"]
        finally:
            service.close()


@pytest.mark.parametrize("claim", ["identity", "import"])
def test_random_gemma_cannot_claim_pinned_import(random_artifact, claim):
    folder, model, dictionary = random_artifact
    metadata = deepcopy(dictionary.metadata)
    if claim == "identity":
        metadata["model_hash"] = imported._GEMMA_IDENTITY
    else:
        metadata["import_source"] = {"kind": "gemma_scope"}
    with pytest.raises(ValueError, match="import binding"):
        imported.verified_import_binding(folder, metadata, model=model)


@pytest.mark.parametrize("declaration", ["missing", "changed"])
@pytest.mark.parametrize("identity", ["pinned", "content", "missing"])
def test_verified_pinned_model_cannot_skip_import_checks(artifact, declaration, identity):
    folder, model, dictionary, _ = artifact
    metadata = deepcopy(dictionary.metadata)
    if declaration == "missing":
        del metadata["import_source"]
    else:
        metadata["import_source"]["kind"] = "unchecked"
    if identity == "content":
        metadata["model_hash"] = "sha256:" + "a" * 64
    elif identity == "missing":
        del metadata["model_hash"]
    with pytest.raises(ValueError, match="import binding"):
        imported.verified_import_binding(folder, metadata, model=model)


def corrupt(folder, dictionary, fault):
    metadata = dictionary.metadata
    if fault == "file":
        path = folder / "layer_00.safetensors"
        weights = {key: value.clone() for key, value in load_file(path).items()}
        weights["W_dec"][0, 0] += .25
        save_file(weights, path)
    elif fault in ("sha256", "input_gain_hash"):
        metadata["import_source"]["layers"][0][fault] = ("sha256:" if fault == "input_gain_hash" else "") + "b" * 64
    elif fault == "missing":
        del metadata["import_source"]
    elif fault == "kind":
        metadata["import_source"]["kind"] = "unchecked"
    else:
        metadata["import_source"]["layers"][0]["file"] = "../layer_00.safetensors"
    (folder / "dictionary.json").write_text(json.dumps(metadata))


@pytest.mark.parametrize("fault", ["file", "sha256", "input_gain_hash", "missing", "kind", "path"])
def test_import_refusal_precedes_quality_batches(artifact, monkeypatch, fault):
    folder, model, dictionary, _ = artifact
    corrupt(folder, dictionary, fault)
    dictionary = load_dictionary(folder, model=model)
    before = (folder / "dictionary.json").read_bytes()
    monkeypatch.setattr(eval_dictionary, "evaluate_batch", lambda *a, **k: pytest.fail("quality batch ran"))
    tokenizer = SimpleNamespace(encode=lambda text, **kw: [3 + ord(ch) % 34 for ch in text], eos_token_id=2)
    with TextStream(documents(), tokenizer, "quality") as stream:
        with pytest.raises(ValueError, match="import binding|import_source"):
            eval_dictionary.evaluate(model, dictionary, stream, folder, tokens=8, context=4, batch_tokens=8)
    assert (folder / "dictionary.json").read_bytes() == before
    assert not (folder / "evaluation").exists()


@pytest.mark.parametrize("fault", [None, "file", "sha256", "input_gain_hash", "missing", "kind"])
def test_live_import_check_survives_fresh_quality_and_index(artifact, fault):
    folder, model, dictionary, _ = artifact
    if fault is not None:
        corrupt(folder, dictionary, fault)
    dictionary = load_dictionary(folder, model=model)
    dictionary.metadata["quality"].update(status="passed", fvu=.1, dead_fraction=.1, cross_entropy_delta=.1,
                                           dictionary_hash=dictionary_fingerprint(folder, dictionary.metadata))
    (folder / "dictionary.json").write_text(json.dumps(dictionary.metadata))
    tokenizer = SimpleNamespace(eos_token_id=2, convert_ids_to_tokens=lambda value: str(value))
    build_index(folder, [("fixture", [3, 4, 5, 6])], model=model, tokenizer=tokenizer,
                dictionary=dictionary, provenance={"partition": "fixture"}, context_length=4)
    with FeatureIndex(folder, model=model, tokenizer=tokenizer, dictionary=dictionary) as index:
        if fault is None:
            result = dictionary_response(dictionary, index, model=model, tokenizer=tokenizer)
            assert result["import_binding"] == imported.verified_import_binding(folder, dictionary.metadata, model=model)
        else:
            with pytest.raises(ValueError, match="import binding"):
                dictionary_response(dictionary, index, model=model, tokenizer=tokenizer)


def test_reforged_gain_cannot_rebaseline_an_unverified_model(artifact):
    folder, model, dictionary, _ = artifact
    weight = model.model.layers[0].pre_feedforward_layernorm.weight
    version = weight._version
    weight.data.add_(.125)
    assert weight._version == version
    dictionary.metadata["import_source"]["layers"][0]["input_gain_hash"] = _state_hash({"gain": 1 + weight.detach()})
    with pytest.raises(ValueError, match="identity mismatch"):
        imported.verified_import_binding(folder, dictionary.metadata, model=model)


def test_metadata_cannot_supply_the_verified_gain(artifact):
    folder, model, dictionary, _ = artifact
    metadata = deepcopy(dictionary.metadata)
    metadata["import_source"]["layers"][0]["input_gain_hash"] = _state_hash(
        {"gain": model.model.layers[0].pre_feedforward_layernorm.weight.detach()})
    with pytest.raises(ValueError, match="import binding"):
        imported.verified_import_binding(folder, metadata, model=model)


@pytest.mark.parametrize("reforge", [False, True])
def test_reduced_precision_gain_has_no_dtype_alias(artifact, reforge):
    folder, model, dictionary, snapshot = artifact
    model.half()
    verify_hf_model(model, snapshot)
    if reforge:
        for row, block in zip(dictionary.metadata["import_source"]["layers"], model.model.layers):
            row["input_gain_hash"] = _state_hash({"gain": 1 + block.pre_feedforward_layernorm.weight.detach()})
    with pytest.raises(ValueError, match="import binding"):
        imported.verified_import_binding(folder, dictionary.metadata, model=model)


def test_import_file_reads_remain_bounded(artifact, monkeypatch):
    folder, model, dictionary, _ = artifact
    monkeypatch.setattr(imported, "_CHUNK_BYTES", 32)
    original = Path.open
    reads = []

    class Reader:
        def __init__(self, source):
            self.source = source

        def __enter__(self):
            self.source.__enter__()
            return self

        def __exit__(self, *args):
            return self.source.__exit__(*args)

        def fileno(self):
            return self.source.fileno()

        def read(self, size):
            assert 0 < size <= 32
            reads.append(size)
            return self.source.read(size)

    def bounded(path, *args, **kwargs):
        source = original(path, *args, **kwargs)
        return Reader(source) if path.suffix == ".safetensors" else source

    monkeypatch.setattr(Path, "open", bounded)
    imported.verified_import_binding(folder, dictionary.metadata, model=model)
    assert len(reads) > 2
