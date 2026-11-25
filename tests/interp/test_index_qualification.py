"""Reference index admission with tiny local source-verifier fixtures."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from caliper.interp import _index, _qualification, _reference_snapshots
from caliper.interp._dictionary import _VERIFIED_MODELS, _state_hash, load_dictionary, model_identity, verify_hf_model
from caliper.interp._gemma_reference import PROFILE, install_reference_profile
from caliper.interp._memory import CleanStop
from tools.interp import build_index as command, model_load
from tools.interp.text_stream import DATASET_CONFIG, DATASET_ID, DATASET_REVISION, partition_metadata
from .cpu_pressure import deterministic_cpu_pressure
from .test_gemma_reference_artifact import reference_artifact
from .test_gemma_runtime import memory


@pytest.fixture
def qualified_index(reference_artifact, monkeypatch):
    source, folder, snapshot, model, metadata = reference_artifact
    install_reference_profile(model, PROFILE)
    verify_hf_model(model, snapshot)
    _reference_snapshots.remember_model_snapshot(model, snapshot)
    lookups = []

    def lookup(**kwargs):
        lookups.append(kwargs)
        assert kwargs == {"repo_id": _reference_snapshots._SOURCE["repo"],
            "revision": _reference_snapshots._SOURCE["revision"],
            "allow_patterns": _reference_snapshots._SCOPE_FILES, "local_files_only": True}
        return str(source)

    monkeypatch.setattr("huggingface_hub.snapshot_download", lookup)
    tokenizer = SimpleNamespace(bos_token_id=1, pad_token_id=0, eos_token_id=2,
        convert_ids_to_tokens=lambda i: str(i), decode=lambda ids, **kwargs: " ".join(map(str, ids)),
        encode=lambda text, **kwargs: [3, 4, 5])
    return SimpleNamespace(root=folder, source=source, snapshot=snapshot, model=model,
        tokenizer=tokenizer, dictionary=load_dictionary(folder, model=model), lookups=lookups)


def build(case, **kwargs):
    options = dict(model=case.model, tokenizer=case.tokenizer, dictionary=case.dictionary,
                   provenance={"partition": "fixture"}, context_length=4, memory=memory())
    options.update(kwargs)
    return _index.build_index(case.root, [("a", [3, 4, 5])], **options)


def observe_verifier(case, monkeypatch):
    calls = []
    verify = _qualification.verified_import_binding

    def observed(folder, metadata, **kwargs):
        assert folder == case.root and metadata == case.dictionary.metadata
        assert kwargs["model"] is case.model
        assert kwargs["profile"] == PROFILE
        assert kwargs["source"] == case.source and kwargs["model_snapshot"] == case.snapshot
        assert not (case.root / "index").exists()
        calls.append(kwargs)
        result = verify(folder, metadata, **kwargs)
        assert result["version"] == 2 and result["profile"] == PROFILE
        calls[-1]["evidence"] = result
        return result

    monkeypatch.setattr(_qualification, "verified_import_binding", observed)
    return calls


def refuse_work(case, monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("model batch preceded reference qualification")
    monkeypatch.setattr(_index, "_forward", refuse)
    monkeypatch.setattr(_index, "_TopTable", lambda *args: pytest.fail("index table preceded qualification"))


def assert_no_output(case):
    assert not (case.root / "index").exists()
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in case.model.modules())


def alter_artifact(case, fault):
    metadata = case.dictionary.metadata
    row = metadata["import_source"]["layers"][0]
    if fault == "original_encoder":
        path = case.root / row["file"]
        tensors = load_file(path)
        tensors["W_enc_original"].view(-1)[0] += .25
        save_file(tensors, path)
        # Fresh file and tensor hashes must not authorize a different conversion.
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        row["tensor_hashes"] = {name: _state_hash({name: tensor}) for name, tensor in tensors.items()}
    else:
        row[fault] = "0" * 64
    (case.root / "dictionary.json").write_text(json.dumps(metadata))
    case.dictionary = load_dictionary(case.root, model=case.model)


def prepare_cli(case, monkeypatch, guard=None):
    import huggingface_hub
    import transformers

    metadata = case.dictionary.metadata
    metadata["evaluation_source"] = {
        "dataset_id": DATASET_ID, "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION, "split": "train", "partitions": partition_metadata(),
        "source_id": f"{DATASET_ID}/{DATASET_CONFIG}@{DATASET_REVISION}:train"}
    (case.root / "dictionary.json").write_text(json.dumps(metadata))
    events = []

    def documents():
        events.append("documents")
        yield from ({"text": f"Reference index fixture {i}."} for i in range(100))

    source_lookup = huggingface_hub.snapshot_download

    def lookup(*args, **kwargs):
        if args:
            assert args == (model_load.GEMMA_ID,)
            assert kwargs == {"revision": model_load.GEMMA_REVISION, "local_files_only": True}
            return str(case.snapshot)
        return source_lookup(**kwargs)

    def load(folder, **kwargs):
        assert folder == case.snapshot and kwargs["reference_profile"] == PROFILE
        assert kwargs["device"] == "cpu" and kwargs["dtype"] == torch.bfloat16
        events.append("loader")
        return case.model

    def tokenize(folder, **kwargs):
        assert folder == case.snapshot
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        assert _reference_snapshots._MODEL_SNAPSHOTS[case.model] == case.snapshot
        return case.tokenizer

    # Only the real CLI loader can restore the registry entry for this model.
    _reference_snapshots._MODEL_SNAPSHOTS.pop(case.model)
    case.model.config._name_or_path = model_load.GEMMA_ID
    case.model.config._commit_hash = model_load.GEMMA_REVISION
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lookup)
    monkeypatch.setattr(model_load, "_load_gemma_model", load)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenize)
    monkeypatch.setattr(command, "fineweb_documents", lambda **kwargs: documents())
    monkeypatch.setattr(command, "MemoryGuard", lambda limit, device: guard if guard is not None else memory(limit))
    args = ["--dictionary", str(case.root), "--device", "cpu", "--model", model_load.GEMMA_ID,
            "--profile", PROFILE, "--context", "4", "--tokens", "3"]
    return args, events


@pytest.mark.parametrize("route", ["library", "library_load", "cli"])
@pytest.mark.parametrize("fault", ["input_gain_hash", "source_sha256", "source_gain_hash", "original_encoder"])
def test_reference_source_refusal_precedes_output_and_batches(qualified_index, monkeypatch, route, fault):
    case = qualified_index
    alter_artifact(case, fault)
    calls = observe_verifier(case, monkeypatch)
    refuse_work(case, monkeypatch)
    if route == "cli":
        args, events = prepare_cli(case, monkeypatch)
    with pytest.raises(ValueError, match="import binding"):
        if route == "cli":
            command.main(args)
        else:
            build(case, dictionary=None if route == "library_load" else case.dictionary)
    assert len(calls) == 1 and "evidence" not in calls[0]
    assert len(case.lookups) == 1
    if route == "cli":
        assert events == ["loader"]
    assert_no_output(case)


@pytest.mark.parametrize("route", ["library", "library_load", "cli"])
def test_valid_reference_qualifies_exact_sources_before_first_batch(qualified_index, monkeypatch, route):
    case = qualified_index
    calls = observe_verifier(case, monkeypatch)
    batches = []
    forward = _index._forward

    def observed_forward(*args):
        assert len(calls) == 1 and "evidence" in calls[0]
        batches.append(1)
        return forward(*args)

    monkeypatch.setattr(_index, "_forward", observed_forward)
    if route == "cli":
        args, events = prepare_cli(case, monkeypatch)
        assert command.main([*args, "--memory-gb", "2"]) == 0
        assert events == ["loader", "documents"]
        assert calls[0]["memory"].limit == 2_000_000_000
        manifest = json.loads((case.root / "index/index.json").read_text())
    else:
        manifest = build(case, dictionary=None if route == "library_load" else case.dictionary)
    assert batches == [1] and len(case.lookups) == 1
    assert manifest["format_version"] == 2 and manifest["encoder_policy"] == _index.REFERENCE_ENCODER_POLICY
    assert manifest["tokens"] == 3 and manifest["complete"] is True
    with _index.FeatureIndex(case.root, model=case.model, tokenizer=case.tokenizer,
                             dictionary=case.dictionary) as index:
        assert index.metadata == manifest
    assert len(calls) == 1


@pytest.mark.parametrize("tensor", ["W_enc_original", "W_enc", "W_dec", "threshold"])
def test_live_reference_tensors_must_match_qualified_artifact(qualified_index, monkeypatch, tensor):
    case = qualified_index
    getattr(case.dictionary.layers[0], tensor).data.view(-1)[0] += .25
    calls = observe_verifier(case, monkeypatch)
    refuse_work(case, monkeypatch)
    with pytest.raises(ValueError, match="artifact binding differs"):
        build(case)
    assert len(calls) == 1 and "evidence" in calls[0]
    assert_no_output(case)


@pytest.mark.parametrize("stage", ["scope_import_admission", "scope_reference_tensor", "evaluation_binding"])
@pytest.mark.parametrize("fault", ["cancel", "lower"])
def test_qualification_keeps_cancellation_and_dynamic_lower_budget(qualified_index, monkeypatch, stage, fault):
    case = qualified_index
    guard = memory(2)
    visits = []

    def check(current):
        if current == stage:
            visits.append(current)
            if fault == "cancel":
                raise CleanStop("fixture cancellation")
            guard.limit = 1

    refuse_work(case, monkeypatch)
    with pytest.raises(CleanStop, match="cancellation|memory limit"):
        build(case, memory=guard, memory_guard=check)
    assert visits
    assert_no_output(case)


@pytest.mark.parametrize("fault", ["cancel", "lower"])
def test_cli_stops_inside_source_qualification(qualified_index, monkeypatch, capsys, fault):
    case = qualified_index
    guard = memory(2)
    stopped = False
    visits = []
    original_check = guard.check

    def check(stage, **kwargs):
        nonlocal stopped
        if stage == "scope_reference_tensor":
            visits.append(stage)
            if fault == "lower":
                guard.limit = 1
            else:
                stopped = True
        return original_check(stage, **kwargs)

    class Stop:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def check(self):
            if stopped:
                raise CleanStop("fixture cancellation")

    monkeypatch.setattr(guard, "check", check)
    monkeypatch.setattr(command, "StopControl", Stop)
    args, events = prepare_cli(case, monkeypatch, guard)
    calls = observe_verifier(case, monkeypatch)
    refuse_work(case, monkeypatch)
    assert command.main([*args, "--memory-gb", "2"]) == 2
    assert visits and events == ["loader"] and len(calls) == 1
    assert "evidence" not in calls[0]
    output = capsys.readouterr().out
    expected = "cancellation" if fault == "cancel" else "memory limit"
    assert expected in output
    assert_no_output(case)


@pytest.mark.parametrize("missing", ["model", "scope"])
def test_reference_requires_available_trusted_snapshots(qualified_index, monkeypatch, missing):
    case = qualified_index
    if missing == "model":
        _reference_snapshots._MODEL_SNAPSHOTS.pop(case.model)
        case.model.config._name_or_path = model_load.GEMMA_ID
        case.model.config._commit_hash = model_load.GEMMA_REVISION
    else:
        monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: str(case.source / "absent"))
    refuse_work(case, monkeypatch)
    with pytest.raises(ValueError, match="local snapshot"):
        build(case)
    assert_no_output(case)


def test_structural_reference_fixture_cannot_bypass_verified_model(qualified_index, monkeypatch):
    case = qualified_index
    del _VERIFIED_MODELS[case.model]
    case.dictionary.metadata["model_hash"] = model_identity(case.model)
    refuse_work(case, monkeypatch)
    with pytest.raises(ValueError, match="verified pinned model"):
        build(case)
    assert not case.lookups
    assert_no_output(case)


@pytest.mark.parametrize("fault", ["source", "partition"])
def test_cli_preserves_source_admission_before_model_load(qualified_index, monkeypatch, fault):
    case = qualified_index
    args, events = prepare_cli(case, monkeypatch)
    metadata = case.dictionary.metadata
    source = metadata["evaluation_source"]
    if fault == "source":
        source["source_id"] = "different-source"
    else:
        source["partitions"] = {}
    (case.root / "dictionary.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="source|partitions"):
        command.main(args)
    assert not events and not case.lookups
    assert_no_output(case)
