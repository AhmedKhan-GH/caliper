"""Published hosted dispatch with local metadata and no weight payload reads."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _dictionary, _hosted, _index, _memory, _qualification as qualification
from caliper.interp._gemma_import_identity import _REFERENCE_INPUT, _REFERENCE_PROFILE
from .test_reference_snapshots import local_snapshots, qualified_model


def reference_manifest():
    return {"format_version": 2, "encoder_policy": "gemma_scope_pre_gain_v1",
            "reference_input": deepcopy(_REFERENCE_INPUT), "dictionary_fingerprint": "fingerprint"}


@pytest.fixture
def loader(monkeypatch, tmp_path, local_snapshots, qualified_model):
    calls = []
    guard = SimpleNamespace(limit=40_000_000_000, device=torch.device("cpu"),
                            check=lambda stage: calls.append(("check", stage)))
    dictionary = SimpleNamespace(metadata={"reference_input": deepcopy(_REFERENCE_INPUT)})
    index = SimpleNamespace(metadata=reference_manifest(), close=lambda: calls.append(("close",)))
    tokenizer = object()
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(_memory, "MemoryGuard", lambda limit, device: guard)
    monkeypatch.setattr(_dictionary, "_read_metadata", lambda *args: (dictionary.metadata, 100))
    def load_model(snapshot, **kwargs):
        calls.append(("model", snapshot, kwargs))
        return qualified_model
    def load_dictionary(folder, **kwargs):
        calls.append(("dictionary", folder, kwargs))
        return dictionary
    def load_index(folder, **kwargs):
        calls.append(("index", folder, kwargs))
        return index
    def verify(folder, metadata, **kwargs):
        calls.append(("verify", folder, metadata, kwargs))
        return {"version": 2, "profile": _REFERENCE_PROFILE}
    monkeypatch.setattr(_hosted, "_load_gemma_model", load_model)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(_dictionary, "load_dictionary", load_dictionary)
    monkeypatch.setattr(_index, "FeatureIndex", load_index)
    monkeypatch.setattr(qualification, "verified_import_binding", verify)
    return SimpleNamespace(calls=calls, guard=guard, dictionary=dictionary, index=index,
        tokenizer=tokenizer, model=qualified_model, folder=tmp_path, snapshots=local_snapshots)


def test_hosted_selects_explicit_arithmetic_and_strict_import_before_index(loader):
    result = _hosted.load_resources(str(loader.snapshots.model), loader.folder,
                                   replication_profile="published_dallas_v1")
    assert result == (loader.model, loader.tokenizer, loader.dictionary, loader.index, loader.guard)
    load = next(row for row in loader.calls if row[0] == "model")
    assert load[1:] == (loader.snapshots.model, {"device": "cpu", "memory": loader.guard,
        "dtype": torch.bfloat16, "reference_profile": _REFERENCE_PROFILE})
    dictionary = next(row for row in loader.calls if row[0] == "dictionary")
    assert dictionary[2]["dtype"] == torch.bfloat16
    assert dictionary[2]["memory"] is loader.guard
    assert dictionary[2]["memory_limit_gb"] == 40
    verify = next(row for row in loader.calls if row[0] == "verify")
    assert verify[3] == {"model": loader.model, "memory": loader.guard, "profile": _REFERENCE_PROFILE,
        "source": loader.snapshots.scope, "model_snapshot": loader.snapshots.model}
    names = [row[0] for row in loader.calls]
    assert names.index("verify") < names.index("index")
    index = next(row for row in loader.calls if row[0] == "index")
    assert index[2]["memory"] is loader.guard and index[2]["memory_limit_gb"] == 40


def test_hosted_accepts_canonical_model_alias(loader, tmp_path):
    alias = tmp_path / "model-alias"
    alias.symlink_to(loader.snapshots.model, target_is_directory=True)
    _hosted.load_resources(str(alias), loader.folder, replication_profile="published_dallas_v1")
    assert next(row for row in loader.calls if row[0] == "model")[1] == loader.snapshots.model


@pytest.mark.parametrize("profile", ["caliper", "circuit_tracer_82996f14", "unknown", False, {}])
def test_unknown_hosted_profile_refuses_before_loads(loader, profile):
    with pytest.raises(ValueError, match="unknown replication profile"):
        _hosted.load_resources(str(loader.snapshots.model), loader.folder, replication_profile=profile)
    assert not loader.calls and not loader.snapshots.calls


@pytest.mark.parametrize("model", ["tiny-lm", "google/gemma-2-2b", "Qwen/Qwen3-0.6B", "missing"])
def test_published_profile_refuses_unsupported_models_before_loads(loader, model):
    with pytest.raises(ValueError, match="local snapshot"):
        _hosted.load_resources(model, loader.folder, replication_profile="published_dallas_v1")
    assert not loader.calls and not loader.snapshots.calls


@pytest.mark.parametrize("declaration", [None, {}, {**_REFERENCE_INPUT, "dtype": "float32"}])
def test_published_profile_refuses_wrong_artifact_before_model_load(loader, declaration):
    loader.dictionary.metadata["reference_input"] = declaration
    with pytest.raises(ValueError, match="original-coordinate artifact"):
        _hosted.load_resources(str(loader.snapshots.model), loader.folder,
                               replication_profile="published_dallas_v1")
    assert not loader.calls and not loader.snapshots.calls


def test_missing_scope_refuses_before_model_load(loader):
    loader.snapshots.scope.rmdir()
    with pytest.raises(ValueError, match="available local snapshot"):
        _hosted.load_resources(str(loader.snapshots.model), loader.folder,
                               replication_profile="published_dallas_v1")
    assert not loader.calls


def test_strict_import_failure_prevents_index_load(loader, monkeypatch):
    def refuse(*args, **kwargs):
        raise ValueError("strict source evidence changed")
    monkeypatch.setattr(qualification, "verified_import_binding", refuse)
    with pytest.raises(ValueError, match="strict source evidence"):
        _hosted.load_resources(str(loader.snapshots.model), loader.folder,
                               replication_profile="published_dallas_v1")
    assert not any(row[0] == "index" for row in loader.calls)


def test_ordinary_index_refusal_closes_loaded_index(loader):
    loader.index.metadata["encoder_policy"] = "live_input_v1"
    with pytest.raises(ValueError, match="original-coordinate index"):
        _hosted.load_resources(str(loader.snapshots.model), loader.folder,
                               replication_profile="published_dallas_v1")
    assert loader.calls[-1] == ("close",)


def test_default_gemma_loader_retains_default_options(loader):
    _hosted.load_resources(str(loader.snapshots.model), loader.folder)
    assert next(row for row in loader.calls if row[0] == "model")[2] == {
        "device": "cpu", "memory": loader.guard}
    assert not any(row[0] == "verify" for row in loader.calls)
    assert not loader.snapshots.calls


@pytest.fixture
def live_qualification(monkeypatch, tmp_path, local_snapshots, qualified_model):
    from caliper.interp._reference_snapshots import remember_model_snapshot
    remember_model_snapshot(qualified_model, local_snapshots.model)
    memory = SimpleNamespace(limit=40_000_000_000, check=lambda *args, **kwargs: None)
    metadata = {"model_hash": _dictionary._GEMMA_IDENTITY, "layers": 0,
                "reference_input": deepcopy(_REFERENCE_INPUT)}
    dictionary = SimpleNamespace(metadata=metadata)
    index = object.__new__(_index.FeatureIndex)
    tokenizer = object()
    manifest = {**reference_manifest(), "context_length": 4, "context_policy": {},
        "example_context_policy": {}, "dictionary_hash": "dictionary", "model_hash": metadata["model_hash"],
        "dictionary_source_hash": "source", "tokenizer_hash": "tokenizer", "files": {}}
    index.root, index.model, index.tokenizer = tmp_path, qualified_model, tokenizer
    index.dictionary, index.memory, index._closed = dictionary, memory, False
    index.close = lambda: setattr(index, "_closed", True)
    (tmp_path / "index").mkdir()
    def write_manifest():
        payload = {key: value for key, value in manifest.items() if key != "index_hash"}
        manifest["index_hash"] = hashlib.sha256(_index._json(payload).encode()).hexdigest()
        index.metadata = deepcopy(manifest)
        (tmp_path / "index/index.json").write_text(json.dumps(manifest))
    write_manifest()
    monkeypatch.setattr(qualification, "model_identity", lambda *args, **kwargs: metadata["model_hash"])
    monkeypatch.setattr(qualification, "evaluation_model_evidence", lambda *args, **kwargs: {"version": 2})
    monkeypatch.setattr(qualification, "context_policy", lambda *args, **kwargs: {})
    monkeypatch.setattr(qualification, "_dictionary_hash", lambda *args, **kwargs: "dictionary")
    monkeypatch.setattr(qualification, "_source_hash", lambda *args, **kwargs: "source")
    monkeypatch.setattr(qualification, "_tokenizer_hash", lambda *args, **kwargs: "tokenizer")
    monkeypatch.setattr(qualification, "_FILES", ())
    monkeypatch.setattr(qualification, "dictionary_fingerprint", lambda *args, **kwargs: "fingerprint")
    monkeypatch.setattr(qualification, "evaluation_dictionary_evidence", lambda *args, **kwargs: {"version": 1})
    monkeypatch.setattr(qualification, "research_evidence", lambda *args, **kwargs: None)
    policies, calls = [], []
    monkeypatch.setattr(qualification, "_encoding_policy", lambda *args: policies.append(args))
    evidence = {"version": 2, "profile": _REFERENCE_PROFILE, "reference_input": deepcopy(_REFERENCE_INPUT),
                "reference_conversion": {"version": 2}, "layers": [{"fixture": "exact validator reply"}]}
    def verify(folder, metadata, **kwargs):
        calls.append((folder, metadata, kwargs))
        return evidence
    monkeypatch.setattr(qualification, "verified_import_binding", verify)
    def response():
        return qualification.dictionary_response(dictionary, index, model=qualified_model, tokenizer=tokenizer)
    return SimpleNamespace(response=response, evidence=evidence, calls=calls, policies=policies,
        manifest=manifest, write_manifest=write_manifest, model=qualified_model, memory=memory,
        snapshots=local_snapshots, dictionary=dictionary, index=index)


def test_live_qualification_returns_exact_version_two_evidence_each_time(live_qualification):
    case = live_qualification
    for _ in range(2):
        assert case.response()["import_binding"] is case.evidence
    assert len(case.calls) == len(case.policies) == 2
    for folder, metadata, arguments in case.calls:
        assert folder == case.index.root and metadata is case.dictionary.metadata
        assert arguments == {"model": case.model, "memory": case.memory, "profile": _REFERENCE_PROFILE,
            "source": case.snapshots.scope, "model_snapshot": case.snapshots.model}


@pytest.mark.parametrize("field,value", [("format_version", 1), ("encoder_policy", "live_input_v1"),
    ("reference_input", None), ("dictionary_fingerprint", "changed")])
def test_reference_live_index_contract_refusal(live_qualification, field, value):
    case = live_qualification
    case.manifest[field] = value
    case.write_manifest()
    with pytest.raises(ValueError, match="reference qualification"):
        case.response()


def test_live_reference_qualification_retains_index_policy_refusal(live_qualification, monkeypatch):
    def refuse(*args):
        raise ValueError("index model encoding differs")
    monkeypatch.setattr(qualification, "_encoding_policy", refuse)
    with pytest.raises(ValueError, match="index model encoding"):
        live_qualification.response()


def test_ordinary_import_dispatch_keeps_default_arguments(monkeypatch, tmp_path):
    calls = []
    model = _hosted.TinyLM()
    memory = object()
    evidence = {"version": 1}
    def verify(*args, **kwargs):
        calls.append((args, kwargs))
        return evidence
    monkeypatch.setattr(qualification, "verified_import_binding", verify)
    assert qualification.qualify_import(tmp_path, {}, model=model, memory=memory) is evidence
    assert calls == [((tmp_path, {}), {"model": model, "memory": memory})]


def test_reference_declaration_cannot_select_profile_for_ordinary_model(tmp_path):
    with pytest.raises(ValueError, match="Gemma import binding"):
        qualification.qualify_import(tmp_path, {"reference_input": _REFERENCE_INPUT},
                                     model=_hosted.TinyLM(), memory=None)


@pytest.mark.parametrize("reference,maximum", [(False, 511), (True, 8191)])
def test_hosted_selection_supports_reference_rows_above_511(monkeypatch, tmp_path, reference, maximum):
    import caliper
    from caliper import _controls, _managed
    from caliper.interp import _replication_actions
    from .fixture import make_fixture
    from .test_dictionary import fixture_manifest
    from .test_intervene import dictionary_for
    model, tokenizer = make_fixture()
    dictionary = dictionary_for(model)
    dictionary.metadata = fixture_manifest(model)
    index = SimpleNamespace(metadata={"dictionary_hash": "fixture", "tokenizer_hash": "fixture"}, close=lambda: None)
    controls = _controls.Controls()
    monkeypatch.setattr(caliper, "_CTL", controls)
    monkeypatch.setattr(_managed, "current", lambda: None)
    monkeypatch.setattr(caliper, "param", controls.param)
    monkeypatch.setattr(caliper, "action", controls.action)
    def watch(*args, **kwargs):
        interventions = controls.interventions_on(model)
        return SimpleNamespace(remove=interventions.remove)
    monkeypatch.setattr(caliper, "watch", watch)
    monkeypatch.setattr(_hosted, "prompt_inputs", lambda *args: {})
    monkeypatch.setattr(_replication_actions, "PublishedCases", lambda *args, **kwargs:
        SimpleNamespace(prepare=lambda: None, close=lambda: None, output_scope=None))
    run = _hosted.HostedInference("tiny-lm", tmp_path, _replication_actions.PROMPTS["dallas"],
        resources=(model, tokenizer, dictionary, index, None),
        replication_profile="published_dallas_v1" if reference else None,
        private_output_directory=tmp_path if reference else None)
    monkeypatch.setattr(run, "boundary", lambda: None)
    try:
        run.start()
        assert run.selected.entry()["max"] == maximum
        run.selected.set(maximum)
        assert run.selected.value == maximum
        if reference:
            node = {"kind": "feature", "layer": 0, "feature": 1, "position": 1}
            run.service.graph = SimpleNamespace(nodes=[node] * (maximum + 1))
            run._selection_graph = run.service.graph
            run._update_selection()
            assert run.selection == {key: node[key] for key in ("layer", "feature", "position")}
    finally:
        run.close()
