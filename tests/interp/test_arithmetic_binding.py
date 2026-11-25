"""Trace caches and evaluation cursors bind effective arithmetic on CPU."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from caliper.interp import _evaluation_identity
from caliper.interp._dictionary import dictionary_fingerprint, model_identity
from caliper.interp._evaluation_identity import _evaluation_settings, _settings_digest, evaluation_model_evidence
from caliper.interp._graph import Graph
from caliper.interp._index import _model_encoding, _encoding_policy
from caliper.interp._trace import Tracer
from tools.interp import eval_dictionary as evaluation
from tools.interp.evaluation_binding import EvaluationBinding
from .fixture import TinyLM, TinyTokenizer
from .test_arithmetic_identity import arithmetic, default_arithmetic
from .test_evaluation_settings import run as run_legacy, settings_fixture
from .test_gemma_import_binding import artifact
from .test_gemma_runtime import gemma_model
from .test_research_evaluation import run as run_research, host_headroom
from .test_trace import exact_dictionary, qwen_for_trace


TRANSITIONS = [("default", "bf16"), ("bf16", "fp16"),
               ("default", "high"), ("high", "medium")]


@pytest.mark.parametrize(("factory", "evaluation_hash", "index_hash"), [
    (gemma_model, "439db547b5ff10fbde702c0c3158d4a88890378e26994a672ea6ba6330c252c2",
     "a9d0efb48f2a6b4cef5c3be0c6d781f91e2bbd70a390042a2f271a73d2c33575"),
    (qwen_for_trace, "8a60057bed25559793f333525ea9b94bc6a94a924b7784fbfc8c221cbb0f34d6",
     "662c457d577c39fc8edbe3c31919ba34442f78533c30889cd0d6ddaa956fb79f"),
])
def test_default_evaluation_and_index_hashes_match_legacy(factory, evaluation_hash, index_hash):
    # Exact parent-revision hashes protect default artifacts without a migration.
    model = factory()
    assert _settings_digest(model, _evaluation_settings(model, None)) == evaluation_hash
    encoding = _model_encoding(model, None)
    assert encoding["forward_settings_hash"] == evaluation_hash
    assert hashlib.sha256(json.dumps(encoding, sort_keys=True).encode()).hexdigest() == index_hash
    manifest = {"format_version": 2, "encoder_policy": "live_input_v1", "model_encoding": encoding}
    dictionary = SimpleNamespace(metadata={}, layers=[])
    saved = deepcopy(manifest)
    _encoding_policy(manifest, model, dictionary, None)
    with arithmetic("bf16"), pytest.raises(ValueError, match="model encoding differs"):
        _encoding_policy(manifest, model, dictionary, None)
    _encoding_policy(manifest, model, dictionary, None)
    assert manifest == saved


@pytest.mark.parametrize(("initial", "changed"), TRANSITIONS)
def test_trace_context_change_clears_caches_before_forward_or_publication(monkeypatch, initial, changed):
    monkeypatch.setattr(Graph, "publish", lambda graph: graph)
    model = TinyLM(layers=2, width=8, heads=2, latents=8, k=2).eval()
    dictionary = exact_dictionary(model)
    with arithmetic(initial):
        tracer = Tracer(model, TinyTokenizer(), dictionary)
        first = tracer.trace("AB", 0)
        assert tracer.trace("AB", 0) is first
        assert tracer._cache and tracer._forward_cache is not None
        with monkeypatch.context() as blocked, arithmetic(changed):
            blocked.setattr(model, "forward", lambda *a, **kw: pytest.fail("changed context reached forward"))
            blocked.setattr(Graph, "publish", lambda *a: pytest.fail("changed context reached publication"))
            with pytest.raises(ValueError, match="forward setting changed.*__arithmetic_context__"):
                tracer.trace("AB", 0)
            assert not tracer._cache and tracer._forward_cache is None
        restored = tracer.trace("AB", 0)
        assert restored is not first
        assert restored.nodes == first.nodes and restored.edges == first.edges
        assert restored.trace_id == first.trace_id


def test_separate_trace_admissions_have_distinct_arithmetic_identities(monkeypatch):
    monkeypatch.setattr(Graph, "publish", lambda graph: graph)
    model = TinyLM(layers=2, width=8, heads=2, latents=8, k=2).eval()
    dictionary, tokenizer = exact_dictionary(model), TinyTokenizer()
    first = Tracer(model, tokenizer, dictionary).trace("AB", 0)
    with arithmetic("high"):
        second = Tracer(model, tokenizer, dictionary).trace("AB", 0)
    assert second.trace_id != first.trace_id
    restored = Tracer(model, tokenizer, dictionary).trace("AB", 0)
    assert restored.trace_id == first.trace_id


@pytest.mark.parametrize(("initial", "changed"), TRANSITIONS)
def test_evaluation_binding_and_evidence_reject_arithmetic_changes(artifact, initial, changed):
    folder, model, dictionary, _ = artifact
    identity = dictionary_fingerprint(folder, dictionary.metadata)
    with arithmetic(initial):
        binding = EvaluationBinding(folder, dictionary, identity, model=model)
        original = evaluation_model_evidence(model, model_identity(model))
        with arithmetic(changed):
            current = evaluation_model_evidence(model, model_identity(model))
            assert current["model_hash"] == original["model_hash"]
            assert current["tensor_hash"] == original["tensor_hash"]
            assert current["forward_settings_hash"] != original["forward_settings_hash"]
            with pytest.raises(ValueError, match="forward setting changed.*__arithmetic_context__"):
                binding.check()
            with pytest.raises(ValueError, match="forward setting changed.*__arithmetic_context__"):
                binding.verify()
        binding.verify()
        assert evaluation_model_evidence(model, model_identity(model)) == original


@pytest.mark.parametrize(("initial", "changed"), TRANSITIONS + [("bf16", "default")])
def test_supplied_evaluation_snapshot_cannot_relabel_changed_arithmetic(monkeypatch, initial, changed):
    model = gemma_model()
    with arithmetic(initial):
        settings = _evaluation_settings(model, None)
        saved = deepcopy(settings)
        with arithmetic(changed):
            monkeypatch.setattr(_evaluation_identity, "_hf_digest", lambda *a, **kw: pytest.fail("tensor readback ran"))
            with pytest.raises(ValueError, match="arithmetic context changed"):
                evaluation_model_evidence(model, "fixture", settings=settings)
        assert settings == saved


@pytest.mark.parametrize(("initial", "changed"), TRANSITIONS)
def test_research_resume_refuses_context_change_before_forward_or_stats_load(artifact, monkeypatch, initial, changed):
    folder, model, dictionary, _ = artifact
    with arithmetic(initial):
        original = run_research(folder, model, dictionary)
        directory = folder / "evaluation-research-v1"
        saved = {path.name: path.read_bytes() for path in directory.iterdir()}
        metadata = (folder / "dictionary.json").read_bytes()
        with monkeypatch.context() as blocked, arithmetic(changed):
            blocked.setattr(evaluation, "hidden_forward", lambda *a, **kw: pytest.fail("resume forward ran"))
            blocked.setattr(evaluation, "load_file", lambda *a, **kw: pytest.fail("statistics restore ran"))
            blocked.setattr(evaluation, "_validate_tensor_headers", lambda *a, **kw: pytest.fail("statistics opened"))
            with pytest.raises(ValueError, match="changed across resume"):
                run_research(folder, model, dictionary, resume=True)
        assert {path.name: path.read_bytes() for path in directory.iterdir()} == saved
        assert (folder / "dictionary.json").read_bytes() == metadata
        restored = run_research(folder, model, dictionary, resume=True)
        assert restored["model_binding"] == original["model_binding"]
        assert restored["source_tokens"] == original["source_tokens"]
        assert restored["per_layer_fvu"] == original["per_layer_fvu"]


@pytest.mark.parametrize("mode", ["bf16", "high"])
def test_legacy_resume_refuses_context_change_and_accepts_restoration(tmp_path, monkeypatch, mode):
    model, dictionary = settings_fixture(tmp_path, "tiny")
    original = run_legacy(tmp_path, model, dictionary, interrupt=True)
    assert original["evaluated_tokens"] == 4
    directory = tmp_path / "evaluation"
    saved = {path.name: path.read_bytes() for path in directory.iterdir()}
    with monkeypatch.context() as blocked, arithmetic(mode):
        blocked.setattr(evaluation, "hidden_forward", lambda *a, **kw: pytest.fail("resume forward ran"))
        blocked.setattr(evaluation, "load_file", lambda *a, **kw: pytest.fail("statistics restore ran"))
        with pytest.raises(ValueError, match="cursor binding"):
            run_legacy(tmp_path, model, dictionary, resume=True)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == saved
    restored = run_legacy(tmp_path, model, dictionary, resume=True)
    assert restored["evaluated_tokens"] == 8
    assert restored["model_binding"] == original["model_binding"]
