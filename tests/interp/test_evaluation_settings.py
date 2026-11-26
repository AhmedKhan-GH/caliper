"""CPU fixture checks for persistent and runtime evaluation settings."""
import copy
import json

import pytest
import torch
from safetensors.torch import save_file

from caliper.interp import load_dictionary
from caliper.interp._dictionary import model_identity
from tools.interp import eval_dictionary as evaluation
from tools.interp.checkpoint_state import dictionary_fingerprint, write_json
from tools.interp.memory_limits import CleanStop
from tools.interp.text_stream import TextStream
from .fixture import TinyTokenizer
from .test_dictionary import memory_readers
from .test_evaluation_binding import fixture, publications
from .test_training_stream import documents


def settings_fixture(folder, family):
    assert family == "tiny"
    return fixture(folder)


def run(folder, model, dictionary, *, interrupt=False, **kwargs):
    class Tokenizer(TinyTokenizer):
        # Keep the source EOS inside both fixture vocabularies, including saved suffixes.
        eos_token_id = 0

        def encode(self, *args, **kwargs):
            return list(range(1, 9))

    class Stream(TextStream):
        def next_tokens(self, *args, **kwargs):
            if interrupt and self.emitted == 4:
                raise CleanStop("fixture batch complete")
            return super().next_tokens(*args, **kwargs)

    return evaluation.evaluate(model, dictionary, Stream(documents(), Tokenizer(), "quality"),
                               folder, tokens=8, context=4, batch_tokens=4, chunk_tokens=2, **kwargs)


@pytest.mark.parametrize("family", ["tiny"])
def test_epsilon_resume_refuses_mixed_forward_statistics(tmp_path, monkeypatch, family):
    model, dictionary = settings_fixture(tmp_path, family)
    partial = run(tmp_path, model, dictionary, interrupt=True)
    assert partial["evaluated_tokens"] == 4
    cursor = json.loads((tmp_path / "evaluation" / "state.json").read_text())
    assert cursor["model_binding"] == partial["model_binding"]
    assert cursor["model_binding"]["version"] == 2
    assert len(cursor["model_binding"]["forward_settings_hash"]) == 64
    identity = dictionary_fingerprint(tmp_path, dictionary.metadata)
    model_hash = model_identity(model)
    norm = model.norm if family == "tiny" else model.model.norm
    with torch.no_grad():
        ids = torch.tensor([[1, 2, 3, 4]])
        before_logits = model(ids).logits
        norm.eps = 1.0
        assert not torch.equal(before_logits, model(ids).logits)
    assert (model_identity(model) == model_hash) is (family == "tiny")
    assert dictionary_fingerprint(tmp_path, dictionary.metadata) == identity
    before = publications(tmp_path)
    monkeypatch.setattr(evaluation, "load_file", lambda *a, **k: pytest.fail("statistics restore ran"))
    with pytest.raises(ValueError, match="cursor binding" if family == "tiny" else "model identity mismatch"):
        run(tmp_path, model, dictionary, resume=True)
    assert publications(tmp_path) == before


@pytest.mark.parametrize("family", ["tiny"])
def test_equivalent_model_instance_resumes_forward_evidence(tmp_path, family):
    model, dictionary = settings_fixture(tmp_path, family)
    identity = dictionary_fingerprint(tmp_path, dictionary.metadata)
    partial = run(tmp_path, model, dictionary, interrupt=True)
    other = copy.deepcopy(model)
    resumed = run(tmp_path, other, dictionary, resume=True)
    assert resumed["evaluated_tokens"] == 8
    assert resumed["model_binding"] == partial["model_binding"]
    assert dictionary_fingerprint(tmp_path, dictionary.metadata) == identity


@pytest.mark.parametrize("family", ["tiny"])
@pytest.mark.parametrize("forward", [1, 2])
def test_epsilon_change_during_actual_forward_refuses_publication(tmp_path, monkeypatch, family, forward):
    model, dictionary = settings_fixture(tmp_path, family)
    before = publications(tmp_path)
    implementation = evaluation.hidden_forward
    calls = 0

    def changed(*args, **kwargs):
        nonlocal calls
        result = implementation(*args, **kwargs)
        calls += 1
        if calls == forward:
            (model.norm if family == "tiny" else model.model.norm).eps = 1.0
        return result

    monkeypatch.setattr(evaluation, "hidden_forward", changed)
    with pytest.raises(ValueError, match=r"forward setting changed.*norm.eps"):
        run(tmp_path, model, dictionary)
    assert calls == forward
    assert publications(tmp_path) == before


def test_change_between_forwards_refuses_replacement(tmp_path, monkeypatch):
    model, dictionary = fixture(tmp_path)
    before = publications(tmp_path)
    entropy = evaluation.cross_entropy_sum
    calls = 0

    def changed(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = entropy(*args, **kwargs)
        model.norm.eps = 1.0
        return result

    monkeypatch.setattr(evaluation, "cross_entropy_sum", changed)
    with pytest.raises(ValueError, match=r"forward setting changed.*norm.eps"):
        run(tmp_path, model, dictionary)
    assert calls == 1
    assert publications(tmp_path) == before


def test_plain_tensor_forward_override_binds_bytes(tmp_path, monkeypatch):
    model, dictionary = fixture(tmp_path)
    model.norm.frozen_scale = torch.ones(1, 4, 1, dtype=torch.float64)
    run(tmp_path, model, dictionary, interrupt=True)
    version = model.norm.frozen_scale._version
    model.norm.frozen_scale.data.mul_(.5)
    assert model.norm.frozen_scale._version == version
    before = publications(tmp_path)
    monkeypatch.setattr(evaluation, "load_file", lambda *a, **k: pytest.fail("statistics restore ran"))
    with pytest.raises(ValueError, match="cursor binding"):
        run(tmp_path, model, dictionary, resume=True)
    assert publications(tmp_path) == before
