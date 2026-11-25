"""Reference prompt consumers with random CPU weights and a local tokenizer."""
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from caliper._controls import Controls
from caliper.interp import _publish, _trace
from caliper.interp._dictionary import model_identity
from caliper.interp._gemma_reference import PROFILE, install_reference_profile, profile_metadata
from caliper.interp._index import _tokenizer_hash
from caliper.interp._output import Output, prompt_inputs
from caliper.interp._validate import validate_trace
from .test_gemma_frozen import original_encoder_dictionary
from .test_gemma_reference import reference_fixture
from .test_reference_interventions import independent, objective
from .fixture import TinyLM
from .test_trace import exact_dictionary


def local_tokenizer(automatic_bos=True):
    """Use an in-memory vocabulary with the Gemma automatic BOS behavior."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    vocab = {f"t{i}": i for i in range(37)}
    for word, index in {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "A": 4, "B": 6}.items():
        del vocab[f"t{index}"]
        vocab[word] = index
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A" if automatic_bos else "$A", special_tokens=[("<bos>", 1)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>",
                                  pad_token="<pad>", unk_token="<unk>",
                                  model_input_names=["input_ids", "attention_mask"])


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setattr(_publish, "_TENSORS", {})
    monkeypatch.setattr(_publish, "_VERSIONS", {})
    model = reference_fixture()
    install_reference_profile(model, PROFILE)
    dictionary = original_encoder_dictionary(model, dtype=torch.bfloat16)
    controls = Controls()
    service = controls.interventions_on(model)
    yield SimpleNamespace(model=model, dictionary=dictionary, controls=controls, service=service)
    service.remove()


@pytest.mark.parametrize("automatic_bos", [True, False])
@pytest.mark.parametrize("prefix,first", [("", 1), ("<bos> ", 1), ("<eos> ", 2)])
def test_completed_trace_output_and_validation_share_reference_positions(case, automatic_bos, prefix, first):
    case.tokenizer = local_tokenizer(automatic_bos)
    # Give random features enough output effect to survive ordinary reference pruning.
    with torch.no_grad():
        for layer in case.dictionary.layers:
            layer.W_dec.mul_(64)
    case.service.configure(dictionary=case.dictionary, tokenizer=case.tokenizer)
    prompt, expected_ids = prefix + "A B", torch.tensor([[first, 4, 6]])
    before = (model_identity(case.model), profile_metadata(case.model),
              deepcopy(case.model.config.to_dict()), _tokenizer_hash(case.tokenizer, 37))
    graph = _trace.Tracer(case.model, case.tokenizer, case.dictionary).trace(
        prompt, profile=PROFILE, max_targets=8, max_n_logits=1, batch_size=2)
    assert [token["token_id"] for token in graph.tokens] == expected_ids[0].tolist()
    assert any(node["kind"] == "feature" for node in graph.nodes)
    measured = validate_trace(case.model, case.tokenizer, case.dictionary, case.service, graph, prompt=prompt)
    _, baseline = independent(case, expected_ids)
    target = next(node for node in graph.nodes if node["kind"] == "logit")
    assert measured["baseline_centered_logit"] == objective(case, baseline, target["token_id"])
    assert measured["measurements"]

    feature = int(baseline["activation"][0, -1].argmax())
    spec = {"kind": "clamp", "point": "model.layers.0.mlp_out", "feature": feature,
            "positions": [2], "value": 0.}
    expected, _ = independent(case, expected_ids.repeat(2, 1), spec, rows=[1])
    case.service.set(spec)
    case.controls.step()
    observed = []
    with ExitStack() as stack:
        stack.callback(case.model.register_forward_pre_hook(
            lambda module, args, kwargs: observed.append({key: value.clone() for key, value in kwargs.items()
                                                        if key in ("input_ids", "attention_mask")}),
            with_kwargs=True).remove)
        result = Output(case.model, case.tokenizer, case.service).compare(prompt)
    assert len(observed) == 1
    torch.testing.assert_close(observed[0]["input_ids"], expected_ids.repeat(2, 1), rtol=0, atol=0)
    assert observed[0]["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 1]]
    assert all(value.device == next(case.model.parameters()).device for value in observed[0].values())
    torch.testing.assert_close(_publish._resolve("output.probs"),
                               expected[:, -1].softmax(-1)[:, result["tokens"]].float(), rtol=0, atol=0)
    assert not torch.equal(expected[0], expected[1])
    assert before == (model_identity(case.model), profile_metadata(case.model),
                      case.model.config.to_dict(), _tokenizer_hash(case.tokenizer, 37))


@pytest.mark.parametrize("automatic_bos", [True, False])
@pytest.mark.parametrize("prefix", ["", "<bos> ", "<eos> "])
def test_explicit_ordinary_profile_keeps_tokenizer_defaults(case, automatic_bos, prefix):
    tokenizer = local_tokenizer(automatic_bos)
    prompt = prefix + "A B"
    expected = tokenizer(prompt, return_tensors="pt")
    actual = prompt_inputs(case.model, tokenizer, prompt, profile="caliper")
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_generation_and_patch_source_use_reference_prefixes(case):
    case.tokenizer = local_tokenizer()
    case.service.configure(dictionary=case.dictionary, tokenizer=case.tokenizer)
    case.service.set({"kind": "patch", "point": "model.layers.0.resid_pre",
                      "source_prompt": "<eos> B A", "source_position": 2, "positions": [2]})
    case.controls.step()
    observed = []
    with ExitStack() as stack:
        stack.callback(case.model.register_forward_pre_hook(
            lambda module, args, kwargs: observed.append({key: value.clone() for key, value in kwargs.items()
                                                        if key in ("input_ids", "attention_mask")}),
            with_kwargs=True).remove)
        result = Output(case.model, case.tokenizer, case.service).generate("<bos> A B")
    generated = [row for row in observed if row["input_ids"].shape[0] == 2]
    sources = [row for row in observed if row["input_ids"].shape[0] == 1]
    assert len(generated) == 24 and sources
    assert all(row["input_ids"].tolist() == [[2, 6, 4]] for row in sources)
    for step, row in enumerate(generated):
        assert row["input_ids"][:, :3].tolist() == [[1, 4, 6], [1, 4, 6]]
        assert row["input_ids"].shape == row["attention_mask"].shape == (2, 3 + step)
        assert torch.all(row["attention_mask"] == 1)
    assert [len(row) for row in result["token_ids"]] == [24, 24]


@pytest.mark.parametrize("automatic_bos", [True, False])
@pytest.mark.parametrize("prefix", ["", "<bos>"])
def test_ordinary_trace_and_output_keep_tokenizer_defaults(monkeypatch, automatic_bos, prefix):
    monkeypatch.setattr(_publish, "_TENSORS", {})
    model = TinyLM(layers=1, width=4, heads=1, latents=4, k=2, vocab=37)
    tokenizer = local_tokenizer(automatic_bos)
    prompt = prefix + "A B"
    dictionary, controls = exact_dictionary(model), Controls()
    service = controls.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    try:
        graph = _trace.Tracer(model, tokenizer, dictionary).trace(prompt, max_targets=4)
        ids = torch.tensor([[token["token_id"] for token in graph.tokens]])
        expected_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
        expected = model(ids).logits[0, -1].softmax(-1)
        result = Output(model, tokenizer, service).compare(prompt)
        torch.testing.assert_close(_publish._resolve("output.probs")[0],
                                   expected[result["tokens"]].float(), rtol=0, atol=0)
    finally:
        service.remove()
