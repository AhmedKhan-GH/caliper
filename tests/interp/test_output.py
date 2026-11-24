"""Paired output checks on the seeded CPU fixture."""
import pytest
import torch

from caliper._controls import Controls
from caliper.interp._output import Output, forward_logits, is_internal_forward, packed_labels
from caliper.interp._publish import _clear, _resolve
from .fixture import make_fixture
from .test_intervene import dictionary_for, explicit_forward


@pytest.fixture
def output_case():
    model, tokenizer = make_fixture()
    controls = Controls()
    interventions = controls.interventions_on(model)
    interventions.configure(dictionary=dictionary_for(model), tokenizer=tokenizer)
    output = Output(model, tokenizer, interventions)
    yield model, tokenizer, controls, interventions, output
    interventions.remove()
    _clear()


def test_paired_probabilities_use_one_batch_and_preserve_baseline(output_case):
    model, tokenizer, controls, service, output = output_case
    ids = tokenizer("AB")["input_ids"]
    logits, acts = explicit_forward(model, ids)
    feature = int(acts[-1][0, -1].argmax())
    service.set({"kind": "clamp", "point": "layers.3.mlp_out", "feature": feature, "value": 0, "positions": [2]})
    controls.step()
    expected, _ = explicit_forward(model, ids, edit=lambda layer, a: a.scatter(
        -1, torch.full_like(a[..., :1], feature, dtype=torch.long), 0) if layer == 3 else a)
    # The oracle changes every position in the last MLP. Only its last position reaches these logits.
    calls = []
    handle = model.register_forward_hook(lambda m, a, y: calls.append((y.logits.shape, is_internal_forward())))
    result = output.compare("AB")
    handle.remove()
    assert calls == [(torch.Size([2, 3, 512]), True)]
    assert not is_internal_forward()
    selected = result["tokens"]
    assert int(logits[0, -1].argmax()) in selected
    torch.testing.assert_close(_resolve("output.probs")[0], logits[0, -1].softmax(-1)[selected].float())
    torch.testing.assert_close(_resolve("output.probs")[1], expected[0, -1].softmax(-1)[selected].float())
    assert _resolve("output.tokens").dtype == torch.int32
    assert _resolve("output.probs").dtype == torch.float32
    assert _resolve("output.label_text").shape == (10, 16)
    assert controls.n == 1
    assert service.specifications()[0]["feature"] == feature


def test_fork_matches_full_logits_after_overlapping_replacement(output_case):
    model, tokenizer, controls, service, output = output_case
    ablation = {"kind": "ablate", "point": "layers.3.mlp_out"}
    service.set(ablation)
    service.set({"kind": "add", "point": "layers.3.mlp_out", "vector": [1.] * 64, "coefficient": .25})
    controls.step(3)
    service.set(dict(ablation, until_step=20))
    controls.step(4)
    inputs = tokenizer("AB")
    expected = forward_logits(model, inputs)
    expected_bars = output.compare("AB")
    with service.suspend():
        baseline = forward_logits(model, inputs)
    assert not torch.equal(expected, baseline)
    fork_model, fork_tokenizer = make_fixture()
    fork_controls = Controls()
    fork = fork_controls.interventions_on(fork_model)
    try:
        fork.configure(dictionary=dictionary_for(fork_model), tokenizer=fork_tokenizer)
        fork.load_state_dict(service.state_dict())
        fork_controls.step(controls.n)
        torch.testing.assert_close(forward_logits(fork_model, inputs), expected, rtol=0, atol=0)
        assert Output(fork_model, fork_tokenizer, fork).compare("AB") == expected_bars
        assert fork.specifications() == service.specifications()
    finally:
        fork.remove()


def test_generation_has_exactly_24_greedy_tokens_per_row(output_case):
    model, tokenizer, controls, service, output = output_case
    service.set({"kind": "ablate", "point": "layers.3.mlp"})
    controls.step()
    before = service.state_dict()
    calls = []
    handle = model.register_forward_hook(lambda m, a, y: calls.append(y.logits.detach()))
    result = output.generate("A")
    handle.remove()
    assert len(calls) == 24
    assert [len(row) for row in result["token_ids"]] == [24, 24]
    for step, logits in enumerate(calls):
        assert logits.shape[:2] == (2, 2 + step)
        assert logits[:, -1].argmax(-1).tolist() == [row[step] for row in result["token_ids"]]
    assert result["baseline"] == tokenizer.decode(result["token_ids"][0])
    assert result["steered"] == tokenizer.decode(result["token_ids"][1])
    assert service.state_dict() == before
    assert controls.n == 1


@pytest.mark.parametrize("mapping", [{"positions": [1]}, {"source_position": 1}])
def test_generation_preserves_patch_state_and_requires_a_defined_source(output_case, mapping):
    model, tokenizer, controls, service, output = output_case
    spec = {"kind": "patch", "point": "layers.1.resid_pre", "source_prompt": "AB"}
    service.set(spec)
    controls.step()
    saved = service.state_dict()
    calls = []
    hook = model.register_forward_hook(lambda *args: calls.append(1))
    try:
        with pytest.raises(ValueError, match="patch positions"):
            output.generate("AC")
        assert not calls and service.state_dict() == saved
        service.set(dict(spec, **mapping))
        controls.step()
        saved = service.state_dict()
        result = output.generate("AC")
        assert [len(row) for row in result["token_ids"]] == [24, 24]
        assert service.state_dict() == saved
        assert not service.suspended and controls.n == 2
    finally:
        hook.remove()


def test_utf8_labels_end_on_character_boundaries():
    labels = packed_labels(["a" * 63 + "é", "😀" * 17, "A\0B"], "cpu")
    raw = [b"".join(int(i).to_bytes(4, "little") for i in row) for row in labels.tolist()]
    assert raw[0].rstrip(b"\0").decode() == "a" * 63
    assert raw[1].rstrip(b"\0").decode() == "😀" * 15
    assert raw[2].rstrip(b"\0").decode() == r"A\x00B"


def test_comparison_labels_preserve_sparse_rows_and_shared_display_text(output_case, monkeypatch):
    from caliper.interp import _output
    from caliper.interp._labels import token_label
    model, tokenizer, _, _, output = output_case
    logits = torch.full((2, 1, model.config.vocab_size), -100., dtype=torch.float64)
    logits[:, 0, :10] = torch.arange(10, 0, -1, dtype=torch.float64)
    spellings = {0: "A\0B", 1: None, 2: "None", 3: "", 4: "é" * 40,
                 5: "\ud800", 6: "<eos>", 7: "Ġtail", 8: r"\x00", 9: "word"}
    decoded = {**spellings, 7: " tail"}
    def decode(ids, *, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(decoded[i] for i in ids)

    monkeypatch.setattr(tokenizer, "convert_ids_to_tokens", spellings.get)
    monkeypatch.setattr(tokenizer, "decode", decode)
    monkeypatch.setattr(_output, "forward_logits", lambda *a, **k: logits)
    result = output.compare("A")
    assert result["tokens"] == list(range(10))
    assert result["labels"] == [r"A\x00B", "[unmapped token 1]", "None", "#3", "é" * 31,
                                "?", "<eos>", " tail", r"\x00", "word"]
    packed = _resolve("output.label_text").tolist()
    for token, label, row in zip(result["tokens"], result["labels"], packed):
        assert label == token_label(tokenizer, token).encode("utf-8")[:63].decode("utf-8", errors="ignore")
        raw = b"".join(int(word).to_bytes(4, "little") for word in row)
        assert raw.split(b"\0", 1)[0].decode("utf-8") == label
    expected = logits[:, 0].softmax(-1)[:, :10].float()
    torch.testing.assert_close(_resolve("output.probs"), expected, atol=0, rtol=0)


@pytest.mark.parametrize("prompt", ["", "x" * 32, "x" * 4097])
def test_prompt_caps_do_not_truncate(output_case, prompt):
    with pytest.raises(ValueError):
        output_case[-1].compare(prompt)


def test_failed_forward_restores_row_and_internal_scope(output_case, monkeypatch):
    model, tokenizer, controls, service, output = output_case
    from caliper.interp._intervene import selected_rows
    def fail(*a, **kw):
        assert selected_rows(model) == (1,)
        raise RuntimeError("fixture failure")
    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        output.compare("A")
    assert selected_rows(model) is None
    assert not is_internal_forward()


def test_supported_text_model_requests_only_next_token_logits(output_case, monkeypatch):
    from caliper.interp._output import forward_logits
    model, tokenizer, controls, service, output = output_case
    original = model.forward
    seen = []
    model.config.model_type = "qwen3_5_text"
    def forward(*a, **kwargs):
        seen.append(kwargs.pop("logits_to_keep", None))
        result = original(*a, **kwargs)
        result.logits = result.logits[:, -1:]
        return result
    monkeypatch.setattr(model, "forward", forward)
    value = forward_logits(model, tokenizer("AB"))
    assert value.shape == (1, 1, 512) and seen == [1]


@pytest.mark.parametrize("steered_winner", [0, 10])
def test_comparison_reserves_both_true_winners(output_case, monkeypatch, steered_winner):
    from caliper.interp import _output
    model, _, _, _, output = output_case
    probabilities = torch.empty((2, model.config.vocab_size), dtype=torch.float64)
    probabilities[0, :10] = torch.tensor([.11] + [.09] * 9)
    probabilities[0, 10:] = .08 / (model.config.vocab_size - 10)
    probabilities[1, :10] = torch.tensor([.07] + [.06] * 9)
    probabilities[1, 10] = .08
    probabilities[1, 11:] = .31 / (model.config.vocab_size - 11)
    if steered_winner == 0:
        probabilities[1, 0], probabilities[1, 10] = .08, .07
    logits = probabilities.log().unsqueeze(1)
    monkeypatch.setattr(_output, "forward_logits", lambda *args, **kwargs: logits)
    result = output.compare("AB")
    assert result["argmax_token_ids"] == [0, steered_winner]
    assert set(result["argmax_token_ids"]) <= set(result["tokens"])
    assert len(result["tokens"]) == len(set(result["tokens"])) == 10
    assert _resolve("output.probs").shape == (2, 10)
    expected = logits[:, 0].softmax(-1)[:, result["tokens"]].float()
    torch.testing.assert_close(_resolve("output.probs"), expected)
