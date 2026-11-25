"""Bounded token prefix checks with fake tokenizers and existing CPU fixtures."""
from copy import deepcopy

import pytest
import torch

from caliper.interp import _trace
from caliper.interp._dictionary import model_identity
from caliper.interp._index import _tokenizer_hash
from caliper.interp._reference_graph import PROFILE
from .fixture import TinyLM, TinyTokenizer
from .test_trace import exact_dictionary


DALLAS = "Fact: the capital of the state containing Dallas is"
# Recorded IDs from the pinned reference run for the lowercase tutorial prompt.
DALLAS_IDS = [2, 18143, 235292, 573, 6037, 576, 573, 2329, 10751, 26865, 603]


class PrefixTokenizer(TinyTokenizer):
    """Return explicit content IDs and model the automatic BOS option."""

    all_special_ids = [256, 257, 258, 259]

    def __init__(self, ids, *, automatic_bos=True):
        self.ids = torch.tensor([ids], dtype=torch.long)
        self.automatic_bos = automatic_bos
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        ids = self.ids.clone()
        if kwargs.get("add_special_tokens", True) and self.automatic_bos:
            ids = torch.cat((ids.new_full((1, 1), self.bos_token_id), ids), dim=1)
        return {"input_ids": ids}


class CapturedTokens(Exception):
    pass


@pytest.fixture
def token_capture(monkeypatch):
    """Capture admitted IDs before any frozen path or graph calculation."""
    captured = []

    def capture(model, ids, dictionary, **kwargs):
        captured.append(ids.clone())
        raise CapturedTokens

    monkeypatch.setattr(_trace, "FrozenPath", capture)
    return captured


def tracer_for(tokenizer, *, vocab=512):
    model = TinyLM(layers=1, width=4, heads=1, latents=4, k=2, vocab=vocab)
    return _trace.Tracer(model, tokenizer, exact_dictionary(model))


def trace_tokens(tracer, captured, prompt="AB", *, profile=PROFILE):
    with pytest.raises(CapturedTokens):
        tracer.trace(prompt, max_targets=0, profile=profile)
    return captured[-1][0].tolist()


@pytest.mark.parametrize("automatic_bos", [True, False])
@pytest.mark.parametrize("prefix", [[], [256], [257], [258], [259], [256, 257]])
def test_reference_preserves_special_prefix_or_adds_one_bos(token_capture, automatic_bos, prefix):
    tokenizer = PrefixTokenizer([*prefix, 65, 66], automatic_bos=automatic_bos)
    tracer = tracer_for(tokenizer)
    expected = [*(prefix or [256]), 65, 66]
    assert trace_tokens(tracer, token_capture) == expected
    assert tokenizer.calls == [("AB", {"return_tensors": "pt", "truncation": False,
                                       "add_special_tokens": False})]


@pytest.mark.parametrize("automatic_bos", [True, False])
@pytest.mark.parametrize("explicit_bos", [True, False])
def test_reference_keeps_exact_dallas_ids(token_capture, automatic_bos, explicit_bos):
    tokenizer = PrefixTokenizer(DALLAS_IDS if explicit_bos else DALLAS_IDS[1:],
                                automatic_bos=automatic_bos)
    tokenizer.bos_token_id, tokenizer.all_special_ids = 2, [0, 1, 2]
    tracer = tracer_for(tokenizer, vocab=256000)
    prompt = ("<bos>" if explicit_bos else "") + DALLAS
    assert trace_tokens(tracer, token_capture, prompt) == DALLAS_IDS
    assert len(token_capture[-1][0]) == 11


@pytest.mark.parametrize("bos", [None, True, 2.5, -1, 512])
def test_reference_refuses_missing_or_invalid_bos(token_capture, bos):
    tokenizer = PrefixTokenizer([65, 66], automatic_bos=False)
    tokenizer.bos_token_id = bos
    tracer = tracer_for(tokenizer)
    with pytest.raises(ValueError, match="BOS"):
        tracer.trace("AB", max_targets=0, profile=PROFILE)
    assert not token_capture


@pytest.mark.parametrize("prefix", [257, 258, 259])
def test_reference_existing_special_prefix_needs_no_bos(token_capture, prefix):
    tokenizer = PrefixTokenizer([prefix, 65], automatic_bos=False)
    tokenizer.bos_token_id = None
    assert trace_tokens(tracer_for(tokenizer), token_capture) == [prefix, 65]


def test_reference_accepts_zero_bos_without_tokenizer_mutation(token_capture):
    tokenizer = PrefixTokenizer([65, 66], automatic_bos=False)
    tokenizer.bos_token_id = 0
    tokenizer.all_special_ids = [0, 257, 258]
    tracer = tracer_for(tokenizer)
    settings = deepcopy(tracer.model.config.__dict__)
    identity = model_identity(tracer.model), _tokenizer_hash(tokenizer, 512)
    assert trace_tokens(tracer, token_capture) == [0, 65, 66]
    assert identity == (model_identity(tracer.model), _tokenizer_hash(tokenizer, 512))
    assert tracer.model.config.__dict__ == settings
    assert tokenizer.ids.tolist() == [[65, 66]]
    assert tokenizer.bos_token_id == 0 and tokenizer.automatic_bos is False
    assert tokenizer.all_special_ids == [0, 257, 258]


@pytest.mark.parametrize("profile", ["caliper", PROFILE])
@pytest.mark.parametrize("ids,message", [
    ([], "prompt tokens"),
    ([65] * 33, "prompt tokens"),
    ([-1, 65], "out-of-range"),
    ([512, 65], "out-of-range"),
    (torch.tensor([65]), "one integer token sequence"),
    (torch.tensor([[65], [66]]), "one integer token sequence"),
    (torch.tensor([[65.]]), "one integer token sequence"),
    (torch.tensor([[True]]), "one integer token sequence"),
])
def test_token_gates_precede_forward(token_capture, profile, ids, message):
    tokenizer = PrefixTokenizer([], automatic_bos=False)
    tokenizer.ids = ids if isinstance(ids, torch.Tensor) else torch.tensor([ids], dtype=torch.long)
    with pytest.raises(ValueError, match=message):
        tracer_for(tokenizer).trace("AB", max_targets=0, profile=profile)
    assert not token_capture


@pytest.mark.parametrize("ids,allowed", [([65] * 31, True), ([65] * 32, False),
                                        ([258] + [65] * 31, True)])
def test_reference_length_gate_counts_the_prefix(token_capture, ids, allowed):
    tracer = tracer_for(PrefixTokenizer(ids, automatic_bos=False))
    if allowed:
        assert len(trace_tokens(tracer, token_capture)) == 32
    else:
        with pytest.raises(ValueError, match="prompt tokens"):
            tracer.trace("AB", max_targets=0, profile=PROFILE)
        assert not token_capture


@pytest.mark.parametrize("automatic_bos", [True, False])
@pytest.mark.parametrize("prefix", [[], [256]])
def test_ordinary_tokenization_keeps_defaults(token_capture, automatic_bos, prefix):
    tokenizer = PrefixTokenizer([*prefix, 65, 66], automatic_bos=automatic_bos)
    expected = ([256] if automatic_bos else []) + prefix + [65, 66]
    assert trace_tokens(tracer_for(tokenizer), token_capture, profile="caliper") == expected
    assert tokenizer.calls == [("AB", {"return_tensors": "pt", "truncation": False})]
