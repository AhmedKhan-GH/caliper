"""Prompt admission, cancellation, and hosted reuse on small CPU models."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _hosted, _publish, _trace
from caliper.interp._gemma_reference import PROFILE
from caliper.interp._memory import CleanStop
from caliper.interp._output import Output, prompt_inputs
from .test_prompt_consistency import case, local_tokenizer
from .test_reference_tokens import DALLAS, DALLAS_IDS, PrefixTokenizer, tracer_for
from .test_service_memory import RequestGuard
from .test_service_trace_admission import enqueue, run, unchanged_publication
from .test_hosted_trace_admission import case as ordinary_hosted


@contextmanager
def no_forward(model):
    handle = model.register_forward_pre_hook(lambda *args: pytest.fail("refused prompt reached the model"))
    try:
        yield
    finally:
        handle.remove()


@pytest.fixture
def hosted(case):
    case.tokenizer = local_tokenizer()
    case.service.configure(dictionary=case.dictionary, tokenizer=case.tokenizer)
    index = SimpleNamespace(model=case.model, tokenizer=case.tokenizer, dictionary=case.dictionary,
        memory=RequestGuard(40_000_000_000), label_identity=("prompt-fixture", "labels"),
        labels=lambda features, **kw: {key: f"Feature {key[0]}:{key[1]}" for key in features})
    failures = []
    owner = SimpleNamespace(
        model=case.model, tokenizer=case.tokenizer, dictionary=case.dictionary, index=index,
        guard=RequestGuard(40_000_000_000), controls=case.controls, interventions=case.service,
        sweep=lambda **kw: None, validate=lambda **kw: None,
        prompt=case.controls.param("prompt", "A B"), status=SimpleNamespace(set=lambda value: None),
        _check_stop=lambda: None, _latch_failure=failures.append,
        update_controls=case.controls.step, update_output=lambda: None, _reconcile_selection=lambda graph: None)
    service = _hosted.HostedService(owner)
    service._bind_controls(case.controls)
    try:
        yield SimpleNamespace(case=case, service=service, owner=owner, failures=failures)
    finally:
        service.close()


@pytest.mark.parametrize("prefix,first", [("", 1), ("<bos> ", 1), ("<eos> ", 2)])
def test_hosted_accepts_exactly_32_reference_tokens_and_reuses_graph(hosted, prefix, first):
    prompt = prefix + " ".join(["A"] * 31)
    graph = run(hosted.service, prompt=prompt, profile=PROFILE, max_n_logits=1, batch_size=2)
    assert [token["token_id"] for token in graph.tokens] == [first, *([4] * 31)]
    assert hosted.owner.prompt.value == hosted.owner.traced_prompt == hosted.owner.checkpoint_prompt == prompt
    assert run(hosted.service, prompt=prompt, profile=PROFILE, max_n_logits=1, batch_size=2) is graph
    case = hosted.case
    assert Output(case.model, case.tokenizer, case.service).compare(prompt)["tokens"]
    before = dict(_publish._TENSORS)
    rejected = prompt + " A"
    replies = enqueue(hosted.service, prompt=rejected, profile=PROFILE, max_n_logits=1, batch_size=2)
    assert hosted.service.step() == 1
    assert replies[0][0] == "error" and "prompt tokens" in replies[0][2]["message"]
    assert hosted.service.graph is graph and hosted.owner.prompt.value == prompt
    unchanged_publication(before)


def test_hosted_explicit_ordinary_profile_keeps_automatic_bos(ordinary_hosted):
    hosted = ordinary_hosted
    tokenizer = local_tokenizer()
    hosted.service.tokenizer = hosted.owner.tokenizer = hosted.index.tokenizer = tokenizer
    hosted.owner.interventions.configure(dictionary=hosted.dictionary, tokenizer=tokenizer)
    prompt = "<bos> " + " ".join(["A"] * 30)
    graph = run(hosted.service, prompt=prompt, profile="caliper", max_targets=0)
    assert [token["token_id"] for token in graph.tokens] == [1, 1, *([4] * 30)]
    replies = enqueue(hosted.service, prompt=prompt + " A", profile="caliper")
    assert hosted.service.step() == 1 and replies[0][0] == "error"
    assert hosted.service.graph is graph and hosted.owner.prompt.value == prompt


@pytest.mark.parametrize("stage", ["tokenization admission", "tokenization"])
def test_reference_deadline_preserves_prior_graph_before_forward(case, monkeypatch, stage):
    tokenizer = local_tokenizer()
    tracer = _trace.Tracer(case.model, tokenizer, case.dictionary)
    graph = tracer.trace("<bos> A B", profile=PROFILE, max_targets=0, max_n_logits=1, batch_size=2)
    published = dict(_publish._TENSORS)
    calls = []
    def encode(*args, **kwargs):
        calls.append(kwargs)
        return tokenizer(*args, **kwargs)
    tracer.tokenizer = encode
    def check(self, current):
        if current == stage:
            raise _trace.TraceTimeout("fixture token deadline")
    monkeypatch.setattr(_trace._Deadline, "__call__", check)
    with no_forward(case.model), pytest.raises(_trace.TraceTimeout, match="token deadline"):
        tracer.trace("<eos> A B", profile=PROFILE, max_targets=0)
    assert len(calls) == (stage == "tokenization")
    assert graph in tracer._cache.values()
    unchanged_publication(published)


def test_hosted_cancellation_after_reference_tokenizer_preserves_prompt(hosted, monkeypatch):
    graph = run(hosted.service, prompt="<bos> A B", profile=PROFILE, max_n_logits=1, batch_size=2)
    published = dict(_publish._TENSORS)
    tokenizer = hosted.service.tokenizer
    call = type(tokenizer).__call__
    calls = []
    def encode(self, *args, **kwargs):
        calls.append(kwargs)
        encoded = call(self, *args, **kwargs)
        hosted.owner.guard.failure = CleanStop("fixture token cancellation")
        return encoded
    monkeypatch.setattr(type(tokenizer), "__call__", encode)
    replies = enqueue(hosted.service, prompt="<eos> A B", profile=PROFILE)
    with no_forward(hosted.case.model), pytest.raises(CleanStop, match="token cancellation"):
        hosted.service.step()
    assert calls and all(call["add_special_tokens"] is False for call in calls)
    assert replies[0][0] == "error" and replies[0][2]["code"] == "interpret.stopped"
    assert hosted.service.graph is graph and hosted.owner.prompt.value == "<bos> A B"
    assert hosted.failures and hosted.service._request_stack is None
    unchanged_publication(published)


class RawTokenizer:
    bos_token_id = 1
    all_special_ids = [0, 1, 2]

    def __init__(self, ids, mask=None):
        self.encoded = {"input_ids": ids}
        if mask is not None:
            self.encoded["attention_mask"] = mask

    def __call__(self, prompt, **kwargs):
        assert kwargs == {"return_tensors": "pt", "truncation": False, "add_special_tokens": False}
        return self.encoded


@pytest.mark.parametrize("ids,mask,message", [
    ([[4, 6]], None, "one integer"),
    (None, None, "one integer"),
    (torch.empty((1, 0), dtype=torch.long), None, "prompt tokens"),
    (torch.full((1, 33), 4), None, "prompt tokens"),
    (torch.full((1, 32), 4), None, "prompt tokens"),
    (torch.tensor([4, 6]), None, "one integer"),
    (torch.tensor([[4], [6]]), None, "one integer"),
    (torch.tensor([[4.]]), None, "one integer"),
    (torch.tensor([[True]]), None, "one integer"),
    (torch.tensor([[-1, 4]]), None, "out-of-range"),
    (torch.tensor([[1, 37]]), None, "out-of-range"),
    (torch.tensor([[4, 6]]), torch.ones(1, 1), "unpadded"),
    (torch.tensor([[4, 6]]), [[1, 1]], "unpadded"),
    (torch.tensor([[4, 6]]), torch.tensor([[1, 0]]), "unpadded"),
    (torch.tensor([[1, 4]]), torch.tensor([[1, 2]]), "unpadded"),
    (torch.tensor([[2, 4]]), torch.tensor([[1., float("nan")]]), "unpadded"),
])
def test_reference_output_rejects_invalid_inputs_before_forward(case, ids, mask, message):
    tokenizer = RawTokenizer(ids, mask)
    case.service.configure(dictionary=case.dictionary, tokenizer=tokenizer)
    with no_forward(case.model), pytest.raises(ValueError, match=message):
        Output(case.model, tokenizer, case.service).compare("A B")
    with no_forward(case.model), pytest.raises(ValueError, match=message):
        _trace.Tracer(case.model, tokenizer, case.dictionary).trace("A B", profile=PROFILE, max_targets=0)
    assert not _publish._TENSORS


@pytest.mark.parametrize("bos", [None, True, 1.5, -1, 37])
def test_output_requires_valid_bos_only_without_a_special_prefix(case, bos):
    tokenizer = RawTokenizer(torch.tensor([[4, 6]]))
    tokenizer.bos_token_id = bos
    with pytest.raises(ValueError, match="BOS"):
        prompt_inputs(case.model, tokenizer, "A B")
    tokenizer.encoded["input_ids"] = torch.tensor([[2, 4, 6]])
    assert prompt_inputs(case.model, tokenizer, "<eos> A B")["input_ids"].tolist() == [[2, 4, 6]]


@pytest.mark.parametrize("mask_dtype", [None, torch.bool, torch.int32, torch.float32])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_prefix_preserves_tensor_dtype_device_and_tokenizer_storage(case, mask_dtype, ids_dtype):
    ids = torch.tensor([[4, 6]], dtype=ids_dtype)
    mask = None if mask_dtype is None else torch.ones_like(ids, dtype=mask_dtype)
    tokenizer = RawTokenizer(ids, mask)
    tokenizer.bos_token_id = 0
    actual = prompt_inputs(case.model, tokenizer, "A B")
    assert actual["input_ids"].tolist() == [[0, 4, 6]]
    assert actual["input_ids"].dtype == ids_dtype
    assert ids.tolist() == [[4, 6]] and tokenizer.encoded["input_ids"] is ids
    assert all(value.device == next(case.model.parameters()).device for value in actual.values())
    if mask is None:
        assert "attention_mask" not in actual
    else:
        assert actual["attention_mask"].tolist() == [[1, 1, 1]]
        assert actual["attention_mask"].dtype == mask_dtype
        assert tokenizer.encoded["attention_mask"] is mask and mask.shape == (1, 2)


@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("automatic", [True, False])
def test_output_preserves_recorded_dallas_ids(explicit, automatic):
    tokenizer = PrefixTokenizer(DALLAS_IDS if explicit else DALLAS_IDS[1:], automatic_bos=automatic)
    tokenizer.bos_token_id, tokenizer.all_special_ids = 2, [0, 1, 2]
    model = tracer_for(tokenizer, vocab=256000).model
    prompt = ("<bos>" if explicit else "") + DALLAS
    assert prompt_inputs(model, tokenizer, prompt, profile=PROFILE)["input_ids"].tolist() == [DALLAS_IDS]


@pytest.mark.parametrize("prefix,first", [("<bos>", 1), ("<eos>", 2)])
def test_single_special_token_is_a_complete_reference_prompt(case, prefix, first):
    tokenizer = local_tokenizer()
    case.service.configure(dictionary=case.dictionary, tokenizer=tokenizer)
    graph = _trace.Tracer(case.model, tokenizer, case.dictionary).trace(
        prefix, profile=PROFILE, max_targets=0, max_n_logits=1, batch_size=2)
    assert [token["token_id"] for token in graph.tokens] == [first]
    assert Output(case.model, tokenizer, case.service).compare(prefix)["tokens"]
