"""CPU checks for source spans, synthetic prefixes, and next-token targets."""
from types import SimpleNamespace

import pytest
import torch

from caliper.interp._context import context_policy, prepare_contexts, source_segments


def model(family="gemma2"):
    return SimpleNamespace(config=SimpleNamespace(model_type=family, bos_token_id=2,
                                                  pad_token_id=0, vocab_size=16))


def test_prefix_and_padding_are_not_source_observations():
    rows = [[3, 4, 5], [6], [0, 2]]
    batch = prepare_contexts(rows, context_policy(model(), 4), device="cpu", pad_to=4)
    assert batch.input_ids.tolist() == [[2, 3, 4, 5], [2, 6, 0, 0], [2, 0, 2, 0]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]]
    assert batch.input_ids[batch.content_mask].tolist() == [3, 4, 5, 6, 0, 2]
    assert batch.input_ids[:, 1:][batch.prediction_mask].tolist() == [3, 4, 5, 6, 0, 2]
    assert batch.input_ids[:, :-1][batch.prediction_mask].tolist() == [2, 3, 4, 2, 2, 0]
    assert rows == [[3, 4, 5], [6], [0, 2]]


def test_context_limit_reserves_one_prefix_position():
    policy = context_policy(model(), 256)
    batch = prepare_contexts([list(range(255))], policy, device="cpu")
    assert batch.input_ids.shape == (1, 256)
    with pytest.raises(ValueError, match="lengths"):
        prepare_contexts([list(range(256))], policy, device="cpu")
    for length in (1, 257, True):
        with pytest.raises(ValueError, match="reserve"):
            context_policy(model(), length)


def test_document_spans_survive_partial_source_batches():
    docs, positions = [7, 7, 8, 8, 8, 8, 8, 8], [0, 1, 0, 1, 2, 3, 4, 5]
    assert list(source_segments(docs[:4], positions[:4], 3)) == [(0, 2)]
    assert list(source_segments(docs, positions, 3)) == [(0, 2), (2, 5), (5, 8)]
    assert list(source_segments([7, 7, 7], [0, 1, 0], 3, final=True)) == [(0, 2), (2, 3)]
    assert list(source_segments([7, 7], [12, 13], 3, final=True)) == [(0, 2)]
    with pytest.raises(ValueError, match="contiguous"):
        list(source_segments([7] * 4, [0, 1, 2, 9], 3))


def test_public_tokenizer_default_matches_context_prefix():
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"<pad>": 0, "<unk>": 1, "<bos>": 2, "alpha": 3, "beta": 4}, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend.post_processor = tokenizers.processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<bos>", pad_token="<pad>")
    policy = context_policy(model(), 4, tokenizer=tokenizer)
    content = tokenizer.encode("alpha beta", add_special_tokens=False)
    batch = prepare_contexts([content], policy, device="cpu")
    assert batch.input_ids.tolist() == [tokenizer.encode("alpha beta")]
    tokenizer.bos_token = "<unk>"
    with pytest.raises(ValueError, match="special IDs"):
        context_policy(model(), 4, tokenizer=tokenizer)


@pytest.mark.parametrize("family", ["unsupported_decoder", "caliper_tiny"])
def test_other_families_retain_their_existing_context_path(family):
    assert context_policy(model(family), 256) is None
