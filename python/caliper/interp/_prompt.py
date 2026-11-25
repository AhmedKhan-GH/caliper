"""Shared token prefixes for the pinned reference profile."""
import torch

from ._graph import bounded_integer


def reference_prompt_inputs(model, tokenizer, prompt, *, check=None):
    """Preserve a special prefix or prepend one valid BOS without tokenizer edits."""
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False, add_special_tokens=False)
    if check is not None:
        check("tokenization")
    ids = encoded["input_ids"]
    if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1
            or ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("tokenizer must return one integer token sequence")
    bounded_integer(ids.shape[1], "prompt tokens including special tokens", 1, 32)
    if ids.min() < 0 or ids.max() >= model.config.vocab_size:
        raise ValueError("tokenizer returned an out-of-range token")
    mask = encoded.get("attention_mask")
    if "attention_mask" in encoded and (not isinstance(mask, torch.Tensor)
            or mask.shape != ids.shape or not torch.all(mask == 1)):
        raise ValueError("one unpadded prompt is required")
    # The pinned reference retains any special token at position zero, such as EOS.
    if ids[0, 0].item() not in getattr(tokenizer, "all_special_ids", ()):
        bos = getattr(tokenizer, "bos_token_id", None)
        if type(bos) is not int or not 0 <= bos < model.config.vocab_size:
            raise ValueError("reference tokenization requires a valid BOS token")
        bounded_integer(ids.shape[1] + 1, "prompt tokens including special tokens", 1, 32)
        ids = torch.cat((ids.new_full((1, 1), bos), ids), dim=1)
        if mask is not None:
            mask = torch.cat((mask.new_ones((1, 1)), mask), dim=1)
    inputs = {"input_ids": ids}
    if mask is not None:
        inputs["attention_mask"] = mask
    return inputs
