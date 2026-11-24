"""Display text for token metadata and packed piece labels."""

from inspect import Parameter, signature


def display_text(value):
    """Return valid UTF-8 text with control characters as visible escapes."""
    text = str(value).encode("utf-8", errors="replace").decode("utf-8")
    for character, escape in (("\x00", r"\x00"), ("\n", r"\n"), ("\r", r"\r"), ("\t", r"\t")):
        text = text.replace(character, escape)
    return text


def token_label(tokenizer, token_id):
    """Decode one token, or identify a row with no token spelling.

    Vocabulary lookup detects sparse IDs and padded output rows. The decoder
    supplies display text with special tokens and boundary spaces intact.
    Raw vocabulary text remains separate from display text for identity checks.
    """
    token_id = int(token_id)
    lookup = getattr(tokenizer, "convert_ids_to_tokens", None)
    if lookup is not None and lookup(token_id) is None:
        return f"[unmapped token {token_id}]"
    decode = tokenizer.decode
    parameters = signature(decode).parameters
    accepts_kwargs = any(p.kind == Parameter.VAR_KEYWORD for p in parameters.values())
    options = {name: False for name in ("skip_special_tokens", "clean_up_tokenization_spaces")
               if accepts_kwargs or (name in parameters and parameters[name].kind in
                                     (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY))}
    # Select supported options before the call. Decoder errors must propagate.
    decoded = display_text(decode([token_id], **options))
    return decoded if decoded else f"[empty token {token_id}]"


def pack_labels(values, count):
    """Pack at most 63 UTF-8 bytes per label, with zero padding."""
    import torch
    labels = torch.zeros((count, 16), dtype=torch.uint32)
    for i, value in enumerate(values):
        raw = display_text(value).encode("utf-8")[:63].decode("utf-8", errors="ignore").encode("utf-8")
        raw = raw.ljust(64, b"\0")
        labels[i] = torch.tensor([int.from_bytes(raw[j:j + 4], "little") for j in range(0, 64, 4)],
                                 dtype=torch.uint32)
    return labels
