"""Display text for token metadata and packed piece labels."""


def display_text(value):
    """Return valid UTF-8 text with NUL as a visible escape."""
    return str(value).encode("utf-8", errors="replace").decode("utf-8").replace("\x00", r"\x00")


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
