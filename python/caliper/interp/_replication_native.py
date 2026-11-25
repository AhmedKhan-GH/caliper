"""Native private tensor records for the explicit published transaction."""
import hashlib
from contextlib import contextmanager
from pathlib import Path

import torch


def write_native(path, tensors, *, root, created, reserve):
    """Retain actual tensor precision before any float32 archive conversion."""
    path, root = Path(path), Path(root).resolve(strict=True)
    if not path.resolve().is_relative_to(root):
        raise ValueError("native output must remain inside its private root")
    if not tensors or any(not isinstance(value, torch.Tensor) or value.layout != torch.strided
                          or value.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64)
                          for value in tensors.values()):
        raise ValueError("native output requires dense floating tensors")
    reserve(2 * sum(value.numel() * value.element_size() for value in tensors.values()) + 4 * 1024 * 1024)
    copies = {name: value.detach().to(device="cpu").contiguous() for name, value in tensors.items()}
    # The caller removes all candidate paths if the enclosing transaction fails.
    with path.open("xb") as stream:
        created.append(path)
        torch.save(copies, stream)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"file": str(path.relative_to(root)), "sha256": digest,
            "tensors": {name: {"dtype": str(value.dtype), "shape": list(value.shape)}
                        for name, value in tensors.items()}}


def write_native_tensor(path, name, tensor, *, root, created, reserve):
    """Keep a single-tensor observation within the hosted status depth limit."""
    record = write_native(path, {name: tensor}, root=root, created=created, reserve=reserve)
    return {"file": record["file"], "sha256": record["sha256"], "tensor": name,
            **record["tensors"][name]}


@contextmanager
def observe_features(model, dictionary, nodes, *, input_shape, guard, reserve):
    """Observe encoder features from the actual forward under the current clamps."""
    from ._frozen import _trace_features
    from ._intervene import _reference_features, _reference_input_scope, _uses_reference_inputs
    from ._memory import autorelease_scope
    from ._points import capture

    reads = dictionary.metadata["read_points"]
    if len(input_shape) != 2 or input_shape[0] != 1 or not nodes:
        raise ValueError("published feature observations require one prompt and explicit nodes")
    if any(len(node) != 3 or any(type(value) is not int for value in node)
           or not 0 <= node[0] < len(reads) or not 0 <= node[2] < input_shape[1]
           or not 0 <= node[1] < dictionary.layers[node[0]].W_enc.shape[0] for node in nodes):
        raise ValueError("published observation coordinates exceed the current resources")
    original = _uses_reference_inputs(dictionary)
    layers = sorted({node[0] for node in nodes})
    length = input_shape[1]
    width = model.config.hidden_size
    largest = max(dictionary.layers[layer].W_enc.shape[0] for layer in layers)
    reserve(length * width * (len(reads) if original else len(layers)) * 8 + length * largest * 32 + 4 * 1024 * 1024)
    observations = {}
    scope = _reference_input_scope(model) if original else capture(model, [reads[layer] for layer in layers])
    with scope as captured:
        yield observations
        for layer in layers:
            if reads[layer] not in captured:
                raise ValueError("published feature observation has no completed forward")
            with autorelease_scope(captured[reads[layer]].device), torch.no_grad():
                coordinate, weights = captured[reads[layer]], dictionary.layers[layer]
                # Retain the complete encoder matrix shape used by the published forward.
                features = (_reference_features(weights, coordinate) if original else
                            _trace_features(weights, coordinate, guard))
                for node in nodes:
                    if node[0] == layer:
                        observations[node] = features[0, node[2], node[1]].detach().reshape(1).clone()
                del features, coordinate
