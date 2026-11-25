"""Independent pinned source hooks for small CPU fixtures."""
import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class Geometry(TorchDispatchMode):
    """Record exact gather and product operands without device work."""

    def __init__(self):
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if (func == torch.ops.aten.bmm.default or
                (func == torch.ops.aten.index.Tensor and args[0].ndim == 3)):
            tensors = [args[0], args[1], result] if func == torch.ops.aten.bmm.default else [args[0], result]
            self.calls.append((str(func), [(tuple(x.shape), x.stride(), x.dtype, x.clone()) for x in tensors]))
        return result


@pytest.fixture
def pinned_decoder(pinned_context):
    """Compile the actual pinned sparse decoder and its row selection method."""
    source = Path(os.environ["CALIPER_PINNED_ATTRIBUTION_SOURCE"])
    source = source.parent.parent / "transcoder" / "single_layer_transcoder.py"
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "8fb5719906c96653efed86353bbf83fd7217d646edf9f7c05e6ab157cd1a55e7"
    cls = next(node for node in ast.parse(raw).body if isinstance(node, ast.ClassDef)
               and node.name == "SingleLayerTranscoder")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ("encode_sparse", "decode_sparse", "_get_decoder_vectors")]
    assert len(methods) == 3
    namespace = dict(torch=torch, np=np, F=torch.nn.functional)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)

    def decode(rows, activations):
        owner = SimpleNamespace(W_dec=rows, dtype=activations.dtype, lazy_decoder=False,
                                W_skip=None, d_model=rows.shape[1], b_dec=rows.new_zeros(rows.shape[1]))
        owner._get_decoder_vectors = lambda ids: namespace["_get_decoder_vectors"](owner, ids)
        return namespace["decode_sparse"](owner, activations)[1]

    def non_bos(activations):
        columns = activations.shape[1]
        owner = SimpleNamespace(W_enc=torch.eye(columns, dtype=activations.dtype),
                                b_enc=activations.new_zeros(columns), activation_function=lambda x: x)
        return namespace["encode_sparse"](owner, activations)[0]

    decode.non_bos = non_bos
    return decode


def source_case(dtype, counts, width=37):
    """Use a small linear path with distinct positions and all bias categories."""
    generator = torch.Generator().manual_seed(910)

    def values(*shape, dtype=dtype):
        return (torch.randint(-16, 17, shape, generator=generator).to(dtype) / 16)

    layers, tokens, columns = len(counts), len(counts[0]), 7
    path = SimpleNamespace(device=torch.device("cpu"), dtype=dtype, layers=layers, tokens=tokens,
                           width=width, mixer_width=width, raw_reference_input=True)
    path.embedding = values(1, tokens, width)
    path.errors = [values(1, tokens, width, dtype=torch.float32) for _ in counts]
    path.decoder_biases = [values(1, tokens, width, dtype=torch.float32) for _ in counts]
    path.bias_final = values(4, tokens, width)
    path.bias_reads = [values(4, tokens, width) for _ in counts]
    path.logits = values(1, 3)
    directions = values(3, width)
    path.logit_target = lambda token, **kwargs: (directions[token], path.embedding.new_tensor((token + 1) / 8))
    path.reserves = []
    path.guard = SimpleNamespace(reserve=lambda *args: path.reserves.append(args), check=lambda *args: None)
    factors = torch.arange(1, tokens + 1, dtype=torch.float32).reshape(1, tokens, 1) / 4

    def run(embedding, writes, **kwargs):
        x, reads = embedding, []
        for layer, write in enumerate(writes):
            reads.append(x + x.sum(1, keepdim=True))
            x = x + write * (factors + layer)
        return reads, x.sum(1, keepdim=True).expand(-1, tokens, -1).to(dtype)

    path.run = run
    _, path.reference_final = run(path.embedding, path.errors)
    path.feature_outputs, weights = [], []
    for layer_counts in counts:
        indices = torch.stack([torch.randperm(columns, generator=generator) for _ in layer_counts])
        activations = path.embedding.new_zeros(tokens, columns)
        for position, count in enumerate(layer_counts):
            activations[position, :count] = torch.tensor([.1, -1.3, .5, 2.1, -.25, 1., -.7][:count], dtype=dtype)
        path.feature_outputs.append(SimpleNamespace(indices=indices, activations=activations,
            coefficients=values(tokens, columns), constants=values(tokens, columns)))
        encoder = values(columns, width)
        decoder = values(columns, width, dtype=torch.float64).T.contiguous().T
        weights.append(SimpleNamespace(W_dec=decoder, W_enc=encoder,
                                       W_enc_original=encoder, b_enc=values(columns)))
    dictionary = SimpleNamespace(layers=weights)
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "x")
    return path, dictionary, list(range(tokens)), tokenizer


def whole_layer_scores(effects, gradients, context_class, decode):
    """Run actual non-BOS layer hooks and separate full-source BOS hooks."""
    path = effects.path
    sparse, vectors, source_rows, scopes = [], [], [], []
    for layer, weights in enumerate(effects.dictionary.layers):
        dense = path.embedding.new_zeros(path.tokens, len(weights.W_dec))
        output = path.feature_outputs[layer]
        dense.scatter_(1, output.indices, output.activations)
        acts = decode.non_bos(dense)
        assert not (acts.indices()[0] == 0).any()
        sparse.append(acts)
        vectors.append(decode(weights.W_dec, acts))
        nodes = {(node["position"], node["feature"]): index for index, node in enumerate(effects.nodes)
                 if node["kind"] == "feature" and node["layer"] == layer}
        rows = [nodes[tuple(coordinate)] for coordinate in acts.indices().T.tolist()]
        source_rows.extend(rows)
        scopes.append(dict(layer=layer, bos=False, rows=rows, vectors=vectors[-1], positions=acts.indices()[0]))
        dense[1:] = 0
        bos = dense.to_sparse().coalesce()
        scopes.append(dict(layer=layer, bos=True,
            rows=[nodes[tuple(coordinate)] for coordinate in bos.indices().T.tolist()],
            vectors=decode(weights.W_dec, bos), positions=bos.indices()[0]))
    activations = torch.stack(sparse).coalesce()
    feature_count = activations._nnz()
    context = context_class(activations, torch.stack([error[0] for error in path.errors]),
        path.embedding[0], torch.cat(vectors), torch.empty(0), torch.arange(feature_count),
        activations.indices()[:2], path.logits)
    context._batch_buffer = path.embedding.new_zeros(context._row_size, len(gradients[0]))
    published_scopes = [scope for scope in scopes if not scope["bos"] and scope["rows"]]
    for hook_index, (name, hook) in enumerate(context._make_attribution_hooks("synthetic")):
        layer = None if name == "hook_embed" else int(name.split(".")[1])
        gradient = gradients[0 if layer is None else layer + 1]
        scope = published_scopes[hook_index] if hook_index < len(published_scopes) else None
        if gradient is not None:
            with Geometry() as geometry:
                hook(gradient, None)
            if scope is not None:
                scope["geometry"] = geometry.calls
    source_rows.extend(index for index, node in enumerate(effects.nodes) if node["kind"] == "error")
    source_rows.extend(range(path.tokens))
    expected = path.embedding.new_zeros(len(effects.nodes), len(gradients[0]))
    expected[source_rows] = context._batch_buffer
    bos_context = object.__new__(context_class)
    bos_context._batch_buffer = expected
    for scope in scopes:
        if not scope["bos"] or not scope["rows"] or gradients[scope["layer"] + 1] is None:
            continue
        _, hook = bos_context._compute_score_hook("bos", scope["vectors"], scope["rows"],
                                                  np.s_[:, scope["positions"]])
        with Geometry() as geometry:
            hook(gradients[scope["layer"] + 1], None)
        scope["geometry"] = geometry.calls
    return expected, scopes


def contraction_evidence(effects, targets, monkeypatch):
    """Observe actual gathers and products without a candidate-shaped oracle."""
    effects.prepare_decoders()
    decoders = {decoder.untyped_storage().data_ptr(): rows
                for blocks in effects._decoder_blocks for rows, position, decoder in blocks}
    gradients, calls = [], []
    grad = torch.autograd.grad

    def record(*args, **kwargs):
        result = grad(*args, **kwargs)
        gradients.extend(result)
        return result

    class Contracts(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func == torch.ops.aten.mm.default:
                calls.append(dict(mm=True))
            if func == torch.ops.aten.bmm.default:
                matching = [x for x in args[:2] if x.untyped_storage().data_ptr() in decoders]
                if matching:
                    rows = decoders[matching[0].untyped_storage().data_ptr()]
                    calls.append(dict(rows=rows,
                        operands=[(tuple(x.shape), x.stride(), x.dtype, x.clone()) for x in (*args[:2], result)]))
            return result

    with monkeypatch.context() as patch:
        patch.setattr(torch.autograd, "grad", record)
        with Geometry() as geometry, Contracts():
            result = effects.backward(targets)
    return result, gradients, calls, geometry.calls


def compare_source_products(effects, calls, scopes, batch):
    """Check source matrices and disclose the bounded target-row pitch."""
    expected = {}
    for scope in scopes:
        if "geometry" not in scope:
            continue
        product, = [operands for name, operands in scope["geometry"] if name == "aten.bmm.default"]
        for index, row in enumerate(scope["rows"]):
            expected[row] = (scope, product, index)
    observed = []
    assert not any(call.get("mm") for call in calls)
    for call in calls:
        rows, operands = call["rows"], call["operands"]
        count = rows.stop - rows.start
        for index, row in enumerate(range(rows.start, rows.stop)):
            scope, product, ref_index = expected[row]
            width = effects.path.width
            multi = len(scope["rows"]) > 1 or batch == 1
            assert [x[0][1:] for x in operands] == [x[0][1:] for x in product]
            for actual, reference in zip(operands, product):
                assert actual[2] == reference[2]
                torch.testing.assert_close(actual[3][index], reference[3][ref_index], rtol=0, atol=0)
            if multi and batch > 1:
                # Full gathers use P*D between targets. Bounded gathers use C*D.
                assert operands[0][1] == (width, count * width, 1)
                assert product[0][1] == (width, len(scope["rows"]) * width, 1)
                assert operands[1][1][0:2] == product[1][1][0:2] == (width, 1)
            elif not multi:
                assert [x[1] for x in operands] == [x[1] for x in product]
            observed.append(row)
    assert observed == [row for row in effects.feature_nodes if row in expected]
