"""Frozen paths, direct source effects, completeness, and graph contracts."""
import json

import pytest
import torch

from caliper.interp._graph import Graph


def test_graph_fixed_shapes_padding_and_paging():
    nodes = [{"id": "token:0:7", "kind": "token", "layer": -1, "position": 0,
              "token_id": 7, "activation": 1., "influence": .5, "effect": -.2,
              "label": "seven"},
             {"id": "logit:0:8", "kind": "logit", "layer": 4, "position": 0,
              "token_id": 8, "activation": .2, "influence": .5, "effect": 0.,
              "label": "eight"}]
    graph = Graph.build(nodes=nodes, edges=[{"source": 0, "target": 1, "weight": -.2}],
                        tokens=[{"token_id": 7, "text": "seven"}], layers=4,
                        completeness={"max_absolute_residual": 0., "targets": []},
                        omitted_influence=.1, error_share=0., feature_share=.4,
                        trace_id="example", device=torch.device("cpu"))
    expected = {"trace.counts": ((4,), torch.int32), "trace.node_i": ((512, 4), torch.int32),
                "trace.node_f": ((512, 4), torch.float32), "trace.edge_i": ((8192, 2), torch.int32),
                "trace.edge_f": ((8192, 2), torch.float32)}
    assert set(graph.tensors) == set(expected)
    for name, (shape, dtype) in expected.items():
        assert graph.tensors[name].shape == shape and graph.tensors[name].dtype == dtype
        if name != "trace.counts":
            assert graph.tensors[name][2 if "node" in name else 1:].count_nonzero() == 0
    assert graph.tensors["trace.counts"].tolist() == [2, 1, 1, 4]
    assert graph.tensors["trace.edge_f"][0].tolist() == pytest.approx([-.2, -1.])
    page = json.loads(graph.to_json(node_limit=1, edge_limit=1))
    assert page["next_node_offset"] == 1 and page["nodes"][0]["id"] == "token:0:7"
    assert graph.read(node_offset=1)["nodes"][0]["id"] == "logit:0:8"
    for argument in ({"node_limit": 257}, {"edge_offset": -1}, {"edge_limit": True}):
        with pytest.raises(ValueError):
            graph.read(**argument)


def exact_dictionary(model):
    from caliper.interp._dictionary import Dictionary, TopKTranscoder
    return Dictionary({}, [TopKTranscoder(**{name: getattr(block.mlp, name).detach().clone()
                                            for name in ("W_enc", "b_enc", "W_dec", "b_dec")},
                                         k=block.mlp.k) for block in model.layers])


def random_dictionary(model, *, latents=12, k=3):
    from caliper.interp._dictionary import Dictionary, TopKTranscoder
    width = model.config.hidden_size
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        layers = [TopKTranscoder(W_enc=torch.randn(latents, width) * .1,
                                b_enc=torch.randn(latents) * .1,
                                W_dec=torch.randn(latents, width) * .03,
                                b_dec=torch.randn(width) * .01, k=k)
                  for _ in range(model.config.num_hidden_layers)]
    return Dictionary({}, layers).to(dtype=next(model.parameters()).dtype)


def test_fixture_frozen_forward_linearity_and_zero_errors():
    from caliper.interp._frozen import FrozenPath
    from .fixture import TinyLM
    model = TinyLM()
    path = FrozenPath(model, torch.tensor([[256, 65, 66]]), exact_dictionary(model))
    assert path.forward_residual < 1e-9
    assert all(error.count_nonzero() == 0 for error in path.errors)
    zero_masks = torch.zeros(3, 1, 1, 1, dtype=torch.float64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        a = [torch.randn_like(path.embedding) for _ in range(path.layers + 1)]
        b = [torch.randn_like(path.embedding) for _ in range(path.layers + 1)]
    ar, af = path.run(a[0], a[1:], bias_masks=zero_masks)
    br, bf = path.run(b[0], b[1:], bias_masks=zero_masks)
    cr, cf = path.run(a[0] - .7 * b[0], [x - .7 * y for x, y in zip(a[1:], b[1:])], bias_masks=zero_masks)
    for x, y, z in zip(ar + [af], br + [bf], cr + [cf]):
        torch.testing.assert_close(z, x - .7 * y, atol=1e-10, rtol=0)


def test_real_qwen_modules_forward_and_linearity():
    from caliper.interp._frozen import FrozenPath
    from .test_points import qwen
    model = qwen()
    # Cover nonzero full-attention biases with actual primary module code.
    full = model.model.layers[3].self_attn
    for projection in (full.v_proj, full.o_proj):
        projection.bias = torch.nn.Parameter(torch.linspace(-.02, .02, projection.out_features))
    observed = {}
    handles = []
    for i, block in enumerate(model.model.layers):
        mixer = block.linear_attn if block.block_type == "linear_attention" else block.self_attn
        def record(mod, args, kwargs, result, i=i):
            observed[i] = (kwargs["hidden_states"].detach(),
                           (result[0] if isinstance(result, tuple) else result).detach())
        handles.append(mixer.register_forward_hook(record, with_kwargs=True))
    try:
        path = FrozenPath(model, torch.tensor([[1, 2, 3, 4]]), random_dictionary(model))
    finally:
        for handle in handles:
            handle.remove()
    assert path.forward_residual < 2e-5
    assert len(path.mixers) == 4
    assert all(error.abs().sum() > 0 for error in path.errors)
    for i, mixer in enumerate(path.mixers):
        x, y = observed[i]
        torch.testing.assert_close(mixer(x), y, atol=2e-6, rtol=2e-5)
        perturbation = torch.linspace(-.1, .1, x.numel()).reshape_as(x)
        torch.testing.assert_close(mixer(x + perturbation) - mixer(x), mixer(perturbation, 0.),
                                   atol=3e-6, rtol=1e-4)
        torch.testing.assert_close(mixer(1.3 * x - .7 * perturbation, 0.),
                                   1.3 * mixer(x, 0.) - .7 * mixer(perturbation, 0.), atol=3e-6, rtol=1e-4)
    # The whole independent-write path also obeys superposition.
    zero_masks = torch.zeros(3, 1, 1, 1)
    x = path.embedding
    writes = path.writes
    first = path.run(x, writes, bias_masks=zero_masks)
    second = path.run(2 * x, [2 * y for y in writes], bias_masks=zero_masks)
    for a, b in zip(first[0] + [first[1]], second[0] + [second[1]]):
        torch.testing.assert_close(b, 2 * a, atol=2e-6, rtol=1e-5)
