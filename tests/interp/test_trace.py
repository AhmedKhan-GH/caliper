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


def test_trace_fixture_completeness_publication_caps_and_determinism():
    from caliper.interp import Tracer
    from caliper.interp._publish import _resolve
    from .fixture import TinyLM, TinyTokenizer
    model = TinyLM()
    tracer = Tracer(model, TinyTokenizer(), exact_dictionary(model))
    graph = tracer.trace("AB", max_targets=32, max_nodes=48, max_edges=80)
    assert graph.completeness["max_absolute_residual"] < 1e-9
    assert graph.completeness["feature_targets"] == 32
    assert graph.error_share == 0 and graph.feature_share > 0
    assert 2 <= len(graph.nodes) <= 48 and 1 <= len(graph.edges) <= 80
    assert graph.completeness["logit_targets"] == 10
    for name, tensor in graph.tensors.items():
        assert _resolve(name).data_ptr() == tensor.data_ptr()
    assert tracer.trace("AB", 32, 48, 80) is graph
    second = Tracer(model, TinyTokenizer(), exact_dictionary(model)).trace("AB", 32, 48, 80)
    assert second.nodes == graph.nodes and second.edges == graph.edges
    assert any(graph.nodes[e["source"]]["kind"] == "token" and
               graph.nodes[e["target"]]["kind"] == "logit" for e in graph.edges)
    minimal = tracer.trace("AB", 0, 1, 0)
    assert len(minimal.nodes) == 1 and len(minimal.edges) == 0
    assert minimal.completeness["max_absolute_residual"] < 1e-9


@pytest.mark.parametrize("imperfect", [False, True])
def test_direct_edges_match_independent_source_finite_differences(imperfect):
    from caliper.interp._frozen import FrozenPath
    from caliper.interp._trace import _DirectEffects
    from .fixture import TinyLM, TinyTokenizer
    model = TinyLM(layers=3, width=12, heads=3, latents=12, k=2)
    dictionary = random_dictionary(model) if imperfect else exact_dictionary(model)
    path = FrozenPath(model, torch.tensor([[1, 2]]), dictionary)
    effects = _DirectEffects(path, dictionary, [1, 2], TinyTokenizer())
    target_indices = effects.logit_nodes[:1] + effects.feature_nodes
    incoming, diagnostics = effects.backward(target_indices)
    assert max(d["absolute_residual"] for d in diagnostics) < 1e-9
    epsilon = 1e-4
    for source, node in enumerate(effects.nodes):
        if node["kind"] == "logit":
            continue
        plus = [path.embedding.clone()] + [x.clone() for x in path.writes]
        minus = [x.clone() for x in plus]
        position = node["position"]
        if node["kind"] == "token":
            write, vector = 0, path.embedding[0, position]
        elif node["kind"] == "feature":
            write = node["layer"] + 1
            vector = node["activation"] * dictionary.layers[node["layer"]].W_dec[node["feature"]]
        else:
            write, vector = node["layer"] + 1, path.errors[node["layer"]][0, position]
        plus[write][0, position] += epsilon * vector
        minus[write][0, position] -= epsilon * vector
        plus_reads, plus_final = path.run(plus[0], plus[1:])
        minus_reads, minus_final = path.run(minus[0], minus[1:])
        for column, target_index in enumerate(target_indices):
            target = effects.target(target_index)
            after = plus_final if target.layer == path.layers else plus_reads[target.layer]
            before = minus_final if target.layer == path.layers else minus_reads[target.layer]
            difference = ((after[0, target.position] - before[0, target.position]) @ target.direction) / (2 * epsilon)
            torch.testing.assert_close(incoming[source, column], difference, atol=1e-6, rtol=0)
            if node["kind"] in ("feature", "error") and node["layer"] >= target.layer:
                assert incoming[source, column] == 0


def test_imperfect_dictionary_and_explicit_bias_accounting():
    from caliper.interp import Tracer
    from .fixture import TinyLM, TinyTokenizer, RMSNorm
    model = TinyLM(layers=2, width=12, heads=3, latents=16, k=3)
    class BiasedNorm(RMSNorm):
        def __init__(self, width):
            super().__init__(width)
            self.bias = torch.nn.Parameter(torch.linspace(-.02, .02, width))
        def forward(self, x):
            return super().forward(x) + self.bias
    for block in model.layers:
        block.input_layernorm = BiasedNorm(12).double()
        block.post_attention_layernorm = BiasedNorm(12).double()
        block.self_attn.v_proj.bias = torch.nn.Parameter(torch.linspace(-.03, .03, 12).double())
        block.self_attn.o_proj.bias = torch.nn.Parameter(torch.linspace(-.01, .01, 12).double())
    model.norm = BiasedNorm(12).double()
    model.lm_head.bias = torch.nn.Parameter(torch.linspace(-.1, .1, 512).double())
    dictionary = random_dictionary(model)
    graph = Tracer(model, TinyTokenizer(), dictionary).trace("A", max_targets=32)
    assert graph.error_share > 0 and graph.feature_share > 0
    assert graph.completeness["max_absolute_residual"] < 1e-9
    targets = graph.completeness["targets"]
    for kind in ("decoder", "encoder", "normalization", "attention", "unembedding"):
        assert any(abs(t["biases"][kind]) > 1e-9 for t in targets)
    assert all(abs(t["incoming"] + sum(t["biases"].values()) - t["expected"]) < 1e-9 for t in targets)


def test_small_real_qwen_trace_with_all_twenty_four_blocks():
    from caliper.interp import Tracer
    from .test_points import qwen
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]])}
        def decode(self, ids):
            return str(ids[0])
    model = qwen(layers=24)
    assert sum(b.block_type == "linear_attention" for b in model.model.layers) == 18
    dictionary = random_dictionary(model, latents=8, k=2)
    graph = Tracer(model, Tokenizer(), dictionary).trace("small real architecture", max_targets=16)
    assert graph.layers == 24
    assert graph.completeness["max_absolute_residual"] < 2e-2
    assert graph.completeness["forward_residual"] < 2e-5
    assert graph.error_share > 0


def test_target_batches_are_independent_of_later_encoders(monkeypatch):
    from caliper.interp._frozen import FrozenPath
    from caliper.interp._trace import _DirectEffects
    from .fixture import TinyLM, TinyTokenizer
    model = TinyLM()
    dictionary = exact_dictionary(model)
    path = FrozenPath(model, torch.tensor([[1, 2, 3]]), dictionary)
    effects = _DirectEffects(path, dictionary, [1, 2, 3], TinyTokenizer())
    def forbidden(*args, **kwargs):
        raise AssertionError("a direct-effect backward must not evaluate later feature activations")
    monkeypatch.setattr(dictionary, "encode", forbidden)
    monkeypatch.setattr(model, "forward", forbidden)
    batch = effects.logit_nodes + effects.feature_nodes[:54]
    batched, _ = effects.backward(batch)
    assert batched.shape[1] == 64
    for column in (0, 9, 10, 30, 63):
        single, _ = effects.backward([batch[column]])
        torch.testing.assert_close(single[:, 0], batched[:, column], atol=1e-9, rtol=0)
    assert all(p.grad is None for p in model.parameters())


def test_trace_full_caps_and_token_limit():
    from caliper.interp import Tracer
    from .fixture import TinyLM, TinyTokenizer
    model = TinyLM()
    tracer = Tracer(model, TinyTokenizer(), exact_dictionary(model))
    graph = tracer.trace("a" * 31)
    assert graph.completeness["feature_targets"] == 512
    assert graph.completeness["target_count"] == 522
    assert len(graph.nodes) <= 512 and len(graph.edges) <= 8192 and len(graph.tokens) == 32
    assert graph.completeness["max_absolute_residual"] < 1e-9
    assert len({n["id"] for n in graph.nodes}) == len(graph.nodes)
    with pytest.raises(ValueError, match="prompt tokens"):
        tracer.trace("a" * 32)


@pytest.mark.parametrize("options", [{"max_targets": -1}, {"max_targets": 513}, {"max_targets": True},
                                     {"max_nodes": 0}, {"max_nodes": 513}, {"max_edges": -1},
                                     {"max_edges": 8193}, {"max_nodes": 2.5}])
def test_invalid_trace_bounds(options):
    from caliper.interp import Tracer
    from .fixture import TinyLM, TinyTokenizer
    model = TinyLM(layers=1)
    with pytest.raises(ValueError):
        Tracer(model, TinyTokenizer(), exact_dictionary(model)).trace("A", **options)


def test_trace_cache_identity_full_state_and_lru(monkeypatch):
    from caliper.interp import Tracer
    from .fixture import TinyLM, TinyTokenizer
    model = TinyLM(layers=1, width=8, heads=2, latents=8, k=2)
    dictionary = exact_dictionary(model)
    tracer = Tracer(model, TinyTokenizer(), dictionary)
    state = []
    invalidations = []
    service = type("Service", (), {"invalidate_source_cache": lambda self: invalidations.append(True)})()
    monkeypatch.setattr(tracer, "_service", lambda: service)
    monkeypatch.setattr(tracer, "_intervention_state", lambda: state)
    initial = tracer.trace("A", 0)
    assert tracer.trace("A", 0) is initial
    # Full specification fields participate, even when the point and kind stay the same.
    state.append({"point": "layers.0.mlp_out", "kind": "add", "coefficient": 1., "vector": [0.] * 8,
                  "positions": [0]})
    changed = tracer.trace("A", 0)
    assert changed is not initial
    state[0]["vector"][2] = .2
    vector_changed = tracer.trace("A", 0)
    assert vector_changed is not changed
    state.clear()
    assert tracer.trace("A", 0) is initial
    before = tracer._forward_cache
    tracer.trace("B", 0)
    assert tracer._forward_cache is not before and len(invalidations) == 2
    assert tracer.trace("A", 0) is initial and tracer._forward_cache is None
    with torch.no_grad():
        model.lm_head.weight[0, 0] += .1
    model_changed = tracer.trace("A", 0)
    assert model_changed is not initial
    with torch.no_grad():
        dictionary.layers[0].b_dec[0] += .1
    assert tracer.trace("A", 0) is not model_changed
    # Eight distinct cache entries bound retained graph storage.
    for prompt in "CDEFGHIJ":
        tracer.trace(prompt, 0)
    assert len(tracer._cache) == 8
    assert initial not in tracer._cache.values()
