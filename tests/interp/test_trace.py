"""Frozen paths, direct source effects, completeness, and graph contracts."""
import json

import pytest
import torch

from caliper.interp._graph import Graph


def test_trace_memory_records_keep_recent_stages_and_lifetime_peak(monkeypatch):
    from caliper.interp import _frozen
    from caliper.interp._memory import MemoryGuard as ProcessMemoryGuard
    sample = {"current": 800, "peak": 1000}
    def deterministic_guard(limit_gb, device, **kwargs):
        return ProcessMemoryGuard(limit_gb, device, pressure=lambda: 80.,
                                  resident=lambda: 50, footprint=lambda: sample["current"],
                                  peak=lambda: sample["peak"], driver=lambda: 0, clock=lambda: 0.)
    monkeypatch.setattr(_frozen, "ProcessMemoryGuard", deterministic_guard)
    monkeypatch.setattr(_frozen.MemoryGuard, "record_limit", 4)
    guard = _frozen.MemoryGuard("cpu")
    assert guard.records[0]["stage"] == "trace start"
    sample["peak"] = 0
    # Six scalar checks evict the original peak record without any model trace.
    for stage in range(6):
        sample["current"] = 100 + stage
        assert guard.check(f"stage {stage}") == sample["current"]
        assert len(guard.records) <= 4
    assert [record["stage"] for record in guard.records] == ["stage 2", "stage 3", "stage 4", "stage 5"]
    assert guard.records[-1]["process_bytes"] == 105
    assert guard._guard.peak == 1000
    assert all(record["peak_memory_bytes"] == 1000 for record in guard.records)
    assert all(record["allocated_bytes"] == record["driver_bytes"] == 0 for record in guard.records)


def test_trace_memory_guard_uses_physical_footprint_and_prior_peak(monkeypatch):
    from caliper.interp import _frozen
    from caliper.interp._memory import CleanStop
    monkeypatch.setattr(_frozen, "free_percent", lambda: 80.)
    guard = _frozen.MemoryGuard(torch.device("cpu"))
    guard._guard.resident = lambda: 100
    guard._guard.footprint = lambda: 12_000_000_001
    guard._guard.peak_reader = lambda: 100
    with pytest.raises(CleanStop, match="memory limit"):
        guard.check("injected physical cap")
    guard._guard.footprint = lambda: 100
    with pytest.raises(CleanStop, match="memory limit"):
        guard.check("prior peak remains over cap")


def test_target_batch_admission_reduces_before_allocation(monkeypatch):
    from caliper.interp import _frozen
    from caliper.interp._memory import CleanStop
    monkeypatch.setattr(_frozen, "free_percent", lambda: 80.)
    guard = _frozen.MemoryGuard(torch.device("cpu"))
    # Only the accounting device changes. No MPS tensor or dispatch exists in this test.
    guard.device = torch.device("mps")
    monkeypatch.setattr(guard, "check", lambda stage: 9_000_000_000)
    admitted = guard.batch_size(64, layers=24, tokens=32, width=1024, mixer_width=2048)
    assert 1 <= admitted < 64
    monkeypatch.setattr(guard, "check", lambda stage: 11_800_000_000)
    with pytest.raises(CleanStop, match="no target batch fits"):
        guard.batch_size(64, layers=24, tokens=32, width=1024, mixer_width=2048)


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
    from .test_dictionary import fixture_manifest
    layers = [TopKTranscoder(**{name: getattr(block.mlp, name).detach().clone()
                               for name in ("W_enc", "b_enc", "W_dec", "b_dec")},
                            k=block.mlp.k) for block in model.layers]
    return Dictionary(fixture_manifest(model), layers)


def qwen_for_trace(layers=4):
    from caliper.interp._dictionary import weights_hash
    from .test_points import qwen
    model = qwen(layers=layers)
    # Synthetic checkpoint metadata identifies these random CPU test weights.
    model.config._name_or_path = "caliper-test/random-qwen"
    model.config._commit_hash = weights_hash(model).split(":", 1)[1][:40]
    return model


def random_dictionary(model, *, latents=12, k=3):
    from caliper.interp._dictionary import Dictionary, TopKTranscoder, model_identity
    from caliper.interp._points import semantic_points
    from .test_dictionary import fixture_manifest
    width = model.config.hidden_size
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        layers = [TopKTranscoder(W_enc=torch.randn(latents, width) * .1,
                                b_enc=torch.randn(latents) * .1,
                                W_dec=torch.randn(latents, width) * .03,
                                b_dec=torch.randn(width) * .01, k=k)
                  for _ in range(model.config.num_hidden_layers)]
    if model.config.model_type == "caliper_tiny":
        metadata = fixture_manifest(model)
    else:
        metadata = {"format_version": 1, "model_id": model.config._name_or_path,
                    "model_hash": model_identity(model), "revision": model.config._commit_hash,
                    "layers": model.config.num_hidden_layers, "width": width,
                    "activation": "topk",
                    "training": {"dataset_id": "fixture", "dataset_config": "random-inputs",
                                 "dataset_revision": "seed-29", "split": "train", "partitions": {},
                                 "tokens": 0, "elapsed_seconds": 0, "peak_memory_bytes": 0},
                    "quality": {"fvu": None, "dead_fraction": None, "cross_entropy_delta": None,
                                "evaluated_tokens": 0}}
    probes = semantic_points(model)
    metadata.update(latents=latents, k=k,
                    read_points=[p["point"] for p in probes if p["probe"] == "mlp_in"],
                    write_points=[p["point"] for p in probes if p["probe"] == "mlp_out"])
    return Dictionary(metadata, layers).to(dtype=next(model.parameters()).dtype)


@pytest.mark.parametrize("expected_hash_only", [False, True])
@pytest.mark.parametrize("family", ["fixture", "qwen"])
def test_trace_rejects_loaded_dictionary_for_other_model(tmp_path, monkeypatch, expected_hash_only, family):
    from safetensors.torch import save_file
    from caliper.interp import Tracer, load_dictionary
    from caliper.interp._frozen import FrozenPath
    from .fixture import TinyLM, TinyTokenizer
    if family == "fixture":
        options = dict(layers=1, width=8, heads=2, latents=8, k=2)
        model, other = TinyLM(**options), TinyLM(seed=11, **options)
        source = exact_dictionary(model)
    else:
        model, other = qwen_for_trace(layers=1), qwen_for_trace(layers=1)
        other.config._commit_hash = "f" * 40
        source = random_dictionary(model, latents=8, k=2)
    (tmp_path / "dictionary.json").write_text(json.dumps(source.metadata), encoding="utf-8")
    save_file(source.layers[0].state_dict(), tmp_path / "layer_00.safetensors")
    identity = {"model_hash": source.metadata["model_hash"]} if expected_hash_only else {"model": model}
    dictionary = load_dictionary(tmp_path, **identity)
    assert Tracer(model, TinyTokenizer(), dictionary).dictionary is dictionary
    assert [(name, tuple(p.shape)) for name, p in model.named_parameters()] == [
        (name, tuple(p.shape)) for name, p in other.named_parameters()]
    def forbidden(*args, **kwargs):
        raise AssertionError("identity refusal must precede the model forward")
    monkeypatch.setattr(other, "forward", forbidden)
    with pytest.raises(ValueError, match="dictionary model identity mismatch"):
        Tracer(other, TinyTokenizer(), dictionary)
    with pytest.raises(ValueError, match="dictionary model identity mismatch"):
        FrozenPath(other, torch.tensor([[1, 2]]), dictionary)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in other.modules())


@pytest.mark.parametrize("field,value,message", [
    ("model_hash", None, "identity mismatch"),
    ("read_points", ["other.0.mlp_in"], "read_points differ from model probes"),
    ("write_points", ["other.0.mlp_out"], "write_points differ from model probes"),
    ("width", 9, "dimensions differ"),
    ("latents", 9, "shape differs"),
    ("k", 1, "TopK differs"),
])
def test_trace_rejects_incompatible_dictionary_metadata(monkeypatch, field, value, message):
    from caliper.interp import Tracer
    from caliper.interp._frozen import FrozenPath
    from .fixture import TinyLM, TinyTokenizer
    model = TinyLM(layers=1, width=8, heads=2, latents=8, k=2)
    dictionary = exact_dictionary(model)
    dictionary.metadata[field] = value
    def forbidden(*args, **kwargs):
        raise AssertionError("compatibility refusal must precede the model forward")
    monkeypatch.setattr(model, "forward", forbidden)
    with pytest.raises(ValueError, match=message):
        Tracer(model, TinyTokenizer(), dictionary)
    with pytest.raises(ValueError, match=message):
        FrozenPath(model, torch.tensor([[1, 2]]), dictionary)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


@pytest.mark.parametrize("changed_resource", ["model", "dictionary"])
def test_trace_rechecks_compatibility_before_cache_use(monkeypatch, changed_resource):
    from caliper.interp import Tracer
    from .fixture import TinyLM, TinyTokenizer
    options = dict(layers=1, width=8, heads=2, latents=8, k=2)
    model = TinyLM(**options)
    dictionary = exact_dictionary(model)
    tracer = Tracer(model, TinyTokenizer(), dictionary)
    tracer.trace("A", 0)
    if changed_resource == "model":
        tracer.model = TinyLM(seed=11, **options)
    else:
        dictionary.metadata.clear()
    def forbidden(*args, **kwargs):
        raise AssertionError("changed identity must fail before another model forward")
    monkeypatch.setattr(tracer.model, "forward", forbidden)
    with pytest.raises(ValueError, match="identity mismatch"):
        tracer.trace("A", 0)


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


@pytest.mark.parametrize("delta_biases", [False, True])
def test_real_qwen_modules_forward_and_linearity(delta_biases):
    from caliper.interp._frozen import FrozenPath
    model = qwen_for_trace()
    # Cover nonzero full-attention biases with actual primary module code.
    full = model.model.layers[3].self_attn
    for projection in (full.v_proj, full.o_proj):
        projection.bias = torch.nn.Parameter(torch.linspace(-.02, .02, projection.out_features))
    if delta_biases:
        for block in model.model.layers[:3]:
            delta = block.linear_attn
            for projection in (delta.in_proj_qkv, delta.out_proj):
                projection.bias = torch.nn.Parameter(torch.linspace(-.02, .02, projection.out_features))
            delta.conv1d.bias = torch.nn.Parameter(torch.linspace(-.01, .01, delta.conv_dim))
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
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]])}
        def decode(self, ids):
            return str(ids[0])
    model = qwen_for_trace(layers=24)
    assert sum(b.block_type == "linear_attention" for b in model.model.layers) == 18
    dictionary = random_dictionary(model, latents=8, k=2)
    graph = Tracer(model, Tokenizer(), dictionary).trace("small real architecture", max_targets=16)
    assert graph.layers == 24
    assert graph.completeness["max_absolute_residual"] < 2e-2
    assert graph.completeness["direct_forward_residual"] < 2e-5
    assert graph.completeness["replacement_forward_residual"] < 2e-2
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
    executor = type("Executor", (), {"invalidate_cache": lambda self: invalidations.append(True)})()
    service = type("Service", (), {"executor": executor})()
    monkeypatch.setattr(tracer, "_service", lambda: service)
    monkeypatch.setattr(tracer, "_intervention_state", lambda *_: state)
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
    with pytest.raises(ValueError, match="identity mismatch"):
        tracer.trace("A", 0)
    # A new dictionary must establish identity for the changed fixture weights.
    dictionary = tracer.dictionary = exact_dictionary(model)
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


@pytest.mark.parametrize("new_limits", [False, True])
def test_dictionary_alias_write_invalidates_graph_and_frozen_caches(new_limits):
    from caliper.interp import Tracer
    from .fixture import TinyLM, TinyTokenizer

    model = TinyLM(layers=2, width=8, heads=2, latents=8, k=2)
    dictionary = exact_dictionary(model)
    tracer = Tracer(model, TinyTokenizer(), dictionary)
    first = tracer.trace("AB", max_targets=8, max_nodes=32)
    path = tracer._forward_cache[1]
    weight = dictionary.layers[0].W_dec
    version = weight._version
    weight.data.zero_()
    assert weight._version == version
    options = {"max_targets": 4 if new_limits else 8, "max_nodes": 24 if new_limits else 32}
    changed = tracer.trace("AB", **options)
    assert changed is not first and tracer._forward_cache[1] is not path
    fresh = Tracer(model, TinyTokenizer(), dictionary).trace("AB", **options)
    assert changed.completeness == fresh.completeness
    for name, tensor in fresh.tensors.items():
        torch.testing.assert_close(changed.tensors[name], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("real_qwen", [False, True])
def test_full_replacement_keeps_active_masks_and_constant_errors(real_qwen):
    from caliper.interp._frozen import FrozenPath
    from .fixture import TinyLM
    model = qwen_for_trace() if real_qwen else TinyLM(layers=2)
    dictionary = random_dictionary(model)
    path = FrozenPath(model, torch.tensor([[1, 2, 3]]), dictionary)
    x = path.embedding
    # Large changes must not select new TopK entries or reapply ReLU.
    zero = path.replacement(torch.zeros_like(x), dictionary)[1]
    first = path.replacement(3 * x, dictionary)[1]
    second = path.replacement(-2 * x, dictionary)[1]
    combined = path.replacement(x, dictionary)[1]
    torch.testing.assert_close(first + second - zero, combined, atol=2e-5 if real_qwen else 1e-9, rtol=1e-5 if real_qwen else 0)
    assert all(not e.requires_grad for e in path.errors)


def test_probability_coverage_and_no_invented_path():
    from caliper.interp import Tracer
    from .fixture import TinyLM, TinyTokenizer
    model = TinyLM(layers=1)
    model.lm_head.bias = torch.nn.Parameter(torch.zeros(512, dtype=torch.float64))
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias[42] = 20
    graph = Tracer(model, TinyTokenizer(), exact_dictionary(model)).trace("A", 0)
    assert graph.completeness["logit_targets"] == 1
    assert graph.completeness["logit_probability_mass"] >= .95
    assert graph.nodes[0]["token_id"] == 42 and graph.nodes[0]["kind"] == "logit"
    assert graph.edges == [] and graph.error_share == graph.feature_share == 0
    assert graph.completeness["max_absolute_residual"] < 1e-9


def test_input_validation_and_inference_context():
    from caliper.interp import Tracer
    from .fixture import TinyLM, TinyTokenizer
    model = TinyLM(layers=1)
    dictionary = exact_dictionary(model)
    tracer = Tracer(model, TinyTokenizer(), dictionary)
    with torch.inference_mode():
        graph = tracer.trace("A", 0)
    assert graph.completeness["max_absolute_residual"] < 1e-9
    with pytest.raises(TypeError, match="string"):
        tracer.trace([1, 2])
    model.train()
    with pytest.raises(ValueError, match="evaluation"):
        tracer.trace("A", 0)
    model.eval()
    for ids in (torch.tensor([1]), torch.tensor([[1.]]), torch.tensor([[512]]), torch.empty((1, 0), dtype=torch.long)):
        tracer.tokenizer = lambda *args, **kwargs: {"input_ids": ids}
        with pytest.raises(ValueError):
            tracer.trace("invalid")
    with pytest.raises(ValueError, match="dictionary"):
        Tracer(model, TinyTokenizer(), None)


def test_mid_forward_intervention_change_is_not_cached(monkeypatch):
    from caliper.interp import Tracer
    from .fixture import TinyLM, TinyTokenizer
    model = TinyLM(layers=1)
    tracer = Tracer(model, TinyTokenizer(), exact_dictionary(model))
    state = []
    monkeypatch.setattr(tracer, "_intervention_state", lambda *_: list(state))
    def changed(*args):
        state.append("changed")
    handle = model.register_forward_hook(changed)
    try:
        with pytest.raises(RuntimeError, match="state changed"):
            tracer.trace("A", 0)
    finally:
        handle.remove()
    assert not tracer._cache and tracer._forward_cache is None


def test_nested_source_forward_preserves_outer_frozen_capture():
    from caliper.interp._frozen import FrozenPath
    from .fixture import TinyLM
    model = TinyLM(layers=2)
    dictionary = exact_dictionary(model)
    baseline = FrozenPath(model, torch.tensor([[1, 2, 3]]), dictionary)
    nested = False
    def clean_source(mod, args, output):
        nonlocal nested
        if not nested:
            nested = True
            try:
                model(torch.tensor([[9, 8]]))
            finally:
                nested = False
    handle = model.layers[0].mlp.register_forward_hook(clean_source)
    try:
        actual = FrozenPath(model, torch.tensor([[1, 2, 3]]), dictionary)
    finally:
        handle.remove()
    assert actual.forward_residual < 1e-9
    for a, b in zip(actual.inputs + actual.outputs, baseline.inputs + baseline.outputs):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_m8_live_service_cache_clamp_and_nested_patch(monkeypatch):
    from caliper import _controls
    if not hasattr(_controls.Controls, "intervention_service"):
        pytest.skip("M8 service integration requires the M8 controls dependency")
    from caliper.interp import Tracer
    from .fixture import TinyLM, TinyTokenizer
    model, tokenizer = TinyLM(layers=2), TinyTokenizer()
    dictionary = exact_dictionary(model)
    from .test_dictionary import fixture_manifest
    dictionary.metadata = fixture_manifest(model)
    controls = _controls.Controls()
    monkeypatch.setattr(_controls, "CONTROLS", controls)
    service = controls.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    tracer = Tracer(model, tokenizer, dictionary)
    try:
        baseline = tracer.trace("AB", 16)
        service.set({"kind": "clamp", "point": "layers.0.mlp_out", "feature": 0, "value": 1.})
        # Pending changes must not invalidate the graph of the actual current effects.
        assert tracer.trace("AB", 16) is baseline
        controls.step()
        clamped = tracer.trace("AB", 16)
        assert clamped is not baseline and clamped.completeness["max_absolute_residual"] < 1e-9
        with service.suspend():
            suspended = tracer.trace("AB", 16)
        assert suspended is not clamped
        for name in baseline.tensors:
            torch.testing.assert_close(suspended.tensors[name], baseline.tensors[name], atol=0, rtol=0)
        service.clear()
        controls.step()
        assert tracer.trace("AB", 16) is baseline
        service.set({"kind": "patch", "point": "layers.0.mlp_out", "source_prompt": "CD"})
        controls.step()
        patched = tracer.trace("AB", 16)
        assert patched.completeness["max_absolute_residual"] < 1e-9 and service.executor._cache
        other = tracer.trace("EF", 16)
        assert other.completeness["max_absolute_residual"] < 1e-9
        assert tracer.trace("AB", 16) is patched
        # A cache hit for a new external prompt still releases the clean-source cache.
        assert not service.executor._cache
        # A rebound intervention dictionary changes the same clamp specification's effect.
        other_dictionary = exact_dictionary(model)
        other_dictionary.metadata = fixture_manifest(model)
        with torch.no_grad():
            other_dictionary.layers[0].W_dec[0].mul_(2)
        service.configure(dictionary=other_dictionary, tokenizer=tokenizer)
        service.set({"kind": "clamp", "point": "layers.0.mlp_out", "feature": 0, "value": 1.})
        controls.step()
        # A different decoder basis cannot use the original trace's feature coefficients.
        with pytest.raises(ValueError, match="same dictionary instance"):
            tracer.trace("AB", 16)
        rebound = Tracer(model, tokenizer, other_dictionary, controls=controls).trace("AB", 16)
        assert rebound.completeness["max_absolute_residual"] < 1e-9
    finally:
        service.remove()


@pytest.mark.parametrize("legacy_owner", [True, False])
def test_trace_uses_separate_controls_owner_for_ablation(monkeypatch, legacy_owner):
    from caliper import _controls
    from caliper.interp import Tracer
    from caliper.interp._trace import _versions
    from .fixture import TinyLM, TinyTokenizer
    if legacy_owner:
        # M7 exposes the ablation owner without the M8 service property.
        monkeypatch.delattr(_controls.Controls, "intervention_service", raising=False)
    owner = _controls.Controls()
    monkeypatch.setattr(_controls, "CONTROLS", _controls.Controls())
    model = TinyLM(layers=1, width=8, heads=2, latents=8, k=2)
    watched = owner.interventions_on(model)
    tracer = Tracer(model, TinyTokenizer(), exact_dictionary(model), controls=owner)
    forwards = []
    handle = model.register_forward_hook(lambda *args: forwards.append(True))
    try:
        assert tracer.controls is owner and tracer.controls is not _controls.CONTROLS
        initial_state = tracer._intervention_state()
        baseline = tracer.trace("A", 0)
        baseline_logits = tracer._forward_cache[1].logits.clone()
        assert tracer.trace("A", 0) is baseline and len(forwards) == 1
        versions = _versions(model)
        with owner.lock:
            assert owner._intervene({"set": {"point": "layers.0.mlp", "op": "ablate"}}) is None
        owner.step()
        state = tracer._intervention_state()
        assert state != initial_state
        records = state if isinstance(state, list) else state["specifications"]
        assert len(records) == 1
        assert records[0]["point"] == "layers.0.mlp" and records[0]["op"] == "ablate"
        assert _versions(model) == versions
        ablated = tracer.trace("A", 0)
        assert ablated is not baseline and ablated.trace_id != baseline.trace_id
        assert len(forwards) == 2
        assert not torch.equal(tracer._forward_cache[1].logits, baseline_logits)
        assert ablated.completeness["max_absolute_residual"] < 1e-9
        assert tracer.trace("A", 0) is ablated and len(forwards) == 2
        with owner.lock:
            assert owner._intervene({"clear_all": True}) is None
        owner.step()
        assert tracer._intervention_state() == initial_state
        assert tracer.trace("A", 0) is baseline and len(forwards) == 2
    finally:
        handle.remove()
        watched.remove()


def test_full_json_export_keeps_all_records_and_diagnostics():
    nodes = [{"id": f"token:0:{i}", "kind": "token", "layer": -1,
              "position": 0, "token_id": i, "activation": 1.,
              "influence": 0., "effect": 0.} for i in range(300)]
    edges = [{"source": 0, "target": i, "weight": -float(i)} for i in range(300)]
    graph = Graph.build(nodes=nodes, edges=edges, tokens=[{"text": "x"}], layers=1,
                        completeness={"targets": [{"id": "all-source-accounting"}]},
                        omitted_influence=0., error_share=0., feature_share=0.,
                        trace_id="full", device="cpu", settings={"profile": "circuit_tracer_82996f14"})
    exported = json.loads(graph.to_json())
    assert len(exported["nodes"]) == len(exported["edges"]) == 300
    assert exported["completeness"]["targets"] == [{"id": "all-source-accounting"}]
    assert exported["settings"]["profile"] == "circuit_tracer_82996f14"
    assert exported["edges"][-1]["weight"] == -299
    assert graph.tensors["trace.edge_f"][299, 0] == -299
    assert graph.tensors["trace.node_i"].shape == (8192, 4)
    assert graph.tensors["trace.edge_i"].shape == (262144, 2)
    assert len(graph.read()["nodes"]) == 128
    assert len(graph.read()["edges"]) == 256
    assert graph.read(node_offset=299, edge_offset=299)["nodes"] == nodes[299:]
    assert len(json.loads(graph.to_json(node_limit=1))["nodes"]) == 1


def test_reference_trace_preserves_full_sources_and_logical_queues(monkeypatch):
    from caliper.interp._trace import Tracer
    from .fixture import TinyLM, TinyTokenizer
    with torch.random.fork_rng():
        torch.manual_seed(73)
        model = TinyLM(layers=2, width=8, heads=2, latents=12, k=4).cpu().eval()
    tracer = Tracer(model, TinyTokenizer(), random_dictionary(model, latents=12, k=4))
    graph = tracer.trace("abc", max_targets=8, profile="circuit_tracer_82996f14",
                         max_n_logits=2, batch_size=2, update_interval=2,
                         node_threshold=1., edge_threshold=1.)
    assert graph.settings["profile"] == "circuit_tracer_82996f14"
    assert graph.settings["objective"] == "centered_pre_softcap_logit"
    assert graph.settings["batch_size"] == 2
    assert graph.completeness["source_scope"] == "full_sources_before_reference_projection"
    assert not any(node["kind"] in ("feature", "error") and node["position"] == 0 for node in graph.nodes)
    assert all("reference_projection" in row for row in graph.completeness["targets"])
    assert graph is tracer.trace("abc", max_targets=8, profile="circuit_tracer_82996f14",
                                max_n_logits=2, batch_size=2, update_interval=2,
                                node_threshold=1., edge_threshold=1.)
    other = Tracer(model, TinyTokenizer(), tracer.dictionary)
    admitted = other.guard.batch_size
    monkeypatch.setattr(other.guard, "batch_size", lambda requested, **kw: min(1, admitted(requested, **kw)))
    split = other.trace("abc", max_targets=8, profile="circuit_tracer_82996f14",
                        max_n_logits=2, batch_size=2, update_interval=2,
                        node_threshold=1., edge_threshold=1.)
    assert graph.settings["logical_queues"] == split.settings["logical_queues"]
    assert set(split.settings["physical_batches"]) == {1}
    assert [n["id"] for n in graph.nodes] == [n["id"] for n in split.nodes]
    torch.testing.assert_close(torch.tensor([e["weight"] for e in graph.edges]),
                               torch.tensor([e["weight"] for e in split.edges]))
    changed = tracer.trace("abc", max_targets=8, profile="circuit_tracer_82996f14",
                           max_n_logits=2, batch_size=2, update_interval=1,
                           node_threshold=1., edge_threshold=1.)
    assert changed.trace_id != graph.trace_id


@pytest.mark.parametrize("setting", ["max_n_logits", "desired_logit_prob", "node_threshold", "edge_threshold",
                                     "batch_size", "update_interval", "profile"])
def test_trace_settings_enter_cache_identity(setting):
    from caliper.interp._trace import Tracer
    from .fixture import TinyLM, TinyTokenizer
    model = TinyLM(layers=1, width=8, heads=2, latents=8, k=2).cpu().eval()
    tracer = Tracer(model, TinyTokenizer(), exact_dictionary(model))
    first = tracer.trace("abc", max_targets=4)
    value = {"max_n_logits": 2, "desired_logit_prob": .8, "node_threshold": .7,
             "edge_threshold": .9, "batch_size": 2, "update_interval": 2,
             "profile": "circuit_tracer_82996f14"}[setting]
    second = tracer.trace("abc", max_targets=4, **{setting: value})
    assert first.trace_id != second.trace_id
    assert second.settings[setting] == value


def test_reference_schedule_summary_respects_protocol_bounds():
    from caliper._interpret_protocol import bounded
    graph = Graph.build(nodes=[], edges=[], tokens=[{"text": "x"}], layers=1,
                        completeness={"targets": []}, omitted_influence=0., error_share=0.,
                        feature_share=0., trace_id="schedule", device="cpu",
                        settings={"profile": "circuit_tracer_82996f14", "batch_size": 512,
                                  "logical_queues": [[str(i) for i in range(5000)]],
                                  "physical_batches": [1] * 5000})
    page = graph.read()
    assert bounded(page)
    assert page["settings"]["physical_batch_count"] == 5000
    assert page["settings"]["logical_queue_count"] == 1
    assert "logical_queues" not in page["settings"]
    assert len(json.loads(graph.to_json())["settings"]["logical_queues"][0]) == 5000


def test_json_export_rechecks_the_caller_guard():
    from caliper.interp._memory import CleanStop
    class Guard:
        stopped = False
        def reserve(self, stage, amount):
            if self.stopped:
                raise CleanStop("supplied memory limit")
    guard = Guard()
    graph = Graph.build(nodes=[], edges=[], tokens=[{"text": "x"}], layers=1,
                        completeness={}, omitted_influence=0., error_share=0., feature_share=0.,
                        trace_id="guard", device="cpu", memory=guard)
    graph.to_json()
    guard.stopped = True
    with pytest.raises(CleanStop, match="supplied memory limit"):
        graph.to_json()
    with pytest.raises(CleanStop, match="supplied memory limit"):
        graph.publish()
