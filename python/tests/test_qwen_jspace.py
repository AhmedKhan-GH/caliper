"""The Jacobian lens and J-space interventions on a tiny random Qwen3.5 built from mlx-lm's own model code.

No downloaded weights: four blocks (three linear-attention, one full-attention), 64 wide, 97 tokens. The real
model and lens are checked against an independent torch readout by hand (see qwen_jspace.py).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
qwen3_5 = pytest.importorskip("mlx_lm.models.qwen3_5")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/python"))
# Checks multiply in float64 NumPy: MLX's float32 GPU matmul is good to about 1e-3 relative, so an intervention
# is held to removing (or moving) at least 99% of a coordinate.
import qwen_jspace  # noqa: E402
from qwen_jspace import Intervention, JSpace, project_out, project_out_each, trace  # noqa: E402

WIDTH, VOCAB = 64, 97


def f64(a):
    return np.array(a).astype(np.float64)


def coordinates(x, V):
    """Lens coordinates V^+ x, row by row, in float64."""
    return f64(x) @ np.linalg.pinv(f64(V)).T


def removed(after, before, V):
    """At most 1% of the coordinates along V's columns remain."""
    return np.abs(coordinates(after, V)).max() <= 0.01 * np.abs(coordinates(before, V)).max()


def tiny_model():
    mx.random.seed(3)
    config = dict(model_type="qwen3_5", hidden_size=WIDTH, intermediate_size=128, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=VOCAB,
                  linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
                  linear_value_head_dim=16, linear_conv_kernel_dim=4, full_attention_interval=4,
                  tie_word_embeddings=True, rms_norm_eps=1e-6)
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(dict(model_type="qwen3_5", text_config=config)))
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def space(tmp_path_factory):
    model = tiny_model()
    rng = np.random.default_rng(0)
    path = tmp_path_factory.mktemp("lens") / "lens.npz"
    lenses = {f"J{l}": (np.eye(WIDTH) + 0.1 * rng.standard_normal((WIDTH, WIDTH))).astype(np.float16)
              for l in range(3)}
    lenses["J1"] = np.eye(WIDTH, dtype=np.float16)                 # an identity map: the logit lens
    np.savez(path, source_layers=np.arange(3), n_prompts=np.array(1), d_model=np.array(WIDTH), **lenses)
    return model, JSpace(model, mx, path)


class Tokenizer:
    def decode(self, ids):
        return f"<{ids[0]}>"


def test_forward_matches_the_model_and_visits_every_block(space):
    model, js = space
    tokens = mx.array([[5, 17, 3, 88, 41, 2]])
    seen = []
    ours = js.forward(tokens, model.make_cache(), lambda i, h: seen.append(i) or h)
    theirs = model(tokens, cache=model.make_cache())
    assert seen == [0, 1, 2, 3]
    assert np.allclose(np.array(ours), np.array(theirs), atol=1e-4)


def test_readout_is_the_logit_lens_through_an_identity_jacobian(space):
    _, js = space
    h = mx.random.normal((3, WIDTH))
    assert np.allclose(np.array(js.readout(1, h)), np.array(js.readout(1, h, jacobian=False)), atol=1e-3)
    assert not np.allclose(np.array(js.readout(0, h)), np.array(js.readout(0, h, jacobian=False)), atol=1e-3)
    ids, probs = js.top(js.readout(0, h), 4)
    full = np.array(mx.softmax(js.readout(0, h), axis=-1))
    assert np.array_equal(np.array(ids), np.argsort(-full, axis=-1)[:, :4])
    assert np.allclose(np.array(probs), np.sort(full, axis=-1)[:, ::-1][:, :4], atol=1e-5)


def test_directions_raise_their_own_token_and_are_unit(space):
    _, js = space
    V = js.directions(0, [7, 40])
    assert np.allclose(np.linalg.norm(np.array(V), axis=0), 1, atol=1e-5)
    h = mx.random.normal((WIDTH,)) * 0.01
    before = np.array(js.readout(0, h[None]))[0]
    after = np.array(js.readout(0, (h + 20 * V[:, 0])[None]))[0]
    assert after[7] - before[7] > after.mean() - before.mean()


def test_projections_remove_exactly_the_span():
    V = mx.random.normal((WIDTH, 3))
    x = mx.random.normal((5, WIDTH))
    assert removed(project_out(mx, x, V), x, V)
    rows = mx.random.normal((5, 3, WIDTH))
    each = project_out_each(mx, x, rows)
    for p in range(5):
        assert removed(each[p][None], x[p][None], rows[p].T)


def test_swap_exchanges_two_coordinates_and_keeps_the_rest(space):
    _, js = space
    h = mx.random.normal((1, 4, WIDTH))
    swapped = js.intervene(2, h, [Intervention("swap", 0, 2, tokens=[7, 40])], 0, [set()] * 4)
    V = f64(js.directions(2, [7, 40]))
    before, after = coordinates(h[0], V), coordinates(swapped[0], V)
    assert np.abs(after - before[:, ::-1]).max() <= 0.01 * np.abs(before).max()
    # The rest of the stream, outside the two directions, is unchanged.
    rest = lambda x: f64(x) - coordinates(x, V) @ V.T
    assert np.abs(rest(swapped[0]) - rest(h[0])).max() <= 0.01 * np.abs(f64(h)).max()


def test_inject_remove_and_band_and_positions(space):
    _, js = space
    h = mx.random.normal((1, 4, WIDTH))
    js.mean_norm = {0: 2.0, 1: 2.0, 2: 2.0}
    injected = js.intervene(0, h, [Intervention("inject", 0, 0, tokens=[9], strength=3)], 0, [set()] * 4)
    assert np.allclose(f64(injected) - f64(h), 6 * f64(js.directions(0, [9]))[:, 0][None, None, :], atol=1e-2)
    # Outside its band or its positions an intervention leaves the stream alone.
    assert np.array_equal(np.array(js.intervene(2, h, [Intervention("inject", 0, 1, tokens=[9])], 0, [set()] * 4)), np.array(h))
    only = js.intervene(0, h, [Intervention("remove", 0, 2, tokens=[9, 11], positions=[11])], 10, [set()] * 4)
    changed = np.abs(np.array(only - h)).max(axis=-1)[0]
    assert changed[1] > 1e-4 and changed[0] == 0 and changed[2] == 0 and changed[3] == 0
    assert removed(only[0, 1][None], h[0, 1][None], js.directions(0, [9, 11]))


def test_ablation_removes_the_top_lens_tokens_but_spares_protected_ones(space):
    _, js = space
    h = mx.random.normal((1, 2, WIDTH))
    ids, _ = js.top(js.readout(1, h[0]), 4)
    protect = {ids[0, 0].item()}
    ablated = js.intervene(1, h, [Intervention("ablate", 1, 1, k=3)], 0, [protect, set()])
    first = [t for t in ids[0].tolist() if t not in protect][:3]
    assert removed(ablated[0, 0][None], h[0, 0][None], js.directions(1, first))
    assert removed(ablated[0, 1][None], h[0, 1][None], js.directions(1, ids[1, :3].tolist()))


def test_trace_streams_every_position_and_stops_at_the_end_token(space):
    from caliper.requests import MAX_MESSAGE, encode
    from mlx_lm.sample_utils import make_sampler
    _, js = space
    tokens = list(range(1, 71))                              # more than one prompt chunk
    events = []
    result = trace(js, Tokenizer(), tokens, max_new_tokens=5, sampler=make_sampler(temp=0.0), k=3,
                   interventions=[Intervention("ablate", 0, 2, k=2)], emit=events.append,
                   cancelled=threading.Event(), eos=set())
    assert result["tokens"] == 5 and result["finish_reason"] == "length"
    assert [e["first"] for e in events] == [0, 64, 70, 71, 72, 73, 74]
    words = {}
    for e in events:
        n, L, k = e["count"], len(e["layers"]), e["k"]
        assert len(e["jlens"]) == len(e["jprob"]) == len(e["logit"]) == len(e["lprob"]) == n * L * k
        assert len(e["output"]) == len(e["oprob"]) == n * k
        assert e["prompt"] == len(tokens)
        words.update(e["words"])
        assert all(str(i) in words for i in e["tokens"] + e["jlens"] + e["logit"] + e["output"])
        assert len(encode(dict(id=1, done=False, event=e))) < MAX_MESSAGE
    # Greedy: each generated token is the previous position's top output.
    for before, after in zip(events[1:], events[2:]):
        assert after["tokens"][0] == before["output"][(before["count"] - 1) * before["k"]]
    stop = events[2]["output"][0]
    events.clear()
    result = trace(js, Tokenizer(), tokens, max_new_tokens=5, sampler=make_sampler(temp=0.0), k=3,
                   interventions=[Intervention("ablate", 0, 2, k=2)], emit=events.append,
                   cancelled=threading.Event(), eos={stop})
    assert result["tokens"] == 0 and result["finish_reason"] == "stop"


@pytest.mark.parametrize("item", [dict(kind="erase"), dict(kind="ablate", first=2, last=1),
                                  dict(kind="ablate", last=3), dict(kind="remove"),
                                  dict(kind="remove", tokens=[VOCAB]), dict(kind="swap", tokens=[4, 4]),
                                  dict(kind="swap", tokens=[4]), dict(kind="inject", tokens=[4], strength=1e9),
                                  dict(kind="ablate", positions=[-1]), dict(kind="ablate", k=0)])
def test_host_refuses_malformed_interventions(space, item):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qwen_inspector_for_jspace", Path(__file__).resolve().parents[2] / "examples/python/qwen_inspector.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    host = module.Inspector.__new__(module.Inspector)
    host._jspace = space[1]
    assert host.interventions(dict(interventions=[dict(kind="ablate"), dict(kind="swap", tokens=[4, 5])]))
    with pytest.raises(ValueError):
        host.interventions(dict(interventions=[item]))


def test_trace_streams_the_attention_patterns_of_full_attention_layers(space):
    import base64
    from mlx_lm.sample_utils import make_sampler
    model, js = space
    tokens = list(range(1, 13))
    events = []
    trace(js, Tokenizer(), tokens, max_new_tokens=2, sampler=make_sampler(temp=0.0), k=3, interventions=[],
          emit=events.append, cancelled=threading.Event(), eos=set(), attention=True)
    assert all(e["attention_layers"] == [3] and e["heads"] == 4 for e in events)    # one full-attention block
    for e in events:
        data = np.frombuffer(base64.b64decode(e["attention"]), np.uint8)
        weights = data.reshape(e["count"], 1, 4, e["keys"]).astype(np.float64) / 255
        for i in range(e["count"]):
            position = e["first"] + i
            # Causal: nothing beyond the position itself; each row is a distribution.
            assert np.all(weights[i, :, :, position + 1:] == 0)
            assert np.allclose(weights[i].sum(axis=-1), 1, atol=e["keys"] / 255)
    assert [e["keys"] for e in events] == [12, 13, 14]
    # Weights times values, gated and projected, is the block's own attention output.
    layer = model.language_model.model.layers[3]
    cache = model.make_cache()
    h = js.text.embed_tokens(mx.array([tokens]))
    for l, c in zip(js.layers[:3], cache[:3]):
        h = l(h, mask=None, cache=c)
    x = layer.input_layernorm(h)
    fused = layer.self_attn(x, "causal", cache[3])
    from qwen_jspace import attention_weights
    w = attention_weights(layer.self_attn, x, cache[3], 0)
    attn = layer.self_attn
    values = mx.repeat(cache[3].keys_and_values()[1][0], attn.num_attention_heads // attn.num_key_value_heads, axis=0)
    out = (w @ values.astype(mx.float32)).transpose(1, 0, 2).reshape(1, len(tokens), -1)
    gate = mx.split(attn.q_proj(x).reshape(1, len(tokens), attn.num_attention_heads, -1), 2, axis=-1)[1]
    mine = attn.o_proj(out * mx.sigmoid(gate.reshape(1, len(tokens), -1)))
    assert np.abs(f64(mine) - f64(fused)).max() <= 0.02 * np.abs(f64(fused)).max()


def test_run_to_max_bans_the_end_tokens(space):
    from mlx_lm.sample_utils import make_sampler
    _, js = space
    tokens = list(range(1, 9))
    events = []
    trace(js, Tokenizer(), tokens, max_new_tokens=6, sampler=make_sampler(temp=0.0), k=2, interventions=[],
          emit=events.append, cancelled=threading.Event(), eos=set())
    first = events[1]["tokens"][0]                    # the greedy first token
    # With that token as the end, the run stops at once; banned, it writes all six.
    events.clear()
    stopped = trace(js, Tokenizer(), tokens, max_new_tokens=6, sampler=make_sampler(temp=0.0), k=2,
                    interventions=[], emit=events.append, cancelled=threading.Event(), eos={first})
    assert stopped["tokens"] == 0 and stopped["finish_reason"] == "stop"
    events.clear()
    full = trace(js, Tokenizer(), tokens, max_new_tokens=6, sampler=make_sampler(temp=0.0), k=2,
                 interventions=[], emit=events.append, cancelled=threading.Event(), eos={first}, run_to_max=True)
    assert full["tokens"] == 6 and full["finish_reason"] == "length"
    assert first not in [e["tokens"][0] for e in events[1:]]
