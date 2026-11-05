"""The Jacobian lens and J-space interventions for the Qwen inspector (MLX).

Follows Anthropic's "Verbalizable Representations Form a Global Workspace in Language Models" (Transformer
Circuits, 2026) and its reference code (github.com/anthropics/jacobian-lens):

- The residual stream h_l is a decoder block's output. The lens maps it into the last block's basis with the
  average Jacobian J_l = E[dh_final / dh_l] and reads it like the model does:
  lens_l(h) = unembed(final_norm(J_l h)). With J_l = I it is the plain logit lens.
- The J-lens vector of token t at layer l is the direction in h_l that raises t's lens logit:
  J_l^T (g * W_U[t]), with the final norm's gain g folded into the unembedding, unit length.
- J-space ablation projects out, at every position and band layer, the k most active J-lens vectors
  (top k of the lens readout), except tokens in the clean run's own top outputs.
- A swap exchanges two lens coordinates, c = V^+ h, h' = h + V(swap(c) - c), leaving the rest of h alone.
- An injection adds a J-lens vector scaled by the layer's mean residual norm times a strength.

The lens itself (J_l for every block but the last) is fitted by the reference code; Neuronpedia publishes one
for Qwen3.5-0.8B (neuronpedia/jacobian-lens, qwen3.5-0.8b/jlens), converted here to lens.npz.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import base64
from pathlib import Path

import numpy as np

DEFAULT_LENS = Path.home() / "PycharmProjects/weight-space/models/Qwen3.5-0.8B-jlens/lens.npz"


@dataclass
class Intervention:
    """One change to the residual stream at band layers [first, last] and the given positions.

    kind: "ablate" (J-space ablation: the k most active lens tokens), "remove" (project out the given tokens'
    directions), "swap" (exchange the coordinates of tokens[0] and tokens[1]) or "inject" (add tokens[0]'s
    direction at strength). positions None means every position, including generated ones.
    """
    kind: str
    first: int
    last: int
    tokens: list[int] = field(default_factory=list)
    k: int = 10
    strength: float = 1.0
    positions: list[int] | None = None


class JSpace:
    def __init__(self, model, mx, lens_path: Path = DEFAULT_LENS):
        self.mx = mx
        self.language = model.language_model
        self.text = model.language_model.model
        self.layers = self.text.layers
        lens = np.load(lens_path)
        self.source_layers = [int(layer) for layer in lens["source_layers"]]
        if int(lens["d_model"]) != self.text.embed_tokens.weight.shape[1]:
            raise ValueError("lens width does not match the model")
        self.J = {layer: mx.array(lens[f"J{layer}"]).astype(mx.float32) for layer in self.source_layers}
        self.n_prompts = int(lens["n_prompts"])
        from mlx_lm.models import qwen3_5
        self._attention_mask = qwen3_5.create_attention_mask
        self._ssm_mask = qwen3_5.create_ssm_mask
        self.mean_norm = {}

    # -- the model, block by block ------------------------------------------------------------------------
    def unembed(self, h):
        """Final norm, then the (tied) unembedding: what the model does to its last residual."""
        normed = self.text.norm(h)
        if getattr(self.language.args, "tie_word_embeddings", True):
            return self.text.embed_tokens.as_linear(normed)
        return self.language.lm_head(normed)

    def forward(self, tokens, cache, visit=None, attend=None):
        """The text model's forward pass with a visit(layer, h) -> h after every block; returns logits.

        attend(layer, weights) receives each full-attention block's softmax weights [heads, L, T].
        """
        h = self.text.embed_tokens(tokens)
        fa_mask = self._attention_mask(h, cache[self.text.fa_idx])
        ssm_mask = self._ssm_mask(h, cache[self.text.ssm_idx])
        for index, (layer, c) in enumerate(zip(self.layers, cache)):
            before, offset = h, (None if layer.is_linear else c.offset)
            h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
            if attend is not None and not layer.is_linear:
                attend(index, attention_weights(layer.self_attn, layer.input_layernorm(before), c, offset))
            if visit is not None:
                h = visit(index, h)
        return self.unembed(h)

    # -- the lens -------------------------------------------------------------------------------------------
    def readout(self, layer, h, jacobian=True):
        """Lens logits for residuals h [..., d] at a source layer."""
        if jacobian:
            h = h.astype(self.mx.float32) @ self.J[layer].T
        return self.unembed(h.astype(self.text.embed_tokens.weight.dtype)).astype(self.mx.float32)

    def top(self, logits, k):
        """The k most likely tokens per row, most likely first, with their probabilities."""
        mx = self.mx
        ids = mx.argpartition(-logits, k - 1, axis=-1)[..., :k]
        values = mx.take_along_axis(logits, ids, axis=-1)
        order = mx.argsort(-values, axis=-1)
        ids = mx.take_along_axis(ids, order, axis=-1)
        logprobs = mx.take_along_axis(logits, ids, axis=-1) - mx.logsumexp(logits, axis=-1, keepdims=True)
        return ids, mx.exp(logprobs)

    def token_rows(self, layer, ids):
        """Unit J-lens vectors of token ids (any shape) at a source layer, as rows: [..., d]."""
        mx = self.mx
        rows = self.text.embed_tokens.weight[ids].astype(mx.float32) * self.text.norm.weight.astype(mx.float32)
        v = rows @ self.J[layer]
        return v / mx.linalg.norm(v, axis=-1, keepdims=True)

    def directions(self, layer, tokens):
        """Unit J-lens vectors of the given tokens at a source layer: columns of a [d, n] matrix."""
        return self.token_rows(layer, self.mx.array(tokens)).T

    # -- interventions --------------------------------------------------------------------------------------
    def calibrate(self, cache_factory, tokens):
        """Each source layer's mean residual norm over a text, which scales an injection."""
        mx = self.mx
        norms = {}

        def visit(index, h):
            if index in self.J:
                norms[index] = float(mx.mean(mx.linalg.norm(h.astype(mx.float32), axis=-1)).item())
            return h
        self.forward(mx.array([tokens]), cache_factory(), visit)
        self.mean_norm = norms

    def intervene(self, index, h, interventions, start, protected):
        """Apply every intervention whose band holds this layer to a chunk of positions start, start + 1, ...

        protected[i] is the set of tokens ablation leaves alone at the chunk's i-th position: the clean run's
        top outputs there, so ablation removes what the model holds in mind rather than what it is about to say.
        """
        mx = self.mx
        if index not in self.J:
            return h
        for change in interventions:
            if not change.first <= index <= change.last:
                continue
            x = h.astype(mx.float32)
            length = x.shape[1]
            chosen_positions = None if change.positions is None else set(change.positions)
            where = mx.ones((length,), dtype=mx.bool_) if chosen_positions is None else mx.array(
                [start + i in chosen_positions for i in range(length)])
            if change.kind == "ablate":
                spare = max((len(p) for p in protected), default=0)
                ids, _ = self.top(self.readout(index, x), change.k + spare)
                chosen = [[t for t in row if t not in protected[p]][:change.k] for p, row in enumerate(ids[0].tolist())]
                y = project_out_each(mx, x[0], self.token_rows(index, mx.array(chosen)))[None]
            elif change.kind == "remove":
                y = project_out(mx, x, self.directions(index, change.tokens))
            elif change.kind == "swap":
                V = self.directions(index, change.tokens[:2])
                c = x @ mx.linalg.pinv(V, stream=mx.cpu).T
                y = x + (c[..., ::-1] - c) @ V.T
            elif change.kind == "inject":
                v = self.directions(index, change.tokens[:1])[:, 0]
                y = x + change.strength * self.mean_norm.get(index, 1.0) * v
            else:
                raise ValueError(f"unknown intervention: {change.kind}")
            h = mx.where(where[None, :, None], y, x).astype(h.dtype)
        return h


def attention_weights(attention, x, cache, offset):
    """The softmax attention weights of a full-attention layer, [heads, L, T], for the L positions of x.

    Called after the layer ran on x (its cache now holds T = offset + L keys), it recomputes what MLX's fused
    attention used: the queries (normed, rotated at their positions), every cached key, the scale and the causal
    mask. Grouped key/value heads are shared by attention // key-value heads query heads.
    """
    import mlx.core as mx
    B, L, _ = x.shape
    queries = attention.q_proj(x).reshape(B, L, attention.num_attention_heads, -1)
    queries = mx.split(queries, 2, axis=-1)[0]
    queries = attention.rope(attention.q_norm(queries).transpose(0, 2, 1, 3), offset=offset)
    keys = cache.keys_and_values()[0]                                   # the T cached keys
    group = attention.num_attention_heads // attention.num_key_value_heads
    keys = mx.repeat(keys, group, axis=1)                                # [B, heads, T, d]
    scores = (queries.astype(mx.float32) * attention.scale) @ keys.astype(mx.float32).swapaxes(-1, -2)
    T = keys.shape[2]
    causal = mx.arange(T)[None, :] <= (offset + mx.arange(L))[:, None]   # [L, T]
    scores = mx.where(causal, scores, -mx.inf)
    return mx.softmax(scores, axis=-1)[0]


def end_ban(mx, vocabulary, eos):
    """Added to logits, makes the end tokens impossible: 0 everywhere but -inf at each of eos."""
    ban = np.zeros(vocabulary, dtype=np.float32)
    ban[sorted(eos)] = -np.inf
    return mx.array(ban)


def project_out(mx, x, V):
    """x minus its projection onto the span of V's columns: x - V V^+ x."""
    pinv = mx.linalg.pinv(V, stream=mx.cpu)
    return x - (x @ pinv.T) @ V.T


def project_out_each(mx, x, rows):
    """Row by row, x[p] minus its projection onto the span of rows[p] ([n, d] and [n, k, d]): one batched solve."""
    gram = rows @ rows.swapaxes(-1, -2)                                   # [n, k, k]
    coefficients = mx.linalg.solve(gram, (rows @ x[..., None]), stream=mx.cpu)   # [n, k, 1]
    return x - (rows.swapaxes(-1, -2) @ coefficients)[..., 0]


TRACE_CHUNK = 64        # prompt positions read out per step: 64 x 248k float32 logits, 64 MB
PROTECTED = 10          # the clean run's top outputs that J-space ablation spares, as in the paper
MAX_ATTENTION_KEYS = 1024   # attention is sent while the context is at most this long
ATTENTION_BYTES = 1_500_000 # attention bytes per event, so a reply stays under the 4 MB request frame
BATCHED_READOUT_BYTES = 256 * 1024 * 1024   # logits read out in one unembedding when they fit


def trace(js, tokenizer, tokens, *, max_new_tokens, sampler, k, interventions, emit, cancelled, eos,
          attention=False, run_to_max=False):
    """Run tokens through the model and generate, emitting the lens grid as it goes.

    Every event covers positions first .. first + count - 1 (prompt is the prompt's length): their token ids, and
    for each position and source layer the top k tokens of the J-lens (jlens, jprob) and of the plain logit lens
    (logit, lprob), then the model's own top k at that position (output, oprob). Ids are flat, position-major,
    then layer, then rank. words maps every id not yet sent in this trace to its text.

    run_to_max bans the end tokens (eos), so the model writes until max_new_tokens.

    With attention, each event also carries the attention patterns of the full-attention layers
    (attention_layers): for each of its positions, layer and head, the softmax weights over the keys (the first
    keys positions), as bytes 0-255 of weight, base64. keys is 0 when the context is longer than
    MAX_ATTENTION_KEYS. The model's linear-attention layers form no such weights.
    """
    mx = js.mx
    layers = js.source_layers
    attention_layers = [i for i, layer in enumerate(js.layers) if not layer.is_linear] if attention else []
    heads = js.layers[attention_layers[0]].self_attn.num_attention_heads if attention_layers else 0
    attended = {}
    sent = set()

    def words(ids):
        new = set(ids) - sent
        sent.update(new)
        return {str(i): tokenizer.decode([i]) for i in sorted(new)}

    def send(first, ids, captured, logits):
        vocabulary = js.text.embed_tokens.weight.shape[0]
        if len(ids) * 2 * len(layers) * vocabulary * 4 <= BATCHED_READOUT_BYTES:
            # A few positions (each generated token): every layer through both lenses as one unembedding.
            transported = [captured[l].astype(mx.float32) @ js.J[l].T for l in layers]
            stream = mx.stack(transported + [captured[l].astype(mx.float32) for l in layers])   # [2 layers, n, d]
            logits_all = js.unembed(stream.astype(js.text.embed_tokens.weight.dtype)).astype(mx.float32)
            top_ids, top_p = js.top(logits_all, k)                                            # [2 layers, n, k]
            jl, ll = top_ids[:len(layers)].transpose(1, 0, 2), top_ids[len(layers):].transpose(1, 0, 2)
            jp, lp = top_p[:len(layers)].transpose(1, 0, 2), top_p[len(layers):].transpose(1, 0, 2)
        else:
            # A prompt chunk: a layer at a time, so its logits stay small.
            jl, jp, ll, lp = [], [], [], []
            for layer in layers:
                a, b = js.top(js.readout(layer, captured[layer]), k)
                c, d = js.top(js.readout(layer, captured[layer], jacobian=False), k)
                jl.append(a); jp.append(b); ll.append(c); lp.append(d)
            jl, jp, ll, lp = (mx.stack(x, axis=1) for x in (jl, jp, ll, lp))                 # [n, layers, k]
        out_ids, out_p = js.top(logits.astype(mx.float32), k)
        jl, jp, ll, lp = (np.array(x) for x in (jl, jp, ll, lp))
        out_ids, out_p = np.array(out_ids), np.array(out_p)
        pattern = dict()
        if attention_layers:
            keys = attended[attention_layers[0]].shape[-1] if attended else 0
            if 0 < keys <= MAX_ATTENTION_KEYS:
                rows = slice(first - offset_of[0], first - offset_of[0] + len(ids))
                weights = mx.stack([attended[l][:, rows, :] for l in attention_layers])     # [layers, heads, n, keys]
                weights = np.array(weights.transpose(2, 0, 1, 3).astype(mx.float32))         # [n, layers, heads, keys]
                data = np.clip(np.rint(weights * 255), 0, 255).astype(np.uint8).tobytes()
            else:
                keys, data = 0, b""
            pattern = dict(attention_layers=attention_layers, heads=heads, keys=keys,
                           attention=base64.b64encode(data).decode("ascii"))
        emit(dict(first=first, count=len(ids), prompt=len(tokens), layers=layers, k=k, tokens=ids, **pattern,
                  jlens=jl.reshape(-1).tolist(), jprob=np.round(jp, 4).reshape(-1).tolist(),
                  logit=ll.reshape(-1).tolist(), lprob=np.round(lp, 4).reshape(-1).tolist(),
                  output=out_ids.reshape(-1).tolist(), oprob=np.round(out_p, 4).reshape(-1).tolist(),
                  words=words(list(ids) + jl.reshape(-1).tolist() + ll.reshape(-1).tolist() + out_ids.reshape(-1).tolist())))

    # With interventions, a clean run goes alongside: ablation spares its top outputs at each position, and an
    # injection is scaled by its layers' mean residual norms.
    clean = js.language.make_cache() if interventions else None
    protected = [set() for _ in tokens]
    if clean is not None:
        norms = {}

        def measure(index, h):
            if index in js.J:
                norms[index] = float(mx.mean(mx.linalg.norm(h.astype(mx.float32), axis=-1)).item())
            return h
        clean_logits = js.forward(mx.array([tokens]), clean, measure)
        protected = [set(row) for row in js.top(clean_logits[0].astype(mx.float32), PROTECTED)[0].tolist()]
        js.mean_norm = norms

    cache = js.language.make_cache()
    captured = {}
    state = dict(start=0, protected=protected)

    def visit(index, h):
        h = js.intervene(index, h, interventions, state["start"], state["protected"])
        if index in js.J:
            captured[index] = h[0]
        return h

    # The positions the attention just recorded start at offset_of[0].
    offset_of = [0]

    def attend(index, weights):
        attended[index] = weights

    logits = js.forward(mx.array([tokens]), cache, visit, attend if attention_layers else None)
    chunk = TRACE_CHUNK
    if attention_layers and len(tokens) <= MAX_ATTENTION_KEYS:
        chunk = max(1, min(TRACE_CHUNK, ATTENTION_BYTES // (len(attention_layers) * heads * len(tokens))))
    for first in range(0, len(tokens), chunk):
        if cancelled.is_set():
            return dict(tokens=0, cancelled=True, finish_reason="cancelled", prompt_tokens=len(tokens))
        end = min(len(tokens), first + chunk)
        send(first, tokens[first:end], {l: captured[l][first:end] for l in layers}, logits[0, first:end])

    count, finish = 0, "length"
    position = len(tokens)
    banned = end_ban(mx, js.text.embed_tokens.weight.shape[0], eos) if run_to_max else None
    for _ in range(max_new_tokens):
        if cancelled.is_set():
            finish = "cancelled"
            break
        last = logits[:, -1, :].astype(mx.float32)
        if banned is not None:
            last = last + banned
        token = int(sampler(last - mx.logsumexp(last, axis=-1, keepdims=True)).item())
        if token in eos:
            finish = "stop"
            break
        if clean is not None:
            clean_logits = js.forward(mx.array([[token]]), clean)
            state["protected"] = [set(js.top(clean_logits[0].astype(mx.float32), PROTECTED)[0][0].tolist())]
        state["start"] = position
        offset_of[0] = position
        logits = js.forward(mx.array([[token]]), cache, visit, attend if attention_layers else None)
        send(position, [token], {l: captured[l] for l in layers}, logits[0])
        position += 1
        count += 1
    return dict(tokens=count, cancelled=finish == "cancelled", finish_reason=finish, prompt_tokens=len(tokens))
