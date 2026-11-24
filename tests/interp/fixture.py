"""Deterministic decoder with exact TopK transcoders for CPU tests.

Every block uses pre-normalization and causal attention. The model has no
dropout. The default dtype is float64 for completeness and derivative checks.
"""

from contextlib import contextmanager
import hashlib
import json
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps
        self.frozen_scale = None

    def forward(self, x):
        scale = self.frozen_scale
        if scale is None:
            scale = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        self.last_scale = scale.detach()
        return x * scale * self.weight


class Attention(nn.Module):
    def __init__(self, width=64, heads=4):
        super().__init__()
        self.num_heads = heads
        self.head_dim = width // heads
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.pattern = nn.Identity()
        self.frozen_pattern = None

    def forward(self, x):
        batch, length, width = x.shape
        def split(proj):
            return proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.q_proj), split(self.k_proj), split(self.v_proj)
        pattern = self.frozen_pattern
        if pattern is None:
            scores = q @ k.transpose(-1, -2) / self.head_dim ** 0.5
            mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
            pattern = scores.masked_fill(mask, -torch.inf).softmax(-1)
        pattern = self.pattern(pattern)
        self.last_pattern = pattern.detach()
        y = (pattern @ v).transpose(1, 2).reshape(batch, length, width)
        return self.o_proj(y)


class TranscoderMLP(nn.Module):
    def __init__(self, width=64, latents=256, k=8):
        super().__init__()
        self.k = k
        # Rows of W_enc read a feature. Rows of W_dec write a feature.
        self.W_enc = nn.Parameter(torch.randn(latents, width) / width ** 0.5)
        self.b_enc = nn.Parameter(torch.zeros(latents))
        self.W_dec = nn.Parameter(torch.randn(latents, width) / (latents * k) ** 0.5)
        self.b_dec = nn.Parameter(torch.zeros(width))
        self.features = nn.Identity()

    def encode(self, x):
        z = F.linear(x, self.W_enc, self.b_enc).relu()
        values, indices = z.topk(self.k, dim=-1)
        return torch.zeros_like(z).scatter(-1, indices, values)

    def decode(self, a):
        return a @ self.W_dec + self.b_dec

    def forward(self, x):
        return self.decode(self.features(self.encode(x)))


class Block(nn.Module):
    def __init__(self, width=64, heads=4, latents=256, k=8):
        super().__init__()
        self.input_layernorm = RMSNorm(width)
        self.self_attn = Attention(width, heads)
        self.post_attention_layernorm = RMSNorm(width)
        self.mlp = TranscoderMLP(width, latents, k)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyLM(nn.Module):
    def __init__(self, *, seed=7, layers=4, width=64, heads=4, vocab=512,
                 latents=256, k=8, dtype=torch.float64):
        super().__init__()
        self.config = SimpleNamespace(model_type="caliper_tiny", _name_or_path="tiny-lm",
                                      num_hidden_layers=layers, hidden_size=width,
                                      num_attention_heads=heads, vocab_size=vocab,
                                      rms_norm_eps=1e-6, latents=latents, k=k)
        # Model construction must not change the caller's random state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embed_tokens = nn.Embedding(vocab, width)
            self.layers = nn.ModuleList(Block(width, heads, latents, k) for _ in range(layers))
            self.norm = RMSNorm(width)
            self.lm_head = nn.Linear(width, vocab, bias=False)
        self.to(dtype=dtype)
        self.eval()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, *, inputs_embeds=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("supply input_ids or inputs_embeds")
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        for block in self.layers:
            x = block(x)
        return SimpleNamespace(logits=self.lm_head(self.norm(x)))

    @contextmanager
    def frozen_attention(self, *, norms=False):
        """Freeze constants from the last forward pass, then restore prior values."""
        saved = []
        for module in self.modules():
            key = "frozen_pattern" if isinstance(module, Attention) else (
                "frozen_scale" if norms and isinstance(module, RMSNorm) else None)
            if key is not None:
                saved.append((module, key, getattr(module, key)))
                source = "last_pattern" if key == "frozen_pattern" else "last_scale"
                setattr(module, key, getattr(module, source).detach())
        try:
            yield self
        finally:
            for module, key, value in saved:
                setattr(module, key, value)


class TinyTokenizer:
    """A byte tokenizer with a fixed 512-token vocabulary and a BOS token."""
    vocab_size = 512
    bos_token_id = 256
    eos_token_id = 257
    pad_token_id = 258

    def encode(self, text, add_special_tokens=True):
        ids = list(text.encode("utf-8"))
        return ([self.bos_token_id] if add_special_tokens else []) + ids

    def decode(self, ids, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        return bytes(int(x) for x in ids if 0 <= int(x) < 256).decode("utf-8", errors="replace")

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.decode([ids]) if ids < 256 else f"<token{ids}>"
        return [self.convert_ids_to_tokens(int(x)) for x in ids]

    def __call__(self, text, return_tensors="pt", **kwargs):
        ids = self.encode(text, add_special_tokens=kwargs.get("add_special_tokens", True))
        return {"input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}


def make_fixture(**kwargs):
    return TinyLM(**kwargs), TinyTokenizer()


def forward_logits(model=None):
    """Return the fixed prompt logits for numerical reference checks."""
    model = model if model is not None else TinyLM()
    with torch.no_grad():
        return model(torch.tensor([[256, 65, 66, 67, 32, 68]])).logits


def forward_hash(model=None):
    """Hash rounded logits for local diagnostics, without platform tolerance."""
    rounded = (forward_logits(model) * 1e8).round().to(torch.int64).tolist()
    data = json.dumps(rounded, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(data).hexdigest()
