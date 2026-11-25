"""Explicit Gemma dense arithmetic for the pinned circuit-tracer profile."""
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
from types import CodeType

import torch
from transformers import __version__ as transformers_version
from transformers.models.gemma2 import modeling_gemma2 as hf

PROFILE = "circuit_tracer_82996f14"
_PROFILE_FIELD = "_caliper_reference_profile"
_DTYPE_FIELD = "_caliper_reference_dtype"
_CONFIG_FIELD = "_caliper_reference_config"
_CONFIG_KEYS = (
    "model_type", "hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size",
    "head_dim", "num_attention_heads", "num_key_value_heads", "rms_norm_eps",
    "hidden_activation", "attention_bias", "attention_dropout", "tie_word_embeddings",
    "layer_types", "sliding_window", "query_pre_attn_scalar", "attn_logit_softcapping",
    "final_logit_softcapping", "rope_parameters", "use_bidirectional_attention",
)


def _config_signature(config):
    return json.dumps({key: getattr(config, key, None) for key in _CONFIG_KEYS},
                      sort_keys=True, allow_nan=False)


def validate_profile(profile):
    if profile is not None and (type(profile) is not str or profile != PROFILE):
        raise ValueError("unsupported Gemma reference profile")


def normalized_input(module, x):
    """Return original Scope coordinates without an inverse of rounded gain."""
    work = x.float()
    return (work / (work.square().mean(-1, keepdim=True) + module.eps).sqrt()).to(module.weight.dtype)


class ReferenceInput(torch.nn.Module):
    """A hook point at the exact model-dtype input before learned gain."""

    def forward(self, value):
        return value


class ReferenceRMSNorm(hf.Gemma2RMSNorm):
    def forward(self, x):
        value = self.reference_input(normalized_input(self, x))
        return value * (self.weight.float() + 1).to(self.weight.dtype)


class ReferenceRotaryEmbedding(hf.Gemma2RotaryEmbedding):
    def forward(self, x, position_ids):
        # The reference divides positions by frequencies before trigonometry.
        # Retain HF buffers for checkpoint verification, without rounded inverses.
        dim = self.config.head_dim
        # The reference constructs its default rotary table on CPU, then moves it.
        exponent = torch.arange(dim // 2, device="cpu", dtype=torch.float32) / (dim / 2)
        frequency = self.config.rope_parameters["rope_theta"] ** exponent
        angles = position_ids.to(device="cpu", dtype=torch.float32)[..., None] / frequency
        angles = torch.cat((angles, angles), dim=-1)
        return (angles.cos().to(device=x.device, dtype=x.dtype),
                angles.sin().to(device=x.device, dtype=x.dtype))


class ReferenceAttention(hf.Gemma2Attention):
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, cache=None, **kwargs):
        if cache is not None or kwargs.get("past_key_values") is not None:
            raise ValueError("Gemma reference arithmetic requires an uncached dense forward")
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        query = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        cos, sin = (part.unsqueeze(1) for part in position_embeddings)
        query = query * cos + hf.rotate_half(query) * sin
        key = key * cos + hf.rotate_half(key) * sin
        key = key.repeat_interleave(self.num_key_value_groups, dim=1)
        value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        scores = (query @ key.transpose(-1, -2)) / math.sqrt(self.config.query_pre_attn_scalar)
        cap = self.attn_logit_softcapping
        if cap is not None:
            scores = cap * (scores / cap).tanh()
        if attention_mask is None:
            raise ValueError("Gemma reference arithmetic requires an explicit causal mask")
        if attention_mask.dtype == torch.bool:
            allowed = attention_mask
        else:
            allowed = attention_mask == 0
        scores = scores.masked_fill(~allowed, -torch.inf)
        pattern = scores.softmax(dim=-1)
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        mixed = (pattern.to(value.dtype) @ value).transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
        return self.o_proj(mixed), pattern


class ReferenceGemmaForCausalLM(hf.Gemma2ForCausalLM):
    def forward(self, *args, **kwargs):
        profile_metadata(self)
        devices = {"cpu", *(parameter.device.type for parameter in self.parameters())}
        if (any(torch.is_autocast_enabled(device) for device in devices)
                or torch.get_float32_matmul_precision() != "highest"):
            raise ValueError("Gemma reference arithmetic requires disabled autocast and highest matmul precision")
        if kwargs.get("use_cache") or kwargs.get("cache") is not None or kwargs.get("past_key_values") is not None:
            raise ValueError("Gemma reference arithmetic requires an uncached dense forward")
        kwargs["use_cache"] = False
        return super().forward(*args, **kwargs)


_ADAPTERS = {
    hf.Gemma2ForCausalLM: ReferenceGemmaForCausalLM,
    hf.Gemma2RMSNorm: ReferenceRMSNorm,
    hf.Gemma2Attention: ReferenceAttention,
    hf.Gemma2RotaryEmbedding: ReferenceRotaryEmbedding,
}
# Retain function objects to reject instance and class overrides at admission.
_FORWARDS = {cls: cls.forward for cls in (*_ADAPTERS.values(), hf.Gemma2ForCausalLM, hf.Gemma2Model,
    hf.Gemma2DecoderLayer, hf.Gemma2MLP, hf.Gemma2TextScaledWordEmbedding, torch.nn.Linear,
    ReferenceInput, type(hf.ACT2FN["gelu_pytorch_tanh"]))}
_NORMALIZED_INPUT = normalized_input
_ROTATE_HALF = hf.rotate_half


def _operations():
    result = {}
    for function in (*_FORWARDS.values(), normalized_input, hf.rotate_half):
        name = function.__module__ + "." + function.__qualname__
        while True:
            result[name] = function
            if not hasattr(function, "__wrapped__"):
                break
            function = function.__wrapped__
            name += ".__wrapped__"
    return result


_OPERATIONS = _operations()
_CODES = {function: function.__code__ for function in _OPERATIONS.values()}


def _code_identity(code):
    """Bind executable expressions without source paths or line numbers."""
    def constant(value):
        if isinstance(value, CodeType):
            return _code_identity(value)
        if isinstance(value, (tuple, frozenset)):
            return [constant(item) for item in sorted(value, key=repr)] if isinstance(value, frozenset) else [
                constant(item) for item in value]
        return {"type": type(value).__name__, "value": repr(value)}
    content = {"bytecode": code.co_code.hex(), "constants": [constant(value) for value in code.co_consts],
               "names": code.co_names, "variables": code.co_varnames, "freevars": code.co_freevars,
               "cellvars": code.co_cellvars, "arguments": [code.co_argcount, code.co_posonlyargcount,
               code.co_kwonlyargcount], "flags": code.co_flags, "exceptions": code.co_exceptiontable.hex()}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


_OPERATION_HASHES = {name: _code_identity(function.__code__) for name, function in _OPERATIONS.items()}


def _check_model(model, *, adapted):
    expected = ReferenceGemmaForCausalLM if adapted else hf.Gemma2ForCausalLM
    if type(model) is not expected:
        raise ValueError("Gemma reference profile requires the supported Gemma2 model")
    config = model.config
    if adapted and getattr(model, _CONFIG_FIELD, None) != _config_signature(config):
        raise ValueError("Gemma reference adapter configuration changed")
    if (config._attn_implementation != "eager" or config.attention_bias
            or config.hidden_activation != "gelu_pytorch_tanh"
            or config.rope_parameters != {"rope_type": "default", "rope_theta": config.rope_parameters.get("rope_theta")}
            or getattr(config, "use_bidirectional_attention", False)
            or config.attention_dropout != 0 or not config.tie_word_embeddings
            or config.head_dim % 2 or config.num_attention_heads % config.num_key_value_heads):
        raise ValueError("unsupported Gemma reference configuration")
    if (not math.isfinite(config.query_pre_attn_scalar) or config.query_pre_attn_scalar <= 0
            or any(cap is not None and (not math.isfinite(cap) or cap <= 0)
                   for cap in (config.attn_logit_softcapping, config.final_logit_softcapping))):
        raise ValueError("unsupported Gemma reference scale")
    if any(module.training for module in model.modules()):
        raise ValueError("Gemma reference profile requires evaluation mode")
    dtypes = {value.dtype for value in model.parameters()}
    if len(dtypes) != 1 or next(iter(dtypes)) not in (torch.float32, torch.bfloat16):
        raise ValueError("Gemma reference profile requires uniform float32 or bfloat16 parameters")
    dtype = next(iter(dtypes))
    if adapted and getattr(model, _DTYPE_FIELD, None) != str(dtype):
        raise ValueError("Gemma reference adapter dtype changed")
    if normalized_input is not _NORMALIZED_INPUT or hf.rotate_half is not _ROTATE_HALF:
        raise ValueError("Gemma reference adapter operation changed")
    if any(cls.forward is not forward for cls, forward in _FORWARDS.items()):
        raise ValueError("Gemma reference adapter class operation changed")
    if any(function.__code__ is not code for function, code in _CODES.items()):
        raise ValueError("Gemma reference adapter implementation changed")
    if (type(model.model) is not hf.Gemma2Model or model.model.config is not config
            or type(model.model.embed_tokens) is not hf.Gemma2TextScaledWordEmbedding
            or type(model.lm_head) is not torch.nn.Linear
            or model.lm_head.bias is not None
            or model.lm_head.weight is not model.model.embed_tokens.weight):
        raise ValueError("Gemma reference embedding or output state changed")
    for module in model.modules():
        if any(key in vars(module) for key in ("forward", "_call_impl", "_compiled_call_impl")):
            raise ValueError("Gemma reference adapter callable override")
        cls = type(module)
        if adapted and cls in _ADAPTERS:
            raise ValueError("Gemma reference adapter is incomplete")
    norms = [model.model.norm]
    if len(model.model.layers) != config.num_hidden_layers:
        raise ValueError("Gemma reference layer count changed")
    for i, block in enumerate(model.model.layers):
        if type(block) is not hf.Gemma2DecoderLayer or type(block.mlp) is not hf.Gemma2MLP:
            raise ValueError("Gemma reference block changed")
        if (block.config is not config or block.mlp.config is not config
                or type(block.mlp.act_fn) is not type(hf.ACT2FN["gelu_pytorch_tanh"])):
            raise ValueError("Gemma reference MLP state changed")
        attention = block.self_attn
        if type(attention) is not (ReferenceAttention if adapted else hf.Gemma2Attention):
            raise ValueError("Gemma reference attention changed")
        if (attention.config is not config or attention.layer_idx != i
                or attention.layer_type != config.layer_types[i]
                or attention.head_dim != config.head_dim
                or attention.num_key_value_groups != config.num_attention_heads // config.num_key_value_heads
                or attention.attn_logit_softcapping != config.attn_logit_softcapping):
            raise ValueError("Gemma reference attention state changed")
        projections = [attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj,
                       block.mlp.gate_proj, block.mlp.up_proj, block.mlp.down_proj]
        if any(type(module) is not torch.nn.Linear or module.bias is not None for module in projections):
            raise ValueError("Gemma reference projection state changed")
        norms.extend(getattr(block, name) for name in ("input_layernorm", "post_attention_layernorm",
                     "pre_feedforward_layernorm", "post_feedforward_layernorm"))
    if any(type(norm) is not (ReferenceRMSNorm if adapted else hf.Gemma2RMSNorm)
           or norm.eps != config.rms_norm_eps for norm in norms):
        raise ValueError("Gemma reference normalization state changed")
    if adapted and any(type(getattr(norm, "reference_input", None)) is not ReferenceInput for norm in norms):
        raise ValueError("Gemma reference input hook changed")
    rotary = model.model.rotary_emb
    if (type(rotary) is not (ReferenceRotaryEmbedding if adapted else hf.Gemma2RotaryEmbedding)
            or rotary.config is not config or rotary.rope_type != "default"):
        raise ValueError("Gemma reference rotary adapter changed")
    return dtype


def install_reference_profile(model, profile):
    """Adapt operations without a parameter copy or a verified-load claim."""
    validate_profile(profile)
    if profile is None:
        return model
    if hasattr(model, _PROFILE_FIELD):
        profile_metadata(model)
        return model
    dtype = _check_model(model, adapted=False)
    for module in model.modules():
        replacement = _ADAPTERS.get(type(module))
        if replacement is not None:
            module.__class__ = replacement
            if replacement is ReferenceRMSNorm:
                module.reference_input = ReferenceInput().eval()
    setattr(model, _PROFILE_FIELD, profile)
    setattr(model, _DTYPE_FIELD, str(dtype))
    setattr(model, _CONFIG_FIELD, _config_signature(model.config))
    profile_metadata(model)
    return model


def profile_metadata(model):
    """Bind the installed operations, gain preparation, and arithmetic dtype."""
    profile = getattr(model, _PROFILE_FIELD, None)
    if profile is None:
        if any(type(module) in _ADAPTERS.values() for module in model.modules()):
            raise ValueError("Gemma reference adapter profile is missing")
        return None
    validate_profile(profile)
    dtype = _check_model(model, adapted=True)
    return {
        "profile": profile, "dtype": str(dtype).removeprefix("torch."), "version": 1,
        "reference_revision": "82996f144b3371ae858ea5713590d004c218f0c2",
        "reference_transformer_lens": "3.9.0",
        "torch_version": str(torch.__version__), "transformers_version": transformers_version,
        "normalization": "float32_divide_sqrt_then_cast_then_gain",
        "gain": "source_weight_float32_plus_one_then_cast",
        "rotary": "cpu_float32_position_divide_frequency_then_trig_then_cast_then_transfer",
        "attention": "matmul_divide_sqrt_scalar_softcap_mask_negative_inf_softmax_score_dtype",
        "encoder_coordinate": "gemma_scope_pre_gain",
        "capture": "pre_feedforward_layernorm.reference_input",
        "runtime": "autocast_disabled_float32_matmul_highest",
        "operation_sha256": dict(_OPERATION_HASHES),
        "operations": {name: cls.__module__ + "." + cls.__qualname__ + ".forward"
                       for name, cls in sorted((base.__name__, cls) for base, cls in _ADAPTERS.items())},
    }


@contextmanager
def capture_reference_inputs(model):
    """Capture live original encoder coordinates for each complete dense forward.

    Callers serialize forwards and admit retained tensors before entry. The map
    keeps the last complete forward. Failure clears it. Exit removes all hooks.
    """
    if profile_metadata(model) is None:
        raise ValueError("original encoder capture requires a Gemma reference profile")
    values, frames = {}, []
    with ExitStack() as stack:
        def start(*_):
            profile_metadata(model)
            frames.append(dict(values) if frames else None)
            values.clear()

        def finish(module, args, output):
            if not frames:
                values.clear()
                return
            saved = frames.pop()
            if saved is not None:
                values.clear()
                values.update(saved)
            elif output is None:
                values.clear()

        stack.callback(model.register_forward_pre_hook(start).remove)
        stack.callback(model.register_forward_hook(finish, always_call=True).remove)
        for i, block in enumerate(model.model.layers):
            def read(module, args, output, i=i):
                values[f"model.layers.{i}.mlp_in"] = output.detach()
            stack.callback(block.pre_feedforward_layernorm.reference_input.register_forward_hook(read).remove)
        yield values
