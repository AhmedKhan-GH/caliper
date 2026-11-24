"""Forward settings for trusted checkpoint compatibility, without trace imports."""
from __future__ import annotations

import json

import torch


_MAX_CHECKPOINT_BUFFER_BYTES = 1024 * 1024


_MODULE_FIELDS = (
    "training", "eps", "variance_epsilon", "layer_norm_epsilon", "frozen_scale", "frozen_pattern",
    "num_heads", "n_heads", "n_head", "head_dim", "in_features", "out_features", "k",
    "num_key_value_groups", "scaling", "attention_dropout", "is_causal", "layer_idx",
    "block_type", "layer_type", "num_v_heads", "num_k_heads", "head_k_dim", "head_v_dim",
    "key_dim", "value_dim", "hidden_size", "intermediate_size", "conv_dim", "conv_kernel_size", "activation",
    "rope_type", "attention_scaling", "mrope_section", "max_seq_len_cached",
    "original_max_seq_len", "long_inv_freq", "max_norm", "norm_type",
    "padding_idx", "scale_grad_by_freq", "sparse", "inplace", "approximate",
    "kernel_size", "stride", "padding", "dilation", "groups", "padding_mode",
)
_CONFIG_FIELDS = (
    "model_type", "num_hidden_layers", "hidden_size", "intermediate_size", "vocab_size",
    "layer_types", "is_causal", "rope_parameters", "max_position_embeddings", "head_dim",
    "num_attention_heads", "num_key_value_heads", "hidden_act", "rms_norm_eps",
    "attention_dropout", "attention_bias", "linear_num_value_heads", "linear_num_key_heads",
    "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
)


def _type_name(value):
    return type(value).__module__ + "." + type(value).__qualname__


def forward_settings(model, *, tensor_hash):
    """Record effective settings with stable callable names and tensor contents.

    Eager and SDPA attention are supported backend choices. They share checkpoint
    identity. Cache outputs and placement are absent from this signature.
    """
    settings = {}

    def record(name, value):
        if isinstance(value, torch.Tensor):
            value = {"tensor": tensor_hash({name: value})}
        settings[name] = json.dumps(value, sort_keys=True, allow_nan=False)

    for name, module in model.named_modules():
        prefix = name + "." if name else ""
        record(prefix + "type", _type_name(module))
        for field in _MODULE_FIELDS:
            if hasattr(module, field):
                record(prefix + field, getattr(module, field))
        if hasattr(module, "act_fn"):
            function = module.act_fn
            record(prefix + "act_fn", _type_name(function) if isinstance(function, torch.nn.Module)
                   else function.__module__ + "." + function.__qualname__)
        config = getattr(module, "config", None)
        backend = getattr(config, "_attn_implementation", None)
        if backend not in (None, "eager", "sdpa"):
            raise ValueError("checkpoint identity supports eager or sdpa attention")
        for field in _CONFIG_FIELDS:
            if hasattr(config, field):
                record(prefix + "config." + field, getattr(config, field))
    return settings


def nonpersistent_buffers(model):
    """Include every buffer registration, even aliases and absent buffers."""
    return {f"{name}.{key}" if name else key: value
            for name, module in model.named_modules(remove_duplicate=False)
            for key, value in module._buffers.items()
            if key in module._non_persistent_buffers_set}


def _checkpoint_buffers(reference, model, *, admit):
    """Reconstruct supported buffers on CPU after a bounded meta inspection."""
    expected = nonpersistent_buffers(reference)
    actual = nonpersistent_buffers(model)
    if expected.keys() != actual.keys():
        raise ValueError("model nonpersistent buffers differ from the local checkpoint")
    if not expected:
        return {}
    total = 0
    for name, value in expected.items():
        live = actual[name]
        if (not isinstance(value, torch.Tensor) or not isinstance(live, torch.Tensor)
                or live.shape != value.shape or live.dtype not in
                (torch.float16, torch.bfloat16, torch.float32, torch.float64)):
            raise ValueError("unsupported checkpoint nonpersistent buffers")
        total += value.numel() * max(value.element_size(), live.element_size())
    if total > _MAX_CHECKPOINT_BUFFER_BYTES:
        raise ValueError("checkpoint nonpersistent buffers exceed the reconstruction limit")
    # Cover the result, constructor intermediates, and dtype conversion together.
    admit(4 * _MAX_CHECKPOINT_BUFFER_BYTES)

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    buffers = {}
    for name, module in reference.named_modules(remove_duplicate=False):
        keys = module._non_persistent_buffers_set
        if not keys:
            continue
        # This constructor uses only two frequency vectors. Unknown constructors
        # must not allocate model parameters or establish an unchecked baseline.
        if (type(module) is not Qwen3_5TextRotaryEmbedding or module.rope_type != "default"
                or keys != {"inv_freq", "original_inv_freq"}):
            raise ValueError("unsupported checkpoint nonpersistent buffers")
        with torch.device("cpu"):
            rotary = Qwen3_5TextRotaryEmbedding(module.config)
        for key, value in rotary.named_buffers():
            full_name = f"{name}.{key}" if name else key
            buffers[full_name] = value.to(dtype=actual[full_name].dtype)
    return buffers


def checkpoint_settings(model, data, *, tensor_hash, admit_buffers):
    """Derive settings and buffer identity without checkpoint parameter storage."""
    # The text-only class uses the nested text config of a multimodal snapshot.
    if getattr(model.config, "model_type", None) == "qwen3_5_text" and "text_config" in data:
        data = data["text_config"]
    factory = getattr(type(model.config), "from_dict", None)
    if factory is None:
        raise ValueError("checkpoint verification requires a reconstructible model config")
    config = factory(data)
    # Reference construction must not select a backend from runtime configuration.
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        reference = type(model)(config).eval()
    settings = forward_settings(reference, tensor_hash=tensor_hash)
    return settings, tensor_hash(_checkpoint_buffers(reference, model, admit=admit_buffers))
