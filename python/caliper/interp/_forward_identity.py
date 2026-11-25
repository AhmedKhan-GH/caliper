"""Effective settings for checkpoint compatibility and clean patch sources."""
from __future__ import annotations

import json
from functools import partial
from itertools import chain

import torch

from . import _qwen_family


_MAX_CHECKPOINT_BUFFER_BYTES = 1024 * 1024
_ARITHMETIC_CONTEXT = "__arithmetic_context__"


_MODULE_FIELDS = (
    "training", "eps", "variance_epsilon", "layer_norm_epsilon", "frozen_scale", "frozen_pattern",
    "num_heads", "n_heads", "n_head", "head_dim", "in_features", "out_features", "k",
    "num_key_value_groups", "scaling", "attention_dropout", "is_causal", "layer_idx",
    *_qwen_family.ROUTING_FIELDS, "layer_type", *_qwen_family.DELTA_HEAD_FIELDS,
    *_qwen_family.DELTA_DIM_FIELDS, "hidden_size", "intermediate_size", *_qwen_family.DELTA_CONV_FIELDS, "activation",
    "rope_type", "attention_scaling", *_qwen_family.ROTARY_FIELDS, "max_seq_len_cached",
    "original_max_seq_len", "long_inv_freq", "max_norm", "norm_type",
    "padding_idx", "scale_grad_by_freq", "sparse", "inplace", "approximate",
    "kernel_size", "stride", "padding", "dilation", "groups", "padding_mode",
    "sliding_window", "attn_logit_softcapping", "scalar_embed_scale",
)
_CONFIG_FIELDS = (
    "model_type", "num_hidden_layers", "hidden_size", "intermediate_size", "vocab_size",
    "layer_types", "is_causal", "rope_parameters", "max_position_embeddings", "head_dim",
    "num_attention_heads", "num_key_value_heads", "hidden_act", "rms_norm_eps",
    "attention_dropout", "attention_bias", *_qwen_family.CONFIG_FIELDS,
    "sliding_window", "query_pre_attn_scalar", "attn_logit_softcapping", "final_logit_softcapping",
    "tie_word_embeddings", "use_bidirectional_attention",
)
_SOURCE_MODULE_FIELDS = ("batch_first", "norm_first", "activation_relu_or_gelu", "dropout", "p")
_NATIVE_ACTIVATIONS = (torch.nn.GELU, torch.nn.Identity, torch.nn.Hardswish, torch.nn.LeakyReLU,
                       torch.nn.ReLU, torch.nn.ReLU6, torch.nn.Sigmoid, torch.nn.SiLU,
                       torch.nn.Tanh, torch.nn.PReLU)


def _type_name(value):
    return type(value).__module__ + "." + type(value).__qualname__


def _activation_callable(function, owner=None):
    """Bind supported callables and options without process-specific addresses."""
    functions = (
        (torch.nn.functional.gelu, {"approximate": ("none", "tanh")}),
        (torch.nn.functional.relu, {"inplace": (False, True)}),
        (torch.nn.functional.silu, {"inplace": (False, True)}),
        (torch.nn.functional.mish, {"inplace": (False, True)}),
        (torch.tanh, {}), (torch.sigmoid, {}),
    )
    base = function.func if type(function) is partial else function
    for supported, options in functions:
        if base is not supported:
            continue
        name = supported.__module__ + "." + supported.__qualname__
        if type(function) is not partial:
            return name
        if (function.args or any(key not in options or
                not any(type(value) is type(choice) and value == choice for choice in options[key])
                for key, value in function.keywords.items())):
            raise ValueError("unsupported effective activation callable options")
        return {"function": name, "keywords": dict(function.keywords)}
    if owner is None or type(owner).__module__ != "transformers.activations":
        raise ValueError("unsupported effective activation callable")
    from transformers.activations import GELUActivation, GELUTanh, MishActivation

    for cls, name in ((GELUTanh, "_gelu_tanh_python"), (GELUActivation, "_gelu_python"),
                      (MishActivation, "_mish_python")):
        if (type(owner) is cls and getattr(function, "__self__", None) is owner
                and getattr(function, "__func__", None) is getattr(cls, name)):
            return cls.__module__ + "." + cls.__qualname__ + "." + name
    raise ValueError("unsupported effective activation callable")


def _activation_settings(function, prefix, record):
    if not isinstance(function, torch.nn.Module):
        record(prefix, _activation_callable(function))
        return
    # Known activation modules use public scalar options or a supported `act`.
    # The stateful kernel module requires a separate implementation contract.
    supported = set(_NATIVE_ACTIVATIONS)
    if type(function).__module__ == "transformers.activations":
        from transformers import activations
        supported.update(value[0] if isinstance(value, tuple) else value
                         for key, value in activations.ACT2CLS.items() if key != "xielu")
    if type(function) not in supported:
        raise ValueError("unsupported effective activation module")
    if any(key in vars(function) for key in ("forward", "_call_impl", "_compiled_call_impl")):
        raise ValueError("unsupported effective activation callable override")
    record(prefix, _type_name(function))
    if hasattr(function, "act"):
        record(prefix + ".act", _activation_callable(function.act, owner=function))
    for key, value in vars(function).items():
        if key.startswith("_") or key in ("training", "act"):
            continue
        if type(value) in (type(None), bool, int, float, str):
            record(prefix + "." + key, value)
        else:
            raise ValueError("unsupported effective activation option")


def forward_settings(model, *, tensor_hash):
    """Record effective settings with stable callable names and tensor contents.

    Qwen permits eager and SDPA attention. Gemma requires eager attention for
    its softcap. Cache outputs and placement are absent from this signature.
    Runtime arithmetic belongs to source, trace, and evaluation evidence. It
    does not change checkpoint verification against a reference on meta.
    """
    return _settings(model, tensor_hash=tensor_hash, source=False)


def source_settings(model, *, tensor_hash):
    """Bind generic source settings without checkpoint backend admission.

    Known decoder families retain their backend restrictions. Callable activation
    fields accept only the bounded functions and modules above.
    """
    settings = _settings(model, tensor_hash=tensor_hash, source=True)
    settings.update(arithmetic_settings(model))
    return settings


def arithmetic_settings(model):
    """Bind arithmetic without tensor copies or device allocations.

    The canonical default is disabled autocast and highest float32 matmul
    precision. It adds no fields, which preserves exact legacy evaluation and
    index hashes. A disabled autocast dtype has no effect and is absent too.

    Check CPU and every device type in the model's parameters and buffers.
    CPU can also execute host operations for a model on another device.
    Checkpoint settings remain independent of this runtime context.
    """
    devices = {"cpu"}
    devices.update(value.device.type for value in chain(model.parameters(), model.buffers()))
    autocast = {device: str(torch.get_autocast_dtype(device))
                for device in sorted(devices) if torch.is_autocast_enabled(device)}
    context = {}
    if autocast:
        context["autocast"] = autocast
    precision = torch.get_float32_matmul_precision()
    if precision != "highest":
        context["float32_matmul_precision"] = precision
    return ({_ARITHMETIC_CONTEXT: json.dumps(context, sort_keys=True, allow_nan=False)}
            if context else {})


def _settings(model, *, tensor_hash, source):
    settings = {}

    def record(name, value):
        if isinstance(value, torch.Tensor):
            value = {"tensor": tensor_hash({name: value})}
        settings[name] = json.dumps(value, sort_keys=True, allow_nan=False)

    profile = _reference_profile(model)
    if profile is not None:
        record("__gemma_reference__", profile)

    for name, module in model.named_modules():
        prefix = name + "." if name else ""
        record(prefix + "type", _type_name(module))
        for field in _MODULE_FIELDS:
            if hasattr(module, field):
                value = getattr(module, field)
                if source and field == "activation" and callable(value):
                    _activation_settings(value, prefix + field, record)
                else:
                    record(prefix + field, value)
        if source:
            if isinstance(module, _NATIVE_ACTIVATIONS):
                # Standalone modules have no activation or act_fn owner field.
                _activation_settings(module, prefix + "activation_module", record)
            for field in _SOURCE_MODULE_FIELDS:
                value = getattr(module, field, None)
                # Dropout modules appear separately in named_modules.
                if hasattr(module, field) and not isinstance(value, torch.nn.Module):
                    record(prefix + field, value)
        if hasattr(module, "act_fn"):
            _activation_settings(module.act_fn, prefix + "act_fn", record)
        config = getattr(module, "config", None)
        backend = getattr(config, "_attn_implementation", None)
        family = getattr(config, "model_type", None)
        if family == "gemma2" and backend != "eager":
            raise ValueError("Gemma checkpoint identity requires eager attention for its softcap")
        if (not source or family in ("gemma2", _qwen_family.FAMILY_NAME)) and backend not in (None, "eager", "sdpa"):
            raise ValueError("checkpoint identity supports eager or sdpa attention")
        if source and hasattr(config, "_attn_implementation"):
            record(prefix + "config._attn_implementation", backend)
        for field in _CONFIG_FIELDS:
            if hasattr(config, field):
                record(prefix + "config." + field, getattr(config, field))
    return settings


def _reference_profile(model):
    # Preserve optional Transformers imports and all default identity fields.
    if (hasattr(model, "_caliper_reference_profile")
            or any(type(module).__module__ == "caliper.interp._gemma_reference"
                   for module in model.modules())):
        from ._gemma_reference import profile_metadata
        return profile_metadata(model)
    return None


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

    qwen_rotary = _qwen_family.rotary_class()
    from transformers.models.gemma2.modeling_gemma2 import Gemma2RotaryEmbedding, Gemma2TextScaledWordEmbedding

    buffers = {}
    for name, module in reference.named_modules(remove_duplicate=False):
        keys = module._non_persistent_buffers_set
        if not keys:
            continue
        # This constructor uses only two frequency vectors. Unknown constructors
        # must not allocate model parameters or establish an unchecked baseline.
        if type(module) is Gemma2TextScaledWordEmbedding and keys == {"embed_scale"}:
            full_name = f"{name}.embed_scale" if name else "embed_scale"
            # Construct only the scalar, never the vocabulary embedding matrix.
            buffers[full_name] = torch.tensor(module.scalar_embed_scale, dtype=torch.float32,
                                               device="cpu").to(dtype=actual[full_name].dtype)
            continue
        rotary_types = (qwen_rotary, Gemma2RotaryEmbedding)
        if _reference_profile(model) is not None:
            from ._gemma_reference import ReferenceRotaryEmbedding
            rotary_types += (ReferenceRotaryEmbedding,)
        if (type(module) not in rotary_types or module.rope_type != "default"
                or keys != {"inv_freq", "original_inv_freq"}):
            raise ValueError("unsupported checkpoint nonpersistent buffers")
        with torch.device("cpu"):
            rotary = type(module)(module.config)
        for key, value in rotary.named_buffers():
            full_name = f"{name}.{key}" if name else key
            buffers[full_name] = value.to(dtype=actual[full_name].dtype)
    return buffers


def checkpoint_settings(model, data, *, tensor_hash, admit_buffers):
    """Derive settings and buffer identity without checkpoint parameter storage."""
    # The text-only class uses the nested text config of a multimodal snapshot.
    if getattr(model.config, "model_type", None) == _qwen_family.FAMILY_NAME:
        data = _qwen_family.checkpoint_config(data)
    factory = getattr(type(model.config), "from_dict", None)
    if factory is None:
        raise ValueError("checkpoint verification requires a reconstructible model config")
    config = factory(data)
    # Reference construction must not select a backend from runtime configuration.
    config._attn_implementation = "eager"
    profile = _reference_profile(model)
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        if profile is None:
            reference = type(model)(config).eval()
        else:
            from transformers import Gemma2ForCausalLM
            from ._gemma_reference import install_reference_profile
            reference = Gemma2ForCausalLM(config).eval().to(dtype=getattr(torch, profile["dtype"]))
            install_reference_profile(reference, profile["profile"])
    settings = forward_settings(reference, tensor_hash=tensor_hash)
    return settings, tensor_hash(_checkpoint_buffers(reference, model, admit=admit_buffers))
