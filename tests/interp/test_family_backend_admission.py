"""Backend admission follows the supported family's actual attention path."""
import pytest

from caliper.interp._dictionary import _state_hash
from caliper.interp._forward_identity import forward_settings
from .test_gemma_family import gemma_model
from .test_points import qwen


def test_gemma_refuses_sdpa_after_eager_verification():
    model = gemma_model(1)
    model.config._attn_implementation = "eager"
    forward_settings(model, tensor_hash=_state_hash)
    model.config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="Gemma.*eager.*softcap"):
        forward_settings(model, tensor_hash=_state_hash)


def test_qwen_retains_eager_and_sdpa_identity():
    model = qwen()
    model.config._attn_implementation = "eager"
    eager = forward_settings(model, tensor_hash=_state_hash)
    model.config._attn_implementation = "sdpa"
    assert forward_settings(model, tensor_hash=_state_hash) == eager
