"""CPU admission checks for the dictionary command limits."""
from types import SimpleNamespace

import pytest
import safetensors

from caliper.interp import _dictionary
from caliper.interp._memory import MemoryGuard
from tools.interp import build_index, eval_dictionary
from .fixture import TinyLM, TinyTokenizer
from .test_index import _t2_source_manifest


@pytest.mark.parametrize("command", [build_index, eval_dictionary])
def test_requested_limit_refuses_before_tensor_materialization(tmp_path, monkeypatch, command):
    model = TinyLM(layers=1, width=8, heads=2, latents=16, k=2)
    _t2_source_manifest(tmp_path, model)
    limits = []

    def guard(limit, device):
        limits.append(limit)
        return MemoryGuard(limit, device, resident=lambda: 0, footprint=lambda: 5_999_999_900,
                           peak=lambda: 0, driver=lambda: 0, pressure=lambda: 50)

    opened = safetensors.safe_open

    class HeadersOnly:
        def __init__(self, *args, **kwargs):
            self.source = opened(*args, **kwargs)

        def __enter__(self):
            source = self.source.__enter__()
            return SimpleNamespace(keys=source.keys, get_slice=source.get_slice,
                                   get_tensor=lambda *_: pytest.fail("tensor materialization ran"))

        def __exit__(self, *args):
            return self.source.__exit__(*args)

    monkeypatch.setattr(command, "MemoryGuard", guard)
    monkeypatch.setattr(_dictionary, "MemoryGuard", guard)
    monkeypatch.setattr(command, "load_text_model", lambda **_: (model, TinyTokenizer()))
    monkeypatch.setattr(command, "fineweb_documents", lambda **_: pytest.fail("dataset access ran"))
    monkeypatch.setattr(safetensors, "safe_open", HeadersOnly)
    arguments = ["--dictionary", str(tmp_path), "--device", "cpu", "--memory-gb", "6"]
    assert command.main(arguments) == 2
    assert limits == [6, 6]
