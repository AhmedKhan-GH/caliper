"""The tests of the tensor output and the params op of the PROGRAM message in the script.

See design/instrument-interface.md, sections Tensor Output and Program Parameters.
"""
from __future__ import annotations

import pytest
import torch

from caliper import _programs as P
from test_programs import MPS, _train, run  # noqa: F401

try:
    from caliper import _producer
    ABI = int(getattr(_producer, "ABI_VERSION", 0))
except ImportError:
    ABI = 0

pytestmark = pytest.mark.skipif(
    not (MPS and ABI >= 2),
    reason="needs the MPS device and caliper._producer with ABI version 2 (tensor output and params)")

# Four numbers of the weight, scaled and shifted by the uniform params.
AFFINE = """struct Params { scale: f32, bias: f32 }
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(4)
fn affine(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= 4u) { return; }
    out[id.x] = w[id.x] * params.scale + params.bias;
}
"""

DECL = {"inputs": [{"var": "w", "source": "live:fc.weight"}],
        "output": {"kind": "tensor", "shape": [2, 2]},
        "params": {"scale": 2.0, "bias": 0.5}, "workgroups": [1, 1, 1]}


def test_a_tensor_program_publishes_its_array_and_reads_new_params_at_the_next_pass(run):
    net, opt, _, shown = run
    item = P.install("t", DECL, AFFINE, sha256="ab")
    assert item["state"] == "compiled", item["error"]
    assert item["kind"] == "tensor"
    _train(net, opt, P.WARMUP_STEPS)
    w = net.fc.weight.detach().cpu().flatten()[:4]   # the weight the pass reads: before the update
    _train(net, opt, 1, first=P.WARMUP_STEPS)
    name, out = shown[-1]
    assert name == "program.t" and out.shape == (2, 2) and out.dtype == torch.float32
    torch.testing.assert_close(out.cpu().flatten(), w * 2.0 + 0.5, atol=1e-5, rtol=1e-5)

    sha = P.status()[0]["sha256"]
    P.set_params("t", {"scale": -1.0})               # no compile: the same program and hash
    assert P.status()[0]["sha256"] == sha == "ab"
    w = net.fc.weight.detach().cpu().flatten()[:4]
    _train(net, opt, 1, first=P.WARMUP_STEPS + 1)
    out = shown[-1][1]
    torch.testing.assert_close(out.cpu().flatten(), w * -1.0 + 0.5, atol=1e-5, rtol=1e-5)
    P.remove("t")
    assert shown[-1] == ("removed", "program.t")
