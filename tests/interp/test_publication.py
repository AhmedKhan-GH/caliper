"""Interpretation inputs before publication and complete standalone trace bindings."""
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest
import torch

from caliper import _controls, _programs as P
from caliper.interp import Tracer, _publish, publish
from caliper.interp._backend import make_tiny_backend
from caliper.interp._index import FeatureIndex, build_index
from caliper.interp._service import register_service, unregister_service
from .test_token_labels import execute

ROOT = Path(__file__).resolve().parents[2]
BINDINGS = {
    "attribution_graph": {name: "live:trace." + name.removeprefix("graph_") for name in
                          ("graph_counts", "node_i", "node_f", "edge_i", "edge_f", "label_text")},
    "logit_bars": {"probs": "live:trace.probs", "tokens": "live:trace.logit_tokens",
                   "label_text": "live:trace.logit_label_text"},
    "token_strip": {name: "live:examples." + name for name in
                    ("tokens", "activations", "example_ids", "valid_tokens", "label_text")},
}


@pytest.fixture
def registry(monkeypatch):
    for name in ("_PROGS", "_DRAWN", "_IMPORTS", "_LIVE", "_ACTS", "_ACT_HOOKS"):
        monkeypatch.setattr(P, name, {})
    monkeypatch.setattr(P, "_STATE", {**P._STATE, "model": None, "optimizer": None, "rt": None, "marks": None})
    monkeypatch.setattr(P, "_CAD", P._Cadence())
    monkeypatch.setattr(P, "_VIEWS", {"ring": None})
    monkeypatch.setattr(P, "ACTIVE", False)
    monkeypatch.setattr(P, "listeners", [])
    monkeypatch.setattr(_publish, "_TENSORS", {})
    monkeypatch.setattr(P, "_relay", lambda p: None)
    monkeypatch.setattr(P, "_relay_removal", lambda p: None)
    yield
    P.detach()


def install_piece(name, bindings=None):
    folder = ROOT / "src/builder/programs/library/pieces" / name / "1"
    manifest = tomllib.loads((folder / "piece.toml").read_text())
    return P.install_piece(
        name,
        piece={"name": name, "version": 1, "hash": "sha256:" + "0" * 64, "from": "library"},
        manifest=manifest,
        bind=bindings or BINDINGS[name],
        derive=(folder / "derive.wgsl").read_text(),
    )


def mock_producer(monkeypatch):
    imports, compiled = {}, []

    def bind(name, tensor):
        imports[name] = tensor
        return P._Import(name, len(imports), P._storage_key(tensor))

    def derive(*args, **kwargs):
        compiled.append(args[0])
        return len(compiled)

    runtime = SimpleNamespace(derive=derive, status=lambda handle: {}, remove=lambda handle: None)
    monkeypatch.setattr(P, "_import", bind)
    monkeypatch.setattr(P, "_runtime", lambda: runtime)
    return imports, compiled


@pytest.mark.parametrize("watched", [False, True])
def test_piece_creation_waits_then_compiles_after_real_queries(registry, monkeypatch, tmp_path, watched):
    backend = make_tiny_backend(n_layers=1)
    build_index(
        tmp_path, [("fixture", [2, 37, 38, 39, 40])], model=backend.model,
        tokenizer=backend.tokenizer, dictionary=backend.dictionary, provenance={"partition": "fixture"},
    )
    if watched:
        P.attach(backend.model, None)
    imports, compiled = mock_producer(monkeypatch)
    controls = _controls.Controls()
    with FeatureIndex(tmp_path, model=backend.model, tokenizer=backend.tokenizer, dictionary=backend.dictionary) as index:
        service = register_service(
            controls=controls, model=backend.model, tokenizer=backend.tokenizer,
            dictionary=backend.dictionary, index=index,
        )
        try:
            for name in BINDINGS:
                assert install_piece(name)["state"] == "waiting"
            assert not imports and not compiled
            execute(service, "trace.run", {"prompt": "A", "max_targets": 0, "max_nodes": 32, "max_edges": 16})
            with P._LOCK:
                P._refresh(P._PROGS["attribution_graph"])
                P._refresh(P._PROGS["logit_bars"])
                with pytest.raises(P._Waiting):
                    P._refresh(P._PROGS["token_strip"])
            assert set(compiled) == {"attribution_graph", "logit_bars"}
            execute(service, "feature.examples", {"layer": 0, "feature": int(index.counts[0].argmax())})
            with P._LOCK:
                P._refresh(P._PROGS["token_strip"])
            assert all(item["state"] == "compiled" for item in P.status())
            expected_tensors = {n for n in _publish.INTERPRETATION_TENSORS if n.startswith(("trace.", "examples."))}
            assert set(imports) == expected_tensors
            assert controls.interpretation_service is service
        finally:
            unregister_service(service, controls=controls)


@pytest.mark.parametrize("name", ["trace.probz", "examples.token", "trace.unknown", "custom.future"])
def test_unknown_input_refuses_even_after_known_pending_input(registry, name):
    P.attach(torch.nn.Linear(2, 2), None)
    bindings = {**BINDINGS["logit_bars"], "tokens": "live:" + name}
    result = install_piece("logit_bars", bindings)
    assert result["state"] == "refused"
    assert result["error"]["code"] == "program.unknown_input"
    assert P._resolve(name) is None


def test_fixed_names_wait_without_a_tensor_and_cpu_import_still_refuses(registry):
    with pytest.raises(P._Waiting):
        P._resolve("trace.probs")
    model = torch.nn.Linear(2, 2)
    P.attach(model, None)
    for name in _publish.INTERPRETATION_TENSORS:
        with pytest.raises(P._Waiting):
            P._resolve(name)
    value = torch.ones(2, 10)
    publish("trace.probs", value)
    resolved = P._resolve("trace.probs")
    assert resolved.data_ptr() == value.data_ptr() and resolved.device.type == "cpu"
    with pytest.raises(P._Refused) as refused:
        P._import("trace.probs", resolved)
    assert refused.value.error["code"] == "program.import_refused"
    P.detach(model)
    assert _publish._resolve("trace.probs") is None


def test_standalone_trace_and_cache_publish_all_piece_inputs(registry):
    backend = make_tiny_backend(n_layers=1)
    tracer = Tracer(backend.model, backend.tokenizer, backend.dictionary, backend=backend)
    graph = tracer.trace("A", max_targets=0, max_nodes=1, max_edges=0)
    assert len(graph.nodes) == 1 and len(graph.tensors) == 5
    published = dict(_publish._TENSORS)
    assert set(published) == {name for name in _publish.INTERPRETATION_TENSORS if name.startswith("trace.")}
    assert tracer.trace("A", max_targets=0, max_nodes=1, max_edges=0) is graph
