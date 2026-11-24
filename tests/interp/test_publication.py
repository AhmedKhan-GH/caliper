"""Interpretation inputs before publication and complete standalone trace bindings."""
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest
import torch

from caliper import _controls, _programs as P
from caliper.interp import Tracer, publish
from caliper.interp import _publish
from caliper.interp._index import FeatureIndex, build_index
from caliper.interp._labels import display_text
from caliper.interp._service import register_service, unregister_service
from .fixture import TinyLM, TinyTokenizer
from .test_dictionary import write_fixture
from .test_token_labels import execute, unpack_label
from .test_trace import exact_dictionary


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
    return P.install_piece(name, piece={"name": name, "version": 1, "hash": "sha256:" + "0" * 64,
                                       "from": "library"}, manifest=manifest,
                           bind=bindings or BINDINGS[name], derive=(folder / "derive.wgsl").read_text())


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


def test_piece_creation_waits_then_compiles_after_real_queries(registry, monkeypatch, tmp_path):
    model, tokenizer = TinyLM(layers=1, width=8, heads=2, latents=8, k=2), TinyTokenizer()
    write_fixture(tmp_path, model)
    dictionary = exact_dictionary(model)
    build_index(tmp_path, [("fixture", [65, 66, 67, 68])], model=model, tokenizer=tokenizer,
                dictionary=dictionary, provenance={"partition": "fixture"})
    P.attach(model, None)
    imports, compiled = mock_producer(monkeypatch)
    controls = _controls.Controls()
    with FeatureIndex(tmp_path, model=model, tokenizer=tokenizer, dictionary=dictionary) as index:
        service = register_service(controls=controls, model=model, tokenizer=tokenizer,
                                   dictionary=dictionary, index=index)
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
            assert set(imports) == _publish.INTERPRETATION_TENSORS
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


def test_fixed_names_wait_only_with_a_watch_and_cpu_import_still_refuses(registry):
    assert P._resolve("trace.probs") is None
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
    assert P._STATE["rt"] is None
    P.detach(model)
    assert P._resolve("trace.probs") is None


def test_standalone_trace_and_cache_publish_all_piece_inputs_before_pruning(registry):
    model, tokenizer = TinyLM(layers=1, width=8, heads=2, latents=8, k=2), TinyTokenizer()
    tracer = Tracer(model, tokenizer, exact_dictionary(model))
    forwards = []
    handle = model.register_forward_hook(lambda *args: forwards.append(True))
    try:
        graph = tracer.trace("A", max_targets=0, max_nodes=1, max_edges=0)
        assert len(forwards) == 1 and len(graph.nodes) == 1
        assert len(graph.tensors) == 5
        published = dict(_publish._TENSORS)
        assert set(published) == {name for name in _publish.INTERPRETATION_TENSORS if name.startswith("trace.")}
        expected = tracer._forward_cache[1].logits[0].softmax(-1)
        order = torch.argsort(expected, descending=True, stable=True)[:10]
        count = min(10, int((expected[order].cumsum(0) < .95).sum()) + 1)
        assert count > len(graph.nodes)
        assert published["trace.logit_tokens"][:count].tolist() == order[:count].tolist()
        torch.testing.assert_close(published["trace.probs"][:, :count], expected[order[:count]].float().repeat(2, 1))
        assert published["trace.label_text"].shape == (544, 16)
        assert unpack_label(published["trace.label_text"][1]) == "A"
        assert unpack_label(published["trace.label_text"][32]) == graph.nodes[0]["label"]
        for index, token in enumerate(order[:count].tolist()):
            assert unpack_label(published["trace.logit_label_text"][index]) == display_text(tokenizer.decode([token]))
        assert all(tensor.device.type == "cpu" for tensor in published.values())
        _publish._clear()
        assert tracer.trace("A", max_targets=0, max_nodes=1, max_edges=0) is graph
        assert len(forwards) == 1
        for name, tensor in published.items():
            torch.testing.assert_close(_publish._TENSORS[name], tensor)
    finally:
        handle.remove()
