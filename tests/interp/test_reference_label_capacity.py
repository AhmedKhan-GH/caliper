"""Complete reference labels through CPU trace publication and private export."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from caliper._controls import Controls
from caliper.interp import Tracer, load_dictionary, _publish, _trace
from caliper.interp._graph import REFERENCE_NODE_CAP, TOKEN_CAP
from caliper.interp._index import FeatureIndex, LABEL_FEATURES, build_index
from caliper.interp._memory import CleanStop, MemoryGuard
from caliper.interp._reference_graph import PROFILE
from caliper.interp._replication_export import export_graph
from .fixture import TinyLM, TinyTokenizer
from .test_dictionary import write_fixture
from .test_token_labels import unpack_label


FEATURES = LABEL_FEATURES + 1
OPTIONS = dict(profile=PROFILE, max_targets=FEATURES, max_nodes=REFERENCE_NODE_CAP,
               max_edges=8192, max_n_logits=1, batch_size=128, update_interval=1,
               node_threshold=1., edge_threshold=1.)


@pytest.fixture
def capacity_case(tmp_path, monkeypatch, request):
    model = TinyLM(layers=1, width=8, heads=2, latents=FEATURES, k=FEATURES)
    with torch.no_grad():
        model.layers[0].mlp.b_enc.fill_(10.)
    root = tmp_path / "dictionary"
    write_fixture(root, model)
    tokenizer = TinyTokenizer()
    dictionary = load_dictionary(root, model=model)
    memory = MemoryGuard(12, "cpu", resident=lambda: 0, footprint=lambda: 0,
                         peak=lambda: 0, driver=lambda: 0, pressure=lambda: 80)
    index = None
    if getattr(request, "param", True):
        build_index(root, [("fixture", [65, 66])], model=model, tokenizer=tokenizer,
                    dictionary=dictionary, provenance={"partition": "fixture"}, context_length=2)
        index = FeatureIndex(root, model=model, tokenizer=tokenizer, dictionary=dictionary, memory=memory)
    monkeypatch.setattr(_publish, "_TENSORS", {})
    monkeypatch.setattr(_publish, "_VERSIONS", {})
    monkeypatch.setattr(_publish, "_VERSION", 0)
    tracer = Tracer(model, tokenizer, dictionary, controls=Controls(), index=index, memory=memory)
    inputs, released = [], []
    decorate, release = _trace.label_graph_nodes, _trace._DirectEffects.release_decoders

    def labels(nodes, *args, **kwargs):
        inputs.append((nodes, deepcopy(nodes)))
        return decorate(nodes, *args, **kwargs)

    def release_decoders(effects):
        release(effects)
        released.append(effects)

    monkeypatch.setattr(_trace, "label_graph_nodes", labels)
    monkeypatch.setattr(_trace._DirectEffects, "release_decoders", release_decoders)
    yield SimpleNamespace(model=model, tracer=tracer, index=index, memory=memory,
                          inputs=inputs, released=released, root=root)
    if index is not None:
        index.close()


def hooks(model):
    return [(tuple(module._forward_hooks), tuple(module._forward_pre_hooks)) for module in model.modules()]


def assert_cleanup(case, before):
    assert hooks(case.model) == before
    assert case.released and all(effects._decoder_blocks is None for effects in case.released)
    assert all(nodes == original for nodes, original in case.inputs)
    for module in case.model.modules():
        assert getattr(module, "frozen_scale", None) is None
        assert getattr(module, "frozen_pattern", None) is None


@pytest.mark.parametrize("capacity_case", [False, True], indirect=True, ids=["without-index", "with-index"])
def test_reference_trace_publishes_complete_labels(capacity_case, monkeypatch, tmp_path):
    case = capacity_case
    batches, expected, callbacks = [], {}, []
    contents = {}
    if case.index is not None:
        contents = {file.name: file.read_bytes() for file in (case.root / "index").iterdir()}
        labels = case.index.labels

        def record(features, *, check):
            batches.append(list(features))
            callbacks.append(check)
            result = labels(features, check=check)
            expected.update(result)
            return result

        monkeypatch.setattr(case.index, "labels", record)
    before = hooks(case.model)
    graph = case.tracer.trace("A", **OPTIONS)
    features = [node for node in graph.nodes if node["kind"] == "feature"]
    keys = [(node["layer"], node["feature"]) for node in features]
    assert len(features) == len(set(keys)) == FEATURES
    assert len(graph.nodes) > LABEL_FEATURES
    assert graph.completeness["feature_targets"] == FEATURES
    assert graph.completeness["max_absolute_residual"] <= 1e-9
    assert graph.completeness["source_scope"] == "full_sources_before_reference_projection"
    assert all(node["position"] == 1 for node in features)
    if case.index is not None:
        assert batches == [keys[:LABEL_FEATURES], keys[LABEL_FEATURES:]]
        assert callbacks[0] is callbacks[1]
        assert len(expected) == FEATURES and len(case.index._summaries) == LABEL_FEATURES
        assert all(node["label"] == expected[key] for node, key in zip(features, keys))
        assert all(label.startswith("Output: ") for label in expected.values())
        assert {file.name: file.read_bytes() for file in (case.root / "index").iterdir()} == contents
    else:
        assert all(node["label"] == f"L{key[0]} F{key[1]}" for node, key in zip(features, keys))
    assert all(isinstance(node["label"], str) and node["label"] for node in graph.nodes)
    assert _publish._TENSORS["trace.counts"].tolist() == [len(graph.nodes), len(graph.edges), 2, 1]
    assert _publish._TENSORS["trace.node_i"].shape == (REFERENCE_NODE_CAP, 4)
    assert _publish._TENSORS["trace.label_text"].shape == (TOKEN_CAP + REFERENCE_NODE_CAP, 16)
    assert [unpack_label(row) for row in _publish._TENSORS["trace.label_text"][
        TOKEN_CAP:TOKEN_CAP + len(graph.nodes)]] == [node["label"] for node in graph.nodes]
    assert [unpack_label(row) for row in _publish._TENSORS["trace.label_text"][:2]] == [
        token["text"] for token in graph.tokens]
    for name, tensor in graph.tensors.items():
        assert torch.equal(_publish._TENSORS[name], tensor)

    nodes, edges = [], []
    node_offset = edge_offset = 0
    while node_offset is not None or edge_offset is not None:
        page = graph.read(node_offset=len(graph.nodes) if node_offset is None else node_offset,
                          edge_offset=len(graph.edges) if edge_offset is None else edge_offset)
        nodes.extend(page["nodes"])
        edges.extend(page["edges"])
        node_offset, edge_offset = page["next_node_offset"], page["next_edge_offset"]
    assert nodes == graph.nodes and edges == graph.edges
    exported = json.loads(graph.to_json())
    assert exported["nodes"] == graph.nodes and exported["edges"] == graph.edges
    assert exported["completeness"] == graph.completeness and exported["settings"] == graph.settings
    destination = tmp_path / "evidence"
    destination.mkdir()
    descriptor = export_graph(graph, destination)
    raw = (destination / descriptor["file"]).read_bytes()
    assert json.loads(raw) == exported
    assert hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
    assert export_graph(graph, destination) == descriptor
    assert len(list(destination.iterdir())) == 1

    def refuse(*args, **kwargs):
        pytest.fail("a cached graph must retain its complete labels")

    if case.index is not None:
        monkeypatch.setattr(case.index, "labels", refuse)
    assert case.tracer.trace("A", **OPTIONS) is graph
    assert len(case.tracer._cache) == 1 and case.tracer._forward_cache is not None
    assert_cleanup(case, before)


@pytest.mark.parametrize("failure", ["index", "deadline", "memory"])
def test_later_reference_label_batch_failure_preserves_publication(capacity_case, monkeypatch, tmp_path, failure):
    case = capacity_case
    prior = case.tracer.trace("A", max_targets=0)
    destination = tmp_path / "evidence"
    destination.mkdir()
    descriptor = export_graph(prior, destination)
    evidence = {file.name: file.read_bytes() for file in destination.iterdir()}
    publication = dict(_publish._TENSORS)
    versions, version = dict(_publish._VERSIONS), _publish._VERSION
    cached = dict(case.tracer._cache)
    before = hooks(case.model)
    case.index.clear_cache()
    batches = []
    labels = case.index.labels

    def fail_later(features, *, check):
        batches.append(list(features))
        if len(batches) == 2:
            if failure == "index":
                raise RuntimeError("later index batch failed")
            if failure == "deadline":
                check.expires = 0
            if failure == "memory":
                case.memory.limit = 1
        return labels(features, check=check)

    monkeypatch.setattr(case.index, "labels", fail_later)
    error, message = {"index": (RuntimeError, "later index batch failed"),
                      "deadline": (_trace.TraceTimeout, "20 second request budget"),
                      "memory": (CleanStop, "feature label allocation")}[failure]
    with pytest.raises(error, match=message):
        graph = case.tracer.trace("A", **OPTIONS)
        export_graph(graph, destination)
    assert [len(batch) for batch in batches] == [LABEL_FEATURES, 1]
    assert len(case.inputs[-1][0]) > LABEL_FEATURES
    assert list(case.index._summaries) == batches[0]
    assert case.tracer._cache == cached and case.tracer._forward_cache is None
    assert _publish._TENSORS.keys() == publication.keys()
    assert all(_publish._TENSORS[name] is tensor for name, tensor in publication.items())
    assert _publish._VERSIONS == versions and _publish._VERSION == version
    assert {file.name: file.read_bytes() for file in destination.iterdir()} == evidence
    assert descriptor["file"] in evidence
    assert_cleanup(case, before)


def test_ordinary_trace_refuses_expanded_capacity(capacity_case, monkeypatch):
    case = capacity_case

    def refuse(*args, **kwargs):
        pytest.fail("ordinary overflow must fail before model or label work")

    monkeypatch.setattr(case.model, "forward", refuse)
    monkeypatch.setattr(case.index, "labels", refuse)
    with pytest.raises(ValueError, match="max_nodes must be an integer from 1 to 512"):
        case.tracer.trace("A", max_nodes=LABEL_FEATURES + 1)
    assert not _publish._TENSORS and not case.tracer._cache
    assert case.tracer._forward_cache is None and not case.inputs
