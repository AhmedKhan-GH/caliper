"""Frozen paths, direct source effects, completeness, and graph contracts."""
import json

import pytest
import torch

from caliper.interp._graph import Graph


def test_graph_fixed_shapes_padding_and_paging():
    nodes = [{"id": "token:0:7", "kind": "token", "layer": -1, "position": 0,
              "token_id": 7, "activation": 1., "influence": .5, "effect": -.2,
              "label": "seven"},
             {"id": "logit:0:8", "kind": "logit", "layer": 4, "position": 0,
              "token_id": 8, "activation": .2, "influence": .5, "effect": 0.,
              "label": "eight"}]
    graph = Graph.build(nodes=nodes, edges=[{"source": 0, "target": 1, "weight": -.2}],
                        tokens=[{"token_id": 7, "text": "seven"}], layers=4,
                        completeness={"max_absolute_residual": 0., "targets": []},
                        omitted_influence=.1, error_share=0., feature_share=.4,
                        trace_id="example", device=torch.device("cpu"))
    expected = {"trace.counts": ((4,), torch.int32), "trace.node_i": ((512, 4), torch.int32),
                "trace.node_f": ((512, 4), torch.float32), "trace.edge_i": ((8192, 2), torch.int32),
                "trace.edge_f": ((8192, 2), torch.float32)}
    assert set(graph.tensors) == set(expected)
    for name, (shape, dtype) in expected.items():
        assert graph.tensors[name].shape == shape and graph.tensors[name].dtype == dtype
        if name != "trace.counts":
            assert graph.tensors[name][2 if "node" in name else 1:].count_nonzero() == 0
    assert graph.tensors["trace.counts"].tolist() == [2, 1, 1, 4]
    assert graph.tensors["trace.edge_f"][0].tolist() == pytest.approx([-.2, -1.])
    page = json.loads(graph.to_json(node_limit=1, edge_limit=1))
    assert page["next_node_offset"] == 1 and page["nodes"][0]["id"] == "token:0:7"
    assert graph.read(node_offset=1)["nodes"][0]["id"] == "logit:0:8"
    for argument in ({"node_limit": 257}, {"edge_offset": -1}, {"edge_limit": True}):
        with pytest.raises(ValueError):
            graph.read(**argument)
