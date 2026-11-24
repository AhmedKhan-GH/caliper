"""Trace completeness for an exported dictionary with error-only lower layers."""
import torch

from caliper.interp import Tracer, load_dictionary
from tools.interp.checkpoint_state import export_dictionary
from tools.interp.train_transcoders import initial_progress
from .fixture import TinyTokenizer
from .test_dictionary import write_fixture
from .test_upper_half import fallback_manifest, tiny_model, upper_trainer


def test_exported_upper_half_trace_keeps_lower_errors_and_exact_completeness(tmp_path):
    model = tiny_model(2).double()
    write_fixture(tmp_path / "source", model)
    trainer = upper_trainer(fallback_manifest(model))
    trainer.initialize_upper(tmp_path / "source")
    folder = tmp_path / "dictionary"
    export_dictionary(folder, trainer, initial_progress())
    dictionary = load_dictionary(folder, model=model, dtype=torch.float64)
    tracer = Tracer(model, TinyTokenizer(), dictionary)
    graph = tracer.trace("AB", max_targets=32, max_nodes=64, max_edges=256)
    assert graph.completeness["max_absolute_residual"] <= 1e-9
    assert graph.completeness["replacement_forward_residual"] <= 1e-9
    assert graph.error_share > 0 and graph.feature_share > 0
    assert not any(node["kind"] == "feature" and node["layer"] == 0 for node in graph.nodes)
    path = tracer._forward_cache[1]
    assert path.features[0].count_nonzero() == 0
    assert path.errors[0].count_nonzero() > 0
    assert tracer.trace("AB", max_targets=32, max_nodes=64, max_edges=256) is graph
