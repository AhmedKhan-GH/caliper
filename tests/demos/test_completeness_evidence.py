"""Account coverage and residual measurements without model execution."""
from copy import deepcopy

import pytest

from test_circuit_trace import driver  # Install the demo import path.
from _completeness import BIAS_FIELDS, account_evidence, summary_evidence
from _mcp import Failure


def complete_accounts():
    targets = [{"id": identifier, "expected": 1., "incoming": 4.,
                "biases": dict.fromkeys(BIAS_FIELDS, 0.), "absolute_residual": 3., "relative_residual": 3.}
               for identifier in ("logit:10:1", "feature:9:1:13520")]
    return {"counts": {"layers": 26, "tokens": 11},
            "settings": {"physical_batches": [1, 1], "logical_queues": [[targets[0]["id"]], [targets[1]["id"]]]},
            "logits": [{"id": targets[0]["id"]}],
            "completeness": {"numeric_status": "measured", "source_scope": "full_sources_before_reference_projection",
                "objective": "centered_pre_softcap_logit", "target_count": 2, "feature_targets": 1, "logit_targets": 1,
                "max_absolute_residual": 3., "max_relative_residual": 3.,
                "forward_residual": 7., "direct_forward_residual": 5., "replacement_forward_residual": 7.,
                "forward_residuals": {"direct_reads": [5.] * 26, "direct_logits": 2.,
                                      "replacement_reads": [7.] * 26, "replacement_logits": 6.},
                "targets": targets}}


def test_large_residuals_remain_measurements_without_equivalence():
    graph = complete_accounts()
    original = deepcopy(graph)
    result = account_evidence(graph)
    assert result["inventory_verified"] and result["numeric_status"] == "measured"
    assert result["reference_equivalence"] == "not_established"
    assert result["source_inventory"] == "requires_independent_review"
    assert result["summary"]["max_absolute_residual"] == 3.
    assert result["summary"]["forward_residual"] == 7.
    assert graph == original


@pytest.mark.parametrize("fault", ["missing_targets", "short_targets", "duplicate", "queue_order", "missing_queue",
    "batch_count", "batch_boolean", "crossed_queue", "logits", "counts", "boolean_count", "missing_bias", "extra_bias",
    "nonfinite_bias", "missing_expected", "nonfinite_incoming", "negative_residual", "relative_definition", "summary",
    "malformed_id", "invalid_coordinate", "missing_numeric_status", "wrong_scope", "missing_forward_group",
    "short_forward_reads", "nonfinite_forward", "boolean_forward", "forward_summary"])
def test_incomplete_or_malformed_accounts_refuse(fault):
    graph = complete_accounts()
    record, settings = graph["completeness"], graph["settings"]
    target = record["targets"][0]
    if fault == "missing_targets": record.pop("targets")
    elif fault == "short_targets": record["targets"].pop()
    elif fault == "duplicate": record["targets"][1] = deepcopy(target)
    elif fault == "queue_order": settings["logical_queues"].reverse()
    elif fault == "missing_queue": settings.pop("logical_queues")
    elif fault == "batch_count": settings["physical_batches"] = [1]
    elif fault == "batch_boolean": settings["physical_batches"] = [True, 1]
    elif fault == "crossed_queue": settings["physical_batches"] = [2]
    elif fault == "logits": graph["logits"][0]["id"] = "logit:10:2"
    elif fault == "counts": record["feature_targets"] += 1
    elif fault == "boolean_count": record["feature_targets"] = True
    elif fault == "missing_bias": target["biases"].pop("attention")
    elif fault == "extra_bias": target["biases"]["balancing"] = 0.
    elif fault == "nonfinite_bias": target["biases"]["attention"] = float("nan")
    elif fault == "missing_expected": target.pop("expected")
    elif fault == "nonfinite_incoming": target["incoming"] = float("inf")
    elif fault == "negative_residual": target["absolute_residual"] = -3.
    elif fault == "relative_definition": target["relative_residual"] = 1.
    elif fault == "summary": record["max_absolute_residual"] = .001
    elif fault in ("malformed_id", "invalid_coordinate"):
        value = "feature:26:11:1" if fault == "invalid_coordinate" else "unknown:1"
        target["id"] = settings["logical_queues"][0][0] = graph["logits"][0]["id"] = value
    elif fault == "missing_numeric_status": record.pop("numeric_status")
    elif fault == "wrong_scope": record["source_scope"] = "projection_only"
    elif fault == "missing_forward_group": record["forward_residuals"].pop("direct_logits")
    elif fault == "short_forward_reads": record["forward_residuals"]["replacement_reads"].pop()
    elif fault == "nonfinite_forward": record["forward_residuals"]["direct_logits"] = float("inf")
    elif fault == "boolean_forward": record["forward_residuals"]["direct_reads"][0] = True
    elif fault == "forward_summary": record["replacement_forward_residual"] = .001
    with pytest.raises(Failure):
        account_evidence(graph)


def test_account_recomputation_keeps_the_original_residual():
    graph = complete_accounts()
    graph["completeness"]["targets"][0]["incoming"] = 1000.
    result = account_evidence(graph)
    row = result["account_recomputation"]["targets"][0]
    assert row == {"id": "logit:10:1", "recorded_residual": 3., "binary64_residual": 999., "absolute_difference": 996.}
    assert result["reference_equivalence"] == "not_established"


@pytest.mark.parametrize("value", [None, "missing", [], 1])
def test_malformed_target_records_refuse(value):
    graph = complete_accounts()
    graph["completeness"]["targets"][0] = value
    with pytest.raises(Failure):
        account_evidence(graph)


@pytest.mark.parametrize("identifier", ["logit:10:256000", "feature:9:1:16384"])
def test_target_coordinates_stay_within_model_identity(identifier):
    graph = complete_accounts()
    index = int(identifier.startswith("feature"))
    graph["completeness"]["targets"][index]["id"] = identifier
    graph["settings"]["logical_queues"][index][0] = identifier
    if not index:
        graph["logits"][0]["id"] = identifier
    with pytest.raises(Failure):
        account_evidence(graph)
