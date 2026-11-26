"""Complete account evidence with numerical residuals kept as measurements."""
from __future__ import annotations

import math
import re

from _published import digest, finite, require
from _published_qualification import VOCAB_SIZE

BIAS_FIELDS = ("decoder", "normalization", "attention", "intervention", "encoder", "unembedding")
FORWARD_FIELDS = {"direct_reads", "direct_logits", "replacement_reads", "replacement_logits"}
SOURCE_SCOPE = "full_sources_before_reference_projection"
OBJECTIVE = "centered_pre_softcap_logit"


def summary_evidence(graph):
    """Validate complete forward diagnostics without an equivalence claim."""
    record = graph["completeness"]
    layers = graph["counts"]["layers"]
    tokens = graph["counts"]["tokens"]
    require(type(tokens) is int and tokens > 0, "Completeness token count is invalid.")
    require(type(layers) is int and layers > 0 and record.get("numeric_status") == "measured"
            and record.get("source_scope") == SOURCE_SCOPE and record.get("objective") == OBJECTIVE,
            "Completeness numerical status or account scope is absent.")
    for key in ("target_count", "feature_targets", "logit_targets"):
        require(type(record.get(key)) is int and record[key] >= 0, "Completeness target counts are invalid.")
    require(record["logit_targets"] > 0 and record["target_count"] == record["feature_targets"] + record["logit_targets"],
            "Completeness target counts do not partition the inventory.")
    for key in ("max_absolute_residual", "max_relative_residual", "forward_residual",
                "direct_forward_residual", "replacement_forward_residual"):
        require(finite(record.get(key)), "Completeness residual measurement is absent or nonfinite.")
    forward = record.get("forward_residuals")
    require(isinstance(forward, dict) and set(forward) == FORWARD_FIELDS,
            "All four forward residual groups are required.")
    for prefix in ("direct", "replacement"):
        reads, logits = forward[prefix + "_reads"], forward[prefix + "_logits"]
        require(isinstance(reads, list) and len(reads) == layers and all(finite(value) for value in reads)
                and finite(logits), "Forward residual layer coverage is incomplete or invalid.")
        require(record[prefix + "_forward_residual"] == max(*reads, logits),
                "Forward residual summary differs from its measurements.")
    require(record["forward_residual"] == max(record["direct_forward_residual"], record["replacement_forward_residual"]),
            "Forward residual maximum differs from its measurements.")
    return {key: value for key, value in record.items() if key != "targets"}


def account_evidence(graph):
    """Check target, queue, and bias inventories in the complete export."""
    summary = summary_evidence(graph)
    targets = graph["completeness"].get("targets")
    settings = graph["settings"]
    require(isinstance(targets, list) and len(targets) == summary["target_count"]
            and all(isinstance(row, dict) for row in targets),
            "Full graph target diagnostics are incomplete.")
    queues, batches = settings.get("logical_queues"), settings.get("physical_batches")
    require(isinstance(queues, list) and queues and all(isinstance(queue, list) and queue for queue in queues)
            and isinstance(batches, list) and batches and all(type(size) is int and size > 0 for size in batches),
            "Full graph schedules are absent or invalid.")
    ids = [row.get("id") for row in targets]
    require(all(isinstance(value, str) for value in ids) and len(set(ids)) == len(ids)
            and [value for queue in queues for value in queue] == ids
            and sum(batches) == len(ids), "Target and queue inventories differ.")
    queue_ends, batch_ends = set(), set()
    for values, ends in (([len(queue) for queue in queues], queue_ends), (batches, batch_ends)):
        total = 0
        for size in values:
            total += size
            ends.add(total)
    require(queue_ends <= batch_ends, "A physical batch crosses a logical queue boundary.")
    logits = graph.get("logits")
    require(isinstance(logits, list) and all(isinstance(row, dict) for row in logits) and [row.get("id") for row in logits] == queues[0]
            and len(logits) == summary["logit_targets"], "The first queue differs from the logit inventory.")
    kinds = []
    discrepancies = []
    for row in targets:
        match = re.fullmatch(r"(logit|feature):([0-9]+):([0-9]+)(?::([0-9]+))?", row["id"])
        require(match is not None, "A target identity is malformed.")
        kind, first, second, feature = match.groups()
        require((kind == "logit" and feature is None and int(first) == graph["counts"]["tokens"] - 1
                 and int(second) < VOCAB_SIZE)
                or (kind == "feature" and feature is not None and int(first) < graph["counts"]["layers"]
                    and int(second) < graph["counts"]["tokens"] and int(feature) < 16384), "A target coordinate is invalid.")
        kinds.append(kind)
        biases = row.get("biases")
        require(isinstance(biases, dict) and set(biases) == set(BIAS_FIELDS)
                and all(finite(value, -math.inf) for value in biases.values())
                and all(finite(row.get(key), -math.inf) for key in ("expected", "incoming"))
                and all(finite(row.get(key)) for key in ("absolute_residual", "relative_residual")),
                "A target account is absent or nonfinite.")
        require(row["relative_residual"] == row["absolute_residual"] / max(1., abs(row["expected"])),
                "A relative residual differs from its declared definition.")
        # Python binary64 evaluates the recorded account scalars in this order.
        # The producer uses its recorded tensor dtype. Preserve both results.
        recomputed = abs(row["incoming"] + sum(biases[key] for key in BIAS_FIELDS[:4])
                         + biases["encoder"] + biases["unembedding"] - row["expected"])
        discrepancy = abs(recomputed - row["absolute_residual"])
        require(finite(recomputed) and finite(discrepancy), "Account recomputation is nonfinite.")
        discrepancies.append({"id": row["id"], "recorded_residual": row["absolute_residual"],
                              "binary64_residual": recomputed, "absolute_difference": discrepancy})
    require(kinds == ["logit"] * summary["logit_targets"] + ["feature"] * summary["feature_targets"],
            "Target kinds differ from their inventory counts.")
    for field, key in (("max_absolute_residual", "absolute_residual"), ("max_relative_residual", "relative_residual")):
        require(summary[field] == max(row[key] for row in targets), "Target residual summary differs from its measurements.")
    return {"inventory_verified": True, "numeric_status": "measured", "reference_equivalence": "not_established",
            "source_scope": SOURCE_SCOPE, "target_count": len(ids), "target_ids_sha256": digest(ids),
            "summary": summary, "account_recomputation": {"dtype": "python_binary64",
                "expression": "abs(incoming + sum(decoder, normalization, attention, intervention) + encoder + unembedding - expected)",
                "targets": discrepancies},
            "source_inventory": "requires_independent_review"}
