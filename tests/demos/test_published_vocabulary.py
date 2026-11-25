"""The published acceptance path requires every Gemma output column."""
import json

import numpy as np
import pytest
import torch

from test_published_dallas import (callback_boundary, evidence_factory, published, reviewed_collection, write_json)
from _replication_review import main, write_collection


def rebind_result(root, review, result):
    write_json(root / "driver/result.json", result)
    write_collection(root / "driver", root, result)
    review["collection_sha256"] = published.file_hash(root / "driver/collection.json")
    write_json(root / "review.json", review)


def assert_cli_refusal(root, message, *, failed_checks=("external_replication_review",)):
    assert main(["--private-output-directory", str(root), "--collection", "driver/collection.json",
                 "--review", "review.json", "--out", str(root / "refused.json")]) == 1
    verified = json.loads((root / "refused.json").read_text())
    assert not any(verified[key] for key in ("accepted_replication", "gemma_acceptance", "acceptance_complete"))
    gate = next(item for item in verified["checks"] if item["name"] == "external_replication_review")
    assert message in gate["measured"]["reason"]
    assert verified["failed_checks"] == list(failed_checks)


@pytest.mark.parametrize("reviewed_collection,compared", [(5, 10), (5, 512000), (255999, 511998), (256001, 512002)],
                         indirect=["reviewed_collection"])
def test_review_cli_refuses_truncated_vocabulary(reviewed_collection, compared):
    root, review, result = reviewed_collection
    width = result["published_actions"][0]["evidence"]["result"]["evidence"]["shape"][1]
    assert result["dictionary_evidence"]["index"]["vocab_size"] == 256000
    assert {tuple(action["evidence"]["result"]["evidence"]["shape"])
            for action in result["published_actions"]} == {(2, width)}
    assert {row["probabilities_compared"] for row in review["cases"]} == {2 * width}
    for row in review["cases"]:
        row["probabilities_compared"] = compared
    write_json(root / "review.json", review)
    assert_cli_refusal(root, "all 256000 vocabulary columns")


@pytest.mark.parametrize("action_index", [0, 13, 2, 14], ids=["initial-baseline", "fresh-baseline", "first-case", "last-case"])
def test_every_callback_requires_profile_width(reviewed_collection, evidence_factory, action_index):
    root, review, result = reviewed_collection
    _, _, make = evidence_factory
    action = result["published_actions"][action_index]
    original = action["evidence"]["result"]
    action["status"] = make(original["cell"], original["sequence"], vocab_size=5)
    action["evidence"] = published.validate_case(root, action["status"], original["cell"],
                                                previous_sequence=original["sequence"] - 1)
    for row in review["cases"]:
        if row["caliper_sequence"] == original["sequence"]:
            row["caliper_npz_sha256"] = action["evidence"]["result"]["evidence"]["sha256"]
            row["probabilities_compared"] = 10
    rebind_result(root, review, result)
    assert_cli_refusal(root, "all 256000 vocabulary columns")


@pytest.mark.parametrize("cell,compared", [
    (10, 256000), (14, 10), (29, 511999), (14, 512001), (14, 512000.), (14, True), (14, "512000"),
])
def test_full_width_requires_exact_integer_comparison_count(reviewed_collection, cell, compared):
    root, review, _ = reviewed_collection
    row = next(row for row in review["cases"] if row["case_id"] == published.action_name(cell))
    row["probabilities_compared"] = compared
    write_json(root / "review.json", review)
    assert_cli_refusal(root, "full-vocabulary reference agreement failed")


@pytest.mark.parametrize("reviewed_collection", [5, 256000], indirect=True)
def test_dictionary_width_cannot_redefine_profile(reviewed_collection):
    root, review, result = reviewed_collection
    dictionary = result["dictionary_evidence"]
    dictionary["index"]["vocab_size"] = 5
    dictionary["index"]["index_hash"] = published.digest({
        key: value for key, value in dictionary["index"].items() if key != "index_hash"})
    dictionary["identity"]["index_hash"] = dictionary["index"]["index_hash"]
    rebind_result(root, review, result)
    assert main(["--private-output-directory", str(root), "--collection", "driver/collection.json",
                 "--review", "review.json", "--out", str(root / "refused.json")]) == 1
    verified = json.loads((root / "refused.json").read_text())
    assert not any(verified[key] for key in ("accepted_replication", "gemma_acceptance", "acceptance_complete"))
    assert "exact example policy" in verified["dictionary_qualification"]["reason"]
    width = result["published_actions"][0]["evidence"]["result"]["evidence"]["shape"][1]
    assert verified["failed_checks"] == ["dictionary_qualification"] + (
        ["external_replication_review"] if width == 5 else [])


@pytest.mark.parametrize("fault,message", [
    ("npz_logits_width", "Private arrays have invalid values or shape"),
    ("npz_probabilities_width", "Private arrays have invalid values or shape"),
    ("npz_baseline_only", "Private arrays have invalid values or shape"),
    ("native_actual_width", "Native tensor shape or storage differs"),
    ("native_declared_width", "Native tensor declarations differ"),
    ("native_baseline_tail", "Native bytes differ from private arrays"),
    ("native_case_tail", "Native bytes differ from private arrays"),
    ("baseline_restore_tail", "fresh baseline does not restore the original output"),
])
def test_profile_width_binds_actual_arrays_and_native_rows(reviewed_collection, fault, message):
    root, review, result = reviewed_collection
    action = result["published_actions"][0 if fault == "baseline_restore_tail" else 2]
    record = action["status"]["inference"]["replication"]["result"]
    with np.load(root / record["evidence"]["file"], allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    if fault.startswith("npz_"):
        if fault == "npz_baseline_only":
            arrays = {name: value[:1] for name, value in arrays.items()}
        else:
            name = "logits" if fault == "npz_logits_width" else "probabilities"
            arrays[name] = arrays[name][:, :5]
    elif fault == "baseline_restore_tail":
        arrays["logits"][1, -1] = .25
    if fault.startswith("npz_") or fault == "baseline_restore_tail":
        archive = root / "changed-arrays.npz"
        np.savez(archive, **arrays)
        record["evidence"].update(published.descriptor(root, archive))
    if fault.startswith("native_") or fault == "baseline_restore_tail":
        tensors = {name: torch.from_numpy(value).to(torch.bfloat16) for name, value in arrays.items()}
        if fault == "native_actual_width":
            tensors = {name: value[:, :5].contiguous() for name, value in tensors.items()}
        elif fault in ("native_baseline_tail", "native_case_tail"):
            tensors["probabilities"][0 if fault == "native_baseline_tail" else 1, -1] = .25
        native = root / "changed-native.pt"
        with native.open("wb") as stream:
            torch.save(tensors, stream)
        record["native_evidence"].update(published.descriptor(root, native))
        if fault == "native_declared_width":
            for tensor in record["native_evidence"]["tensors"].values():
                tensor["shape"] = [2, 5]
    write_json(root / record["metadata_file"], {key: value for key, value in record.items()
                                              if key not in ("metadata_file", "action_seconds")})
    action["evidence"].update(status=action["status"], result=record,
                               metadata=published.descriptor(root, root / record["metadata_file"]))
    rebind_result(root, review, result)
    assert_cli_refusal(root, message)
