"""Tamper checks for native case evidence and complete collection coverage."""
import pickle
import zipfile

import pytest
import torch

from test_published_dallas import evidence_factory, published, write_json
from _mcp import Failure


def sync_metadata(root, record):
    write_json(root / record["metadata_file"],
               {key: value for key, value in record.items() if key not in ("metadata_file", "action_seconds")})


def replace_native(root, descriptor, tensors):
    target = root / descriptor["file"]
    with target.open("wb") as stream:
        torch.save(tensors, stream)
    descriptor.update(published.descriptor(root, target))
    descriptor["tensors"] = {name: {"dtype": str(value.dtype), "shape": list(value.shape)}
                             for name, value in tensors.items()}


@pytest.mark.parametrize("fault", ["omitted", "tampered", "dtype", "shape", "values", "declaration"])
def test_native_arrays_bind_actual_bytes(evidence_factory, fault):
    root, _, make = evidence_factory
    status = make(14)
    record = status["inference"]["replication"]["result"]
    native = record["native_evidence"]
    if fault == "omitted":
        record.pop("native_evidence")
    elif fault == "tampered":
        (root / native["file"]).write_bytes(b"changed")
    elif fault == "declaration":
        native["tensors"]["logits"]["dtype"] = "torch.float32"
    else:
        values = torch.load(root / native["file"], weights_only=True)
        if fault == "dtype":
            values["logits"] = values["logits"].float()
        elif fault == "shape":
            values["logits"] = values["logits"].reshape(-1)
        else:
            values["logits"][0, 0] = 2
        replace_native(root, native, values)
    sync_metadata(root, record)
    with pytest.raises(Failure):
        published.validate_case(root, status, 14, previous_sequence=0)


@pytest.mark.parametrize("fault", ["omitted", "node", "native_omitted", "tampered", "nonzero", "nan", "dtype", "shape"])
def test_cell29_requires_actual_zero_observation(evidence_factory, fault):
    root, _, make = evidence_factory
    status = make(29)
    record = status["inference"]["replication"]["result"]
    observation = record["observations"]["Say Victoria"]
    native = observation["native_evidence"]
    if fault == "omitted":
        record.pop("observations")
    elif fault == "node":
        observation["node"]["feature"] += 1
    elif fault == "native_omitted":
        observation.pop("native_evidence")
    elif fault == "tampered":
        (root / native["file"]).write_bytes(b"changed")
    else:
        activation = torch.tensor([1. if fault == "nonzero" else float("nan") if fault == "nan" else 0.],
                                  dtype=torch.float32 if fault == "dtype" else torch.bfloat16)
        if fault == "shape":
            activation = activation.reshape(1, 1)
        replace_native(root, native, {"activation": activation})
    sync_metadata(root, record)
    with pytest.raises(Failure):
        published.validate_case(root, status, 29, previous_sequence=0)


@pytest.mark.parametrize("fault", ["pickle_global", "large_member", "large_file", "duplicate"])
def test_observation_reader_refuses_unqualified_archives(evidence_factory, fault):
    root, _, make = evidence_factory
    status = make(29)
    record = status["inference"]["replication"]["result"]
    native = record["observations"]["Say Victoria"]["native_evidence"]
    target = root / native["file"]
    with zipfile.ZipFile(target) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    metadata = next(name for name in entries if name.endswith("/data.pkl"))
    if fault == "pickle_global":
        entries[metadata] = pickle.dumps(len)
    elif fault == "large_member":
        entries[metadata] = b"0" * (128 * 1024)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
        if fault == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate"):
                archive.writestr(metadata, entries[metadata])
    if fault == "large_file":
        with target.open("ab") as stream:
            stream.truncate(32 * 1024 * 1024 + 1)
    native.update(published.descriptor(root, target))
    sync_metadata(root, record)
    with pytest.raises(Failure):
        published.validate_case(root, status, 29, previous_sequence=0)
