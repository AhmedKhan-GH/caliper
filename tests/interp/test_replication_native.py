"""Native evidence preserves actual precision and candidate ownership."""
import hashlib

import pytest
import torch

from caliper.interp._replication_native import write_native, write_native_tensor


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_actual_native_bytes_survive_export(tmp_path, dtype):
    values = {"logits": torch.tensor([[0., -0., 1.003], [2.09, -1., 0.]], dtype=dtype)}
    values["probabilities"] = values["logits"].softmax(-1)
    created, admitted = [], []
    target = tmp_path / "output.pt"
    descriptor = write_native(target, values, root=tmp_path, created=created, reserve=admitted.append)
    saved = torch.load(target, weights_only=True)
    assert created == [target] and len(admitted) == 1 and admitted[0] > 4 * 1024 * 1024
    assert descriptor["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    for name, tensor in values.items():
        assert saved[name].dtype == dtype and torch.equal(saved[name].view(torch.uint8), tensor.view(torch.uint8))
        assert descriptor["tensors"][name] == {"dtype": str(dtype), "shape": list(tensor.shape)}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_single_tensor_descriptor_preserves_actual_native_bytes(tmp_path, dtype):
    activation = torch.tensor([1.003], dtype=dtype)
    created, admitted = [], []
    target = tmp_path / "observation.pt"
    descriptor = write_native_tensor(target, "activation", activation,
                                     root=tmp_path, created=created, reserve=admitted.append)
    saved = torch.load(target, weights_only=True)
    assert set(saved) == {"activation"}
    assert saved["activation"].dtype == dtype
    assert torch.equal(saved["activation"].view(torch.uint8), activation.view(torch.uint8))
    assert created == [target] and len(admitted) == 1 and admitted[0] > 4 * 1024 * 1024
    assert descriptor == {"file": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                          "tensor": "activation", "dtype": str(dtype), "shape": [1]}


def test_native_reserve_refuses_before_copy_and_file_creation(tmp_path):
    created = []
    def refuse(amount):
        raise RuntimeError("fixture native reserve refusal")
    with pytest.raises(RuntimeError, match="native reserve refusal"):
        write_native(tmp_path / "output.pt", {"activation": torch.ones(1, dtype=torch.bfloat16)},
                     root=tmp_path, created=created, reserve=refuse)
    assert not created and not list(tmp_path.iterdir())


def test_existing_evidence_is_not_owned_or_replaced(tmp_path):
    target = tmp_path / "output.pt"
    target.write_bytes(b"existing evidence")
    created = []
    with pytest.raises(FileExistsError):
        write_native(target, {"activation": torch.zeros(1)}, root=tmp_path, created=created, reserve=lambda amount: None)
    assert target.read_bytes() == b"existing evidence" and created == []


def test_partial_native_write_retains_cleanup_ownership(tmp_path, monkeypatch):
    created = []
    def fail(values, stream):
        stream.write(b"partial")
        raise OSError("fixture native write failure")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="native write failure"):
        write_native(tmp_path / "output.pt", {"activation": torch.zeros(1)},
                     root=tmp_path, created=created, reserve=lambda amount: None)
    assert created == [tmp_path / "output.pt"]
    for target in created:
        target.unlink()
    assert not list(tmp_path.iterdir())
