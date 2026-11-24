"""Deterministic file identity checks for Gemma import bytes."""
from copy import copy
import hashlib
from io import BytesIO
import stat
from types import SimpleNamespace

import pytest

from caliper.interp import _gemma_import_identity as imported
from caliper.interp._memory import CleanStop


@pytest.fixture
def file_case(monkeypatch):
    payload = b"Gemma import bytes\x00\xff\x01"
    path_info = SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_dev=3, st_ino=7,
                                st_size=len(payload), st_mtime_ns=2_000,
                                st_ctime_ns=1_000, st_birthtime_ns=1_000)
    descriptor_info = copy(path_info)
    # CPython 3.12.10 Windows stat() uses CreationTime for ctime, but
    # fstat() uses ChangeTime. Both report the same birthtime.
    descriptor_info.st_ctime_ns = 2_000
    source = BytesIO(payload)
    source.fileno = lambda: 7
    reads = []
    read = source.read

    def tracked(size):
        reads.append(size)
        return read(size)

    source.read = tracked
    path = SimpleNamespace(stat=lambda: copy(path_info), open=lambda mode: source)
    monkeypatch.setattr(imported, "os", SimpleNamespace(fstat=lambda fd: copy(descriptor_info)))
    yield SimpleNamespace(path=path, path_info=path_info, descriptor_info=descriptor_info,
                          source=source, payload=payload, reads=reads)
    source.close()


@pytest.mark.parametrize("same_ctime", [False, True])
def test_file_digest_accepts_stable_api_specific_ctime(file_case, same_ctime):
    if same_ctime:
        file_case.descriptor_info.st_ctime_ns = file_case.path_info.st_ctime_ns
    assert imported._file_digest(file_case.path, None) == hashlib.sha256(file_case.payload).hexdigest()
    assert file_case.source.closed


@pytest.mark.parametrize("api", ["path_info", "descriptor_info"])
@pytest.mark.parametrize("field", ["st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"])
def test_file_digest_rejects_change_during_read(file_case, monkeypatch, api, field):
    read = file_case.source.read

    def changed(size):
        chunk = read(size)
        info = getattr(file_case, api)
        # Only the selected field changes. Birthtime stays fixed.
        setattr(info, field, getattr(info, field) + 100)
        return chunk

    monkeypatch.setattr(file_case.source, "read", changed)
    with pytest.raises(ValueError, match="import binding"):
        imported._file_digest(file_case.path, None)
    assert file_case.source.closed


@pytest.mark.parametrize("field", ["st_dev", "st_ino", "st_size", "st_mtime_ns"])
def test_file_digest_rejects_path_descriptor_mismatch_before_read(file_case, field):
    info = file_case.descriptor_info
    setattr(info, field, getattr(info, field) + 1)
    with pytest.raises(ValueError, match="import binding"):
        imported._file_digest(file_case.path, None)
    assert file_case.reads == []
    assert file_case.source.closed


@pytest.mark.parametrize("size", [0, 1, 31])
def test_file_digest_matches_full_sha256_for_local_bytes(tmp_path, monkeypatch, size):
    payload = bytes(range(size))
    path = tmp_path / "layer.safetensors"
    path.write_bytes(payload)
    monkeypatch.setattr(imported, "_CHUNK_BYTES", 7)
    assert imported._file_digest(path, None) == hashlib.sha256(payload).hexdigest()


def test_file_digest_bounds_requests_and_accepts_short_reads(file_case, monkeypatch):
    read = file_case.source.read
    requests = []

    def short_read(size):
        assert 0 < size <= 7
        requests.append(size)
        return read(min(size, 3))

    monkeypatch.setattr(imported, "_CHUNK_BYTES", 7)
    monkeypatch.setattr(file_case.source, "read", short_read)
    assert imported._file_digest(file_case.path, None) == hashlib.sha256(file_case.payload).hexdigest()
    assert len(requests) > 3 and requests[-1] == 1


@pytest.mark.parametrize("difference", [-1, 1])
def test_file_digest_rejects_extra_bytes_and_early_eof(file_case, difference):
    file_case.path_info.st_size += difference
    file_case.descriptor_info.st_size += difference
    with pytest.raises(ValueError, match="import binding"):
        imported._file_digest(file_case.path, None)
    assert file_case.source.closed


def test_file_digest_rejects_oversized_file_before_read(file_case, monkeypatch):
    monkeypatch.setattr(imported, "_WIDTH", 1)
    monkeypatch.setattr(imported, "_LATENTS", 1)
    file_case.path_info.st_size = file_case.descriptor_info.st_size = 8 * 5 + 65_536 + 1
    with pytest.raises(ValueError, match="import binding"):
        imported._file_digest(file_case.path, None)
    assert file_case.reads == []
    assert file_case.source.closed


@pytest.mark.parametrize("api", ["path_info", "descriptor_info"])
@pytest.mark.parametrize("mode", [stat.S_IFDIR, stat.S_IFIFO])
def test_file_digest_rejects_nonregular_files_before_read(file_case, api, mode):
    getattr(file_case, api).st_mode = mode
    with pytest.raises(ValueError, match="import binding"):
        imported._file_digest(file_case.path, None)
    assert file_case.reads == []
    assert file_case.source.closed == (api == "descriptor_info")


@pytest.mark.parametrize("completed_reads,stop", [(0, "guard"), (1, "guard"), (0, "budget")])
def test_file_digest_honors_guard_stops(file_case, monkeypatch, completed_reads, stop):
    checks = []

    def check(stage, *, force):
        checks.append((stage, force))
        if len(file_case.reads) == completed_reads:
            if stop == "guard":
                raise CleanStop("fixture stop")
            memory.limit = 0

    memory = SimpleNamespace(limit=1_000_000_000, peak=0, check=check)
    monkeypatch.setattr(imported, "_CHUNK_BYTES", 7)
    with pytest.raises(CleanStop, match="fixture stop|allocation exceeds"):
        imported._file_digest(file_case.path, memory)
    assert len(file_case.reads) == completed_reads
    assert checks == [("model_identity", True)] + [("model_identity", False)] * (2 * completed_reads)
    assert file_case.source.closed == bool(completed_reads)
