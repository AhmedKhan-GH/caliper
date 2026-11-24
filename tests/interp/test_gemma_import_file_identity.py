"""Deterministic file identity checks for Gemma import bytes."""
from copy import copy
import hashlib
from io import BytesIO
import stat
from types import SimpleNamespace

import pytest

from caliper.interp import _gemma_import_identity as imported


@pytest.fixture
def file_case(monkeypatch):
    payload = b"Gemma import bytes\x00\xff\x01"
    path_info = SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_dev=3, st_ino=7,
                                st_size=len(payload), st_mtime_ns=3_000,
                                st_ctime_ns=1_000, st_birthtime_ns=1_000)
    descriptor_info = copy(path_info)
    # CPython 3.12.10 Windows stat() uses CreationTime for ctime, but
    # fstat() uses ChangeTime. Both report the same birthtime.
    descriptor_info.st_ctime_ns = 2_000
    source = BytesIO(payload)
    source.fileno = lambda: 7
    path = SimpleNamespace(stat=lambda: copy(path_info), open=lambda mode: source)
    monkeypatch.setattr(imported, "os", SimpleNamespace(fstat=lambda fd: copy(descriptor_info)))
    return SimpleNamespace(path=path, path_info=path_info, descriptor_info=descriptor_info,
                           source=source, payload=payload)


@pytest.mark.parametrize("same_ctime", [False, True])
def test_file_digest_accepts_stable_api_specific_ctime(file_case, same_ctime):
    if same_ctime:
        file_case.descriptor_info.st_ctime_ns = file_case.path_info.st_ctime_ns
    assert imported._file_digest(file_case.path, None) == hashlib.sha256(file_case.payload).hexdigest()
    assert file_case.source.closed


@pytest.mark.parametrize("api", ["path_info", "descriptor_info"])
def test_file_digest_rejects_ctime_change_with_stable_birthtime(file_case, monkeypatch, api):
    read = file_case.source.read

    def changed(size):
        chunk = read(size)
        getattr(file_case, api).st_ctime_ns += 100
        return chunk

    monkeypatch.setattr(file_case.source, "read", changed)
    with pytest.raises(ValueError, match="import binding"):
        imported._file_digest(file_case.path, None)
    assert file_case.source.closed
