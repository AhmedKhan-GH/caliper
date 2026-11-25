"""Bounded complete graph evidence and unchanged metadata limits."""
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_published_dallas import published
from _mcp import Failure


@pytest.mark.parametrize("reader,limit", [
    (published.read_json, published.JSON_EVIDENCE_LIMIT),
    (published.read_full_graph, published.FULL_GRAPH_JSON_LIMIT),
])
def test_oversized_json_refuses_before_open(tmp_path, monkeypatch, reader, limit):
    target = tmp_path / "oversized.json"
    with target.open("wb") as stream:
        stream.truncate(limit + 1)

    def refuse_open(*args, **kwargs):
        pytest.fail("Oversized evidence reached the file read.")

    monkeypatch.setattr(Path, "open", refuse_open)
    with pytest.raises(Failure, match="JSON exceeds the evidence limit"):
        reader(target)
    assert published.JSON_EVIDENCE_LIMIT == 16 * 1024 * 1024


def test_full_graph_rechecks_open_file_size(tmp_path, monkeypatch):
    target = tmp_path / "replaced.json"
    with target.open("wb") as stream:
        stream.truncate(published.FULL_GRAPH_JSON_LIMIT + 1)
    monkeypatch.setattr(Path, "stat", lambda self: SimpleNamespace(st_size=0))
    with pytest.raises(Failure, match="Full graph JSON exceeds"):
        published.read_full_graph(target)


@pytest.mark.parametrize("extra", [-1, 0, 1])
def test_full_graph_bounds_read_after_file_growth(monkeypatch, extra):
    limit = 64
    raw = b'{"label":"\\u03b1"}'
    raw += b" " * (limit + extra - len(raw))
    reads = []

    class GrowingFile(io.BytesIO):
        def fileno(self):
            return -1

        def read(self, size=-1):
            reads.append(size)
            assert size == limit + 1
            return super().read(size)

    target = SimpleNamespace(stat=lambda: SimpleNamespace(st_size=0),
                             open=lambda mode: GrowingFile(raw))
    monkeypatch.setattr(published, "FULL_GRAPH_JSON_LIMIT", limit)
    monkeypatch.setattr(published.os, "fstat", lambda fd: SimpleNamespace(st_size=0))
    if extra > 0:
        with pytest.raises(Failure, match="Full graph JSON exceeds"):
            published.read_full_graph(target)
    else:
        assert published.read_full_graph(target) == {"label": "α"}
    assert reads == [limit + 1]


@pytest.mark.parametrize("reader", [published.read_json, published.read_full_graph])
@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_nonfinite_json_refuses_in_nested_fields(tmp_path, reader, number):
    target = tmp_path / "nonfinite.json"
    target.write_text('{"completeness":{"targets":[{"expected":' + number + '}]}}', encoding="utf-8")
    with pytest.raises(Failure, match="Nonfinite JSON evidence"):
        reader(target)
