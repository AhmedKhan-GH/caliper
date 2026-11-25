"""Complete private graph evidence for the explicit published runtime."""
import hashlib
import os
from pathlib import Path
import re
import uuid


def export_graph(graph, directory):
    """Preserve full diagnostics and actual schedules outside bounded replies."""
    if not isinstance(graph.trace_id, str) or re.fullmatch(r"[0-9a-f]{64}", graph.trace_id) is None:
        raise ValueError("private graph evidence requires a complete trace identity")
    directory = Path(directory).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("private graph evidence requires an existing directory")
    payload = graph.to_json()
    # Graph JSON uses ASCII escapes, so character count equals encoded bytes.
    if graph._memory is not None:
        graph._memory.reserve("private graph evidence", len(payload) + 4 * 1024 * 1024)
    raw = payload.encode("ascii")
    digest = hashlib.sha256(raw).hexdigest()
    target = directory / ("trace-" + graph.trace_id + ".json")
    temporary = directory / (".trace-" + uuid.uuid4().hex + ".json")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A hard link publishes complete bytes without replacing prior evidence.
            os.link(temporary, target)
        except FileExistsError:
            with target.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError("private graph evidence changed for the same trace")
    finally:
        temporary.unlink(missing_ok=True)
    return {"file": target.name, "sha256": digest}
