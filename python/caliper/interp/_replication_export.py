"""Complete private graph evidence for the explicit published runtime."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import re
import uuid


def _workspace(graph, check):
    """Bound JSON chunks, their joined text, encoded bytes, and file workspace."""
    # These references match Graph.to_json without a full record copy or buffer.
    value = graph.read(node_limit=0, edge_limit=0)
    value.update(nodes=graph.nodes, edges=graph.edges, completeness=graph.completeness,
                 logits=graph.logits, settings=graph.settings, next_node_offset=None, next_edge_offset=None)
    count = 0

    def size(item):
        nonlocal count
        count += 1
        if count % 1024 == 0:
            check("private graph sizing")
        if item is None or type(item) is bool:
            return 5
        if type(item) is str:
            # An astral character can require two six-byte ASCII escapes.
            return 2 + 12 * len(item)
        if type(item) is int:
            return 2 + (abs(item).bit_length() * 30103 // 100000)
        if type(item) is float:
            return 32
        if type(item) in (list, tuple):
            return 2 + sum(size(child) + 1 for child in item)
        if type(item) is dict:
            return 2 + sum(size(key) + size(child) + 2 for key, child in item.items())
        raise TypeError("private graph evidence requires JSON values")

    encoded = size(value)
    # The encoder can retain small string chunks and list pointers before join.
    return 4 * 1024 * 1024 + 3 * encoded + 128 * count


def _no_check(stage):
    pass


class _Evidence:
    def __init__(self):
        self.created = []

    def rollback(self):
        for target, identity in reversed(self.created):
            try:
                current = target.stat()
                if (current.st_dev, current.st_ino) == identity:
                    target.unlink()
            except FileNotFoundError:
                pass


@contextmanager
def _evidence_scope(evidence=None):
    owned = evidence is None
    evidence = _Evidence() if owned else evidence
    try:
        yield evidence
    except BaseException:
        if owned:
            evidence.rollback()
        raise


def export_graph(graph, directory, *, memory=None, check=_no_check, evidence=None):
    """Preserve full diagnostics and actual schedules outside bounded replies."""
    if not isinstance(graph.trace_id, str) or re.fullmatch(r"[0-9a-f]{64}", graph.trace_id) is None:
        raise ValueError("private graph evidence requires a complete trace identity")
    directory = Path(directory).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("private graph evidence requires an existing directory")
    check("private graph admission")
    memory = graph._memory if memory is None else memory
    workspace = _workspace(graph, check)
    if memory is not None:
        memory.reserve("private graph evidence", workspace)
    check("private graph serialization")
    payload = graph.to_json()
    raw = payload.encode("ascii")
    digest = hashlib.sha256(raw).hexdigest()
    check("private graph serialized")
    target = directory / ("trace-" + graph.trace_id + ".json")
    temporary = directory / (".trace-" + uuid.uuid4().hex + ".json")
    with _evidence_scope(evidence) as evidence:
        try:
            with temporary.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
                stat = os.fstat(stream.fileno())
            check("private graph publication")
            try:
                # A hard link publishes complete bytes without replacement.
                os.link(temporary, target)
                evidence.created.append((target, (stat.st_dev, stat.st_ino)))
            except FileExistsError:
                with target.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                        raise ValueError("private graph evidence changed for the same trace")
        finally:
            temporary.unlink(missing_ok=True)
        if memory is not None:
            memory.check("private graph complete")
        check("private graph complete")
    return {"file": target.name, "sha256": digest}


class _TraceTransaction:
    def __init__(self, service):
        self.service = service
        self.evidence = _Evidence()
        self.attributes = []
        self.mappings = []

    def remember(self, owner, *names):
        for name in names:
            self.attributes.append((owner, name, name in vars(owner), getattr(owner, name, None)))

    def cache(self, owner, *names):
        for name in names:
            value = getattr(owner, name, None)
            if value is not None:
                self.mappings.append((value, value.copy()))

    def controls(self):
        """Snapshot controls after any concurrent edit and before hosted commit."""
        owner = self.service.owner
        self.parameters = [(param, param._value, param._version, param._pending)
                           for param in (owner.prompt, getattr(owner, "selected", None)) if param is not None]
        self.remember(owner, "checkpoint_prompt", "traced_prompt", "traced_specifications", "traced_state",
                      "selection", "_selected_version", "_selection_graph")

    def rollback(self):
        service = self.service
        for owner, name, present, value in reversed(self.attributes):
            if present:
                setattr(owner, name, value)
            else:
                vars(owner).pop(name, None)
        for mapping, previous in self.mappings:
            mapping.clear()
            mapping.update(previous)
        if self.tracer is None and service.tracer is not None:
            service.tracer._cache.clear()
            service.tracer._forward_cache = service.tracer._prompt = None
        with service.owner.controls.lock:
            for param, value, version, pending in getattr(self, "parameters", ()):
                param._value, param._version, param._pending = value, version, pending
            service.owner.controls._update_busy()


@contextmanager
def trace_transaction(service):
    """Keep published trace state private until the complete request succeeds."""
    from .. import _programs
    from . import _publish

    transaction = _TraceTransaction(service)
    # Producers and watch replacement use this lock. They cannot observe the
    # candidate tensors while export or hosted commit can still fail.
    with _programs._LOCK:
        epoch = _programs._WATCH_EPOCH
        tensors, versions, version = dict(_publish._TENSORS), dict(_publish._VERSIONS), _publish._VERSION
        transaction.remember(service, "graph")
        transaction.tracer = service.tracer
        if service.tracer is not None:
            transaction.remember(service.tracer, "_forward_cache", "_prompt")
            transaction.cache(service.tracer, "_cache")
        executor = service.owner.interventions.executor
        transaction.remember(executor, "_cache_identity")
        transaction.cache(executor, "_cache")
        transaction.cache(service.index, "_summaries")
        try:
            yield transaction
        except BaseException:
            transaction.rollback()
            # A same-thread watch replacement owns its own publication state.
            if _programs._WATCH_EPOCH == epoch:
                _publish._TENSORS, _publish._VERSIONS, _publish._VERSION = tensors, versions, version
            transaction.evidence.rollback()
            raise
