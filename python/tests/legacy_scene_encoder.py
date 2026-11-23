"""Legacy scene encoders used only by codec fixtures and saved-recording tests."""
from caliper import _wire as w
from caliper._wire import (SceneUpdate, Section, MAX_FRAME_BYTES, MAX_UPDATE_BYTES, MAX_PIECES,
                          MAX_TIMELINES, SECTION_ALIGN, HEADER_BYTES, PART_SECTIONS, WireError,
                          frame_parts, canonical_json, np)


def legacy_sections(self):
    out = [w._section(name, self.arrays[name]) for name in w.PART_SECTIONS.get(self.part, ())
           if self.arrays.get(name) is not None]
    if self.stamp is not None:
        out.append(w._section("stamp", self.stamp))
    return out


def split_update(u: SceneUpdate, update: int, max_frame: int = MAX_FRAME_BYTES) -> list:
    """The pieces of scene update ``u`` (split counter ``update``), each frame
    within ``max_frame`` bytes, or ``[u]`` when it fits one frame. Rows are cut
    along the first axis of each array in the part's order; piece 0 carries
    the part's keys and ``rows``, every piece the stamp. Raises ValueError
    for an update over MAX_UPDATE_BYTES or one that needs more than MAX_PIECES
    pieces."""
    try:
        _, size = frame_parts(u.TYPE, u.to_meta(), u.to_sections())
    except WireError:                                # over MAX_FRAME_BYTES: it does not fit
        size = max_frame + 1
    if size <= max_frame:
        return [u]
    arrays = []
    for name in PART_SECTIONS[u.part]:
        a = u.arrays.get(name)
        if a is not None:
            arrays.append((name, a.array() if isinstance(a, Section) else np.asarray(a)))
    total = sum(a.nbytes for _, a in arrays)
    if total > MAX_UPDATE_BYTES:
        raise ValueError(f"a scene update of {total} bytes, more than {MAX_UPDATE_BYTES}")
    # A bound on every piece's head: the header, the section table, padding,
    # and metadata with the part's keys, the largest "split" key and the
    # descriptors of the whole arrays.
    rows = {name: int(a.shape[0]) for name, a in arrays}
    probe = u.to_meta()
    probe["split"] = {"update": update, "piece": MAX_PIECES - 1, "pieces": MAX_PIECES,
                      "first": rows, "rows": rows}
    sections = [Section.from_array(name, a[:0]) for name, a in arrays]
    probe["sections"] = [dict(x.descriptor(), shape=[int(n) for n in a.shape])
                         for x, (_, a) in zip(sections, arrays)]
    nsec = len(arrays) + (u.stamp is not None)
    stamp_bytes = 0 if u.stamp is None else 8 * MAX_TIMELINES + SECTION_ALIGN
    head = HEADER_BYTES + 4 * nsec + len(canonical_json(probe)) + 64 + SECTION_ALIGN * nsec
    room = max_frame - head - stamp_bytes
    if room <= 0:
        raise ValueError(f"a frame of {max_frame} bytes leaves no room for rows")
    pieces, cur, used = [], {}, 0
    for name, a in arrays:
        rowbytes = max(1, a.nbytes // a.shape[0]) if a.shape[0] else 1
        r = 0
        while r < a.shape[0]:
            take = min(a.shape[0] - r, max(0, (room - used) // rowbytes))
            if take == 0:
                if not cur:
                    raise ValueError(f"one row of {name!r} needs more than {room} bytes")
                pieces.append(cur)
                cur, used = {}, 0
                continue
            cur[name] = (r, a[r:r + take])
            used += take * rowbytes
            r += take
    if cur:
        pieces.append(cur)
    if not 2 <= len(pieces) <= MAX_PIECES:
        raise ValueError(f"the update needs {len(pieces)} pieces; a split update has 2 to "
                         f"{MAX_PIECES}")
    out = []
    for k, cut in enumerate(pieces):
        sp = {"update": update, "piece": k, "pieces": len(pieces),
              "first": {name: int(r) for name, (r, _) in cut.items()}}
        if k == 0:
            sp["rows"] = rows
        out.append(SceneUpdate(u.sub, u.view, u.seq, u.stamp_ns, u.path, u.part,
                               {name: a for name, (_, a) in cut.items()},
                               dict(u.params) if k == 0 else {}, u.static, u.state, u.axes,
                               u.items, u.stamp, u.cursor, sp))
        if frame_parts(u.TYPE, out[-1].to_meta(), out[-1].to_sections())[1] > max_frame:
            raise ValueError(f"piece {k} would pass {max_frame} bytes")
    return out



def install():
    w.SceneUpdate.to_sections = legacy_sections
    w.split_update = split_update
