// The codec against the golden frames of the protocol 3.2 draft's scenes subset
// (tests/fixtures/wire_v3/v3_2): every frame decodes to what its .json says,
// every invalid frame is refused as expect.json says, and every frame but the
// one that carries a section the subset ignores encodes back at minor 2 to the
// same metadata and sections, and to the same bytes when its metadata text is
// the canonical one (JavaScript writes 2 where Python writes 2.0, so a frame
// with a float key such as vmin is compared by metadata and section bytes).
// Runs in Node, once.

import fs from 'node:fs';
import path from 'node:path';
import {
  canonicalJson, decode, decodeFrame, encode, encodeFrame, entityPathOk, MessageError, resolveSceneDataView, sceneViewInfo, SplitResources, SplitUpdates,
  stampTextOk, textFromLines, textLines, WireError,
  type Section,
} from '../src/protocol/codec.js';
import {
  CAMERA_REQUIRED, CAP_SCENES, CLEAR_PARTS, COLOR_PER_EDGE, COLOR_PER_STRIP, COLOR_PER_VERTEX, CONVENTION_RDF, CONVENTION_RUB,
  DTYPE, DTYPE_SIZE, FORM_COLOR, FORM_DEPTH, FORM_ENCODED, FORM_SEGMENTATION, FORMAT_JPEG, FORMAT_PNG, KIND_SCENE, LIMITS_32,
  LINES_EDGES, LINES_SEGMENTS, LINES_STRIPS, MAX_CLASS_ID, MAX_CLASSES, MAX_ENCODED_BYTES, MAX_IMAGE_SIDE, MAX_LABELS, MAX_TIMELINES,
  MAX_TRANSFORMS, MSG, PART_ARROWS, PART_BOXES, PART_CAMERA, PART_IMAGE, PART_KEYS, PART_LINES, PART_MESH, PART_POINTS,
  PART_SECTIONS, PART_TRANSFORM, PROTOCOL_MINOR_3_2, SCENE_DATA_VIEW, SCENE_PARTS, STAMP_ABSENT,
  CAP_BARS, CAP_RECORDINGS, CAP_TEXT, CAP_TIMELINES, CURSOR_RATE_HZ, ERR_BAD_CURSOR, ERR_BAD_RECORDING, ERR_NOT_SEEKABLE,
  ERR_UNKNOWN_RECORDING, KIND_IDS, LEVEL_DEBUG, LEVEL_ERROR, LEVEL_INFO, LEVEL_WARNING, MAX_BARS, MAX_CURSOR_STEP, MAX_INCOMPLETE_UPDATES,
  MAX_PIECES, MAX_RECORDED, MAX_RECORDINGS, MAX_SPEED, MAX_UPDATE_BYTES, MIN_SPEED, PACE_EVEN, PACE_RECORDED, RECORDING_LINGER_MS,
  STAMP_TEXT_MAX, TEXT_LINE_MAX, TEXT_MAX_LINES, TEXT_RETAIN_MIN, TIME_FOLLOWING, TIME_PAUSED, TIME_PLAYING, TIME_STATE_RATE_HZ,
  TIMELINE_SEQUENCE, TIMELINE_TIME,
  CAP_INTERVENTIONS, CAP_PROGRAMS, INTERVENE_ABLATE, MAX_INTERVENTIONS, MAX_POINTS_LISTED, MAX_PROGRAMS, POINT_MAX, PROGRAM_EVERY_MAX,
  PROGRAM_ID_MAX, PROGRAM_MAX_COUNT, PROGRAM_MAX_INPUTS, PROGRAM_MEMORY_MB_MAX, PROGRAM_REMOVE, PROGRAM_SET, PROGRAM_SIDE_MAX,
  PROGRAM_SOURCE_MAX, PROGRAM_STATE_RATE_HZ, PROGRAM_TIME_MS_MAX, PROGRAM_WORKGROUPS_MAX, TENSOR_NAME_MAX,
  PIECE_COMMON_MAX, PIECE_NAME_MAX, PIECE_SEGMENTS_MAX, PIECE_VERSION_MAX, RESOURCE_MAX_BYTES, RESOURCE_PART_WAIT, RESOURCE_SIDE_MAX, RESOURCE_STRIDE_MAX,
  RESOURCE_VERSION_MAX_BYTES, STATUS_PIECE,
  type ControlMsg, type ControlsMsg, type Message, type ProgramMsg, type ProgramStateMsg, type ResourceMsg,
  type CursorMsg, type SceneClassesMsg, type SceneTransformsMsg, type SceneUpdateMsg, type SnapshotMsg, type TextMsg, type TimeStateMsg,
  type ViewEntry, type ViewsMsg,
} from '../src/protocol/messages.js';
import { expect, goldenFrames, REPO, test } from './util.js';

test.beforeEach(({}, info) => test.skip(info.project.name !== 'canvas2d', 'the codec runs in Node once'));

const dir = path.join(goldenFrames(), 'v3_2');
const names = fs.readdirSync(dir).filter((f) => f.endsWith('.bin')).map((f) => f.slice(0, -4)).sort();
const DECODE_ONLY = ['scene_update_points_future'];
const typeName = (id: number) => Object.entries(MSG).find(([, v]) => v === id)?.[0] ?? null;
/** The values of a raw section as a typed array over its bytes (copied when unaligned). */
function values(s: Section): Float32Array | Int32Array | Uint8Array | Uint16Array | BigInt64Array {
  const c = s.bytes.slice();
  const b = c.buffer;
  switch (s.dtype) {
    case DTYPE.u8: return c;
    case DTYPE.u16: return new Uint16Array(b);
    case DTYPE.i32: return new Int32Array(b);
    case DTYPE.i64: return new BigInt64Array(b);
    default: return new Float32Array(b);
  }
}
/** A decoded i64 stamp is a BigInt64Array over the frame's bytes, present exactly when the section is. */
function checkStamp(f: ReturnType<typeof decodeFrame>, got: BigInt64Array | undefined, name: string): void {
  const s = f.sections.find((x) => x.name === name);
  if (!s) {
    expect(got, name).toBeUndefined();
    return;
  }
  expect(got, name).toBeInstanceOf(BigInt64Array);
  expect(Array.from(bytesOf(got!)), name).toEqual(Array.from(s.bytes));
  expect(got!.buffer).toBe(f.sections[0].bytes.buffer);
}
const bytesOf = (a: ArrayBufferView) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
const read = (...p: string[]) => new Uint8Array(fs.readFileSync(path.join(dir, ...p)));

function catchError(fn: () => unknown): unknown {
  try {
    fn();
  } catch (e) {
    return e;
  }
  return null;
}

test('the 3.2 golden frames are there', () => {
  expect(names.length).toBeGreaterThanOrEqual(66);
  expect(names).toContain('scene_update_mesh');
  expect(names).toContain('scene_transforms_stamps');
  expect(typeName(0x70)).toBe('scene_update');
  expect(typeName(0x71)).toBe('scene_transforms');
  expect(typeName(0x72)).toBe('scene_clear');
  expect(typeName(0x73)).toBe('scene_classes');
  expect([0x74, 0x78, 0x79, 0x7a, 0x7b].map(typeName)).toEqual(['text', 'recordings', 'recording_open', 'cursor', 'time_state']);
  for (const n of ['text', 'cursor_follow', 'series_decimated', 'snapshot_bars', 'time_state_following', 'scene_update_split_0']) expect(names).toContain(n);
});

for (const name of names) {
  test(`decodes and encodes v3_2/${name}`, () => {
    const buf = read(`${name}.bin`);
    const want = JSON.parse(fs.readFileSync(path.join(dir, `${name}.json`), 'utf8'));
    const f = decodeFrame(buf);
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    expect({
      magic: dv.getUint32(0, true), major: f.major, minor: f.minor, type: f.type, flags: f.flags,
      frame_len: buf.byteLength, meta_len: dv.getUint32(16, true), section_count: f.sections.length, reserved: dv.getUint16(22, true),
    }).toEqual(want.header);
    expect(f.minor).toBe(PROTOCOL_MINOR_3_2);
    expect(f.meta).toEqual(want.meta);
    expect(f.sections.map((s) => ({ name: s.name, dtype: s.dtype, shape: s.shape, encoding: s.encoding, offset: s.bytes.byteOffset, length: s.bytes.byteLength })))
      .toEqual(want.sections.map((s: Record<string, unknown>) => ({ encoding: 0, name: s.name, dtype: s.dtype, shape: s.shape, offset: s.offset, length: s.length })));
    // Section values: f32, i32, u8 and u16 as numbers; i64 is compared as bytes only (JSON loses it above 2^53).
    f.sections.forEach((s, i) => {
      const w = want.sections[i];
      if (s.dtype === 3) return;
      expect(Array.from(values(s) as Float32Array), s.name).toEqual(w.values);
    });

    const msg = decode(buf);
    expect(msg.type === 'unknown' ? null : msg.type).toBe(want.type_name);
    if (msg.type === 'scene_update') {
      for (const [k, a] of Object.entries(msg.arrays)) {
        const s = f.sections.find((x) => x.name === k)!;
        expect([a.dtype, a.shape]).toEqual([s.dtype, s.shape]);
        expect(Array.from(a.data)).toEqual(Array.from(values(s) as Float32Array));
        expect(a.data.buffer).toBe(buf.buffer); // a view, not a copy
        expect(a.data.constructor).toBe({ 0: Float32Array, 4: Int32Array, 5: Uint8Array, 6: Uint16Array }[a.dtype]);
      }
      checkStamp(f, msg.stamp, 'stamp');
      expect(msg.cursor).toBe(want.meta.cursor ?? undefined);
      expect([msg.sub, msg.view, msg.seq, msg.stamp_ns, msg.path, msg.part]).toEqual([want.meta.sub, want.meta.view, want.meta.seq, want.meta.stamp_ns, want.meta.path, want.meta.part]);
      expect(msg.static ?? false).toBe(want.meta.static ?? false);
      expect(msg.state ?? false).toBe(want.meta.state ?? false);
      if (want.meta.stats) expect(msg.params.stats).toEqual(want.meta.stats.map((x: number | null) => (x === null ? NaN : x)));
      if (want.meta.mode) expect(msg.params.mode).toBe(want.meta.mode);
    }
    if (msg.type === 'scene_clear') {
      expect([msg.path, msg.recursive ?? false, msg.reset ?? false, msg.parts]).toEqual([want.meta.path, want.meta.recursive ?? false, want.meta.reset ?? false, want.meta.parts]);
      checkStamp(f, msg.stamp, 'stamp');
      expect(msg.cursor).toBe(want.meta.cursor ?? undefined);
    }
    if (msg.type === 'scene_transforms') {
      expect([msg.sub, msg.view, msg.seq, msg.stamp_ns, msg.paths, msg.static ?? false, msg.state ?? false, msg.cursor])
        .toEqual([want.meta.sub, want.meta.view, want.meta.seq, want.meta.stamp_ns, want.meta.paths, want.meta.static ?? false, want.meta.state ?? false, want.meta.cursor]);
      const m = f.sections.find((x) => x.name === 'matrices')!;
      expect(Array.from(msg.matrices)).toEqual(Array.from(values(m) as Float32Array));
      expect(msg.matrices.buffer).toBe(buf.buffer);
      checkStamp(f, msg.stamp, 'stamp');
      checkStamp(f, msg.stamps, 'stamps');
      expect(msg.stamps_t).toBe(f.sections.find((x) => x.name === 'stamps')?.shape[1]);
    }
    if (msg.type === 'scene_classes') {
      expect([msg.sub, msg.view, msg.seq, msg.stamp_ns, msg.classes, msg.static ?? false, msg.state ?? false, msg.cursor])
        .toEqual([want.meta.sub, want.meta.view, want.meta.seq, want.meta.stamp_ns, want.meta.classes, want.meta.static ?? false, want.meta.state ?? false, want.meta.cursor]);
      checkStamp(f, msg.stamp, 'stamp');
    }
    if (msg.type === 'snapshot' || msg.type === 'series' || msg.type === 'text' || msg.type === 'controls') {
      checkStamp(f, msg.stamp, 'stamp');
      if (msg.type !== 'snapshot' && msg.type !== 'controls') {
        checkStamp(f, msg.stamps, 'stamps');
        expect(msg.stamps_t).toBe(f.sections.find((x) => x.name === 'stamps')?.shape[1]);
      }
      if (msg.type !== 'controls') expect(msg.cursor).toBe(want.meta.cursor ?? undefined);
      if (msg.type === 'series') checkStamp(f, msg.indices, 'indices');
      if (msg.type === 'snapshot') {
        expect(msg.static ?? false).toBe(want.meta.static ?? false);
        expect(msg.empty ?? false).toBe(want.meta.empty ?? false);
        expect(msg.labels).toEqual(want.meta.labels ?? undefined);
      }
    }
    if (msg.type === 'text') {
      expect([msg.sub, msg.view, msg.start, msg.total, msg.stamp_ns, msg.dropped]).toEqual([want.meta.sub, want.meta.view, want.meta.start, want.meta.total, want.meta.stamp_ns, want.meta.dropped ?? undefined]);
      expect(msg.text.buffer).toBe(buf.buffer);
      expect(msg.offsets.buffer).toBe(buf.buffer);
    }
    if (msg.type === 'cursor') expect(msg).toEqual({ type: 'cursor', ...want.meta, sections: undefined });
    if (msg.type === 'time_state') expect(msg).toEqual({ type: 'time_state', ...want.meta, cursor: want.meta.cursor ?? null });
    if (msg.type === 'recordings') expect(msg.recordings).toEqual(want.meta.recordings);
    if (msg.type === 'recording_open') expect(msg.recording).toBe(want.meta.recording);
    if (msg.type === 'scene_update' && want.meta.split) expect(msg.split).toEqual(want.meta.split);
    expect(typeName(f.type)).toBe(want.type_name);
    if (msg.type === 'unknown' || DECODE_ONLY.includes(name)) return;

    const again = encode(msg, f.minor);
    const g = decodeFrame(again);
    expect(g.type).toBe(f.type);
    expect(g.minor).toBe(f.minor);
    expect(g.meta).toEqual(f.meta);
    expect(g.sections.map((s) => [s.name, Array.from(s.bytes)])).toEqual(f.sections.map((s) => [s.name, Array.from(s.bytes)]));
    const text = new TextDecoder().decode(buf.subarray(24 + 4 * f.sections.length, 24 + 4 * f.sections.length + want.header.meta_len));
    if (text === canonicalJson(f.meta)) {
      expect(Array.from(again)).toEqual(Array.from(buf));
    } else {
      // Only a float written with a trailing .0 by Python can differ.
      expect(text.replace(/(\d)\.0(?![0-9])/g, '$1')).toBe(canonicalJson(f.meta));
    }
  });
}

test('some 3.2 frames re-encode byte for byte', () => {
  // Counted here: the tests above run in several workers, which share no state.
  let byteEqual = 0;
  for (const name of names) {
    if (DECODE_ONLY.includes(name)) continue;
    const buf = read(`${name}.bin`);
    const f = decodeFrame(buf);
    const msg = decode(buf);
    if (msg.type === 'unknown') continue;
    if (Array.from(encode(msg, f.minor)).join() === Array.from(buf).join()) byteEqual++;
  }
  expect(byteEqual).toBeGreaterThanOrEqual(20);
});

test('refuses every invalid 3.2 frame as expect.json says', () => {
  const expected: Record<string, { reason?: string; close_code?: number; error?: string }> = JSON.parse(fs.readFileSync(path.join(dir, 'invalid', 'expect.json'), 'utf8'));
  expect(Object.keys(expected).length).toBeGreaterThanOrEqual(96);
  for (const [file, want] of Object.entries(expected)) {
    const buf = read('invalid', file);
    if (want.error === undefined) {
      const err = catchError(() => decodeFrame(buf));
      expect(err, file).toBeInstanceOf(WireError);
      expect({ file, reason: (err as WireError).reason, close: (err as WireError).closeCode }).toEqual({ file, reason: want.reason, close: want.close_code });
    } else {
      expect(() => decodeFrame(buf), file).not.toThrow();
      const err = catchError(() => decode(buf));
      expect(err, file).toBeInstanceOf(MessageError);
      expect({ file, code: (err as MessageError).code }).toEqual({ file, code: want.error });
    }
  }
});

test('refuses every truncation of scene_update_mesh', () => {
  const buf = read('scene_update_mesh.bin');
  for (let n = 0; n < buf.length; n++) {
    const err = catchError(() => decodeFrame(buf.subarray(0, n)));
    expect(err, `${n} bytes`).toBeInstanceOf(WireError);
  }
});

type Typed = Float32Array | Int32Array | Uint8Array | Uint16Array | BigInt64Array;
const sec = (name: string, dtype: number, shape: number[], data: Typed): Section =>
  ({ name, dtype, shape, encoding: 0, bytes: new Uint8Array(data.buffer, data.byteOffset, data.byteLength) });
const base = { sub: 1, view: 'w', seq: 1, stamp_ns: 1, path: 'a' };

test('refuses a batch over a limit', () => {
  const points = encodeFrame(MSG.scene_update, { ...base, part: 'points' },
    [sec('positions', 0, [LIMITS_32.maxBatchPoints + 1, 2], new Float32Array((LIMITS_32.maxBatchPoints + 1) * 2))], PROTOCOL_MINOR_3_2);
  expect((catchError(() => decode(points)) as MessageError).code).toBe('bad_request');

  const n = LIMITS_32.maxBatchItems + 1;
  const strips = encodeFrame(MSG.scene_update, { ...base, part: 'lines', mode: 'strips' },
    [sec('positions', 0, [n, 2], new Float32Array(n * 2)), sec('strip_lengths', 4, [n], new Int32Array(n).fill(1))], PROTOCOL_MINOR_3_2);
  expect((catchError(() => decode(strips)) as MessageError).message).toContain('strips, more than');

  const t = LIMITS_32.maxBatchPoints + 1;
  const mesh = encodeFrame(MSG.scene_update, { ...base, part: 'mesh' },
    [sec('positions', 0, [3, 3], new Float32Array(9)), sec('indices', 4, [t, 3], new Int32Array(t * 3))], PROTOCOL_MINOR_3_2);
  expect((catchError(() => decode(mesh)) as MessageError).message).toContain('triangles, more than');

  const ok = encodeFrame(MSG.scene_update, { ...base, part: 'points' },
    [sec('positions', 0, [LIMITS_32.maxBatchPoints, 2], new Float32Array(LIMITS_32.maxBatchPoints * 2))], PROTOCOL_MINOR_3_2);
  expect((decode(ok) as SceneUpdateMsg).arrays.positions.shape).toEqual([LIMITS_32.maxBatchPoints, 2]);
});

test('entity paths', () => {
  for (const p of ['', 'a', 'world/cloud', 'a.b/c-d_e', 'x'.repeat(63), Array(16).fill('a').join('/')]) expect(entityPathOk(p), p).toBe(true);
  for (const p of ['/', 'a/', '/a', 'a//b', '.', '..', 'a/../b', 'x'.repeat(64), Array(17).fill('a').join('/'), 'a b', 'é', Array(52).fill('abcde').join('/'), 5, null])
    expect(entityPathOk(p), String(p)).toBe(false);
});

test('scene view info and the scene data view', () => {
  const entry = (scene?: unknown): ViewEntry => ({ id: 1, name: 's', kind: KIND_SCENE, sources: [], ...(scene === undefined ? {} : { scene }) });
  expect(sceneViewInfo(entry())).toEqual({ space: '3d', up: '+y', units: 'm' });
  expect(sceneViewInfo(entry({}))).toEqual({ space: '3d', up: '+y', units: 'm' });
  expect(sceneViewInfo(entry({ space: '2d' }))).toEqual({ space: '2d', up: '-y', units: 'm' });
  expect(sceneViewInfo(entry({ space: '2d', up: '+y', units: '' }))).toEqual({ space: '2d', up: '+y', units: '' });
  expect(sceneViewInfo(entry({ up: '+z', units: 'x'.repeat(15) }))).toEqual({ space: '3d', up: '+z', units: 'x'.repeat(15) });
  for (const bad of [[], 'x', { space: '4d' }, { up: 'y' }, { space: '2d', up: '+z' }, { units: 'x'.repeat(16) }, { units: 3 }]) {
    const err = catchError(() => sceneViewInfo(entry(bad)));
    expect(err, JSON.stringify(bad)).toBeInstanceOf(MessageError);
    expect((err as MessageError).code).toBe('bad_request');
  }
  const view = entry({});
  expect(resolveSceneDataView(view, undefined)).toEqual(SCENE_DATA_VIEW);
  expect(resolveSceneDataView(view, {})).toEqual(SCENE_DATA_VIEW);
  expect(resolveSceneDataView(view, JSON.parse(JSON.stringify(SCENE_DATA_VIEW)))).toEqual(SCENE_DATA_VIEW);
  expect(resolveSceneDataView({ ...view, default: {} }, null)).toEqual(SCENE_DATA_VIEW);
  expect(resolveSceneDataView(view, undefined)).not.toBe(SCENE_DATA_VIEW);
  for (const [v, r] of [[view, { budget: { max_points: 5 } }], [view, { entities: { include: ['a'], exclude: [] } }],
    [{ ...view, default: { budget: { max_points: 5 } } }, undefined], [view, [] as unknown as object]] as [ViewEntry, object | undefined][]) {
    const err = catchError(() => resolveSceneDataView(v, r));
    expect(err, JSON.stringify(r)).toBeInstanceOf(MessageError);
    expect((err as MessageError).code).toBe('bad_data_view');
  }
  expect(Object.keys(SCENE_DATA_VIEW.budget).sort()).toEqual(['max_extent', 'max_items', 'max_points', 'max_vertices']);
});

test('a null key of a scene means the same as an absent one, as in the golden VIEWS', () => {
  const v = decode(read('views_scene_nulls.bin'));
  expect(v.type).toBe('views');
  const entry = (v as { views: ViewEntry[] }).views[0];
  expect(entry.scene).toEqual({ space: null, up: null, units: null });
  expect(sceneViewInfo(entry)).toEqual({ space: '3d', up: '+y', units: 'm' });
});

test('the 3.2 constants agree with include/caliper/live_net.h', () => {
  const text = fs.readFileSync(path.join(REPO, 'include', 'caliper', 'live_net.h'), 'utf8');
  const block = text.slice(text.indexOf('PROTOCOL 3.2 DRAFT: THE SCENES SUBSET'));
  expect(block.length).toBeGreaterThan(100);
  const defs: Record<string, string | number> = {};
  // A define whose value is a parenthesised expression (STAMP_ABSENT) does not match and is checked below.
  for (const m of block.matchAll(/^#define\s+(CALIPER_NET_\w+)\s+("[^"]*"|0x[0-9a-fA-F]+u?|\d+(?:\.\d+)?u?)/gm)) {
    const v = m[2];
    defs[m[1]] = v.startsWith('"') ? v.slice(1, -1) : Number(v.replace(/u$/, ''));
  }
  expect(defs).toMatchObject({
    CALIPER_NET_MINOR_3_2: PROTOCOL_MINOR_3_2,
    CALIPER_NET_CAP_SCENES: CAP_SCENES,
    CALIPER_NET_KIND_SCENE: KIND_SCENE,
    CALIPER_NET_DTYPE_U16: DTYPE.u16,
    CALIPER_NET_MSG_SCENE_UPDATE: MSG.scene_update,
    CALIPER_NET_MSG_SCENE_TRANSFORMS: MSG.scene_transforms,
    CALIPER_NET_MSG_SCENE_CLEAR: MSG.scene_clear,
    CALIPER_NET_MSG_SCENE_CLASSES: MSG.scene_classes,
    CALIPER_NET_PART_POINTS: PART_POINTS,
    CALIPER_NET_PART_LINES: PART_LINES,
    CALIPER_NET_PART_ARROWS: PART_ARROWS,
    CALIPER_NET_PART_BOXES: PART_BOXES,
    CALIPER_NET_PART_MESH: PART_MESH,
    CALIPER_NET_PART_IMAGE: PART_IMAGE,
    CALIPER_NET_PART_CAMERA: PART_CAMERA,
    CALIPER_NET_PART_TRANSFORM: PART_TRANSFORM,
    CALIPER_NET_LINES_SEGMENTS: LINES_SEGMENTS,
    CALIPER_NET_LINES_STRIPS: LINES_STRIPS,
    CALIPER_NET_LINES_EDGES: LINES_EDGES,
    CALIPER_NET_CAP_PROGRAMS: CAP_PROGRAMS,
    CALIPER_NET_CAP_INTERVENTIONS: CAP_INTERVENTIONS,
    CALIPER_NET_MSG_PROGRAM: MSG.program,
    CALIPER_NET_MSG_PROGRAM_STATE: MSG.program_state,
    CALIPER_NET_PROGRAM_SET: PROGRAM_SET,
    CALIPER_NET_PROGRAM_REMOVE: PROGRAM_REMOVE,
    CALIPER_NET_OP_INTERVENE: 'intervene',
    CALIPER_NET_INTERVENE_ABLATE: INTERVENE_ABLATE,
    CALIPER_NET_MAX_PROGRAMS: MAX_PROGRAMS,
    CALIPER_NET_PROGRAM_ID_MAX: PROGRAM_ID_MAX,
    CALIPER_NET_PROGRAM_SOURCE_MAX: PROGRAM_SOURCE_MAX,
    CALIPER_NET_MSG_RESOURCE: MSG.resource,
    CALIPER_NET_RESOURCE_MAX_BYTES: RESOURCE_MAX_BYTES,
    CALIPER_NET_RESOURCE_VERSION_MAX_BYTES: RESOURCE_VERSION_MAX_BYTES,
    CALIPER_NET_RESOURCE_PART_WAIT: RESOURCE_PART_WAIT,
    CALIPER_NET_RESOURCE_STRIDE_MAX: RESOURCE_STRIDE_MAX,
    CALIPER_NET_RESOURCE_SIDE_MAX: RESOURCE_SIDE_MAX,
    CALIPER_NET_PIECE_NAME_MAX: PIECE_NAME_MAX,
    CALIPER_NET_PIECE_VERSION_MAX: PIECE_VERSION_MAX,
    CALIPER_NET_PIECE_COMMON_MAX: PIECE_COMMON_MAX,
    CALIPER_NET_PIECE_SEGMENTS_MAX: PIECE_SEGMENTS_MAX,
    CALIPER_NET_STATUS_PIECE: STATUS_PIECE,
    CALIPER_NET_PROGRAM_STATE_RATE_HZ: PROGRAM_STATE_RATE_HZ,
    CALIPER_NET_POINT_MAX: POINT_MAX,
    CALIPER_NET_MAX_POINTS_LISTED: MAX_POINTS_LISTED,
    CALIPER_NET_MAX_INTERVENTIONS: MAX_INTERVENTIONS,
    CALIPER_NET_PROGRAM_MAX_INPUTS: PROGRAM_MAX_INPUTS,
    CALIPER_NET_PROGRAM_MAX_COUNT: PROGRAM_MAX_COUNT,
    CALIPER_NET_PROGRAM_SIDE_MAX: PROGRAM_SIDE_MAX,
    CALIPER_NET_PROGRAM_TIME_MS_MAX: PROGRAM_TIME_MS_MAX,
    CALIPER_NET_PROGRAM_MEMORY_MB_MAX: PROGRAM_MEMORY_MB_MAX,
    CALIPER_NET_PROGRAM_EVERY_MAX: PROGRAM_EVERY_MAX,
    CALIPER_NET_PROGRAM_WORKGROUPS_MAX: PROGRAM_WORKGROUPS_MAX,
    CALIPER_NET_TENSOR_NAME_MAX: TENSOR_NAME_MAX,
    CALIPER_NET_COLOR_PER_VERTEX: COLOR_PER_VERTEX,
    CALIPER_NET_COLOR_PER_STRIP: COLOR_PER_STRIP,
    CALIPER_NET_COLOR_PER_EDGE: COLOR_PER_EDGE,
    CALIPER_NET_FORM_COLOR: FORM_COLOR,
    CALIPER_NET_FORM_DEPTH: FORM_DEPTH,
    CALIPER_NET_FORM_SEGMENTATION: FORM_SEGMENTATION,
    CALIPER_NET_FORM_ENCODED: FORM_ENCODED,
    CALIPER_NET_FORMAT_JPEG: FORMAT_JPEG,
    CALIPER_NET_FORMAT_PNG: FORMAT_PNG,
    CALIPER_NET_CONVENTION_RDF: CONVENTION_RDF,
    CALIPER_NET_CONVENTION_RUB: CONVENTION_RUB,
    CALIPER_NET_PATH_MAX: LIMITS_32.pathMax,
    CALIPER_NET_PATH_SEGMENTS_MAX: LIMITS_32.pathSegmentsMax,
    CALIPER_NET_UNITS_MAX: LIMITS_32.unitsMax,
    CALIPER_NET_MAX_ENTITIES: LIMITS_32.maxEntities,
    CALIPER_NET_MAX_BATCHES: LIMITS_32.maxBatches,
    CALIPER_NET_MAX_BATCH_POINTS: LIMITS_32.maxBatchPoints,
    CALIPER_NET_MAX_BATCH_ITEMS: LIMITS_32.maxBatchItems,
    CALIPER_NET_MAX_LABELS: MAX_LABELS,
    CALIPER_NET_MAX_CLASSES: MAX_CLASSES,
    CALIPER_NET_MAX_CLASS_ID: MAX_CLASS_ID,
    CALIPER_NET_MAX_IMAGE_SIDE: MAX_IMAGE_SIDE,
    CALIPER_NET_MAX_ENCODED_BYTES: MAX_ENCODED_BYTES,
    CALIPER_NET_MAX_TRANSFORMS: MAX_TRANSFORMS,
    CALIPER_NET_MAX_TIMELINES: MAX_TIMELINES,
    CALIPER_NET_CAP_TIMELINES: CAP_TIMELINES,
    CALIPER_NET_CAP_TEXT: CAP_TEXT,
    CALIPER_NET_CAP_BARS: CAP_BARS,
    CALIPER_NET_CAP_RECORDINGS: CAP_RECORDINGS,
    CALIPER_NET_KIND_TEXT: KIND_IDS.text,
    CALIPER_NET_KIND_BARS: KIND_IDS.bars,
    CALIPER_NET_MSG_TEXT: MSG.text,
    CALIPER_NET_MSG_RECORDINGS: MSG.recordings,
    CALIPER_NET_MSG_RECORDING_OPEN: MSG.recording_open,
    CALIPER_NET_MSG_CURSOR: MSG.cursor,
    CALIPER_NET_MSG_TIME_STATE: MSG.time_state,
    CALIPER_NET_ERR_UNKNOWN_RECORDING: ERR_UNKNOWN_RECORDING,
    CALIPER_NET_ERR_BAD_RECORDING: ERR_BAD_RECORDING,
    CALIPER_NET_ERR_NOT_SEEKABLE: ERR_NOT_SEEKABLE,
    CALIPER_NET_ERR_BAD_CURSOR: ERR_BAD_CURSOR,
    CALIPER_NET_TIMELINE_SEQUENCE: TIMELINE_SEQUENCE,
    CALIPER_NET_TIMELINE_TIME: TIMELINE_TIME,
    CALIPER_NET_LEVEL_DEBUG: LEVEL_DEBUG,
    CALIPER_NET_LEVEL_INFO: LEVEL_INFO,
    CALIPER_NET_LEVEL_WARNING: LEVEL_WARNING,
    CALIPER_NET_LEVEL_ERROR: LEVEL_ERROR,
    CALIPER_NET_PACE_RECORDED: PACE_RECORDED,
    CALIPER_NET_PACE_EVEN: PACE_EVEN,
    CALIPER_NET_TIME_FOLLOWING: TIME_FOLLOWING,
    CALIPER_NET_TIME_PAUSED: TIME_PAUSED,
    CALIPER_NET_TIME_PLAYING: TIME_PLAYING,
    CALIPER_NET_STAMP_TEXT_MAX: STAMP_TEXT_MAX,
    CALIPER_NET_MAX_PIECES: MAX_PIECES,
    CALIPER_NET_MAX_UPDATE_BYTES: MAX_UPDATE_BYTES,
    CALIPER_NET_MAX_INCOMPLETE_UPDATES: MAX_INCOMPLETE_UPDATES,
    CALIPER_NET_TEXT_LINE_MAX: TEXT_LINE_MAX,
    CALIPER_NET_TEXT_MAX_LINES: TEXT_MAX_LINES,
    CALIPER_NET_TEXT_RETAIN_MIN: TEXT_RETAIN_MIN,
    CALIPER_NET_MAX_BARS: MAX_BARS,
    CALIPER_NET_MAX_RECORDINGS: MAX_RECORDINGS,
    CALIPER_NET_MAX_RECORDED: MAX_RECORDED,
    CALIPER_NET_MAX_CURSOR_STEP: MAX_CURSOR_STEP,
    CALIPER_NET_CURSOR_RATE_HZ: CURSOR_RATE_HZ,
    CALIPER_NET_TIME_STATE_RATE_HZ: TIME_STATE_RATE_HZ,
    CALIPER_NET_MIN_SPEED: MIN_SPEED,
    CALIPER_NET_MAX_SPEED: MAX_SPEED,
    CALIPER_NET_RECORDING_LINGER_MS: RECORDING_LINGER_MS,
    CALIPER_NET_SCENE_DEFAULT_MAX_ITEMS: SCENE_DATA_VIEW.budget.max_items,
    CALIPER_NET_SCENE_DEFAULT_MAX_VERTICES: SCENE_DATA_VIEW.budget.max_vertices,
    CALIPER_NET_SCENE_DEFAULT_MAX_EXTENT: SCENE_DATA_VIEW.budget.max_extent,
    CALIPER_NET_SCENE_LIMIT_MAX_ITEMS: 1048576,
    CALIPER_NET_SCENE_LIMIT_MAX_VERTICES: 4194304,
    CALIPER_NET_SCENE_LIMIT_MAX_EXTENT: 8192,
  });
  expect(block).toMatch(/#define\s+CALIPER_NET_STAMP_ABSENT\s+\(-0x7fffffffffffffffLL - 1\)/);
  expect(STAMP_ABSENT).toBe(-(2n ** 63n));
  expect(DTYPE_SIZE[DTYPE.u16]).toBe(2);
  expect(SCENE_PARTS).toEqual([PART_POINTS, PART_LINES, PART_ARROWS, PART_BOXES, PART_MESH, PART_IMAGE, PART_CAMERA]);
  expect(CLEAR_PARTS).toEqual([...SCENE_PARTS, PART_TRANSFORM]);
  expect(CAMERA_REQUIRED).toEqual(['fx', 'fy', 'cx', 'cy', 'width', 'height', 'convention']);
  expect(Object.keys(PART_SECTIONS).sort()).toEqual([...SCENE_PARTS].sort());
  expect(Object.keys(PART_KEYS).sort()).toEqual([...SCENE_PARTS].sort());
});

// --- the rules and golden values of the other parts, mirroring python/tests/test_wire_32.py ---

const frame32 = (name: string) => decode(read(`${name}.bin`));
const upd = (name: string) => frame32(name) as SceneUpdateMsg;
const nums = (a: ArrayLike<number | bigint>) => Array.from(a, Number);

test('golden values of the other parts', () => {
  const st = upd('scene_update_points_stamped');
  expect(Array.from(st.stamp!)).toEqual([1200n, 1763200000123456789n]);
  expect(upd('scene_update_points').stamp).toBeUndefined();
  const lab = upd('scene_update_points_labels');
  expect(lab.params.labels).toEqual(['a', 'b', 'c']);
  expect(nums(lab.arrays.class_ids.data)).toEqual([0, 1, 1]);
  const e = upd('scene_update_lines_edges');
  expect([e.params.mode, e.params.color_per]).toEqual(['edges', 'edge']);
  expect(nums(e.arrays.indices.data)).toEqual([0, 1, 1, 2, 2, 0]);
  expect(e.arrays.indices.shape).toEqual([3, 2]);
  expect(e.arrays.values.shape).toEqual([3]);
  expect(nums(upd('scene_update_arrows').arrays.vectors.data)).toEqual([0, 0, 1, 0, 0.5, 0]);
  const b = upd('scene_update_boxes');
  expect([b.static, b.stamp]).toEqual([true, undefined]);
  expect(nums(b.arrays.rotations.data)).toEqual([0, 0, 0, 1]);
  const b2 = upd('scene_update_boxes_2d');
  expect(b2.params).toEqual({ labels: ['car'], fill: true });
  expect(b2.arrays.half_sizes.shape).toEqual([2, 2]);
  const t = upd('scene_update_mesh_textured');
  expect([t.arrays.texture.shape, t.arrays.uvs.shape]).toEqual([[2, 2, 4], [4, 2]]);
  expect(Object.keys(upd('scene_update_mesh_normals').arrays).sort()).toEqual(['indices', 'normals', 'positions']);
  const dep = upd('scene_update_image_depth');
  expect(dep.arrays.pixels.dtype).toBe(DTYPE.u16);
  expect(dep.arrays.pixels.data).toBeInstanceOf(Uint16Array);
  expect(nums(dep.arrays.pixels.data)).toEqual([500, 1000, 0, 1500, 2000, 2500]);
  expect([dep.params.form, dep.params.scale, dep.params.cloud]).toEqual(['depth', 0.001, true]);
  const enc = upd('scene_update_image_encoded');
  expect(nums(enc.arrays.pixels.data).slice(0, 8)).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  expect(enc.params.format).toBe('png');
  expect(nums(upd('scene_update_image_segmentation_u16').arrays.pixels.data)).toEqual([0, 300, 65535, 7]);
  const cam = upd('scene_update_camera');
  expect(cam.arrays).toEqual({});
  expect(cam.params).toEqual({ fx: 525, fy: 525, cx: 319.5, cy: 239.5, width: 640, height: 480, convention: 'rdf' });
  expect(upd('scene_update_camera_rub').params.plane_distance).toBe(0.25);
  const tr = frame32('scene_transforms') as SceneTransformsMsg;
  expect(tr.type).toBe('scene_transforms');
  expect(tr.paths).toEqual(['arm/link0', 'arm/link0/link1']);
  expect(tr.matrices.length).toBe(32);
  expect(nums(tr.matrices.subarray(16 + 12, 16 + 15))).toEqual([0, 0, 0.5]); // column-major: the translation is the last column
  expect(Array.from(tr.stamp!)).toEqual([1200n, 1763200000123456789n]);
  expect(tr.stamps).toBeUndefined();
  const ts = frame32('scene_transforms_stamps') as SceneTransformsMsg;
  expect(Array.from(ts.stamps!)).toEqual([1200n, 1763200000123456789n, 1201n, 1763200000128456789n]);
  expect(ts.stamps_t).toBe(2);
  const stat = frame32('scene_transforms_static') as SceneTransformsMsg;
  expect([stat.static, stat.state]).toEqual([true, true]);
  const cl = frame32('scene_classes') as SceneClassesMsg;
  expect(cl.type).toBe('scene_classes');
  expect(cl.classes.map((c) => c.id)).toEqual([0, 1, 65535]);
  expect((frame32('scene_classes_stamped') as SceneClassesMsg).classes).toEqual([]);
  const c = frame32('scene_clear_stamped');
  expect(c.type === 'scene_clear' && c.parts).toEqual(['transform', 'boxes']);
  expect(c.type === 'scene_clear' && c.stamp instanceof BigInt64Array).toBe(true);
});

const update = (part: string, sections: Record<string, Typed | [number, number[], Typed]>, keys: Record<string, unknown> = {}) => {
  const secs = Object.entries(sections).map(([n, v]) => {
    if (Array.isArray(v)) return sec(n, v[0], v[1], v[2]);
    return sec(n, v instanceof Float32Array ? DTYPE.f32 : v instanceof Int32Array ? DTYPE.i32 : v instanceof Uint8Array ? DTYPE.u8
      : v instanceof Uint16Array ? DTYPE.u16 : DTYPE.i64, [v.length], v);
  });
  return encodeFrame(MSG.scene_update, { part, path: 'p', seq: 1, stamp_ns: 1, sub: 2, view: 's', ...keys }, secs, PROTOCOL_MINOR_3_2);
};
const f32s = (shape: number[]): [number, number[], Float32Array] => [DTYPE.f32, shape, new Float32Array(shape.reduce((a, b) => a * b, 1))];
const i32s = (shape: number[], fill = 0): [number, number[], Int32Array] => [DTYPE.i32, shape, new Int32Array(shape.reduce((a, b) => a * b, 1)).fill(fill)];
const u8s = (shape: number[]): [number, number[], Uint8Array] => [DTYPE.u8, shape, new Uint8Array(shape.reduce((a, b) => a * b, 1))];
const u16s = (shape: number[]): [number, number[], Uint16Array] => [DTYPE.u16, shape, new Uint16Array(shape.reduce((a, b) => a * b, 1))];
const i64s = (vals: bigint[], shape = [vals.length]): [number, number[], BigInt64Array] => [DTYPE.i64, shape, BigInt64Array.from(vals)];
const refused = (buf: Uint8Array, code = 'bad_request', match?: string) => {
  const err = catchError(() => decode(buf));
  expect(err).toBeInstanceOf(MessageError);
  expect((err as MessageError).code, (err as Error).message).toBe(code);
  if (match) expect((err as Error).message).toContain(match);
};
const upd2 = (b: Uint8Array) => decode(b) as SceneUpdateMsg;

test('message rules', () => {
  const p3 = f32s([4, 3]);
  const rgba = u8s([4, 4]);
  refused(update('points', { positions: p3, values: f32s([3]) }));
  refused(update('points', { positions: p3, radii: f32s([4, 1]) }));
  refused(update('points', { positions: p3 }, { color: [1, 2, 3] }));
  refused(update('points', { positions: p3 }, { radius: 0 }));
  refused(update('lines', { positions: p3 }, { mode: 'segments', width: -1 }));
  refused(update('lines', { positions: p3 }, { mode: 'segments', color_per: 'strip' }));
  refused(update('lines', { positions: p3, strip_lengths: i32s([1], 4) }, { mode: 'segments' }));
  refused(update('lines', { positions: p3, strip_lengths: [DTYPE.i32, [2], new Int32Array([2, 2])], colors: rgba }, { mode: 'strips', color_per: 'strip' }));
  refused(update('lines', { positions: p3, strip_lengths: [DTYPE.i32, [2], new Int32Array([4, 0])] }, { mode: 'strips' }));
  refused(update('lines', { positions: p3, strip_lengths: i32s([1], 4) }, { mode: 'strips', color_per: 'edge' }));
  refused(update('lines', { positions: p3, indices: i32s([1, 2]) }, { mode: 'strips' }));
  refused(update('lines', { positions: p3 }, { mode: 'loops' }));
  refused(update('lines', { positions: p3 }, {}));
  refused(update('mesh', { positions: p3 }));
  refused(update('mesh', { positions: f32s([3, 3]), indices: i32s([2, 2]) }));
  refused(update('points', { positions: [DTYPE.f16, [4, 3], new Uint16Array(12)] }));
  refused(update('points', { positions: p3 }, { labels: [] }));
  refused(update('points', { positions: p3 }, { labels: 'a' }));
  refused(update('points', { positions: p3 }, { labels: ['a', 'b'] }));
  refused(update('boxes', { centers: p3, half_sizes: p3 }, { fill: 1 }));
  refused(update('image', { pixels: u8s([2, 2]) }, { form: 'color', opacity: -0.5 }));
  refused(update('image', { pixels: u8s([2, 2]) }, { form: 'encoded', format: 'gif' }));
  refused(update('image', { pixels: f32s([2, 2]) }, { form: 'segmentation' }));
  refused(update('image', { pixels: u8s([4]) }, { form: 'color' }));
  refused(update('image', { pixels: u16s([2, 2]) }, { form: 'color' }));
  refused(update('image', { pixels: u8s([2]) }, { form: 'depth', scale: 0 }));
  refused(update('image', { pixels: u8s([2]) }, { form: 'encoded' }));
  const cam = { fx: 1, fy: 1, cx: 0, cy: 0, width: 2, height: 2, convention: 'rdf' };
  refused(update('camera', {}, { ...cam, width: 8193 }));
  refused(update('camera', {}, { ...cam, convention: 'flu' }));
  for (const k of CAMERA_REQUIRED) refused(update('camera', {}, { ...cam, [k]: undefined }));
  refused(update('arrows', { origins: p3 }));
  refused(update('arrows', { origins: p3, vectors: f32s([4, 2]) }));
  refused(update('boxes', { centers: f32s([4, 2]), half_sizes: f32s([4, 2]), rotations: f32s([4, 4]) }));
  refused(update('mesh', { positions: f32s([3, 3]), uvs: f32s([3, 2]) }));
  refused(update('lines', { positions: p3 }, { mode: 'edges' }));
  // per-strip colours, values per vertex, per-edge and per-vertex colours of edges are fine
  let ok = upd2(update('lines', { positions: p3, strip_lengths: [DTYPE.i32, [2], new Int32Array([3, 1])], colors: u8s([2, 4]) }, { mode: 'strips', color_per: 'strip' }));
  expect(ok.arrays.colors.shape).toEqual([2, 4]);
  ok = upd2(update('lines', { positions: p3, values: f32s([4]) }, { mode: 'segments', vmin: 0, vmax: 1 }));
  expect(ok.params.vmin).toBe(0);
  const edges: [number, number[], Int32Array] = [DTYPE.i32, [3, 2], new Int32Array([0, 1, 2, 3, 3, 0])];
  ok = upd2(update('lines', { positions: p3, indices: edges, colors: rgba }, { mode: 'edges' }));
  expect([ok.arrays.colors.shape, ok.arrays.indices.shape]).toEqual([[4, 4], [3, 2]]);
  ok = upd2(update('lines', { positions: p3, indices: edges, colors: u8s([3, 4]) }, { mode: 'edges', color_per: 'edge' }));
  expect(ok.arrays.colors.shape).toEqual([3, 4]);
  // labels: one for all or one per item; a camera holds no array
  expect(upd2(update('points', { positions: p3 }, { labels: ['x'] })).params.labels).toEqual(['x']);
  expect(upd2(update('boxes', { centers: p3, half_sizes: p3 }, { labels: ['a', 'b', 'c', 'd'] })).params.labels?.length).toBe(4);
  const c = upd2(update('camera', { positions: p3 }, { ...cam, fx: 1.5, fy: 2, width: 2, height: 3, convention: 'rub' }));
  expect(c.arrays).toEqual({});
  expect(c.params.width).toBe(2);
  // unknown keys and sections are ignored
  const un = upd2(update('points', { positions: p3, normals: f32s([4, 3]) }, { future: 1, fill: true }));
  expect(Object.keys(un.arrays)).toEqual(['positions']);
  expect(un.params).toEqual({});
  // the colour pixels may be (H, W); an empty batch is well formed
  expect(upd2(update('image', { pixels: u8s([2, 3]) }, { form: 'color' })).arrays.pixels.shape).toEqual([2, 3]);
  expect(upd2(update('points', { positions: f32s([0, 3]) })).arrays.positions.shape).toEqual([0, 3]);
  // an unsupported split and an unknown part
  refused(update('points', { positions: p3 }, { split: { axis: 0 } }));
  refused(update('points', { positions: p3 }, { split: 3 }));
  refused(update('points', { positions: p3 }, { part: 'nope' }));
  refused(update('points', { positions: p3 }, { part: 'points', path: 'a//b' }));
  refused(update('points', { positions: p3 }, { cursor: -1 }));
  expect(upd2(update('points', { positions: p3 }, { cursor: 7 })).cursor).toBe(7);
});

test('stamps', () => {
  const p3 = f32s([2, 3]);
  const m = upd2(update('points', { positions: p3, stamp: i64s([5n, 7n]) }));
  expect(Array.from(m.stamp!)).toEqual([5n, 7n]);
  expect(m.stamp).toBeInstanceOf(BigInt64Array);
  expect(upd2(update('points', { positions: p3, stamp: i64s(Array(MAX_TIMELINES).fill(0n)) })).stamp!.length).toBe(MAX_TIMELINES);
  expect(upd2(update('points', { positions: p3, stamp: i64s([1n, STAMP_ABSENT]) })).stamp![1]).toBe(-(1n << 63n));
  for (const bad of [i64s([]), i64s(Array(9).fill(0n)), [DTYPE.i32, [2], new Int32Array(2)] as [number, number[], Int32Array]])
    refused(update('points', { positions: p3, stamp: bad }));
  refused(update('points', { positions: p3, stamp: i64s([5n, 7n]) }, { static: true }));
  // a scene update takes one stamp; "stamps" is not one of its sections
  const s = upd2(update('points', { positions: p3, stamps: i64s([0n, 0n, 0n, 0n], [2, 2]) }));
  expect(s.stamp).toBeUndefined();
  expect(Object.keys(s.arrays)).toEqual(['positions']);
  // the stamp is the last section written, and survives an encode
  const msg: SceneUpdateMsg = { type: 'scene_update', sub: 1, view: 's', seq: 1, stamp_ns: 1, path: 'p', part: 'points',
    arrays: { positions: { dtype: DTYPE.f32, shape: [2, 3], data: new Float32Array(6) } }, params: {}, stamp: BigInt64Array.from([5n, 7n]) };
  const f = decodeFrame(encode(msg, PROTOCOL_MINOR_3_2));
  expect(f.sections.map((x) => x.name)).toEqual(['positions', 'stamp']);
  expect(Array.from((decode(encode(msg, PROTOCOL_MINOR_3_2)) as SceneUpdateMsg).stamp!)).toEqual([5n, 7n]);
  // a stamp at an unaligned offset is copied
  const raw = encode(msg, PROTOCOL_MINOR_3_2);
  const shifted = new Uint8Array(raw.length + 4);
  shifted.set(raw, 4);
  expect(Array.from((decode(shifted.subarray(4)) as SceneUpdateMsg).stamp!)).toEqual([5n, 7n]);
});

test('transforms and classes', () => {
  const eye = new Float32Array(48);
  for (let k = 0; k < 3; k++) for (let i = 0; i < 4; i++) eye[16 * k + 5 * i] = 1;
  const msg: SceneTransformsMsg = { type: 'scene_transforms', sub: 1, view: 's', seq: 2, stamp_ns: 3, paths: ['a', 'a/b', 'c'], matrices: eye, cursor: 4 };
  const back = decode(encode(msg, PROTOCOL_MINOR_3_2)) as SceneTransformsMsg;
  expect([back.paths, back.cursor]).toEqual([['a', 'a/b', 'c'], 4]);
  expect(Array.from(back.matrices)).toEqual(Array.from(eye));
  // one stamp or one each
  const one = decode(encode({ ...msg, stamp: BigInt64Array.from([9n]) }, PROTOCOL_MINOR_3_2)) as SceneTransformsMsg;
  expect([Array.from(one.stamp!), one.stamps]).toEqual([[9n], undefined]);
  const each = decode(encode({ ...msg, stamps: BigInt64Array.from([1n, 2n, 3n]), stamps_t: 1 }, PROTOCOL_MINOR_3_2)) as SceneTransformsMsg;
  expect([Array.from(each.stamps!), each.stamps_t, each.stamp]).toEqual([[1n, 2n, 3n], 1, undefined]);
  // 'stamps' first when both are given, as Python writes
  expect(decodeFrame(encode({ ...msg, stamp: BigInt64Array.from([9n]), stamps: BigInt64Array.from([1n, 2n, 3n]), stamps_t: 1 }, PROTOCOL_MINOR_3_2))
    .sections.map((s) => s.name)).toEqual(['matrices', 'stamps']);

  const frame = (meta: Record<string, unknown>, ...secs: Section[]) => encodeFrame(MSG.scene_transforms, { sub: 1, view: 's', seq: 2, stamp_ns: 3, ...meta }, secs, PROTOCOL_MINOR_3_2);
  const mats = (k: number) => sec('matrices', DTYPE.f32, [k, 16], new Float32Array(16 * k));
  const names = (k: number) => Array.from({ length: k }, (_, i) => `e${i}`);
  refused(frame({ paths: ['a', 'a'] }, mats(2)));
  refused(frame({ paths: [''] }, mats(1)));
  refused(frame({ paths: [] }, mats(0)));
  refused(frame({ paths: ['a'] }));
  refused(frame({ paths: ['a'] }, mats(2)));
  refused(frame({ paths: ['a'], static: true }, mats(1), sec('stamp', DTYPE.i64, [1], new BigInt64Array(1))));
  refused(frame({ paths: ['a'] }, mats(1), sec('stamp', DTYPE.i64, [1], new BigInt64Array(1)), sec('stamps', DTYPE.i64, [1, 1], new BigInt64Array(1))));
  refused(frame({ paths: ['a', 'b'] }, mats(2), sec('stamps', DTYPE.i64, [1, 1], new BigInt64Array(1))));
  refused(frame({ paths: names(MAX_TRANSFORMS + 1) }, mats(MAX_TRANSFORMS + 1)));
  expect((decode(frame({ paths: names(MAX_TRANSFORMS) }, mats(MAX_TRANSFORMS))) as SceneTransformsMsg).paths.length).toBe(MAX_TRANSFORMS);

  const table = Array.from({ length: MAX_CLASSES }, (_, i) => ({ id: i, label: `c${i}`, color: [i % 256, 0, 0, 255] }));
  const classes = (list: unknown, ...secs: Section[]) => encodeFrame(MSG.scene_classes, { sub: 1, view: 's', seq: 2, stamp_ns: 3, classes: list }, secs, PROTOCOL_MINOR_3_2);
  expect((decode(classes(table)) as SceneClassesMsg).classes.length).toBe(MAX_CLASSES);
  const rt = decode(encode({ type: 'scene_classes', sub: 1, view: 's', seq: 2, stamp_ns: 3, classes: table.slice(0, 3), cursor: 2 }, PROTOCOL_MINOR_3_2)) as SceneClassesMsg;
  expect([rt.classes.length, rt.cursor]).toEqual([3, 2]);
  refused(classes([...table, { id: 65535, label: 'x', color: [0, 0, 0, 0] }]));
  refused(classes([{ id: 1, label: 'x'.repeat(64), color: [0, 0, 0, 0] }]));
  refused(classes([{ id: 65536, label: 'x', color: [0, 0, 0, 0] }]));
  refused(classes([{ id: 1, label: 'x', color: [0, 0, 0] }]));
  refused(classes([{ id: 2, label: 'x', color: [0, 0, 0, 0] }, { id: 2, label: 'y', color: [0, 0, 0, 0] }]));
  refused(classes([{ id: 2, label: 'x', color: [0, 0, 0, 0] }, { id: 1, label: 'y', color: [0, 0, 0, 0] }]));
  refused(classes('x'));
  refused(classes([], sec('stamp', DTYPE.i64, [0], new BigInt64Array(0))));
  refused(encodeFrame(MSG.scene_classes, { sub: 1, view: 's', seq: 2, stamp_ns: 3, classes: [], static: true }, [sec('stamp', DTYPE.i64, [1], new BigInt64Array(1))], PROTOCOL_MINOR_3_2));

  // scene_clear: stamp and cursor, and the transform part
  const clear = decode(encode({ type: 'scene_clear', sub: 1, view: 's', seq: 1, stamp_ns: 1, path: 'a', parts: ['transform', 'camera'], cursor: 5, stamp: BigInt64Array.from([1n]) }, PROTOCOL_MINOR_3_2));
  expect(clear.type === 'scene_clear' && [clear.parts, clear.cursor, Array.from(clear.stamp!)]).toEqual([['transform', 'camera'], 5, [1n]]);
  refused(encodeFrame(MSG.scene_clear, { sub: 1, view: 's', seq: 1, stamp_ns: 1, path: 'a', parts: ['arrows', 'nope'] }, [], PROTOCOL_MINOR_3_2));
});

test('limits of every part', () => {
  const big = f32s([LIMITS_32.maxBatchPoints + 1, 2]);
  refused(update('points', { positions: big }), 'bad_request', 'more than 4194304');
  refused(update('lines', { positions: f32s([2, 3]), indices: i32s([LIMITS_32.maxBatchPoints + 1, 2]) }, { mode: 'edges' }), 'bad_request', 'edges, more than');
  const items = f32s([LIMITS_32.maxBatchItems + 1, 2]);
  refused(update('arrows', { origins: items, vectors: items }), 'bad_request', 'arrows, more than');
  refused(update('boxes', { centers: items, half_sizes: items }), 'bad_request', 'boxes, more than');
  refused(update('points', { positions: f32s([2, 2]) }, { labels: Array(MAX_LABELS + 1).fill('l') }));
  expect(upd2(update('points', { positions: f32s([MAX_LABELS, 2]) }, { labels: Array(MAX_LABELS).fill('l') })).params.labels?.length).toBe(MAX_LABELS);
  expect(upd2(update('image', { pixels: u8s([1, MAX_IMAGE_SIDE]) }, { form: 'segmentation' })).arrays.pixels.shape).toEqual([1, MAX_IMAGE_SIDE]);
  refused(update('image', { pixels: u8s([1, MAX_IMAGE_SIDE + 1]) }, { form: 'segmentation' }));
  refused(update('image', { pixels: u8s([MAX_ENCODED_BYTES + 1]) }, { form: 'encoded', format: 'png' }), 'bad_request', 'encoded frame');
  expect(upd2(update('image', { pixels: u8s([MAX_ENCODED_BYTES]) }, { form: 'encoded', format: 'jpeg' })).arrays.pixels.shape).toEqual([MAX_ENCODED_BYTES]);
  refused(update('mesh', { positions: f32s([3, 2]), uvs: f32s([3, 2]), texture: u8s([1, MAX_IMAGE_SIDE + 1, 4]) }));
});

test('scene view info with a view camera', () => {
  const entry = (scene?: unknown): ViewEntry => ({ id: 1, name: 's', kind: KIND_SCENE, sources: [], ...(scene === undefined ? {} : { scene }) });
  const base = { space: '3d', up: '+y', units: 'm', view_camera: undefined };
  expect(sceneViewInfo(entry())).toEqual(base);
  expect(sceneViewInfo(entry({ view_camera: { azimuth: 30 } }))).toEqual({ ...base, view_camera: { azimuth: 30 } });
  expect(sceneViewInfo(entry({ view_camera: { azimuth: 30, distance: null } })).view_camera).toEqual({ azimuth: 30 });
  expect(sceneViewInfo(entry({ space: '2d' })).up).toBe('-y');
  for (const sc of [{ space: '4d' }, { space: '2d', up: '+z' }, { up: 'z' }, { units: 'x'.repeat(16) }, { units: 3 }, '3d'])
    expect(() => sceneViewInfo(entry(sc)), JSON.stringify(sc)).toThrow(MessageError);
  const good = [{}, { target: [0, 1, 2.5] }, { distance: 3 }, { elevation: -89 },
    { elevation: 89, azimuth: -720.5, target: [0, 0, 0], distance: 0.5 }, 'rig/left'];
  for (const vc of good) expect(sceneViewInfo(entry({ view_camera: vc })).view_camera, JSON.stringify(vc)).toEqual(vc);
  // a view_camera a viewer cannot read is ignored, never refused, and a 2-D scene takes none
  const unreadable = [{ target: [0, 1] }, { distance: 0 }, { elevation: 90 }, { azimuth: 'x' }, { fov: 45 }, '', 'a/../b', 3, [1, 2, 3],
    { target: [0, 0, NaN] }, { target: [0, 0, Infinity] }, true];
  for (const vc of unreadable) expect(sceneViewInfo(entry({ view_camera: vc })).view_camera, String(vc)).toBeUndefined();
  expect(sceneViewInfo(entry({ space: '2d', view_camera: {} })).view_camera).toBeUndefined();
  const views = frame32('views_scene_camera');
  const infos = (views.type === 'views' ? views.views : []).map((e) => sceneViewInfo(e));
  expect(infos[0].view_camera).toEqual({ target: [0, 0, 0.5], azimuth: 30, elevation: 25 });
  expect(infos[1]).toEqual({ space: '2d', up: '-y', units: 'px', view_camera: undefined });
});

// --- timelines, text logs, bar charts, recordings and split updates ---

test('golden values of timelines, text, bars and recordings', () => {
  const v = frame32('views_timelines') as ViewsMsg;
  expect(v.timelines!.map((t) => t.name)).toEqual(['step', 'time', 'episode']);
  expect(v.views[0].recorded).toEqual([{ budget: { max_backfill: 4096 } }]);
  expect((decode(new Uint8Array(fs.readFileSync(path.join(goldenFrames(), 'views.bin')))) as ViewsMsg).timelines).toBeUndefined();
  const se = frame32('sessions_recording');
  expect(se.type === 'sessions' && se.sessions[0].recording).toEqual({ id: 'train-20261115-120314', seekable: true, complete: true });
  const sr = frame32('series_stamps');
  expect(sr.type === 'series' && Array.from(sr.stamps!)).toEqual([1200n, 1763200000123456789n, 1201n, 1763200000128456789n, 1202n, 1763200000133456789n]);
  expect(sr.type === 'series' && sr.stamps_t).toBe(2);
  const s1 = frame32('series_stamp');
  expect(s1.type === 'series' && Array.from(s1.stamp!)).toEqual([1200n, 1763200000123456789n]);
  const dec = frame32('series_decimated');
  expect(dec.type === 'series' && [Array.from(dec.indices!), dec.decimated, dec.cursor]).toEqual([[0n, 2048n, 4095n], 1024, 2]);
  const sn = frame32('snapshot_stamped') as SnapshotMsg;
  expect([sn.stamp instanceof BigInt64Array, sn.static]).toEqual([true, undefined]);
  const st = frame32('snapshot_static') as SnapshotMsg;
  expect([st.static, st.stamp]).toEqual([true, undefined]);
  const em = frame32('snapshot_empty') as SnapshotMsg;
  expect([em.empty, em.data.length, em.cursor]).toEqual([true, 0, 3]);
  const b = frame32('snapshot_bars') as SnapshotMsg;
  expect([b.kind, b.labels, b.horizontal, b.colors!.length]).toEqual([8, ['cat', 'dog', 'fish'], true, 12]);
  expect(Array.from(b.data)).toEqual([0.125, 0.75, -0.25]);
  const cs = frame32('controls_stamped');
  expect(cs.type === 'controls' && Array.from(cs.stamp!)).toEqual([1200n, 1763200000123456789n]);
  const t = frame32('text') as TextMsg;
  expect(textLines(t)).toEqual(['epoch 1 starts', 'loss rose\nto 2.5', 'pas de données é']);
  expect([Array.from(t.levels), t.stamps!.length, t.stamps_t]).toEqual([[1, 2, 3], 6, 2]);
  const ts = frame32('text_stamp') as TextMsg;
  expect([ts.start, ts.total, ts.dropped, ts.cursor]).toEqual([10, 12, 10, 1]);
  expect(textLines(ts)).toEqual(['a', '']);
  const r = frame32('recordings');
  expect(r.type === 'recordings' && r.recordings[0].id).toBe('train-20261115-120314');
  expect((frame32('recordings_none') as { recordings: unknown[] }).recordings).toEqual([]);
  expect((frame32('recording_open') as { recording: string }).recording).toBe('train-20261115-120314');
  const c = frame32('cursor_play') as CursorMsg;
  expect([c.op, c.timeline, c.speed, c.pace, c.loop]).toEqual(['play', 'step', 2, 'even', ['1000', '5000']]);
  expect((frame32('cursor_step') as CursorMsg).count).toBe(-5);
  expect((frame32('cursor_follow') as CursorMsg).timeline).toBeUndefined();
  const p = frame32('time_state_paused') as TimeStateMsg;
  expect([p.request_id, p.cursor, p.generation]).toEqual([2, '5000', 2]);
  const fo = frame32('time_state_following') as TimeStateMsg;
  expect([fo.cursor, fo.live]).toEqual([null, true]);
  expect(new TextDecoder().decode(encode(fo, PROTOCOL_MINOR_3_2))).toContain('"cursor":null');
});

test('3.0 and 3.1 encodings are unchanged', () => {
  const root = goldenFrames();
  const files = [
    ...fs.readdirSync(root).filter((f) => f.endsWith('.bin')).map((f) => path.join(root, f)),
    ...fs.readdirSync(path.join(root, 'v3_1')).filter((f) => f.endsWith('.bin')).map((f) => path.join(root, 'v3_1', f)),
  ].filter((f) => !['ping_future.bin', 'unknown_type.bin'].includes(path.basename(f)));
  expect(files.length).toBeGreaterThan(30);
  let exact = 0;
  for (const file of files) {
    const buf = new Uint8Array(fs.readFileSync(file));
    const f = decodeFrame(buf);
    const msg = decode(buf);
    if (msg.type === 'unknown') continue;
    const again = encode(msg, f.minor);
    const g = decodeFrame(again);
    expect(g.meta, file).toEqual(f.meta);
    expect(g.sections.map((x) => [x.name, Array.from(x.bytes)]), file).toEqual(f.sections.map((x) => [x.name, Array.from(x.bytes)]));
    const text = new TextDecoder().decode(buf.subarray(24 + 4 * f.sections.length, 24 + 4 * f.sections.length + new DataView(buf.buffer, buf.byteOffset).getUint32(16, true)));
    if (text === canonicalJson(f.meta)) {
      expect(Array.from(again), file).toEqual(Array.from(buf));
      exact++;
    }
  }
  expect(exact).toBeGreaterThan(20);
});

test('stamp text', () => {
  for (const ok of ['0', '-1', '1763200000123456789', '9223372036854775807', '-9223372036854775808', '007']) expect(stampTextOk(ok), ok).toBe(true);
  for (const bad of ['', '-', '+1', '1.5', '1e3', ' 1', '9223372036854775808', '-9223372036854775809', '1'.repeat(21), 12, null, '١٢', '1\n'])
    expect(stampTextOk(bad), String(bad)).toBe(false);
});

const cursorFrame = (meta: Record<string, unknown>) => encodeFrame(MSG.cursor, meta, [], PROTOCOL_MINOR_3_2);

test('a cursor keeps only the fields of its op', () => {
  const meta = { request_id: 1, op: 'pause', timeline: 'step', value: '5', count: 3, speed: 2, pace: 'even', loop: null };
  let c = decode(cursorFrame(meta)) as CursorMsg;
  expect([c.timeline, c.value, c.count, c.speed]).toEqual([undefined, undefined, undefined, undefined]);
  expect(Array.from(encode(c, PROTOCOL_MINOR_3_2))).toEqual(Array.from(cursorFrame({ request_id: 1, op: 'pause' })));
  c = decode(cursorFrame({ ...meta, op: 'seek' })) as CursorMsg;
  expect([c.timeline, c.value, c.count]).toEqual(['step', '5', undefined]);
  for (const [op, extra, code] of [['step', { count: 1001 }, 'bad_cursor'], ['step', { count: 1.5 }, 'bad_cursor'], ['step', { count: 0 }, 'bad_cursor'],
    ['play', { speed: 0.05 }, 'bad_cursor'], ['play', { pace: 'fast' }, 'bad_cursor'], ['play', { timeline: 'bad name' }, 'bad_cursor'],
    ['play', { loop: ['5', '4'] }, 'bad_cursor'], ['seek', { value: '1.5' }, 'bad_cursor'], ['seek', { value: null }, 'bad_cursor'],
    ['rewind', {}, 'bad_request']] as [string, Record<string, unknown>, string][])
    refused(cursorFrame({ request_id: 1, op, timeline: 'step', ...extra }), code);
  refused(cursorFrame({ op: 'pause' }));
  refused(cursorFrame({ request_id: 1, op: 'step', timeline: 'bad name' }), 'bad_cursor');
  const ok = decode(cursorFrame({ request_id: 1, op: 'play', timeline: 't', speed: 1 / 16 })) as CursorMsg;
  expect([ok.speed, ok.loop]).toEqual([0.0625, undefined]);
});

test('text rules', () => {
  const enc = (m: TextMsg) => encode(m, PROTOCOL_MINOR_3_2);
  const t = textFromLines(1, 'log', 0, 2, 5, ['x'.repeat(TEXT_LINE_MAX), 'é'.repeat(3)], [0, 3]);
  expect(textLines(decode(enc(t)) as TextMsg)).toEqual(['x'.repeat(4096), 'ééé']);
  const many = textFromLines(1, 'log', 0, TEXT_MAX_LINES, 5, Array(TEXT_MAX_LINES).fill(''), Array(TEXT_MAX_LINES).fill(1));
  expect(textLines(decode(enc(many)) as TextMsg).length).toBe(TEXT_MAX_LINES);
  refused(enc(textFromLines(1, 'log', 0, TEXT_MAX_LINES + 1, 5, Array(TEXT_MAX_LINES + 1).fill(''), Array(TEXT_MAX_LINES + 1).fill(1))));
  refused(enc(textFromLines(1, 'log', 0, 1, 5, ['x'.repeat(TEXT_LINE_MAX + 1)], [1])));
  refused(enc(textFromLines(1, 'log', 0, 1, 5, ['a'], [4])));
  refused(enc(textFromLines(1, 'log', 1, 1, 5, ['a'], [1])));                 // 1 + 1 lines past a total of 1
  const text = (extra: Section[], over: Record<string, unknown> = {}) => encodeFrame(MSG.text, { sub: 1, view: 'log', start: 0, total: 1, stamp_ns: 5, ...over },
    [sec('text', DTYPE.u8, [1], Uint8Array.of(97)), sec('offsets', DTYPE.i32, [2], Int32Array.of(0, 1)), sec('levels', DTYPE.u8, [1], Uint8Array.of(1)), ...extra], PROTOCOL_MINOR_3_2);
  expect(decode(text([])).type).toBe('text');
  refused(text([sec('stamps', DTYPE.i64, [2, 2], new BigInt64Array(4))]));
  expect((decode(text([sec('stamps', DTYPE.i64, [1, 2], new BigInt64Array(2))])) as TextMsg).stamps_t).toBe(2);
  refused(text([sec('stamp', DTYPE.i64, [1], new BigInt64Array(1)), sec('stamps', DTYPE.i64, [1, 1], new BigInt64Array(1))]));
  // offsets that do not start at 0, decrease, or miss the end; bytes that are not UTF-8
  const bad = (text_: Uint8Array, offs: number[], lv = [1]) => encodeFrame(MSG.text, { sub: 1, view: 'log', start: 0, total: offs.length - 1, stamp_ns: 5 },
    [sec('text', DTYPE.u8, [text_.length], text_), sec('offsets', DTYPE.i32, [offs.length], Int32Array.from(offs)), sec('levels', DTYPE.u8, [lv.length], Uint8Array.from(lv))], PROTOCOL_MINOR_3_2);
  refused(bad(Uint8Array.of(97, 98), [1, 2]));
  refused(bad(Uint8Array.of(97, 98), [0, 1]));
  refused(bad(Uint8Array.of(97, 98), [0, 2, 1], [1, 1]));
  refused(bad(Uint8Array.of(0xc3, 0x28), [0, 2]));
  refused(bad(Uint8Array.of(0xc3, 0xa9), [0, 1, 2], [1, 1]));                  // a line cut inside a character
  expect(decode(bad(Uint8Array.of(0xc3, 0xa9), [0, 2])).type).toBe('text');
});

const pieces = () => [0, 1, 2, 3, 4].map((k) => frame32(`scene_update_split_${k}`) as SceneUpdateMsg);
const asArray = (a: { data: ArrayLike<number> }) => Array.from(a.data);

test('a split update assembles to the whole', () => {
  const ps = pieces();
  expect(ps.map((p) => p.split!.piece)).toEqual([0, 1, 2, 3, 4]);
  expect(ps[0].params).toEqual({ radius: 0.25 });
  expect(ps.slice(1).every((p) => Object.keys(p.params).length === 0)).toBe(true);
  expect(ps.every((p) => p.stamp instanceof BigInt64Array)).toBe(true);
  const su = new SplitUpdates();
  const got = ps.map((p) => su.add(p));
  expect(got.slice(0, 4)).toEqual([null, null, null, null]);
  expect(su.open.size).toBe(0);
  const u = got[4]!;
  expect(u.type).toBe('scene_update');
  expect(u.split).toBeUndefined();
  // the pieces' rows, concatenated in order of 'first'
  const cat = (name: string) => ps.filter((p) => name in p.arrays).sort((a, b) => a.split!.first[name] - b.split!.first[name]).flatMap((p) => asArray(p.arrays[name]));
  expect(u.arrays.positions.shape).toEqual([20, 3]);
  expect(u.arrays.colors.shape).toEqual([20, 4]);
  expect(asArray(u.arrays.positions)).toEqual(cat('positions'));
  expect(asArray(u.arrays.colors)).toEqual(cat('colors'));
  expect(u.arrays.positions.data.length).toBe(60);
  expect([u.path, u.seq, u.params, Array.from(u.stamp!)]).toEqual(['world/split', 37, { radius: 0.25 }, Array.from(ps[0].stamp!)]);
  expect(u.arrays.positions.data.buffer).not.toBe(ps[0].arrays.positions.data.buffer); // the assembly's own buffer
  // the pieces are not changed by the assembly, and encode back to their frames
  ps.forEach((p, k) => expect(Array.from(decodeFrame(encode(p, PROTOCOL_MINOR_3_2)).sections.map((x) => x.name))).toEqual(decodeFrame(read(`scene_update_split_${k}.bin`)).sections.map((x) => x.name)));
  // it encodes as an unsplit update
  expect(decodeFrame(encode(u, PROTOCOL_MINOR_3_2)).meta.split).toBeUndefined();
});

test('split update rules', () => {
  const ps = pieces();
  const withSplit = (p: SceneUpdateMsg, split: Partial<NonNullable<SceneUpdateMsg['split']>>, over: Partial<SceneUpdateMsg> = {}): SceneUpdateMsg =>
    ({ ...p, ...over, split: { ...p.split!, ...split } });
  const fails = (fn: () => unknown) => expect(catchError(fn)).toBeInstanceOf(MessageError);
  // a piece without its piece 0 is dropped and counted
  let su = new SplitUpdates();
  expect(su.add(ps[1])).toBeNull();
  expect([su.dropped, su.open.size]).toEqual([1, 0]);
  // a piece twice
  su = new SplitUpdates();
  su.add(ps[0]);
  su.add(ps[1]);
  fails(() => su.add(ps[1]));
  expect(su.open.size).toBe(0);
  // a newer update of the same entity part discards the incomplete one
  su = new SplitUpdates();
  su.add(ps[0]);
  su.add(withSplit(ps[0], { update: 5 }));
  expect([Array.from(su.open.keys()), su.discarded]).toEqual([[5], 1]);
  expect(su.add(ps[1])).toBeNull();
  expect(su.dropped).toBe(1);
  // at most two incomplete updates; the oldest goes
  su = new SplitUpdates();
  ['a', 'b', 'c'].forEach((path, k) => su.add(withSplit(ps[0], { update: 10 + k }, { path })));
  expect(Array.from(su.open.keys())).toEqual([11, 12]);
  // a piece that differs from its piece 0, or whose rows overlap
  for (const bad of [{ ...ps[1], seq: 38 }, withSplit(ps[1], { first: { positions: 4 } })]) {
    su = new SplitUpdates();
    su.add(ps[0]);
    fails(() => su.add(bad));
  }
  // the last piece arrives with rows missing
  su = new SplitUpdates();
  su.add(withSplit(ps[0], { pieces: 2 }));
  fails(() => su.add(withSplit(ps[4], { piece: 1, pieces: 2 })));
  // the whole update is checked as an unsplit one: colours of the wrong row count
  su = new SplitUpdates();
  const p0 = withSplit(ps[0], { pieces: 2, rows: { positions: 20, colors: 2 } });
  const rest: SceneUpdateMsg = {
    type: 'scene_update', sub: 2, view: 'world', seq: 37, stamp_ns: ps[0].stamp_ns, path: 'world/split', part: 'points',
    arrays: {
      positions: { dtype: DTYPE.f32, shape: [14, 3], data: new Float32Array(42) },
      colors: { dtype: DTYPE.u8, shape: [2, 4], data: new Uint8Array(8) },
    },
    params: {}, stamp: ps[0].stamp, split: { update: 4, piece: 1, pieces: 2, first: { positions: 6, colors: 0 } },
  };
  su.add(decode(encode(p0, PROTOCOL_MINOR_3_2)) as SceneUpdateMsg);
  fails(() => su.add(decode(encode(rest, PROTOCOL_MINOR_3_2)) as SceneUpdateMsg));
  // key checks of a piece, refused when decoded
  const frameOf = (split: unknown, extra: Record<string, unknown> = {}) => update('points', { positions: f32s([2, 3]) }, { split, ...extra });
  const ok = { update: 1, pieces: 2, piece: 0, first: { positions: 0 }, rows: { positions: 4 } };
  expect((decode(frameOf(ok)) as SceneUpdateMsg).split).toEqual(ok);
  refused(frameOf({ ...ok, update: -1 }));
  refused(frameOf({ ...ok, pieces: 1 }));
  refused(frameOf({ ...ok, pieces: MAX_PIECES + 1 }));
  refused(frameOf({ ...ok, piece: 2 }));
  refused(frameOf({ ...ok, first: { colors: 0 } }));                          // first names an array the piece does not hold
  refused(frameOf({ ...ok, rows: undefined }));                               // piece 0 needs rows
  refused(frameOf({ ...ok, rows: { positions: 1 } }));                        // rows pass split.rows
  refused(frameOf({ ...ok, rows: { positions: 4, nope: 1 } }));
  refused(frameOf({ ...ok, piece: 1 }));                                      // only piece 0 carries rows
  refused(frameOf({ ...ok, first: { positions: -1 } }));
  // piece 0 checks the part's keys, the other pieces carry none
  refused(frameOf(ok, { radius: 0 }));
  expect(Object.keys((decode(frameOf({ update: 1, pieces: 2, piece: 1, first: { positions: 2 } }, { radius: 0 })) as SceneUpdateMsg).params)).toEqual([]);
  // a piece holds any dtype; the assembly copies its bytes
  const f16 = update('points', { positions: [DTYPE.f16, [2, 3], new Uint16Array(6)] }, { split: ok });
  expect((decode(f16) as SceneUpdateMsg).arrays.positions.dtype).toBe(DTYPE.f16);
});

test('a split update of a large mesh', () => {
  // small frames stand in for 64 MiB: the pieces are cut here, 8192 rows to a piece
  const n = 30000;
  const pos = new Float32Array(n * 3).map((_, i) => (i * 7919 % 10007) / 10007);
  const tri = new Int32Array(n * 3).map((_, i) => i % n);
  const cut = 8192;
  const ps: SceneUpdateMsg[] = [];
  const count = Math.ceil(n / cut);
  for (let k = 0; k < count; k++) {
    const a = k * cut;
    const b = Math.min(n, a + cut);
    ps.push({
      type: 'scene_update', sub: 1, view: 's', seq: 3, stamp_ns: 4, path: 'm', part: 'mesh',
      arrays: { positions: { dtype: DTYPE.f32, shape: [b - a, 3], data: pos.slice(a * 3, b * 3) }, indices: { dtype: DTYPE.i32, shape: [b - a, 3], data: tri.slice(a * 3, b * 3) } },
      params: k === 0 ? { color: [1, 2, 3, 255] } : {},
      split: { update: 9, piece: k, pieces: count, first: { positions: a, indices: a }, ...(k === 0 ? { rows: { positions: n, indices: n } } : {}) },
    });
  }
  const su = new SplitUpdates();
  const out = ps.map((p) => su.add(decode(encode(p, PROTOCOL_MINOR_3_2)) as SceneUpdateMsg));
  expect(out.slice(0, -1).every((x) => x === null)).toBe(true);
  const u = out[count - 1]!;
  expect(u.arrays.positions.shape).toEqual([n, 3]);
  expect(Array.from(u.arrays.positions.data as Float32Array)).toEqual(Array.from(pos));
  expect(Array.from(u.arrays.indices.data as Int32Array)).toEqual(Array.from(tri));
  expect(u.params).toEqual({ color: [1, 2, 3, 255] });
  // pieces in any order, once piece 0 is first
  const su2 = new SplitUpdates();
  const order = [0, ...[...Array(count).keys()].slice(1).reverse()];
  const res = order.map((k) => su2.add(decode(encode(ps[k], PROTOCOL_MINOR_3_2)) as SceneUpdateMsg));
  expect(Array.from(res[count - 1]!.arrays.positions.data as Float32Array)).toEqual(Array.from(pos));
  // a whole update over MAX_UPDATE_BYTES is refused before it allocates
  const huge = { ...ps[0], split: { ...ps[0].split!, rows: { positions: MAX_UPDATE_BYTES / 12 + 1, indices: n } } };
  fails2(() => new SplitUpdates().add(decode(encode(huge, PROTOCOL_MINOR_3_2)) as SceneUpdateMsg));
});

function fails2(fn: () => unknown): void {
  expect(catchError(fn)).toBeInstanceOf(MessageError);
}

test('golden values of programs and interventions', async () => {
  const { createHash } = await import('node:crypto');
  const g = frame32('program_set_geometry') as ProgramMsg;
  expect([g.op, g.id, g.scene, g.request_id]).toEqual(['set', 'rows', 'world', 11]);
  expect(g.wgsl!.endsWith('\n')).toBe(true);
  expect(createHash('sha256').update(g.wgsl!, 'utf8').digest('hex')).toBe(g.sha256);
  expect(g.wgsl).toContain('\u00b5');
  expect((g.declaration!.output as Record<string, unknown>).max_count).toBe(4096);
  const gf = decodeFrame(read('program_set_geometry.bin'));
  expect(gf.sections.map((x) => x.name)).toEqual(['wgsl']);
  expect(gf.sections[0].dtype).toBe(DTYPE.u8);
  const p = frame32('program_set_pixels') as ProgramMsg;
  expect(p.scene).toBeUndefined();
  expect((p.declaration!.output as Record<string, unknown>).width).toBe(64);
  const r = frame32('program_remove') as ProgramMsg;
  expect([r.op, r.declaration, r.wgsl, r.sha256]).toEqual(['remove', undefined, undefined, undefined]);
  expect(decodeFrame(read('program_remove.bin')).sections).toEqual([]);
  const st = frame32('program_state_reply') as ProgramStateMsg;
  expect([st.request_id, st.step, st.step_ms]).toEqual([12, 1200, 31.25]);
  expect(st.programs).toHaveLength(2);
  expect(st.programs.map((x) => x.state)).toEqual(['running', 'refused']);
  expect([st.programs[0].overhead_share, st.programs[0].passes_per_s]).toEqual([0.03125, 2.5]);
  expect((st.programs[1].error as Record<string, unknown>).code).toBe('program.unknown_input');
  const wt = frame32('program_state_waiting') as ProgramStateMsg;
  expect([wt.request_id, wt.step, wt.programs.length]).toEqual([undefined, undefined, 1]);

  const s = frame32('control_intervene_set') as ControlMsg;
  expect(s.intervene).toEqual({ set: { point: 'blocks.0.attn#3', op: 'ablate', from_step: 1201, until_step: null } });
  expect((frame32('control_intervene_clear') as ControlMsg).intervene).toEqual({ clear: 'blocks.0.attn#3' });
  expect((frame32('control_intervene_clear_all') as ControlMsg).intervene).toEqual({ clear_all: true });
  expect((frame32('control_reply_intervene') as { op: string }).op).toBe('intervene');
  const c = frame32('controls_interventions') as ControlsMsg;
  expect(c.interventions!.points).toHaveLength(2);
  expect(c.interventions!.points.map((x) => x.kind)).toEqual(['module', 'head']);
  expect(c.interventions!.active[0].state).toBe('active');
  expect((decode(new Uint8Array(fs.readFileSync(path.join(goldenFrames(), 'v3_1', 'controls.bin')))) as ControlsMsg).interventions).toBeUndefined();
});

test('golden values of instances of pieces and resources', () => {
  const sec = (name: string) => decodeFrame(read(`${name}.bin`)).sections.map((x) => x.name);
  const p = frame32('program_set_piece') as ProgramMsg;
  expect(sec('program_set_piece')).toEqual(['common', 'derive', 'draw']);
  expect([p.op, p.id, p.request_id, p.panel]).toEqual(['set', 'heads', 31, 'bars']);
  expect(p.piece).toEqual({ name: 'head_bars', version: 2, hash: 'sha256:6aa44eda0c257cd658221f0f7d0fddfc7cd77c0d9529e812d63b62c5cfb1a040' });
  expect(p.bind).toEqual({ attn: 'live:blocks.0.attn' });
  expect(p.params).toEqual({ gain: 2 });
  expect(p.segments).toEqual([{ part: 'lib/colormap@1', first_line: 1, lines: 1 }, { part: 'common.wgsl', first_line: 2, lines: 1 }]);
  expect(p.declaration).toBeUndefined();
  expect(p.wgsl).toBeUndefined();
  expect(p.derive).toContain('fn mean_heads');
  expect(p.draw).toContain('fn bar_vertex');
  expect(p.draw).toContain('\u00b5');
  expect(p.common).toContain('fn bar_height');
  expect((p.manifest as { name: string }).name).toBe('head_bars');
  const d = frame32('program_set_piece_draw') as ProgramMsg;
  expect(sec('program_set_piece_draw')).toEqual(['common', 'draw']);
  expect([d.request_id, d.derive, d.piece!.name]).toEqual([0, undefined, 'head_bars']);
  expect(d.draw).toBe(p.draw);
  expect(d.common).toBe(p.common);
  const r = frame32('program_remove_piece') as ProgramMsg;
  expect([r.op, r.request_id]).toEqual(['remove', 0]);
  const st = frame32('program_state_piece') as ProgramStateMsg;
  expect([st.step, st.programs[0].kind, st.programs[0].state, st.programs[0].runs]).toEqual([1200, STATUS_PIECE, 'running', 4]);

  const b = frame32('resource_buffer') as ResourceMsg;
  expect([b.type, b.program, b.resource, b.step, b.count, b.stride, b.width, b.height]).toEqual(['resource', 'heads', 'heads', 1200, 4, 4, undefined, undefined]);
  const f = decodeFrame(read('resource_buffer.bin'));
  expect(Array.from(new Float32Array(b.data.slice().buffer))).toEqual([0.5, 0.25, 0.125, 0.0625]);
  const t = frame32('resource_texture') as ResourceMsg;
  expect([t.program, t.resource, t.width, t.height, t.count, t.stride, t.step, t.data.length]).toEqual(['arcs', 'glow', 2, 2, 4, 4, 1210, 16]);
  const u = frame32('resource_struct') as ResourceMsg;
  expect(u.data.length).toBe(u.count * u.stride);
  const e = frame32('resource_empty') as ResourceMsg;
  expect([e.count, e.stride, e.step, e.data.length]).toEqual([0, 24, 1220, 0]);

  // The data is a view of the frame's own buffer, not a copy.
  const bytes = read('resource_buffer.bin');
  const view = decode(bytes) as ResourceMsg;
  expect(view.data.buffer).toBe(bytes.buffer);
  expect(f.sections[0].bytes.byteLength).toBe(16);
  // A decoded golden frame without a float key encodes to its bytes (the generic round trip above covers the others by metadata).
  for (const n of ['program_state_piece', 'resource_empty', 'resource_struct', 'resource_texture']) {
    const raw = read(`${n}.bin`);
    expect(Array.from(encode(decode(raw) as Message, PROTOCOL_MINOR_3_2)), n).toEqual(Array.from(raw));
  }
});

/** 10 segments (k, -k) of two f32: the version that resource_split_0 to _2 carry. */
function wholeSegments(): Uint8Array {
  const v = new Float32Array(20);
  for (let k = 0; k < 10; k++) [v[2 * k], v[2 * k + 1]] = [k, 0 - k];   // 0 - 0 is +0, as in the generator
  return new Uint8Array(v.buffer);
}

test('a resource in parts carries its golden values and is put together whole', () => {
  const parts = [0, 1, 2].map((k) => frame32(`resource_split_${k}`) as ResourceMsg);
  expect(parts.map((p) => p.split)).toEqual([0, 1, 2].map((k) => ({ version: 5, part: k, parts: 3, first: 4 * k })));
  const sr = new SplitResources();
  expect(sr.add(parts[0])).toBeNull();
  expect(sr.add(parts[1])).toBeNull();
  const whole = sr.add(parts[2])!;
  expect([whole.program, whole.resource, whole.step, whole.count, whole.stride, whole.split]).toEqual(['lines', 'segments', 1300, 10, 8, undefined]);
  expect(Array.from(whole.data)).toEqual(Array.from(wholeSegments()));
  expect(sr.open.size).toBe(0);
  const again = decode(encode(whole, PROTOCOL_MINOR_3_2)) as ResourceMsg;
  expect(new SplitResources().add(again)!.count).toBe(10);
  for (const k of [0, 1, 2]) {
    const raw = read(`resource_split_${k}.bin`);
    expect(Array.from(encode(parts[k], PROTOCOL_MINOR_3_2))).toEqual(Array.from(raw));
  }
});

test('a receiver drops or refuses the parts of a resource that break the order', () => {
  const [p0, p1, p2] = [0, 1, 2].map((k) => frame32(`resource_split_${k}`) as ResourceMsg);
  const orphan = frame32('resource_split_orphan') as ResourceMsg;
  const refused = (sr: SplitResources, r: ResourceMsg, why: RegExp) => {
    let err: unknown;
    try {
      sr.add(r);
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(MessageError);
    expect((err as MessageError).message).toMatch(why);
  };
  let sr = new SplitResources();
  expect(sr.add(orphan)).toBeNull();
  sr.add(p0);
  expect(sr.add(orphan)).toBeNull();
  expect([sr.dropped, sr.open.size]).toEqual([2, 1]);
  refused(sr, p2, /out of order/);
  expect(sr.open.size).toBe(0);
  sr.add(p0);
  sr.add(p1);
  refused(sr, frame32('resource_split_short') as ResourceMsg, /end before its count/);
  sr.add(p0);
  refused(sr, frame32('resource_split_differs') as ResourceMsg, /differs from its part 0/);
  // A version that never completes expires after RESOURCE_PART_WAIT further frames.
  sr = new SplitResources();
  sr.add(p0);
  const other = frame32('resource_buffer') as ResourceMsg;
  for (let k = 0; k + 1 < RESOURCE_PART_WAIT; k++) expect(sr.add(other)).not.toBeNull();
  expect(sr.add(p1)).toBeNull();
  for (let k = 0; k < RESOURCE_PART_WAIT; k++) sr.add(other);
  expect(sr.open.size).toBe(1);
  sr.add(other);
  expect([sr.open.size, sr.expired]).toEqual([0, 1]);
  expect(sr.add(p2)).toBeNull();
  expect(sr.dropped).toBe(1);
  // A newer version discards an incomplete one; at most MAX_INCOMPLETE_UPDATES are held.
  sr = new SplitResources();
  sr.add(p0);
  sr.add(p0);
  expect([sr.open.size, sr.discarded]).toEqual([1, 1]);
  for (const name of ['a', 'b', 'c']) sr.add({ ...p0, resource: name });
  expect([...sr.open.keys()]).toEqual(['lines/b', 'lines/c']);
  expect(MAX_INCOMPLETE_UPDATES).toBe(2);
});

test('the parts of a resource bound every size before a product', () => {
  const zeros = new Uint8Array(256);
  const meta = { program: 'a', resource: 'r', step: 0, count: 2 ** 32 - 1, stride: 256, split: { version: 0, part: 0, parts: 2, first: 0 } };
  const frame = encodeFrame(MSG.resource, meta, [{ name: 'resource', dtype: DTYPE.u8, shape: [256], encoding: 0, bytes: zeros }], PROTOCOL_MINOR_3_2);
  expect(() => decode(frame)).toThrow(/in parts holds at most 134217728 bytes/);
  expect(RESOURCE_VERSION_MAX_BYTES).toBe(128 * 1024 * 1024);
});
