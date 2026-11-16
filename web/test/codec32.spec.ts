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
  canonicalJson, decode, decodeFrame, encode, encodeFrame, entityPathOk, MessageError, resolveSceneDataView, sceneViewInfo, WireError,
  type Section,
} from '../src/protocol/codec.js';
import {
  CAMERA_REQUIRED, CAP_SCENES, CLEAR_PARTS, COLOR_PER_EDGE, COLOR_PER_STRIP, COLOR_PER_VERTEX, CONVENTION_RDF, CONVENTION_RUB,
  DTYPE, DTYPE_SIZE, FORM_COLOR, FORM_DEPTH, FORM_ENCODED, FORM_SEGMENTATION, FORMAT_JPEG, FORMAT_PNG, KIND_SCENE, LIMITS_32,
  LINES_EDGES, LINES_SEGMENTS, LINES_STRIPS, MAX_CLASS_ID, MAX_CLASSES, MAX_ENCODED_BYTES, MAX_IMAGE_SIDE, MAX_LABELS, MAX_TIMELINES,
  MAX_TRANSFORMS, MSG, PART_ARROWS, PART_BOXES, PART_CAMERA, PART_IMAGE, PART_KEYS, PART_LINES, PART_MESH, PART_POINTS,
  PART_SECTIONS, PART_TRANSFORM, PROTOCOL_MINOR_3_2, SCENE_DATA_VIEW, SCENE_PARTS, STAMP_ABSENT,
  type SceneClassesMsg, type SceneTransformsMsg, type SceneUpdateMsg, type ViewEntry,
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
  expect(names.length).toBeGreaterThanOrEqual(38);
  expect(names).toContain('scene_update_mesh');
  expect(names).toContain('scene_transforms_stamps');
  expect(typeName(0x70)).toBe('scene_update');
  expect(typeName(0x71)).toBe('scene_transforms');
  expect(typeName(0x72)).toBe('scene_clear');
  expect(typeName(0x73)).toBe('scene_classes');
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
  expect(Object.keys(expected).length).toBeGreaterThanOrEqual(59);
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
  for (const m of block.matchAll(/^#define\s+(CALIPER_NET_\w+)\s+("[^"]*"|0x[0-9a-fA-F]+u?|\d+u?)/gm)) {
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
  refused(update('points', { positions: p3 }, { split: { axis: 0 } }), 'unsupported');
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
