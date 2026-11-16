// The codec against the golden frames of the protocol 3.2 draft's scenes subset
// (tests/fixtures/wire_v3/v3_2): every frame decodes to what its .json says,
// every invalid frame is refused as expect.json says, and every frame but the
// two that carry sections the subset ignores encodes back at minor 2 to the
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
  CAP_SCENES, KIND_SCENE, LIMITS_32, MSG, PROTOCOL_MINOR_3_2, SCENE_DATA_VIEW, type SceneUpdateMsg, type ViewEntry,
} from '../src/protocol/messages.js';
import { expect, goldenFrames, REPO, test } from './util.js';

test.beforeEach(({}, info) => test.skip(info.project.name !== 'canvas2d', 'the codec runs in Node once'));

const dir = path.join(goldenFrames(), 'v3_2');
const names = fs.readdirSync(dir).filter((f) => f.endsWith('.bin')).map((f) => f.slice(0, -4)).sort();
const DECODE_ONLY = ['scene_update_points_stamped', 'scene_update_mesh_normals'];
const typeName = (id: number) => Object.entries(MSG).find(([, v]) => v === id)?.[0] ?? null;
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
  expect(names.length).toBeGreaterThanOrEqual(16);
  expect(names).toContain('scene_update_mesh');
  expect(typeName(0x70)).toBe('scene_update');
  expect(typeName(0x72)).toBe('scene_clear');
});

let byteEqual = 0;
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
    // Section values: f32, i32 and u8 as numbers; i64 is compared as bytes only (JSON loses it above 2^53).
    f.sections.forEach((s, i) => {
      const w = want.sections[i];
      if (s.dtype === 3) return;
      const a = s.dtype === 5 ? s.bytes : s.dtype === 4
        ? new Int32Array(s.bytes.buffer, s.bytes.byteOffset, s.bytes.byteLength / 4)
        : new Float32Array(s.bytes.buffer, s.bytes.byteOffset, s.bytes.byteLength / 4);
      expect(Array.from(a), s.name).toEqual(w.values);
    });

    const msg = decode(buf);
    expect(msg.type === 'unknown' ? null : msg.type).toBe(want.type_name);
    if (msg.type === 'scene_update') {
      for (const [k, a] of Object.entries(msg.arrays)) {
        const s = f.sections.find((x) => x.name === k)!;
        expect([a.dtype, a.shape]).toEqual([s.dtype, s.shape]);
        expect(Array.from(a.data)).toEqual(Array.from(a.dtype === 5 ? s.bytes : a.dtype === 4
          ? new Int32Array(s.bytes.buffer, s.bytes.byteOffset, s.bytes.byteLength / 4)
          : new Float32Array(s.bytes.buffer, s.bytes.byteOffset, s.bytes.byteLength / 4)));
        expect(a.data.buffer).toBe(buf.buffer); // a view, not a copy
        expect(a.data.constructor).toBe(a.dtype === 5 ? Uint8Array : a.dtype === 4 ? Int32Array : Float32Array);
      }
      expect([msg.sub, msg.view, msg.seq, msg.stamp_ns, msg.path, msg.part]).toEqual([want.meta.sub, want.meta.view, want.meta.seq, want.meta.stamp_ns, want.meta.path, want.meta.part]);
      expect(msg.static ?? false).toBe(want.meta.static ?? false);
      expect(msg.state ?? false).toBe(want.meta.state ?? false);
      if (want.meta.stats) expect(msg.params.stats).toEqual(want.meta.stats.map((x: number | null) => (x === null ? NaN : x)));
      if (want.meta.mode) expect(msg.params.mode).toBe(want.meta.mode);
    }
    if (msg.type === 'scene_clear') {
      expect([msg.path, msg.recursive ?? false, msg.reset ?? false, msg.parts]).toEqual([want.meta.path, want.meta.recursive ?? false, want.meta.reset ?? false, want.meta.parts]);
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
      byteEqual++;
    } else {
      // Only a float written with a trailing .0 by Python can differ.
      expect(text.replace(/(\d)\.0(?![0-9])/g, '$1')).toBe(canonicalJson(f.meta));
    }
  });
}

test('some 3.2 frames re-encode byte for byte', () => {
  expect(byteEqual).toBeGreaterThanOrEqual(3);
});

test('refuses every invalid 3.2 frame as expect.json says', () => {
  const expected: Record<string, { reason?: string; close_code?: number; error?: string }> = JSON.parse(fs.readFileSync(path.join(dir, 'invalid', 'expect.json'), 'utf8'));
  expect(Object.keys(expected).length).toBeGreaterThanOrEqual(18);
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

const sec = (name: string, dtype: number, shape: number[], data: Float32Array | Int32Array | Uint8Array): Section =>
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

test('the 3.2 constants agree with include/caliper/live_net.h', () => {
  const text = fs.readFileSync(path.join(REPO, 'include', 'caliper', 'live_net.h'), 'utf8');
  const block = text.slice(text.indexOf('PROTOCOL 3.2 DRAFT: THE SCENES SUBSET'));
  expect(block.length).toBeGreaterThan(100);
  const defs: Record<string, string | number> = {};
  for (const m of block.matchAll(/^#define\s+(CALIPER_NET_\w+)\s+("[^"]*"|0x[0-9a-fA-F]+u?|\d+u?)/gm)) {
    const v = m[2];
    defs[m[1]] = v.startsWith('"') ? v.slice(1, -1) : Number(v.replace(/u$/, ''));
  }
  expect(defs).toMatchObject({
    CALIPER_NET_MINOR_3_2: PROTOCOL_MINOR_3_2,
    CALIPER_NET_CAP_SCENES: CAP_SCENES,
    CALIPER_NET_KIND_SCENE: KIND_SCENE,
    CALIPER_NET_MSG_SCENE_UPDATE: MSG.scene_update,
    CALIPER_NET_MSG_SCENE_CLEAR: MSG.scene_clear,
    CALIPER_NET_PATH_MAX: LIMITS_32.pathMax,
    CALIPER_NET_PATH_SEGMENTS_MAX: LIMITS_32.pathSegmentsMax,
    CALIPER_NET_UNITS_MAX: LIMITS_32.unitsMax,
    CALIPER_NET_MAX_ENTITIES: LIMITS_32.maxEntities,
    CALIPER_NET_MAX_BATCHES: LIMITS_32.maxBatches,
    CALIPER_NET_MAX_BATCH_POINTS: LIMITS_32.maxBatchPoints,
    CALIPER_NET_MAX_BATCH_ITEMS: LIMITS_32.maxBatchItems,
    CALIPER_NET_SCENE_DEFAULT_MAX_ITEMS: SCENE_DATA_VIEW.budget.max_items,
    CALIPER_NET_SCENE_DEFAULT_MAX_VERTICES: SCENE_DATA_VIEW.budget.max_vertices,
    CALIPER_NET_SCENE_DEFAULT_MAX_EXTENT: SCENE_DATA_VIEW.budget.max_extent,
  });
});
