// The codec against the wire v3 golden frames (tests/fixtures/wire_v3, and the
// 3.1 frames in its v3_1 folder): every frame decodes to what its .json says,
// every invalid frame is rejected with the stated reason and close code, and
// every message encodes back, at the frame's own minor version, to the same
// metadata and sections, and to the same bytes when its metadata text is the
// canonical one. Runs in Node, once.

import fs from 'node:fs';
import path from 'node:path';
import { canonicalJson, decode, decodeFrame, encode, WireError } from '../src/protocol/codec.js';
import { MSG } from '../src/protocol/messages.js';
import { expect, goldenFrames, test } from './util.js';

test.beforeEach(({}, info) => test.skip(info.project.name !== 'canvas2d', 'the codec runs in Node once'));

const dir = goldenFrames();
const names = [
  ...fs.readdirSync(dir).filter((f) => f.endsWith('.bin')).map((f) => f.slice(0, -4)),
  ...(fs.existsSync(path.join(dir, 'v3_1')) ? fs.readdirSync(path.join(dir, 'v3_1')).filter((f) => f.endsWith('.bin')).map((f) => `v3_1/${f.slice(0, -4)}`) : []),
].sort();
const typeName = (id: number) => Object.entries(MSG).find(([, v]) => v === id)?.[0] ?? null;

test('the golden frames are there, with the 3.1 frames', () => {
  expect(names.length).toBeGreaterThanOrEqual(20);
  expect(names.filter((n) => n.startsWith('v3_1/')).length).toBeGreaterThanOrEqual(16);
});

for (const name of names) {
  test(`decodes ${name}`, () => {
    const buf = new Uint8Array(fs.readFileSync(path.join(dir, `${name}.bin`)));
    const want = JSON.parse(fs.readFileSync(path.join(dir, `${name}.json`), 'utf8'));
    const f = decodeFrame(buf);
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    expect({
      magic: dv.getUint32(0, true), major: f.major, minor: f.minor, type: f.type, flags: f.flags,
      frame_len: buf.byteLength, meta_len: dv.getUint32(16, true), section_count: f.sections.length, reserved: dv.getUint16(22, true),
    }).toEqual(want.header);
    expect(f.meta).toEqual(want.meta);
    expect(f.sections.map((s) => ({
      name: s.name, dtype: s.dtype, shape: s.shape, encoding: s.encoding, offset: s.bytes.byteOffset, length: s.bytes.byteLength,
      values: Array.from(s.dtype === 5 ? s.bytes : new Float32Array(s.bytes.buffer, s.bytes.byteOffset, s.bytes.byteLength / 4)),
    }))).toEqual(want.sections.map((s: Record<string, unknown>) => ({ encoding: 0, ...s })));

    const msg = decode(buf);
    expect(msg.type === 'unknown' ? null : msg.type).toBe(want.type_name);
    if (msg.type === 'unknown') return;
    if (msg.type === 'snapshot') {
      expect(Array.from(msg.data)).toEqual(want.sections[0].values);
      expect(msg.shape).toEqual(want.sections[0].shape);
      if (want.sections[1]) expect(Array.from(msg.aux!)).toEqual(want.sections[1].values);
      expect(msg.stats).toEqual((want.meta.stats ?? []).map((x: number | null) => (x === null ? NaN : x)));
    }
    if (msg.type === 'series') expect(Array.from(msg.values)).toEqual(want.sections[0].values);
    if (msg.type === 'event') {
      expect(msg.event).toBe(want.meta.type);
      expect(msg.detail).toEqual(want.meta.detail);
    }
    if (msg.type === 'controls') expect({ run: msg.run, params: msg.params, actions: msg.actions }).toEqual(want.meta);
    if (msg.type === 'control') expect(msg.op).toBe(want.meta.op);
    if (msg.type === 'control_reply') expect([msg.status, msg.error?.code]).toEqual([want.meta.status, want.meta.error?.code]);
    if (msg.type === 'selection') expect([msg.space, msg.items, msg.details ?? null]).toEqual([want.meta.space, want.meta.items, want.meta.details ?? null]);
    if (msg.type === 'welcome') expect(msg.control).toBe(want.meta.control);

    // Encoding the decoded message gives the same metadata and sections, and
    // the same bytes whenever the canonical metadata text is the same.
    const again = encode(msg, f.minor);
    const g = decodeFrame(again);
    expect(g.type).toBe(f.type);
    const known = { ...f.meta };
    if (name === 'ping_future') delete known.future_key;
    expect(g.meta).toEqual(known);
    expect(g.sections.map((s) => Array.from(s.bytes))).toEqual(f.sections.map((s) => Array.from(s.bytes)));
    const text = new TextDecoder().decode(buf.subarray(24 + 4 * f.sections.length, 24 + 4 * f.sections.length + want.header.meta_len));
    if (text === canonicalJson(f.meta) && f.flags === 0) expect(Array.from(again)).toEqual(Array.from(buf));
    expect(typeName(f.type)).toBe(want.type_name);
  });
}

test('rejects every invalid frame with its reason and close code', () => {
  const expected: Record<string, { reason: string; close_code: number }> = JSON.parse(fs.readFileSync(path.join(dir, 'invalid', 'expect.json'), 'utf8'));
  expect(Object.keys(expected).length).toBeGreaterThanOrEqual(10);
  for (const [file, want] of Object.entries(expected)) {
    const buf = new Uint8Array(fs.readFileSync(path.join(dir, 'invalid', file)));
    let err: unknown = null;
    try {
      decodeFrame(buf);
    } catch (e) {
      err = e;
    }
    expect(err, file).toBeInstanceOf(WireError);
    expect({ file, reason: (err as WireError).reason, close: (err as WireError).closeCode }).toEqual({ file, reason: want.reason, close: want.close_code });
  }
});
