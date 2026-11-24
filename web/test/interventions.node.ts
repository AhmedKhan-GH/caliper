import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { canonicalJson, decode, encode, encodeFrame, HEADER_BYTES, MessageError } from '../src/protocol/codec.js';
import { INTERVENTION_LIMITS, MAX_INTERVENTIONS, MAX_POINTS_LISTED, MSG, type InterventionSpec, type Message } from '../src/protocol/messages.js';

const fixtures = process.env.CALIPER_WIRE_FIXTURES ?? fileURLToPath(new URL('../../tests/fixtures/wire_v3/', import.meta.url));
const plain = (value: unknown) => JSON.parse(JSON.stringify(value));
const request = (intervene: unknown) => ({ request_id: 1, op: 'intervene', intervene });
const controls = (active: unknown[], points: unknown[] = []) => ({ run: { state: 'none' }, params: [], actions: [], interventions: { points, active } });
const ablate = { kind: 'ablate', point: 'layers.0.mlp' };
const add = { kind: 'add', point: 'layers.0.mlp_out', vector: [1, -1], coefficient: 0.5 };
const scale = { kind: 'scale', point: 'layers.0.mlp_out', feature: 0, multiplier: -2 };
const clamp = { kind: 'clamp', point: 'layers.0.mlp_out', feature: 1, value: 0 };
const patch = { kind: 'patch', point: 'layers.0.resid_pre', source_prompt: 'A café' };
const message = (type: number, meta: Record<string, unknown>) => ({ type: type === MSG.control ? 'control' : 'controls', ...meta }) as Message;
function accepts(type: number, meta: Record<string, unknown>) {
  const bytes = encodeFrame(type, meta);
  const decoded = decode(bytes);
  assert.deepEqual(plain(decoded), message(type, meta));
  assert.deepEqual(encode(decoded as Message), bytes);
  assert.deepEqual(encode(message(type, meta)), bytes);
}
function refuses(type: number, meta: Record<string, unknown>) {
  assert.throws(() => decode(encodeFrame(type, meta)), MessageError);
  assert.throws(() => encode(message(type, meta)), MessageError);
}
function specCase(spec: Record<string, unknown>, accepted: boolean) {
  const check = accepted ? accepts : refuses;
  check(MSG.control, request({ set: spec }));
  check(MSG.controls, controls([{ ...spec, state: 'active' }]));
}
function rawFrame(type: number, json: string): Uint8Array {
  const meta = new TextEncoder().encode(json);
  const bytes = new Uint8Array(HEADER_BYTES + meta.length);
  bytes.set(encodeFrame(type, {}).subarray(0, HEADER_BYTES));
  const view = new DataView(bytes.buffer);
  view.setUint32(12, bytes.length, true);
  view.setUint32(16, meta.length, true);
  bytes.set(meta, HEADER_BYTES);
  return bytes;
}
function rawSpec(spec: Record<string, unknown>, token: string, accepted: boolean) {
  for (const [type, meta] of [
    [MSG.control, request({ set: spec })], [MSG.controls, controls([{ ...spec, state: 'active' }])],
  ] as const) {
    const bytes = rawFrame(type, canonicalJson(meta).replaceAll('"NUMBER"', token));
    if (accepted) assert.deepEqual(encode(decode(bytes) as Message), bytes);
    else assert.throws(() => decode(bytes), MessageError, token);
  }
}

test('Shared M8 Intervention Contract', () => {
  const cases = JSON.parse(readFileSync(join(fixtures, 'm8_interventions.json'), 'utf8'));
  assert.equal(cases.length, 23);
  for (const item of cases) specCase(item.specification, item.accepted);
  const typed: InterventionSpec[] = [
    { point: 'layers.0.mlp', op: 'ablate' }, { point: 'layers.0.mlp', kind: 'ablate' },
    { point: 'layers.0.mlp_out', kind: 'add', vector: [1], coefficient: 0 },
    { point: 'layers.0.mlp_out', kind: 'scale', feature: 0, multiplier: 1 },
    { point: 'layers.0.mlp_out', kind: 'clamp', feature: 0, value: 0 },
    { point: 'layers.0.resid_pre', kind: 'patch', source_prompt: 'AB' },
  ];
  for (const spec of typed) specCase(spec, true);
});

test('Intervention Discriminators and Exact Fields', () => {
  for (const base of [ablate, add, scale, clamp, patch, { ...ablate, kind: 'steer' }]) {
    const { kind, ...rest } = base;
    for (const tags of [{ kind }, { op: kind }, { op: kind, kind }, { op: null, kind }, { op: kind, kind: null }])
      specCase({ ...rest, ...tags }, true);
    for (const tags of [{}, { kind: null }, { op: null }, { op: kind, kind: 'unknown' }, { kind: false }, { kind: 'toString' }, { kind: '__proto__' }])
      specCase({ ...rest, ...tags }, false);
    for (const key of ['unknown', '__proto__', 'state', 'since_step', 'error', 'error_code'])
      refuses(MSG.control, request({ set: { ...base, [key]: 1 } }));
    specCase({ ...base, unexpected: 1 }, false);
  }
  for (const [base, extra] of [[ablate, { positions: [] }], [add, { dictionary: 'x' }], [scale, { value: 0 }],
    [clamp, { multiplier: 1 }], [patch, { feature: 0 }]] as const) specCase({ ...base, ...extra }, false);
  for (const set of [null, [], true, 'ablate', {}]) refuses(MSG.control, request({ set }));
  for (const intervene of [null, [], {}, { clear_all: false }, { clear_all: 1 }, { clear_all: true, unknown: 1 },
    { set: ablate, clear: 'x' }, { clear: 'x', clear_all: true }, { set: ablate, clear_all: true }]) refuses(MSG.control, request(intervene));
  for (const intervene of [{ clear: 'layers.0.mlp' }, { clear_all: true }, { set: ablate, clear: null, clear_all: null },
    { set: null, clear: 'x', clear_all: null }, { set: null, clear: null, clear_all: true }]) accepts(MSG.control, request(intervene));
});

test('Intervention UTF-8 Text Bounds', () => {
  const fields: [Record<string, unknown>, string, number][] = [
    [ablate, 'point', 255], [scale, 'dictionary', 255], [clamp, 'dictionary', 255], [patch, 'source_prompt', 4096],
  ];
  for (const [base, key, limit] of fields) {
    for (const value of ['x'.repeat(limit), 'é'.repeat(Math.floor(limit / 2)) + 'x'.repeat(limit % 2), '\ufefftext\n', 'quote"\\text'])
      specCase({ ...base, [key]: value }, true);
    for (const value of ['', 'x'.repeat(limit + 1), 'é'.repeat(Math.floor(limit / 2) + 1), 'a\0b', '\ud800', '\udc00', null, true, 1])
      specCase({ ...base, [key]: value }, false);
  }
  for (const clear of ['x'.repeat(255), 'é'.repeat(127) + 'x']) accepts(MSG.control, request({ clear }));
  for (const clear of ['', 'x'.repeat(256), 'é'.repeat(128), '\0', '\ud800', 1]) refuses(MSG.control, request({ clear }));
  const bytes = encodeFrame(MSG.control, request({ set: patch }));
  const index = bytes.findIndex((byte, i) => byte === 0xc3 && bytes[i + 1] === 0xa9);
  assert.ok(index >= 0);
  bytes[index + 1] = 0x28;
  assert.throws(() => decode(bytes));
});

test('Intervention Schedules and Lifecycle Fields', () => {
  for (const spec of [{ ...ablate, from_step: 0, until_step: Number.MAX_SAFE_INTEGER },
    { ...ablate, from_step: null, until_step: null }, { ...ablate, until_step: 0 }]) specCase(spec, true);
  for (const key of ['from_step', 'until_step']) {
    for (const value of [-1, 2 ** 53, 0.5, true, '1']) specCase({ ...ablate, [key]: value }, false);
    for (const token of ['1.0', '1e0', '9007199254740992', '9007199254740992.0']) rawSpec({ ...ablate, [key]: 'NUMBER' }, token, false);
  }
  for (const until_step of [0, 1]) specCase({ ...ablate, from_step: 1, until_step }, false);
  for (const state of ['pending', 'active', 'ended'])
    accepts(MSG.controls, controls([{ ...patch, state, since_step: Number.MAX_SAFE_INTEGER, error: '', error_code: 'intervene.hook_failed' }]));
  accepts(MSG.controls, controls([{ ...ablate, state: 'ended', since_step: null, error: 'é'.repeat(512), error_code: 'é'.repeat(31) + 'x' }]));
  for (const extra of [{ state: null }, { state: 'queued' }, { since_step: -1 }, { since_step: 2 ** 53 }, { since_step: true },
    { error: null }, { error: 'x'.repeat(1025) }, { error: '\0' }, { error: '\ud800' },
    { error_code: '' }, { error_code: null }, { error_code: 'é'.repeat(32) }, { error_code: '\0' }])
    refuses(MSG.controls, controls([{ ...ablate, state: 'active', ...extra }]));
  refuses(MSG.controls, controls([ablate]));
  for (const token of ['1.0', '1e0']) {
    const json = canonicalJson(controls([{ ...ablate, state: 'active', since_step: 'NUMBER' }])).replace('"NUMBER"', token);
    assert.throws(() => decode(rawFrame(MSG.controls, json)), MessageError);
  }
});

test('Intervention Positions and Feature IDs', () => {
  for (const base of [add, scale, clamp, patch]) {
    for (const positions of [[], [0, 31], [31, 0], Array.from({ length: 32 }, (_, i) => i)]) specCase({ ...base, positions }, true);
    for (const positions of [null, 0, [0, 0], [-1], [32], [true], [0.5], ['1'], Array.from({ length: 33 }, (_, i) => i)])
      specCase({ ...base, positions }, false);
    for (const token of ['1.0', '1e0']) rawSpec({ ...base, positions: ['NUMBER'] }, token, false);
  }
  for (const [base, key, high] of [[patch, 'source_position', 31], [scale, 'feature', 2147483647], [clamp, 'feature', 2147483647]] as const) {
    for (const value of [0, high]) specCase({ ...base, [key]: value }, true);
    for (const value of [-1, high + 1, true, null, '1', 0.5]) specCase({ ...base, [key]: value }, false);
    for (const token of ['1.0', '1e0', '1.0000000000000001']) rawSpec({ ...base, [key]: 'NUMBER' }, token, false);
  }
});

test('Intervention Vectors and Numeric Tokens', () => {
  assert.deepEqual(INTERVENTION_LIMITS, { maxVector: 16384, maxPositions: 32, maxBytes: 262144 });
  specCase({ ...add, vector: Array(16384).fill(0) }, true);
  for (const vector of [[], Array(16385).fill(0), null, 1, [true], ['1'], [null], [2 ** 53]]) specCase({ ...add, vector }, false);
  const numbers: [Record<string, unknown>, string][] = [[add, 'coefficient'], [scale, 'multiplier'], [clamp, 'value']];
  for (const [base, key] of numbers) {
    for (const value of [0, -1.25, Number.MAX_SAFE_INTEGER, -Number.MAX_SAFE_INTEGER, 1e100]) specCase({ ...base, [key]: value }, true);
    for (const value of [null, true, '1', 2 ** 53, -(2 ** 53)]) specCase({ ...base, [key]: value }, false);
    for (const token of ['1.0', '-0.0', '1e-05', '1e+100', '9007199254740992.0', '-9007199254740992.0']) rawSpec({ ...base, [key]: 'NUMBER' }, token, true);
    for (const token of ['9007199254740992', '-9007199254740992', '1e309', '-1e309']) rawSpec({ ...base, [key]: 'NUMBER' }, token, false);
  }
  for (const token of ['1.0', '-0.0', '1e-05', '1e+100', '9007199254740992.0']) rawSpec({ ...add, vector: ['NUMBER'] }, token, true);
  for (const token of ['9007199254740992', '1e309', '-1e309']) rawSpec({ ...add, vector: ['NUMBER'] }, token, false);
  for (const bad of [NaN, Infinity, -Infinity]) {
    for (const spec of [{ ...add, vector: [bad] }, ...numbers.map(([base, key]) => ({ ...base, [key]: bad }))]) {
      assert.throws(() => encode(message(MSG.control, request({ set: spec }))), MessageError);
      assert.throws(() => encode(message(MSG.controls, controls([{ ...spec, state: 'active' }]))), MessageError);
    }
  }
});

test('Intervention Canonical Byte Limit', () => {
  for (const lifecycle of [false, true]) {
    const spec = { ...add, point: 'x', vector: Array(16370).fill('NUMBER'), ...(lifecycle ? { state: 'active' } : {}) };
    const json = () => canonicalJson(spec).replaceAll('"NUMBER"', '1234567890123.0');
    const padding = 262144 - Buffer.byteLength(json());
    assert.ok(padding >= 0 && padding < 255);
    spec.point += 'x'.repeat(padding);
    const bytes = () => lifecycle ? rawFrame(MSG.controls, `{"actions":[],"interventions":{"active":[${json()}],"points":[]},"params":[],"run":{"state":"none"}}`)
      : rawFrame(MSG.control, `{"intervene":{"set":${json()}},"op":"intervene","request_id":1}`);
    assert.equal(Buffer.byteLength(json()), 262144);
    assert.deepEqual(encode(decode(bytes()) as Message), bytes());
    spec.point += 'x';
    assert.equal(Buffer.byteLength(json()), 262145);
    assert.throws(() => decode(bytes()), MessageError);
  }
});

test('Intervention Metadata Collection Limits', () => {
  const point = { point: 'layers.0.mlp_out', module: 'layers.0', kind: 'module', probe: 'mlp_out', layer: 0, pattern: 'softmax' };
  const active = { ...clamp, state: 'active' };
  accepts(MSG.controls, controls(Array(MAX_INTERVENTIONS).fill(active), Array(MAX_POINTS_LISTED).fill(point)));
  refuses(MSG.controls, controls(Array(MAX_INTERVENTIONS + 1).fill(active)));
  refuses(MSG.controls, controls([], Array(MAX_POINTS_LISTED + 1).fill(point)));
  for (const interventions of [[], {}, { points: [], active: null }, { points: null, active: [] }])
    refuses(MSG.controls, { ...controls([]), interventions });
});
