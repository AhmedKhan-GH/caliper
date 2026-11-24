import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { canonicalJson, decode, decodeFrame, encode, encodeFrame, HEADER_BYTES, MessageError } from '../src/protocol/codec.js';
import { CAP, INTERPRET_LIMITS, MSG, type InterpretationRequest, type Message } from '../src/protocol/messages.js';

const fixtures = process.env.CALIPER_WIRE_FIXTURES ?? fileURLToPath(new URL('../../tests/fixtures/wire_v3/', import.meta.url));
const plain = (value: unknown) => JSON.parse(JSON.stringify(value));
const request = (method: string, params: unknown = {}, extra = {}) => ({
  request_id: 1, op: 'interpret', interpretation: { version: 1, method, params, ...extra },
});
const reply = (result: unknown) => ({ request_id: 1, op: 'interpret', status: 'ok', result });
const accepts = (type: number, meta: Record<string, unknown>) => decode(encodeFrame(type, meta));
const refuses = (type: number, meta: Record<string, unknown>) =>
  assert.throws(() => accepts(type, meta), (error: unknown) => error instanceof MessageError && error.code === 'bad_request');

// This preserves numeric tokens that JSON.parse would convert to JS numbers.
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

test('Shared Control Fixture Bytes', () => {
  let valid = 0;
  for (const directory of ['v3_1', 'v3_2']) {
    for (const name of readdirSync(join(fixtures, directory)).filter((name) => /^(controls?|hello|welcome).*\.bin$/.test(name))) {
      const bytes = readFileSync(join(fixtures, directory, name));
      const expected = JSON.parse(readFileSync(join(fixtures, directory, name.replace('.bin', '.json')), 'utf8'));
      assert.deepEqual(decodeFrame(bytes).meta, expected.meta, name);
      const message = decode(bytes);
      assert.notEqual(message.type, 'unknown', name);
      assert.deepEqual(encode(message as Message, expected.header.minor), new Uint8Array(bytes), name);
      valid++;
    }
  }
  const invalid = JSON.parse(readFileSync(join(fixtures, 'v3_2/invalid/expect.json'), 'utf8'));
  const names = Object.keys(invalid).filter((name) => /^controls?_/.test(name));
  for (const name of names)
    assert.throws(() => decode(readFileSync(join(fixtures, 'v3_2/invalid', name))),
      (error: unknown) => error instanceof MessageError && error.code === invalid[name].error, name);
  assert.ok(valid >= 18);
  assert.equal(names.length, 11);
});

test('Native Interpretation Metadata Cases', () => {
  // Read the native literals directly. Its frame_of helper canonicalizes them.
  const source = readFileSync(join(fixtures, '../../test_net_wire.cpp'), 'utf8');
  const cases = [...source.matchAll(/CHECK\(status\(CALIPER_NET_MSG_(CONTROL(?:_REPLY)?), R"\(([^\n]+)\)"\) == (valid|bad)\);/g)];
  assert.equal(cases.length, 11);
  for (const [, kind, json, status] of cases) {
    const type = kind === 'CONTROL' ? MSG.control : MSG.control_reply;
    const bytes = rawFrame(type, canonicalJson(JSON.parse(json)));
    if (status === 'valid') {
      const message = decode(bytes);
      assert.deepEqual(plain(message), { type: kind.toLowerCase(), ...JSON.parse(json) });
      assert.deepEqual(encode(message as Message), bytes);
    } else assert.throws(() => decode(bytes), MessageError, json);
  }
});

test('Interpretation Capabilities and Methods', () => {
  assert.equal(CAP.interpretation, 'interpretation');
  assert.equal(CAP.interventions, 'interventions');
  const capabilities = [CAP.controls, CAP.interpretation, CAP.interventions];
  for (const type of ['hello', 'welcome'] as const) {
    const message: Message = type === 'hello' ? { type, role: 'producer', capabilities }
      : { type, role: 'producer', capabilities, minor: 2 };
    const decoded = decode(encode(message, 2));
    assert.ok(decoded.type === 'hello' || decoded.type === 'welcome');
    assert.deepEqual(decoded.capabilities, capabilities);
  }
  const requests: InterpretationRequest[] = [
    { version: 1, method: 'feature.dictionary', params: {} },
    { version: 1, method: 'feature.show', params: { layer: 0, feature: 2 } },
    { version: 1, method: 'feature.show', params: { node: 511 } },
    { version: 1, method: 'feature.examples', params: { layer: 4095, feature: 2147483647, limit: 16 } },
    { version: 1, method: 'feature.examples', params: { node: 0, limit: 0 } },
    { version: 1, method: 'trace.status', params: {} },
    { version: 1, method: 'trace.validate', params: {} },
    { version: 1, method: 'patch.sweep', params: { clean_prompt: 'AB', corrupted_prompt: 'AC', metric: { positive_token: 1, negative_token: 2 } } },
    { version: 1, method: 'trace.run', params: { prompt: 'The capital', max_targets: 512, max_nodes: 512, max_edges: 8192 } },
    { version: 1, method: 'trace.read', params: {} },
    { version: 1, method: 'trace.read', params: { node_offset: 512, node_limit: 256, edge_offset: 8192, edge_limit: 256 } },
  ];
  for (const interpretation of requests) {
    const message: Message = { type: 'control', request_id: 1, op: 'interpret', interpretation };
    assert.deepEqual(plain(decode(encode(message))), message);
  }
  for (const method of ['unknown', 'toString', '__proto__', 'patch.sweep']) refuses(MSG.control, request(method));
  for (const extra of [{ version: true }, { version: 2 }, { version: null }, { extra: 1 }]) refuses(MSG.control, request('trace.status', {}, extra));
  for (const params of [null, [], true, { unknown: 1 }]) refuses(MSG.control, request('trace.status', params));
  for (const method of ['feature.show', 'feature.examples'])
    for (const params of [{}, { layer: 0 }, { feature: 0 }, { node: 0, layer: 0 }, { node: 0, feature: 0 }, { node: 0, layer: 0, feature: 0 }])
      refuses(MSG.control, request(method, params));
  for (const interpretation of [undefined, null, [], {}]) refuses(MSG.control, { request_id: 1, op: 'interpret', interpretation });
});

test('Interpretation Integer and Page Bounds', () => {
  const bounds: [string, string, number, number, Record<string, unknown>][] = [
    ['feature.show', 'layer', 0, 4095, { feature: 0 }],
    ['feature.show', 'feature', 0, 2147483647, { layer: 0 }],
    ['feature.show', 'node', 0, 511, {}],
    ['feature.examples', 'limit', 0, 16, { node: 0 }],
    ['trace.run', 'max_targets', 0, 512, { prompt: 'x' }],
    ['trace.run', 'max_nodes', 1, 512, { prompt: 'x' }],
    ['trace.run', 'max_edges', 0, 8192, { prompt: 'x' }],
    ['trace.read', 'node_offset', 0, 512, {}], ['trace.read', 'node_limit', 0, 256, {}],
    ['trace.read', 'edge_offset', 0, 8192, {}], ['trace.read', 'edge_limit', 0, 256, {}],
  ];
  for (const [method, key, low, high, base] of bounds) {
    for (const value of [low, high]) accepts(MSG.control, request(method, { ...base, [key]: value }));
    for (const value of [low - 1, high + 1, 0.5, true, null, '1']) refuses(MSG.control, request(method, { ...base, [key]: value }));
    for (const number of ['1.0', '1e0', '1.0000000000000001']) {
      const json = canonicalJson(request(method, { ...base, [key]: 'REPLACE' })).replace('"REPLACE"', number);
      assert.throws(() => decode(rawFrame(MSG.control, json)), MessageError);
    }
  }
  assert.throws(() => decode(rawFrame(MSG.control, canonicalJson(request('trace.status')).replace('"version":1', '"version":1.0'))), MessageError);
});

test('Interpretation Text and Key Bounds', () => {
  for (const prompt of ['x'.repeat(4096), 'é'.repeat(2048), '😀'.repeat(1024), '\ufefftext\n', 'quote"\\text']) {
    const message = { type: 'control', ...request('trace.run', { prompt }) } as Message;
    assert.deepEqual(plain(decode(encode(message))), message);
  }
  for (const prompt of ['', 'x'.repeat(4097), 'é'.repeat(2048) + 'x', 'a\0b', '\ud800', '\udc00']) {
    const meta = request('trace.run', { prompt });
    refuses(MSG.control, meta);
    assert.throws(() => encode({ type: 'control', ...meta } as Message), MessageError);
  }
  refuses(MSG.control, request('trace.run'));
  for (const key of ['a'.repeat(63), 'é'.repeat(31) + 'x', '__proto__']) accepts(MSG.control_reply, reply({ [key]: '' }));
  for (const key of ['', 'a'.repeat(64), 'é'.repeat(32), 'a\0b', '\ud800']) refuses(MSG.control_reply, reply({ [key]: 0 }));
  for (const value of ['x'.repeat(4097), 'é'.repeat(2049), '\0', '\ud800']) refuses(MSG.control_reply, reply({ value }));
  const malformed = encodeFrame(MSG.control, request('trace.run', { prompt: 'é' }));
  const index = malformed.findIndex((byte, i) => byte === 0xc3 && malformed[i + 1] === 0xa9);
  assert.ok(index >= 0);
  malformed[index + 1] = 0x28;
  assert.throws(() => decode(malformed));
});

test('Interpretation Reply States and Legacy Separation', () => {
  for (const meta of [
    { request_id: 1, op: 'interpret', status: 'queued' },
    reply({ nodes: [{ id: 'a', effect: -0.125 }], next: null }),
    { request_id: 1, op: 'interpret', status: 'error', error: { code: 'interpret.missing_dictionary', message: 'No dictionary.' } },
  ]) {
    const message = { type: 'control_reply', ...meta } as Message;
    assert.deepEqual(plain(decode(encode(message))), message);
  }
  for (const op of ['pause', 'resume', 'step', 'set', 'run', 'intervene']) {
    for (const interpretation of [null, {}]) refuses(MSG.control, { request_id: 1, op, name: 'x', value: 0, intervene: { clear_all: true }, interpretation });
    for (const result of [null, {}]) refuses(MSG.control_reply, { request_id: 1, op, status: 'ok', result });
    if (op !== 'run') refuses(MSG.control_reply, { request_id: 1, op, status: 'queued' });
  }
  for (const result of [null, [], 0, 'text']) refuses(MSG.control_reply, reply(result));
  refuses(MSG.control_reply, { request_id: 1, op: 'interpret', status: 'ok' });
  for (const status of ['queued', 'error'])
    for (const result of [null, {}]) refuses(MSG.control_reply, { ...reply(result), status, ...(status === 'error' ? { error: { code: 'refused' } } : {}) });
  for (const status of ['ok', 'queued']) refuses(MSG.control_reply, { ...reply({}), status, error: { code: 'refused' } });
  for (const error of [undefined, null, {}, { code: 1 }, { code: 'refused', message: 'x'.repeat(1025) }])
    refuses(MSG.control_reply, { request_id: 1, op: 'interpret', status: 'error', error });
  refuses(MSG.control_reply, { ...reply({}), message: 'x'.repeat(1025) });
  accepts(MSG.control_reply, { ...reply({}), message: 'x'.repeat(1024) });
});

test('Interpretation Result Structure Bounds', () => {
  assert.deepEqual(INTERPRET_LIMITS, { maxBytes: 262144, maxString: 4096, maxList: 512, maxFields: 64, maxDepth: 8, maxValues: 16384 });
  const fields = (count: number) => Object.fromEntries(Array.from({ length: count }, (_, i) => [`k${i}`, null]));
  accepts(MSG.control_reply, reply(fields(64)));
  refuses(MSG.control_reply, reply(fields(65)));
  accepts(MSG.control_reply, reply({ values: Array(512).fill(null) }));
  refuses(MSG.control_reply, reply({ values: Array(513).fill(null) }));
  const nested = (depth: number): Record<string, unknown> => depth === 1 ? { x: null } : { x: nested(depth - 1) };
  accepts(MSG.control_reply, reply(nested(8)));
  refuses(MSG.control_reply, reply(nested(9)));
  // Root + 32 arrays + 31 * 512 values + 479 values = 16384.
  const values = Object.fromEntries(Array.from({ length: 32 }, (_, i) => [`k${i}`, Array(i === 31 ? 479 : 512).fill(null)]));
  accepts(MSG.control_reply, reply(values));
  values.k31.push(null);
  refuses(MSG.control_reply, reply(values));
  const bytes: Record<string, string> = Object.fromEntries(Array.from({ length: 64 }, (_, i) => [`k${i}`, 'x'.repeat(4096)]));
  bytes.k0 = bytes.k0.slice(0, 4096 - (Buffer.byteLength(canonicalJson(bytes)) - 262144));
  assert.equal(Buffer.byteLength(canonicalJson(bytes)), 262144);
  accepts(MSG.control_reply, reply(bytes));
  bytes.k0 += 'x';
  refuses(MSG.control_reply, reply(bytes));
  for (const value of [Number.MAX_SAFE_INTEGER, -Number.MAX_SAFE_INTEGER, 0.25, 1e100]) accepts(MSG.control_reply, reply({ value }));
  for (const value of [2 ** 53, -(2 ** 53)]) refuses(MSG.control_reply, reply({ value }));
  const floatReply = decode(rawFrame(MSG.control_reply, '{"request_id":1,"op":"interpret","status":"ok","result":{"value":9007199254740992.0}}'));
  assert.deepEqual(decode(encode(floatReply as Message)), floatReply);
  for (const value of ['1e309', '-1e309'])
    assert.throws(() => decode(rawFrame(MSG.control_reply, `{"request_id":1,"op":"interpret","status":"ok","result":{"value":${value}}}`)), MessageError);
  for (const result of [{ value: Infinity }, { value: NaN }, { value: 2 ** 53 }, { value: '\0' }])
    assert.throws(() => encode({ type: 'control_reply', ...reply(result) } as Message), MessageError);
});

test('Interpretation Canonical Numeric Bytes', () => {
  for (const [token, expected] of [
    ['1.0', '1.0'], ['-0.0', '-0.0'], ['1e-5', '1e-05'], ['1e16', '1e+16'],
    ['9007199254740992.0', '9007199254740992.0'], ['1e100', '1e+100'], ['1e-100', '1e-100'],
  ]) {
    const metadata = (value: string) => `{"op":"interpret","request_id":1,"result":{"value":${value}},"status":"ok"}`;
    const message = decode(rawFrame(MSG.control_reply, metadata(token)));
    assert.deepEqual(encode(message as Message), rawFrame(MSG.control_reply, metadata(expected)));
  }
  for (const token of ['1.0', '1e-05', '1e+16']) {
    const padding: Record<string, unknown> = Object.fromEntries(Array.from({ length: 64 }, (_, i) => [`k${i}`, 'x'.repeat(4096)]));
    padding.k63 = { text: 'x'.repeat(4096), value: 'FLOAT' };
    const resultJson = () => canonicalJson(padding).replace('"FLOAT"', token);
    padding.k0 = (padding.k0 as string).slice(0, 4096 - (Buffer.byteLength(resultJson()) - 262144));
    assert.equal(Buffer.byteLength(resultJson()), 262144);
    const frame = () => rawFrame(MSG.control_reply, `{"op":"interpret","request_id":1,"result":${resultJson()},"status":"ok"}`);
    const bytes = frame();
    assert.deepEqual(encode(decode(bytes) as Message), bytes);
    padding.k0 += 'x';
    assert.throws(() => decode(frame()), MessageError);
  }
});

test('Legacy Control Bounds', () => {
  const controls = { run: { state: 'none' }, params: [], actions: [] };
  const params = Array.from({ length: 64 }, (_, i) => ({ name: `p${i}`, type: 'number', value: 0 }));
  const actions = Array.from({ length: 64 }, (_, i) => ({ name: `a${i}`, running: false, queued: false }));
  accepts(MSG.controls, { ...controls, params, actions });
  refuses(MSG.controls, { ...controls, params: [...params, { ...params[0], name: 'extra' }] });
  refuses(MSG.controls, { ...controls, actions: [...actions, { ...actions[0], name: 'extra' }] });
  refuses(MSG.controls, { ...controls, params: [params[0], params[0]] });
  refuses(MSG.controls, { ...controls, actions: [actions[0], actions[0]] });
  for (const value of [-1, 2 ** 32, true, 0.5]) {
    refuses(MSG.control, { request_id: value, op: 'pause' });
    refuses(MSG.control_reply, { request_id: value, op: 'interpret', status: 'queued' });
  }
  for (const count of [0, -1, 2 ** 32, true, 0.5]) refuses(MSG.control, { request_id: 1, op: 'step', count });
  for (const op of ['set', 'run']) refuses(MSG.control, { request_id: 1, op, value: 1 });
  for (const op of ['unknown', 'clamp', 'patch']) refuses(MSG.control, { request_id: 1, op });
  refuses(MSG.control, { request_id: 1, op: 'intervene', intervene: { set: { point: 'layer', kind: 'clamp', value: 0 } } });
});

test('Native M8 Interpretation Literals', () => {
  const source = readFileSync(join(fixtures, '../../test_net_wire.cpp'), 'utf8');
  const cases = [...source.matchAll(/CHECK(_FALSE)?\(accepted\(R"\(([^\n]+)\)"\)\);/g)];
  assert.equal(cases.length, 5);
  for (const [, invalid, json] of cases) {
    const interpretation = JSON.parse(json);
    const message = { type: 'control', request_id: 1, op: 'interpret', interpretation } as Message;
    const bytes = rawFrame(MSG.control, canonicalJson({ request_id: 1, op: 'interpret', interpretation }));
    if (invalid) {
      assert.throws(() => decode(bytes), MessageError);
      assert.throws(() => encode(message), MessageError);
    } else {
      assert.deepEqual(plain(decode(bytes)), message);
      assert.deepEqual(encode(decode(bytes) as Message), bytes);
      assert.deepEqual(encode(message), bytes);
    }
  }
});

test('M8 Interpretation Command Fields', () => {
  const sweep = { clean_prompt: 'AB', corrupted_prompt: 'AC', metric: { positive_token: 1, negative_token: 2 } };
  const invalid = (method: string, params: unknown) => {
    const meta = request(method, params);
    refuses(MSG.control, meta);
    assert.throws(() => encode({ type: 'control', ...meta } as Message), MessageError);
  };
  for (const params of [null, [], true, { prompt: 'AB' }, { max_targets: 10 }, { limit: 10 }, { metric: sweep.metric }]) invalid('trace.validate', params);
  for (const key of Object.keys(sweep)) {
    const params = { ...sweep } as Record<string, unknown>;
    delete params[key];
    invalid('patch.sweep', params);
  }
  for (const key of ['prompt', 'positions', 'batch_size', 'layer', 'max_nodes', '__proto__']) invalid('patch.sweep', { ...sweep, [key]: 1 });
  for (const metric of [null, [], true, {}, { positive_token: 1 }, { negative_token: 2 },
    { positive_token: 1, negative_token: 2, extra: 0 }, { positive_token: 1, unknown: 2 },
    { positive_token: 1, negative_token: 1 }]) invalid('patch.sweep', { ...sweep, metric });
  for (const key of ['positive_token', 'negative_token']) {
    const other = key === 'positive_token' ? 'negative_token' : 'positive_token';
    for (const value of [0, 2147483647]) accepts(MSG.control, request('patch.sweep', { ...sweep, metric: { [key]: value, [other]: 1 } }));
    for (const value of [-1, 2147483648, 2 ** 53, null, true, '1', 0.5]) invalid('patch.sweep', { ...sweep, metric: { [key]: value, [other]: 2 } });
    for (const token of ['1.0', '1e0', '1.0000000000000001', '1e309']) {
      const meta = request('patch.sweep', { ...sweep, metric: { [key]: 'NUMBER', [other]: 2 } });
      assert.throws(() => decode(rawFrame(MSG.control, canonicalJson(meta).replace('"NUMBER"', token))), MessageError);
    }
  }
  for (const key of ['clean_prompt', 'corrupted_prompt']) {
    for (const value of ['x'.repeat(4096), 'é'.repeat(2048), '😀'.repeat(1024), '\ufeffAB\n']) {
      const message = { type: 'control', ...request('patch.sweep', { ...sweep, [key]: value }) } as Message;
      assert.deepEqual(plain(decode(encode(message))), message);
    }
    for (const value of ['', 'x'.repeat(4097), 'é'.repeat(2048) + 'x', 'a\0b', '\ud800', '\udc00', null, 1]) invalid('patch.sweep', { ...sweep, [key]: value });
  }
  // Token counts belong to execution. The wire contract accepts bounded text.
  accepts(MSG.control, request('patch.sweep', { ...sweep, clean_prompt: 'A', corrupted_prompt: 'A longer prompt' }));
});

test('M8 Interpretation Result Contracts', () => {
  // These field names follow _validate.py and _sweep.py. Wire results share the M7 object bounds.
  const results = [
    { trace_id: 'fixture', token: 1, position: 2, baseline_centered_logit: 0.5, count: 1, required: 10,
      spearman: null, sign_agreement: 1, correct_signs: 1, passed: false,
      reasons: ['insufficient_active_features', 'zero_variance', 'fewer_than_eight_correct_signs'],
      measurements: [{ layer: 0, position: 1, feature: 2, predicted: -0.25, measured: -0.125, sign_match: true }] },
    { clean_metric: 0.75, corrupted_metric: -0.25, normalizer: 1, recovery: [[0, -1.5], [1, 2.25]],
      layers: 2, positions: 2, binding: 'live:patch.grid' },
    {},
  ];
  for (const result of results) {
    const message = { type: 'control_reply', ...reply(result) } as Message;
    assert.deepEqual(plain(decode(encode(message))), message);
  }
  for (const status of ['queued', 'error'] as const) {
    const message: Message = { type: 'control_reply', request_id: 1, op: 'interpret', status,
      ...(status === 'error' ? { error: { code: 'interpret.unsupported', message: 'No execution handler.' } } : {}) };
    assert.deepEqual(plain(decode(encode(message))), message);
  }
  // No command-specific result cap can replace the shared native bounds.
  accepts(MSG.control_reply, reply({ recovery: [Array(512).fill(0)] }));
  refuses(MSG.control_reply, reply({ recovery: [Array(513).fill(0)] }));
  for (const value of [NaN, Infinity, -Infinity]) {
    assert.throws(() => encode({ type: 'control_reply', ...reply({ spearman: value }) } as Message), MessageError);
    assert.throws(() => encode({ type: 'control_reply', ...reply({ recovery: [[value]] }) } as Message), MessageError);
  }
});
