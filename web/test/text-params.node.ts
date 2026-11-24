import assert from 'node:assert/strict';
import test from 'node:test';
import { decode, encode, encodeFrame } from '../src/protocol/codec.js';
import { MSG, PARAM_TEXT_MAX, LIMITS_31, type Message } from '../src/protocol/messages.js';
import { isParamText } from '../src/protocol/param-text.js';
import './interpretation.node.js';

test('Free-text values preserve exact UTF-8 bytes through metadata and controls', () => {
  for (const value of ['', 'x'.repeat(1025), 'x'.repeat(4096), 'é'.repeat(2048), '😀'.repeat(1024), '\ufefftext\n']) {
    assert.equal(isParamText(value), true);
    const messages: Message[] = [
      { type: 'controls', run: { state: 'running' }, params: [{ name: 'prompt', type: 'text', value, pending: value }], actions: [] },
      { type: 'control', request_id: 1, op: 'set', name: 'prompt', value },
      { type: 'control_reply', request_id: 1, op: 'set', status: 'ok', name: 'prompt', value },
    ];
    for (const message of messages) assert.deepEqual(JSON.parse(JSON.stringify(decode(encode(message)))), message);
  }
});

test('Free-text validation refuses excess bytes, NUL, and unpaired surrogates', () => {
  for (const value of ['x'.repeat(4097), 'é'.repeat(2048) + 'x', '😀'.repeat(1024) + 'x', 'a\0b', '\ud800', '\udc00']) {
    assert.equal(isParamText(value), false);
    assert.throws(() => encode({ type: 'control', request_id: 1, op: 'set', name: 'prompt', value }));
    assert.throws(() => decode(encodeFrame(MSG.control, { request_id: 1, op: 'set', name: 'prompt', value })));
    assert.throws(() => decode(encodeFrame(MSG.control_reply, { request_id: 1, op: 'set', status: 'ok', value })));
    for (const param of [{ name: 'prompt', type: 'text', value }, { name: 'prompt', type: 'text', value: 'ok', pending: value }]) {
      assert.throws(() => decode(encodeFrame(MSG.controls, { run: { state: 'none' }, params: [param], actions: [] })));
    }
  }
});

test('Legacy help and choice limits remain unchanged', () => {
  assert.equal(PARAM_TEXT_MAX, 4096);
  assert.equal(LIMITS_31.textMax, 1024);
  for (const param of [
    { name: 'prompt', type: 'text', value: '', help: 'x'.repeat(1025) },
    { name: 'choice', type: 'choice', value: 'x'.repeat(64), choices: ['x'.repeat(64)] },
    { name: 'x'.repeat(64), type: 'text', value: '' },
  ]) assert.throws(() => decode(encodeFrame(MSG.controls, { run: { state: 'none' }, params: [param], actions: [] })));
});


test('Malformed UTF-8 bytes are refused before parameter metadata', () => {
  const frame = encodeFrame(MSG.control, { request_id: 1, op: 'set', name: 'prompt', value: 'é' });
  const first = frame.findIndex((byte, index) => byte === 0xc3 && frame[index + 1] === 0xa9);
  assert.ok(first >= 0);
  frame[first + 1] = 0x28;
  assert.throws(() => decode(frame));
});
