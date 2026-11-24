import assert from 'node:assert/strict';
import test, { type TestContext } from 'node:test';
import { SocketClient } from '../src/client/socket.js';
import { decode, encode, MessageError } from '../src/protocol/codec.js';
import { CAP, type ControlReplyMsg, type InterpretationRequest, type Message } from '../src/protocol/messages.js';

// Exercise the public client through encoded peer frames without a network.
class PeerSocket {
  static readonly OPEN = 1;
  static current: PeerSocket;
  readyState = 1;
  binaryType = '';
  onopen?: () => void;
  onmessage?: (event: { data: ArrayBuffer }) => void;
  onclose?: (event: { code: number }) => void;
  sent: Message[] = [];
  failSend = false;

  constructor() { PeerSocket.current = this; }
  send(bytes: Uint8Array): void {
    if (this.failSend) throw new Error('Socket send failed');
    this.sent.push(decode(bytes) as Message);
  }
  close(code: number): void { this.readyState = 3; this.onclose?.({ code }); }
  receive(message: Message): void {
    this.onmessage?.({ data: Uint8Array.from(encode(message)).buffer });
  }
}

const caps = [CAP.controls, CAP.interpretation];
function connected(t: TestContext, serverCaps: string[] = caps, sessionCaps: string[] = caps, hub = true) {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'WebSocket');
  Object.defineProperty(globalThis, 'WebSocket', { configurable: true, value: PeerSocket });
  const client = new SocketClient('ws://localhost/v3/stream');
  t.after(() => {
    client.close();
    if (original) Object.defineProperty(globalThis, 'WebSocket', original);
    else Reflect.deleteProperty(globalThis, 'WebSocket');
  });
  client.connect();
  const peer = PeerSocket.current;
  peer.onopen?.();
  const session = { id: 'fixture', script: 'fixture.py', pid: 1, start_unix_ms: 1, state: 'live' as const, views: 0, capabilities: sessionCaps };
  peer.receive({ type: 'welcome', role: hub ? 'hub' : 'producer', capabilities: serverCaps, control: true, minor: 2, session });
  if (hub) peer.receive({ type: 'sessions', sessions: [session] });
  peer.receive({ type: 'views', session: session.id, views: [] });
  peer.receive({ type: 'controls', run: { state: 'running' }, params: [], actions: [] });
  assert.equal(peer.sent.some((m) => m.type === 'error'), false);
  assert.equal(client.controlAccess().allowed, true);
  return { client, peer };
}

test('Socket Interpretation Handshake and Payloads', (t) => {
  const { client, peer } = connected(t);
  const hello = peer.sent[0];
  assert.equal(hello.type, 'hello');
  if (hello.type === 'hello') assert.ok(hello.capabilities.includes(CAP.interpretation));
  const requests: InterpretationRequest[] = [
    { version: 1, method: 'trace.status', params: {} },
    { version: 1, method: 'trace.run', params: { prompt: 'é\nnext', max_targets: 8 } },
    { version: 1, method: 'feature.examples', params: { node: 3, limit: 2 } },
    { version: 1, method: 'trace.read', params: { node_offset: 2, node_limit: 8 } },
  ];
  for (const interpretation of requests) {
    const id = client.control({ op: 'interpret', interpretation });
    assert.ok(id && id > 0);
    const sent = peer.sent.at(-1);
    assert.equal(sent?.type, 'control');
    if (sent?.type === 'control') {
      assert.equal(sent.request_id, id);
      assert.deepEqual(sent.interpretation, interpretation);
    }
  }
});

test('Socket Interpretation Replies Retain Queued Callbacks Until Completion', (t) => {
  const { client, peer } = connected(t);
  const replies: ControlReplyMsg[] = [];
  const id = client.control({ op: 'interpret', interpretation: { version: 1, method: 'trace.status', params: {} } }, (r) => replies.push(r))!;
  peer.receive({ type: 'control_reply', request_id: id, op: 'interpret', status: 'queued' });
  peer.receive({ type: 'control_reply', request_id: id, op: 'interpret', status: 'ok', result: { state: 'done', nodes: [1, 2] } });
  peer.receive({ type: 'control_reply', request_id: id, op: 'interpret', status: 'ok', result: {} });
  assert.deepEqual(replies.map((r) => r.status), ['queued', 'ok']);
  assert.deepEqual(replies[1].result, { state: 'done', nodes: [1, 2] });
});

for (const [label, serverCaps, sessionCaps, hub] of [
  ['Hub Capability', [CAP.controls], caps, true],
  ['Producer Capability', caps, [CAP.controls], true],
  ['Direct Producer Capability', [CAP.controls], caps, false],
] as const) {
  test(`Socket Interpretation Requires ${label}`, (t) => {
    const { client, peer } = connected(t, [...serverCaps], [...sessionCaps], hub);
    const before = peer.sent.length;
    assert.equal(client.control({ op: 'interpret', interpretation: { version: 1, method: 'trace.status', params: {} } }), null);
    assert.equal(peer.sent.length, before);
    assert.ok(client.control({ op: 'pause' }));
    assert.equal(peer.sent.at(-1)?.type, 'control');
  });
}

test('Socket Direct Producer Interpretation', (t) => {
  const { client, peer } = connected(t, caps, [], false);
  const interpretation = { version: 1, method: 'trace.status', params: {} } as const;
  assert.ok(client.control({ op: 'interpret', interpretation }));
  const sent = peer.sent.at(-1);
  if (sent?.type === 'control') assert.deepEqual(sent.interpretation, interpretation);
  else assert.fail('Expected a control frame');
});

test('Socket Encoding Failure Removes the Callback and Permits the Next Request', (t) => {
  const { client, peer } = connected(t);
  let calls = 0;
  const before = peer.sent.length;
  assert.throws(() => client.control({ op: 'interpret' }, () => calls++), MessageError);
  assert.equal(peer.sent.length, before);
  // A reply for the rejected request must not reach its discarded callback.
  peer.receive({ type: 'control_reply', request_id: 1, op: 'interpret', status: 'error', error: { code: 'bad_request' } });
  assert.equal(calls, 0);
  const id = client.control({ op: 'interpret', interpretation: { version: 1, method: 'trace.status', params: {} } }, () => calls++);
  assert.equal(id, 2);
  peer.receive({ type: 'control_reply', request_id: id!, op: 'interpret', status: 'ok', result: {} });
  assert.equal(calls, 1);
});

test('Socket Send Failure Removes the Callback', (t) => {
  const { client, peer } = connected(t);
  let calls = 0;
  peer.failSend = true;
  assert.throws(() => client.control({ op: 'pause' }, () => calls++), /Socket send failed/);
  peer.failSend = false;
  peer.receive({ type: 'control_reply', request_id: 1, op: 'pause', status: 'ok' });
  assert.equal(calls, 0);
});
