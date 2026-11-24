import assert from 'node:assert/strict';
import test, { type TestContext } from 'node:test';
import { SocketClient } from '../src/client/socket.js';
import { decode, encode, MessageError } from '../src/protocol/codec.js';
import { CAP, CLOSE, type ControlMsg, type ControlReplyMsg, type InterpretationRequest, type Message } from '../src/protocol/messages.js';

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
const reads: InterpretationRequest[] = [
  { version: 1, method: 'feature.dictionary', params: {} },
  { version: 1, method: 'feature.show', params: { node: 1 } },
  { version: 1, method: 'feature.examples', params: { layer: 0, feature: 1, limit: 2 } },
  { version: 1, method: 'trace.status', params: {} },
  { version: 1, method: 'trace.read', params: { node_offset: 0, node_limit: 8 } },
];
const read = { op: 'interpret', interpretation: reads[3] } as const;
function connected(t: TestContext, {
  serverCaps = caps, sessionCaps = caps, hub = true, control = true, controls = true, ready = true,
}: { serverCaps?: readonly string[]; sessionCaps?: readonly string[]; hub?: boolean; control?: boolean; controls?: boolean; ready?: boolean } = {}) {
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
  const session = { id: 'fixture', script: 'fixture.py', pid: 1, start_unix_ms: 1, state: 'live' as const, views: 0, capabilities: [...sessionCaps] };
  peer.receive({ type: 'welcome', role: hub ? 'hub' : 'producer', capabilities: [...serverCaps], control, minor: 2, session });
  if (hub) peer.receive({ type: 'sessions', sessions: [session] });
  if (ready) peer.receive({ type: 'views', session: session.id, views: [] });
  if (controls) peer.receive({ type: 'controls', run: { state: 'running' }, params: [], actions: [] });
  assert.equal(peer.sent.some((m) => m.type === 'error'), false);
  assert.equal(client.controlAccess().allowed, control && controls && ready
    && serverCaps.includes(CAP.controls) && (!hub || sessionCaps.includes(CAP.controls)));
  return { client, peer, session };
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

test('Socket Read-Only Queued Callbacks and Final Results', (t) => {
  const { client, peer } = connected(t, { control: false });
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
    const { client, peer } = connected(t, { serverCaps, sessionCaps, hub });
    const before = peer.sent.length;
    assert.equal(client.control({ op: 'interpret', interpretation: { version: 1, method: 'trace.status', params: {} } }), null);
    assert.equal(peer.sent.length, before);
    assert.ok(client.control({ op: 'pause' }));
    assert.equal(peer.sent.at(-1)?.type, 'control');
  });
}

test('Socket Direct Producer Interpretation', (t) => {
  const { client, peer } = connected(t, { sessionCaps: [], hub: false });
  const interpretation = { version: 1, method: 'trace.status', params: {} } as const;
  assert.ok(client.control({ op: 'interpret', interpretation }));
  const sent = peer.sent.at(-1);
  if (sent?.type === 'control') assert.deepEqual(sent.interpretation, interpretation);
  else assert.fail('Expected a control frame');
});

for (const hub of [false, true]) {
  for (const controls of [false, true]) {
    test(`Socket ${hub ? 'Hub' : 'Direct'} Read-Only Reads ${controls ? 'With' : 'Without'} Controls`, (t) => {
      const { client, peer } = connected(t, { hub, control: false, controls,
        serverCaps: controls ? caps : [CAP.interpretation],
        sessionCaps: controls ? caps : [CAP.interpretation] });
      // Check the public entry and the deferred-send gate independently.
      for (const method of ['control', 'sendControl'] as const) {
        for (const interpretation of reads) {
          let calls = 0;
          const id = client[method]({ op: 'interpret', interpretation }, () => calls++);
          assert.ok(id && id > 0);
          const sent = peer.sent.at(-1);
          if (sent?.type !== 'control') assert.fail('Expected a control frame');
          assert.equal(sent.request_id, id);
          assert.equal(sent.op, 'interpret');
          assert.deepEqual(sent.interpretation, interpretation);
          peer.receive({ type: 'control_reply', request_id: id, op: 'interpret', status: 'ok', result: {} });
          assert.equal(calls, 1);
        }
      }
      assert.equal(client.controlAccess().allowed, false);
    });
  }

  test(`Socket ${hub ? 'Hub' : 'Direct'} Read-Only Change Rejection`, (t) => {
    const { client, peer } = connected(t, { hub, control: false });
    const changes: Omit<ControlMsg, 'type' | 'request_id'>[] = [
      { op: 'interpret', interpretation: { version: 1, method: 'trace.run', params: { prompt: 'test' } } },
      { op: 'pause' }, { op: 'resume' }, { op: 'step', count: 1 },
      { op: 'set', name: 'rate', value: 1 }, { op: 'run', name: 'action' },
      { op: 'intervene', intervene: { clear_all: true } },
      { op: 'interpret' },
      // Future methods must not gain read permission by their prefix.
      { op: 'interpret', interpretation: { version: 1, method: 'feature.change', params: {} } as unknown as InterpretationRequest },
    ];
    const before = peer.sent.length;
    let calls = 0;
    for (const method of ['control', 'sendControl'] as const) {
      for (const change of changes) assert.equal(client[method](change, () => calls++), null);
    }
    assert.equal(peer.sent.length, before);
    peer.receive({ type: 'control_reply', request_id: 1, op: 'interpret', status: 'ok', result: {} });
    assert.equal(calls, 0);
    assert.equal(client.control(read), 1);
  });

  test(`Socket ${hub ? 'Hub' : 'Direct'} Read Readiness`, (t) => {
    const { client, peer, session } = connected(t, { hub, control: false, ready: false });
    const before = peer.sent.length;
    assert.equal(client.control(read), null);
    assert.equal(client['sendControl'](read), null);
    assert.equal(peer.sent.length, before);
    peer.receive({ type: 'views', session: session.id, views: [] });
    assert.ok(client.control(read));
    peer.readyState = 2;
    const sent = peer.sent.length;
    assert.equal(client.control(read), null);
    assert.equal(client['sendControl'](read), null);
    assert.equal(peer.sent.length, sent);
    peer.close(CLOSE.normal);
    assert.equal(client.control(read), null);
    assert.equal(client['sendControl'](read), null);
  });
}

for (const [label, serverCaps, sessionCaps, hub] of [
  ['Hub', [CAP.controls], caps, true],
  ['Hub Session', caps, [CAP.controls], true],
  ['Direct', [CAP.controls], caps, false],
] as const) {
  test(`Socket Read-Only ${label} Capability Rejection`, (t) => {
    const { client, peer } = connected(t, { serverCaps, sessionCaps, hub, control: false });
    const before = peer.sent.length;
    let calls = 0;
    for (const method of ['control', 'sendControl'] as const) {
      for (const interpretation of reads) {
        assert.equal(client[method]({ op: 'interpret', interpretation }, () => calls++), null);
      }
    }
    assert.equal(peer.sent.length, before);
    peer.receive({ type: 'control_reply', request_id: 1, op: 'interpret', status: 'ok', result: {} });
    assert.equal(calls, 0);
  });
}

test('Socket Read-Only Session Callback Isolation', (t) => {
  const { client, peer, session } = connected(t, { control: false });
  let calls = 0;
  const oldId = client.control(read, () => calls++)!;
  peer.receive({ type: 'views', views: [] });
  const before = peer.sent.length;
  assert.equal(client.control(read), null);
  assert.equal(client['sendControl'](read), null);
  assert.equal(peer.sent.length, before);
  peer.receive({ type: 'control_reply', request_id: oldId, op: 'interpret', status: 'ok', result: {} });
  assert.equal(calls, 0);
  const nextSession = { ...session, id: 'next' };
  peer.receive({ type: 'sessions', sessions: [nextSession] });
  peer.receive({ type: 'views', session: nextSession.id, views: [] });
  const id = client.control(read, () => calls++)!;
  assert.ok(id > oldId);
  peer.receive({ type: 'control_reply', request_id: oldId, op: 'interpret', status: 'ok', result: {} });
  peer.receive({ type: 'control_reply', request_id: id, op: 'interpret', status: 'error', error: { code: 'refused' } });
  peer.receive({ type: 'control_reply', request_id: id, op: 'interpret', status: 'ok', result: {} });
  assert.equal(calls, 1);
  peer.receive({ type: 'sessions', sessions: [{ ...nextSession, state: 'ended' }] });
  assert.equal(client.control(read), null);
  assert.equal(client['sendControl'](read), null);
});

test('Socket Read-Only Attached Session Capabilities', (t) => {
  const { client, peer, session } = connected(t, { control: false });
  peer.receive({ type: 'views', session: 'next', views: [] });
  const before = peer.sent.length;
  assert.equal(client.control(read), null);
  assert.equal(client['sendControl'](read), null);
  peer.receive({ type: 'sessions', sessions: [{ ...session, id: 'next', capabilities: [CAP.controls] }] });
  assert.equal(client.control(read), null);
  assert.equal(client['sendControl'](read), null);
  assert.equal(peer.sent.length, before);
  peer.receive({ type: 'sessions', sessions: [{ ...session, id: 'next' }] });
  assert.ok(client.control(read));
  assert.ok(client['sendControl'](read));
});

test('Socket Read-Only Encoding and Send Callback Cleanup', (t) => {
  const { client, peer } = connected(t, { control: false });
  let calls = 0;
  assert.throws(() => client.control({ op: 'interpret', interpretation: {
    version: 1, method: 'feature.show', params: { node: -1 },
  } }, () => calls++), MessageError);
  peer.failSend = true;
  assert.throws(() => client.control(read, () => calls++), /Socket send failed/);
  peer.failSend = false;
  for (const id of [1, 2]) {
    peer.receive({ type: 'control_reply', request_id: id, op: 'interpret', status: 'ok', result: {} });
  }
  assert.equal(calls, 0);
  assert.equal(client.control(read, () => calls++), 3);
  client.close();
  peer.receive({ type: 'control_reply', request_id: 3, op: 'interpret', status: 'ok', result: {} });
  assert.equal(calls, 0);
  assert.equal(client.control(read), null);
  assert.equal(client['sendControl'](read), null);
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
