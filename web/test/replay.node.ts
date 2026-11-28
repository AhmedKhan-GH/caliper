// Unit tests for static recording parsing and ReplaySource timeline/peer logic.

import assert from 'node:assert/strict';
import test from 'node:test';
import { isStreamUrl, parseRecording, ReplaySource } from '../src/client/replay.js';
import { decode, encode } from '../src/protocol/codec.js';
import {
  KIND_IDS, PROTOCOL_MINOR_3_2, SNAP_AUTO_VMAX, SNAP_AUTO_VMIN,
  type Message, type SeriesMsg, type SnapshotMsg, type ViewEntry,
} from '../src/protocol/messages.js';

function concatFrames(msgs: Message[]): Uint8Array {
  const parts = msgs.map((m) => encode(m, PROTOCOL_MINOR_3_2));
  const total = parts.reduce((sum, p) => sum + p.byteLength, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.byteLength;
  }
  return out;
}

function makeSampleRecording(steps: number[]): Uint8Array {
  const views: ViewEntry[] = [
    { id: 1, name: 'loss', kind: KIND_IDS.series, sources: [] },
    { id: 2, name: 'weights', kind: KIND_IDS.heatmap, sources: [] },
    { id: 3, name: 'embedding', kind: KIND_IDS.points, sources: [] },
  ];
  const msgs: Message[] = [
    {
      type: 'welcome', role: 'producer', capabilities: [], minor: PROTOCOL_MINOR_3_2,
      session: { id: 'mnist-test', script: 'examples/mnist/train.py', pid: 0, start_unix_ms: 1700000000000 },
    },
    { type: 'views', views, timelines: [{ name: 'step', kind: 'sequence' }, { name: 'time', kind: 'time' }] },
  ];

  // One full series with stamps at steps 0..maxStep
  const maxStep = steps[steps.length - 1];
  const seriesCount = Math.max(1, maxStep + 1);
  const vals = new Float32Array(seriesCount);
  const stamps = new BigInt64Array(seriesCount * 2);
  for (let i = 0; i < seriesCount; i++) {
    vals[i] = 2.0 / (1 + i * 0.1);
    stamps[i * 2] = BigInt(i);
    stamps[i * 2 + 1] = BigInt(i * 1000);
  }
  msgs.push({
    type: 'series', sub: 1, view: 'loss', start: 0, total: seriesCount, stamp_ns: seriesCount * 1000,
    values: vals, stamps, stamps_t: 2,
  });

  // Snapshots at each checkpoint step
  steps.forEach((step, idx) => {
    const heat = new Float32Array(4 * 4);
    for (let i = 0; i < heat.length; i++) heat[i] = step * 0.01 + i * 0.1;
    msgs.push({
      type: 'snapshot', sub: 2, view: 'weights', seq: idx + 1, stamp_ns: step * 1000, kind: KIND_IDS.heatmap,
      data_view: {}, data: heat, shape: [4, 4], colormap: 0, flags: SNAP_AUTO_VMIN | SNAP_AUTO_VMAX,
      vmin: 0, vmax: 1, stats: [heat[0], heat[heat.length - 1]], axes: [],
      stamp: new BigInt64Array([BigInt(step), BigInt(step * 1000)]),
    });
    const pts = new Float32Array([step * 0.1, 1.0, 0.0, -step * 0.1, -1.0, 0.5]);
    const aux = new Float32Array([0, 1]);
    msgs.push({
      type: 'snapshot', sub: 3, view: 'embedding', seq: idx + 1, stamp_ns: step * 1000, kind: KIND_IDS.points,
      data_view: {}, data: pts, shape: [2, 3], aux, colormap: 0, flags: 0, vmin: 0, vmax: 9, stats: [], axes: [],
      points_dims: 3, label_count: 10, stamp: new BigInt64Array([BigInt(step), BigInt(step * 1000)]),
    });
  });
  return concatFrames(msgs);
}

test('isStreamUrl distinguishes WebSocket streams from static recording paths', () => {
  assert.equal(isStreamUrl('ws://127.0.0.1:7341/v3/stream?token=a'), true);
  assert.equal(isStreamUrl('wss://example.com/v3/stream'), true);
  assert.equal(isStreamUrl('./recording/recording.calrec'), false);
  assert.equal(isStreamUrl('https://example.com/mnist/recording.calrec'), false);
});

test('parseRecording extracts session, views, steps, snapshots and series', () => {
  const bytes = makeSampleRecording([0, 10, 25]);
  const rec = parseRecording(bytes);
  assert.equal(rec.session.id, 'mnist-test');
  assert.equal(rec.session.script, 'examples/mnist/train.py');
  assert.deepEqual(rec.views.map((v) => v.name), ['loss', 'weights', 'embedding']);
  // The series has a sample at every step 0..25, so the timeline holds every step.
  assert.deepEqual(rec.steps, Array.from({ length: 26 }, (_, i) => i));
  assert.equal(rec.snapshots.get('weights')?.length, 3);
  assert.equal(rec.series.get('loss')?.values.length, 26);
});

test('the timeline is the union of snapshot and series steps, and views hold their latest value', async () => {
  const views: ViewEntry[] = [
    { id: 1, name: 'loss', kind: KIND_IDS.series, sources: [] },
    { id: 2, name: 'weights', kind: KIND_IDS.heatmap, sources: [] },
  ];
  const snap = (step: number): Message => ({
    type: 'snapshot', sub: 2, view: 'weights', seq: step + 1, stamp_ns: step, kind: KIND_IDS.heatmap,
    data_view: {}, data: new Float32Array([step]), shape: [1, 1], colormap: 0, flags: 0, vmin: 0, vmax: 1,
    stats: [step, step], axes: [], stamp: new BigInt64Array([BigInt(step), BigInt(step)]),
  });
  // Series samples at steps 0, 5, 10 and snapshots at steps 0 and 7.
  const bytes = concatFrames([
    { type: 'views', views },
    {
      type: 'series', sub: 1, view: 'loss', start: 0, total: 3, stamp_ns: 10,
      values: new Float32Array([3, 2, 1]), stamps: new BigInt64Array([0n, 0n, 5n, 5n, 10n, 10n]), stamps_t: 2,
    },
    snap(0),
    snap(7),
  ]);
  assert.deepEqual(parseRecording(bytes).steps, [0, 5, 7, 10]);

  const src = new ReplaySource('http://localhost/r.calrec', { bytes, initialStep: 'first' });
  const snaps: SnapshotMsg[] = [];
  const series: SeriesMsg[] = [];
  src.listen({ onSnapshot: (s) => snaps.push(s), onSeries: (s) => series.push(s) });
  src.connect();
  await src.ready;
  src.subscribe('loss');
  src.subscribe('weights');
  await new Promise((r) => queueMicrotask(() => r(null)));
  src.seekStep(5);
  assert.equal(series.at(-1)?.total, 2);
  assert.equal(snaps.at(-1)?.data[0], 0);   // no snapshot at step 5: the step 0 snapshot holds
  src.seekStep(7);
  assert.equal(series.at(-1)?.total, 2);    // no series sample at step 7: the step 5 value holds
  assert.equal(snaps.at(-1)?.data[0], 7);
  src.close();
});

test('every points snapshot carries the bounding box of the whole run', () => {
  const pts = (step: number, half: number): Message => ({
    type: 'snapshot', sub: 3, view: 'embedding', seq: step + 1, stamp_ns: step, kind: KIND_IDS.points,
    data_view: {}, data: new Float32Array([-half, -half, -half, half, half, half]), shape: [2, 3],
    aux: new Float32Array([0, 1]), colormap: 0, flags: 0, vmin: 0, vmax: 9,
    stats: [0, 9, -half, half, -2 * half, 2 * half, -half, half], axes: [], points_dims: 3, label_count: 10,
    stamp: new BigInt64Array([BigInt(step), BigInt(step)]),
  });
  const rec = parseRecording(concatFrames([
    { type: 'views', views: [{ id: 3, name: 'embedding', kind: KIND_IDS.points, sources: [] }] },
    pts(0, 0.1),
    pts(10, 3),
  ]));
  for (const s of rec.snapshots.get('embedding')!) {
    assert.deepEqual(s.msg.stats, [0, 9, -3, 3, -6, 6, -3, 3]);
  }
});

test('play advances one step per animation frame and stops at the last step', async () => {
  const frames: (() => void)[] = [];
  const g = globalThis as unknown as Record<string, unknown>;
  const saved = [g.requestAnimationFrame, g.cancelAnimationFrame];
  g.requestAnimationFrame = (cb: () => void) => frames.push(cb);
  g.cancelAnimationFrame = (id: number) => { frames[id - 1] = () => {}; };
  try {
    const src = new ReplaySource('http://localhost/r.calrec', { bytes: makeSampleRecording([0, 3]), initialStep: 'first' });
    src.connect();
    await src.ready;
    assert.deepEqual(src.steps, [0, 1, 2, 3]);
    src.play();
    assert.equal(src.playing, true);
    assert.equal(src.currentStep, 0);
    for (const want of [1, 2, 3]) {
      frames.at(-1)!();
      assert.equal(src.currentStep, want);
    }
    assert.equal(src.playing, false);
    assert.equal(frames.length, 3);   // no frame is requested after the last step
    src.close();
  } finally {
    [g.requestAnimationFrame, g.cancelAnimationFrame] = saved;
  }
});

test('reset pauses, returns to the first step and re-sends every view with a new cursor', async () => {
  const src = new ReplaySource('http://localhost/r.calrec', { bytes: makeSampleRecording([0, 10, 25]), initialStep: 'last' });
  src.connect();
  await src.ready;
  const peerFrames: Message[] = [];
  const peer = src.attachFramePeer((frame) => {
    const decoded = decode(frame);
    if (decoded.type !== 'unknown') peerFrames.push(decoded);
  });
  peer.receive(encode({ type: 'hello', role: 'viewer', capabilities: [] }, PROTOCOL_MINOR_3_2));
  peer.receive(encode({ type: 'subscribe', sub: 7, view: 'loss', request_id: 1 }, PROTOCOL_MINOR_3_2));
  peer.receive(encode({ type: 'subscribe', sub: 8, view: 'weights', request_id: 2 }, PROTOCOL_MINOR_3_2));
  const lastSeries = () => peerFrames.filter((m): m is SeriesMsg => m.type === 'series').at(-1)!;
  const lastSnap = () => peerFrames.filter((m): m is SnapshotMsg => m.type === 'snapshot').at(-1)!;
  assert.equal(lastSeries().total, 26);

  src.reset();
  assert.equal(src.playing, false);
  assert.equal(src.stepIndex, 0);
  assert.equal(lastSeries().total, 1);
  assert.equal(lastSeries().values[0], 2);
  assert.ok(Math.abs(lastSnap().data[0] - 0) < 1e-6);
  const cursor = lastSeries().cursor!;

  // At the first step already, a reset still re-sends the views with a newer cursor.
  const before = peerFrames.length;
  src.reset();
  assert.ok(peerFrames.length > before);
  assert.ok(lastSeries().cursor! > cursor);
  assert.equal(lastSeries().total, 1);
  peer.detach();
  src.close();
});

test('ReplaySource seeks snapshots and series cleanly for both StreamClient and WASM frame peers', async () => {
  const bytes = makeSampleRecording([0, 10, 25]);
  const src = new ReplaySource('http://localhost/recording.calrec', { bytes, initialStep: 'last' });
  const receivedSnaps: SnapshotMsg[] = [];
  const receivedSeries: SeriesMsg[] = [];
  src.listen({
    onSnapshot: (s) => receivedSnaps.push(s),
    onSeries: (s) => receivedSeries.push(s),
  });
  src.connect();
  await src.ready;

  assert.equal(src.state, 'open');
  assert.equal(src.currentStep, 25);
  assert.equal(src.stepIndex, 25);

  // StreamClient subscriptions deliver initial step (25)
  const subLoss = src.subscribe('loss');
  const subWeights = src.subscribe('weights');
  await new Promise((r) => queueMicrotask(() => r(null)));
  assert.equal(receivedSeries.at(-1)?.total, 26);
  assert.ok(Math.abs((receivedSnaps.at(-1)?.data[0] ?? 0) - 0.25) < 1e-5);

  // WASM frame peer handshake and seeking backwards
  const peerFrames: Message[] = [];
  const peer = src.attachFramePeer((frame) => {
    const decoded = decode(frame);
    if (decoded.type !== 'unknown') peerFrames.push(decoded);
  });
  peer.receive(encode({ type: 'hello', role: 'viewer', capabilities: [] }, PROTOCOL_MINOR_3_2));
  assert.deepEqual(peerFrames.map((m) => m.type), ['welcome', 'sessions', 'views']);
  assert.equal((peerFrames[1] as { sessions: { state?: string }[] }).sessions[0].state, 'live');

  peerFrames.length = 0;
  peer.receive(encode({ type: 'subscribe', sub: 7, view: 'loss', request_id: 1 }, PROTOCOL_MINOR_3_2));
  peer.receive(encode({ type: 'subscribe', sub: 8, view: 'weights', request_id: 2 }, PROTOCOL_MINOR_3_2));
  assert.deepEqual(peerFrames.map((m) => m.type), ['view_reply', 'series', 'view_reply', 'snapshot']);
  assert.equal((peerFrames[1] as SeriesMsg).total, 26);
  const seqAt25 = (peerFrames[3] as SnapshotMsg).seq;

  // Seek backwards to step 0: series truncates to 1 value, snapshot seq still increases monotonically!
  peerFrames.length = 0;
  src.seekIndex(0);
  assert.equal(src.currentStep, 0);
  assert.equal(receivedSeries.at(-1)?.total, 1);
  assert.ok(Math.abs((receivedSnaps.at(-1)?.data[0] ?? 99) - 0.0) < 1e-5);
  const peerSeries0 = peerFrames.find((m): m is SeriesMsg => m.type === 'series');
  const peerSnap0 = peerFrames.find((m): m is SnapshotMsg => m.type === 'snapshot');
  assert.equal(peerSeries0?.total, 1);
  assert.ok(peerSeries0?.cursor !== undefined);
  assert.ok((peerSnap0?.seq ?? 0) > seqAt25);

  // Seek forward to step 10
  src.seekStep(10);
  assert.equal(src.currentStep, 10);
  assert.equal(receivedSeries.at(-1)?.total, 11);

  src.unsubscribe(subLoss);
  src.unsubscribe(subWeights);
  peer.detach();
  src.close();
});

test('parseRecording rejects truncated or non-CALW bytes', () => {
  assert.throws(() => parseRecording(new Uint8Array([1, 2, 3, 4])), /truncated/);
  assert.throws(() => parseRecording(new Uint8Array(24)), /magic/);
});
