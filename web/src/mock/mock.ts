// A producer simulated in the page: it implements StreamClient and makes the
// data a small training run would stream (a decaying noisy loss, a drifting
// weight matrix, attention maps with layers and heads to pick from, an
// embedding whose clusters separate, a histogram of weights and an input
// image). It answers subscriptions, data view requests and events like a real
// producer, without a socket, so the views can be built and tested alone.

import { emit, type ConnectionState, type StreamClient, type StreamListener } from '../client/stream.js';
import {
  KIND_IDS, SNAP_AUTO_VMAX, SNAP_AUTO_VMIN, type DataView, type SeriesMsg, type SessionIdentity, type SnapshotMsg,
  type ViewEntry, type ViewerEvent,
} from '../protocol/messages.js';
import { applyDataView, type Tensor } from './tensor.js';

/** A small deterministic generator (mulberry32) and a normal sampler. */
export function rng(seed: number): { u(): number; n(): number } {
  let a = seed >>> 0;
  const u = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const n = () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u());
  return { u, n };
}

interface MockSub {
  view: string;
  dataView: DataView;
  resolved: DataView | null;
  seq: number;
  /** Next series index to send. */
  sent: number;
}

export interface MockOptions {
  /** Points in the embedding (default 4000). */
  points?: number;
  /** 2 or 3 (default 3). */
  dims?: 2 | 3;
  /** Milliseconds between producer steps (default 50); 0 steps only on step(). */
  interval?: number;
  seed?: number;
}

const LAYERS = 4, HEADS = 8, SEQ = 32;
const W_ROWS = 48, W_COLS = 64;
const CLASSES = 8;

export class MockSource implements StreamClient {
  state: ConnectionState = 'connecting';
  session: SessionIdentity | null = null;
  views: readonly ViewEntry[] = [];
  /** Every event a viewer sent, newest last. */
  readonly events: { event: ViewerEvent; sub?: number; seq?: number; dataView?: DataView }[] = [];
  /** Every data view request, newest last. */
  readonly requests: { sub: number; view: string; dataView: DataView }[] = [];

  private readonly listeners = new Set<StreamListener>();
  private readonly subs = new Map<number, MockSub>();
  private nextSub = 1;
  private nextReq = 1;
  private timer: ReturnType<typeof setInterval> | null = null;
  private step_ = 0;
  private readonly r;
  private readonly nPoints: number;
  private readonly pDims: 2 | 3;
  private readonly interval: number;
  private loss: number[] = [];
  private weights = new Float32Array(W_ROWS * W_COLS);
  private readonly centers: number[][];
  private readonly offsets: Float32Array;
  private readonly labels: Float32Array;

  constructor(opts: MockOptions = {}) {
    this.r = rng(opts.seed ?? 7);
    this.nPoints = opts.points ?? 4000;
    this.pDims = opts.dims ?? 3;
    this.interval = opts.interval ?? 50;
    for (let i = 0; i < this.weights.length; i++) this.weights[i] = this.r.n() * 0.5;
    this.centers = Array.from({ length: CLASSES }, (_, k) => {
      const a = (2 * Math.PI * k) / CLASSES;
      return [Math.cos(a) * 4, Math.sin(a) * 4, (k % 2 ? 1 : -1) * 2];
    });
    this.offsets = new Float32Array(this.nPoints * 3);
    this.labels = new Float32Array(this.nPoints);
    for (let i = 0; i < this.nPoints; i++) {
      this.labels[i] = i % CLASSES;
      for (let d = 0; d < 3; d++) this.offsets[i * 3 + d] = this.r.n();
    }
    const kind = KIND_IDS;
    this.views = [
      { id: 1, name: 'loss', kind: kind.series, sources: [] },
      {
        id: 2, name: 'weights', kind: kind.heatmap,
        sources: [{ name: 'fc1.weight', dims: [{ name: 'out', size: W_ROWS }, { name: 'in', size: W_COLS }], dtype: 'f32' }],
        default: {},
      },
      {
        id: 3, name: 'attention', kind: kind.heatmap,
        sources: ['attn', 'scores'].map((name) => ({
          name, dtype: 'f32',
          dims: [{ name: 'layer', size: LAYERS }, { name: 'head', size: HEADS }, { name: 'query', size: SEQ }, { name: 'key', size: SEQ }],
        })),
        default: { source: 'attn', index: { layer: 0, head: 0 } },
      },
      {
        id: 4, name: 'embedding', kind: kind.points,
        sources: [{ name: 'hidden', dims: [{ name: 'layer', size: 3 }, { name: 'sample', size: this.nPoints }], dtype: 'f32' }],
        default: { index: { layer: 2 } },
      },
      { id: 5, name: 'weight_hist', kind: kind.histogram, sources: [{ name: 'fc1.weight', dims: [], dtype: 'f32' }] },
      { id: 6, name: 'input', kind: kind.image, sources: [{ name: 'batch', dims: [{ name: 'height', size: 32 }, { name: 'width', size: 32 }], dtype: 'u8' }] },
    ];
  }

  connect(): void {
    if (this.state !== 'connecting') return;
    this.session = { id: 'mock', script: 'train_mock.py', pid: 1, start_unix_ms: Date.now() };
    this.state = 'open';
    emit(this.listeners, 'onStatus', 'open');
    emit(this.listeners, 'onSession', this.session);
    emit(this.listeners, 'onViews', this.views);
    for (let i = 0; i < 40; i++) this.advance();
    if (this.interval > 0) this.timer = setInterval(() => this.tick(), this.interval);
  }

  close(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.state = 'ended';
    emit(this.listeners, 'onStatus', 'ended');
  }

  listen(l: StreamListener): () => void {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  }

  subscribe(view: string, dataView: DataView = {}): number {
    const sub = this.nextSub++;
    this.subs.set(sub, { view, dataView, resolved: null, seq: 0, sent: Math.max(0, this.loss.length - 100000) });
    queueMicrotask(() => {
      const s = this.subs.get(sub);
      if (!s) return;
      this.resolve(sub, s, this.nextReq++);
      this.publish(sub, s);
    });
    return sub;
  }

  unsubscribe(sub: number): void {
    this.subs.delete(sub);
  }

  requestDataView(sub: number, dataView: DataView): number {
    const id = this.nextReq++;
    const s = this.subs.get(sub);
    if (!s) return id;
    this.requests.push({ sub, view: s.view, dataView });
    const before = s.dataView;
    s.dataView = dataView;
    queueMicrotask(() => {
      if (!this.resolve(sub, s, id)) s.dataView = before;
      else this.publish(sub, s);
    });
    return id;
  }

  sendEvent(event: ViewerEvent, sub?: number, seq?: number, dataView?: DataView): void {
    this.events.push({ event, sub, seq, dataView });
  }

  /** Advances the simulated training by n steps and publishes. */
  step(n = 1): void {
    for (let i = 0; i < n; i++) this.advance();
    for (const [sub, s] of this.subs) this.publish(sub, s);
  }

  /** Stops the clock (step() still works). */
  pause(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  // --- the simulated run ------------------------------------------------------

  private tick(): void {
    this.step(1);
  }

  private advance(): void {
    const t = ++this.step_;
    for (let k = 0; k < 4; k++) {
      const i = this.loss.length;
      const base = 0.08 + 2.2 * Math.exp(-i / 600) + 0.3 * Math.exp(-i / 60);
      this.loss.push(base * (1 + 0.12 * this.r.n()) + 0.01 * Math.abs(this.r.n()));
    }
    for (let i = 0; i < this.weights.length; i++) {
      const y = Math.floor(i / W_COLS), x = i % W_COLS;
      const target = 0.8 * Math.sin(x / 7 + t / 40) * Math.cos(y / 5 - t / 60);
      this.weights[i] += 0.04 * (target - this.weights[i]) + 0.02 * this.r.n();
    }
  }

  private progress(): number {
    return 1 - Math.exp(-this.step_ / 150);
  }

  private attention(source: string): Tensor {
    const data = new Float32Array(LAYERS * HEADS * SEQ * SEQ);
    const t = this.step_;
    let o = 0;
    for (let l = 0; l < LAYERS; l++) for (let h = 0; h < HEADS; h++) {
      const width = 1.5 + h * 0.8 + l, shift = (h % 3) - 1;
      for (let q = 0; q < SEQ; q++) {
        let sum = 0;
        const row = new Float32Array(SEQ);
        for (let k = 0; k < SEQ; k++) {
          const d = k - q - shift + Math.sin(t / 30 + h) * 2;
          const s = -(d * d) / (2 * width * width) + (k === 0 ? l * 0.7 : 0);
          row[k] = source === 'scores' ? s : Math.exp(s);
          sum += row[k];
        }
        for (let k = 0; k < SEQ; k++) data[o++] = source === 'scores' ? row[k] : row[k] / sum;
      }
    }
    return { data, shape: [LAYERS, HEADS, SEQ, SEQ] };
  }

  private embedding(): Tensor {
    const n = this.nPoints, p = this.progress();
    const data = new Float32Array(3 * n * this.pDims);
    for (let layer = 0; layer < 3; layer++) {
      const sep = p * (0.4 + 0.3 * layer);
      for (let i = 0; i < n; i++) {
        const c = this.centers[this.labels[i]];
        for (let d = 0; d < this.pDims; d++) {
          data[(layer * n + i) * this.pDims + d] = c[d] * sep + this.offsets[i * 3 + d] * (1.2 - 0.5 * sep);
        }
      }
    }
    return { data, shape: [3, n, this.pDims] };
  }

  private image(): Uint8Array {
    const out = new Uint8Array(32 * 32 * 4);
    const t = this.step_ / 10;
    for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) {
      const o = (y * 32 + x) * 4;
      const r = Math.hypot(x - 16 - 6 * Math.cos(t), y - 16 - 6 * Math.sin(t));
      out[o] = Math.max(0, 255 - r * 18);
      out[o + 1] = (x * 8) & 255;
      out[o + 2] = (y * 8) & 255;
      out[o + 3] = 255;
    }
    return out;
  }

  // --- the producer side of subscriptions ------------------------------------

  private viewOf(name: string): ViewEntry | undefined {
    return this.views.find((v) => v.name === name);
  }

  /** Resolves the sub's data view; sends VIEW_REPLY. False on an error. */
  private resolve(sub: number, s: MockSub, requestId: number): boolean {
    const v = this.viewOf(s.view);
    if (!v) {
      emit(this.listeners, 'onError', { type: 'error', code: 'unknown_view', message: `no view '${s.view}'`, fatal: false, sub, view: s.view });
      return false;
    }
    try {
      s.resolved = this.compute(v, s.dataView).resolved;
    } catch (e) {
      emit(this.listeners, 'onReply', { type: 'view_reply', request_id: requestId, sub, view: s.view, error: { code: 'bad_data_view', message: (e as Error).message } });
      return false;
    }
    emit(this.listeners, 'onReply', { type: 'view_reply', request_id: requestId, sub, view: s.view, data_view: s.resolved });
    return true;
  }

  private compute(v: ViewEntry, dv: DataView) {
    const merged: DataView = { ...(v.default ?? {}), ...dv };
    const src = v.sources.find((x) => x.name === (merged.source ?? v.sources[0]?.name)) ?? v.sources[0];
    if (v.name === 'attention') return applyDataView(this.attention(src.name), src, merged, 2);
    if (v.name === 'weights') return applyDataView({ data: this.weights, shape: [W_ROWS, W_COLS] }, src, merged, 2);
    if (v.name === 'embedding') {
      const e = this.embedding();
      // The sample dim is the point axis; the coordinate dim travels whole.
      const per = this.pDims;
      const layer = typeof merged.index?.layer === 'number' ? merged.index.layer : 2;
      if (merged.reduce && Object.keys(merged.reduce).length) throw new Error('the embedding cannot be reduced');
      if (layer < 0 || layer > 2) throw new Error(`index ${layer} out of range for 'layer' of size 3`);
      const n = this.nPoints;
      return {
        data: e.data.slice(layer * n * per, (layer + 1) * n * per), shape: [n, per],
        axes: [{ dim: 'sample', start: 0, step: 1, count: n, size: n }],
        resolved: { source: src.name, index: { layer }, reduce: {}, budget: { max_points: 100000 } } as DataView, tile: null,
      };
    }
    return { data: new Float32Array(0), shape: [0], axes: [], resolved: { budget: {} } as DataView, tile: null };
  }

  private publish(sub: number, s: MockSub): void {
    const v = this.viewOf(s.view);
    if (!v || !s.resolved) return;
    const base = { type: 'snapshot' as const, sub, view: s.view, stamp_ns: Date.now() * 1e6, kind: v.kind, data_view: s.resolved };
    if (v.kind === KIND_IDS.series) {
      if (s.sent >= this.loss.length) return;
      const values = new Float32Array(this.loss.slice(s.sent));
      const msg: SeriesMsg = { type: 'series', sub, view: s.view, start: s.sent, total: this.loss.length, stamp_ns: base.stamp_ns, values };
      s.sent = this.loss.length;
      emit(this.listeners, 'onSeries', msg);
      return;
    }
    let snap: SnapshotMsg;
    if (v.kind === KIND_IDS.histogram) {
      const bins = 64, lo = -2.5, hi = 2.5;
      const counts = new Float32Array(bins);
      for (const w of this.weights) {
        const b = Math.floor(((w - lo) / (hi - lo)) * bins);
        if (b >= 0 && b < bins) counts[b]++;
      }
      snap = { ...base, seq: ++s.seq, data: counts, shape: [bins], colormap: 0, flags: 0, vmin: 0, vmax: 1, stats: [lo, hi], axes: [] };
    } else if (v.kind === KIND_IDS.image) {
      snap = { ...base, seq: ++s.seq, data: this.image(), shape: [32, 32, 4], colormap: 0, flags: 0, vmin: 0, vmax: 1, stats: [], axes: [] };
    } else {
      let c;
      try {
        c = this.compute(v, s.dataView);
      } catch {
        return;
      }
      let lo = Infinity, hi = -Infinity;
      for (const x of c.data) if (Number.isFinite(x)) { lo = Math.min(lo, x); hi = Math.max(hi, x); }
      if (v.kind === KIND_IDS.points) {
        snap = {
          ...base, seq: ++s.seq, data: c.data, shape: c.shape, aux: this.labels, colormap: 0, flags: 0, vmin: 0, vmax: CLASSES - 1,
          stats: [], axes: c.axes, points_dims: this.pDims, label_count: CLASSES,
        };
      } else if (v.name === 'weights') {
        const m = Math.max(Math.abs(lo), Math.abs(hi));
        snap = { ...base, seq: ++s.seq, data: c.data, shape: c.shape, colormap: 2, flags: 0, vmin: -m, vmax: m, stats: [lo, hi], axes: c.axes };
      } else {
        snap = {
          ...base, seq: ++s.seq, data: c.data, shape: c.shape, colormap: 0, flags: SNAP_AUTO_VMIN | SNAP_AUTO_VMAX, vmin: 0, vmax: 1,
          stats: [lo, hi], axes: c.axes, tile: c.tile,
        };
      }
    }
    emit(this.listeners, 'onSnapshot', snap);
  }
}
