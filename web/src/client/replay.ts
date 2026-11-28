// A StreamClient that replays a static recording file (.calrec: concatenated
// wire protocol v3 binary frames) with no server and no WebSocket. It drives
// both Canvas 2D <caliper-view> subscriptions and WebAssembly views running
// with feed "frames", and exposes a shared step timeline for time scrubbing.

import { decode, decodeFrame, encode, HEADER_BYTES, MAGIC } from '../protocol/codec.js';
import {
  KIND_IDS, PROTOCOL_MINOR_3_2, type DataView as DataViewSpec, type Message, type SeriesMsg,
  type SessionIdentity, type SnapshotMsg, type ViewEntry, type ViewerEvent,
} from '../protocol/messages.js';
import { emit, type ConnectionState, type StreamClient, type StreamListener } from './stream.js';

export interface ReplayOptions {
  /** Which step to show first once loaded: 'last' (default), 'first', or a step number. */
  initialStep?: 'first' | 'last' | number;
  /** Preloaded recording bytes (skips fetch when provided). */
  bytes?: Uint8Array | ArrayBuffer;
}

export interface RecordedSnapshot {
  step: number;
  msg: SnapshotMsg;
}

export interface RecordedSeries {
  values: Float32Array;
  /** Training step associated with each sample index. */
  steps: Float64Array;
  stamp_ns: number;
}

export interface ParsedRecording {
  session: SessionIdentity;
  views: ViewEntry[];
  steps: number[];
  snapshots: Map<string, RecordedSnapshot[]>;
  series: Map<string, RecordedSeries>;
}

interface Sub {
  view: string;
  dataView?: DataViewSpec;
  seq: number;
}

interface FramePeer {
  send: (bytes: Uint8Array) => void;
  greeted: boolean;
  sentViews: boolean;
  subs: Map<number, { view: string; dataView?: DataViewSpec }>;
  nextSeq: number;
}

export interface FramePeerHandle {
  receive(bytes: Uint8Array): void;
  detach(): void;
}

/** True when src names a live WebSocket stream rather than a static recording file. */
export function isStreamUrl(src: string): boolean {
  return /^wss?:\/\//i.test(src.trim());
}

/** Resolves a possibly relative URL against document.baseURI or location.href. */
export function resolvePageUrl(raw: string, base?: string): string {
  const origin = base ?? (typeof document !== 'undefined' && document.baseURI ? document.baseURI
    : typeof location !== 'undefined' ? location.href : 'http://localhost/');
  return new URL(raw, origin).href;
}

function defaultScriptName(src: string): string {
  try {
    const u = new URL(src, 'http://localhost/');
    const seg = u.pathname.split('/').filter(Boolean).pop();
    return seg || 'recording';
  } catch {
    return src || 'recording';
  }
}

/** Parses concatenated wire protocol v3 binary frames into an indexed replayable run. */
export function parseRecording(input: Uint8Array | ArrayBuffer, fallbackScript = 'recording'): ParsedRecording {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  let session: SessionIdentity | null = null;
  let views: ViewEntry[] = [];
  const snapshots = new Map<string, RecordedSnapshot[]>();
  const seriesChunks = new Map<string, { start: number; values: Float32Array; steps: Float64Array; stamp_ns: number }[]>();

  let pos = 0;
  let frameIndex = 0;
  while (pos < bytes.byteLength) {
    if (bytes.byteLength - pos < HEADER_BYTES) {
      throw new Error(`recording truncated at byte ${pos} (frame ${frameIndex})`);
    }
    const dv = new DataView(bytes.buffer, bytes.byteOffset + pos, HEADER_BYTES);
    const magic = dv.getUint32(0, true);
    if (magic !== MAGIC) {
      throw new Error(`recording has invalid frame magic 0x${magic.toString(16)} at byte ${pos}`);
    }
    const frameLen = dv.getUint32(12, true);
    if (frameLen < HEADER_BYTES || pos + frameLen > bytes.byteLength) {
      throw new Error(`recording frame ${frameIndex} length ${frameLen} exceeds buffer at byte ${pos}`);
    }
    const slice = bytes.subarray(pos, pos + frameLen);
    // Validate framing and decode the protocol message.
    decodeFrame(slice);
    const msg = decode(slice);
    pos += frameLen;
    frameIndex++;

    if (msg.type === 'welcome' && msg.session) {
      session = { ...msg.session };
    } else if (msg.type === 'sessions' && msg.sessions.length > 0 && !session) {
      session = { ...msg.sessions[0] };
    } else if (msg.type === 'views') {
      views = msg.views;
    } else if (msg.type === 'snapshot') {
      const step = msg.stamp && msg.stamp.length > 0 ? Number(msg.stamp[0]) : msg.seq;
      let list = snapshots.get(msg.view);
      if (!list) {
        list = [];
        snapshots.set(msg.view, list);
      }
      list.push({ step, msg });
    } else if (msg.type === 'series') {
      const n = msg.values.length;
      const sampleSteps = new Float64Array(n);
      if (msg.stamps && msg.stamps_t && msg.stamps_t > 0) {
        for (let i = 0; i < n; i++) sampleSteps[i] = Number(msg.stamps[i * msg.stamps_t]);
      } else if (msg.stamp && msg.stamp.length > 0) {
        const s = Number(msg.stamp[0]);
        sampleSteps.fill(s);
      } else {
        for (let i = 0; i < n; i++) sampleSteps[i] = msg.start + i;
      }
      let list = seriesChunks.get(msg.view);
      if (!list) {
        list = [];
        seriesChunks.set(msg.view, list);
      }
      list.push({ start: msg.start, values: msg.values, steps: sampleSteps, stamp_ns: msg.stamp_ns });
    }
  }

  // Reconstruct contiguous series arrays from recorded deltas or full frames.
  const series = new Map<string, RecordedSeries>();
  for (const [view, chunks] of seriesChunks) {
    let total = 0;
    let stampNs = 0;
    for (const c of chunks) {
      total = Math.max(total, c.start + c.values.length);
      stampNs = Math.max(stampNs, c.stamp_ns);
    }
    const values = new Float32Array(total).fill(NaN);
    const sampleSteps = new Float64Array(total);
    for (let i = 0; i < total; i++) sampleSteps[i] = i;
    for (const c of chunks) {
      values.set(c.values, c.start);
      sampleSteps.set(c.steps, c.start);
    }
    series.set(view, { values, steps: sampleSteps, stamp_ns: stampNs });
  }

  // Synthesize missing view entries if a recording omitted VIEWS (defensive).
  if (!views.length) {
    let nextId = 1;
    for (const [name, list] of snapshots) {
      if (list.length) views.push({ id: nextId++, name, kind: list[0].msg.kind, sources: [] });
    }
    for (const name of series.keys()) {
      views.push({ id: nextId++, name, kind: KIND_IDS.series, sources: [] });
    }
  }

  // Sort each view's snapshots by step while preserving frame order on ties.
  for (const list of snapshots.values()) {
    list.sort((a, b) => a.step - b.step);
  }

  // Collect checkpoint steps: prefer snapshot steps so every tick advances every visual view.
  const stepSet = new Set<number>();
  for (const list of snapshots.values()) {
    for (const s of list) stepSet.add(s.step);
  }
  if (stepSet.size === 0) {
    for (const s of series.values()) {
      for (let i = 0; i < s.steps.length; i++) stepSet.add(s.steps[i]);
    }
  }
  const steps = [...stepSet].sort((a, b) => a - b);

  const id: SessionIdentity = session ?? {
    id: 'recording',
    script: fallbackScript,
    pid: 0,
    start_unix_ms: 0,
    state: 'ended',
    views: views.length,
  };
  return { session: id, views, steps, snapshots, series };
}

/** Binary search for the last index with steps[i] <= target (at least 0 when non-empty). */
function lastAtOrBefore(items: readonly { step: number }[], target: number): number {
  let lo = 0;
  let hi = items.length - 1;
  let best = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    if (items[mid].step <= target) {
      best = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return best;
}

/** Count of prefix samples whose step is <= target, clamped to at least 1 when non-empty. */
function seriesCountAt(steps: Float64Array, target: number): number {
  const n = steps.length;
  if (!n) return 0;
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (steps[i] <= target) count = i + 1;
  }
  return Math.max(1, count);
}

export class ReplaySource implements StreamClient {
  state: ConnectionState = 'connecting';
  session: SessionIdentity | null = null;
  views: readonly ViewEntry[] = [];
  readonly capabilities: readonly string[] = [];
  readonly controls = null;
  readonly selections = new Map<string, never>();

  private recording: ParsedRecording | null = null;
  private error_: string | null = null;
  private stepIndex_ = 0;
  private seekGen = 1;
  private playing_ = false;
  private playTimer: ReturnType<typeof setInterval> | null = null;
  private started = false;
  private closed = false;
  private readonly listeners = new Set<StreamListener>();
  private readonly timelineListeners = new Set<() => void>();
  private readonly subs = new Map<number, Sub>();
  private readonly peers = new Set<FramePeer>();
  private nextSub = 1;
  private nextReq = 1;
  private readonly initialStep: 'first' | 'last' | number;
  private readonly preloaded?: Uint8Array | ArrayBuffer;
  private resolveReady!: () => void;
  readonly ready: Promise<void>;

  constructor(readonly src: string, opts: ReplayOptions = {}) {
    this.initialStep = opts.initialStep ?? 'last';
    this.preloaded = opts.bytes;
    this.ready = new Promise<void>((resolve) => { this.resolveReady = resolve; });
  }

  get steps(): readonly number[] {
    return this.recording?.steps ?? [];
  }

  get stepIndex(): number {
    return this.stepIndex_;
  }

  get currentStep(): number {
    const s = this.steps;
    return s.length ? s[Math.min(this.stepIndex_, s.length - 1)] : 0;
  }

  get playing(): boolean {
    return this.playing_;
  }

  get error(): string | null {
    return this.error_;
  }

  connect(): void {
    if (this.started || this.closed) return;
    this.started = true;
    if (this.preloaded) {
      this.loadBytes(this.preloaded);
      return;
    }
    void this.fetchAndLoad();
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.pause();
    this.state = 'ended';
    emit(this.listeners, 'onStatus', 'ended');
    this.notifyTimeline();
  }

  listen(listener: StreamListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  onTimeline(cb: () => void): () => void {
    this.timelineListeners.add(cb);
    return () => this.timelineListeners.delete(cb);
  }

  subscribe(view: string, dataView?: DataViewSpec): number {
    const sub = this.nextSub++;
    this.subs.set(sub, { view, dataView, seq: 0 });
    if (this.state === 'open') {
      queueMicrotask(() => {
        if (!this.closed && this.subs.has(sub)) this.deliverSub(sub, true);
      });
    }
    return sub;
  }

  unsubscribe(sub: number): void {
    this.subs.delete(sub);
  }

  requestDataView(sub: number, dataView: DataViewSpec): number {
    const id = this.nextReq++;
    const s = this.subs.get(sub);
    if (!s) return id;
    s.dataView = dataView;
    const resolved = this.resolvedDataView(s.view, dataView);
    queueMicrotask(() => {
      if (this.closed || !this.subs.has(sub)) return;
      emit(this.listeners, 'onReply', { type: 'view_reply', request_id: id, sub, view: s.view, data_view: resolved });
      this.deliverSub(sub, false);
    });
    return id;
  }

  sendEvent(_event: ViewerEvent, _sub?: number, _seq?: number, _dataView?: DataViewSpec): void {
    /* static recordings are read-only */
  }

  // --- timeline controls ----------------------------------------------------

  seekIndex(index: number): void {
    const s = this.steps;
    if (!s.length) return;
    const clamped = Math.max(0, Math.min(s.length - 1, Math.trunc(index)));
    if (clamped === this.stepIndex_ && this.seekGen > 1) return;
    this.stepIndex_ = clamped;
    this.seekGen++;
    this.deliverAll();
    this.notifyTimeline();
  }

  seekStep(step: number): void {
    const s = this.steps;
    if (!s.length) return;
    let best = 0;
    for (let i = 0; i < s.length; i++) {
      if (s[i] <= step) best = i;
      else break;
    }
    this.seekIndex(best);
  }

  play(intervalMs = 450): void {
    const s = this.steps;
    if (s.length <= 1 || this.playing_ || this.closed) return;
    if (this.stepIndex_ >= s.length - 1) this.seekIndex(0);
    this.playing_ = true;
    this.notifyTimeline();
    this.playTimer = setInterval(() => {
      if (this.closed || !this.playing_) {
        this.pause();
        return;
      }
      if (this.stepIndex_ + 1 >= this.steps.length) {
        this.pause();
        return;
      }
      this.seekIndex(this.stepIndex_ + 1);
    }, Math.max(50, intervalMs));
  }

  pause(): void {
    if (this.playTimer) clearInterval(this.playTimer);
    this.playTimer = null;
    if (!this.playing_) return;
    this.playing_ = false;
    this.notifyTimeline();
  }

  togglePlay(intervalMs = 450): void {
    if (this.playing_) this.pause();
    else this.play(intervalMs);
  }

  // --- WASM feed:"frames" bridge --------------------------------------------

  /**
   * Connects a WebAssembly view instance running with feed "frames": incoming
   * wire v3 frames (HELLO, SUBSCRIBE, VIEW_REQUEST, PING) are answered over
   * `send`, and every seek pushes updated SNAPSHOT and SERIES frames.
   */
  attachFramePeer(send: (bytes: Uint8Array) => void): FramePeerHandle {
    const peer: FramePeer = { send, greeted: false, sentViews: false, subs: new Map(), nextSeq: 0 };
    this.peers.add(peer);
    return {
      receive: (bytes: Uint8Array) => this.onPeerFrame(peer, bytes),
      detach: () => { this.peers.delete(peer); },
    };
  }

  private onPeerFrame(peer: FramePeer, bytes: Uint8Array): void {
    if (this.closed || !this.peers.has(peer)) return;
    let msg: Message | { type: 'unknown' };
    try {
      msg = decode(bytes);
    } catch {
      return;
    }
    switch (msg.type) {
      case 'hello':
        peer.greeted = true;
        if (this.recording) this.sendPeerHandshake(peer);
        break;
      case 'subscribe':
        peer.subs.set(msg.sub, { view: msg.view, dataView: msg.data_view });
        this.sendPeerFrame(peer, {
          type: 'view_reply',
          request_id: msg.request_id,
          sub: msg.sub,
          view: msg.view,
          data_view: this.resolvedDataView(msg.view, msg.data_view),
        });
        this.deliverPeerSub(peer, msg.sub, msg.view);
        break;
      case 'unsubscribe':
        peer.subs.delete(msg.sub);
        break;
      case 'view_request': {
        const entry = peer.subs.get(msg.sub);
        if (entry) entry.dataView = msg.data_view;
        this.sendPeerFrame(peer, {
          type: 'view_reply',
          request_id: msg.request_id,
          sub: msg.sub,
          view: msg.view,
          data_view: this.resolvedDataView(msg.view, msg.data_view),
        });
        this.deliverPeerSub(peer, msg.sub, msg.view);
        break;
      }
      case 'ping':
        this.sendPeerFrame(peer, { type: 'pong', nonce: msg.nonce });
        break;
      default:
        break;
    }
  }

  private sendPeerHandshake(peer: FramePeer): void {
    if (!this.recording || peer.sentViews) return;
    peer.sentViews = true;
    const sess: SessionIdentity = {
      id: this.recording.session.id,
      script: this.recording.session.script,
      pid: this.recording.session.pid,
      start_unix_ms: this.recording.session.start_unix_ms,
      // Keep "live" on the internal peer so caliper-views does not draw a "session ended" pill over the plot.
      state: 'live',
      views: this.recording.views.length,
    };
    this.sendPeerFrame(peer, { type: 'welcome', role: 'producer', capabilities: [], session: sess, minor: PROTOCOL_MINOR_3_2 });
    this.sendPeerFrame(peer, { type: 'sessions', sessions: [sess] });
    this.sendPeerFrame(peer, { type: 'views', session: sess.id, views: this.recording.views });
  }

  private sendPeerFrame(peer: FramePeer, msg: Message): void {
    if (this.closed || !this.peers.has(peer)) return;
    peer.send(encode(msg, PROTOCOL_MINOR_3_2));
  }

  private deliverPeerSub(peer: FramePeer, sub: number, view: string): void {
    if (!this.recording) return;
    const step = this.currentStep;
    const snaps = this.recording.snapshots.get(view);
    if (snaps && snaps.length) {
      const idx = lastAtOrBefore(snaps, step);
      const base = snaps[idx].msg;
      const out: SnapshotMsg = {
        ...base,
        sub,
        seq: ++peer.nextSeq,
        cursor: this.seekGen,
      };
      this.sendPeerFrame(peer, out);
      return;
    }
    const ser = this.recording.series.get(view);
    if (ser && ser.values.length) {
      const count = seriesCountAt(ser.steps, step);
      const out: SeriesMsg = {
        type: 'series',
        sub,
        view,
        start: 0,
        total: count,
        stamp_ns: ser.stamp_ns,
        values: ser.values.subarray(0, count),
        cursor: this.seekGen,
      };
      this.sendPeerFrame(peer, out);
    }
  }

  // --- loading and Canvas 2D delivery ---------------------------------------

  private async fetchAndLoad(): Promise<void> {
    try {
      const res = await fetch(this.src, { mode: 'cors', credentials: 'omit' });
      if (!res.ok) throw new Error(`HTTP ${res.status} loading ${this.src}`);
      const buf = await res.arrayBuffer();
      if (this.closed) return;
      this.loadBytes(buf);
    } catch (e) {
      if (this.closed) return;
      this.error_ = e instanceof Error ? e.message : String(e);
      this.state = 'ended';
      emit(this.listeners, 'onStatus', 'ended');
      emit(this.listeners, 'onError', { type: 'error', code: 'recording_load', message: this.error_, fatal: true });
      this.notifyTimeline();
      this.resolveReady();
    }
  }

  private loadBytes(buf: Uint8Array | ArrayBuffer): void {
    try {
      const rec = parseRecording(buf, defaultScriptName(this.src));
      this.recording = rec;
      this.session = rec.session;
      this.views = rec.views;
      if (rec.steps.length) {
        if (this.initialStep === 'first') this.stepIndex_ = 0;
        else if (typeof this.initialStep === 'number') {
          let best = 0;
          for (let i = 0; i < rec.steps.length; i++) if (rec.steps[i] <= this.initialStep) best = i;
          this.stepIndex_ = best;
        } else {
          this.stepIndex_ = rec.steps.length - 1;
        }
      } else {
        this.stepIndex_ = 0;
      }
      this.state = 'open';
      emit(this.listeners, 'onStatus', 'open');
      emit(this.listeners, 'onSession', this.session);
      emit(this.listeners, 'onViews', this.views);
      for (const sub of this.subs.keys()) this.deliverSub(sub, true);
      for (const peer of this.peers) {
        if (peer.greeted && !peer.sentViews) this.sendPeerHandshake(peer);
      }
      this.notifyTimeline();
      this.resolveReady();
    } catch (e) {
      this.error_ = e instanceof Error ? e.message : String(e);
      this.state = 'ended';
      emit(this.listeners, 'onStatus', 'ended');
      emit(this.listeners, 'onError', { type: 'error', code: 'bad_recording', message: this.error_, fatal: true });
      this.notifyTimeline();
      this.resolveReady();
    }
  }

  private resolvedDataView(view: string, requested?: DataViewSpec): DataViewSpec {
    const entry = this.views.find((v) => v.name === view);
    return { ...(entry?.default ?? {}), ...(requested ?? {}) };
  }

  private deliverSub(sub: number, withReply: boolean): void {
    const s = this.subs.get(sub);
    if (!s || !this.recording) return;
    if (withReply) {
      emit(this.listeners, 'onReply', {
        type: 'view_reply',
        request_id: this.nextReq++,
        sub,
        view: s.view,
        data_view: this.resolvedDataView(s.view, s.dataView),
      });
    }
    const step = this.currentStep;
    const snaps = this.recording.snapshots.get(s.view);
    if (snaps && snaps.length) {
      const idx = lastAtOrBefore(snaps, step);
      const base = snaps[idx].msg;
      emit(this.listeners, 'onSnapshot', { ...base, sub, seq: ++s.seq, cursor: this.seekGen });
      return;
    }
    const ser = this.recording.series.get(s.view);
    if (ser && ser.values.length) {
      const count = seriesCountAt(ser.steps, step);
      emit(this.listeners, 'onSeries', {
        type: 'series',
        sub,
        view: s.view,
        start: 0,
        total: count,
        stamp_ns: ser.stamp_ns,
        values: ser.values.subarray(0, count),
        cursor: this.seekGen,
      });
    }
  }

  private deliverAll(): void {
    for (const sub of this.subs.keys()) this.deliverSub(sub, false);
    for (const peer of this.peers) {
      for (const [sub, info] of peer.subs) this.deliverPeerSub(peer, sub, info.view);
    }
  }

  private notifyTimeline(): void {
    for (const cb of this.timelineListeners) {
      try {
        cb();
      } catch (e) {
        console.error('caliper-replay timeline listener', e);
      }
    }
  }
}
