// The WebSocket implementation of StreamClient, and the registry that lets
// every element on a page with the same src share one socket.

import { decode, encode, MessageError, WireError } from '../protocol/codec.js';
import {
  CAP, CAP_SCENES, CLOSE, LIMITS, LIMITS_31, type ControlMsg, type ControlReplyMsg, type ControlsMsg, type DataView, type Message, type SelectionMsg,
  type SessionIdentity, type ViewEntry, type ViewerEvent,
} from '../protocol/messages.js';
import {
  emit, type ConnectionState, type ControlAccess, type StreamClient, type StreamListener,
} from './stream.js';

const VERSION = '0.1.0';
type ControlRequest = Omit<ControlMsg, 'type' | 'request_id'>;

interface Sub {
  view: string;
  dataView?: DataView;
  /** SUBSCRIBE was sent on the current connection, for the view of this id. */
  sent: boolean;
  viewId: number;
  subscribeId: number;
}

/** Close codes after which reconnecting cannot help. */
const FINAL_CODES = new Set<number>([
  CLOSE.normal, CLOSE.goingAway, CLOSE.unsupportedMajor, CLOSE.roleRefused, CLOSE.protocolError,
]);

export class SocketClient implements StreamClient {
  state: ConnectionState = 'connecting';
  session: SessionIdentity | null = null;
  views: readonly ViewEntry[] = [];
  /** 3.1: the server's capabilities, its control flag for this viewer, the session's controls and selections. */
  capabilities: readonly string[] = [];
  controls: ControlsMsg | null = null;
  readonly selections = new Map<string, SelectionMsg>();
  private controlFlag = false;
  private readonly replies = new Map<number, (r: ControlReplyMsg) => void>();
  private readonly setTimes = new Map<string, number>();
  private readonly setWaiting = new Map<string, { req: ControlRequest; onReply?: (r: ControlReplyMsg) => void; timer: ReturnType<typeof setTimeout> }>();

  private ws: WebSocket | null = null;
  private readonly listeners = new Set<StreamListener>();
  private readonly subs = new Map<number, Sub>();
  private nextSub = 1;
  private nextRequest = 1;
  private nextNonce = 1;
  /** Attached to a session: subscriptions may be sent. */
  private ready = false;
  private welcomed = false;
  private hub = false;
  private sessions: SessionIdentity[] = [];
  private attached: string | null = null;
  private attachSent: string | null = null;
  private closedByUs = false;
  private ended = false;
  private retries = 0;
  private everOpen = false;
  private lastInput = 0;
  private pinged = false;
  private timer: ReturnType<typeof setInterval> | null = null;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly wantSession: string | null;

  constructor(readonly src: string) {
    let s: string | null = null;
    try {
      s = new URL(src).searchParams.get('session');
    } catch {
      /* a relative or malformed src fails in connect() */
    }
    this.wantSession = s;
  }

  connect(): void {
    if (this.ws || this.closedByUs || this.ended) return;
    this.open();
  }

  close(): void {
    this.closedByUs = true;
    this.stopTimers();
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) {
      try {
        if (this.ws.readyState === WebSocket.OPEN) this.send({ type: 'bye', reason: 'normal', message: '' });
        this.ws.close(CLOSE.normal);
      } catch {
        /* already closing */
      }
    }
    this.ws = null;
    this.setState('ended');
  }

  listen(listener: StreamListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  subscribe(view: string, dataView?: DataView): number {
    const sub = this.nextSub++;
    this.subs.set(sub, { view, dataView, sent: false, viewId: -1, subscribeId: -1 });
    this.sendSub(sub);
    return sub;
  }

  unsubscribe(sub: number): void {
    const s = this.subs.get(sub);
    if (!s) return;
    this.subs.delete(sub);
    if (s.sent && this.ready) this.send({ type: 'unsubscribe', sub });
  }

  requestDataView(sub: number, dataView: DataView): number {
    const s = this.subs.get(sub);
    const id = this.nextRequest++;
    if (!s) return id;
    s.dataView = dataView;
    if (s.sent && this.ready) this.send({ type: 'view_request', request_id: id, sub, view: s.view, data_view: dataView });
    else this.sendSub(sub);
    return id;
  }

  sendEvent(event: ViewerEvent, sub?: number, seq?: number, dataView?: DataView): void {
    if (!this.ready) return;
    this.send({ type: 'event', view: event.view, event: event.type, detail: event.detail, sub, seq, data_view: dataView });
  }

  // --- 3.1: controls ----------------------------------------------------------

  controlAccess(): ControlAccess {
    const session = this.session;
    const sessionCaps = this.hub ? session?.capabilities ?? [] : this.capabilities;
    const shown = this.capabilities.includes(CAP.controls) && sessionCaps.includes(CAP.controls) && this.controls !== null;
    if (!shown) return { shown: false, allowed: false, reason: 'The script has no controls' };
    if (this.state === 'ended' || session?.state === 'ended') return { shown, allowed: false, reason: 'The session has ended' };
    if (!this.controlFlag) {
      return { shown, allowed: false, reason: this.hub ? 'The hub does not allow control (it runs without --allow-control)' : 'This viewer may not control the run' };
    }
    if (!this.ready) return { shown, allowed: false, reason: 'Not connected' };
    return { shown, allowed: true, reason: '' };
  }

  control(req: ControlRequest, onReply?: (reply: ControlReplyMsg) => void): number | null {
    if (!this.controlAccess().allowed) return null;
    if (req.op !== 'set' || !req.name) return this.sendControl(req, onReply);
    // At most 30 sets a second for one parameter, ending with the last value.
    const name = req.name;
    const gap = 1000 / LIMITS_31.controlRateHz;
    const waiting = this.setWaiting.get(name);
    const since = performance.now() - (this.setTimes.get(name) ?? -Infinity);
    if (!waiting && since >= gap) {
      this.setTimes.set(name, performance.now());
      return this.sendControl(req, onReply);
    }
    if (waiting) {
      waiting.req = req;
      waiting.onReply = onReply;
      return 0;
    }
    const timer = setTimeout(() => {
      const w = this.setWaiting.get(name);
      this.setWaiting.delete(name);
      if (!w) return;
      this.setTimes.set(name, performance.now());
      this.sendControl(w.req, w.onReply);
    }, gap - since);
    this.setWaiting.set(name, { req, onReply, timer });
    return 0;
  }

  private sendControl(req: ControlRequest, onReply?: (reply: ControlReplyMsg) => void): number | null {
    if (!this.controlAccess().allowed) return null;
    if (req.op === 'interpret' && (!this.capabilities.includes(CAP.interpretation)
      || (this.hub && !this.session?.capabilities?.includes(CAP.interpretation)))) return null;
    const id = this.nextRequest++;
    if (onReply) this.replies.set(id, onReply);
    try {
      this.send({ ...req, type: 'control', request_id: id });
    } catch (error) {
      this.replies.delete(id);
      throw error;
    }
    return id;
  }

  /** The session changed or went: its controls and selections go too. */
  private dropSessionState(): void {
    for (const w of this.setWaiting.values()) clearTimeout(w.timer);
    this.setWaiting.clear();
    this.replies.clear();
    const had = this.controls !== null || this.selections.size > 0;
    this.controls = null;
    this.selections.clear();
    if (had) emit(this.listeners, 'onControls', null);
  }

  // --- connection -----------------------------------------------------------

  private open(): void {
    this.ready = false;
    this.welcomed = false;
    this.hub = false;
    this.attached = null;
    this.attachSent = null;
    this.capabilities = [];
    this.controlFlag = false;
    this.dropSessionState();
    for (const s of this.subs.values()) s.sent = false;
    let ws: WebSocket;
    try {
      ws = new WebSocket(this.src);
    } catch (e) {
      console.error('caliper-view: cannot open', e);
      this.ended = true;
      this.setState('ended');
      return;
    }
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    ws.onopen = () => {
      this.everOpen = true;
      this.retries = 0;
      this.lastInput = performance.now();
      // 3.1: this viewer shows controls and draws linked selections.
      // 3.2 draft: scenes, so that the producer lists its scene views; the WebAssembly views draw them.
      this.send({ type: 'hello', role: 'viewer', capabilities: [CAP.controls, CAP.selection, CAP.textParams, CAP.interpretation, CAP_SCENES], software: { name: 'caliper-view', version: VERSION } });
      this.startTimers();
    };
    ws.onmessage = (ev) => {
      if (ws !== this.ws) return;
      this.lastInput = performance.now();
      this.pinged = false;
      if (!(ev.data instanceof ArrayBuffer)) {
        ws.close(CLOSE.unsupportedData);
        return;
      }
      this.receive(ev.data);
    };
    ws.onclose = (ev) => {
      if (ws !== this.ws) return;
      this.ws = null;
      this.ready = false;
      this.stopTimers();
      if (this.closedByUs) return;
      if (this.ended || FINAL_CODES.has(ev.code) || (!this.everOpen && this.retries >= 3) || this.retries >= 10) {
        this.ended = true;
        this.setState('ended');
        return;
      }
      this.setState('connecting');
      const delay = Math.min(5000, 500 * 2 ** this.retries++);
      this.retryTimer = setTimeout(() => {
        this.retryTimer = null;
        if (!this.closedByUs && !this.ended) this.open();
      }, delay);
    };
  }

  private startTimers(): void {
    this.stopTimers();
    this.timer = setInterval(() => {
      const idle = performance.now() - this.lastInput;
      if (!this.welcomed && idle > LIMITS.handshakeTimeoutMs) {
        this.ws?.close(CLOSE.handshakeTimeout);
      } else if (idle > LIMITS.idleTimeoutMs) {
        this.ws?.close(CLOSE.idleTimeout);
      } else if (idle > LIMITS.pingIntervalMs && !this.pinged && this.welcomed) {
        this.pinged = true;
        this.send({ type: 'ping', nonce: this.nextNonce++ });
      }
    }, 1000);
  }

  private stopTimers(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.retryTimer = null;
  }

  private send(msg: Message): void {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(encode(msg) as Uint8Array<ArrayBuffer>);
  }

  private setState(s: ConnectionState): void {
    if (this.state === s) return;
    this.state = s;
    emit(this.listeners, 'onStatus', s);
  }

  private sendSub(sub: number): void {
    const s = this.subs.get(sub);
    const v = s && this.views.find((x) => x.name === s.view);
    if (!s || !v || !this.ready) return;
    s.sent = true;
    s.viewId = v.id;
    s.subscribeId = this.nextRequest++;
    this.send({ type: 'subscribe', sub, view: s.view, request_id: s.subscribeId, data_view: s.dataView });
  }

  private setSession(info: SessionIdentity): void {
    const changed = JSON.stringify(info) !== JSON.stringify(this.session);
    this.session = info;
    if (changed) emit(this.listeners, 'onSession', info);
  }

  private becomeReady(): void {
    this.ready = true;
    this.setState('open');
  }

  // --- messages -------------------------------------------------------------

  private receive(buf: ArrayBuffer): void {
    let msg;
    try {
      msg = decode(buf);
    } catch (e) {
      if (e instanceof WireError) {
        console.error('caliper-view:', e.message);
        this.ws?.close(e.closeCode);
      } else if (e instanceof MessageError) {
        console.warn('caliper-view:', e.message);
        this.send({ type: 'error', code: e.code, message: e.message, fatal: false, in_reply_to: e.messageType || undefined });
      } else {
        throw e;
      }
      return;
    }
    switch (msg.type) {
      case 'welcome':
        // The server sends SESSIONS next, and VIEWS once this viewer is attached.
        this.hub = msg.role === 'hub';
        this.welcomed = true;
        this.capabilities = msg.capabilities;
        this.controlFlag = msg.control === true;
        if (msg.session) this.setSession(msg.session);
        break;
      case 'sessions': {
        this.sessions = msg.sessions;
        const mine = this.attached && this.sessions.find((s) => s.id === this.attached);
        if (mine) {
          this.setSession(mine);
          break;
        }
        // A server with one live session attaches a new viewer to it; a viewer
        // chooses only when it was told to, or when there are several.
        const live = this.sessions.filter((s) => s.state === 'live');
        const wanted = this.wantSession ? this.sessions.find((s) => s.id === this.wantSession) : undefined;
        const pick = wanted ?? (live.length > 1 ? live.reduce((a, b) => (b.start_unix_ms > a.start_unix_ms ? b : a)) : undefined);
        if (pick && pick.id !== this.attachSent) {
          this.attachSent = pick.id;
          this.send({ type: 'attach', session: pick.id });
        }
        break;
      }
      case 'views': {
        // A viewer drops its controls and selections when VIEWS names another session or none.
        if ((msg.session ?? null) !== this.attached) this.dropSessionState();
        this.attached = msg.session ?? null;
        const info = this.attached ? this.sessions.find((s) => s.id === this.attached) : undefined;
        if (info) this.setSession(info);
        this.views = this.attached || !this.hub ? msg.views : [];
        const ids = new Map(this.views.map((v) => [v.name, v.id]));
        // A producer ends the subscriptions of a view that disappears or is re-created.
        for (const s of this.subs.values()) if (ids.get(s.view) !== s.viewId) s.sent = false;
        if (this.views.length || this.attached) this.becomeReady();
        emit(this.listeners, 'onViews', this.views);
        for (const [sub, s] of this.subs) if (!s.sent) this.sendSub(sub);
        break;
      }
      case 'snapshot':
        if (this.subs.has(msg.sub)) emit(this.listeners, 'onSnapshot', msg);
        break;
      case 'series':
        if (this.subs.has(msg.sub)) emit(this.listeners, 'onSeries', msg);
        break;
      case 'view_reply': {
        const s = this.subs.get(msg.sub);
        if (!s) break;
        // A failed SUBSCRIBE leaves no subscription; it is sent again when the views change.
        if (msg.error && msg.request_id === s.subscribeId) s.sent = false;
        emit(this.listeners, 'onReply', msg);
        break;
      }
      case 'controls':
        this.controls = msg;
        emit(this.listeners, 'onControls', msg);
        break;
      case 'control_reply': {
        const cb = this.replies.get(msg.request_id);
        // A queued run is answered once more, with its result.
        if (msg.status !== 'queued') this.replies.delete(msg.request_id);
        cb?.(msg);
        break;
      }
      case 'selection':
        this.selections.set(msg.space, msg);
        emit(this.listeners, 'onSelection', msg);
        break;
      case 'ping':
        this.send({ type: 'pong', nonce: msg.nonce });
        break;
      case 'error':
        emit(this.listeners, 'onError', msg);
        break;
      case 'bye':
        this.ended = true;
        break;
      default:
        break;
    }
  }
}

// --- one socket per src -------------------------------------------------------

const shared = new Map<string, { client: SocketClient; refs: number; timer: ReturnType<typeof setTimeout> | null }>();

/** The shared client of src, connected. Elements with the same src share it. */
export function connect(src: string): StreamClient {
  return acquire(src);
}

/** Takes a reference on the shared client of src (released with release()). */
export function acquire(src: string): StreamClient {
  let e = shared.get(src);
  if (!e || e.client.state === 'ended') {
    e = { client: new SocketClient(src), refs: 0, timer: null };
    shared.set(src, e);
  }
  e.refs++;
  if (e.timer) {
    clearTimeout(e.timer);
    e.timer = null;
  }
  e.client.connect();
  return e.client;
}

/** Drops a reference; the socket closes shortly after the last one goes. */
export function release(src: string): void {
  const e = shared.get(src);
  if (!e) return;
  e.refs = Math.max(0, e.refs - 1);
  if (e.refs === 0 && !e.timer) {
    e.timer = setTimeout(() => {
      if (e.refs === 0 && shared.get(src) === e) {
        shared.delete(src);
        e.client.close();
      }
    }, 2000);
  }
}
