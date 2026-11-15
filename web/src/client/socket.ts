// The WebSocket implementation of StreamClient, and the registry that lets
// every element on a page with the same src share one socket.

import { decode, encode, MessageError, WireError } from '../protocol/codec.js';
import {
  CLOSE, LIMITS, type DataView, type Message, type SessionIdentity, type ViewEntry, type ViewerEvent,
} from '../protocol/messages.js';
import { emit, type ConnectionState, type StreamClient, type StreamListener } from './stream.js';

const VERSION = '0.1.0';

interface Sub {
  view: string;
  dataView?: DataView;
  /** SUBSCRIBE was sent on the current connection. */
  sent: boolean;
}

/** Close codes after which reconnecting cannot help. */
const FINAL_CODES = new Set<number>([
  CLOSE.normal, CLOSE.goingAway, CLOSE.unsupportedMajor, CLOSE.roleRefused, CLOSE.protocolError,
]);

export class SocketClient implements StreamClient {
  state: ConnectionState = 'connecting';
  session: SessionIdentity | null = null;
  views: readonly ViewEntry[] = [];

  private ws: WebSocket | null = null;
  private readonly listeners = new Set<StreamListener>();
  private readonly subs = new Map<number, Sub>();
  private nextSub = 1;
  private nextRequest = 1;
  private nextNonce = 1;
  /** WELCOME received (and, from a hub, a session attached). */
  private ready = false;
  private hub = false;
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
    this.subs.set(sub, { view, dataView, sent: false });
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

  // --- connection -----------------------------------------------------------

  private open(): void {
    this.ready = false;
    this.hub = false;
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
      this.send({ type: 'hello', role: 'viewer', capabilities: [], software: { name: 'caliper-view', version: VERSION } });
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
      if (idle > LIMITS.idleTimeoutMs) {
        this.ws?.close(CLOSE.idleTimeout);
      } else if (idle > LIMITS.pingIntervalMs && !this.pinged && this.ready) {
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
    if (!s || !this.ready || !this.views.some((v) => v.name === s.view)) return;
    s.sent = true;
    this.send({ type: 'subscribe', sub, view: s.view, request_id: this.nextRequest++, data_view: s.dataView });
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
        this.hub = msg.role === 'hub';
        if (!this.hub) {
          this.session = msg.session ?? null;
          emit(this.listeners, 'onSession', this.session);
          this.becomeReady();
        }
        break;
      case 'sessions': {
        if (!this.hub) break;
        const list = msg.sessions;
        const current = this.session && list.find((s) => s.id === this.session!.id);
        if (current) {
          this.session = current;
          emit(this.listeners, 'onSession', current);
          break;
        }
        const pick = (this.wantSession && list.find((s) => s.id === this.wantSession))
          || list.find((s) => s.state === 'live') || list[0];
        if (pick) {
          this.session = pick;
          this.send({ type: 'attach', session: pick.id });
          emit(this.listeners, 'onSession', pick);
        }
        break;
      }
      case 'views': {
        if (this.hub && this.session && msg.session && msg.session !== this.session.id) break;
        this.views = msg.views;
        const names = new Set(msg.views.map((v) => v.name));
        for (const s of this.subs.values()) if (!names.has(s.view)) s.sent = false;
        if (!this.ready) this.becomeReady();
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
      case 'view_reply':
        if (this.subs.has(msg.sub)) emit(this.listeners, 'onReply', msg);
        break;
      case 'ping':
        this.send({ type: 'pong', nonce: msg.nonce });
        break;
      case 'error':
        if (msg.code === 'unknown_view' && msg.sub !== undefined) {
          const s = this.subs.get(msg.sub);
          if (s) s.sent = false;
        }
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
