// The <caliper-view> element: one live view of a Caliper stream, with a header
// (name, data view picker, status dot) and a drawing area. Elements with the
// same src share one socket; the `source` property takes any StreamClient.

import { acquire, release } from './client/socket.js';
import type { StreamClient, StreamListener } from './client/stream.js';
import {
  KIND_IDS, kindName, type DataView, type Kind, type SeriesMsg, type SnapshotMsg, type ViewEntry,
} from './protocol/messages.js';
import { adapterInfo, gpuDevice, onDeviceLost } from './render/gpu.js';
import type { LiveView, ViewHost } from './views/base.js';
import { HeatmapView } from './views/heatmap.js';
import { HistogramView } from './views/histogram.js';
import { Picker } from './views/picker.js';
import { PointsView } from './views/points.js';
import { SeriesView } from './views/series.js';
import { C, FONT_MONO, FONT_TEXT } from './views/theme.js';

const STYLE = `
:host { display: block; height: var(--caliper-view-height, 320px); min-width: 220px; color: ${C.parchment};
  font: 13px/1.4 ${FONT_TEXT}; -webkit-font-smoothing: antialiased; }
:host([hidden]) { display: none; }
* { box-sizing: border-box; }
.plate { height: 100%; display: flex; flex-direction: column; background: ${C.plate}; border: 1px solid ${C.ruleFaint};
  border-radius: 6px; overflow: hidden; }
.head { display: flex; align-items: center; flex-wrap: wrap; gap: 4px 10px; padding: 5px 10px; min-height: 33px;
  border-bottom: 1px solid ${C.ruleFaint}; }
.name { font-size: 13.5px; color: ${C.parchment}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 18em; }
.picker { display: contents; }
.picker label { display: inline-flex; align-items: center; gap: 4px; color: ${C.faint}; font-size: 12px; }
select, input { font: 12px ${FONT_TEXT}; color: ${C.parchment}; background: #1c1b18; border: 1px solid ${C.rule};
  border-radius: 3px; padding: 1px 3px; height: 21px; }
input { width: 5.5em; font-family: ${FONT_MONO}; }
select:focus-visible, input:focus-visible, button:focus-visible { outline: 1px solid ${C.bronze}; outline-offset: 1px; }
.info { margin-left: auto; color: ${C.faint}; font: 11px ${FONT_MONO}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }
.tools { display: inline-flex; gap: 4px; }
.tool { font: 11px ${FONT_TEXT}; letter-spacing: .06em; color: ${C.faint}; background: transparent; border: 1px solid ${C.rule};
  border-radius: 3px; padding: 1px 7px; height: 21px; cursor: pointer; }
.tool:hover { color: ${C.parchment}; }
.tool[aria-pressed="true"] { color: ${C.bronze}; border-color: ${C.bronze}; }
.dot { width: 8px; height: 8px; border-radius: 50%; flex: none; border: 1px solid ${C.faint}; }
.dot[data-state="connected"] { background: ${C.bronze}; border-color: ${C.bronze}; }
.dot[data-state="ended"] { background: ${C.rule}; border-color: ${C.rule}; }
.body { position: relative; flex: 1; min-height: 0; }
.body canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.body canvas.overlay { z-index: 1; cursor: crosshair; touch-action: none; }
.tip { position: absolute; z-index: 2; pointer-events: none; background: rgba(24, 23, 21, .94); border: 1px solid ${C.rule};
  border-radius: 4px; padding: 4px 7px; font: 11.5px/1.45 ${FONT_MONO}; color: ${C.parchment}; white-space: pre; }
.msg { position: absolute; inset: 0; z-index: 2; display: flex; align-items: center; justify-content: center; padding: 16px;
  text-align: center; color: ${C.faint}; font-size: 13px; pointer-events: none; }
.msg.error { color: ${C.secondary}; align-items: flex-end; }
.msg.error span { background: rgba(24, 23, 21, .94); border: 1px solid ${C.rule}; border-radius: 4px; padding: 5px 9px; }
`;

const NEEDS_GPU: ReadonlySet<Kind> = new Set(['heatmap', 'image', 'points']);
const HOVER_MS = 100;

export class CaliperView extends HTMLElement {
  static get observedAttributes(): string[] {
    return ['src', 'view', 'kind', 'height', 'renderer'];
  }

  private sourceProp: StreamClient | null = null;
  private client: StreamClient | null = null;
  private acquired: string | null = null;
  private unlisten: (() => void) | null = null;
  private names: string[] = [];
  private subs: number[] = [];
  private live: LiveView | null = null;
  private liveKind: Kind | null = null;
  private creating: Kind | null = null;
  private pending: SnapshotMsg | null = null;
  private current: SnapshotMsg | null = null;
  private resolved: DataView | null = null;
  private wanted: DataView | null = null;
  private gotData = false;
  private dirtyData = false;
  private raf = 0;
  private device: GPUDevice | null = null;
  private hoverTimer: ReturnType<typeof setTimeout> | null = null;
  private hoverQueued: Record<string, unknown> | null = null;
  private lastHover = 0;
  private msgTimer: ReturnType<typeof setTimeout> | null = null;
  private generation = 0;
  private built = false;
  private ro: ResizeObserver | null = null;
  private offLost: (() => void) | null = null;
  private readonly picker = new Picker((dv) => this.pick(dv));
  private readonly root: ShadowRoot;
  private readonly parts: {
    name: HTMLElement; info: HTMLElement; tools: HTMLElement; dot: HTMLElement; body: HTMLElement;
    overlay: HTMLCanvasElement; tip: HTMLElement; msg: HTMLElement;
  };

  constructor() {
    super();
    this.root = this.attachShadow({ mode: 'open' });
    const el = <K extends keyof HTMLElementTagNameMap>(tag: K, cls: string) => {
      const e = document.createElement(tag);
      e.className = cls;
      return e;
    };
    this.parts = {
      name: el('span', 'name'), info: el('span', 'info'), tools: el('span', 'tools'), dot: el('span', 'dot'),
      body: el('div', 'body'), overlay: el('canvas', 'overlay'), tip: el('div', 'tip'), msg: el('div', 'msg'),
    };
  }

  // --- public interface -------------------------------------------------------

  /** A StreamClient to read from instead of the socket named by src. */
  get source(): StreamClient | null {
    return this.sourceProp;
  }

  set source(c: StreamClient | null) {
    this.sourceProp = c;
    if (this.isConnected) this.reattach();
  }

  /** The resolved data view of the view, or the one asked for while none is resolved. */
  get dataView(): DataView | null {
    return this.resolved ?? this.wanted;
  }

  /** Asks the producer for a data view, as the picker does. */
  set dataView(dv: DataView | null) {
    const text = JSON.stringify(dv);
    if (!dv || text === JSON.stringify(this.resolved) || text === JSON.stringify(this.wanted)) return;
    this.wanted = dv;
    if (this.client && this.subs.length) this.client.requestDataView(this.subs[0], dv);
  }

  /** 'webgpu' or 'canvas2d' once the live view exists. */
  get renderer(): string | null {
    return this.live ? String(this.live.debug().renderer) : null;
  }

  /** Read-only state for tests and debugging. */
  debugState(): Record<string, unknown> {
    return {
      view: this.names.join(','), subs: [...this.subs], status: this.parts.dot.dataset.state, dataView: this.dataView,
      adapter: this.device ? adapterInfo() : null, ...(this.live?.debug() ?? {}),
    };
  }

  /** Resolves once pending drawing is done, including the GPU's work. */
  async settled(): Promise<void> {
    while (this.raf || this.creating) await new Promise((r) => requestAnimationFrame(() => r(null)));
    if (this.device) await this.device.queue.onSubmittedWorkDone();
  }

  // --- lifecycle --------------------------------------------------------------

  connectedCallback(): void {
    if (!this.built) this.build();
    this.applyHeight();
    this.reattach();
    this.offLost = onDeviceLost(() => {
      if (this.liveKind && NEEDS_GPU.has(this.liveKind)) this.recreate();
    });
  }

  disconnectedCallback(): void {
    this.detach();
    this.destroyLive();
    this.offLost?.();
    this.offLost = null;
  }

  attributeChangedCallback(name: string, old: string | null, value: string | null): void {
    if (!this.isConnected || old === value) return;
    if (name === 'height') this.applyHeight();
    else if (name === 'src' || name === 'view') this.reattach();
    else if (name === 'kind' || name === 'renderer') this.recreate();
  }

  private build(): void {
    this.built = true;
    const style = document.createElement('style');
    style.textContent = STYLE;
    const plate = document.createElement('div');
    plate.className = 'plate';
    const head = document.createElement('div');
    head.className = 'head';
    const p = this.parts;
    p.dot.setAttribute('role', 'status');
    head.append(p.name, this.picker.el, p.info, p.tools, p.dot);
    p.tip.hidden = true;
    p.body.append(p.overlay, p.tip, p.msg);
    plate.append(head, p.body);
    this.root.append(style, plate);
    this.ro = new ResizeObserver(() => {
      this.live?.resize();
      this.request('data');
    });
    this.ro.observe(p.body);
    document.fonts?.ready.then(() => this.request('data'));
  }

  private applyHeight(): void {
    const h = Number(this.getAttribute('height'));
    if (h > 0) this.style.setProperty('--caliper-view-height', `${h}px`);
    else this.style.removeProperty('--caliper-view-height');
  }

  // --- the stream -------------------------------------------------------------

  private reattach(): void {
    this.detach();
    this.destroyLive();
    this.gotData = false;
    this.current = this.pending = null;
    this.resolved = null;
    const gen = ++this.generation;
    const view = this.getAttribute('view') ?? '';
    const src = this.getAttribute('src');
    let client = this.sourceProp;
    if (!client && src) {
      client = acquire(src);
      this.acquired = src;
    }
    this.parts.name.textContent = view;
    this.parts.name.title = view;
    this.client = client;
    if (!client || !view) {
      this.status();
      this.message(client ? 'No view given' : 'Not connected');
      return;
    }
    const listener: StreamListener = {
      onStatus: () => this.status(),
      onViews: (views) => this.views(views),
      onSnapshot: (s) => this.snapshot(s),
      onSeries: (d) => this.seriesDelta(d),
      onReply: (r) => {
        if (r.sub !== this.subs[0]) return;
        if (r.error) {
          this.message(`${r.error.message ?? r.error.code}`, true);
          if (this.resolved) this.picker.show(this.resolved);
        } else if (r.data_view) {
          this.resolved = r.data_view;
          this.wanted = null;
          this.picker.show(r.data_view);
        }
      },
      onError: (e) => {
        if (e.sub !== undefined && this.subs.includes(e.sub) && e.code !== 'unknown_view') this.message(e.message || e.code, true);
      },
    };
    this.unlisten = client.listen(listener);
    client.connect();
    if (gen !== this.generation) return;
    // Several series may share one plate: view="loss,val_loss".
    const known = (n: string) => client!.views.some((v) => v.name === n);
    this.names = view.includes(',') && !known(view) ? view.split(',').map((s) => s.trim()).filter(Boolean) : [view];
    this.subs = this.names.map((n, i) => client!.subscribe(n, i === 0 ? this.wanted ?? undefined : undefined));
    this.views(client.views);
    this.status();
  }

  private detach(): void {
    if (this.client) for (const s of this.subs) this.client.unsubscribe(s);
    this.subs = [];
    this.unlisten?.();
    this.unlisten = null;
    if (this.acquired) release(this.acquired);
    this.acquired = null;
    this.client = null;
  }

  private entry(): ViewEntry | undefined {
    return this.client?.views.find((v) => v.name === this.names[0]);
  }

  private kind(): Kind | null {
    const attr = this.getAttribute('kind') as Kind | null;
    if (attr && attr in KIND_IDS) return attr;
    const e = this.entry();
    if (e) return kindName(e.kind);
    if (this.current) return kindName(this.current.kind);
    return null;
  }

  private views(_views: readonly ViewEntry[]): void {
    const e = this.entry();
    this.picker.offer(e);
    if (this.resolved) this.picker.show(this.resolved);
    const k = this.kind();
    if (k) this.ensureLive(k);
    if (this.gotData) return;
    const missing = !e && this.client?.state === 'open';
    this.message(missing ? `No view named ${this.names.join(', ')} yet` : 'Waiting for data');
  }

  private snapshot(s: SnapshotMsg): void {
    if (s.sub !== this.subs[0]) return;
    this.pending = s;
    if (!this.gotData) {
      this.gotData = true;
      this.message(null);
      this.status();
    }
    if (s.data_view && JSON.stringify(s.data_view) !== JSON.stringify(this.resolved) && !this.wanted) {
      this.resolved = s.data_view;
      this.picker.show(s.data_view);
    }
    const k = this.kind() ?? kindName(s.kind);
    if (k) this.ensureLive(k);
    this.request('data');
  }

  private seriesDelta(d: SeriesMsg): void {
    const curve = this.subs.indexOf(d.sub);
    if (curve < 0) return;
    if (!this.gotData) {
      this.gotData = true;
      this.message(null);
      this.status();
    }
    this.ensureLive('series');
    this.live?.series?.(d, curve);
    this.request('data');
  }

  private status(): void {
    const c = this.client;
    const state = !c || c.state === 'ended' ? 'ended' : c.state === 'open' && this.gotData ? 'connected' : 'waiting';
    const dot = this.parts.dot;
    if (dot.dataset.state !== state) {
      dot.dataset.state = state;
      dot.title = state === 'connected' ? 'Connected' : state === 'waiting' ? 'Waiting for data' : 'Ended';
      dot.setAttribute('aria-label', dot.title);
    }
    if (state === 'ended' && !this.gotData) this.message('The stream has ended');
  }

  private pick(dv: DataView): void {
    this.wanted = dv;
    if (this.client && this.subs.length) this.client.requestDataView(this.subs[0], dv);
    this.dispatchEvent(new CustomEvent('caliper-dataview', { detail: dv, bubbles: true, composed: true }));
  }

  // --- the live view ------------------------------------------------------------

  private ensureLive(kind: Kind): void {
    if (this.liveKind === kind || this.creating === kind) return;
    this.destroyLive();
    const wantGpu = NEEDS_GPU.has(kind) && this.getAttribute('renderer') !== 'canvas';
    if (!wantGpu) {
      this.makeLive(kind, null);
      return;
    }
    this.creating = kind;
    const gen = this.generation;
    gpuDevice().then((device) => {
      if (gen !== this.generation || this.creating !== kind) return;
      this.creating = null;
      this.makeLive(kind, device);
    });
  }

  private makeLive(kind: Kind, device: GPUDevice | null): void {
    this.device = device;
    const host = this.host();
    let live: LiveView;
    try {
      live = kind === 'series' ? new SeriesView(host, this.names)
        : kind === 'histogram' ? new HistogramView(host)
          : kind === 'points' ? new PointsView(host)
            : new HeatmapView(host, kind === 'image');
    } catch (e) {
      if (!device) throw e;
      console.warn('caliper-view: WebGPU setup failed; drawing with Canvas 2D', e);
      this.makeLive(kind, null);
      return;
    }
    this.live = live;
    this.liveKind = kind;
    if (this.current && !this.pending) this.pending = this.current;
    this.request('data');
  }

  private destroyLive(): void {
    this.creating = null;
    if (!this.live) return;
    this.live.destroy();
    this.live = null;
    this.liveKind = null;
    this.parts.tools.replaceChildren();
    this.parts.info.textContent = '';
    this.tooltip(null);
  }

  private recreate(): void {
    const k = this.liveKind ?? this.creating;
    this.destroyLive();
    const kind = this.kind() ?? k;
    if (kind) this.ensureLive(kind);
  }

  private host(): ViewHost {
    const p = this.parts;
    return {
      name: this.names.join(', '),
      body: p.body,
      overlay: p.overlay,
      tools: p.tools,
      device: this.device,
      size: () => {
        const r = p.body.getBoundingClientRect();
        return { w: Math.max(1, r.width), h: Math.max(1, r.height), dpr: window.devicePixelRatio || 1 };
      },
      request: (what) => this.request(what),
      tooltip: (text, x, y) => this.tooltip(text, x, y),
      report: (type, detail) => this.report(type, detail),
      info: (text) => {
        p.info.textContent = text;
        p.info.title = text;
      },
    };
  }

  private request(what: 'data' | 'overlay'): void {
    if (what === 'data') this.dirtyData = true;
    if (!this.raf) this.raf = requestAnimationFrame(() => this.frame());
  }

  private frame(): void {
    this.raf = 0;
    const live = this.live;
    if (!live) return;
    if (this.pending) {
      const s = this.pending;
      this.pending = null;
      this.current = s;
      live.snapshot(s);
      this.dirtyData = true;
    }
    const data = this.dirtyData;
    this.dirtyData = false;
    live.draw(data);
  }

  // --- tooltips, messages and events --------------------------------------------

  private tooltip(text: string | null, x = 0, y = 0): void {
    const tip = this.parts.tip;
    if (text === null) {
      tip.hidden = true;
      return;
    }
    tip.textContent = text;
    tip.hidden = false;
    const bw = this.parts.body.clientWidth, bh = this.parts.body.clientHeight;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = x + 14, top = y + 14;
    if (left + tw > bw - 4) left = Math.max(4, x - tw - 10);
    if (top + th > bh - 4) top = Math.max(4, y - th - 10);
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  }

  private message(text: string | null, error = false): void {
    const m = this.parts.msg;
    if (this.msgTimer) clearTimeout(this.msgTimer);
    this.msgTimer = null;
    m.replaceChildren();
    m.className = error ? 'msg error' : 'msg';
    if (!text) return;
    const span = document.createElement('span');
    span.textContent = text;
    m.append(span);
    if (error) this.msgTimer = setTimeout(() => this.message(this.gotData ? null : 'Waiting for data'), 5000);
  }

  private report(type: 'select' | 'pick' | 'hover', detail: Record<string, unknown>): void {
    const view = this.names[0];
    const event = { view, type, detail };
    if (type === 'select') this.dispatchEvent(new CustomEvent('caliper-select', { detail: event, bubbles: true, composed: true }));
    if (type === 'hover') {
      // Hovers reach the producer at most every HOVER_MS, the newest one last.
      this.hoverQueued = detail;
      if (this.hoverTimer) return;
      const wait = Math.max(0, HOVER_MS - (performance.now() - this.lastHover));
      this.hoverTimer = setTimeout(() => {
        this.hoverTimer = null;
        this.lastHover = performance.now();
        const d = this.hoverQueued;
        this.hoverQueued = null;
        if (d) this.send({ view, type: 'hover', detail: d });
      }, wait);
      return;
    }
    this.send(event);
  }

  private send(event: { view: string; type: 'select' | 'pick' | 'hover'; detail: Record<string, unknown> }): void {
    const s = this.current;
    this.client?.sendEvent(event, this.subs[0], s?.seq, s?.data_view ?? this.resolved ?? undefined);
  }
}

export function define(): void {
  if (!customElements.get('caliper-view')) customElements.define('caliper-view', CaliperView);
}
