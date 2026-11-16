// The <caliper-view> element: one live view of a Caliper stream, with a header
// (name, data view picker, status dot) and a drawing area. Elements with the
// same src share one socket; the `source` property takes any StreamClient.
//
// What draws the area is decided per view once its kind is known: with
// renderer="auto" (the default), Caliper's own views compiled to WebAssembly
// (wasm.ts) draw the kinds their loader lists in CALIPER_VIEWS_KINDS when
// WebGPU gives an adapter, and this element's TypeScript views draw the rest,
// so one page can mix both. The header is the element's in both cases: with
// the WebAssembly views the shared socket carries only the metadata (VIEWS,
// no subscription), the module streams the view on its own connection, the
// picker sends its choice through the module's setDataView, and the module's
// data view reply updates the picker. A selection, pick or hover in a
// WebAssembly view goes to the producer on the module's connection, with the
// same detail the TypeScript views send, and its selections come out of the
// element as caliper-select events just the same; the header line about the
// view comes from the facts the module reports.

import { acquire, release } from './client/socket.js';
import type { StreamClient, StreamListener } from './client/stream.js';
import {
  KIND_IDS, kindName, type DataView, type Kind, type SeriesMsg, type SnapshotMsg, type ViewEntry,
} from './protocol/messages.js';
import { adapterInfo, gpuDevice, onDeviceLost } from './render/gpu.js';
import type { LiveView, ViewHost } from './views/base.js';
import { HeatmapView } from './views/heatmap.js';
import { HistogramView } from './views/histogram.js';
import { toggle } from './views/base.js';
import { Picker } from './views/picker.js';
import { PointsView } from './views/points.js';
import { SeriesView } from './views/series.js';
import { assetBase, hasAdapter, mountWasm, wasmKinds, type WasmHandle } from './wasm.js';
import { factsInfo } from './views/info.js';
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
.plate[hidden], .wasm[hidden], canvas.overlay[hidden] { display: none; }
.wasm { position: absolute; inset: 0; }
.tip { position: absolute; z-index: 2; pointer-events: none; background: rgba(24, 23, 21, .94); border: 1px solid ${C.rule};
  border-radius: 4px; padding: 4px 7px; font: 11.5px/1.45 ${FONT_MONO}; color: ${C.parchment}; white-space: pre; }
.msg { position: absolute; inset: 0; z-index: 2; display: flex; align-items: center; justify-content: center; padding: 16px;
  text-align: center; color: ${C.faint}; font-size: 13px; pointer-events: none; }
.msg.error { color: ${C.secondary}; align-items: flex-end; }
.msg.error span { background: rgba(24, 23, 21, .94); border: 1px solid ${C.rule}; border-radius: 4px; padding: 5px 9px; }
`;

const NEEDS_GPU: ReadonlySet<Kind> = new Set(['heatmap', 'image', 'points']);
const HOVER_MS = 1000 / 30;

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
  private lastHover = 0;
  private msgTimer: ReturnType<typeof setTimeout> | null = null;
  private generation = 0;
  private built = false;
  private ro: ResizeObserver | null = null;
  private offLost: (() => void) | null = null;
  private wasm: WasmHandle | null = null;
  private wasmReady = false;
  /** The header toggles of a view the WebAssembly views draw, by name. */
  private readonly wasmToggles = new Map<string, { set(v: boolean): void }>();
  /** null until the view's kind is known; then which views draw it. */
  private drawer: 'wasm' | 'ts' | null = null;
  private deciding = false;
  private readonly picker = new Picker((dv) => this.pick(dv));
  private readonly root: ShadowRoot;
  private readonly parts: {
    name: HTMLElement; info: HTMLElement; tools: HTMLElement; dot: HTMLElement; body: HTMLElement;
    overlay: HTMLCanvasElement; tip: HTMLElement; msg: HTMLElement; plate: HTMLElement; wasm: HTMLElement;
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
      plate: el('div', 'plate'), wasm: el('div', 'wasm'),
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
    if (this.wasm) this.wasm.setDataView(dv);
    else if (this.client && this.subs.length) this.client.requestDataView(this.subs[0], dv);
  }

  /**
   * What draws the view: 'wasm' (Caliper's own view layer compiled to
   * WebAssembly), or the element's own views with 'webgpu' or 'canvas2d'.
   */
  get renderer(): string | null {
    if (this.wasm) return 'wasm';
    return this.live ? String(this.live.debug().renderer) : null;
  }

  /** Read-only state for tests and debugging. */
  debugState(): Record<string, unknown> {
    return {
      view: this.names.join(','), renderer: this.renderer, drawer: this.drawer, wasmReady: this.wasmReady,
      subs: [...this.subs], status: this.parts.dot.dataset.state, dataView: this.dataView,
      adapter: this.device ? adapterInfo() : null, wasmStats: this.wasm?.stats ?? null, info: this.parts.info.textContent,
      ...(this.wasm ? this.wasmDebug() : {}), ...(this.live?.debug() ?? {}),
    };
  }

  /** A WebAssembly view's counters and state under the names the TypeScript views use. */
  private wasmDebug(): Record<string, unknown> {
    const s = (this.wasm?.stats ?? {}) as Record<string, unknown>;
    const out: Record<string, unknown> = { kind: this.kind(), updates: s.updates ?? 0, selected: s.selected ?? 0 };
    let geo: { area: number[]; window: number[]; hover: number; cols: number; follow: boolean; log: boolean } | undefined;
    try {
      geo = JSON.parse(String(s.geometry ?? '{}'))[this.names[0]];
    } catch {
      geo = undefined;
    }
    if (geo) {
      const [x, y, w, h] = geo.area, [x0, x1, y0, y1] = geo.window;
      out.area = { x, y, w, h };
      out.window = { x0, x1, y0, y1 };
      out.follow = geo.follow;
      out.logY = geo.log;
      const k = this.kind();
      out.hover = geo.hover < 0 ? null
        : k === 'heatmap' || k === 'image' ? { r: Math.floor(geo.hover / geo.cols), c: geo.hover % geo.cols } : geo.hover;
    }
    return out;
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
    else if (name === 'src' || name === 'view' || name === 'renderer') this.reattach();
    else if (name === 'kind') this.wasm ? this.reattach() : this.recreate();
  }

  private build(): void {
    this.built = true;
    const style = document.createElement('style');
    style.textContent = STYLE;
    const plate = this.parts.plate;
    const head = document.createElement('div');
    head.className = 'head';
    const p = this.parts;
    p.dot.setAttribute('role', 'status');
    head.append(p.name, this.picker.el, p.info, p.tools, p.dot);
    p.tip.hidden = true;
    p.wasm.hidden = true;
    p.body.append(p.wasm, p.overlay, p.tip, p.msg);
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
    this.drawer = null;
    this.deciding = false;
    this.wasmReady = false;
    const gen = ++this.generation;
    const view = this.getAttribute('view') ?? '';
    const src = this.getAttribute('src');
    this.attach(gen, view, src);
  }

  /**
   * Decides, once the view's kind is known, what draws it: renderer="auto"
   * (the default) uses the WebAssembly views for the kinds their loader lists
   * when WebGPU gives an adapter; "wasm" uses them for those kinds whatever
   * the adapter; "ts" uses this element's own views. The element's own views
   * also draw a stream given through the source property, several series in
   * one plate, and anything the WebAssembly views fail to mount.
   */
  private async decide(gen: number): Promise<void> {
    const kind = this.kind();
    if (this.drawer || this.deciding || !kind) return;
    this.deciding = true;
    const mode = this.getAttribute('renderer') ?? 'auto';
    const src = this.getAttribute('src');
    const base = src && assetBase(src);
    let handle: WasmHandle | null = null;
    if (!this.sourceProp && base && this.names.length === 1 && (mode === 'auto' || mode === 'wasm')) {
      const kinds = await wasmKinds(base);
      const ok = !!kinds && kinds.includes(kind) && (mode === 'wasm' || (await hasAdapter()));
      if (ok && gen === this.generation) handle = await this.mountWasm(src!, base, kind);
    }
    if (gen !== this.generation) {
      handle?.destroy();
      return;
    }
    this.deciding = false;
    if (handle) {
      this.useWasm(handle);
      return;
    }
    this.drawer = 'ts';
    this.subscribe();
    this.views(this.client?.views ?? []);
  }

  private mountWasm(src: string, base: string, kind: Kind): Promise<WasmHandle | null> {
    const view = this.names[0];
    this.parts.wasm.hidden = false;
    return mountWasm(this.parts.wasm, {
      src, view, kind, header: false, assetBase: base,
      // The module sent the selection to the producer itself; the page hears it here.
      onSelect: (e) => {
        const event = 'detail' in e && 'type' in e ? e : { view, type: 'select', detail: e };
        this.dispatchEvent(new CustomEvent('caliper-select', { detail: event, bubbles: true, composed: true }));
      },
      onInfo: (facts) => {
        if (this.drawer !== 'wasm' && !this.deciding) return;
        const text = factsInfo(facts);
        if (text !== null) this.parts.info.textContent = text;
      },
      onToggle: (name, on) => this.wasmToggles.get(name)?.set(on),
      // The reply to the module's own subscription or data view request.
      onDataView: (dv) => {
        if (this.drawer !== 'wasm') return;
        this.resolved = dv;
        this.wanted = null;
        this.picker.show(dv);
      },
    });
  }

  private useWasm(handle: WasmHandle): void {
    this.wasm = handle;
    this.drawer = 'wasm';
    this.wasmTools(handle);
    this.parts.overlay.hidden = true;
    this.parts.wasm.hidden = false;
    if (this.wanted) handle.setDataView(this.wanted);
    const gen = this.generation;
    const ready = handle.ready ?? Promise.resolve();
    ready.then(() => {
      if (gen !== this.generation) return;
      this.wasmReady = true;
      this.message(null);
      this.status();
    });
    this.message('Waiting for data');
    this.status();
  }

  /**
   * The header toggles of a series or histogram the WebAssembly views draw,
   * the same buttons with the same names as the TypeScript views': log for
   * both, and follow for a series, which the view turns off itself when the
   * user pans or zooms and on again on a double click.
   */
  private wasmTools(handle: WasmHandle): void {
    const kind = this.kind();
    if (!handle.setToggle || (kind !== 'series' && kind !== 'histogram')) return;
    const set = (name: string) => (on: boolean) => handle.setToggle?.(name, on);
    const host = { tools: this.parts.tools };
    this.wasmToggles.set('log', toggle(host, 'log', kind === 'series' ? 'Log scale for y' : 'Log scale for counts', false, set('log')));
    if (kind === 'series') this.wasmToggles.set('follow', toggle(host, 'follow', 'Follow the newest values', true, set('follow')));
  }

  /** The stream for the header: the element's shared client, without subscriptions until it draws itself. */
  private attach(gen: number, view: string, src: string | null): void {
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
    this.views(client.views);
    this.status();
  }

  /** The element's own views draw: subscribe on the shared client. */
  private subscribe(): void {
    const client = this.client;
    if (!client) return;
    this.subs = this.names.map((n, i) => client.subscribe(n, i === 0 ? this.wanted ?? undefined : undefined));
  }

  private detach(): void {
    if (this.wasm) {
      try {
        this.wasm.destroy();
      } catch (e) {
        console.warn('caliper-view:', e);
      }
      this.wasm = null;
      this.parts.info.textContent = '';
      this.parts.tools.replaceChildren();
      this.wasmToggles.clear();
    }
    this.parts.wasm.replaceChildren();
    this.parts.wasm.hidden = true;
    this.parts.overlay.hidden = false;
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
    if (!this.drawer) this.decide(this.generation);
    const k = this.kind();
    if (k && this.drawer === 'ts') this.ensureLive(k);
    if (this.gotData || this.wasmReady) return;
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
    const drawn = this.drawer === 'wasm' ? this.wasmReady : this.gotData;
    const state = !c || c.state === 'ended' ? 'ended' : c.state === 'open' && drawn ? 'connected' : 'waiting';
    const dot = this.parts.dot;
    if (dot.dataset.state !== state) {
      dot.dataset.state = state;
      dot.title = state === 'connected' ? 'Connected' : state === 'waiting' ? 'Waiting for data' : 'Ended';
      dot.setAttribute('aria-label', dot.title);
    }
    if (state === 'ended' && !drawn) this.message('The stream has ended');
  }

  private pick(dv: DataView): void {
    this.wanted = dv;
    if (this.wasm) this.wasm.setDataView(dv);
    else if (this.client && this.subs.length) this.client.requestDataView(this.subs[0], dv);
    this.dispatchEvent(new CustomEvent('caliper-dataview', { detail: dv, bubbles: true, composed: true }));
  }

  // --- the live view ------------------------------------------------------------

  private ensureLive(kind: Kind): void {
    if (this.liveKind === kind || this.creating === kind) return;
    this.destroyLive();
    const wantGpu = NEEDS_GPU.has(kind);
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
      // At most 30 hovers per second reach the producer; a hover inside that
      // interval is dropped, not queued (wire protocol, Events).
      const now = performance.now();
      if (now - this.lastHover < HOVER_MS) return;
      this.lastHover = now;
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
