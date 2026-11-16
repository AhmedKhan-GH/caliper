// The browser viewer of spec applets: the page that Caliper API serves at
// /v0/spec?token=<token>&applet=<id>. It reads the token and the applet's id
// from its own address, as the viewer page reads the stream's, asks the app
// for the saved spec (spec.get) and for each shown panel's data (panel.data),
// and draws them with the native runtime's layout and look. The app resolves
// every panel, transforms included; the page only draws. Every request goes
// out from the viewer origin, so the command layer runs reads only for it.
//
// Refresh: once a second the page asks for the spec again and, for each panel
// it shows, for its data with the version it has; the app answers "unchanged"
// without resolving until the panel's sources, the spec or the catalog change.

import { computeLayout, effectiveLayout, findPanel, metricsForTextScale, panelTitle, type SpecLayout, type TabChoice } from './layout.js';
import { COL, ColorKey } from './look.js';
import { drawPanel, drawState, drawTabs, type PanelView } from './draw.js';
import { preparePanel, type PreparedPanel } from './prepare.js';
import type { PanelDataResult, Spec } from './types.js';

export interface CommandEnvelope { ok: boolean; result?: any; error?: { code: string; message: string; where: string } }

/** POST /v0/commands/<name> from the viewer origin. */
export async function command(base: URL, token: string, name: string, args: Record<string, unknown>): Promise<CommandEnvelope> {
  const res = await fetch(new URL(`/v0/commands/${name}`, base), {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json', 'X-Caliper-Origin': 'viewer' },
    body: JSON.stringify(args),
    cache: 'no-store',
  });
  if (res.status !== 200) {
    let message = `HTTP ${res.status}`;
    try {
      message = (await res.json()).error ?? message;
    } catch { /* not JSON */ }
    return { ok: false, error: { code: `http.${res.status}`, message, where: '' } };
  }
  return (await res.json()) as CommandEnvelope;
}

export interface PageParts { title: HTMLElement; area: HTMLElement; canvas: HTMLCanvasElement; status: HTMLElement; list: HTMLElement }

interface PanelEntry { version: string; prepared: PreparedPanel | null; busy: boolean; view: PanelView }

/** What a test reads of the page. */
export interface SpecPageState {
  status: string;
  applet: string;
  title: string;
  panels: { id: string; title: string; primitive: string; drawn: boolean; state: string | null; info: string;
            rect: { x: number; y: number; w: number; h: number } }[];
  tabs: { path: string; labels: string[]; active: number }[];
  requests: number;
}

export class SpecPage {
  private spec: Spec | null = null;
  private specText = '';
  private key: ColorKey | null = null;
  private layout: SpecLayout = { panels: [], tabs: [] };
  private readonly panels = new Map<string, PanelEntry>();
  private readonly tabs: TabChoice = {};
  private tabHits: { path: string; index: number; rect: { x: number; y: number; w: number; h: number } }[] = [];
  private status = 'connecting';
  private whole: { label: string; detail: string } | null = null;
  private requests = 0;
  private timer = 0;

  constructor(private readonly parts: PageParts, private readonly base: URL, private readonly token: string,
              private readonly applet: string, private readonly period = 1000) {
    new ResizeObserver(() => this.redraw()).observe(parts.area);
    parts.canvas.addEventListener('click', (e) => this.onClick(e));
    parts.canvas.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
  }

  start(): void {
    if (!this.applet) {
      void this.listApplets();
      return;
    }
    const loop = async () => {
      await this.refresh();
      this.timer = window.setTimeout(loop, this.period);
    };
    void loop();
  }

  stop(): void {
    window.clearTimeout(this.timer);
  }

  /** Without an applet in the address, the page lists the workspace's saved spec applets as links. */
  private async listApplets(): Promise<void> {
    this.parts.title.textContent = 'Spec applets';
    this.parts.canvas.hidden = true;
    let env: CommandEnvelope;
    try {
      env = await command(this.base, this.token, 'applet.list', {});
    } catch {
      this.setStatus('offline');
      return;
    }
    this.setStatus(env.ok ? 'connected' : env.error?.code ?? 'error');
    const list = this.parts.list;
    list.hidden = false;
    list.replaceChildren();
    for (const a of env.ok ? (env.result.applets as { id: string; title: string }[]) : []) {
      const li = document.createElement('li');
      const link = document.createElement('a');
      const url = new URL(location.href);
      url.searchParams.set('applet', a.id);
      link.href = url.toString();
      link.textContent = a.title;
      const id = document.createElement('span');
      id.textContent = a.id;
      li.append(link, id);
      list.append(li);
    }
  }

  private setStatus(s: string): void {
    if (s === this.status) return;
    this.status = s;
    this.redraw();
  }

  /** One round: the spec, then the data of every panel shown whose version may have moved. */
  async refresh(): Promise<void> {
    let env: CommandEnvelope;
    try {
      this.requests++;
      env = await command(this.base, this.token, 'spec.get', { applet: this.applet });
    } catch {
      this.setStatus('offline');
      return;
    }
    if (!env.ok) {
      this.whole = { label: env.error?.code === 'applet.not_found' ? 'Missing applet' : 'No spec', detail: env.error?.code ?? '' };
      this.spec = null;
      this.specText = '';
      this.setStatus('connected');
      this.redraw();
      return;
    }
    this.whole = null;
    if (env.result.text !== this.specText) {
      this.specText = env.result.text;
      this.spec = env.result.json as Spec;
      this.key = new ColorKey(this.spec);
      for (const id of [...this.panels.keys()]) if (!findPanel(this.spec, id)) this.panels.delete(id);
      const t = this.spec.applet.title ?? this.spec.applet.id;
      this.parts.title.textContent = t;
      document.title = `${t} · Caliper`;
    }
    this.setStatus('connected');
    this.redraw();
    await Promise.all(this.layout.panels.map((lp) => this.fetchPanel(lp.id)));
  }

  private async fetchPanel(id: string): Promise<void> {
    const spec = this.spec;
    if (!spec) return;
    const panel = findPanel(spec, id);
    if (!panel) return;
    let e = this.panels.get(id);
    if (!e) {
      e = { version: '', prepared: null, busy: false, view: { scroll: 0 } };
      this.panels.set(id, e);
    }
    if (e.busy) return;
    e.busy = true;
    try {
      this.requests++;
      const env = await command(this.base, this.token, 'panel.data', { applet: this.applet, panel: id, known: e.version });
      if (!env.ok) {
        e.prepared = preparePanel(spec, panel, { panel: id, primitive: panel.primitive, version: '', unchanged: false,
          error: env.error!, state: null, bounds: [], data: null });
        e.version = '';
      } else {
        const r = env.result as PanelDataResult;
        if (!r.unchanged && this.spec === spec) {
          e.prepared = preparePanel(spec, panel, r);
          e.version = r.version;
            }
      }
    } catch {
      this.setStatus('offline');
    } finally {
      e.busy = false;
    }
    this.redraw();
  }

  private redrawQueued = false;

  redraw(): void {
    if (this.redrawQueued) return;
    this.redrawQueued = true;
    requestAnimationFrame(() => {
      this.redrawQueued = false;
      this.draw();
    });
  }

  private draw(): void {
    const { canvas, area, status } = this.parts;
    const w = Math.max(1, Math.floor(area.clientWidth)), h = Math.max(1, Math.floor(area.clientHeight));
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
    }
    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // The rules between the panels: the area in the rule colour, which the
    // panels leave showing in the gaps of the layout.
    ctx.fillStyle = COL.rule;
    ctx.fillRect(0, 0, w, h);
    const spec = this.spec;
    const root = spec ? effectiveLayout(spec) : null;
    const before = this.layout.panels.map((p) => p.id).join(' ');
    this.layout = spec && root ? computeLayout(root, spec, { x: 0, y: 0, w, h }, metricsForTextScale(1), this.tabs) : { panels: [], tabs: [] };
    if (!spec || !root) {
      ctx.fillStyle = COL.ground;
      ctx.fillRect(0, 0, w, h);
      const st = this.whole ?? (spec ? { label: 'No panels', detail: spec.applet.id } : { label: this.status === 'offline' ? 'Offline' : 'Connecting', detail: this.applet });
      drawState(ctx, st, { x: 0, y: 0, w, h });
    } else {
      this.tabHits = [];
      for (const t of this.layout.tabs) {
        const shown = t.path ? `${t.path}.${t.active}` : String(t.active);
        const lp = this.layout.panels.find((p) => p.path === shown && p.tab);
        const e = lp ? this.panels.get(lp.id) : undefined;
        const info = e?.prepared && !e.prepared.state ? e.prepared.info : '';
        drawTabs(ctx, t, info).forEach((rect, index) => this.tabHits.push({ path: t.path, index, rect }));
      }
      for (const lp of this.layout.panels) {
        const panel = findPanel(spec, lp.id);
        if (!panel || lp.rect.w < 1 || lp.rect.h < 1) continue;
        let e = this.panels.get(lp.id);
        if (!e) {
          e = { version: '', prepared: null, busy: false, view: { scroll: 0 } };
          this.panels.set(lp.id, e);
        }
        drawPanel(ctx, e.prepared, panelTitle(spec, panel), lp.rect, !lp.tab, this.key!, e.view);
      }
      // A panel that a tab has just shown is fetched at once.
      if (this.layout.panels.map((p) => p.id).join(' ') !== before)
        for (const lp of this.layout.panels) if (!this.panels.get(lp.id)?.prepared) void this.fetchPanel(lp.id);
    }
    const nPanels = spec?.panel?.length ?? 0, nSources = spec?.source?.length ?? 0;
    const items = [this.status, ...(spec ? [`applet ${spec.applet.id}`, `${nPanels} panel${nPanels === 1 ? '' : 's'}`,
      `${nSources} source${nSources === 1 ? '' : 's'}`] : [])];
    const text = items.join('  ·  ');
    if (status.textContent !== text) status.textContent = text;
    status.dataset.state = this.status;
  }

  private onClick(e: MouseEvent): void {
    const r = this.parts.canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    for (const h of this.tabHits) {
      if (x >= h.rect.x && x < h.rect.x + h.rect.w && y >= h.rect.y && y < h.rect.y + h.rect.h) {
        this.tabs[h.path] = h.index;
        this.redraw();
        return;
      }
    }
  }

  private onWheel(e: WheelEvent): void {
    const r = this.parts.canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    for (const lp of this.layout.panels) {
      const q = lp.rect;
      if (x < q.x || x >= q.x + q.w || y < q.y || y >= q.y + q.h) continue;
      const ent = this.panels.get(lp.id);
      if (ent?.prepared?.primitive !== 'table') return;
      e.preventDefault();
      ent.view.scroll = Math.max(0, ent.view.scroll + Math.sign(e.deltaY) * 3);
      this.redraw();
      return;
    }
  }

  debugState(): SpecPageState {
    const spec = this.spec;
    return {
      status: this.status,
      applet: this.applet,
      title: spec ? spec.applet.title ?? spec.applet.id : '',
      panels: this.layout.panels.map((lp) => {
        const e = this.panels.get(lp.id);
        const p = e?.prepared ?? null;
        const panel = spec ? findPanel(spec, lp.id) : undefined;
        return { id: lp.id, title: p?.title ?? (spec && panel ? panelTitle(spec, panel) : lp.id), primitive: panel?.primitive ?? '',
          drawn: !!p && !p.state, state: p?.state?.label ?? null, info: p?.info ?? '', rect: { ...lp.rect } };
      }),
      tabs: this.layout.tabs.map((t) => ({ path: t.path, labels: [...t.labels], active: t.active })),
      requests: this.requests,
    };
  }
}
