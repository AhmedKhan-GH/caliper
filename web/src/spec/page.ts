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
// A round draws once, after every answer, so that the colour key gives the
// lines their colours in the layout's order, as the app's first frame does.
//
// Programs: a scene panel's instances of pieces come with its panel.data and
// draw in the WebAssembly views over the panel's content (programs.ts); a
// pick there stays local to the page.
//
// Controls: a control leaf of the layout takes one control row, and the page
// shows each control with its state from panel.data, disabled, because every
// control changes the run or the app's values and the viewer origin only
// reads (controls.ts).

import { computeLayout, effectiveLayout, findControl, findPanel, metricsForTextScale, panelTitle, type SpecLayout, type TabChoice } from './layout.js';
import { drawControl, formatProgressStatus, progressElapsedSeconds, type ControlData } from './controls.js';
import { ProgramViews, type ProgramPick } from './programs.js';
import { COL, ColorKey } from './look.js';
import { drawPanel, drawProgressOverlay, drawState, drawTabs, type PanelProgress, type PanelView } from './draw.js';
import { preparePanel, type PreparedPanel } from './prepare.js';
import type { PanelDataResult, ProgressPanelState, RunProgress, Spec } from './types.js';

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

export interface PageParts {
  title: HTMLElement; area: HTMLElement; canvas: HTMLCanvasElement; status: HTMLElement; list: HTMLElement;
  /** Optional top canvas over WebGPU mounts for panel progress overlays. */
  overlay?: HTMLCanvasElement;
  /** The line under the bar that says why no live data shows (a dashboard the app does not have open). */
  banner?: HTMLElement;
}

/** What the page says when the app does not have the dashboard open: no live source, value or kept step reaches it. */
export const NOT_OPEN_BANNER = 'The app does not have this dashboard open. It shows data when the dashboard is open in the app.';
export const NOT_OPEN_STATE = { label: 'Dashboard not open in the app', detail: 'shows data when it is open in the app' };

interface PanelEntry { version: string; prepared: PreparedPanel | null; busy: boolean; view: PanelView; control: ControlData | null }

/** What a test reads of the page. */
export interface SpecPageState {
  status: string;
  /** The state the whole page shows in place of the applet, or null. */
  state: string | null;
  /** The app does not have this dashboard open (spec.get's open_in_app), and the banner that says so. */
  notOpen: boolean;
  banner: string;
  applet: string;
  title: string;
  panels: { id: string; title: string; primitive: string; drawn: boolean; state: string | null; info: string;
            progress: PanelProgress | null;
            rect: { x: number; y: number; w: number; h: number } }[];
  /** Hosted run progress, when reported by the app. */
  progress: RunProgress | null;
  /** The control leaves shown, with the state each shows. */
  controls: { id: string; kind: string; title: string; value: unknown; text: string; enabled: boolean; why: string;
              rect: { x: number; y: number; w: number; h: number } }[];
  /** The series lines drawn, by panel: each line's name and colour. */
  lines: Record<string, { name: string; color: string }[]>;
  /** The scene panels' programs (programs.ts, ProgramViews.debug). */
  programs: ReturnType<ProgramViews['debug']>;
  /** The page's own state and the picks it saw, which never reach the app. */
  localState: Record<string, unknown>;
  picks: { panel: string; detail: ProgramPick }[];
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
  private animId = 0;
  private notOpen = false;
  private runProgress: RunProgress | null = null;
  private readonly lastPanelProgress = new Map<string, PanelProgress>();
  private readonly startedAtMs = performance.now();
  readonly programs: ProgramViews;

  constructor(private readonly parts: PageParts, private readonly base: URL, private readonly token: string,
              private readonly applet: string, private readonly period = 1000) {
    this.programs = new ProgramViews(parts.area, base, token, () => this.redraw());
    new ResizeObserver(() => this.redraw()).observe(parts.area);
    parts.canvas.addEventListener('click', (e) => this.onClick(e));
    parts.canvas.addEventListener('mousemove', (e) => this.onMove(e));
    parts.canvas.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
  }

  private isRunProgressActive(): boolean {
    const st = this.runProgress?.state ?? '';
    return !!st && st !== 'ready' && st !== 'idle' && st !== 'error';
  }

  start(): void {
    if (!this.applet) {
      void this.listApplets();
      return;
    }
    this.draw();
    const loop = async () => {
      await this.refresh();
      const delay = this.isRunProgressActive() ? Math.min(this.period, 250) : this.period;
      this.timer = window.setTimeout(loop, delay);
    };
    void loop();
  }

  stop(): void {
    window.clearTimeout(this.timer);
    if (this.animId) {
      cancelAnimationFrame(this.animId);
      this.animId = 0;
    }
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
      const code = env.error?.code ?? '';
      const label = code === 'applet.not_found' ? 'Missing applet'
        : code === 'http.401' || code === 'http.403' ? 'Access refused' : code === 'http.404' ? 'No command layer' : 'No spec';
      this.whole = { label, detail: code === 'applet.not_found' ? this.applet : code };
      this.spec = null;
      this.specText = '';
      this.setStatus('connected');
      this.redraw();
      return;
    }
    this.whole = null;
    if (env.result.progress && typeof env.result.progress === 'object') {
      this.runProgress = env.result.progress as RunProgress;
    } else if (env.result.open_in_app === false) {
      this.runProgress = null;
    }
    // A saved dashboard that the app does not have open gets no live data: the page says so, and asks
    // for every panel again when that changes.
    const notOpen = env.result.open_in_app === false;
    if (notOpen !== this.notOpen) {
      this.notOpen = notOpen;
      for (const e of this.panels.values()) e.version = '';
      const b = this.parts.banner;
      if (b) {
        b.textContent = notOpen ? NOT_OPEN_BANNER : '';
        b.hidden = !notOpen;
      }
    }
    if (env.result.text !== this.specText) {
      this.specText = env.result.text;
      this.spec = env.result.json as Spec;
      this.key = new ColorKey(this.spec);
      this.programs.setSpec(this.spec);
      for (const id of [...this.panels.keys()]) if (!findPanel(this.spec, id) && !findControl(this.spec, id)) this.panels.delete(id);
      const t = this.spec.applet.title ?? this.spec.applet.id;
      this.parts.title.textContent = t;
      document.title = `${t} · Caliper`;
    }
    this.setStatus('connected');
    this.draw();   // the layout of this spec, whose leaves the round fetches
    await Promise.all(this.layout.panels.map((lp) => this.fetchPanel(lp.id, false)));
    this.redraw();
  }

  private entry(id: string): PanelEntry {
    let e = this.panels.get(id);
    if (!e) {
      e = { version: '', prepared: null, busy: false, view: { scroll: 0 }, control: null };
      this.panels.set(id, e);
    }
    return e;
  }

  /** The data of one leaf, a panel or a control; `redraw` false when the caller draws after a round. */
  private async fetchPanel(id: string, redraw = true): Promise<void> {
    const spec = this.spec;
    if (!spec) return;
    const panel = findPanel(spec, id);
    const control = panel ? undefined : findControl(spec, id);
    if (!panel && !control) return;
    const e = this.entry(id);
    if (e.busy) return;
    e.busy = true;
    try {
      this.requests++;
      const env = await command(this.base, this.token, 'panel.data', { applet: this.applet, panel: id, known: e.version });
      if (this.spec !== spec) return;
      if (control) {
        if (env.ok && !env.result.unchanged && env.result.data) {
          e.control = env.result.data as ControlData;
          e.version = env.result.version;
          if (e.control.progress && typeof e.control.progress === 'object') {
            this.runProgress = e.control.progress;
          } else if ((e.control.bind === 'param:output.status' || e.control.id === 'trace_status') && e.control.progress === null) {
            this.runProgress = null;
          }
        } else if (!env.ok) {
          e.control = null;
          e.version = '';
        }
      } else if (!env.ok) {
        e.prepared = preparePanel(spec, panel!, { panel: id, primitive: panel!.primitive, version: '', unchanged: false,
          error: env.error!, state: null, bounds: [], data: null });
        e.version = '';
      } else {
        const r = env.result as PanelDataResult;
        if (!r.unchanged) {
          e.prepared = preparePanel(spec, panel!, r);
          // An offline source of a dashboard that the app does not have open: say that, not "Source offline".
          if (this.notOpen && e.prepared.state?.label === 'Source offline') e.prepared.state = { ...NOT_OPEN_STATE };
          e.version = r.version;
          if (e.prepared.programs) this.programs.update(panel!, e.prepared.programs);
        }
      }
    } catch {
      this.setStatus('offline');
    } finally {
      e.busy = false;
    }
    if (redraw) this.redraw();
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

  private isPlaceholderPanel(p: PreparedPanel | null): boolean {
    if (!p || p.state) return true;
    if (p.primitive === 'text' && p.text && !p.text.hasValue) {
      const t = p.text.text;
      return !t || t === 'The trace is not available.' || t === 'No feature is selected.';
    }
    return false;
  }

  private panelProgressFromRun(obj: ProgressPanelState | RunProgress, elapsedS: number | null, translucent: boolean): PanelProgress | null {
    const state = obj.state || '';
    if (!state || state === 'ready' || state === 'idle') return null;
    const isErr = state === 'error';
    return {
      active: true,
      error: isErr,
      stage: obj.stage || state,
      label: obj.label || (isErr ? 'Error' : state),
      detail: isErr ? (obj.error ?? undefined) : undefined,
      done: typeof obj.done === 'number' ? obj.done : null,
      total: typeof obj.total === 'number' ? obj.total : null,
      elapsedS,
      translucent,
    };
  }

  private resolvePanelProgress(panelId: string, primitive: string, e: PanelEntry, nowMs: number): PanelProgress | null {
    if (this.notOpen) return null;
    const rp = this.runProgress;
    const runState = rp?.state ?? '';
    const runActive = !!runState && runState !== 'ready' && runState !== 'idle';
    const elapsedS = progressElapsedSeconds(rp);
    const hasScenePrograms = primitive === 'scene' && this.programs.has(panelId);
    const sceneWaiting = hasScenePrograms && !this.programs.hasDrawnFrame(panelId);
    const emptyOrPlaceholder = this.isPlaceholderPanel(e.prepared) || sceneWaiting;

    if (rp?.panels && rp.panels[panelId]) {
      const p = this.panelProgressFromRun(rp.panels[panelId], elapsedS, !emptyOrPlaceholder);
      if (p) return p;
    } else if (rp && runActive && (emptyOrPlaceholder || runState === 'starting' || runState === 'loading')) {
      const p = this.panelProgressFromRun(rp, elapsedS, !emptyOrPlaceholder);
      if (p) return p;
    }
    if (hasScenePrograms) {
      const bp = this.programs.progressOf(panelId, nowMs);
      if (bp) return bp;
    }
    if (!e.prepared && e.busy) {
      return {
        active: true,
        error: false,
        stage: 'loading',
        label: 'Loading panel data',
        elapsedS: Math.max(0, (nowMs - this.startedAtMs) / 1000),
      };
    }
    return null;
  }

  private draw(): void {
    const { canvas, overlay, area, status } = this.parts;
    const nowMs = performance.now();
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
    let octx: CanvasRenderingContext2D | null = null;
    if (overlay) {
      if (overlay.width !== Math.round(w * dpr) || overlay.height !== Math.round(h * dpr)) {
        overlay.width = Math.round(w * dpr);
        overlay.height = Math.round(h * dpr);
        overlay.style.width = `${w}px`;
        overlay.style.height = `${h}px`;
      }
      octx = overlay.getContext('2d');
      if (octx) {
        octx.setTransform(dpr, 0, 0, dpr, 0, 0);
        octx.clearRect(0, 0, w, h);
      }
    }
    // The rules between the panels: the area in the rule colour, which the
    // panels leave showing in the gaps of the layout.
    ctx.fillStyle = COL.rule;
    ctx.fillRect(0, 0, w, h);
    const spec = this.spec;
    const root = spec ? effectiveLayout(spec) : null;
    const before = this.layout.panels.map((p) => p.id).join(' ');
    this.layout = spec && root ? computeLayout(root, spec, { x: 0, y: 0, w, h }, metricsForTextScale(1), this.tabs) : { panels: [], tabs: [] };
    this.lastPanelProgress.clear();
    let needsAnim = this.isRunProgressActive();
    if (!spec || !root) {
      this.programs.hideAll(new Set());
      ctx.fillStyle = COL.ground;
      ctx.fillRect(0, 0, w, h);
      if (!this.whole && !spec && this.status === 'connecting') {
        drawProgressOverlay(ctx, {
          active: true,
          error: false,
          stage: 'connecting',
          label: 'Connecting to Caliper',
          elapsedS: Math.max(0, (nowMs - this.startedAtMs) / 1000),
        }, { x: 0, y: 0, w, h }, nowMs);
        needsAnim = true;
      } else {
        const st = this.whole ?? (spec ? { label: 'No panels', detail: spec.applet.id } : { label: this.status === 'offline' ? 'Offline' : 'Connecting', detail: this.applet });
        drawState(ctx, st, { x: 0, y: 0, w, h });
      }
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
        if (lp.rect.w < 1 || lp.rect.h < 1) continue;
        const e = this.entry(lp.id);
        if (lp.control) {
          const c = findControl(spec, lp.id)!;
          drawControl(ctx, e.control, c.title ?? c.id, c.kind, lp.rect, this.runProgress);
          continue;
        }
        const panel = findPanel(spec, lp.id);
        if (!panel) continue;
        const prog = this.resolvePanelProgress(panel.id, panel.primitive, e, nowMs);
        if (prog) {
          this.lastPanelProgress.set(panel.id, prog);
          if (!prog.error) needsAnim = true;
        }
        const frame = drawPanel(ctx, e.prepared, panelTitle(spec, panel), lp.rect, !lp.tab, this.key!, e.view, prog, nowMs);
        if (panel.primitive === 'scene' && this.programs.has(panel.id)) {
          const st = this.notOpen ? NOT_OPEN_STATE : this.programs.stateOf(panel.id);
          if (st) {
            drawState(ctx, st, frame.content);
            this.programs.place(panel.id, null);
          } else {
            this.programs.place(panel.id, frame.content);
            if (prog) {
              drawProgressOverlay(octx ?? ctx, prog, frame.content, nowMs);
            }
          }
        } else if (panel.primitive === 'scene' && prog) {
          drawProgressOverlay(octx ?? ctx, prog, frame.content, nowMs);
        }
      }
      this.programs.hideAll(new Set(this.layout.panels.map((p) => p.id)));
      // A panel that a tab has just shown is fetched at once.
      if (this.layout.panels.map((p) => p.id).join(' ') !== before)
        for (const lp of this.layout.panels) {
          const e = this.panels.get(lp.id);
          if (!e?.prepared && !e?.control) void this.fetchPanel(lp.id);
        }
    }
    const nPanels = spec?.panel?.length ?? 0, nSources = spec?.source?.length ?? 0;
    const liveStatus = this.notOpen ? '' : formatProgressStatus(this.runProgress);
    const items = [this.status, ...(spec ? [`applet ${spec.applet.id}`, `${nPanels} panel${nPanels === 1 ? '' : 's'}`,
      `${nSources} source${nSources === 1 ? '' : 's'}`] : []), ...(liveStatus ? [liveStatus] : [])];
    const text = items.join('  ·  ');
    if (status.textContent !== text) status.textContent = text;
    status.dataset.state = this.status;
    if (needsAnim && !this.animId) {
      this.animId = requestAnimationFrame(() => {
        this.animId = 0;
        this.draw();
      });
    }
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

  /** A hover over a control that the viewer cannot change says why, as the app's tooltip does. */
  private onMove(e: MouseEvent): void {
    const r = this.parts.canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    let tip = '';
    for (const lp of this.layout.panels) {
      const q = lp.rect;
      if (!lp.control || x < q.x || x >= q.x + q.w || y < q.y || y >= q.y + q.h) continue;
      const c = this.panels.get(lp.id)?.control;
      if (c && !c.enabled) tip = c.why;
    }
    if (this.parts.canvas.title !== tip) this.parts.canvas.title = tip;
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
      state: this.whole?.label ?? null,
      notOpen: this.notOpen,
      banner: this.parts.banner && !this.parts.banner.hidden ? this.parts.banner.textContent ?? '' : '',
      applet: this.applet,
      title: spec ? spec.applet.title ?? spec.applet.id : '',
      controls: this.layout.panels.filter((lp) => lp.control).map((lp) => {
        const c = this.panels.get(lp.id)?.control ?? null;
        const sc = spec ? findControl(spec, lp.id) : undefined;
        return { id: lp.id, kind: sc?.kind ?? '', title: sc?.title ?? lp.id, value: c?.value ?? null, text: c?.text ?? '',
          enabled: c?.enabled ?? false, why: c?.why ?? '', rect: { ...lp.rect } };
      }),
      lines: Object.fromEntries(this.layout.panels.filter((lp) => this.panels.get(lp.id)?.prepared?.series).map((lp) => [lp.id,
        this.panels.get(lp.id)!.prepared!.series!.lines.map((l) => ({ name: l.name, color: this.key!.find(l.name) ?? '' }))])),
      panels: this.layout.panels.filter((lp) => !lp.control).map((lp) => {
        const e = this.panels.get(lp.id);
        const p = e?.prepared ?? null;
        const panel = spec ? findPanel(spec, lp.id) : undefined;
        // A scene panel's programs may show a state over it: the dashboard not open, or no program views.
        const over = panel?.primitive === 'scene' && this.programs.has(lp.id)
          ? (this.notOpen ? NOT_OPEN_STATE : this.programs.stateOf(lp.id)) : null;
        return { id: lp.id, title: p?.title ?? (spec && panel ? panelTitle(spec, panel) : lp.id), primitive: panel?.primitive ?? '',
          drawn: !!p && !p.state && !over, state: over?.label ?? p?.state?.label ?? null, info: p?.info ?? '',
          progress: this.lastPanelProgress.get(lp.id) ?? null, rect: { ...lp.rect } };
      }),
      progress: this.runProgress,
      tabs: this.layout.tabs.map((t) => ({ path: t.path, labels: [...t.labels], active: t.active })),
      programs: this.programs.debug(),
      localState: { ...this.programs.localState },
      picks: [...this.programs.picks],
      requests: this.requests,
    };
  }
}
