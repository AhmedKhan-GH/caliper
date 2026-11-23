// The programs of the scene panels in the browser's spec page: the instances
// of pieces (library, workspace and inline alike) that the app gives a viewer
// through panel.data (panel_data.h, viewer_programs), drawn by the WebAssembly
// views (web/wasm) on a canvas over each panel's content.
//
// The viewer token opens no stream, so each panel's module instance runs with
// feed "page": the page hands it the instance form of PROGRAM set and each
// new resource version, and the module checks them as it checks a relay of
// the stream: the pin's hash over the files, the manifest, imports and
// WebGPU's compile. A program of a foreign spec that nobody reviewed arrives
// without its files and stays in review here too.
//
// The module's files come from the app's own origin (GET /v0/spec/<name>,
// with the page's token in the query), as the page's CSP allows.
//
// Picks: a click on a program's item gives the item's fields (pick events of
// the module). The app writes a panel's `select` into a state and a pick into
// the panel's selection, but state.set and selection.set change the app's
// session, which the viewer origin may not do (design/command-layer.md). So a
// pick here stays local to the page: it sets the page's own copy of that
// state and selection, and the page sends the programs whose params read them
// their new values (PROGRAM op params). The app does not see it.

import type { Rect } from './layout.js';
import type { Spec, SpecPanel } from './types.js';

export interface ViewerResource {
  resource: string; step: number; count: number; stride: number; width: number; height: number; base64: string;
}
export interface ViewerProgram {
  id: string;
  title: string | null;
  piece: { name: string; version: number; from: string };
  state: 'ready' | 'review' | 'refused';
  error: { code: string; message: string } | null;
  instance: Record<string, unknown> | null;
  params?: Record<string, number>;
  resources: ViewerResource[];
}

/** What the module reports of a program (web_programs.cpp, status_json). */
export interface ModuleProgram { id: string; panel: string; state: string; code: string; message: string; step?: number; frames?: number }

interface Handle {
  canvas: HTMLCanvasElement;
  ready: Promise<void>;
  readonly stats: { programs?: string | ModuleProgram[]; [k: string]: unknown };
  feedProgram(p: Record<string, unknown>): boolean;
  feedResource(r: Omit<ViewerResource, 'base64'> & { program: string }, bytes: Uint8Array): boolean;
  select(picks: { program: string; item: number; index: number[]; newest: boolean }[]): boolean;
  destroy(): void;
}
interface Loader { mountCaliperViews(container: HTMLElement, options: Record<string, unknown>): Promise<Handle> }

/** A pick of a program's item (live_views_scene.cpp, program_pick_detail). */
export interface ProgramPick { program: string; item: number; step: number; fields: Record<string, number>; index: number[] }

/** A reference in a program's params (spec version 1): state, selection or control. */
type Ref = { state?: string; selection?: string; control?: string; field?: string; entity?: string; default?: unknown };

const MAX_RESOURCE_B64 = 48 << 20;

function bytesOf(b64: string): Uint8Array | null {
  if (b64.length > MAX_RESOURCE_B64) return null;
  try {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  } catch {
    return null;
  }
}

class PanelMount {
  readonly box: HTMLDivElement;
  handle: Handle | null = null;
  failed = '';
  starting: Promise<void> | null = null;
  readonly fed = new Map<string, string>();       // program id -> pin hash it was set with
  readonly params = new Map<string, string>();    // program id -> params JSON last sent
  readonly versions = new Map<string, string>();  // program/resource -> step and length last sent
  programs: ViewerProgram[] = [];
  pick: ProgramPick | null = null;

  constructor(readonly panel: string, area: HTMLElement) {
    this.box = document.createElement('div');
    this.box.dataset.panel = panel;
    this.box.style.position = 'absolute';
    this.box.style.display = 'none';
    area.append(this.box);
  }
}

export class ProgramViews {
  private readonly mounts = new Map<string, PanelMount>();
  private loader: Promise<Loader> | null = null;
  /** The page's own state and selections, which its picks write (never the app's). */
  readonly localState: Record<string, unknown> = {};
  readonly localPicks: Record<string, ProgramPick> = {};
  /** Every pick the page saw, newest last (at most 32), for a test or a reader. */
  readonly picks: { panel: string; detail: ProgramPick }[] = [];
  private spec: Spec | null = null;

  constructor(private readonly area: HTMLElement, private readonly base: URL, private readonly token: string,
              private readonly onChange: () => void) {}

  setSpec(spec: Spec | null): void {
    this.spec = spec;
    for (const [id, m] of this.mounts)
      if (!spec?.panel?.some((p) => p.id === id && p.primitive === 'scene')) {
        m.handle?.destroy();
        m.box.remove();
        this.mounts.delete(id);
      }
  }

  private load(): Promise<Loader> {
    if (!this.loader) {
      const url = new URL(`/v0/spec/caliper-views-loader.js?token=${encodeURIComponent(this.token)}`, this.base).href;
      this.loader = import(/* @vite-ignore */ url) as Promise<Loader>;
      this.loader.catch(() => { this.loader = null; });
    }
    return this.loader;
  }

  /** The programs panel.data gave for a scene panel; mounts its module on first use. */
  update(panel: SpecPanel, programs: ViewerProgram[]): void {
    let m = this.mounts.get(panel.id);
    if (!programs.length) {
      if (m) m.programs = [];
      return;
    }
    if (!m) {
      m = new PanelMount(panel.id, this.area);
      this.mounts.set(panel.id, m);
    }
    m.programs = programs;
    if (!m.handle && !m.starting && !m.failed) m.starting = this.start(m, panel);
    if (m.handle) this.feed(m);
  }

  private async start(m: PanelMount, panel: SpecPanel): Promise<void> {
    try {
      if (!('gpu' in navigator)) throw new Error('This browser has no WebGPU.');
      const loader = await this.load();
      const o = (panel.options ?? {}) as Record<string, unknown>;
      const scene: Record<string, unknown> = { space: '3d', units: 'm', up: typeof o.up === 'string' ? o.up : '+y' };
      if (o.view_camera && typeof o.view_camera === 'object') scene.view_camera = o.view_camera;
      m.handle = await loader.mountCaliperViews(m.box, {
        view: panel.id, kind: 'scene', header: false, feed: 'page', scene,
        assetBase: new URL('/v0/spec/', this.base).href,
        assetQuery: `?token=${encodeURIComponent(this.token)}`,
        onEvent: (e: { view: string; type: string; detail: unknown }) => this.onEvent(panel.id, e.type, e.detail),
      });
      // The module starts its WebGPU device after the mount returns: it takes programs once ready.
      void m.handle.ready.then(() => {
        this.feed(m);
        this.onChange();
      });
    } catch (e) {
      m.failed = e instanceof Error ? e.message : String(e);
    } finally {
      m.starting = null;
      this.onChange();
    }
  }

  /** Hands the module what changed: a new pin, new params, new resource versions. */
  private feed(m: PanelMount): void {
    const h = m.handle;
    if (!h) return;
    const wanted = new Set(m.programs.filter((p) => p.state === 'ready').map((p) => p.id));
    for (const id of [...m.fed.keys()])
      if (!wanted.has(id) && h.feedProgram({ op: 'remove', id })) {
        m.fed.delete(id);
        m.params.delete(id);
      }
    for (const p of m.programs) {
      if (p.state !== 'ready' || !p.instance) continue;
      const pin = JSON.stringify((p.instance as { piece?: unknown }).piece ?? null);
      const params = { ...(p.params ?? {}), ...this.localParams(p.id) };
      // Each is marked as sent only when the module took it, so a later round sends it again.
      if (m.fed.get(p.id) !== pin) {
        if (!h.feedProgram({ ...p.instance, params })) continue;
        m.fed.set(p.id, pin);
        m.params.set(p.id, JSON.stringify(params));
        for (const k of [...m.versions.keys()]) if (k.startsWith(`${p.id}/`)) m.versions.delete(k);
      } else if (m.params.get(p.id) !== JSON.stringify(params)) {
        if (h.feedProgram({ op: 'params', id: p.id, params })) m.params.set(p.id, JSON.stringify(params));
      }
      for (const r of p.resources) {
        const key = `${p.id}/${r.resource}`, v = `${r.step} ${r.base64.length} ${r.base64.slice(-16)}`;
        if (m.versions.get(key) === v) continue;
        const bytes = bytesOf(r.base64);
        if (!bytes || bytes.length !== r.count * r.stride) continue;
        const { base64: _, ...meta } = r;
        if (h.feedResource({ ...meta, program: p.id }, bytes)) m.versions.set(key, v);
      }
    }
  }

  /** The values of a program's params that read the page's own picks and state. */
  private localParams(programId: string): Record<string, number> {
    const out: Record<string, number> = {};
    const prog = (this.spec as { program?: { id: string; params?: Record<string, unknown> }[] } | null)?.program
      ?.find((p) => p.id === programId);
    for (const [name, place] of Object.entries(prog?.params ?? {})) {
      if (!place || typeof place !== 'object') continue;
      const r = place as Ref;
      let v: unknown;
      if (r.state !== undefined) v = this.localState[r.state];
      else if (r.selection !== undefined) {
        const pick = this.localPicks[r.selection];
        v = pick ? (r.field ? pick.fields?.[r.field] : pick.item) : undefined;
      }
      if (typeof v === 'number' && Number.isFinite(v)) out[name] = v;
    }
    return out;
  }

  private onEvent(panelId: string, type: string, detail: unknown): void {
    if (type !== 'pick' || !detail || typeof detail !== 'object') return;
    const d = detail as ProgramPick;
    if (typeof d.program !== 'string') return;
    this.picks.push({ panel: panelId, detail: d });
    if (this.picks.length > 32) this.picks.shift();
    this.localPicks[panelId] = d;
    // The panel's `select` writes a field of the pick into a state, here in the page's own copy.
    const write = (this.spec?.panel?.find((p) => p.id === panelId)?.options as { select?: { state?: string; field?: string } } | undefined)?.select;
    if (write?.state && write.field && typeof d.fields?.[write.field] === 'number') this.localState[write.state] = d.fields[write.field];
    // The linked selection, as the app's draw host gives it: each panel's pick, the newest marked.
    const sel = Object.entries(this.localPicks).map(([panel, p]) => ({ program: p.program, item: p.item,
      index: Array.isArray(p.index) ? p.index.slice(0, 4) : [-1, -1, -1, -1], newest: panel === panelId }));
    for (const m of this.mounts.values()) {
      this.feed(m);
      m.handle?.select(sel);
    }
    this.onChange();
  }

  /** Places a panel's module canvas over its content, or hides it when the panel is not shown. */
  place(panelId: string, content: Rect | null): void {
    const m = this.mounts.get(panelId);
    if (!m) return;
    const s = m.box.style;
    if (!content || content.w < 1 || content.h < 1) {
      s.display = 'none';
      return;
    }
    s.display = 'block';
    s.left = `${content.x}px`;
    s.top = `${content.y}px`;
    s.width = `${content.w}px`;
    s.height = `${content.h}px`;
  }

  hideAll(except: Set<string>): void {
    for (const [id, m] of this.mounts) if (!except.has(id)) m.box.style.display = 'none';
  }

  has(panelId: string): boolean {
    return (this.mounts.get(panelId)?.programs.length ?? 0) > 0;
  }

  /** A panel's state for the page to show over its content, or null while it draws. */
  stateOf(panelId: string): { label: string; detail: string } | null {
    const m = this.mounts.get(panelId);
    if (!m) return null;
    if (m.failed) return { label: 'No program views', detail: m.failed };
    const review = m.programs.find((p) => p.state === 'review');
    if (review && m.programs.every((p) => p.state !== 'ready')) return { label: 'Program in review', detail: review.id };
    const refused = m.programs.find((p) => p.state === 'refused');
    if (refused && m.programs.every((p) => p.state !== 'ready')) return { label: 'Program refused', detail: refused.error?.code ?? refused.id };
    return null;
  }

  /** What a test reads: each panel's programs as the app gave them and as the module reports them. */
  debug(): Record<string, { mounted: boolean; failed: string;
    programs: { id: string; state: string; code: string; module: string; step: number; frames: number }[];
    pick: ProgramPick | null }> {
    const out: ReturnType<ProgramViews['debug']> = {};
    for (const [id, m] of this.mounts) {
      // The module publishes its programs as JSON text (web_programs.cpp, status_json).
      const raw = m.handle?.stats.programs as unknown;
      let mod: ModuleProgram[] = [];
      try {
        mod = typeof raw === 'string' ? JSON.parse(raw) as ModuleProgram[] : Array.isArray(raw) ? raw as ModuleProgram[] : [];
      } catch { /* not yet published */ }
      out[id] = {
        mounted: !!m.handle, failed: m.failed, pick: this.localPicks[id] ?? null,
        programs: m.programs.map((p) => {
          const s = mod.find((q) => q.id === p.id);
          return { id: p.id, state: p.state, code: p.error?.code ?? s?.code ?? '', module: s?.state ?? '', step: s?.step ?? -1,
            frames: s?.frames ?? 0 };
        }),
      };
    }
    return out;
  }

  /** The module canvas of a panel, for a test that reads its pixels. */
  canvasOf(panelId: string): HTMLCanvasElement | null {
    return this.mounts.get(panelId)?.handle?.canvas ?? null;
  }
}
