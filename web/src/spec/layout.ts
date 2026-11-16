// The layout of a spec's panels as plain rectangles, a port of the native
// spec runtime's src/host/spec_runtime/spec_layout.cpp, so that the browser
// and the app place every panel on the same pixels. web/test/spec-layout.spec.ts
// runs the cases of the native tests (tests/test_spec_layout.cpp) against it.
//
// A split gives its first child `ratio` of its length (0.5 by default), or
// equal shares to more children; tabs show one child under a strip of
// labels; a 1 px rule separates the children of a split. Every panel has a
// minimum of 12 by 8 lines of the 15 px UI face, which grows with the text
// scale; a split keeps each child at its minimum while the space allows and
// otherwise shares the space in proportion to the minimums. Every edge is a
// whole pixel. Rounding is C++'s std::round, half away from zero.

import type { LayoutNode, SplitNode, Spec, SpecPanel, TabsNode } from './types.js';

export interface Rect { x: number; y: number; w: number; h: number }

export interface LayoutMetrics { minPanelW: number; minPanelH: number; tabBarH: number; gap: number }

export interface LayoutPanel { id: string; path: string; rect: Rect; tab: boolean }
export interface LayoutTabs { path: string; bar: Rect; labels: string[]; active: number }
export interface SpecLayout { panels: LayoutPanel[]; tabs: LayoutTabs[] }

/** The tab shown for each tabs node, by its path; a missing or out-of-range choice shows the first. */
export type TabChoice = Record<string, number>;

/** std::round: half away from zero. */
export const cround = (v: number): number => (v < 0 ? -Math.round(-v) : Math.round(v));

/** Float32 arithmetic where the native code keeps a float. */
const f32 = Math.fround;

export function metricsForTextScale(textScale: number): LayoutMetrics {
  const s = textScale > 0 ? textScale : 1;
  const line = f32(15 * s);
  return { minPanelW: cround(f32(12 * line)), minPanelH: cround(f32(8 * line)), tabBarH: cround(f32(line + f32(9 * s))), gap: 1 };
}

const isSplit = (n: LayoutNode): n is SplitNode => typeof n === 'object' && 'split' in n;
const isTabs = (n: LayoutNode): n is TabsNode => typeof n === 'object' && 'tabs' in n;
const childrenOf = (n: LayoutNode): LayoutNode[] => (isSplit(n) ? n.children : isTabs(n) ? n.tabs : []);

/** The spec's layout, or its default: one panel fills the applet, several stack vertically. */
export function effectiveLayout(spec: Spec): LayoutNode | null {
  if (spec.layout !== undefined) return spec.layout;
  const panels = spec.panel ?? [];
  if (!panels.length) return null;
  if (panels.length === 1) return panels[0].id;
  return { split: 'vertical', children: panels.map((p) => p.id) };
}

export function findPanel(spec: Spec, id: string): SpecPanel | undefined {
  return (spec.panel ?? []).find((p) => p.id === id);
}

/** The name a binding draws under: its column, or its source's name (a catalog name, a query's first table, or the source's id). */
export function bindingName(spec: Spec, binding: string): string {
  const dot = binding.indexOf('.');
  if (dot >= 0) return binding.slice(dot + 1);
  const src = (spec.source ?? []).find((s) => s.id === binding);
  if (!src) return binding;
  if (src.catalog !== undefined) {
    const colon = src.catalog.indexOf(':');
    return colon < 0 ? src.catalog : src.catalog.slice(colon + 1);
  }
  if (src.sql !== undefined) {
    // A query goes by the first table it reads, its FROM's name.
    const m = /(?:^|[^A-Za-z0-9])FROM\s/i.exec(src.sql);
    if (m) {
      const t = /^\s*([A-Za-z0-9_]+)/.exec(src.sql.slice(m.index + m[0].length - 1));
      if (t) return t[1];
    }
  }
  return src.id;
}

/** The title a panel is shown under: its title, else what it draws, never its bare id. */
export function panelTitle(spec: Spec, panel: SpecPanel): string {
  if (panel.title) return panel.title;
  const bound = (slot: string): string[] | null => {
    const b = panel.bind?.[slot];
    if (b === undefined) return null;
    const list = Array.isArray(b) ? b : [b];
    return list.length ? list : null;
  };
  const name = (slot: string) => bindingName(spec, bound(slot)![0]);
  switch (panel.primitive) {
    case 'series': {
      const y = bound('y');
      if (y) return y.map((b) => bindingName(spec, b)).join(', ');
      break;
    }
    case 'points':
      if (bound('positions')) return name('positions');
      if (bound('x') && bound('y')) return `${name('y')} by ${name('x')}`;
      break;
    case 'heatmap':
    case 'histogram':
    case 'image':
      if (bound('values')) return name('values');
      break;
    case 'table':
      if (bound('rows')) return name('rows');
      break;
    case 'text':
      return bound('value') ? name('value') : 'Text';
  }
  return panel.primitive.charAt(0).toUpperCase() + panel.primitive.slice(1);
}

function firstPanel(n: LayoutNode): string {
  if (typeof n === 'string') return n;
  for (const c of childrenOf(n)) {
    const id = firstPanel(c);
    if (id) return id;
  }
  return '';
}

function tabLabel(n: LayoutNode, spec: Spec): string {
  if (typeof n !== 'string' && n.title) return n.title;
  const id = firstPanel(n);
  const p = findPanel(spec, id);
  return p ? panelTitle(spec, p) : id;
}

const childPath = (path: string, i: number) => (path ? `${path}.${i}` : String(i));

/** The minimum size of a node. */
export function layoutMinSize(n: LayoutNode, m: LayoutMetrics): { w: number; h: number } {
  if (typeof n === 'string') return { w: m.minPanelW, h: m.minPanelH };
  if (isTabs(n)) {
    let w = 0, h = 0;
    for (const c of n.tabs) {
      const r = layoutMinSize(c, m);
      w = Math.max(w, r.w);
      h = Math.max(h, r.h);
    }
    return { w, h: f32(h + m.tabBarH) };
  }
  const across = n.split === 'horizontal';
  let along = 0, cross = 0;
  for (const c of n.children) {
    const r = layoutMinSize(c, m);
    along = f32(along + (across ? r.w : r.h));
    cross = Math.max(cross, across ? r.h : r.w);
  }
  if (n.children.length) along = f32(along + f32(m.gap * (n.children.length - 1)));
  return across ? { w: along, h: cross } : { w: cross, h: along };
}

/**
 * The shares of `length` for children with minimums `mins` along one axis:
 * two children with `ratio`, more equally, each at least its minimum while the
 * length allows, else in proportion to the minimums. Whole pixels that sum to the length.
 */
export function layoutShares(length: number, mins: number[], ratio: number): number[] {
  const k = mins.length;
  if (!k) return [];
  const share = new Array<number>(k).fill(0);
  const L = Math.max(0, cround(f32(length)));
  const need = mins.reduce((a, b) => a + b, 0);
  if (need >= L) {
    for (let i = 0; i < k; i++) share[i] = need > 0 ? (L * mins[i]) / need : L / k;
  } else if (k === 2) {
    const r = Math.min(1, Math.max(0, ratio));
    share[0] = Math.min(Math.max(r * L, mins[0]), L - mins[1]);
    share[1] = L - share[0];
  } else {
    // Equal shares, raising each child below its minimum to it and sharing
    // what is left among the others.
    const fixed = new Array<boolean>(k).fill(false);
    for (let changed = true; changed;) {
      changed = false;
      let left = L, free = 0;
      for (let i = 0; i < k; i++) {
        if (fixed[i]) left -= mins[i];
        else free++;
      }
      const each = free ? left / free : 0;
      for (let i = 0; i < k; i++) {
        if (fixed[i]) share[i] = mins[i];
        else if (mins[i] > each) {
          fixed[i] = true;
          changed = true;
        } else share[i] = each;
      }
    }
  }
  // Whole pixels: round the running edges, so the shares sum to the length.
  const out = new Array<number>(k);
  let edge = 0, prev = 0;
  for (let i = 0; i < k; i++) {
    edge += share[i];
    const e = i + 1 === k ? L : f32(cround(edge));
    out[i] = Math.max(0, f32(e - prev));
    prev = e;
  }
  return out;
}

/** The rectangles of the layout `root` in `area`. */
export function computeLayout(root: LayoutNode, spec: Spec, area: Rect, m: LayoutMetrics, choice: TabChoice = {}): SpecLayout {
  const out: SpecLayout = { panels: [], tabs: [] };
  const place = (n: LayoutNode, r: Rect, path: string, depth: number, tabChild = false): void => {
    if (depth > 64) return;
    if (typeof n === 'string') {
      out.panels.push({ id: n, path, rect: r, tab: tabChild });
      return;
    }
    if (isTabs(n)) {
      if (!n.tabs.length) return;
      const barH = Math.min(r.h, cround(m.tabBarH));
      const c = choice[path];
      const active = c !== undefined && c >= 0 && c < n.tabs.length ? c : 0;
      out.tabs.push({ path, bar: { x: r.x, y: r.y, w: r.w, h: barH }, labels: n.tabs.map((t) => tabLabel(t, spec)), active });
      place(n.tabs[active], { x: r.x, y: f32(r.y + barH), w: r.w, h: Math.max(0, f32(r.h - barH)) }, childPath(path, active), depth + 1, true);
      return;
    }
    const k = n.children.length;
    if (!k) return;
    const across = n.split === 'horizontal';
    const total = across ? r.w : r.h;
    const gaps = f32(m.gap * (k - 1));
    const length = Math.max(0, f32(total - gaps));
    const mins = n.children.map((c) => {
      const ms = layoutMinSize(c, m);
      return across ? ms.w : ms.h;
    });
    const shares = layoutShares(length, mins, n.ratio ?? 0.5);
    // With no room for the gaps, the children overlap rather than leave the area.
    const gap = total >= gaps ? m.gap : 0;
    let at = across ? r.x : r.y;
    for (let i = 0; i < k; i++) {
      const cr = across ? { x: at, y: r.y, w: shares[i], h: r.h } : { x: r.x, y: at, w: r.w, h: shares[i] };
      place(n.children[i], cr, childPath(path, i), depth + 1);
      at = f32(f32(at + shares[i]) + gap);
    }
  };
  // Whole-pixel edges of the area.
  const x0 = cround(area.x), y0 = cround(area.y);
  const x1 = cround(area.x + area.w), y1 = cround(area.y + area.h);
  place(root, { x: x0, y: y0, w: Math.max(0, x1 - x0), h: Math.max(0, y1 - y0) }, '', 0);
  return out;
}
