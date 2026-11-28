// The layout of a spec's panels as plain rectangles, a port of the native
// spec runtime's src/host/spec_runtime/spec_layout.cpp, so that the browser
// and the app place every panel on the same pixels. web/test/spec-layout.spec.ts
// runs the cases of the native tests (tests/test_spec_layout.cpp) against it.
//
// A split gives each child its share of `shares` (spec version 1), else its
// first child `ratio` of its length (0.5 by default), or equal shares to more
// children; tabs show one child under a strip of labels; a 1 px rule
// separates the children of a split. Every panel has a minimum of 12 by 8
// lines of the 15 px UI face, which grows with the text scale, or the
// leaf's own `min` in characters and lines; a split keeps each child at its
// minimum while the space allows and otherwise shares the space in proportion
// to the minimums. A leaf that names a control (spec version 1) takes one
// control row in a vertical split, and the other children share the rest.
// Every edge is a whole pixel. Rounding is C++'s std::round, half away from zero.

import type { LayoutNode, LeafNode, SplitNode, Spec, SpecControl, SpecPanel, TabsNode } from './types.js';

export interface Rect { x: number; y: number; w: number; h: number }

export interface LayoutMetrics {
  minPanelW: number; minPanelH: number; tabBarH: number; gap: number;
  /** A control leaf's minimum width and its row's height (spec_control_height: two lines of the UI face). */
  minControlW: number; controlRowH: number;
  /** The width of one character and the height of one line, for a leaf's `min`. */
  charW: number; lineH: number;
}

export interface LayoutPanel { id: string; path: string; rect: Rect; tab: boolean; control: boolean }
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
  return {
    minPanelW: cround(f32(12 * line)), minPanelH: cround(f32(8 * line)), tabBarH: cround(f32(line + f32(9 * s))), gap: 1,
    minControlW: cround(f32(8 * line)), controlRowH: cround(f32(2 * line)),
    charW: cround(f32(f32(0.55) * line)), lineH: cround(f32(f32(1.2) * line)),
  };
}

const isSplit = (n: LayoutNode): n is SplitNode => typeof n === 'object' && n !== null && 'split' in n;
const isTabs = (n: LayoutNode): n is TabsNode => typeof n === 'object' && n !== null && 'tabs' in n;
const isLeaf = (n: LayoutNode): n is string | LeafNode => typeof n === 'string' || (typeof n === 'object' && n !== null && 'panel' in n);
/** The id a leaf names, panel or control. */
export const leafId = (n: string | LeafNode): string => (typeof n === 'string' ? n : n.panel);
const childrenOf = (n: LayoutNode): LayoutNode[] => (isSplit(n) ? n.children ?? [] : isTabs(n) ? n.tabs ?? [] : []);

/**
 * The spec's layout, or its default (spec_edit.cpp, effective_layout): one
 * panel fills the instrument, several stack vertically, and the controls stand in
 * one row under them in their order.
 */
export function effectiveLayout(spec: Spec): LayoutNode | null {
  if (spec.layout !== undefined) return spec.layout;
  const panels = spec.panel ?? [], controls = spec.control ?? [];
  const top: LayoutNode | null = !panels.length ? null : panels.length === 1 ? panels[0].id
    : { split: 'vertical', children: panels.map((p) => p.id) };
  if (!controls.length) return top;
  const row: LayoutNode = controls.length === 1 ? controls[0].id : { split: 'horizontal', children: controls.map((c) => c.id) };
  return top === null ? row : { split: 'vertical', children: [top, row] };
}

export function findPanel(spec: Spec, id: string): SpecPanel | undefined {
  return (spec.panel ?? []).find((p) => p.id === id);
}

export function findControl(spec: Spec, id: string): SpecControl | undefined {
  return (spec.control ?? []).find((c) => c.id === id);
}

/** The control a leaf names, when no panel has its id. */
const controlOf = (spec: Spec | null, id: string): SpecControl | undefined =>
  spec && !findPanel(spec, id) ? findControl(spec, id) : undefined;

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
    case 'scene': {
      // The live scene's name, as catalog.list names it.
      const b = bound('scene');
      if (b && b[0].startsWith('live:') && b[0].length > 5) return b[0].slice(5);
      break;
    }
  }
  return panel.primitive.charAt(0).toUpperCase() + panel.primitive.slice(1);
}

function firstPanel(n: LayoutNode): string {
  if (isLeaf(n)) return leafId(n);
  for (const c of childrenOf(n)) {
    const id = firstPanel(c);
    if (id) return id;
  }
  return '';
}

function tabLabel(n: LayoutNode, spec: Spec): string {
  if (!isLeaf(n) && n.title) return n.title;
  const id = firstPanel(n);
  const p = findPanel(spec, id);
  return p ? panelTitle(spec, p) : id;
}

const childPath = (path: string, i: number) => (path ? `${path}.${i}` : String(i));

/** A control's height in a vertical split: one control row (spec_control_height). */
const controlHeight = (_c: SpecControl, m: LayoutMetrics): number => m.controlRowH;

/**
 * The height a node takes in a vertical split when everything in it is a
 * control (spec_layout_fixed_height): a control leaf's height, the tallest
 * child of a horizontal split of such nodes, the children of a vertical one
 * with their gaps. Null when the node holds a panel or tabs.
 */
export function layoutFixedHeight(n: LayoutNode, spec: Spec, m: LayoutMetrics): number | null {
  if (isLeaf(n)) {
    const c = controlOf(spec, leafId(n));
    return c ? controlHeight(c, m) : null;
  }
  if (isTabs(n) || !isSplit(n)) return null;
  const kids = childrenOf(n);
  if (!kids.length) return null;
  const across = n.split === 'horizontal';
  let h = 0;
  for (const c of kids) {
    const ch = layoutFixedHeight(c, spec, m);
    if (ch === null) return null;
    h = across ? Math.max(h, ch) : f32(h + ch);
  }
  if (!across) h = f32(h + f32(m.gap * (kids.length - 1)));
  return h;
}

/** The minimum size of a node; with the spec, a control leaf has a control's minimum. */
export function layoutMinSize(n: LayoutNode, m: LayoutMetrics, spec: Spec | null = null): { w: number; h: number } {
  if (isLeaf(n)) {
    const min = typeof n === 'string' ? undefined : n.min;
    const c = controlOf(spec, leafId(n));
    if (c) return { w: min ? cround(f32(min[0] * m.charW)) : m.minControlW, h: controlHeight(c, m) };
    if (min) return { w: cround(f32(min[0] * m.charW)), h: cround(f32(min[1] * m.lineH)) };
    return { w: m.minPanelW, h: m.minPanelH };
  }
  if (isTabs(n)) {
    let w = 0, h = 0;
    for (const c of childrenOf(n)) {
      const r = layoutMinSize(c, m, spec);
      w = Math.max(w, r.w);
      h = Math.max(h, r.h);
    }
    return { w, h: f32(h + m.tabBarH) };
  }
  if (!isSplit(n)) return { w: 0, h: 0 };
  const across = n.split === 'horizontal';
  const kids = childrenOf(n);
  let along = 0, cross = 0;
  for (const c of kids) {
    const r = layoutMinSize(c, m, spec);
    along = f32(along + (across ? r.w : r.h));
    cross = Math.max(cross, across ? r.h : r.w);
  }
  if (kids.length) along = f32(along + f32(m.gap * (kids.length - 1)));
  return across ? { w: along, h: cross } : { w: cross, h: along };
}

/**
 * The shares of `length` for children with minimums `mins` along one axis
 * (spec_layout_shares): by the weights of a split's `shares` when there is one
 * per child, else two children with `ratio` and more equally; each at least
 * its minimum while the length allows, else in proportion to the minimums.
 * Whole pixels that sum to the length.
 */
export function layoutShares(length: number, mins: number[], ratio: number, weights: number[] = []): number[] {
  const k = mins.length;
  if (!k) return [];
  const share = new Array<number>(k).fill(0);
  const L = Math.max(0, cround(f32(length)));
  const need = mins.reduce((a, b) => a + b, 0);
  // A weight that is not a positive finite number counts as 1 (the validator refuses it).
  const weighted = weights.length === k;
  const w = mins.map((_, i) => (weighted && Number.isFinite(weights[i]) && weights[i] > 0 ? weights[i] : 1));
  if (need >= L) {
    for (let i = 0; i < k; i++) share[i] = need > 0 ? (L * mins[i]) / need : L / k;
  } else if (k === 2 && !weighted) {
    const r = Math.min(1, Math.max(0, ratio));
    share[0] = Math.min(Math.max(r * L, mins[0]), L - mins[1]);
    share[1] = L - share[0];
  } else {
    // Shares by weight (equal without weights), raising each child below its
    // minimum to it and sharing what is left among the others.
    const fixed = new Array<boolean>(k).fill(false);
    for (let changed = true; changed;) {
      changed = false;
      let left = L, free = 0;
      for (let i = 0; i < k; i++) {
        if (fixed[i]) left -= mins[i];
        else free += w[i];
      }
      for (let i = 0; i < k; i++) {
        const part = free > 0 ? (left * w[i]) / free : 0;
        if (fixed[i]) share[i] = mins[i];
        else if (mins[i] > part) {
          fixed[i] = true;
          changed = true;
        } else share[i] = part;
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

/** The rectangles of the layout `root` in `area` (compute_spec_layout). */
export function computeLayout(root: LayoutNode, spec: Spec, area: Rect, m: LayoutMetrics, choice: TabChoice = {}): SpecLayout {
  const out: SpecLayout = { panels: [], tabs: [] };
  const place = (n: LayoutNode, r: Rect, path: string, depth: number, tabChild = false): void => {
    if (depth > 64) return;
    if (isLeaf(n)) {
      const id = leafId(n);
      out.panels.push({ id, path, rect: r, tab: tabChild, control: !!controlOf(spec, id) });
      return;
    }
    if (isTabs(n)) {
      const tabs = childrenOf(n);
      if (!tabs.length) return;
      const barH = Math.min(r.h, cround(m.tabBarH));
      const c = choice[path];
      const active = c !== undefined && c >= 0 && c < tabs.length ? c : 0;
      out.tabs.push({ path, bar: { x: r.x, y: r.y, w: r.w, h: barH }, labels: tabs.map((t) => tabLabel(t, spec)), active });
      place(tabs[active], { x: r.x, y: f32(r.y + barH), w: r.w, h: Math.max(0, f32(r.h - barH)) }, childPath(path, active), depth + 1, true);
      return;
    }
    if (!isSplit(n)) return;
    const kids = childrenOf(n);
    const k = kids.length;
    if (!k) return;
    const across = n.split === 'horizontal';
    const total = across ? r.w : r.h;
    const gaps = f32(m.gap * (k - 1));
    const length = Math.max(0, f32(total - gaps));
    const mins = kids.map((c) => {
      const ms = layoutMinSize(c, m, spec);
      return across ? ms.w : ms.h;
    });
    const weights = n.shares ?? [];
    // In a vertical split, a row of controls takes its own height and the
    // other children share the rest.
    const fixed = kids.map((c) => (across ? null : layoutFixedHeight(c, spec, m)));
    let shares: number[];
    if (fixed.every((f) => f === null)) {
      shares = layoutShares(length, mins, n.ratio ?? 0.5, weights);
    } else {
      let left = length;
      shares = new Array<number>(k).fill(0);
      for (let i = 0; i < k; i++) {
        const f = fixed[i];
        if (f !== null) {
          shares[i] = Math.min(cround(f), left);
          left = f32(left - shares[i]);
        }
      }
      const flexMins: number[] = [], flexWeights: number[] = [];
      for (let i = 0; i < k; i++) {
        if (fixed[i] !== null) continue;
        flexMins.push(mins[i]);
        if (weights.length === k) flexWeights.push(weights[i]);
      }
      if (flexMins.length) {
        const flex = layoutShares(left, flexMins, flexMins.length === 2 ? n.ratio ?? 0.5 : 0.5, flexWeights);
        for (let i = 0, j = 0; i < k; i++) if (fixed[i] === null) shares[i] = flex[j++];
      }
    }
    // With no room for the gaps, the children overlap rather than leave the area.
    const gap = total >= gaps ? m.gap : 0;
    let at = across ? r.x : r.y;
    for (let i = 0; i < k; i++) {
      const cr = across ? { x: at, y: r.y, w: shares[i], h: r.h } : { x: r.x, y: at, w: r.w, h: shares[i] };
      place(kids[i], cr, childPath(path, i), depth + 1);
      at = f32(f32(at + shares[i]) + gap);
    }
  };
  // Whole-pixel edges of the area.
  const x0 = cround(area.x), y0 = cround(area.y);
  const x1 = cround(area.x + area.w), y1 = cround(area.y + area.h);
  const r = { x: x0, y: y0, w: Math.max(0, x1 - x0), h: Math.max(0, y1 - y0) };
  // A layout of controls only keeps their rows at the top of the area.
  const h = layoutFixedHeight(root, spec, m);
  if (h !== null) r.h = Math.min(r.h, cround(h));
  place(root, r, '', 0);
  return out;
}
