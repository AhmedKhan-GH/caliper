// A 2-D plot as ImPlot lays it out and draws it in the native spec runtime's
// house style (push_plot_style in src/host/spec_runtime/spec_runtime.cpp, and
// third_party/implot's BeginPlot, Locator_Default and legend): the plot
// padding, the axes' labels and tick labels in the mono face at 13, the tick
// marks, the grid of major and minor ticks, the border, and a legend inside
// the north-east corner. The data is drawn by the caller between frame() and
// finish(), clipped to the plot rectangle.

import { fmtG } from './format.js';
import { COL, SIZE, fill, outline, text, textWidth } from './look.js';
import type { Rect } from './layout.js';

/** The house's ImPlot style at text scale 1. */
const PAD = { x: 4, y: 6 };          // PlotPadding
const LABEL_PAD = { x: 5, y: 4 };    // LabelPadding
const MAJOR_TICK = 4, MINOR_TICK = 2;
const LEGEND_PAD = { x: 8, y: 8 }, LEGEND_INNER = { x: 6, y: 4 }, LEGEND_SPACING_Y = 0;
const T = SIZE.axes;                 // the plot's text line height
const MINOR_ALPHA = 0.25;

export interface Tick { value: number; major: boolean; label: string; show: boolean }

export interface Axis {
  lo: number;
  hi: number;
  log: boolean;
  label: string;
  /** Ticks given by the caller (a log axis's), else ImPlot's default locator. */
  ticks?: Tick[];
}

const IM_ROUND = (x: number) => Math.trunc(x + 0.5);

function niceNum(x: number, round: boolean): number {
  const expv = Math.floor(Math.log10(x));
  const f = x / 10 ** expv;
  let nf: number;
  if (round) nf = f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10;
  else nf = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
  return nf * 10 ** expv;
}

/** ImPlot's Locator_Default: majors and minors, labelled with %g, pruned when they crowd. */
export function defaultTicks(lo: number, hi: number, pixels: number, vertical: boolean, width: (s: string) => number): Tick[] {
  const ticks: Tick[] = [];
  if (lo === hi || !Number.isFinite(lo) || !Number.isFinite(hi)) return ticks;
  const contains = (v: number) => v >= lo && v <= hi;
  const nMajor = Math.max(2, IM_ROUND(pixels / (vertical ? 300 : 400)));
  const niceRange = niceNum((hi - lo) * 0.99, false);
  const interval = niceNum(niceRange / (nMajor - 1), true);
  const graphmin = Math.floor(lo / interval) * interval, graphmax = Math.ceil(hi / interval) * interval;
  let firstMajor = -1, total = 0;
  for (let major = graphmin; major < graphmax + 0.5 * interval && ticks.length < 2000; major += interval) {
    if (major - interval < 0 && major + interval > 0) major = 0;
    if (contains(major)) {
      if (firstMajor < 0) firstMajor = ticks.length;
      const label = fmtG(major);
      ticks.push({ value: major, major: true, label, show: true });
      total += vertical ? T : width(label);
    }
    for (let i = 1; i < 10; i++) {
      const minor = major + (i * interval) / 10;
      if (!contains(minor)) continue;
      const label = fmtG(minor);
      ticks.push({ value: minor, major: false, label, show: true });
      total += vertical ? T : width(label);
    }
  }
  if (total > pixels * (vertical ? 1 : 0.8)) {
    const first = Math.max(0, firstMajor);
    for (let i = first - 1; i >= 0; i -= 2) ticks[i].show = false;
    for (let i = first + 1; i < ticks.length; i += 2) ticks[i].show = false;
  }
  return ticks;
}

/** Labelled ticks for a log axis: 1, 2 and 5 of each decade, or only the decades past four of them (setup_log_ticks). */
export function logTicks(lo: number, hi: number, fmt: (v: number) => string): Tick[] {
  const out: Tick[] = [];
  if (!(lo > 0) || !(hi > lo)) return out;
  const d0 = Math.floor(Math.log10(lo)), d1 = Math.ceil(Math.log10(hi));
  const decadesOnly = d1 - d0 > 4;
  for (let d = d0; d <= d1 && out.length < 160; d++)
    for (let m = 1; m <= 9; m++) {
      const t = m * 10 ** d;
      if (t < lo || t > hi) continue;
      if (decadesOnly && d1 - d0 > 12 && m !== 1) continue;
      const label = decadesOnly ? m === 1 : m === 1 || m === 2 || m === 5;
      // Ticks that a plot is given are minor ones in ImPlot (AddTicksCustom).
      out.push({ value: t, major: false, label: label ? fmt(t) : '', show: true });
    }
  return out;
}

/** Limits of values with padding: `pad` of the span, of the decades on a log axis, one around a flat value (fit_limits). */
export function fitLimits(values: Iterable<number>, log: boolean, pad: number): [number, number] | null {
  let a = Infinity, b = -Infinity;
  for (const x of values) {
    if (!Number.isFinite(x) || (log && x <= 0)) continue;
    if (x < a) a = x;
    if (x > b) b = x;
  }
  if (!(b >= a)) return null;
  if (log) {
    const l = Math.log10(a), h = Math.log10(b), p = Math.max(0.05, (h - l) * pad);
    return [10 ** (l - p), 10 ** (h + p)];
  }
  if (b === a) return [a - 0.5, b + 0.5];
  return [a - (b - a) * pad, b + (b - a) * pad];
}

export interface LegendItem { name: string; color: string }

export class Plot2D {
  rect: Rect = { x: 0, y: 0, w: 1, h: 1 };   // the plot area, where the data goes
  private xt: Tick[] = [];
  private yt: Tick[] = [];
  private yTickW = 0;

  constructor(private readonly ctx: CanvasRenderingContext2D, private readonly frameRect: Rect,
              readonly x: Axis, readonly y: Axis) {}

  px(v: number): number {
    const a = this.x, r = this.rect;
    const t = a.log ? (Math.log10(v) - Math.log10(a.lo)) / (Math.log10(a.hi) - Math.log10(a.lo)) : (v - a.lo) / (a.hi - a.lo);
    return r.x + t * r.w;
  }

  py(v: number): number {
    const a = this.y, r = this.rect;
    const t = a.log ? (Math.log10(v) - Math.log10(a.lo)) / (Math.log10(a.hi) - Math.log10(a.lo)) : (v - a.lo) / (a.hi - a.lo);
    return r.y + r.h - t * r.h;
  }

  /** Lays the plot out and draws its background, grid and axes; then the caller draws its data. */
  frame(): void {
    const ctx = this.ctx, f = this.frameRect;
    const width = (s: string) => textWidth(ctx, 'mono', T, s);
    const canvas = { x: f.x + PAD.x, y: f.y + PAD.y, w: f.w - 2 * PAD.x, h: f.h - 2 * PAD.y };
    // Bottom: the x tick labels and the x axis label.
    const padB = (this.x.label ? T + LABEL_PAD.y : 0) + T + LABEL_PAD.y;
    const plotH = canvas.h - padB;
    this.yt = this.y.ticks ?? defaultTicks(this.y.lo, this.y.hi, plotH, true, width);
    this.yTickW = 0;
    // ImPlot sizes the axis by every label it formats (AddTick's MaxSize), the ones that it then hides
    // because they crowd included, so a plot with labels 2, 3, 4 keeps the width of 2.5.
    for (const t of this.yt) if (t.label) this.yTickW = Math.max(this.yTickW, width(t.label));
    const padL = (this.y.label ? T + LABEL_PAD.x : 0) + (this.yt.some((t) => t.label) ? this.yTickW + LABEL_PAD.x : 0);
    const plotW = canvas.w - padL;
    this.xt = this.x.ticks ?? defaultTicks(this.x.lo, this.x.hi, plotW, false, width);
    this.rect = { x: canvas.x + padL, y: canvas.y, w: Math.max(1, plotW), h: Math.max(1, plotH) };
    const r = this.rect;
    fill(ctx, r.x, r.y, r.w, r.h, COL.deep);
    // The grid: majors in the grid colour, minors at a quarter of its alpha,
    // fading out as they crowd and gone at one per five pixels (RenderGridLinesX).
    const minorAlpha = (n: number, len: number) => {
      const density = n / len;
      return density < 0.2 ? MINOR_ALPHA * Math.min(1, Math.max(0, 1 - (density - 0.1) / 0.1)) : 0;
    };
    const ax = minorAlpha(this.xt.length, r.w), ay = minorAlpha(this.yt.length, r.h);
    for (const t of this.xt) {
      const X = Math.round(this.px(t.value));
      if (X < r.x || X > r.x + r.w || (!t.major && !ax)) continue;
      ctx.globalAlpha = t.major ? 1 : ax;
      fill(ctx, X, r.y, 1, r.h, COL.ruleFaint);
    }
    for (const t of this.yt) {
      const Y = Math.round(this.py(t.value));
      if (Y < r.y || Y > r.y + r.h || (!t.major && !ay)) continue;
      ctx.globalAlpha = t.major ? 1 : ay;
      fill(ctx, r.x, Y, r.w, 1, COL.ruleFaint);
    }
    ctx.globalAlpha = 1;
  }

  /** The axes' tick marks, labels and titles, the border and the legend, over the data. */
  finish(legend: LegendItem[] = []): void {
    const ctx = this.ctx, r = this.rect;
    const width = (s: string) => textWidth(ctx, 'mono', T, s);
    const bottom = r.y + r.h;
    for (const t of this.xt) {
      const X = Math.round(this.px(t.value));
      if (X < r.x - 1 || X > r.x + r.w + 1) continue;
      if (X >= r.x && X <= r.x + r.w) fill(ctx, X, bottom - (t.major ? MAJOR_TICK : MINOR_TICK), 1, t.major ? MAJOR_TICK : MINOR_TICK, COL.rule);
      if (t.show && t.label) text(ctx, 'mono', T, X - 0.5 * width(t.label), bottom + LABEL_PAD.y, COL.faint, t.label);
    }
    for (const t of this.yt) {
      const Y = Math.round(this.py(t.value));
      if (Y < r.y - 1 || Y > bottom + 1) continue;
      if (Y >= r.y && Y <= bottom) fill(ctx, r.x, Y, t.major ? MAJOR_TICK : MINOR_TICK, 1, COL.rule);
      if (t.show && t.label) text(ctx, 'mono', T, r.x - LABEL_PAD.x - width(t.label), Y - 0.5 * T, COL.faint, t.label);
    }
    if (this.x.label)
      text(ctx, 'mono', T, r.x + r.w / 2 - width(this.x.label) / 2, bottom + T + 2 * LABEL_PAD.y, COL.faint, this.x.label);
    if (this.y.label) {
      const offset = (this.yt.some((t) => t.show && t.label) ? this.yTickW + LABEL_PAD.x : 0) + LABEL_PAD.x;
      ctx.save();
      ctx.translate(Math.round(r.x - offset - T), Math.round(r.y + r.h / 2 + width(this.y.label) / 2));
      ctx.rotate(-Math.PI / 2);
      text(ctx, 'mono', T, 0, 0, COL.faint, this.y.label);
      ctx.restore();
    }
    outline(ctx, r.x, r.y, r.x + r.w, r.y + r.h, COL.rule);
    if (legend.length) this.legend(legend);
  }

  private legend(items: LegendItem[]): void {
    const ctx = this.ctx, r = this.rect;
    const maxW = Math.max(...items.map((i) => textWidth(ctx, 'mono', T, i.name)));
    const w = LEGEND_INNER.x * 2 + T + maxW, h = LEGEND_INNER.y * 2 + items.length * T + (items.length - 1) * LEGEND_SPACING_Y;
    const x0 = Math.round(r.x + r.w - LEGEND_PAD.x - w), y0 = Math.round(r.y + LEGEND_PAD.y);
    ctx.globalAlpha = 230 / 255;
    fill(ctx, x0, y0, w, h, COL.ground);
    ctx.globalAlpha = 1;
    outline(ctx, x0, y0, x0 + w, y0 + h, COL.rule);
    items.forEach((it, i) => {
      const tx = x0 + LEGEND_INNER.x, ty = y0 + LEGEND_INNER.y + i * (T + LEGEND_SPACING_Y);
      fill(ctx, tx + 2, ty + 2, T - 4, T - 4, it.color);
      text(ctx, 'mono', T, tx + T, ty, COL.secondary, it.name);
    });
  }

  /** Clips the context to the plot area; restore() ends it. */
  clip(): void {
    const r = this.rect;
    this.ctx.save();
    this.ctx.beginPath();
    this.ctx.rect(r.x, r.y, r.w, r.h);
    this.ctx.clip();
  }
}
