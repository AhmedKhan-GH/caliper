// A 2-D plot frame: the visible data window, the plot area inside the canvas,
// the transforms between them, pan and zoom, and the axes with their ticks.
// All positions are CSS pixels relative to the view's body.

import { C, FONT_MONO, fmt } from './theme.js';

export interface Rect { x: number; y: number; w: number; h: number }

export interface Range { lo: number; hi: number }

export class Plot2D {
  x: Range = { lo: 0, hi: 1 };
  y: Range = { lo: 0, hi: 1 };
  /** Rows grow downward (heatmaps and images). */
  yDown = false;
  /** y is drawn on a log10 scale (lo and hi stay in data units). */
  logY = false;
  area: Rect = { x: 0, y: 0, w: 1, h: 1 };

  private ty(v: number): number {
    return this.logY ? Math.log10(v) : v;
  }

  private uty(t: number): number {
    return this.logY ? 10 ** t : t;
  }

  xToPx(v: number): number {
    return this.area.x + ((v - this.x.lo) / (this.x.hi - this.x.lo)) * this.area.w;
  }

  pxToX(p: number): number {
    return this.x.lo + ((p - this.area.x) / this.area.w) * (this.x.hi - this.x.lo);
  }

  yToPx(v: number): number {
    const f = (this.ty(v) - this.ty(this.y.lo)) / (this.ty(this.y.hi) - this.ty(this.y.lo));
    return this.yDown ? this.area.y + f * this.area.h : this.area.y + this.area.h - f * this.area.h;
  }

  pxToY(p: number): number {
    const f = this.yDown ? (p - this.area.y) / this.area.h : (this.area.y + this.area.h - p) / this.area.h;
    return this.uty(this.ty(this.y.lo) + f * (this.ty(this.y.hi) - this.ty(this.y.lo)));
  }

  inside(px: number, py: number): boolean {
    const a = this.area;
    return px >= a.x && px <= a.x + a.w && py >= a.y && py <= a.y + a.h;
  }

  /** Which axis a point is over: the plot area, the x axis band or the y axis band. */
  region(px: number, py: number): 'plot' | 'x' | 'y' | 'none' {
    const a = this.area;
    if (this.inside(px, py)) return 'plot';
    if (px >= a.x && px <= a.x + a.w && py > a.y + a.h) return 'x';
    if (py >= a.y && py <= a.y + a.h && px < a.x) return 'y';
    return 'none';
  }

  /** Zooms by factor (> 1 zooms in) about a point, on the given axes. */
  zoom(px: number, py: number, factor: number, axes: 'x' | 'y' | 'xy'): void {
    if (axes !== 'y') {
      const c = this.pxToX(px);
      this.x = { lo: c + (this.x.lo - c) / factor, hi: c + (this.x.hi - c) / factor };
    }
    if (axes !== 'x') {
      const c = this.ty(this.pxToY(py));
      const lo = this.ty(this.y.lo), hi = this.ty(this.y.hi);
      this.y = { lo: this.uty(c + (lo - c) / factor), hi: this.uty(c + (hi - c) / factor) };
    }
  }

  /** Moves the window so the data under the pointer follows a drag of (dx, dy). */
  /** A function that puts the visible ranges back as they are now. */
  keep(): () => void {
    const x = { ...this.x }, y = { ...this.y };
    return () => {
      this.x = { ...x };
      this.y = { ...y };
    };
  }

  pan(dx: number, dy: number, axes: 'x' | 'y' | 'xy' = 'xy'): void {
    if (axes !== 'y') {
      const d = (dx / this.area.w) * (this.x.hi - this.x.lo);
      this.x = { lo: this.x.lo - d, hi: this.x.hi - d };
    }
    if (axes !== 'x') {
      const lo = this.ty(this.y.lo), hi = this.ty(this.y.hi);
      const d = ((this.yDown ? dy : -dy) / this.area.h) * (hi - lo);
      this.y = { lo: this.uty(lo - d), hi: this.uty(hi - d) };
    }
  }

  /** Sets equal data units per pixel on both axes, growing the shorter range about its centre. */
  equalAspect(): void {
    const sx = (this.x.hi - this.x.lo) / this.area.w;
    const sy = (this.y.hi - this.y.lo) / this.area.h;
    if (sx > sy) {
      const c = (this.y.lo + this.y.hi) / 2, h = (sx * this.area.h) / 2;
      this.y = { lo: c - h, hi: c + h };
    } else {
      const c = (this.x.lo + this.x.hi) / 2, h = (sy * this.area.w) / 2;
      this.x = { lo: c - h, hi: c + h };
    }
  }
}

/** Evenly spaced round tick values in [lo, hi], about n of them. */
export function ticks(lo: number, hi: number, n: number, integer = false): { values: number[]; step: number } {
  if (!(hi > lo) || !Number.isFinite(lo) || !Number.isFinite(hi)) return { values: [], step: 1 };
  const raw = (hi - lo) / Math.max(1, n);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const r = raw / mag;
  let step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
  if (integer) step = Math.max(1, Math.round(step));
  const values: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) values.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { values, step };
}

/** Decade ticks for a log axis. */
export function logTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  if (!(lo > 0) || !(hi > lo)) return out;
  const a = Math.floor(Math.log10(lo)), b = Math.ceil(Math.log10(hi));
  const every = Math.max(1, Math.ceil((b - a) / 6));
  for (let e = a; e <= b; e += every) {
    const v = 10 ** e;
    if (v >= lo && v <= hi) out.push(v);
  }
  return out;
}

export interface AxisOptions {
  /** Ticks at integer positions only (indices). */
  xInteger?: boolean;
  yInteger?: boolean;
  /** Ticks label the centre of cells: value v labels position v + 0.5. */
  cells?: boolean;
  xTitle?: string;
  yTitle?: string;
  /** Turns a tick value into its label (for index axes mapped to source indices). */
  xLabel?: (v: number) => string;
  yLabel?: (v: number) => string;
}

/** Draws the frame, tick marks and labels of a plot into a 2-D context (CSS pixel units). */
export function drawAxes(ctx: CanvasRenderingContext2D, p: Plot2D, o: AxisOptions = {}): void {
  const a = p.area;
  ctx.save();
  ctx.font = `10.5px ${FONT_MONO}`;
  ctx.lineWidth = 1;
  ctx.strokeStyle = C.rule;
  ctx.strokeRect(Math.round(a.x) - 0.5, Math.round(a.y) - 0.5, Math.round(a.w) + 1, Math.round(a.h) + 1);
  ctx.fillStyle = C.faint;
  const off = o.cells ? 0.5 : 0;
  // x axis
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const xlo = p.x.lo - off, xhi = p.x.hi - off;
  const xt = ticks(Math.max(xlo, o.cells ? 0 : -Infinity), xhi, Math.max(2, Math.floor(a.w / 70)), o.xInteger);
  ctx.beginPath();
  for (const v of xt.values) {
    const px = Math.round(p.xToPx(v + off)) + 0.5;
    if (px < a.x - 0.5 || px > a.x + a.w + 0.5) continue;
    ctx.moveTo(px, a.y + a.h + 0.5);
    ctx.lineTo(px, a.y + a.h + 4.5);
    ctx.fillText(o.xLabel ? o.xLabel(v) : fmtTick(v, xt.step), px, a.y + a.h + 6);
  }
  // y axis
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  let yvals: number[];
  let ystep = 1;
  if (p.logY) {
    yvals = logTicks(Math.min(p.y.lo, p.y.hi), Math.max(p.y.lo, p.y.hi));
  } else {
    const lo = Math.min(p.y.lo, p.y.hi) - off, hi = Math.max(p.y.lo, p.y.hi) - off;
    const yt = ticks(Math.max(lo, o.cells ? 0 : -Infinity), hi, Math.max(2, Math.floor(a.h / 36)), o.yInteger);
    yvals = yt.values;
    ystep = yt.step;
  }
  for (const v of yvals) {
    const py = Math.round(p.yToPx(v + (p.logY ? 0 : off))) + 0.5;
    if (py < a.y - 0.5 || py > a.y + a.h + 0.5) continue;
    ctx.moveTo(a.x - 0.5, py);
    ctx.lineTo(a.x - 4.5, py);
    ctx.fillText(o.yLabel ? o.yLabel(v) : p.logY ? fmt(v, 2) : fmtTick(v, ystep), a.x - 6, py);
  }
  ctx.stroke();
  if (o.xTitle) {
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText(o.xTitle, a.x + a.w, a.y + a.h + 19);
  }
  if (o.yTitle) {
    ctx.save();
    ctx.translate(a.x - 6, a.y - 4);
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText(o.yTitle, 0, 0);
    ctx.restore();
  }
  ctx.restore();
}

/** A tick's label: fixed decimals to the step, or exponential far from 1. */
export function fmtTick(v: number, step: number): string {
  if (Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0)) return fmt(v, 3);
  const decimals = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return v.toFixed(Math.min(6, decimals));
}

/** Sizes a canvas to its CSS box at the device pixel ratio; true when it changed. */
export function fitCanvas(c: HTMLCanvasElement, w: number, h: number, dpr: number): boolean {
  const W = Math.max(1, Math.round(w * dpr)), H = Math.max(1, Math.round(h * dpr));
  if (c.width === W && c.height === H) return false;
  c.width = W;
  c.height = H;
  return true;
}
