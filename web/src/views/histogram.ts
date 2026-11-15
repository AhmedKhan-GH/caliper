// The histogram live view: bin counts over [lo, hi] drawn as bars, with pan
// and zoom, hover of a bin's range and count, a log scale for counts and a
// range selection.

import type { SnapshotMsg } from '../protocol/messages.js';
import { toggle, type LiveView, type ViewHost } from './base.js';
import { attachGestures } from './interact.js';
import { drawAxes, fitCanvas, Plot2D } from './plot.js';
import { C, fmt } from './theme.js';

export class HistogramView implements LiveView {
  private readonly plot = new Plot2D();
  private readonly canvas = document.createElement('canvas');
  private readonly ctx: CanvasRenderingContext2D;
  private readonly octx: CanvasRenderingContext2D;
  private readonly detach: () => void;
  private counts: Float32Array = new Float32Array(0);
  private lo = 0;
  private hi = 1;
  private seq: number | null = null;
  private auto = true;
  private hover: number | null = null;
  private hoverPx: { x: number; y: number } | null = null;
  private dragSel: { x0: number; x1: number } | null = null;
  private selection: { b0: number; b1: number } | null = null;
  private frames = 0;

  constructor(private readonly host: ViewHost) {
    this.canvas.className = 'data';
    const ctx = this.canvas.getContext('2d', { alpha: false });
    const octx = host.overlay.getContext('2d');
    if (!ctx || !octx) throw new Error('no 2d context');
    this.ctx = ctx;
    this.octx = octx;
    host.body.prepend(this.canvas);
    toggle(host, 'log', 'Log scale for counts', false, (on) => {
      this.plot.logY = on;
      this.auto = true;
      host.request('data');
    });
    this.detach = attachGestures(host.overlay, {
      hover: (x, y) => this.onHover(x, y),
      leave: () => this.onHover(NaN, NaN),
      drag: (dx, dy) => {
        this.auto = false;
        this.plot.pan(dx, dy);
        host.request('data');
      },
      wheel: (x, y, k) => {
        const reg = this.plot.region(x, y);
        if (reg === 'none') return;
        this.auto = false;
        this.plot.zoom(x, y, k, reg === 'plot' ? 'xy' : reg);
        host.request('data');
      },
      dblclick: () => {
        this.auto = true;
        host.request('data');
      },
      select: (phase, x) => this.onSelect(phase, x),
      click: (x, y) => {
        const b = this.binAt(x, y);
        if (b !== null && !this.dragSel) host.report('pick', this.binDetail(b));
      },
    });
  }

  snapshot(s: SnapshotMsg): void {
    this.counts = s.data as Float32Array;
    const n = this.counts.length;
    // The range is stats[0..1]; without it the bins are indices [0, B).
    const lo = s.stats[0], hi = s.stats[1];
    if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) {
      this.lo = lo;
      this.hi = hi;
    } else {
      this.lo = 0;
      this.hi = n;
    }
    this.seq = s.seq;
    let total = 0;
    for (const c of this.counts) total += c;
    this.host.info(`${n} bins  ${fmt(this.lo)} … ${fmt(this.hi)}  n ${fmt(total)}`);
    if (this.hoverPx) this.onHover(this.hoverPx.x, this.hoverPx.y);
  }

  private get width(): number {
    return (this.hi - this.lo) / Math.max(1, this.counts.length);
  }

  private layout(): void {
    const { w, h } = this.host.size();
    this.plot.area = { x: 54, y: 14, w: Math.max(10, w - 54 - 12), h: Math.max(10, h - 14 - 30) };
    if (!this.auto) return;
    let max = 0, minPos = Infinity;
    for (const c of this.counts) {
      if (c > max) max = c;
      if (c > 0 && c < minPos) minPos = c;
    }
    this.plot.x = { lo: this.lo, hi: this.hi };
    if (this.plot.logY) {
      const top = Math.max(1, max);
      const bottom = Number.isFinite(minPos) ? Math.min(minPos, 1) * 0.5 : 0.5;
      this.plot.y = { lo: bottom, hi: top * 1.6 };
    } else this.plot.y = { lo: 0, hi: Math.max(1, max) * 1.06 };
  }

  private binAt(x: number, y: number): number | null {
    if (Number.isNaN(x) || !this.plot.inside(x, y) || !this.counts.length) return null;
    const b = Math.floor((this.plot.pxToX(x) - this.lo) / this.width);
    return b >= 0 && b < this.counts.length ? b : null;
  }

  private binDetail(b: number): Record<string, unknown> {
    return { bin: b, lo: this.lo + b * this.width, hi: this.lo + (b + 1) * this.width, count: this.counts[b] };
  }

  private onHover(x: number, y: number): void {
    this.hoverPx = Number.isNaN(x) ? null : { x, y };
    const b = this.binAt(x, y);
    const changed = b !== this.hover;
    this.hover = b;
    if (b === null) this.host.tooltip(null);
    else {
      const a = this.lo + b * this.width, z = a + this.width;
      this.host.tooltip(`bin ${b}  [${fmt(a)}, ${fmt(z)})\ncount ${fmt(this.counts[b], 6)}`, x, y);
      if (changed) this.host.report('hover', this.binDetail(b));
    }
    if (changed) this.host.request('overlay');
  }

  private onSelect(phase: 'start' | 'move' | 'end', x: number): void {
    if (phase === 'start') {
      const v = this.plot.pxToX(x);
      this.dragSel = { x0: v, x1: v };
    } else if (this.dragSel && phase === 'move') {
      this.dragSel.x1 = this.plot.pxToX(x);
    } else if (this.dragSel && phase === 'end') {
      const s = this.dragSel;
      this.dragSel = null;
      const n = this.counts.length, wd = this.width;
      const b0 = Math.max(0, Math.floor((Math.min(s.x0, s.x1) - this.lo) / wd));
      const b1 = Math.min(n, Math.ceil((Math.max(s.x0, s.x1) - this.lo) / wd));
      if (!Number.isNaN(x) && b1 > b0) {
        this.selection = { b0, b1 };
        this.host.report('select', { range: [this.lo + b0 * wd, this.lo + b1 * wd] });
      } else if (this.selection) {
        this.selection = null;
        this.host.report('select', {});
      }
    }
    this.host.request('overlay');
  }

  draw(data: boolean): void {
    const { w, h, dpr } = this.host.size();
    this.layout();
    const resized = fitCanvas(this.canvas, w, h, dpr);
    if (data || resized) {
      this.drawData(w, h, dpr);
      this.frames++;
    }
    this.drawOverlay(w, h, dpr);
  }

  private bar(ctx: CanvasRenderingContext2D, b: number): void {
    const p = this.plot, a = p.area;
    const c = this.counts[b];
    if (!(c > 0) || (p.logY && c <= 0)) return;
    const x0 = p.xToPx(this.lo + b * this.width), x1 = p.xToPx(this.lo + (b + 1) * this.width);
    const base = p.logY ? a.y + a.h : p.yToPx(0);
    const top = p.yToPx(c);
    const gap = x1 - x0 > 4 ? 1 : 0;
    ctx.fillRect(x0 + gap / 2, top, Math.max(0.75, x1 - x0 - gap), base - top);
  }

  private drawData(w: number, h: number, dpr: number): void {
    const ctx = this.ctx, a = this.plot.area;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = C.plate;
    ctx.fillRect(0, 0, w, h);
    drawAxes(ctx, this.plot, {});
    ctx.save();
    ctx.beginPath();
    ctx.rect(a.x, a.y, a.w, a.h);
    ctx.clip();
    ctx.fillStyle = C.secondary;
    for (let b = 0; b < this.counts.length; b++) this.bar(ctx, b);
    ctx.restore();
  }

  private drawOverlay(w: number, h: number, dpr: number): void {
    const ctx = this.octx, p = this.plot, a = p.area;
    fitCanvas(this.host.overlay, w, h, dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.beginPath();
    ctx.rect(a.x, a.y, a.w, a.h);
    ctx.clip();
    const band = (x0: number, x1: number, dashed: boolean) => {
      const X0 = p.xToPx(x0), X1 = p.xToPx(x1);
      ctx.fillStyle = 'rgba(179,139,89,0.16)';
      ctx.fillRect(Math.min(X0, X1), a.y, Math.abs(X1 - X0), a.h);
      ctx.strokeStyle = C.bronze;
      ctx.setLineDash(dashed ? [4, 3] : []);
      ctx.strokeRect(Math.min(X0, X1), a.y, Math.abs(X1 - X0), a.h);
      ctx.setLineDash([]);
    };
    if (this.selection) band(this.lo + this.selection.b0 * this.width, this.lo + this.selection.b1 * this.width, false);
    if (this.dragSel) band(this.dragSel.x0, this.dragSel.x1, true);
    if (this.hover !== null) {
      ctx.fillStyle = C.bronze;
      this.bar(ctx, this.hover);
    }
    ctx.restore();
  }

  resize(): void {}

  destroy(): void {
    this.detach();
    this.canvas.remove();
  }

  debug(): Record<string, unknown> {
    return {
      kind: 'histogram', renderer: 'canvas2d', bins: this.counts.length, range: [this.lo, this.hi], seq: this.seq, logY: this.plot.logY,
      window: { x0: this.plot.x.lo, x1: this.plot.x.hi, y0: this.plot.y.lo, y1: this.plot.y.hi },
      area: { ...this.plot.area }, hover: this.hover, selection: this.selection, frames: this.frames,
    };
  }
}
