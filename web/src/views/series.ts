// The series live view: float32 curves appended by deltas into a ring of the
// retained history, drawn with min/max decimation per pixel column, with pan
// and zoom on both axes, auto-follow of the newest values, hover with a
// crosshair, a log scale for y and a range selection.

import { PALETTE } from '../render/colormap.js';
import type { SeriesMsg, SnapshotMsg } from '../protocol/messages.js';
import { toggle, type LiveView, type ViewHost } from './base.js';
import { attachGestures } from './interact.js';
import { drawAxes, fitCanvas, Plot2D } from './plot.js';
import { C, fmt } from './theme.js';

/** Values by absolute append index; keeps the newest `cap`. */
export class Ring {
  readonly buf: Float32Array;
  /** Index after the newest value. */
  end = 0;
  /** Oldest index ever written. */
  private first = Infinity;

  constructor(readonly cap: number) {
    this.buf = new Float32Array(cap);
  }

  /** Oldest index still held. */
  get lo(): number {
    return Math.max(this.first, this.end - this.cap);
  }

  get length(): number {
    return Math.max(0, this.end - this.lo);
  }

  write(start: number, values: Float32Array): void {
    const n = values.length;
    if (!n) return;
    const stop = start + n;
    if (stop <= this.end - this.cap) return;
    if (start > this.end && this.end > 0) {
      // A gap: what the producer no longer retains reads as NaN.
      for (let i = Math.max(this.end, start - this.cap); i < start; i++) this.buf[i % this.cap] = NaN;
    }
    const from = Math.max(start, stop - this.cap);
    for (let i = from; i < stop; i++) this.buf[i % this.cap] = values[i - start];
    this.first = Math.min(this.first, from);
    if (stop > this.end) this.end = stop;
    if (this.first < this.end - this.cap) this.first = this.end - this.cap;
  }

  get(i: number): number {
    return i >= this.lo && i < this.end ? this.buf[i % this.cap] : NaN;
  }
}

const RING = 1 << 17;

export class SeriesView implements LiveView {
  private readonly plot = new Plot2D();
  private readonly canvas = document.createElement('canvas');
  private readonly ctx: CanvasRenderingContext2D;
  private readonly octx: CanvasRenderingContext2D;
  private readonly detach: () => void;
  private readonly curves: Ring[];
  private follow = true;
  private autoY = true;
  private hoverX: number | null = null;
  private hoverPx: { x: number; y: number } | null = null;
  private dragSel: { x0: number; x1: number } | null = null;
  private selection: { s0: number; s1: number } | null = null;
  private readonly followBtn;
  private frames = 0;
  private drawn = 0;

  constructor(private readonly host: ViewHost, private readonly names: string[]) {
    this.curves = names.map(() => new Ring(RING));
    this.canvas.className = 'data';
    const ctx = this.canvas.getContext('2d', { alpha: false });
    const octx = host.overlay.getContext('2d');
    if (!ctx || !octx) throw new Error('no 2d context');
    this.ctx = ctx;
    this.octx = octx;
    host.body.prepend(this.canvas);
    toggle(host, 'log', 'Log scale for y', false, (on) => {
      this.plot.logY = on;
      this.autoY = true;
      host.request('data');
    });
    this.followBtn = toggle(host, 'follow', 'Follow the newest values', true, (on) => {
      this.follow = on;
      if (on) this.autoY = true;
      host.request('data');
    });
    this.detach = attachGestures(host.overlay, {
      hover: (x, y) => this.onHover(x, y),
      leave: () => this.onHover(NaN, NaN),
      drag: (dx, dy) => {
        this.release();
        this.plot.pan(dx, dy);
        host.request('data');
      },
      wheel: (x, y, k) => {
        const reg = this.plot.region(x, y);
        if (reg === 'none') return;
        this.release();
        this.plot.zoom(x, y, k, reg === 'plot' ? 'xy' : reg);
        host.request('data');
      },
      dblclick: () => {
        this.follow = true;
        this.autoY = true;
        this.followBtn.set(true);
        host.request('data');
      },
      select: (phase, x) => this.onSelect(phase, x),
      click: (x) => {
        const i = this.indexAt(x);
        if (i !== null && !this.dragSel) this.host.report('pick', this.stepDetail(i));
      },
    });
  }

  private release(): void {
    this.follow = false;
    this.autoY = false;
    this.followBtn.set(false);
  }

  snapshot(_s: SnapshotMsg): void {
    /* a series view only takes SERIES deltas */
  }

  series(d: SeriesMsg, curve: number): void {
    const ring = this.curves[curve];
    if (!ring) return;
    ring.write(d.start, d.values);
    const last = ring.get(ring.end - 1);
    this.host.info(`${ring.end} steps  last ${fmt(last)}`);
    if (this.hoverPx) this.onHover(this.hoverPx.x, this.hoverPx.y);
  }

  private bounds(): { lo: number; hi: number } {
    let lo = Infinity, hi = 0;
    for (const c of this.curves) {
      if (!c.length) continue;
      lo = Math.min(lo, c.lo);
      hi = Math.max(hi, c.end);
    }
    return { lo: Number.isFinite(lo) ? lo : 0, hi };
  }

  private layout(): void {
    const { w, h } = this.host.size();
    this.plot.area = { x: 54, y: 14, w: Math.max(10, w - 54 - 12), h: Math.max(10, h - 14 - 30) };
    if (this.follow) {
      const b = this.bounds();
      const span = Math.max(1, b.hi - 1 - b.lo);
      this.plot.x = { lo: b.lo, hi: b.lo + span };
    }
    if (this.autoY) this.fitY();
  }

  private fitY(): void {
    let lo = Infinity, hi = -Infinity;
    const x0 = Math.floor(this.plot.x.lo), x1 = Math.ceil(this.plot.x.hi);
    const log = this.plot.logY;
    for (const c of this.curves) {
      const a = Math.max(c.lo, x0), b = Math.min(c.end - 1, x1);
      for (let i = a; i <= b; i++) {
        const v = c.buf[i % c.cap];
        if (!(v === v) || (log && v <= 0) || !Number.isFinite(v)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    if (!(hi >= lo)) {
      this.plot.y = log ? { lo: 0.1, hi: 10 } : { lo: 0, hi: 1 };
      return;
    }
    if (log) {
      const l = Math.log10(lo), hh = Math.log10(hi), pad = Math.max(0.05, (hh - l) * 0.05);
      this.plot.y = { lo: 10 ** (l - pad), hi: 10 ** (hh + pad) };
    } else {
      const pad = hi > lo ? (hi - lo) * 0.06 : Math.max(Math.abs(hi) * 0.1, 0.5);
      this.plot.y = { lo: lo - pad, hi: hi + pad };
    }
  }

  // --- interaction ------------------------------------------------------------

  private indexAt(x: number): number | null {
    if (Number.isNaN(x)) return null;
    const b = this.bounds();
    if (b.hi <= b.lo) return null;
    const i = Math.round(this.plot.pxToX(x));
    return Math.max(b.lo, Math.min(b.hi - 1, i));
  }

  /** A pick or hover: the absolute index, and the first curve's value there. */
  private stepDetail(i: number): Record<string, unknown> {
    const v = this.curves[0]?.get(i);
    return v !== undefined && Number.isFinite(v) ? { step: i, value: v } : { step: i };
  }

  private onHover(x: number, y: number): void {
    this.hoverPx = Number.isNaN(x) || !this.plot.inside(x, y) ? null : { x, y };
    const i = this.hoverPx ? this.indexAt(x) : null;
    const changed = i !== this.hoverX;
    this.hoverX = i;
    if (i === null) this.host.tooltip(null);
    else {
      const lines = [`step ${i}`];
      this.curves.forEach((c, k) => lines.push(`${this.names[k]} ${fmt(c.get(i), 5)}`));
      this.host.tooltip(lines.join('\n'), x, y);
      if (changed) this.host.report('hover', this.stepDetail(i));
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
      const b = this.bounds();
      const s0 = Math.max(b.lo, Math.ceil(Math.min(s.x0, s.x1)));
      const s1 = Math.min(b.hi, Math.floor(Math.max(s.x0, s.x1)) + 1);
      if (!Number.isNaN(x) && s1 > s0) {
        this.selection = { s0, s1 };
        this.host.report('select', { steps: [s0, s1] });
      } else if (this.selection) {
        this.selection = null;
        this.host.report('select', {});
      }
    }
    this.host.request('overlay');
  }

  // --- drawing ----------------------------------------------------------------

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

  private drawData(w: number, h: number, dpr: number): void {
    const ctx = this.ctx, p = this.plot, a = p.area;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = C.plate;
    ctx.fillRect(0, 0, w, h);
    drawAxes(ctx, p, { xInteger: true, xTitle: 'step' });
    ctx.save();
    ctx.beginPath();
    ctx.rect(a.x, a.y, a.w, a.h);
    ctx.clip();
    ctx.lineWidth = 1.25;
    ctx.lineJoin = 'round';
    this.drawn = 0;
    this.curves.forEach((c, k) => {
      ctx.strokeStyle = k === 0 ? C.parchment : `rgb(${PALETTE[(k - 1) % 16].join(',')})`;
      this.drawn += this.drawCurve(ctx, c, Math.max(1, Math.round(a.w * dpr)));
    });
    ctx.restore();
  }

  /** Strokes one curve, at most four vertices per pixel column; returns the vertex count. */
  private drawCurve(ctx: CanvasRenderingContext2D, c: Ring, cols: number): number {
    const p = this.plot, log = p.logY;
    const x0 = Math.max(c.lo, Math.floor(p.x.lo) - 1), x1 = Math.min(c.end - 1, Math.ceil(p.x.hi) + 1);
    if (x1 < x0) return 0;
    const ok = (v: number) => v === v && Number.isFinite(v) && (!log || v > 0);
    let pen = false, verts = 0;
    ctx.beginPath();
    const put = (i: number, v: number) => {
      const X = p.xToPx(i), Y = p.yToPx(v);
      if (pen) ctx.lineTo(X, Y);
      else ctx.moveTo(X, Y);
      pen = true;
      verts++;
    };
    if (x1 - x0 + 1 <= cols * 2) {
      for (let i = x0; i <= x1; i++) {
        const v = c.buf[i % c.cap];
        if (ok(v)) put(i, v);
        else pen = false;
      }
    } else {
      // Min/max decimation (M4): per pixel column the first, lowest, highest
      // and last values, in index order, draw the same picture as every value.
      const span = p.x.hi - p.x.lo;
      let col = -1, fi = 0, fv = 0, li = 0, lv = 0, mini = 0, minv = 0, maxi = 0, maxv = 0, have = false;
      const flush = () => {
        if (!have) return;
        put(fi, fv);
        if (mini < maxi) {
          if (mini !== fi) put(mini, minv);
          if (maxi !== li) put(maxi, maxv);
        } else {
          if (maxi !== fi) put(maxi, maxv);
          if (mini !== li) put(mini, minv);
        }
        if (li !== fi) put(li, lv);
      };
      for (let i = x0; i <= x1; i++) {
        const v = c.buf[i % c.cap];
        if (!ok(v)) {
          flush();
          have = false;
          pen = false;
          continue;
        }
        const k = Math.floor(((i - p.x.lo) / span) * cols);
        if (k !== col || !have) {
          flush();
          col = k;
          fi = li = mini = maxi = i;
          fv = lv = minv = maxv = v;
          have = true;
        } else {
          li = i;
          lv = v;
          if (v < minv) { minv = v; mini = i; }
          if (v > maxv) { maxv = v; maxi = i; }
        }
      }
      flush();
    }
    ctx.stroke();
    return verts;
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
    const band = (s0: number, s1: number, dashed: boolean) => {
      const x0 = p.xToPx(s0), x1 = p.xToPx(s1);
      ctx.fillStyle = 'rgba(179,139,89,0.16)';
      ctx.fillRect(Math.min(x0, x1), a.y, Math.abs(x1 - x0), a.h);
      ctx.strokeStyle = C.bronze;
      ctx.setLineDash(dashed ? [4, 3] : []);
      ctx.beginPath();
      ctx.moveTo(x0, a.y);
      ctx.lineTo(x0, a.y + a.h);
      ctx.moveTo(x1, a.y);
      ctx.lineTo(x1, a.y + a.h);
      ctx.stroke();
      ctx.setLineDash([]);
    };
    if (this.selection) band(this.selection.s0, this.selection.s1 - 1, false);
    if (this.dragSel) band(this.dragSel.x0, this.dragSel.x1, true);
    if (this.hoverX !== null) {
      const X = Math.round(p.xToPx(this.hoverX)) + 0.5;
      ctx.strokeStyle = C.rule;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(X, a.y);
      ctx.lineTo(X, a.y + a.h);
      ctx.stroke();
      this.curves.forEach((c) => {
        const v = c.get(this.hoverX!);
        if (!Number.isFinite(v) || (p.logY && v <= 0)) return;
        const Y = Math.round(p.yToPx(v)) + 0.5;
        ctx.strokeStyle = C.rule;
        ctx.beginPath();
        ctx.moveTo(a.x, Y);
        ctx.lineTo(a.x + a.w, Y);
        ctx.stroke();
        ctx.fillStyle = C.bronze;
        ctx.beginPath();
        ctx.arc(X, Y, 3.5, 0, Math.PI * 2);
        ctx.fill();
      });
    }
    ctx.restore();
  }

  resize(): void {}

  destroy(): void {
    this.detach();
    this.canvas.remove();
  }

  debug(): Record<string, unknown> {
    const b = this.bounds();
    return {
      kind: 'series', renderer: 'canvas2d', follow: this.follow, autoY: this.autoY, logY: this.plot.logY,
      window: { x0: this.plot.x.lo, x1: this.plot.x.hi, y0: this.plot.y.lo, y1: this.plot.y.hi },
      area: { ...this.plot.area }, steps: b, hover: this.hoverX, selection: this.selection, frames: this.frames, vertices: this.drawn,
    };
  }
}


