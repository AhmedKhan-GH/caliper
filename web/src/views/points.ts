// The points live view: an (N, 2) or (N, 3) float32 cloud coloured by class
// label (the fixed 16-colour palette) or by value (a colormap), with pan and
// zoom in 2-D, an orbit camera in 3-D, hover of the nearest point and a lasso
// or rectangle selection.

import { colormapLut, lutIndex, paletteLut } from '../render/colormap.js';
import { PointsGpu } from '../render/points-gpu.js';
import { COLORMAPS, LIMITS, snapshotRange, type SnapshotMsg } from '../protocol/messages.js';
import type { LiveView, ViewHost } from './base.js';
import { Orbit, ortho, type Mat4 } from './camera.js';
import { attachGestures } from './interact.js';
import { drawAxes, fitCanvas, Plot2D } from './plot.js';
import { C, FONT_TEXT, fmt } from './theme.js';

const LITTLE = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
const PLATE = [0x27, 0x26, 0x22];
const FLAT = [0xf0, 0xe9, 0xdb];
const PICK_RADIUS = 10;

function pack(r: number, g: number, b: number): number {
  return LITTLE ? ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0 : ((r << 24) | (g << 16) | (b << 8) | 255) >>> 0;
}

function unpack(c: number): [number, number, number] {
  return LITTLE ? [c & 255, (c >>> 8) & 255, (c >>> 16) & 255] : [c >>> 24, (c >>> 16) & 255, (c >>> 8) & 255];
}

/** The Canvas 2D fallback: round sprites per colour, far points first in 3-D. */
class Points2D {
  readonly canvas = document.createElement('canvas');
  private readonly ctx: CanvasRenderingContext2D;
  private sprites = new Map<number, HTMLCanvasElement>();
  private spriteSize = 0;

  constructor() {
    const ctx = this.canvas.getContext('2d', { alpha: false });
    if (!ctx) throw new Error('no 2d context');
    this.ctx = ctx;
  }

  private sprite(color: number, d: number): HTMLCanvasElement {
    if (d !== this.spriteSize) {
      this.sprites.clear();
      this.spriteSize = d;
    }
    let s = this.sprites.get(color);
    if (!s) {
      s = document.createElement('canvas');
      s.width = s.height = d;
      const c = s.getContext('2d')!;
      const [r, g, b] = unpack(color);
      c.fillStyle = `rgb(${r},${g},${b})`;
      c.beginPath();
      c.arc(d / 2, d / 2, d / 2, 0, Math.PI * 2);
      c.fill();
      this.sprites.set(color, s);
    }
    return s;
  }

  /** xy: device-pixel positions (NaN = hidden); order: draw order. */
  draw(xy: Float32Array, colors: Uint32Array, order: Uint32Array | null, n: number, radius: number, clip: { x: number; y: number; w: number; h: number }): void {
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = C.plate;
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.save();
    ctx.beginPath();
    ctx.rect(clip.x, clip.y, clip.w, clip.h);
    ctx.clip();
    const d = Math.max(1, Math.round(radius * 2));
    const h = d / 2;
    for (let k = 0; k < n; k++) {
      const i = order ? order[k] : k;
      const x = xy[2 * i], y = xy[2 * i + 1];
      if (x !== x) continue;
      ctx.drawImage(this.sprite(colors[i], d), x - h, y - h);
    }
    ctx.restore();
  }

  destroy(): void {
    this.sprites.clear();
  }
}

export class PointsView implements LiveView {
  private readonly plot = new Plot2D();
  private readonly orbit = new Orbit();
  private readonly r: PointsGpu | Points2D;
  private readonly octx: CanvasRenderingContext2D;
  private readonly detach: () => void;
  private snap: SnapshotMsg | null = null;
  private n = 0;
  private dims: 2 | 3 = 2;
  /** Positions as (n, 3), z = 0 in 2-D. */
  private pos = new Float32Array(0);
  private base = new Uint32Array(0);
  private colors = new Uint32Array(0);
  private bbox = [0, 1, 0, 1, 0, 0];
  private fitted = false;
  private classes = 0;
  /** Projected CSS-pixel positions and depths, valid while projKey matches. */
  private proj = new Float32Array(0);
  private depth = new Float32Array(0);
  private projKey = '';
  private hover: number | null = null;
  private hoverPx: { x: number; y: number } | null = null;
  private lasso: { x: number; y: number }[] | null = null;
  private rect = false;
  private selected: Uint8Array | null = null;
  private selCount = 0;
  private frames = 0;
  private lastDrawMs = 0;

  constructor(private readonly host: ViewHost) {
    this.r = host.device ? new PointsGpu(host.device) : new Points2D();
    this.r.canvas.className = 'data';
    host.body.prepend(this.r.canvas);
    const octx = host.overlay.getContext('2d');
    if (!octx) throw new Error('no 2d context');
    this.octx = octx;
    this.detach = attachGestures(host.overlay, {
      hover: (x, y) => this.onHover(x, y),
      leave: () => this.onHover(NaN, NaN),
      drag: (dx, dy, _x, _y, m) => {
        if (this.dims === 3) {
          if (m.button === 2 || m.button === 1 || m.alt || m.ctrl) this.orbit.pan(dx, dy, this.plot.area.h);
          else this.orbit.rotate(dx, dy);
        } else this.plot.pan(dx, dy);
        this.host.request('data');
      },
      wheel: (x, y, k) => {
        if (this.dims === 3) this.orbit.zoom(k);
        else {
          const reg = this.plot.region(x, y);
          if (reg === 'none') return;
          this.plot.zoom(x, y, k, reg === 'plot' ? 'xy' : reg);
        }
        this.host.request('data');
      },
      dblclick: () => {
        this.fit();
        this.host.request('data');
      },
      select: (phase, x, y, m) => this.onSelect(phase, x, y, m.alt),
      click: (x, y) => {
        if (this.lasso) return;
        const i = this.nearest(x, y);
        if (i !== null) this.host.report('pick', this.pointDetail(i));
      },
    });
  }

  // --- data -----------------------------------------------------------------

  snapshot(s: SnapshotMsg): void {
    const n = s.shape[0] ?? 0;
    const dims = (s.points_dims ?? s.shape[1] ?? 2) as 2 | 3;
    if (dims !== this.dims) this.fitted = false;
    if (n !== this.n) {
      this.selected = null;
      this.selCount = 0;
    }
    this.snap = s;
    this.n = n;
    this.dims = dims;
    this.classes = s.label_count ?? 0;
    if (this.pos.length < n * 3) this.pos = new Float32Array(n * 3);
    const src = s.data as Float32Array;
    const bb = [Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < 3; d++) {
        const v = d < dims ? src[i * dims + d] : 0;
        this.pos[i * 3 + d] = v;
        if (Number.isFinite(v)) {
          if (v < bb[2 * d]) bb[2 * d] = v;
          if (v > bb[2 * d + 1]) bb[2 * d + 1] = v;
        }
      }
    }
    if (bb.every(Number.isFinite)) this.bbox = bb;
    this.colorize(s);
    this.projKey = '';
    if (!this.fitted) this.fit();
    if (this.r instanceof PointsGpu) this.r.upload(this.pos, this.colors, n);
    const label = this.classes ? `${this.classes} classes` : s.aux ? 'values' : '';
    this.host.info(`${n} points  ${dims}-D${label ? `  ${label}` : ''}`);
    if (this.hoverPx) this.onHover(this.hoverPx.x, this.hoverPx.y);
  }

  private colorize(s: SnapshotMsg): void {
    const n = this.n;
    if (this.base.length < n) {
      this.base = new Uint32Array(n);
      this.colors = new Uint32Array(n);
    }
    const aux = s.aux;
    if (aux && this.classes > 0) {
      // Label L draws with palette entry clamp(L, 0, 255) mod 16, as natively.
      const lut = paletteLut();
      for (let i = 0; i < n; i++) {
        const o = lutIndex(aux[i], 0, 255) * 4;
        this.base[i] = pack(lut[o], lut[o + 1], lut[o + 2]);
      }
    } else if (aux) {
      const lut = colormapLut(COLORMAPS[s.colormap] ?? 'viridis');
      let [lo, hi] = snapshotRange(s);
      if (!Number.isFinite(lo) || !Number.isFinite(hi)) [lo, hi] = [0, 1];
      if (lo === hi) [lo, hi] = [lo - 0.5, hi + 0.5];
      for (let i = 0; i < n; i++) {
        const o = lutIndex(aux[i], lo, hi) * 4;
        this.base[i] = pack(lut[o], lut[o + 1], lut[o + 2]);
      }
    } else {
      this.base.fill(pack(FLAT[0], FLAT[1], FLAT[2]), 0, n);
    }
    this.applySelection();
  }

  /** Dims the points outside the selection. */
  private applySelection(): void {
    const n = this.n, sel = this.selected;
    for (let i = 0; i < n; i++) {
      if (!sel || sel[i]) this.colors[i] = this.base[i];
      else {
        const [r, g, b] = unpack(this.base[i]);
        this.colors[i] = pack(
          Math.round(r * 0.28 + PLATE[0] * 0.72), Math.round(g * 0.28 + PLATE[1] * 0.72), Math.round(b * 0.28 + PLATE[2] * 0.72));
      }
    }
  }

  private fit(): void {
    const b = this.bbox;
    if (this.dims === 3) this.orbit.fit(b);
    const px = (b[1] - b[0]) * 0.05 || 0.5, py = (b[3] - b[2]) * 0.05 || 0.5;
    this.plot.x = { lo: b[0] - px, hi: b[1] + px };
    this.plot.y = { lo: b[2] - py, hi: b[3] + py };
    this.fitted = true;
    this.pendingAspect = true;
  }

  private pendingAspect = false;

  private layout(): void {
    const { w, h } = this.host.size();
    this.plot.area = this.dims === 3 ? { x: 0, y: 0, w: Math.max(1, w), h: Math.max(1, h) } : { x: 50, y: 14, w: Math.max(10, w - 50 - 12), h: Math.max(10, h - 14 - 30) };
    if (this.pendingAspect && this.dims === 2) this.plot.equalAspect();
    this.pendingAspect = false;
  }

  resize(): void {
    this.pendingAspect = this.dims === 2;
  }

  private radius(): number {
    const n = this.n;
    return n > 50000 ? 1.5 : n > 5000 ? 2 : n > 500 ? 2.75 : 3.5;
  }

  private matrix(): Mat4 {
    const a = this.plot.area;
    return this.dims === 3 ? this.orbit.matrix(a.w / a.h) : ortho(this.plot.x.lo, this.plot.x.hi, this.plot.y.lo, this.plot.y.hi);
  }

  /** Projects every point to CSS pixels (NaN when behind the camera). */
  private project(): void {
    const a = this.plot.area;
    const m = this.matrix();
    const key = `${Array.from(m).join(',')}|${a.x},${a.y},${a.w},${a.h}|${this.snap?.seq}`;
    if (key === this.projKey) return;
    this.projKey = key;
    const n = this.n;
    if (this.proj.length < 2 * n) {
      this.proj = new Float32Array(2 * n);
      this.depth = new Float32Array(n);
    }
    const p = this.pos;
    for (let i = 0; i < n; i++) {
      const x = p[3 * i], y = p[3 * i + 1], z = p[3 * i + 2];
      const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
      const cz = m[2] * x + m[6] * y + m[10] * z + m[14];
      if (!(cw > 0) || cz < 0 || cz > cw) {
        this.proj[2 * i] = NaN;
        this.proj[2 * i + 1] = NaN;
        continue;
      }
      const cx = (m[0] * x + m[4] * y + m[8] * z + m[12]) / cw;
      const cy = (m[1] * x + m[5] * y + m[9] * z + m[13]) / cw;
      this.proj[2 * i] = a.x + ((cx + 1) / 2) * a.w;
      this.proj[2 * i + 1] = a.y + ((1 - cy) / 2) * a.h;
      this.depth[i] = cz / cw;
    }
  }

  // --- interaction ------------------------------------------------------------

  private nearest(x: number, y: number): number | null {
    if (!this.n || Number.isNaN(x) || !this.plot.inside(x, y)) return null;
    this.project();
    let best = -1, bd = PICK_RADIUS * PICK_RADIUS, bz = Infinity;
    for (let i = 0; i < this.n; i++) {
      const dx = this.proj[2 * i] - x, dy = this.proj[2 * i + 1] - y;
      const d = dx * dx + dy * dy;
      if (d < bd || (d === bd && this.depth[i] < bz)) {
        best = i;
        bd = d;
        bz = this.depth[i];
      }
    }
    return best >= 0 ? best : null;
  }

  /** The source index of sample point i along the point dim (the first of its positions). */
  private rowId(i: number): number {
    const ax = this.snap?.axes[this.snap.axes.length - 1];
    if (!ax || !ax.size) return i;
    return ax.start + Math.floor((i * ax.count) / ax.size) * ax.step;
  }

  /** A pick or hover: the index in the sample and the source index. */
  private pointDetail(i: number): Record<string, unknown> {
    return { index: i, row_id: this.rowId(i) };
  }

  private onHover(x: number, y: number): void {
    this.hoverPx = Number.isNaN(x) ? null : { x, y };
    if (this.lasso) return;
    const i = this.nearest(x, y);
    const changed = i !== this.hover;
    this.hover = i;
    if (i === null) this.host.tooltip(null);
    else {
      const lines = [`point ${i}`];
      const row = this.rowId(i);
      if (row !== i) lines[0] += `  row ${row}`;
      const aux = this.snap?.aux;
      if (aux) lines.push(this.classes ? `label ${aux[i]}` : `value ${fmt(aux[i], 5)}`);
      lines.push(Array.from(this.pos.subarray(3 * i, 3 * i + this.dims)).map((v, k) => `${'xyz'[k]} ${fmt(v)}`).join('  '));
      this.host.tooltip(lines.join('\n'), x, y);
      if (changed) this.host.report('hover', this.pointDetail(i));
    }
    if (changed) this.host.request('overlay');
  }

  private onSelect(phase: 'start' | 'move' | 'end', x: number, y: number, rect: boolean): void {
    if (phase === 'start') {
      this.lasso = [{ x, y }];
      this.rect = rect;
      this.hover = null;
      this.host.tooltip(null);
    } else if (this.lasso && phase === 'move') {
      if (this.rect) this.lasso[1] = { x, y };
      else this.lasso.push({ x, y });
    } else if (this.lasso && phase === 'end') {
      const poly = this.lasso;
      const rect = this.rect;
      this.lasso = null;
      if (Number.isNaN(x) || poly.length < 2) {
        // A shift-click clears the selection, which an empty detail reports.
        if (this.selected) this.host.report('select', {});
        this.selected = null;
        this.selCount = 0;
      } else {
        if (rect) {
          const [a, b] = [poly[0], poly[1] ?? poly[0]];
          poly.splice(0, poly.length, { x: a.x, y: a.y }, { x: b.x, y: a.y }, { x: b.x, y: b.y }, { x: a.x, y: b.y });
        }
        this.selectPolygon(poly);
      }
      this.applySelection();
      if (this.r instanceof PointsGpu) this.r.recolor(this.colors);
      this.host.request('data');
    }
    this.host.request('overlay');
  }

  private selectPolygon(poly: { x: number; y: number }[]): void {
    this.project();
    const n = this.n;
    const sel = new Uint8Array(n);
    const idx: number[] = [];
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const p of poly) {
      xmin = Math.min(xmin, p.x); xmax = Math.max(xmax, p.x);
      ymin = Math.min(ymin, p.y); ymax = Math.max(ymax, p.y);
    }
    for (let i = 0; i < n; i++) {
      const px = this.proj[2 * i], py = this.proj[2 * i + 1];
      if (!(px >= xmin && px <= xmax && py >= ymin && py <= ymax)) continue;
      let inside = false;
      for (let a = 0, b = poly.length - 1; a < poly.length; b = a++) {
        const pa = poly[a], pb = poly[b];
        if ((pa.y > py) !== (pb.y > py) && px < ((pb.x - pa.x) * (py - pa.y)) / (pb.y - pa.y) + pa.x) inside = !inside;
      }
      if (inside) {
        sel[i] = 1;
        idx.push(i);
      }
    }
    this.selected = idx.length ? sel : null;
    this.selCount = idx.length;
    const kept = idx.slice(0, LIMITS.maxEventIndices);
    const detail: Record<string, unknown> = { indices: kept, row_ids: kept.map((i) => this.rowId(i)) };
    if (idx.length > LIMITS.maxEventIndices) detail.truncated = true;
    this.host.report('select', detail);
  }

  // --- drawing ----------------------------------------------------------------

  draw(data: boolean): void {
    const { w, h, dpr } = this.host.size();
    this.layout();
    const resized = fitCanvas(this.r.canvas, w, h, dpr);
    if (data || resized) {
      const t0 = performance.now();
      const a = this.plot.area;
      const vp = { x: a.x * dpr, y: a.y * dpr, w: a.w * dpr, h: a.h * dpr };
      if (this.r instanceof PointsGpu) {
        this.r.draw(this.matrix(), vp, this.radius() * dpr, this.dims === 3);
      } else {
        this.project();
        const n = this.n;
        const xy = new Float32Array(2 * n);
        for (let i = 0; i < 2 * n; i++) xy[i] = this.proj[i] * dpr;
        let order: Uint32Array | null = null;
        if (this.dims === 3) {
          order = new Uint32Array(n);
          for (let i = 0; i < n; i++) order[i] = i;
          const z = this.depth;
          order.sort((p, q) => z[q] - z[p]);
        }
        this.r.draw(xy, this.colors, order, n, this.radius() * dpr, vp);
      }
      this.lastDrawMs = performance.now() - t0;
      this.frames++;
    }
    this.drawOverlay(w, h, dpr);
  }

  private drawOverlay(w: number, h: number, dpr: number): void {
    const ctx = this.octx;
    fitCanvas(this.host.overlay, w, h, dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!this.snap) return;
    if (this.dims === 2) drawAxes(ctx, this.plot, {});
    else this.drawTriad(ctx, 30, h - 30);
    if (this.hover !== null) {
      this.project();
      const x = this.proj[2 * this.hover], y = this.proj[2 * this.hover + 1];
      if (Number.isFinite(x)) {
        ctx.strokeStyle = C.parchment;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, this.radius() + 3, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    if (this.lasso && this.lasso.length > 1) {
      const pts = this.lasso;
      ctx.strokeStyle = C.bronze;
      ctx.fillStyle = 'rgba(179,139,89,0.12)';
      ctx.lineWidth = 1.25;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      if (this.rect) {
        const a = pts[0], b = pts[1];
        ctx.rect(a.x, a.y, b.x - a.x, b.y - a.y);
      } else {
        ctx.moveTo(pts[0].x, pts[0].y);
        for (const p of pts) ctx.lineTo(p.x, p.y);
        ctx.closePath();
      }
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (this.selCount) {
      ctx.font = `11px ${FONT_TEXT}`;
      ctx.fillStyle = C.bronze;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillText(`${this.selCount} selected`, w - 10, this.dims === 3 ? 8 : this.plot.area.y + 4);
    }
  }

  private drawTriad(ctx: CanvasRenderingContext2D, cx: number, cy: number): void {
    const ax = this.orbit.axes();
    ctx.font = `10px ${FONT_TEXT}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ax.forEach((v, k) => {
      const x = cx + v[0] * 16, y = cy - v[1] * 16;
      ctx.strokeStyle = C.rule;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.fillStyle = C.faint;
      ctx.fillText('xyz'[k], cx + v[0] * 23, cy - v[1] * 23);
    });
  }

  destroy(): void {
    this.detach();
    this.r.destroy();
    this.r.canvas.remove();
  }

  debug(): Record<string, unknown> {
    return {
      kind: 'points', renderer: this.r instanceof PointsGpu ? 'webgpu' : 'canvas2d', n: this.n, dims: this.dims, seq: this.snap?.seq ?? null,
      window: { x0: this.plot.x.lo, x1: this.plot.x.hi, y0: this.plot.y.lo, y1: this.plot.y.hi },
      camera: { az: this.orbit.az, el: this.orbit.el, dist: this.orbit.dist, target: [...this.orbit.target] },
      area: { ...this.plot.area }, hover: this.hover, selected: this.selCount, frames: this.frames, lastDrawMs: this.lastDrawMs,
    };
  }
}
