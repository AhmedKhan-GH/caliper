// The heatmap and image live views: a float32 (H, W) array through a colormap
// over [vmin, vmax], or an (H, W, 4) RGBA image, with pan, zoom, axes of row
// and column indices, hover and a rectangle selection.

import { colormapLut, lutCss } from '../render/colormap.js';
import { Heatmap2D } from '../render/heatmap-2d.js';
import { HeatmapGpu } from '../render/heatmap-gpu.js';
import { COLORMAPS, snapshotRange, type PayloadAxis, type SnapshotMsg } from '../protocol/messages.js';
import type { LiveView, ViewHost } from './base.js';
import { attachGestures } from './interact.js';
import { drawAxes, fitCanvas, Plot2D } from './plot.js';
import { C, FONT_TEXT, fmt } from './theme.js';

interface Cell { r: number; c: number }

export class HeatmapView implements LiveView {
  private readonly plot = new Plot2D();
  private readonly r: HeatmapGpu | Heatmap2D;
  private readonly octx: CanvasRenderingContext2D;
  private readonly detach: () => void;
  private snap: SnapshotMsg | null = null;
  private rows = 0;
  private cols = 0;
  private lo = 0;
  private hi = 1;
  private lut = colormapLut('viridis');
  private fitted = false;
  private hover: Cell | null = null;
  private hoverPx: { x: number; y: number } | null = null;
  private dragSel: { x0: number; y0: number; x1: number; y1: number } | null = null;
  private selection: { r0: number; r1: number; c0: number; c1: number } | null = null;
  private frames = 0;

  constructor(private readonly host: ViewHost, private readonly image: boolean) {
    this.r = host.device ? new HeatmapGpu(host.device) : new Heatmap2D();
    this.r.canvas.className = 'data';
    host.body.prepend(this.r.canvas);
    const octx = host.overlay.getContext('2d');
    if (!octx) throw new Error('no 2d context');
    this.octx = octx;
    this.plot.yDown = true;
    this.detach = attachGestures(host.overlay, {
      hover: (x, y) => this.onHover(x, y),
      leave: () => this.onHover(NaN, NaN),
      drag: (dx, dy) => {
        this.plot.pan(dx, dy);
        this.host.request('data');
      },
      wheel: (x, y, k) => {
        const reg = this.plot.region(x, y);
        if (reg === 'none') return;
        this.plot.zoom(x, y, k, reg === 'plot' ? 'xy' : reg);
        this.host.request('data');
      },
      dblclick: () => {
        this.fit();
        this.host.request('data');
      },
      select: (phase, x, y) => this.onSelect(phase, x, y),
      click: (x, y) => {
        const cell = this.cellAt(x, y);
        if (cell && !this.dragSel) this.host.report('pick', this.cellDetail(cell));
      },
    });
  }

  // --- data -----------------------------------------------------------------

  snapshot(s: SnapshotMsg): void {
    const rows = s.shape[0] ?? 0, cols = s.shape[1] ?? 0;
    if (rows !== this.rows || cols !== this.cols) {
      this.fitted = false;
      this.selection = null;
    }
    this.snap = s;
    this.rows = rows;
    this.cols = cols;
    if (!this.image) {
      const [lo, hi] = snapshotRange(s);
      // A non-finite range keeps the last finite one; a flat one widens by 0.5.
      if (Number.isFinite(lo) && Number.isFinite(hi)) {
        this.lo = lo;
        this.hi = hi;
        if (lo === hi) {
          this.lo = lo - 0.5;
          this.hi = hi + 0.5;
        }
      }
      this.lut = colormapLut(COLORMAPS[s.colormap] ?? 'viridis');
    }
    this.r.upload(s.data, rows, cols, this.image, this.lo, this.hi, this.lut);
    if (this.hover && (this.hover.r >= rows || this.hover.c >= cols)) this.hover = null;
    const shape = this.image ? `${rows}×${cols} RGBA` : `${rows}×${cols}  ${fmt(this.lo)} … ${fmt(this.hi)}`;
    this.host.info(shape);
    if (this.hoverPx) this.onHover(this.hoverPx.x, this.hoverPx.y);
  }

  private layout(): void {
    const { w, h } = this.host.size();
    const right = this.image ? 10 : 26;
    this.plot.area = { x: 46, y: 16, w: Math.max(10, w - 46 - right), h: Math.max(10, h - 16 - 32) };
    if (!this.fitted && this.rows && this.cols) this.fit();
  }

  private fit(): void {
    this.plot.x = { lo: 0, hi: Math.max(1, this.cols) };
    this.plot.y = { lo: 0, hi: Math.max(1, this.rows) };
    if (this.image) this.plot.equalAspect();
    this.fitted = this.rows > 0;
  }

  resize(): void {
    if (this.image && this.fitted) this.fit();
  }

  // --- source indices -----------------------------------------------------------

  /** Row and column axes of the payload, and the tile axis of a mosaic. */
  private axes(): { row?: PayloadAxis; col?: PayloadAxis; tile?: PayloadAxis } {
    const a = this.snap?.axes ?? [];
    if (this.snap?.tile && a.length === 3) return { tile: a[0], row: a[1], col: a[2] };
    return { row: a[0], col: a[1] };
  }

  /** The source index a payload index stands for (the first position of a pooled cell). */
  private static source(ax: PayloadAxis | undefined, i: number): number {
    if (!ax || !ax.size) return i;
    return ax.start + Math.floor((i * ax.count) / ax.size) * ax.step;
  }

  /** The half-open source range that payload indices [i0, i1) stand for. */
  private static range(ax: PayloadAxis | undefined, i0: number, i1: number): [number, number] {
    if (!ax || !ax.size) return [i0, i1];
    const p0 = Math.floor((i0 * ax.count) / ax.size), p1 = Math.ceil((i1 * ax.count) / ax.size);
    return [ax.start + p0 * ax.step, ax.start + (p1 - 1) * ax.step + 1];
  }

  /** Where a payload cell is in the source: the tile, and the row and column inside it. */
  private locate(r: number, c: number): { tile: number | null; r: number; c: number } {
    const { row, col, tile } = this.axes();
    const t = this.snap?.tile;
    if (!t || !tile || !row || !col) return { tile: null, r, c };
    return { tile: Math.floor(r / row.size) * t.cols + Math.floor(c / col.size), r: r % row.size, c: c % col.size };
  }

  /** A pick or hover: source indices of the cell's first position, and its value. */
  private cellDetail(cell: Cell): Record<string, unknown> {
    const s = this.snap!;
    const { row, col, tile } = this.axes();
    const at = this.locate(cell.r, cell.c);
    const d: Record<string, unknown> = { row: HeatmapView.source(row, at.r), col: HeatmapView.source(col, at.c) };
    if (at.tile !== null) d.tile = HeatmapView.source(tile, at.tile);
    if (!this.image) {
      const v = (s.data as Float32Array)[cell.r * this.cols + cell.c];
      if (Number.isFinite(v)) d.value = v;
    }
    return d;
  }

  // --- interaction ------------------------------------------------------------

  private cellAt(x: number, y: number): Cell | null {
    if (!this.snap || !this.plot.inside(x, y)) return null;
    const c = Math.floor(this.plot.pxToX(x)), r = Math.floor(this.plot.pxToY(y));
    if (r < 0 || c < 0 || r >= this.rows || c >= this.cols) return null;
    return { r, c };
  }

  private onHover(x: number, y: number): void {
    this.hoverPx = Number.isNaN(x) ? null : { x, y };
    const cell = Number.isNaN(x) ? null : this.cellAt(x, y);
    const changed = cell?.r !== this.hover?.r || cell?.c !== this.hover?.c;
    this.hover = cell;
    if (!cell || !this.snap) {
      this.host.tooltip(null);
    } else {
      const d = this.cellDetail(cell);
      const { row, col, tile } = this.axes();
      const parts: string[] = [];
      if (tile && d.tile !== undefined) parts.push(`${tile.dim} ${d.tile}`);
      parts.push(`${row?.dim ?? 'row'} ${d.row}`, `${col?.dim ?? 'col'} ${d.col}`);
      const o = cell.r * this.cols + cell.c;
      const val = this.image ? `rgba ${Array.from(this.snap.data.subarray(o * 4, o * 4 + 4)).join(' ')}` : fmt((this.snap.data as Float32Array)[o], 5);
      this.host.tooltip(`${parts.join('  ')}\n${val}`, x, y);
      if (changed) this.host.report('hover', d);
    }
    if (changed) this.host.request('overlay');
  }

  private onSelect(phase: 'start' | 'move' | 'end', x: number, y: number): void {
    if (phase === 'start') {
      this.dragSel = { x0: this.plot.pxToX(x), y0: this.plot.pxToY(y), x1: this.plot.pxToX(x), y1: this.plot.pxToY(y) };
    } else if (this.dragSel && phase === 'move') {
      this.dragSel.x1 = this.plot.pxToX(x);
      this.dragSel.y1 = this.plot.pxToY(y);
    } else if (this.dragSel && phase === 'end') {
      const s = this.dragSel;
      this.dragSel = null;
      let r0 = 0, r1 = 0, c0 = 0, c1 = 0;
      if (!Number.isNaN(x)) {
        // In a mosaic the selection stays inside the tile where the drag began.
        const { row, col } = this.axes();
        const t = this.snap?.tile;
        let rlo = 0, rhi = this.rows, clo = 0, chi = this.cols;
        if (t && row && col) {
          const tr = Math.floor(Math.max(0, Math.min(this.rows - 1, s.y0)) / row.size);
          const tc = Math.floor(Math.max(0, Math.min(this.cols - 1, s.x0)) / col.size);
          rlo = tr * row.size; rhi = rlo + row.size; clo = tc * col.size; chi = clo + col.size;
        }
        const cr = (v: number) => Math.max(rlo, Math.min(rhi, v));
        const cc = (v: number) => Math.max(clo, Math.min(chi, v));
        c0 = cc(Math.floor(Math.min(s.x0, s.x1))); c1 = cc(Math.ceil(Math.max(s.x0, s.x1)));
        r0 = cr(Math.floor(Math.min(s.y0, s.y1))); r1 = cr(Math.ceil(Math.max(s.y0, s.y1)));
      }
      if (r1 > r0 && c1 > c0) {
        this.selection = { r0, r1, c0, c1 };
        const { row, col, tile } = this.axes();
        const a = this.locate(r0, c0);
        const d: Record<string, unknown> = {
          rows: HeatmapView.range(row, a.r, a.r + (r1 - r0)), cols: HeatmapView.range(col, a.c, a.c + (c1 - c0)),
        };
        if (a.tile !== null) d.tile = HeatmapView.source(tile, a.tile);
        this.host.report('select', d);
      } else if (this.selection) {
        // A shift-click clears the selection, which an empty detail reports.
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
    const resized = fitCanvas(this.r.canvas, w, h, dpr);
    if (data || resized) {
      this.r.draw(this.plot, dpr);
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
    const p = this.plot, a = p.area;
    const { row, col, tile } = this.axes();
    const t = this.snap.tile;
    const label = (ax: PayloadAxis | undefined, mod: number) => (v: number) => String(HeatmapView.source(ax, t ? v % mod : v));
    drawAxes(ctx, p, {
      xInteger: true, yInteger: true, cells: true,
      xTitle: col?.dim, yTitle: row?.dim,
      xLabel: label(col, col?.size ?? 1), yLabel: label(row, row?.size ?? 1),
    });
    ctx.save();
    ctx.beginPath();
    ctx.rect(a.x, a.y, a.w, a.h);
    ctx.clip();
    if (t && tile && row && col) {
      ctx.strokeStyle = C.ground;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 1; i < t.cols; i++) {
        const x = p.xToPx(i * col.size);
        ctx.moveTo(x, a.y);
        ctx.lineTo(x, a.y + a.h);
      }
      for (let i = 1; i < t.rows; i++) {
        const y = p.yToPx(i * row.size);
        ctx.moveTo(a.x, y);
        ctx.lineTo(a.x + a.w, y);
      }
      ctx.stroke();
      const tw = Math.abs(p.xToPx(col.size) - p.xToPx(0));
      if (tw > 44) {
        ctx.font = `10px ${FONT_TEXT}`;
        ctx.textBaseline = 'top';
        for (let k = 0; k < tile.size; k++) {
          const x = p.xToPx((k % t.cols) * col.size) + 4, y = p.yToPx(Math.floor(k / t.cols) * row.size) + 3;
          const text = `${tile.dim} ${HeatmapView.source(tile, k)}`;
          ctx.fillStyle = 'rgba(24,23,21,0.7)';
          ctx.fillRect(x - 2, y - 1, ctx.measureText(text).width + 4, 13);
          ctx.fillStyle = C.secondary;
          ctx.fillText(text, x, y);
        }
      }
    }
    const box = (r0: number, r1: number, c0: number, c1: number, stroke: string, fill?: string) => {
      const x0 = p.xToPx(c0), x1 = p.xToPx(c1), y0 = p.yToPx(r0), y1 = p.yToPx(r1);
      if (fill) {
        ctx.fillStyle = fill;
        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      }
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    };
    if (this.selection) {
      const s = this.selection;
      box(s.r0, s.r1, s.c0, s.c1, C.bronze, 'rgba(179,139,89,0.16)');
    }
    if (this.dragSel) {
      const s = this.dragSel;
      ctx.setLineDash([4, 3]);
      box(s.y0, s.y1, s.x0, s.x1, C.bronze, 'rgba(179,139,89,0.12)');
      ctx.setLineDash([]);
    }
    if (this.hover) box(this.hover.r, this.hover.r + 1, this.hover.c, this.hover.c + 1, C.parchment);
    ctx.restore();
    if (!this.image) this.drawColorbar(ctx, a.x + a.w + 8, a.y, 8, a.h);
  }

  private drawColorbar(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
    const n = Math.max(2, Math.floor(h));
    for (let i = 0; i < n; i++) {
      const idx = Math.round(255 * (1 - i / (n - 1)));
      ctx.fillStyle = lutCss(this.lut, idx);
      ctx.fillRect(x, y + (i * h) / n, w, h / n + 0.5);
    }
    ctx.strokeStyle = C.rule;
    ctx.lineWidth = 1;
    ctx.strokeRect(x - 0.5, y - 0.5, w + 1, h + 1);
  }

  destroy(): void {
    this.detach();
    this.r.destroy();
    this.r.canvas.remove();
  }

  debug(): Record<string, unknown> {
    const a = this.plot.area;
    return {
      kind: this.image ? 'image' : 'heatmap',
      renderer: this.r instanceof HeatmapGpu ? 'webgpu' : 'canvas2d',
      rows: this.rows, cols: this.cols, seq: this.snap?.seq ?? null,
      window: { x0: this.plot.x.lo, x1: this.plot.x.hi, y0: this.plot.y.lo, y1: this.plot.y.hi },
      area: { ...a }, range: [this.lo, this.hi], hover: this.hover, selection: this.selection, frames: this.frames,
    };
  }
}
