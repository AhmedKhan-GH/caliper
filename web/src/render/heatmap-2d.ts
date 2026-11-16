// The Canvas 2D heatmap and image renderer: a colormapped ImageData at the
// snapshot's size, drawn with nearest sampling through the plot transform.

import type { Plot2D } from '../views/plot.js';
import { C } from '../views/theme.js';
import { lutIndexer } from './colormap.js';

/** Colormaps a float32 array into RGBA bytes exactly as the native app does (lutIndexer). */
export function colormapInto(out: Uint8ClampedArray, data: Float32Array, lo: number, hi: number, lut: Uint8Array): void {
  const index = lutIndexer(lo, hi);
  for (let i = 0, n = data.length; i < n; i++) {
    const o = index(data[i]) * 4, p = i * 4;
    out[p] = lut[o];
    out[p + 1] = lut[o + 1];
    out[p + 2] = lut[o + 2];
    out[p + 3] = 255;
  }
}

export class Heatmap2D {
  readonly canvas = document.createElement('canvas');
  private readonly ctx: CanvasRenderingContext2D;
  private readonly img = document.createElement('canvas');
  private cols = 0;
  private rows = 0;

  constructor() {
    const ctx = this.canvas.getContext('2d', { alpha: false });
    if (!ctx) throw new Error('no 2d context');
    this.ctx = ctx;
  }

  upload(data: Float32Array | Uint8Array, rows: number, cols: number, image: boolean, lo: number, hi: number, lut: Uint8Array): void {
    if (this.img.width !== cols || this.img.height !== rows) {
      this.img.width = cols;
      this.img.height = rows;
    }
    this.cols = cols;
    this.rows = rows;
    const ictx = this.img.getContext('2d');
    if (!ictx || cols === 0 || rows === 0) return;
    const id = ictx.createImageData(cols, rows);
    if (image) id.data.set(data as Uint8Array);
    else colormapInto(id.data, data as Float32Array, lo, hi, lut);
    if (image) for (let i = 3; i < id.data.length; i += 4) id.data[i] = 255;
    ictx.putImageData(id, 0, 0);
  }

  draw(plot: Plot2D, dpr: number): void {
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = C.plate;
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.cols || !this.rows) return;
    const a = plot.area;
    ctx.save();
    ctx.beginPath();
    ctx.rect(a.x * dpr, a.y * dpr, a.w * dpr, a.h * dpr);
    ctx.clip();
    const sx = (a.w * dpr) / (plot.x.hi - plot.x.lo);
    const sy = (a.h * dpr) / (plot.y.hi - plot.y.lo);
    ctx.setTransform(sx, 0, 0, sy, a.x * dpr - plot.x.lo * sx, a.y * dpr - plot.y.lo * sy);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.img, 0, 0);
    ctx.restore();
  }

  destroy(): void {
    this.img.width = this.img.height = 0;
  }
}
