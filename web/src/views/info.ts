// The line a view's header shows about it: the element's own views make it
// from their snapshots, and the element makes the same line for a view the
// WebAssembly views draw from the facts their module reports (wasm.ts).

import { fmt } from './theme.js';

export const heatmapInfo = (rows: number, cols: number, lo: number, hi: number): string =>
  `${rows}×${cols}  ${fmt(lo)} … ${fmt(hi)}`;

export const imageInfo = (rows: number, cols: number): string => `${rows}×${cols} RGBA`;

export function pointsInfo(n: number, dims: number, classes: number, values: boolean): string {
  const label = classes ? `${classes} classes` : values ? 'values' : '';
  return `${n} points  ${dims}-D${label ? `  ${label}` : ''}`;
}

export const seriesInfo = (end: number, last: number): string => `${end} steps  last ${fmt(last)}`;

export const histogramInfo = (bins: number, lo: number, hi: number, total: number): string =>
  `${bins} bins  ${fmt(lo)} … ${fmt(hi)}  n ${fmt(total)}`;

/** What the WebAssembly views report about a view; a non-finite number is null. */
export interface ViewFacts {
  kind: string;
  rows?: number; cols?: number; lo?: number | null; hi?: number | null;
  n?: number; dims?: number; classes?: number; values?: boolean;
  end?: number; last?: number | null;
  bins?: number; total?: number | null;
  entities?: number; items?: number; copied?: number; gpu_ms?: number | null; notice?: string;
}

/** A scene's line, as the native scene view's header writes it (src/host/live/live_scene.cpp). */
export function sceneInfo(entities: number, items: number, copied: number, gpuMs: number | null): string {
  const n = (v: number) => v.toLocaleString('en-US');
  const bytes = (b: number) => (b < 1024 ? `${b} B` : b < 1024 * 1024 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(1)} MB`);
  let s = `${n(entities)} ${entities === 1 ? 'entity' : 'entities'}  ${n(items)} items`;
  if (copied > 0) s += `  copied ${bytes(copied)}`;
  if (typeof gpuMs === 'number' && gpuMs >= 0) s += `  gpu ${gpuMs.toFixed(2)} ms`;
  return s;
}

/** The header line for a view's facts, or null for facts of an unknown kind. */
export function factsInfo(f: ViewFacts): string | null {
  const num = (v: number | null | undefined) => (typeof v === 'number' ? v : NaN);
  switch (f.kind) {
    case 'heatmap': return heatmapInfo(num(f.rows), num(f.cols), num(f.lo), num(f.hi));
    case 'image': return imageInfo(num(f.rows), num(f.cols));
    case 'points': return pointsInfo(num(f.n), num(f.dims), f.classes ?? 0, !!f.values);
    case 'series': return seriesInfo(num(f.end), num(f.last));
    case 'histogram': return histogramInfo(num(f.bins), num(f.lo), num(f.hi), num(f.total));
    case 'scene': return f.notice ?? sceneInfo(num(f.entities), num(f.items), num(f.copied), f.gpu_ms ?? null);
    default: return null;
  }
}
