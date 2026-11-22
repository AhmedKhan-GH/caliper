// A panel made ready to draw from the data the app sends (panel.data): the
// browser's part of the native runtime's prepare_panel()
// (src/host/spec_runtime/spec_prepare.cpp). The app has resolved the panel,
// run its transforms and bounded its data; here the names, labels, the
// header's info string, the colours of grids and points, the class-label rule,
// the table's formats and the states are made with the same rules.

import { colormapLut, lutIndex } from '../render/colormap.js';
import type { ColormapName } from '../protocol/messages.js';
import { cellText, columnFormat, fmt4, numericType, withCommas } from './format.js';
import { bindingName, panelTitle } from './layout.js';
import { classRgb } from './look.js';
import {
  decodeArray, type HeatmapData, type HistogramData, type ImageData, type PanelDataResult, type PointsData,
  type SeriesData, type Spec, type SpecPanel, type TableData, type TextData,
} from './types.js';

export interface PanelState { label: string; detail: string }

export interface PreparedSeries {
  lines: { name: string; x: Float64Array; y: Float64Array }[];
  xName: string;
}

export interface PreparedGrid {
  rows: number; cols: number; sourceRows: number; sourceCols: number; strideR: number; strideC: number;
  rgba: Uint8ClampedArray; values: Float32Array | null; vmin: number; vmax: number; colormap: string; channels: number;
}

export interface PreparedPoints {
  threeD: boolean; x: Float32Array; y: Float32Array; z: Float32Array | null;
  rgba: Uint8ClampedArray | null; scale: Float32Array | null;
  cmin: number; cmax: number; colormap: string; classes: number[]; colorName: string; total: number;
}

export interface PreparedHistogram { counts: number[]; lo: number; hi: number }

export interface PreparedTable { columns: string[]; numeric: boolean[]; cells: string[][] }

export interface PreparedText { text: string; value: string; hasValue: boolean }

export interface PreparedPanel {
  id: string;
  title: string;
  primitive: SpecPanel['primitive'];
  state: PanelState | null;
  info: string;
  xLabel: string; yLabel: string; zLabel: string;
  xLog: boolean; yLog: boolean;
  xRange: [number, number] | null; yRange: [number, number] | null;
  series?: PreparedSeries;
  grid?: PreparedGrid;
  points?: PreparedPoints;
  histogram?: PreparedHistogram;
  table?: PreparedTable;
  text?: PreparedText;
  bounds: string[];
}

/** The catalog id at the root of a source's chain of `from`, "sql" for a query, or the id itself. */
export function sourceOrigin(spec: Spec, sourceId: string): string {
  let id = sourceId;
  for (let hops = 0; hops < 64; hops++) {
    const src = (spec.source ?? []).find((s) => s.id === id);
    if (!src) return sourceId;
    if (src.catalog !== undefined) return src.catalog;
    if (src.sql !== undefined) return 'sql';
    if (src.from === undefined) return sourceId;
    id = src.from;
  }
  return sourceId;
}

/** The state of a resolver error code (panel_state_for_error). */
export function stateForError(code: string, detail: string): PanelState {
  const d = detail || code;
  if (code === 'resolve.unavailable') return { label: 'Source offline', detail: d };
  if (code === 'resolve.too_large') return { label: 'Data too large', detail: d };
  if (code === 'resolve.query_failed' || code === 'resolve.timeout') return { label: 'Query error', detail: d };
  if (code === 'resolve.unknown_column') return { label: 'Missing column', detail: d };
  if (code === 'resolve.unknown_source' || code === 'resolve.cycle') return { label: 'Missing source', detail: d };
  if (code === 'panel.not_found') return { label: 'Missing panel', detail: d };
  if (code === 'spec.no_open_spec') return { label: 'No spec', detail: d };
  if (code === 'resolve.not_tabular' || code === 'resolve.not_a_tensor') return { label: 'Mismatched data', detail: d };
  if (code.startsWith('transform.')) return { label: 'Transform error', detail: d };
  if (code.startsWith('loader.')) return { label: 'Missing data', detail: d };
  return { label: 'Data error', detail: d };
}

/**
 * Whether a colour vector holds class labels: an integer type, and only whole
 * values from 0 to 255 with at most 16 distinct ones (the spec format's rule,
 * as the native runtime applies it). Returns the labels in order, or null.
 */
export function classLabels(values: ArrayLike<number>, integer: boolean): number[] | null {
  if (!integer) return null;
  const seen = new Uint8Array(256);
  let distinct = 0;
  for (let i = 0; i < values.length; i++) {
    const x = values[i];
    if (!Number.isFinite(x)) continue;
    if (x < 0 || x > 255 || x !== Math.floor(x)) return null;
    if (!seen[x]) {
      seen[x] = 1;
      if (++distinct > 16) return null;
    }
  }
  if (!distinct) return null;
  const out: number[] = [];
  for (let i = 0; i < 256; i++) if (seen[i]) out.push(i);
  return out;
}

/** The colormap's ends: a flat range widens by 0.5 each way, no finite values give [0, 1]. */
function settle(lo: number, hi: number): [number, number] {
  if (!(hi >= lo)) return [0, 1];
  if (hi === lo) return [lo - 0.5, hi + 0.5];
  return [lo, hi];
}

const cmapName = (name: string | undefined, fallback: string): string =>
  name === 'viridis' || name === 'magma' || name === 'rdbu' || name === 'greys' ? name : fallback;

/** The 256-entry table of a colormap, greys included. */
export function lut(name: string): Uint8Array {
  if (name !== 'greys') return colormapLut(name as ColormapName);
  const t = new Uint8Array(1024);
  for (let i = 0; i < 256; i++) t.set([i, i, i, 255], i * 4);
  return t;
}

const f32 = Math.fround;

function mapValues(values: ArrayLike<number>, vmin: number, vmax: number, table: Uint8Array, out: Uint8ClampedArray): void {
  for (let i = 0; i < values.length; i++) {
    const k = lutIndex(values[i], vmin, vmax) * 4;
    out[i * 4] = table[k];
    out[i * 4 + 1] = table[k + 1];
    out[i * 4 + 2] = table[k + 2];
    out[i * 4 + 3] = table[k + 3];
  }
}

export function preparePanel(spec: Spec, panel: SpecPanel, result: PanelDataResult): PreparedPanel {
  const o = panel.options ?? {};
  const p: PreparedPanel = {
    id: panel.id, title: panelTitle(spec, panel), primitive: panel.primitive, state: null, info: '',
    xLabel: o.x_label ?? '', yLabel: o.y_label ?? '', zLabel: o.z_label ?? '',
    xLog: !!o.x_log, yLog: !!o.y_log, xRange: o.x_range ?? null, yRange: o.y_range ?? null, bounds: result.bounds ?? [],
  };
  const name = (binding: string) => bindingName(spec, binding);
  if (result.error) {
    p.state = stateForError(result.error.code, result.error.where ? sourceOrigin(spec, result.error.where) : result.error.code);
    return p;
  }
  if (result.state) {
    // A state names the source's catalog id rather than the spec's own id.
    const det = result.state.detail;
    const isSource = !det.includes('.') && (spec.source ?? []).some((s) => s.id === det);
    p.state = { label: result.state.kind === 'empty' ? 'No data' : 'Mismatched data', detail: isSource ? sourceOrigin(spec, det) : det };
    return p;
  }
  const data = result.data;
  if (!data) {
    p.state = { label: 'No data', detail: panel.id };
    return p;
  }
  switch (panel.primitive) {
    case 'series': {
      const d = data as SeriesData;
      const yb = panel.bind?.x;
      const xName = d.x_index ? 'step' : name(Array.isArray(yb) ? yb[0] : yb ?? 'step');
      const lines = d.lines.map((l) => ({
        name: name(l.binding) + (l.row === null ? '' : `[${l.row}]`),
        x: decodeArray(l.x) as Float64Array, y: decodeArray(l.y) as Float64Array,
      }));
      p.series = { lines, xName };
      if (!p.xLabel) p.xLabel = xName;
      if (!p.yLabel && lines.length === 1) p.yLabel = lines[0].name;
      const last = d.lines.length === 1 ? d.lines[0].last : null;
      p.info = last !== null ? `${withCommas(d.steps)} steps  last ${fmt4(last)}` : `${withCommas(d.steps)} steps  ${lines.length} lines`;
      break;
    }
    case 'heatmap': {
      const d = data as HeatmapData;
      const [rows, cols] = d.values.shape;
      const values = decodeArray(d.values) as Float32Array;
      let lo = Infinity, hi = -Infinity;
      for (const x of values) {
        if (!Number.isFinite(x)) continue;
        if (x < lo) lo = x;
        if (x > hi) hi = x;
      }
      if (o.range) [lo, hi] = o.range;
      [lo, hi] = settle(lo, hi);
      const colormap = cmapName(o.colormap, 'viridis');
      const rgba = new Uint8ClampedArray(rows * cols * 4);
      mapValues(values, f32(lo), f32(hi), lut(colormap), rgba);
      p.grid = { rows, cols, sourceRows: d.source_rows, sourceCols: d.source_cols, strideR: d.stride_rows, strideC: d.stride_cols,
        rgba, values, vmin: f32(lo), vmax: f32(hi), colormap, channels: 1 };
      p.info = `${d.source_rows}×${d.source_cols}  ${fmt4(lo)} … ${fmt4(hi)}`;
      break;
    }
    case 'image': {
      const d = data as ImageData;
      const [rows, cols, k] = d.values.shape;
      const values = decodeArray(d.values) as Float32Array;
      let [lo, hi] = o.range ?? [0, d.float ? 1 : 255];
      const rgba = new Uint8ClampedArray(rows * cols * 4).fill(255);
      let colormap = '';
      if (k === 1) {
        colormap = cmapName(o.colormap, 'greys');
        mapValues(values, f32(lo), f32(hi), lut(colormap), rgba);
      } else {
        const span = hi - lo;
        for (let i = 0; i < rows * cols; i++)
          for (let c = 0; c < k; c++) {
            const x = values[i * k + c];
            let u = span !== 0 && Number.isFinite(x) ? (x - lo) / span : 0;
            u = Math.min(1, Math.max(0, u));
            rgba[i * 4 + c] = Math.trunc(u * 255 + 0.5);
          }
      }
      p.grid = { rows, cols, sourceRows: d.source_rows, sourceCols: d.source_cols, strideR: d.stride, strideC: d.stride,
        rgba, values: k === 1 ? values : null, vmin: f32(lo), vmax: f32(hi), colormap, channels: k };
      p.info = `${d.source_rows}×${d.source_cols} ${k === 1 ? 'grey' : k === 3 ? 'RGB' : 'RGBA'}`;
      break;
    }
    case 'points': {
      const d = data as PointsData;
      const b = panel.bind ?? {};
      const one = (slot: string) => {
        const v = b[slot];
        return v === undefined ? '' : Array.isArray(v) ? v[0] : v;
      };
      if (one('positions')) {
        if (!p.xLabel) p.xLabel = 'x';
        if (!p.yLabel) p.yLabel = 'y';
        if (d.three_d && !p.zLabel) p.zLabel = 'z';
      } else {
        if (!p.xLabel) p.xLabel = name(one('x'));
        if (!p.yLabel) p.yLabel = name(one('y'));
        if (one('z') && !p.zLabel) p.zLabel = name(one('z'));
      }
      const x = decodeArray(d.x) as Float32Array, y = decodeArray(d.y) as Float32Array;
      const z = d.z ? (decodeArray(d.z) as Float32Array) : null;
      const color = d.color ? (decodeArray(d.color) as Float32Array) : null;
      const size = d.size ? (decodeArray(d.size) as Float32Array) : null;
      const colormap = cmapName(o.colormap, 'viridis');
      const pts: PreparedPoints = { threeD: d.three_d, x, y, z, rgba: null, scale: null, cmin: 0, cmax: 1, colormap,
        classes: [], colorName: one('color') ? name(one('color')) : '', total: d.total };
      if (color) {
        const classes = classLabels(color, d.color_integer);
        let lo = Infinity, hi = -Infinity;
        for (const c of color) if (Number.isFinite(c)) { lo = Math.min(lo, c); hi = Math.max(hi, c); }
        [pts.cmin, pts.cmax] = settle(lo, hi).map(f32) as [number, number];
        pts.rgba = new Uint8ClampedArray(color.length * 4);
        if (classes) {
          pts.classes = classes;
          for (let i = 0; i < color.length; i++) {
            const rgb = classRgb(Number.isFinite(color[i]) ? Math.trunc(color[i]) : 0);
            pts.rgba.set([rgb[0], rgb[1], rgb[2], 255], i * 4);
          }
        } else {
          const table = lut(colormap), span = f32(pts.cmax - pts.cmin);
          for (let i = 0; i < color.length; i++) {
            const c = color[i];
            const t = Number.isFinite(c) && span !== 0 ? Math.min(1, Math.max(0, f32(f32(c - pts.cmin) / span))) : 0;
            const k = Math.trunc(f32(f32(t * 255) + 0.5)) * 4;
            pts.rgba.set([table[k], table[k + 1], table[k + 2], 255], i * 4);
          }
        }
      }
      if (size) {
        let lo = Infinity, hi = -Infinity;
        for (const s of size) if (Number.isFinite(s)) { lo = Math.min(lo, s); hi = Math.max(hi, s); }
        const [smin, smax] = settle(lo, hi);
        const span = f32(smax - smin);
        pts.scale = new Float32Array(size.length);
        for (let i = 0; i < size.length; i++) {
          const t = Number.isFinite(size[i]) && span !== 0 ? Math.min(1, Math.max(0, (size[i] - smin) / span)) : 0;
          pts.scale[i] = 0.5 + 1.5 * t;
        }
      }
      p.points = pts;
      p.info = `${withCommas(d.total)} points  ${d.three_d ? 3 : 2}-D${pts.classes.length ? `  ${pts.classes.length} classes` : ''}`;
      break;
    }
    case 'histogram': {
      const d = data as HistogramData;
      p.histogram = { counts: d.counts, lo: d.lo, hi: d.hi };
      const v = panel.bind?.values;
      if (!p.xLabel) p.xLabel = name(Array.isArray(v) ? v[0] : v ?? '');
      if (!p.yLabel) p.yLabel = 'count';
      p.info = `${d.counts.length} bins  ${fmt4(d.lo)} … ${fmt4(d.hi)}  n ${fmt4(d.total)}`;
      break;
    }
    case 'table': {
      const d = data as TableData;
      const columns = d.columns.map((c) => c.name);
      const numeric = d.columns.map((c) => numericType(c.type));
      const formats = d.columns.map((_, c) => (numeric[c] ? columnFormat(d.rows.map((r) => r[c] ?? null)) : null));
      const cells = d.rows.map((row) => columns.map((_, c) => {
        const cell = row[c] ?? null;
        const f = formats[c];
        return f && typeof cell === 'number' && Number.isFinite(cell) ? f(cell) : cellText(cell);
      }));
      p.table = { columns, numeric, cells };
      if (!columns.length) {
        p.state = { label: 'No data', detail: panel.id };
        break;
      }
      const shown = d.rows.length, total = d.total_rows;
      p.info = shown < total ? `${withCommas(shown)} of ${withCommas(total)}${d.truncated ? '+' : ''} rows`
        : `${withCommas(total)}${d.truncated ? '+' : ''} rows`;
      break;
    }
    case 'text': {
      const d = data as TextData;
      const v = d.value;
      const value = v === null ? '' : typeof v === 'number' ? fmt4(v) : typeof v === 'boolean' ? fmt4(v ? 1 : 0) : v;
      p.text = { text: o.text ?? '', value, hasValue: v !== null };
      break;
    }
  }
  return p;
}

