// Data views applied in the page, for the mock producer only. A real producer
// computes them on its device (python/caliper/_wire.py resolve_data_view and
// payload_layout); the viewer itself never does.

import type { DataView, IndexSpec, PayloadAxis, Reduction, SourceInfo } from '../protocol/messages.js';

export interface Tensor { data: Float32Array; shape: number[] }

export interface Applied {
  data: Float32Array;
  shape: number[];
  axes: PayloadAxis[];
  resolved: DataView;
  tile: { dim: string; rows: number; cols: number } | null;
}

function reduceInit(op: Reduction): number {
  return op === 'max' ? -Infinity : op === 'min' ? Infinity : 0;
}

/**
 * Applies index, reduce and tile to a tensor whose dims are described by src,
 * and checks that `keep` dims remain (2 for a heatmap, 1 for points).
 * Throws an Error whose message is the reply's error text.
 */
export function applyDataView(t: Tensor, src: SourceInfo, dv: DataView, keep: number, maxExtent = 512): Applied {
  const dims = src.dims;
  const index = dv.index ?? {};
  const reduce = dv.reduce ?? {};
  const sel: { fixed?: number; start: number; stop: number; step: number }[] = [];
  const rindex: Record<string, IndexSpec> = {};
  dims.forEach((d) => {
    const s = index[d.name];
    if (typeof s === 'number') {
      const i = s < 0 ? s + d.size : s;
      if (i < 0 || i >= d.size) throw new Error(`index ${s} out of range for '${d.name}' of size ${d.size}`);
      sel.push({ fixed: i, start: i, stop: i + 1, step: 1 });
      rindex[d.name] = i;
    } else if (Array.isArray(s)) {
      const step = (s.length === 3 ? s[2] : 1) ?? 1;
      const start = Math.max(0, Math.min(d.size, s[0] ?? 0));
      const stop = Math.max(0, Math.min(d.size, s[1] ?? d.size));
      if (stop <= start) throw new Error(`empty range for '${d.name}'`);
      sel.push({ start, stop, step });
      rindex[d.name] = [start, stop, step];
    } else {
      sel.push({ start: 0, stop: d.size, step: 1 });
    }
  });
  for (const k of Object.keys(index)) if (!dims.some((d) => d.name === k)) throw new Error(`source '${src.name}' has no dim '${k}'`);
  for (const k of Object.keys(reduce)) {
    if (!dims.some((d) => d.name === k)) throw new Error(`source '${src.name}' has no dim '${k}'`);
    if (typeof index[k] === 'number') throw new Error(`dim '${k}' is indexed and cannot be reduced`);
  }
  const kept = dims.map((_, i) => i).filter((i) => sel[i].fixed === undefined && !(dims[i].name in reduce));
  const tile = dv.tile && kept.length === keep + 1 ? dv.tile : undefined;
  if (dv.tile && !tile) throw new Error(`tile dim '${dv.tile}' is not usable here`);
  if (kept.length !== keep + (tile ? 1 : 0)) {
    throw new Error(`the data view keeps ${kept.length} dims (${kept.map((i) => dims[i].name).join(', ')}); this view draws ${keep}`);
  }
  const reduced = dims.map((_, i) => i).filter((i) => dims[i].name in reduce);
  const strides = t.shape.map((_, i) => t.shape.slice(i + 1).reduce((a, b) => a * b, 1));
  const counts = sel.map((s) => Math.ceil((s.stop - s.start) / s.step));
  const keptCounts = kept.map((i) => counts[i]);
  const outN = keptCounts.reduce((a, b) => a * b, 1);
  const out = new Float32Array(outN);
  const pos = sel.map((s) => s.start);
  const redCounts = reduced.map((i) => counts[i]);
  const redN = redCounts.reduce((a, b) => a * b, 1);
  for (let o = 0; o < outN; o++) {
    let r = o;
    for (let k = kept.length - 1; k >= 0; k--) {
      const i = kept[k];
      pos[i] = sel[i].start + (r % keptCounts[k]) * sel[i].step;
      r = Math.floor(r / keptCounts[k]);
    }
    if (!reduced.length) {
      out[o] = t.data[pos.reduce((a, p, i) => a + p * strides[i], 0)];
      continue;
    }
    const op = reduce[dims[reduced[0]].name] as Reduction;
    let acc = reduceInit(op);
    for (let q = 0; q < redN; q++) {
      let rr = q;
      for (let k = reduced.length - 1; k >= 0; k--) {
        const i = reduced[k];
        pos[i] = sel[i].start + (rr % redCounts[k]) * sel[i].step;
        rr = Math.floor(rr / redCounts[k]);
      }
      const v = t.data[pos.reduce((a, p, i) => a + p * strides[i], 0)];
      if (op === 'max') acc = Math.max(acc, v);
      else if (op === 'min') acc = Math.min(acc, v);
      else if (op === 'norm') acc += v * v;
      else acc += v;
    }
    out[o] = op === 'mean' ? acc / redN : op === 'norm' ? Math.sqrt(acc) : acc;
  }
  const axes: PayloadAxis[] = kept.map((i) => ({ dim: dims[i].name, start: sel[i].start, step: sel[i].step, count: counts[i], size: counts[i] }));
  const resolved: DataView = { source: src.name, index: rindex, reduce: { ...reduce }, budget: keep === 1 ? { max_points: 100000 } : { max_extent: maxExtent, pool: 'mean' } };
  if (!tile) return { data: out, shape: keptCounts, axes, resolved, tile: null };
  // A mosaic: the tile dim's positions laid out on a grid of (rows, cols) tiles.
  resolved.tile = tile;
  const [tc, rc, cc] = keptCounts;
  const cols = Math.floor(Math.sqrt(tc - 1)) + 1;
  const rows = Math.ceil(tc / cols);
  const H = rows * rc, W = cols * cc;
  const m = new Float32Array(H * W).fill(NaN);
  for (let k = 0; k < tc; k++) {
    const ty = Math.floor(k / cols), tx = k % cols;
    for (let y = 0; y < rc; y++) for (let x = 0; x < cc; x++) m[(ty * rc + y) * W + tx * cc + x] = out[(k * rc + y) * cc + x];
  }
  return { data: m, shape: [H, W], axes, resolved, tile: { dim: tile, rows, cols } };
}
