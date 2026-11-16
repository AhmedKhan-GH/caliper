// The colormap tables and the categorical palette, generated exactly as the
// native app generates them (src/host/tensor_bridge.cpp for the colormaps,
// src/host/live/live_views_draw.cpp for the palette), so a value is drawn with
// the same colour in the browser and in the app.

import type { ColormapName } from '../protocol/messages.js';

const f = Math.fround;

// Quantizes one channel the way pack_rgba does: clamp to [0, 1], then
// (uint32_t)(x * 255.0f + 0.5f) in float32.
function q(x: number): number {
  let v = f(x);
  if (v < 0) v = 0;
  if (v > 1) v = 1;
  return Math.trunc(f(f(v * 255) + 0.5));
}

// viridis and magma: degree-6 polynomial fits to matplotlib's tables; c[k] is
// the RGB coefficient of t^k. Evaluated in float32 like gen_poly.
const VIRIDIS: readonly number[] = [
  0.2777273272234177, 0.005407344544966578, 0.3340998053353061,
  0.1050930431085774, 1.404613529898575, 1.384590162594685,
  -0.3308618287255563, 0.214847559468213, 0.09509516302823659,
  -4.634230498983486, -5.799100973351585, -19.33244095627987,
  6.228269936347081, 14.17993336680509, 56.69055260068105,
  4.776384997670288, -13.74514537774601, -65.35303263337234,
  -5.435455855934631, 4.645852612178535, 26.3124352495832,
];
const MAGMA: readonly number[] = [
  -0.002136485053939582, -0.000749655052795221, -0.005386127855323933,
  0.2516605407371642, 0.6775232436837668, 2.494026599312351,
  8.353717279216625, -3.577719514958484, 0.3144679030132573,
  -27.66873308576866, 14.26473078096533, -13.64921318813922,
  52.17613981234068, -27.94360607168351, 12.94416944238394,
  -50.76852536473588, 29.04658282127291, 4.23415299384598,
  18.65570506591883, -11.48977351997711, -5.601961508734096,
];
// RdBu: the ColorBrewer 11-class anchors, piecewise linear.
const RDBU: readonly number[] = [
  103, 0, 31, 178, 24, 43, 214, 96, 77, 244, 165, 130, 253, 219, 199, 247, 247, 247,
  209, 229, 240, 146, 197, 222, 67, 147, 195, 33, 102, 172, 5, 48, 97,
];

function polyLut(c: readonly number[]): Uint8Array {
  const out = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = f(i / 255);
    let r = f(c[0]), g = f(c[1]), b = f(c[2]), tp = f(1);
    for (let k = 1; k < 7; k++) {
      tp = f(tp * t);
      r = f(r + f(f(c[3 * k]) * tp));
      g = f(g + f(f(c[3 * k + 1]) * tp));
      b = f(b + f(f(c[3 * k + 2]) * tp));
    }
    out.set([q(r), q(g), q(b), 255], i * 4);
  }
  return out;
}

function rdbuLut(): Uint8Array {
  const out = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const p = f(f(i / 255) * 10);
    let lo = Math.trunc(p);
    if (lo > 9) lo = 9;
    const fr = f(p - lo);
    const ch = (k: number) => {
      const a = RDBU[lo * 3 + k], b = RDBU[(lo + 1) * 3 + k];
      return f(f(a + f(f(b - a) * fr)) / 255);
    };
    out.set([q(ch(0)), q(ch(1)), q(ch(2)), 255], i * 4);
  }
  return out;
}

/** Tableau 10 and six lighter companions, as RGB triples. */
export const PALETTE: readonly (readonly [number, number, number])[] = [
  [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
  [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
  [188, 189, 34], [23, 190, 207], [174, 199, 232], [255, 187, 120],
  [152, 223, 138], [255, 152, 150], [197, 176, 213], [196, 156, 148],
];

const cache = new Map<string, Uint8Array>();

/** The 256-entry RGBA table of a colormap (4 bytes per entry). */
export function colormapLut(name: ColormapName): Uint8Array {
  let lut = cache.get(name);
  if (!lut) {
    lut = name === 'magma' ? polyLut(MAGMA) : name === 'rdbu' ? rdbuLut() : polyLut(VIRIDIS);
    cache.set(name, lut);
  }
  return lut;
}

/**
 * The categorical table: entry L holds palette colour L mod 16. A label L is
 * looked up as entry clamp(L, 0, 255), as in the native points view.
 */
export function paletteLut(): Uint8Array {
  let lut = cache.get('palette');
  if (!lut) {
    lut = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) lut.set([...PALETTE[i % 16], 255], i * 4);
    cache.set('palette', lut);
  }
  return lut;
}

/**
 * The table index of a value for the range [vmin, vmax], exactly as the native
 * reference map_f32_to_rgba8 (src/host/tensor_bridge.cpp) computes it, in
 * float32: NaN is index 0; otherwise t = (v - vmin) / (vmax - vmin), or 0 for
 * a zero width, clamped to [0, 1], and the index is t * 255 + 0.5 truncated.
 * A t that is NaN (an infinite or NaN width, or an infinite v over an
 * infinite width) is index 0, as the reference's conversion of it gives on
 * every platform it runs on. Every value-to-colour mapping of the browser
 * views goes through this function: lutIndex, colormapInto and the WebGPU
 * heatmap's thresholds (colormapThresholds).
 */
export function lutIndexer(vmin: number, vmax: number): (v: number) => number {
  const lo = f(vmin);
  const denom = f(f(vmax) - lo);
  return (v: number): number => {
    if (Number.isNaN(v)) return 0;
    let t = denom !== 0 ? f(f(f(v) - lo) / denom) : 0;
    if (t < 0) t = 0;
    else if (t > 1) t = 1;
    const idx = Math.trunc(f(f(t * 255) + 0.5));
    return idx > 0 ? (idx > 255 ? 255 : idx) : 0;   // NaN and negatives: 0
  };
}

/** The table index of v for [vmin, vmax] (lutIndexer). */
export function lutIndex(v: number, vmin: number, vmax: number): number {
  return lutIndexer(vmin, vmax)(v);
}

/** Floats in their numeric order as unsigned keys (-inf lowest, +inf highest). */
const kbuf = new DataView(new ArrayBuffer(4));
function floatKey(x: number): number {
  kbuf.setFloat32(0, x);
  const u = kbuf.getUint32(0);
  return (u & 0x80000000 ? ~u : u | 0x80000000) >>> 0;
}
function keyFloat(k: number): number {
  kbuf.setUint32(0, (k & 0x80000000 ? k & 0x7fffffff : ~k) >>> 0);
  return kbuf.getFloat32(0);
}

/**
 * The thresholds a WebGPU shader compares values with to find their index
 * with exact float comparisons, as the native WebGPU renderer derives them
 * (src/host/renderer/webgpu_renderer.cpp, colormap_thresholds): T[k] for k in
 * 1..255 is the first float with index k or more for an increasing range
 * (mode 1), the last one for a decreasing range (mode 2), and the index of v
 * is the number of thresholds it has passed. Mode 0, every value at index 0,
 * for a zero, infinite or NaN width, where the reference maps every value to
 * index 0. NaN is index 0 in every mode.
 */
export function colormapThresholds(vmin: number, vmax: number): { mode: number; T: Float32Array } {
  const T = new Float32Array(256);
  const denom = f(f(vmax) - f(vmin));
  if (!(denom !== 0) || !Number.isFinite(denom)) return { mode: 0, T };
  const index = lutIndexer(vmin, vmax);
  const inc = denom > 0;
  const loKey = floatKey(-Infinity), hiKey = floatKey(Infinity);
  for (let k = 1; k < 256; k++) {
    let lo = loKey, hi = hiKey;
    if (inc) {
      while (lo < hi) {
        const mid = lo + Math.floor((hi - lo) / 2);
        if (index(keyFloat(mid)) >= k) hi = mid;
        else lo = mid + 1;
      }
    } else {
      while (lo < hi) {
        const mid = lo + Math.floor((hi - lo + 1) / 2);
        if (index(keyFloat(mid)) >= k) lo = mid;
        else hi = mid - 1;
      }
    }
    T[k] = keyFloat(lo);
  }
  return { mode: inc ? 1 : 2, T };
}

/** The index a shader finds from colormapThresholds, on the CPU (tests). */
export function thresholdIndex(v: number, th: { mode: number; T: Float32Array }): number {
  const x = f(v);
  if (Number.isNaN(x) || th.mode === 0) return 0;
  let lo = 1, hi = 256;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    const passed = th.mode === 1 ? th.T[mid] <= x : x <= th.T[mid];
    if (passed) lo = mid + 1;
    else hi = mid;
  }
  return lo - 1;
}

/** CSS colour of a table entry. */
export function lutCss(lut: Uint8Array, idx: number): string {
  const o = idx * 4;
  return `rgb(${lut[o]},${lut[o + 1]},${lut[o + 2]})`;
}
