// The colormap tables and the categorical palette, generated exactly as the
// native app generates them (src/host/tensor_bridge.cpp and
// src/host/gfx/colormap_tables.h for the colormaps,
// src/host/live/live_views_draw.cpp for the palette), so a value is drawn with
// the same colour in the browser and in the app.

import type { ColormapName } from '../protocol/messages.js';
import { isSpecColormapName, specColormapLut } from './colormaps.gen.js';
import { CATEGORICAL } from './palette.gen.js';

const f = Math.fround;

/**
 * Tableau 10 and six lighter companions, as RGB triples: the app's own table
 * (caliper_scene::kPalette), through the generated palette.gen.ts.
 */
export const PALETTE: readonly (readonly [number, number, number])[] = CATEGORICAL;

const cache = new Map<string, Uint8Array>();

/** The 256-entry RGBA table of any of the 16 built-in colormaps (4 bytes per entry). */
export function colormapLut(name: ColormapName): Uint8Array {
  return specColormapLut(isSpecColormapName(name) ? name : 'viridis');
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
 * infinite width) is index 0, as the reference gives it. Every value-to-colour
 * mapping of the element's own views goes through this function: lutIndex
 * (the points view) and colormapInto (the heatmap).
 */
export function lutIndexer(vmin: number, vmax: number): (v: number) => number {
  const lo = f(vmin);
  const denom = f(f(vmax) - lo);
  return (v: number): number => {
    if (Number.isNaN(v)) return 0;
    let t = denom !== 0 ? f(f(f(v) - lo) / denom) : 0;
    if (!(t >= 0)) t = 0;   // also a NaN quotient, from an infinite end or span
    else if (t > 1) t = 1;
    const idx = Math.trunc(f(f(t * 255) + 0.5));
    return idx > 0 ? (idx > 255 ? 255 : idx) : 0;
  };
}

/** The table index of v for [vmin, vmax] (lutIndexer). */
export function lutIndex(v: number, vmin: number, vmax: number): number {
  return lutIndexer(vmin, vmax)(v);
}

/** CSS colour of a table entry. */
export function lutCss(lut: Uint8Array, idx: number): string {
  const o = idx * 4;
  return `rgb(${lut[o]},${lut[o + 1]},${lut[o + 2]})`;
}
