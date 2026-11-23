// The look of the native spec runtime (src/host/spec_runtime/spec_runtime.cpp,
// house_spec_look) in a browser: the app's colours (src/host/ui/sternbild_theme.h),
// its two faces at the app's sizes, the data palette and the one colour key,
// and text drawn where ImGui draws it.
//
// ImGui sizes a face by its ascender-to-descender height, a browser by its
// em, so a size here is the app's (15 for the UI face) and the CSS size is it
// divided by the face's height ratio (IBM Plex Sans Condensed 1.30, Source Code
// Pro 1.257); the top of a line is where ImGui puts it, and the baseline is
// the ascender below it.

import { PALETTE } from '../render/colormap.js';
import { HOUSE_PIGMENTS, TOKENS } from '../render/palette.gen.js';
import type { Spec } from './types.js';
import { bindingName } from './layout.js';

const tok = (name: string): string => {
  const c = TOKENS[name];
  return `#${c.map((v) => v.toString(16).padStart(2, '0')).join('')}`;
};

/** The house's colours, from the app's own tokens (palette.gen.ts). */
export const COL = {
  deep: tok('deep'),
  ground: tok('ground'),
  plate: tok('plate'),
  plateHover: tok('plate_hover'),
  rule: tok('rule'),
  ruleFaint: tok('rule_faint'),
  text: tok('parchment'),
  secondary: tok('secondary'),
  faint: tok('faint'),
  accent: tok('bronze'),
  flatPoint: 'rgb(220,220,230)',
} as const;

/** The app's sizes at text scale 1: the UI face, its small size, and the mono face of the live views' info and axes. */
export const SIZE = { ui: 15, small: 14, mono: 14, axes: 13 } as const;
export const ROW = 22;   // a table row (sternbild metric::row)

export type Face = 'ui' | 'mono';
const FAMILY: Record<Face, string> = {
  ui: '"Plex Condensed", "IBM Plex Sans Condensed", -apple-system, "Helvetica Neue", sans-serif',
  mono: '"Source Code Pro", ui-monospace, Menlo, monospace',
};
const RATIO: Record<Face, number> = { ui: 1.3, mono: 1.257 };
const ASCENT: Record<Face, number> = { ui: 1025 / 1300, mono: 984 / 1257 };

export function font(face: Face, size: number): string {
  return `${(size / RATIO[face]).toFixed(3)}px ${FAMILY[face]}`;
}

export function textWidth(ctx: CanvasRenderingContext2D, face: Face, size: number, s: string): number {
  ctx.font = font(face, size);
  return ctx.measureText(s).width;
}

/**
 * The text the page drew, when a test asks for it (globalThis.caliperTextLog
 * set to an array): each string with its box in CSS pixels. A picture
 * comparison masks these boxes, because a browser and the app rasterize text
 * differently, and checks the strings and their places on their own.
 */
function logText(ctx: CanvasRenderingContext2D, s: string, x: number, top: number, size: number): void {
  const log = (globalThis as { caliperTextLog?: { s: string; x: number; y: number; w: number; h: number }[] }).caliperTextLog;
  if (!log) return;
  // The box's corners through the canvas's transform (a rotated axis title too), in CSS pixels.
  const t = ctx.getTransform(), d = window.devicePixelRatio || 1, w = ctx.measureText(s).width;
  const xs: number[] = [], ys: number[] = [];
  for (const [u, v] of [[x, top], [x + w, top], [x, top + size], [x + w, top + size]]) {
    xs.push((t.a * u + t.c * v + t.e) / d);
    ys.push((t.b * u + t.d * v + t.f) / d);
  }
  const x0 = Math.min(...xs), y0 = Math.min(...ys);
  log.push({ s, x: x0, y: y0, w: Math.max(...xs) - x0, h: Math.max(...ys) - y0 });
}

/** Text with the top of its line at `top`, at a whole pixel as ImGui's text_at puts it. */
export function text(ctx: CanvasRenderingContext2D, face: Face, size: number, x: number, top: number, color: string, s: string): void {
  if (!s) return;
  ctx.font = font(face, size);
  ctx.fillStyle = color;
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'left';
  ctx.fillText(s, Math.round(x), Math.round(top) + size * ASCENT[face]);
  logText(ctx, s, Math.round(x), Math.round(top), size);
}

/** Text with `tracking` pixels after each character (the house's tracked capitals). */
export function trackedWidth(ctx: CanvasRenderingContext2D, face: Face, size: number, s: string, tracking: number): number {
  ctx.font = font(face, size);
  let w = 0;
  for (const ch of s) w += ctx.measureText(ch).width + tracking;
  return s.length ? w - tracking : 0;
}

export function trackedText(ctx: CanvasRenderingContext2D, face: Face, size: number, x: number, top: number, color: string, s: string, tracking: number): void {
  ctx.font = font(face, size);
  ctx.fillStyle = color;
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'left';
  let at = x;
  for (const ch of s) {
    ctx.fillText(ch, at, Math.round(top) + size * ASCENT[face]);
    at += ctx.measureText(ch).width + tracking;
  }
  logText(ctx, s, x, Math.round(top), size);
}

/** `s` cut to `w` with an ellipsis, "" when not even that fits (fit_text). */
export function fitText(ctx: CanvasRenderingContext2D, face: Face, size: number, s: string, w: number): string {
  if (textWidth(ctx, face, size, s) <= w) return s;
  if (textWidth(ctx, face, size, '…') > w) return '';
  let t = [...s];
  while (t.length) {
    t.pop();
    while (t.length && t[t.length - 1] === ' ') t.pop();
    const cut = t.join('') + '…';
    if (textWidth(ctx, face, size, cut) <= w) return cut;
  }
  return '';
}

/** A filled rectangle in CSS pixels. */
export function fill(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, color: string): void {
  ctx.fillStyle = color;
  ctx.fillRect(x, y, w, h);
}

/** A 1 px outline of the pixels from (x0, y0) to (x1, y1), both included (ImGui's AddRect). */
export function outline(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, color: string): void {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.strokeRect(Math.round(x0) + 0.5, Math.round(y0) + 0.5, Math.round(x1) - Math.round(x0) - 1, Math.round(y1) - Math.round(y0) - 1);
}

const hex = (c: readonly number[]) => `rgb(${c[0]},${c[1]},${c[2]})`;

/**
 * The data palette (spec_palette_color): the house's class pigments (vehicle,
 * dog, bird, cat, other) in their fixed order, then the live views'
 * categorical palette.
 */
export function paletteColor(i: number): string {
  const n = HOUSE_PIGMENTS.length;
  return hex(i < n ? HOUSE_PIGMENTS[i] : PALETTE[(i - n) % 16]);
}

/** The colour of class label L: the live views' categorical palette, L mod 16. */
export function classColor(label: number): string {
  return hex(PALETTE[(label < 0 ? 0 : label) % 16]);
}
export function classRgb(label: number): readonly number[] {
  return PALETTE[(label < 0 ? 0 : label) % 16];
}

/**
 * One colour key for a spec: every name a series draws gets the next colour of
 * the data palette, in the order of the spec's panels, and keeps it in every
 * panel; a table column of the same name carries it as a swatch.
 */
export class ColorKey {
  private readonly index = new Map<string, number>();

  constructor(spec: Spec) {
    for (const p of spec.panel ?? []) {
      if (p.primitive !== 'series') continue;
      const y = p.bind?.y;
      if (y === undefined) continue;
      for (const b of Array.isArray(y) ? y : [y]) this.color(bindingName(spec, b));
    }
  }

  color(name: string): string {
    let i = this.index.get(name);
    if (i === undefined) {
      i = this.index.size;
      this.index.set(name, i);
    }
    return paletteColor(i);
  }

  find(name: string): string | null {
    const i = this.index.get(name);
    return i === undefined ? null : paletteColor(i);
  }
}

/** The live views' point size for a cloud of n points (point_size_px). */
export function pointSize(n: number, scale = 1): number {
  return (n > 50000 ? 3 : n > 5000 ? 4 : n > 500 ? 5.5 : 7) * scale;
}
