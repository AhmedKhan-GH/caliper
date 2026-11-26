// The palette of Caliper's site and app (site/index.html, sternbild_theme.h),
// derived from the v2 design tokens generated into render/palette.gen.ts.

import { TOKENS, type Rgb } from '../render/palette.gen.js';

function hex(c: Rgb): string {
  return `#${c.map((v) => v.toString(16).padStart(2, '0')).join('')}`;
}

export const C = {
  deep: hex(TOKENS.deep),
  ground: hex(TOKENS.ground),
  plate: hex(TOKENS.plate),
  ruleFaint: hex(TOKENS.rule_faint),
  rule: hex(TOKENS.rule),
  parchment: hex(TOKENS.parchment),
  secondary: hex(TOKENS.secondary),
  faint: hex(TOKENS.faint),
  bronze: hex(TOKENS.bronze),
} as const;

export const FONT_TEXT = '"Plex Condensed", "IBM Plex Sans Condensed", -apple-system, "Helvetica Neue", sans-serif';
export const FONT_MONO = '"Source Code Pro", ui-monospace, Menlo, monospace';

/** A number for a label: short, and exact enough to read. */
export function fmt(v: number, digits = 4): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? 'NaN' : v > 0 ? 'inf' : '-inf';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return v.toExponential(digits - 1).replace('e+', 'e');
  return Number(v.toPrecision(digits)).toString();
}
