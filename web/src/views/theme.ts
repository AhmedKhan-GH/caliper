// The palette of Caliper's site and app (site/index.html, sternbild_theme.h).

export const C = {
  ground: '#181715',
  plate: '#272622',
  ruleFaint: '#36342f',
  rule: '#5a5751',
  parchment: '#f0e9db',
  secondary: '#d0c9ba',
  faint: '#a49d8f',
  bronze: '#b38b59',
} as const;

/** The plate colour as linear 0..1 RGBA, for GPU clears. */
export const PLATE_RGBA = [0x27 / 255, 0x26 / 255, 0x22 / 255, 1] as const;

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
