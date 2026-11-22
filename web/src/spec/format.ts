// How the native spec runtime writes numbers, in TypeScript: spec_fmt4 for
// headers and values, printf's %g for ImPlot's tick labels, thousands
// separators, and a table's one format per numeric column
// (src/host/spec_runtime/spec_prepare.cpp).

import type { Cell } from './types.js';

/** Strips the zeros a fixed or exponent mantissa ends with, and a bare point. */
function stripZeros(m: string): string {
  if (!m.includes('.')) return m;
  return m.replace(/0+$/, '').replace(/\.$/, '');
}

/** Four significant digits, as the live views' headers write numbers (spec_fmt4). */
export function fmt4(v: number): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? 'NaN' : v > 0 ? 'inf' : '-inf';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) {
    const [mant, ex] = v.toExponential(3).split('e');
    const n = parseInt(ex, 10);
    return `${stripZeros(mant)}e${n < 0 ? '-' : ''}${Math.abs(n)}`;
  }
  return String(Number(v.toPrecision(4)));
}

/** printf("%.<p>g"). */
export function fmtG(v: number, p = 6): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? 'nan' : v > 0 ? 'inf' : '-inf';
  if (v === 0) return Object.is(v, -0) ? '-0' : '0';
  const [mant, ex] = v.toExponential(p - 1).split('e');
  const x = parseInt(ex, 10);
  if (x < -4 || x >= p) {
    const e = Math.abs(x);
    return `${stripZeros(mant)}e${x < 0 ? '-' : '+'}${e < 10 ? '0' : ''}${e}`;
  }
  return stripZeros(v.toFixed(Math.max(0, p - 1 - x)));
}

/** printf("%.<d>e"): a mantissa and a signed exponent of at least two digits. */
export function fmtE(v: number, d: number): string {
  const [mant, ex] = v.toExponential(d).split('e');
  const x = parseInt(ex, 10), e = Math.abs(x);
  return `${mant}e${x < 0 ? '-' : '+'}${e < 10 ? '0' : ''}${e}`;
}

export function withCommas(n: number): string {
  return Math.trunc(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

const NUMERIC = ['TINYINT', 'SMALLINT', 'INTEGER', 'BIGINT', 'HUGEINT', 'UTINYINT', 'USMALLINT', 'UINTEGER', 'UBIGINT',
  'UHUGEINT', 'FLOAT', 'DOUBLE', 'REAL', 'DECIMAL', 'BOOLEAN', 'INT', 'FLOAT4', 'FLOAT8'];

/** A DuckDB type of numbers (numeric_type): its figures are right-aligned. */
export const numericType = (t: string): boolean => NUMERIC.some((n) => t.startsWith(n));

/** A cell as table text: integers in full, other numbers in six significant digits, strings as they are, nothing for null. */
export function cellText(c: Cell): string {
  if (c === null) return '';
  if (typeof c === 'boolean') return c ? 'true' : 'false';
  if (typeof c === 'number') {
    if (Number.isFinite(c) && Number.isInteger(c) && Math.abs(c) < 1e15) return c.toFixed(0);
    return fmtG(c, 6);
  }
  return c.replace(/[\u0000-\u001f]/g, ' ');
}

/**
 * The one format of each numeric column, so its figures line up: whole numbers
 * in full, otherwise the decimals that give the largest value five significant
 * digits, and an exponent beyond 1e9 or below 1e-6.
 */
export function columnFormat(values: Cell[]): (v: number) => string {
  let integral = true, big = 0;
  for (const c of values) {
    if (typeof c !== 'number' || !Number.isFinite(c)) continue;
    if (!Number.isInteger(c)) integral = false;
    big = Math.max(big, Math.abs(c));
  }
  if (integral && big < 1e15) return (v) => v.toFixed(0);
  if (big >= 1e9 || (big > 0 && big < 1e-6)) return (v) => fmtE(v, 4);
  const d = big > 0 ? Math.min(8, Math.max(0, 4 - Math.floor(Math.log10(big)))) : 4;
  return (v) => v.toFixed(d);
}
