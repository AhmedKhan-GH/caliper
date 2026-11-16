// The browser views' colour index (lutIndex) against the native reference,
// map_f32_to_rgba8 (src/host/tensor_bridge.cpp), on a table it produced
// (fixtures/lut_index.json, made by fixtures/make_lut_index.py): increasing,
// decreasing and zero-width ranges, ranges with an infinite or NaN end or an
// infinite width, and values at index boundaries, NaN, the infinities, zeros
// and subnormals. The threshold search the WebGPU shaders do, and the Canvas 2D
// heatmap's colormapInto, must give the same index for every case.

import { test } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { colormapLut, colormapThresholds, lutIndex, thresholdIndex } from '../src/render/colormap.js';
import { colormapInto } from '../src/render/heatmap-2d.js';
import { expect, WEB } from './util.js';

const table = JSON.parse(fs.readFileSync(path.join(WEB, 'test', 'fixtures', 'lut_index.json'), 'utf8')) as {
  cases: [string, string, string, number][];
};
const dv = new DataView(new ArrayBuffer(4));
const flt = (hex: string) => {
  dv.setUint32(0, parseInt(hex, 16));
  return dv.getFloat32(0);
};
const cases = table.cases.map(([v, lo, hi, idx]) => ({ v: flt(v), lo: flt(lo), hi: flt(hi), idx, key: `${v} ${lo} ${hi}` }));

test('lutIndex gives the native reference index for every case of the table', () => {
  expect(cases.length).toBeGreaterThan(500);
  const wrong = cases.filter((c) => lutIndex(c.v, c.lo, c.hi) !== c.idx).map((c) => `${c.key}: ${lutIndex(c.v, c.lo, c.hi)} != ${c.idx}`);
  expect(wrong).toEqual([]);
});

test('the WebGPU shaders\' threshold search and the Canvas 2D heatmap give the same indices', () => {
  const ranges = new Map<string, typeof cases>();
  for (const c of cases) {
    const k = c.key.split(' ').slice(1).join(' ');
    if (!ranges.has(k)) ranges.set(k, []);
    ranges.get(k)!.push(c);
  }
  const lut = colormapLut('magma');
  const wrong: string[] = [];
  for (const group of ranges.values()) {
    const { lo, hi } = group[0];
    const th = colormapThresholds(lo, hi);
    for (const c of group) {
      const got = thresholdIndex(c.v, th);
      if (got !== c.idx) wrong.push(`thresholds ${c.key}: ${got} != ${c.idx}`);
    }
    const data = new Float32Array(group.map((c) => c.v));
    const out = new Uint8ClampedArray(data.length * 4);
    colormapInto(out, data, lo, hi, lut);
    group.forEach((c, i) => {
      const want = [...lut.subarray(c.idx * 4, c.idx * 4 + 3), 255];
      const got = [...out.subarray(i * 4, i * 4 + 4)];
      if (got.join() !== want.join()) wrong.push(`colormapInto ${c.key}: ${got} != ${want}`);
    });
  }
  expect(wrong).toEqual([]);
});
