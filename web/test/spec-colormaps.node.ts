import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { FLAT_POINT, SPEC_COLORMAP_NAMES, isSpecColormapName, specColormapLut } from '../src/render/colormaps.gen.js';
import { CATEGORICAL, HOUSE_PIGMENTS, TOKENS } from '../src/render/palette.gen.js';
import { lut, preparePanel } from '../src/spec/prepare.js';
import type { PanelDataResult, Spec, SpecPanel } from '../src/spec/types.js';
import { C } from '../src/views/theme.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '..', '..');
const shared = JSON.parse(fs.readFileSync(path.join(repoRoot, 'scripts', 'shared_colors.json'), 'utf8')) as {
  colormaps: { entries: { id: number; name: string; kind: string; rgba8?: string[] }[] };
  theme: {
    tokens: { name: string; rgb: [number, number, number] }[];
    house_pigments: string[];
    untokenized: { name: string; rgba: [number, number, number, number] }[];
  };
  categorical_palette: [number, number, number][];
};

function encodeF32(values: number[]): string {
  const buf = Buffer.from(new Float32Array(values).buffer);
  return buf.toString('base64');
}

test('specColormapLut and prepare.lut expose all 16 colormaps matching shared_colors.json', () => {
  const wantNames = shared.colormaps.entries.map((e) => e.name);
  assert.deepEqual([...SPEC_COLORMAP_NAMES], wantNames);
  assert.equal(isSpecColormapName('cividis'), true);
  assert.equal(isSpecColormapName('berlin'), true);
  assert.equal(isSpecColormapName('jet'), false);

  for (const entry of shared.colormaps.entries) {
    const a = specColormapLut(entry.name as (typeof SPEC_COLORMAP_NAMES)[number]);
    const b = lut(entry.name);
    assert.equal(a.length, 1024, entry.name);
    assert.deepEqual(b, a, entry.name);
    if (entry.kind === 'lut256' && entry.rgba8) {
      for (let i = 0; i < 256; i++) {
        const packed: number = Number.parseInt(entry.rgba8[i], 16) >>> 0;
        assert.equal(a[i * 4 + 0], packed & 0xff, `${entry.name}[${i}].r`);
        assert.equal(a[i * 4 + 1], (packed >>> 8) & 0xff, `${entry.name}[${i}].g`);
        assert.equal(a[i * 4 + 2], (packed >>> 16) & 0xff, `${entry.name}[${i}].b`);
        assert.equal(a[i * 4 + 3], 255, `${entry.name}[${i}].a`);
      }
    }
  }
});

test('preparePanel preserves and maps all 16 spec colormaps on heatmaps', () => {
  for (const name of SPEC_COLORMAP_NAMES) {
    const panel: SpecPanel = {
      id: 'h',
      primitive: 'heatmap',
      bind: { values: 't' },
      options: { colormap: name, range: [0, 1] },
    };
    const spec: Spec = {
      caliper_spec: 1,
      applet: { id: 'cmap-test' },
      source: [{ id: 't', catalog: 'tensor:t' }],
      panel: [panel],
    };
    const result: PanelDataResult = {
      panel: 'h',
      primitive: 'heatmap',
      version: 'v1',
      unchanged: false,
      error: null,
      state: null,
      bounds: [],
      data: {
        source_rows: 1,
        source_cols: 2,
        stride_rows: 1,
        stride_cols: 1,
        values: { dtype: 'f32', shape: [1, 2], base64: encodeF32([0, 1]) },
      },
    };
    const prepared = preparePanel(spec, panel, result);
    assert.ok(prepared.grid);
    assert.equal(prepared.grid.colormap, name);
    const table = lut(name);
    assert.deepEqual([...prepared.grid.rgba.slice(0, 4)], [...table.slice(0, 4)], `${name} low`);
    assert.deepEqual([...prepared.grid.rgba.slice(4, 8)], [...table.slice(255 * 4, 256 * 4)], `${name} high`);
  }
});

test('palette.gen.ts, FLAT_POINT, and views/theme.ts match shared_colors.json', () => {
  for (const t of shared.theme.tokens) {
    assert.deepEqual([...TOKENS[t.name]], t.rgb, t.name);
  }
  assert.deepEqual(
    HOUSE_PIGMENTS.map((c) => [...c]),
    shared.theme.house_pigments.map((n) => [...TOKENS[n]]),
  );
  const flat = shared.theme.untokenized.find((u) => u.name === 'flat_point')!;
  assert.deepEqual([...FLAT_POINT], flat.rgba.slice(0, 3));
  assert.deepEqual(CATEGORICAL.map((c) => [...c]), shared.categorical_palette);
  const hex = (rgb: readonly [number, number, number]) =>
    `#${rgb.map((v) => v.toString(16).padStart(2, '0')).join('')}`;
  const tokByName = Object.fromEntries(shared.theme.tokens.map((t) => [t.name, t.rgb]));
  assert.deepEqual({ ...C }, {
    deep: hex(tokByName.deep),
    ground: hex(tokByName.ground),
    plate: hex(tokByName.plate),
    ruleFaint: hex(tokByName.rule_faint),
    rule: hex(tokByName.rule),
    parchment: hex(tokByName.parchment),
    secondary: hex(tokByName.secondary),
    faint: hex(tokByName.faint),
    bronze: hex(tokByName.bronze),
  });
});
