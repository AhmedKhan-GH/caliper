// The colour of every colormap edge case in both browser renderers, the
// element's TypeScript views (renderer="ts", which draw with Canvas 2D in every
// project) and Caliper's views compiled to WebAssembly (renderer="wasm", on
// WebGPU, in the projects that have an adapter):
// mock_server.py --edges serves a heatmap and 2-D points coloured by values on
// each side of every index boundary, NaN, the infinities and zeros, over an
// increasing, a decreasing and two very wide ranges (one whose width overflows
// float32). Every cell and every point must have the colour of the native
// reference's index, which lutIndex gives (colormap.spec.ts checks lutIndex
// against the reference's own table); the WebAssembly views also compare
// their heatmap texture with the reference itself, map_f32_to_rgba8.

import { test, type Page } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { colormapLut, lutIndex } from '../src/render/colormap.js';
import { bodyPixels, drawn, expect, shotPath, WEB } from './util.js';
import { HAVE_PYTHON, startProducer, WASM, type Producer } from './producer.js';

test.skip(!HAVE_PYTHON, 'needs a Python with numpy and websockets (CALIPER_PYTHON)');

const HAVE_WASM = ['caliper-views-loader.js', 'caliper-views.js', 'caliper-views.wasm'].every((f) => fs.existsSync(path.join(WASM, f)));
type El = HTMLElement & { debugState(): Record<string, unknown> };
type Renderer = 'ts' | 'wasm';
const renderers = (project: string): Renderer[] => (project === 'canvas2d' || !HAVE_WASM ? ['ts'] : ['ts', 'wasm']);
const RANGES: [number, number][] = [[-3, 5], [1, -1], [-3e38, 3e38], [0, 3e38]];

/** A producer of the edge cases over [vmin, vmax] that also serves points3d.html, and the values it sends. */
async function producer(vmin: number, vmax: number): Promise<Producer & { values: Float32Array }> {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'caliper-edges-'));
  for (const f of fs.readdirSync(path.join(WEB, 'dist'))) fs.copyFileSync(path.join(WEB, 'dist', f), path.join(dir, f));
  fs.copyFileSync(path.join(WEB, 'test', 'points3d.html'), path.join(dir, 'points3d.html'));
  const p = await startProducer(['--dist', dir, '--step-ms', '100000', `--edges=${vmin},${vmax}`]);
  const printed = p.printed.find((j) => 'edges' in j) as { edges: { values: string[] } };
  const values = new Float32Array(new Uint32Array(printed.edges.values.map((h) => parseInt(h, 16))).buffer);
  return { ...p, values, stop: () => { p.stop(); fs.rmSync(dir, { recursive: true, force: true }); } };
}

async function open(page: Page, prod: Producer, view: string, which: Renderer[]): Promise<void> {
  const u = new URL(prod.url);
  await page.setViewportSize({ width: 1200, height: 960 });
  await page.goto(`${u.origin}/points3d.html?token=${u.searchParams.get('token')}&view=${view}&r=${which.join(',')}`);
  await page.waitForFunction(([v, ids]) => (ids as string[]).every((r) => {
    const s = (document.getElementById(`${v}-${r}`) as El | null)?.debugState();
    return !!s && s.status === 'connected' && !!s.renderer && (r !== 'wasm' || s.wasmReady) && !!s.area && !!s.window;
  }), [view, which] as const, { timeout: 30_000 });
  await drawn(page);
}

const state = (page: Page, id: string) => page.evaluate((i) => (document.getElementById(i) as El).debugState(), id) as
  Promise<{ area: { x: number; y: number; w: number; h: number }; window: { x0: number; x1: number; y0: number; y1: number }; wasmStats?: Record<string, unknown> }>;

/** The pixel in the element's body of data point (x, y). */
function pixelOf(s: Awaited<ReturnType<typeof state>>, x: number, y: number, yDown: boolean): [number, number] {
  const a = s.area, w = s.window;
  const fy = (y - w.y0) / (w.y1 - w.y0);
  return [Math.floor(a.x + ((x - w.x0) / (w.x1 - w.x0)) * a.w), Math.floor(yDown ? a.y + fy * a.h : a.y + a.h - fy * a.h)];
}

for (const [vmin, vmax] of RANGES) {
  test(`every edge case has the reference's colour over [${vmin}, ${vmax}]`, async ({ page }, info) => {
    const prod = await producer(vmin, vmax);
    try {
      const lut = colormapLut('magma');
      const want = (v: number) => [...lut.subarray(lutIndex(v, vmin, vmax) * 4, lutIndex(v, vmin, vmax) * 4 + 3), 255];
      const which = renderers(info.project.name);
      for (const [view, n] of [['edges', 64 * 64], ['edge_points', 1024]] as const) {
        await open(page, prod, view, which);
        for (const r of which) {
          const id = `${view}-${r}`;
          const s = await state(page, id);
          const { px } = await bodyPixels(page, `#${id}`);
          const wrong: string[] = [];
          for (let i = 0; i < n; i++) {
            const v = prod.values[i % prod.values.length];
            // A heatmap cell's centre (rows grow down), or a point at (i % 32, i / 32).
            const [x, y] = view === 'edges' ? pixelOf(s, (i % 64) + 0.5, Math.floor(i / 64) + 0.5, true)
              : pixelOf(s, i % 32, Math.floor(i / 32), false);
            const got = px.at(x, y), w = want(v);
            if (got.join() !== w.join()) wrong.push(`${i} (${v}) at ${x},${y}: ${got} != ${w}`);
          }
          await page.locator(`#${id}`).screenshot({ path: shotPath(info, `edges-${view}-${r}-${vmin}-${vmax}`) });
          expect(wrong.slice(0, 8), `${r} ${view}: ${wrong.length} of ${n} differ`).toEqual([]);
        }
        if (view === 'edges' && which.includes('wasm')) {
          // The module's own comparison of its texture with map_f32_to_rgba8: "edges:1:mismatches:maxdiff".
          await expect.poll(async () => String(((await state(page, 'edges-wasm')).wasmStats ?? {}).checks ?? ''), { timeout: 20_000 })
            .toMatch(/(^|,)edges:1:\d+:\d+/);
          const m = /(?:^|,)edges:1:(\d+):(\d+)/.exec(String((await state(page, 'edges-wasm')).wasmStats?.checks))!;
          expect(Number(m[1]), 'texels that differ from map_f32_to_rgba8').toBe(0);
        }
      }
    } finally {
      prod.stop();
    }
  });
}
