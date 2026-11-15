// The element with Caliper's real WebAssembly views (not the stub): the
// producer, mock_server.py, also serves the three files of web/wasm's build,
// as caliper.serve() does from the Python package. The files come from
// CALIPER_WASM_DIR, else from python/caliper/_web (the checked-in copies).

import { test as base, type Page } from '@playwright/test';
import { spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { expect, REPO, shotPath, WEB, wireModule } from './util.js';

const PYTHON = process.env.CALIPER_PYTHON ?? path.join(REPO, '.venv', 'bin', 'python');
const WASM = process.env.CALIPER_WASM_DIR ?? path.join(REPO, 'python', 'caliper', '_web');
const FILES = ['caliper-views-loader.js', 'caliper-views.js', 'caliper-views.wasm'];

interface Server { url: string; ws: string; proc: ChildProcess }

const test = base.extend<{ producer: Server }>({
  producer: async ({}, use) => {
    const proc = spawn(PYTHON, [path.join(WEB, 'test', 'mock_server.py'), '--wire', wireModule(), '--wasm', WASM],
                       { stdio: ['ignore', 'pipe', 'inherit'] });
    const lines = readline.createInterface({ input: proc.stdout! });
    const url = await new Promise<string>((ok, fail) => {
      proc.once('exit', (code) => fail(new Error(`mock_server.py exited with ${code}`)));
      lines.on('line', (line) => { if (line.startsWith('http://')) ok(line.trim()); });
    });
    const u = new URL(url);
    await use({ url, ws: `ws://${u.host}/v3/stream?token=${u.searchParams.get('token')}`, proc });
    proc.kill();
  },
});

test.skip(!fs.existsSync(PYTHON), 'needs a Python with numpy and websockets (CALIPER_PYTHON)');
test.skip(!FILES.every((f) => fs.existsSync(path.join(WASM, f))), `needs the WebAssembly views in ${WASM}`);

type El = HTMLElement & { debugState(): Record<string, unknown>; dataView: unknown };
const state = (page: Page, view: string) => page.evaluate((v) =>
  (document.querySelector(`caliper-view[view="${v}"]`) as El).debugState(), view);

/** A second producer with its own options (the fixture's has the defaults). */
async function producerWith(args: string[]): Promise<Server> {
  const proc = spawn(PYTHON, [path.join(WEB, 'test', 'mock_server.py'), '--wire', wireModule(), '--wasm', WASM, ...args],
                     { stdio: ['ignore', 'pipe', 'inherit'] });
  const lines = readline.createInterface({ input: proc.stdout! });
  const url = await new Promise<string>((ok, fail) => {
    proc.once('exit', (code) => fail(new Error(`mock_server.py exited with ${code}`)));
    lines.on('line', (line) => { if (line.startsWith('http://')) ok(line.trim()); });
  });
  const u = new URL(url);
  return { url, ws: `ws://${u.host}/v3/stream?token=${u.searchParams.get('token')}`, proc };
}

async function wasmStats(page: Page, view: string): Promise<Record<string, unknown>> {
  return ((await state(page, view)).wasmStats ?? {}) as Record<string, unknown>;
}

async function plates(page: Page, n: number) {
  await page.waitForFunction((n) => {
    const els = [...document.querySelectorAll('caliper-view')] as El[];
    return els.length === n && els.every((e) => e.debugState().status === 'connected' && e.debugState().renderer);
  }, n, { timeout: 30_000 });
}

test('auto draws each view with the WebAssembly views when they list its kind, and with the TypeScript views otherwise', async ({ page, producer }, info) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto(producer.url);
  await plates(page, 6);
  const got = await page.evaluate(() => [...document.querySelectorAll('caliper-view')].map((e) => {
    const s = (e as El).debugState();
    const root = e.shadowRoot!;
    return [s.view, s.drawer, s.renderer, root.querySelector('.name')?.textContent, (root.querySelector('.plate') as HTMLElement).hidden];
  }));
  if (info.project.name === 'canvas2d') {
    // No adapter: the TypeScript views draw everything.
    expect(got.map((g) => g[1])).toEqual(['ts', 'ts', 'ts', 'ts', 'ts', 'ts']);
  } else {
    // The loader lists every kind the stream has, so the WebAssembly views draw them all.
    expect(got).toEqual([
      ['loss', 'wasm', 'wasm', 'loss', false], ['weights', 'wasm', 'wasm', 'weights', false],
      ['attention', 'wasm', 'wasm', 'attention', false], ['embedding', 'wasm', 'wasm', 'embedding', false],
      ['weight_hist', 'wasm', 'wasm', 'weight_hist', false], ['input', 'wasm', 'wasm', 'input', false],
    ]);
    // Each WebAssembly view drew on its own canvas inside the element's body.
    const canvases = await page.evaluate(() => [...document.querySelectorAll('caliper-view')]
      .filter((e) => (e as El).debugState().drawer === 'wasm')
      .map((e) => {
        const c = e.shadowRoot!.querySelector('.body .wasm canvas') as HTMLCanvasElement | null;
        return c ? c.width > 0 && c.height > 0 : false;
      }));
    expect(canvases).toEqual([true, true, true, true, true, true]);
    // Each module read its texture back: the image is exactly the bytes that were sent, and the
    // heatmap exactly the CPU reference colormap (web/wasm/README.md, "The colormap").
    const checks = async (view: string) => {
      const st = (await state(page, view)).wasmStats as { checks?: string } | null;
      return st?.checks ?? '';
    };
    await expect.poll(() => checks('input'), { timeout: 15_000 }).toMatch(/^input:2:\d+:\d+$/);
    await expect.poll(() => checks('weights'), { timeout: 15_000 }).toMatch(/^weights:1:\d+:\d+$/);
    const [, , imageBad, imageMax] = (await checks('input')).split(':').map(Number);
    expect([imageBad, imageMax]).toEqual([0, 0]);
    const [, , heatBad, heatMax] = (await checks('weights')).split(':').map(Number);
    expect([heatBad, heatMax]).toEqual([0, 0]);
    // The histogram holds the producer's bins.
    const hist = (await state(page, 'weight_hist')).wasmStats as { viewInfo?: string };
    expect(hist.viewInfo).toMatch(/weight_hist:kind=5,updates=\d+,count=\d+/);
    expect(Number(/count=(\d+)/.exec(hist.viewInfo!)![1])).toBeGreaterThan(0);
  }
  await page.waitForTimeout(600);
  await page.screenshot({ path: shotPath(info, 'wasm-page'), fullPage: true });
  expect(errors).toEqual([]);
});

test('the header picker sets the WebAssembly view\'s data view, and the stream\'s reply confirms it', async ({ page, producer }, info) => {
  test.skip(info.project.name === 'canvas2d', 'the WebAssembly views need a WebGPU adapter');
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto(producer.url);
  await plates(page, 6);
  expect((await state(page, 'attention')).drawer).toBe('wasm');
  // The picker is built from what the view offers (the element's metadata connection).
  const dims = await page.evaluate(() => [...document.querySelector('caliper-view[view="attention"]')!.shadowRoot!.querySelectorAll('select[data-dim]')]
    .map((s) => (s as HTMLSelectElement).dataset.dim));
  expect(dims).toEqual(['layer', 'head', 'query', 'key']);
  const before = JSON.stringify((await state(page, 'attention')).dataView);
  await page.evaluate(() => {
    const sel = document.querySelector('caliper-view[view="attention"]')!.shadowRoot!.querySelector('select[data-dim="head"]') as HTMLSelectElement;
    sel.value = 'i:5';
    sel.dispatchEvent(new Event('change'));
  });
  // The module's VIEW_REQUEST is answered; the reply (not the request) sets the element's data view.
  await expect.poll(async () => JSON.stringify((await state(page, 'attention')).dataView), { timeout: 10_000 }).toContain('"head":5');
  expect(before).not.toContain('"head":5');
  await page.waitForTimeout(400);
  await page.locator('caliper-view[view="attention"]').screenshot({ path: shotPath(info, 'wasm-picker') });
});

test('points draw as quads with their palette colours, and hover finds the point under the mouse', async ({ page, producer }, info) => {
  test.skip(info.project.name === 'canvas2d', 'the WebAssembly views need a WebGPU adapter');
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto(producer.url);
  await plates(page, 6);
  expect((await state(page, 'embedding')).drawer).toBe('wasm');
  // The module read its render target back: at every sampled point no other point covers, the
  // pixel is the palette colour of the point's label ("view:3:mismatches:compared").
  await expect.poll(async () => String((await wasmStats(page, 'embedding')).checks ?? ''), { timeout: 20_000 })
    .toMatch(/^embedding:3:\d+:\d+$/);
  const [, , bad, compared] = String((await wasmStats(page, 'embedding')).checks).split(':').map(Number);
  expect(bad).toBe(0);
  expect(compared).toBeGreaterThan(10);
  // Hover: somewhere around the middle of the cloud the module finds a point.
  const box = (await page.locator('caliper-view[view="embedding"] .body').boundingBox())!;
  const hover = async () => Number(/hover=(-?\d+)/.exec(String((await wasmStats(page, 'embedding')).viewInfo))?.[1] ?? -1);
  let found = -1;
  for (let k = 0; k < 40 && found < 0; ++k) {
    const r = 4 + k * 3, a = k * 0.9;
    await page.mouse.move(box.x + box.width / 2 + r * Math.cos(a), box.y + box.height / 2 + r * Math.sin(a));
    await page.waitForTimeout(300);
    found = await hover();
  }
  expect(found).toBeGreaterThanOrEqual(0);
  await page.locator('caliper-view[view="embedding"]').screenshot({ path: shotPath(info, 'wasm-points-hover') });
});

test('100,000 points orbit at an interactive rate on the GPU', async ({ page }, info) => {
  test.skip(info.project.name !== 'webgpu', 'measured on the machine\'s GPU only');
  const big = await producerWith(['--points', '100000']);
  try {
    await page.setViewportSize({ width: 1000, height: 760 });
    await page.goto(big.url);
    await plates(page, 6);
    expect((await state(page, 'embedding')).drawer).toBe('wasm');
    const box = (await page.locator('caliper-view[view="embedding"] .body').boundingBox())!;
    const frames = async () => Number((await wasmStats(page, 'embedding')).frames ?? 0);
    const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
    await page.mouse.move(cx, cy);
    await page.mouse.down();
    const f0 = await frames();
    const t0 = Date.now();
    for (let k = 0; Date.now() - t0 < 3000; ++k) {
      await page.mouse.move(cx + 120 * Math.sin(k / 20), cy + 60 * Math.cos(k / 27));
      await page.waitForTimeout(8);
    }
    const fps = ((await frames()) - f0) / ((Date.now() - t0) / 1000);
    await page.mouse.up();
    console.log(`${info.project.name} wasm points 100000: ${fps.toFixed(1)} frames per second while orbiting`);
    expect(fps).toBeGreaterThan(30);
    await page.locator('caliper-view[view="embedding"]').screenshot({ path: shotPath(info, 'wasm-points-100k') });
  } finally {
    big.proc.kill();
  }
});
