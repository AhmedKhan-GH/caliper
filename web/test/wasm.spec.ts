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
    // The loader lists heatmap, image, series and histogram; points stay with the TypeScript views.
    expect(got).toEqual([
      ['loss', 'wasm', 'wasm', 'loss', false], ['weights', 'wasm', 'wasm', 'weights', false],
      ['attention', 'wasm', 'wasm', 'attention', false], ['embedding', 'ts', 'webgpu', 'embedding', false],
      ['weight_hist', 'wasm', 'wasm', 'weight_hist', false], ['input', 'wasm', 'wasm', 'input', false],
    ]);
    // Each WebAssembly view drew on its own canvas inside the element's body.
    const canvases = await page.evaluate(() => [...document.querySelectorAll('caliper-view')]
      .filter((e) => (e as El).debugState().drawer === 'wasm')
      .map((e) => {
        const c = e.shadowRoot!.querySelector('.body .wasm canvas') as HTMLCanvasElement | null;
        return c ? c.width > 0 && c.height > 0 : false;
      }));
    expect(canvases).toEqual([true, true, true, true, true]);
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
