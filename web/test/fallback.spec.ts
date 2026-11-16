// The element without Caliper's WebAssembly views, as in a checkout where
// neither scripts/build_wasm.py nor scripts/fetch_wasm.py has run: the server
// answers 404 for caliper-views-loader.js, and renderer="auto" draws every
// view with the element's own TypeScript views, with Canvas 2D also where
// there is an adapter, with no error on the page.

import { expect, test } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { HAVE_PYTHON, startProducer } from './producer.js';

test.skip(!HAVE_PYTHON, 'needs a Python with numpy and websockets (CALIPER_PYTHON)');

type El = HTMLElement & { debugState(): Record<string, unknown> };

test('auto draws every view with the TypeScript views when the WebAssembly views are missing', async ({ page }) => {
  const empty = fs.mkdtempSync(path.join(os.tmpdir(), 'caliper-no-views-'));
  const prod = await startProducer([], empty);
  try {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    const loader: number[] = [];
    page.on('response', (r) => {
      if (new URL(r.url()).pathname === '/caliper-views-loader.js') loader.push(r.status());
    });
    await page.goto(prod.url);
    await page.waitForFunction(() => {
      const els = [...document.querySelectorAll('caliper-view')] as El[];
      return els.length > 0 && els.every((e) => e.debugState().status === 'connected' && e.debugState().renderer);
    }, null, { timeout: 30_000 });
    const got = await page.evaluate(() => [...document.querySelectorAll('caliper-view')].map((e) => {
      const s = (e as El).debugState();
      return [s.view, s.kind, s.drawer, s.renderer] as [string, string, string, string];
    }));
    expect(new Set(got.map((g) => g[1]))).toEqual(new Set(['heatmap', 'image', 'points', 'series', 'histogram']));
    for (const [view, , drawer, renderer] of got) {
      expect(drawer, view).toBe('ts');
      // The TypeScript views draw every kind with Canvas 2D: they have no WebGPU path.
      expect(renderer, view).toBe('canvas2d');
    }
    // The element asked for the loader, was told it is not there, and fell back.
    expect(loader.length).toBeGreaterThan(0);
    expect(loader.every((s) => s === 404)).toBe(true);
    // The views draw: each has a visible canvas with a size.
    const sized = await page.evaluate(() => [...document.querySelectorAll('caliper-view')].map((e) =>
      [...e.shadowRoot!.querySelectorAll('canvas')].some((c) => c.offsetWidth > 0 && c.width > 0)));
    expect(sized.every(Boolean)).toBe(true);
    expect(errors).toEqual([]);
  } finally {
    prod.stop();
    fs.rmSync(empty, { recursive: true, force: true });
  }
});
