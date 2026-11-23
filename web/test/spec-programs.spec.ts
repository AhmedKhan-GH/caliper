// The programs of the spec page on the recorded Arc Tower (tests/draw_parity/dashboard), against the mock of the
// app: a click on an arc of the tower picks it in the WebAssembly views and the page gets the item's fields, as the
// app's pick event carries them; the panel's `select` writes the head into the page's own copy of the state, and
// the grid's program, whose params read that state and the tower's selection, takes the new values. Nothing goes
// to the app: the viewer may not run state.set or selection.set, so the selection stays in the page. Needs WebGPU.

import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { mockApp, openSpec, VIEWS_FILES, type Json, type MockApp } from './spec-mock.js';
import { REPO } from './util.js';

const WASM = process.env.CALIPER_WASM_DIR ?? path.join(REPO, 'python', 'caliper', '_web');
const FIXTURE = path.join(REPO, 'tests', 'draw_parity', 'dashboard');

test.beforeEach(({}, info) => test.skip(info.project.name === 'canvas2d', 'the programs draw with WebGPU, which the canvas2d project disables'));
test.skip(!VIEWS_FILES.every((f) => fs.existsSync(path.join(WASM, f))), `needs the WebAssembly views in ${WASM}`);
test.skip(!fs.existsSync(path.join(FIXTURE, 'spec.json')), 'needs the recording in tests/draw_parity/dashboard');

// Known failure, kept as the check of its fix: the module receives the clicks (pointer and button events count up)
// and reports no WebGPU error, but the page-fed scene view gives no pick event (its pageEvents stay 0), so the
// frame pick of the programs returns no item. The page's handling of a pick (fields, select, params, linked
// selection) waits for it.
test.fixme('a pick of an arc gives its fields and writes the page own state, and sends nothing to the app', async ({ page }) => {
  test.setTimeout(120_000);
  const spec = JSON.parse(fs.readFileSync(path.join(FIXTURE, 'spec.json'), 'utf8')) as Json;
  const data: Record<string, Json> = {};
  for (const f of fs.readdirSync(path.join(FIXTURE, 'panels')))
    if (f.endsWith('.json')) data[f.slice(0, -5)] = JSON.parse(fs.readFileSync(path.join(FIXTURE, 'panels', f), 'utf8'));
  const app: MockApp = { specs: { [spec.applet.id]: spec }, data: { [spec.applet.id]: data }, requests: [] };
  await mockApp(page, app, WASM);
  await page.setViewportSize({ width: 1280, height: 800 });
  await openSpec(page, spec.applet.id);
  await page.waitForFunction(() => {
    const s = (window as any).caliperSpec?.debugState();
    const mounts = s ? Object.values(s.programs) as any[] : [];
    return mounts.length === 2 && mounts.every((m) => m.mounted && m.programs.every((p: any) => p.module === 'running' && p.frames >= 1));
  }, null, { timeout: 60_000, polling: 100 });

  // Coloured pixels of the arcs (not the white token labels), read in the same task as the module's frame.
  const candidates = await page.evaluate(() => new Promise<{ x: number; y: number }[]>((done) => {
    const c = (window as any).caliperSpec.programs.canvasOf('tower') as HTMLCanvasElement;
    const original = window.requestAnimationFrame.bind(window);
    original(() => original(() => {
      const probe = document.createElement('canvas');
      probe.width = c.width;
      probe.height = c.height;
      const g = probe.getContext('2d', { willReadFrequently: true })!;
      g.drawImage(c, 0, 0);
      const d = g.getImageData(0, 0, c.width, c.height).data;
      const all: [number, number, number][] = [];
      for (let y = 0; y < c.height; y += 2)
        for (let x = 0; x < c.width; x += 2) {
          const i = 4 * (y * c.width + x), mx = Math.max(d[i], d[i + 1], d[i + 2]), mn = Math.min(d[i], d[i + 1], d[i + 2]);
          if (mx - mn > 50 && mx > 120) all.push([2 * mx - mn, x, y]);
        }
      all.sort((a, b) => b[0] - a[0]);
      const r = c.getBoundingClientRect(), out: [number, number][] = [];
      for (const [, x, y] of all) {
        if (out.some((q) => Math.abs(q[0] - x) < 12 && Math.abs(q[1] - y) < 12)) continue;
        out.push([x, y]);
        if (out.length >= 24) break;
      }
      done(out.map(([x, y]) => ({ x: r.left + x * r.width / c.width, y: r.top + y * r.height / c.height })));
    }));
  }));
  expect(candidates.length).toBeGreaterThan(0);
  let picked = false;
  for (const at of candidates) {
    await page.mouse.click(at.x, at.y);
    picked = await page.waitForFunction(() => (window as any).caliperSpec.debugState().picks.length > 0, null, { timeout: 3000 })
      .then(() => true, () => false);
    if (picked) break;
  }
  if (!picked) {
    const stats = await page.evaluate(() => {
      const m = (window as any).caliperSpec.programs.mounts.get('tower');
      const s = m?.handle?.stats ?? {};
      return { inputPointer: s.inputPointer, inputButtons: s.inputButtons, pageEvents: s.pageEvents, pageLastEvent: s.pageLastEvent,
        gfxErrors: s.gfxErrors };
    });
    throw new Error(`no pick from ${candidates.length} clicks on the arcs; the tower module counted ${JSON.stringify(stats)}`);
  }
  const s = await page.evaluate(() => (window as any).caliperSpec.debugState());
  const pick = s.picks.at(-1);
  expect(pick.panel).toBe('tower');
  expect(pick.detail.program).toBe('tower_arcs');
  // The item's fields, as the piece's pick names them.
  for (const f of ['layer', 'head', 'from_token', 'to_token', 'weight', 'score']) expect(typeof pick.detail.fields[f]).toBe('number');
  // The panel's select writes the head into the page's own state.
  expect(s.localState.head).toBe(pick.detail.fields.head);
  expect(s.programs.tower.pick.item).toBe(pick.detail.item);
  // Nothing went to the app but reads.
  expect(app.requests.every((r) => r.name === 'spec.get' || r.name === 'panel.data' || r.name.startsWith('GET caliper-views'))).toBe(true);
});
