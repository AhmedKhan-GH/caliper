// The programs of the spec page on the recorded Arc Tower (tests/draw_parity/dashboard), against the mock of the
// app: a click on an arc of the tower picks it in the WebAssembly views and the page gets the item's fields, as the
// app's pick event carries them; the panel's `select` writes the head into the page's own copy of the state, the
// grid's program, whose params read that state and the tower's selection, takes the new values, and the picked arc
// draws highlighted, because the linked selection reaches the programs. Nothing goes to the app: the viewer may
// not run state.set or selection.set, so the selection stays in the page. The programs' text tables give their
// labels, and a panel's view_camera starts the scene's camera as in the app. Needs WebGPU.

import { test, expect, type Page } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { mockApp, openSpec, VIEWS_FILES, type Json, type MockApp } from './spec-mock.js';
import { REPO } from './util.js';

const WASM = process.env.CALIPER_WASM_DIR ?? path.join(REPO, 'python', 'caliper', '_web');
const FIXTURE = path.join(REPO, 'tests', 'draw_parity', 'dashboard');

test.beforeEach(({}, info) => test.skip(info.project.name === 'canvas2d', 'the programs draw with WebGPU, which the canvas2d project disables'));
test.skip(!VIEWS_FILES.every((f) => fs.existsSync(path.join(WASM, f))), `needs the WebAssembly views in ${WASM}`);
test.skip(!fs.existsSync(path.join(FIXTURE, 'spec.json')), 'needs the recording in tests/draw_parity/dashboard');

/** The recorded dashboard against the mock of the app, with `change` applied to its spec first. */
async function openRecorded(page: Page, change?: (spec: Json) => void): Promise<{ spec: Json; app: MockApp }> {
  const spec = JSON.parse(fs.readFileSync(path.join(FIXTURE, 'spec.json'), 'utf8')) as Json;
  change?.(spec);
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
  return { spec, app };
}

/** A panel's module canvas as RGBA bytes, read in the same task as the module's frame (two frames on). */
function pixels(page: Page, panel: string): Promise<{ w: number; h: number; px: number[] }> {
  return page.evaluate((id) => new Promise<{ w: number; h: number; px: number[] }>((done) => {
    const c = (window as any).caliperSpec.programs.canvasOf(id) as HTMLCanvasElement;
    const original = window.requestAnimationFrame.bind(window);
    original(() => original(() => {
      const probe = document.createElement('canvas');
      probe.width = c.width;
      probe.height = c.height;
      const g = probe.getContext('2d', { willReadFrequently: true })!;
      g.drawImage(c, 0, 0);
      done({ w: c.width, h: c.height, px: Array.from(g.getImageData(0, 0, c.width, c.height).data) });
    }));
  }), panel);
}

/** A number of the module's stats of a panel. */
function stat(page: Page, panel: string, name: string): Promise<unknown> {
  return page.evaluate(([id, n]) => (window as any).caliperSpec.programs.mounts.get(id)?.handle?.stats?.[n], [panel, name]);
}

test('a pick of an arc gives its fields, writes the page own state, highlights the arc and sends nothing to the app', async ({ page }) => {
  test.setTimeout(120_000);
  const { app } = await openRecorded(page);

  // The programs' text tables give their labels: the tower's tokens and the grid's rows, drawn over the scene.
  const labels = async (panel: string) => {
    const t = await stat(page, panel, 'pageLabels');
    return typeof t === 'string' ? (JSON.parse(t) as [string, number, number, number, number][]).map((l) => l[0]) : [];
  };
  await expect.poll(async () => (await labels('tower')).length, { timeout: 10_000 }).toBeGreaterThan(0);
  await expect.poll(async () => (await labels('pattern')).length, { timeout: 10_000 }).toBeGreaterThan(0);
  expect(await labels('tower')).toContain('fox');

  // Coloured pixels of the arcs (not the white token labels), at most 24 at least 12 pixels apart.
  const before = await pixels(page, 'tower');
  const all: [number, number, number][] = [];
  for (let y = 0; y < before.h; y += 2)
    for (let x = 0; x < before.w; x += 2) {
      const i = 4 * (y * before.w + x), d = before.px;
      const mx = Math.max(d[i], d[i + 1], d[i + 2]), mn = Math.min(d[i], d[i + 1], d[i + 2]);
      if (mx - mn > 50 && mx > 120) all.push([2 * mx - mn, x, y]);
    }
  all.sort((a, b) => b[0] - a[0]);
  const chosen: [number, number][] = [];
  for (const [, x, y] of all) {
    if (chosen.some((q) => Math.abs(q[0] - x) < 12 && Math.abs(q[1] - y) < 12)) continue;
    chosen.push([x, y]);
    if (chosen.length >= 24) break;
  }
  expect(chosen.length).toBeGreaterThan(0);
  // From the canvas's pixels to the page's CSS pixels, through the device pixel ratio of the canvas.
  const box = await page.evaluate(() => {
    const r = ((window as any).caliperSpec.programs.canvasOf('tower') as HTMLCanvasElement).getBoundingClientRect();
    return { left: r.left, top: r.top, width: r.width, height: r.height };
  });
  let picked = false;
  for (const [x, y] of chosen) {
    await page.mouse.click(box.left + (x + 0.5) * box.width / before.w, box.top + (y + 0.5) * box.height / before.h);
    picked = await page.waitForFunction(() => (window as any).caliperSpec.debugState().picks.length > 0, null, { timeout: 3000 })
      .then(() => true, () => false);
    if (picked) break;
  }
  if (!picked) {
    const stats = await page.evaluate(() => {
      const s = (window as any).caliperSpec.programs.mounts.get('tower')?.handle?.stats ?? {};
      return { inputPointer: s.inputPointer, inputButtons: s.inputButtons, pageEvents: s.pageEvents, pageLastEvent: s.pageLastEvent,
        gfxErrors: s.gfxErrors };
    });
    throw new Error(`no pick from ${chosen.length} clicks on the arcs; the tower module counted ${JSON.stringify(stats)}`);
  }
  const s = await page.evaluate(() => (window as any).caliperSpec.debugState());
  const pick = s.picks.at(-1);
  expect(pick.panel).toBe('tower');
  expect(pick.detail.program).toBe('tower_arcs');
  // The item's fields, as the piece's pick names them.
  for (const f of ['layer', 'head', 'from_token', 'to_token', 'weight', 'score']) expect(typeof pick.detail.fields[f]).toBe('number');
  // The panel's select writes the head into the page's own state, and the grid's params follow the pick.
  expect(s.localState.head).toBe(pick.detail.fields.head);
  expect(s.programs.tower.pick.item).toBe(pick.detail.item);
  await expect.poll(async () => {
    const d = await page.evaluate(() => (window as any).caliperSpec.debugState().programs.pattern.programs[0].params);
    return d && [d.head, d.layer];
  }, { timeout: 5000 }).toEqual([pick.detail.fields.head, pick.detail.fields.layer]);

  // The picked arc draws highlighted: the selection reaches the tower's program, and only the highlight changes
  // its picture (its data and params stay), in the warm colour of the piece's selected arcs.
  let changed = 0, warm = 0;
  await expect.poll(async () => {
    const after = await pixels(page, 'tower');
    changed = 0;
    warm = 0;
    for (let i = 0; i < after.px.length; i += 4) {
      const d = Math.max(Math.abs(after.px[i] - before.px[i]), Math.abs(after.px[i + 1] - before.px[i + 1]),
        Math.abs(after.px[i + 2] - before.px[i + 2]));
      if (d <= 24) continue;
      ++changed;
      if (after.px[i] >= 200 && after.px[i + 1] >= 170 && after.px[i] - after.px[i + 2] >= 30) ++warm;
    }
    return warm;
  }, { timeout: 5000, message: 'the picked arc is drawn in its highlight colour' }).toBeGreaterThan(20);
  console.log(`pick: item ${pick.detail.item} head ${pick.detail.fields.head} layer ${pick.detail.fields.layer}; ` +
    `highlight: ${changed} pixels changed, ${warm} in the highlight colour`);

  // Nothing went to the app but reads.
  expect(app.requests.every((r) => r.name === 'spec.get' || r.name === 'panel.data' || r.name.startsWith('GET caliper-views'))).toBe(true);
});

test('a panel view_camera starts the scene camera with its angles, target and distance, as in the app', async ({ page }) => {
  test.setTimeout(120_000);
  await openRecorded(page, (spec) => {
    const tower = spec.panel.find((p: Json) => p.id === 'tower');
    tower.options.view_camera = { azimuth: 40, elevation: 25, target: [0.5, 1, -0.5], distance: 7 };
  });
  await expect.poll(async () => {
    const o = await stat(page, 'tower', 'pageOrbit');
    return typeof o === 'string' ? JSON.parse(o) : null;
  }, { timeout: 10_000 }).not.toBeNull();
  const orbit = JSON.parse(await stat(page, 'tower', 'pageOrbit') as string) as number[];
  const deg = Math.PI / 180;
  expect(orbit.slice(0, 4)).toEqual([0.5, 1, -0.5, 7]);
  expect(orbit[4]).toBeCloseTo(40 * deg, 9);
  expect(orbit[5]).toBeCloseTo(25 * deg, 9);
});


test('removing the last program hides its canvas and clears its resources', async ({ page }) => {
  const { spec, app } = await openRecorded(page);
  const panels = app.data[spec.applet.id];
  const original = panels.tower;
  panels.tower = { ...original, version: 'empty-programs', data: { ...original.data, programs: [] } };
  await page.waitForFunction(() => {
    const views = (window as any).caliperSpec.programs;
    const mount = views.mounts.get('tower');
    return mount && !views.has('tower') && mount.fed.size === 0 && mount.versions.size === 0 &&
      mount.box.style.display === 'none';
  });
  panels.tower = { ...original, version: 'restored-programs' };
  await page.waitForFunction(() => {
    const views = (window as any).caliperSpec.programs;
    const mount = views.mounts.get('tower');
    const programs = (window as any).caliperSpec.debugState().programs.tower.programs;
    return views.has('tower') && mount.box.style.display === 'block' &&
      programs.length > 0 && programs.every((p: any) => p.module === 'running' && p.frames >= 1);
  });
});
