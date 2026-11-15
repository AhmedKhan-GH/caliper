// The real socket path end to end: mock_server.py (the protocol's Python
// codec over websockets) serves dist/ and a synthetic stream on 127.0.0.1
// with a session token; the viewer page and the anywidget module connect to
// it in headless Chromium.

import { test as base, type Page } from '@playwright/test';
import { spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { expect, REPO, serve, shotPath, WEB, wireModule } from './util.js';

const PYTHON = process.env.CALIPER_PYTHON ?? path.join(REPO, '.venv', 'bin', 'python');

interface Server { url: string; ws: string; origin: string; events: Record<string, unknown>[]; proc: ChildProcess }

const test = base.extend<{ producer: Server }>({
  producer: async ({}, use) => {
    const proc = spawn(PYTHON, [path.join(WEB, 'test', 'mock_server.py'), '--wire', wireModule()], { stdio: ['ignore', 'pipe', 'inherit'] });
    const events: Record<string, unknown>[] = [];
    const lines = readline.createInterface({ input: proc.stdout! });
    const url = await new Promise<string>((ok, fail) => {
      proc.once('exit', (code) => fail(new Error(`mock_server.py exited with ${code}`)));
      lines.on('line', (line) => {
        if (line.startsWith('http://')) ok(line.trim());
        else if (line.startsWith('{')) events.push(JSON.parse(line).event);
      });
    });
    const u = new URL(url);
    await use({ url, ws: `ws://${u.host}/v3/stream?token=${u.searchParams.get('token')}`, origin: u.origin, events, proc });
    proc.kill();
  },
});

test.skip(!fs.existsSync(PYTHON), 'needs a Python with numpy and websockets (CALIPER_PYTHON)');

async function plates(page: Page, n: number) {
  await page.waitForFunction((n) => {
    const els = [...document.querySelectorAll('caliper-view')] as (HTMLElement & { debugState(): Record<string, unknown> })[];
    return els.length === n && els.every((e) => e.debugState().status === 'connected' && e.debugState().renderer);
  }, n, { timeout: 20_000 });
}

const state = (page: Page, view: string) => page.evaluate((v) =>
  (document.querySelector(`caliper-view[view="${v}"]`) as HTMLElement & { debugState(): Record<string, unknown> }).debugState(), view);

test('the viewer page shows every view of the stream', async ({ page, producer }, info) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(producer.url);
  await plates(page, 6);
  await expect(page.locator('.wordmark')).toHaveText('Caliper');
  await expect(page.locator('#script')).toHaveText('train_mock.py');
  await expect(page.locator('#state .text')).toHaveText('connected');
  const gpu = info.project.name === 'canvas2d' ? 'canvas2d' : 'webgpu';
  const got = await page.evaluate(() => [...document.querySelectorAll('caliper-view')].map((e) => {
    const s = (e as HTMLElement & { debugState(): Record<string, unknown> }).debugState();
    return [s.view, s.kind, s.renderer];
  }));
  expect(got).toEqual([
    ['loss', 'series', 'canvas2d'], ['weights', 'heatmap', gpu], ['attention', 'heatmap', gpu],
    ['embedding', 'points', gpu], ['weight_hist', 'histogram', 'canvas2d'], ['input', 'image', gpu],
  ]);
  // Data keeps arriving: the newest snapshot of a view advances.
  const seq0 = (await state(page, 'weights')).seq as number;
  await expect.poll(async () => (await state(page, 'weights')).seq as number).toBeGreaterThan(seq0);
  await page.waitForTimeout(400);
  await page.screenshot({ path: shotPath(info, 'page'), fullPage: true });
  expect(errors).toEqual([]);
});

test('a data view picked in the page travels over the socket and comes back resolved', async ({ page, producer }) => {
  await page.goto(producer.url);
  await plates(page, 6);
  const pick = (dim: string, value: string) => page.evaluate(({ dim, value }) => {
    const el = document.querySelector('caliper-view[view="attention"]')!;
    const sel = el.shadowRoot!.querySelector(`select[data-dim="${dim}"]`) as HTMLSelectElement;
    sel.value = value;
    sel.dispatchEvent(new Event('change'));
  }, { dim, value });
  await pick('head', 'i:3');
  await expect.poll(async () => JSON.stringify((await state(page, 'attention')).dataView)).toContain('"head":3');
  // All heads of a layer: the picker asks for a mosaic, and the producer sends 3 x 3 tiles of 32 x 32.
  await pick('head', 'all');
  await expect.poll(async () => {
    const s = await state(page, 'attention');
    return [s.rows, s.cols, (s.dataView as { tile?: string }).tile];
  }).toEqual([96, 96, 'head']);
  // A request the producer refuses (four kept dims) is shown and leaves the data view as it was.
  await pick('layer', 'all');
  await expect(page.locator('caliper-view[view="attention"] .msg.error')).toContainText('kept dims');
  expect(((await state(page, 'attention')).dataView as { tile?: string }).tile).toBe('head');
});

test('a selection in the page reaches the producer as an event', async ({ page, producer }) => {
  await page.goto(producer.url);
  await plates(page, 6);
  const el = page.locator('caliper-view[view="weights"]');
  const s = await state(page, 'weights');
  const area = s.area as { x: number; y: number; w: number; h: number };
  const box = (await el.locator('.body').boundingBox())!;
  // Cells are area.w / 64 wide and area.h / 48 high; drag from cell (2, 4) to cell (5, 9).
  const at = (r: number, c: number) => [box.x + area.x + (c + 0.5) * (area.w / 64), box.y + area.y + (r + 0.5) * (area.h / 48)];
  const [x0, y0] = at(2, 4), [x1, y1] = at(5, 9);
  await page.keyboard.down('Shift');
  await page.mouse.move(x0, y0);
  await page.mouse.down();
  await page.mouse.move(x1, y1, { steps: 4 });
  await page.mouse.up();
  await page.keyboard.up('Shift');
  await expect.poll(() => producer.events.filter((e) => e.type === 'select')).toEqual([
    expect.objectContaining({ view: 'weights', type: 'select', detail: { rows: [2, 6], cols: [4, 10] } }),
  ]);
  const ev = producer.events.find((e) => e.type === 'select')!;
  expect(typeof ev.sub).toBe('number');
  expect(typeof ev.seq).toBe('number');
});

test('the anywidget module draws a view from the stream and writes its traits back', async ({ page, producer }) => {
  // A blank page on another loopback origin, as a notebook front end is.
  const front = await serve(path.join(WEB, 'test'), (name) => (name === '/' ? path.join(WEB, 'test', 'blank.html') : null));
  test.info().attach('front end', { body: front.origin });
  await page.goto(`http://localhost:${new URL(front.origin).port}/`);
  await page.evaluate(async ({ origin, ws }) => {
    const mod = await import(`${origin}/caliper-widget.js`);
    const traits: Record<string, unknown> = { src: ws, view: 'weights', kind: 'heatmap', height: 300, data_view: {}, selection: {} };
    const handlers = new Map<string, Set<() => void>>();
    const model = {
      saved: 0,
      get: (k: string) => traits[k],
      set: (k: string, v: unknown) => {
        traits[k] = v;
        for (const h of handlers.get(`change:${k}`) ?? []) h();
      },
      save_changes() {
        this.saved++;
      },
      on: (e: string, h: () => void) => handlers.set(e, (handlers.get(e) ?? new Set()).add(h)),
      off: (e: string, h: () => void) => handlers.get(e)?.delete(h),
    };
    Object.assign(window, { model, traits });
    mod.default.render({ model, el: document.getElementById('root') });
  }, { origin: producer.origin, ws: producer.ws });
  await plates(page, 1);
  expect(await page.evaluate(() => {
    const el = document.querySelector('caliper-view')!;
    return [el.getAttribute('view'), el.getAttribute('kind'), el.getAttribute('height')];
  })).toEqual(['weights', 'heatmap', '300']);
  // Python sets data_view: the element asks the producer for it.
  await page.evaluate(() => (window as unknown as { model: { set(k: string, v: unknown): void } }).model.set('data_view', { index: { out: [0, 24] } }));
  await expect.poll(async () => (await state(page, 'weights')).rows).toBe(24);
  // A selection in the element is written to the selection trait and saved.
  const s = await state(page, 'weights');
  const area = s.area as { x: number; y: number; w: number; h: number };
  const box = (await page.locator('caliper-view .body').boundingBox())!;
  await page.keyboard.down('Shift');
  await page.mouse.move(box.x + area.x + 2, box.y + area.y + 2);
  await page.mouse.down();
  await page.mouse.move(box.x + area.x + area.w / 2, box.y + area.y + area.h / 2, { steps: 3 });
  await page.mouse.up();
  await page.keyboard.up('Shift');
  const t = await page.evaluate(() => ({
    selection: (window as unknown as { traits: Record<string, unknown> }).traits.selection,
    saved: (window as unknown as { model: { saved: number } }).model.saved,
  }));
  expect(t.selection).toMatchObject({ view: 'weights', type: 'select', detail: { rows: [0, 12], cols: [0, 32] } });
  expect(t.saved).toBeGreaterThan(0);
  await expect.poll(() => producer.events.some((e) => e.type === 'select')).toBe(true);
  front.close();
});
