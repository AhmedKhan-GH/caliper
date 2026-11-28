// Linked selection with item details (wire protocol 3.1), against
// mock_server.py --controls --fixed: a selection in one view comes back from
// the producer to every view that maps the same item space, on this page and
// on a second one, and the views draw the highlights. With WebGPU the views
// on the first page are Caliper's WebAssembly views, in Canvas 2D the
// element's own; so this spec runs on swiftshader in CI as well as canvas2d.

import { expect, test, type Page } from '@playwright/test';
import { open, second, state } from './pages.js';
import { HAVE_PYTHON, startProducer, type Producer } from './producer.js';
import { drawn, roundTrip, shotPath } from './util.js';

test.skip(!HAVE_PYTHON, 'needs a Python with numpy and websockets (CALIPER_PYTHON)');

// Two pages of WebAssembly views on the software adapter are slow to start
// while other tests run.
test.beforeEach(({}, info) => test.slow(info.project.name === 'swiftshader'));

test.describe('linked selection', () => {
  let prod: Producer;
  test.afterEach(() => prod?.stop());

  /** Page coordinates of data point (x, y) of a view, from its debug area and window. */
  async function at(page: Page, view: string, x: number, y: number) {
    const d = await state(page, view);
    const body = (await page.locator(`caliper-view[view="${view}"] .body`).boundingBox())!;
    const yDown = view === 'h' || view === 'ha';
    const fx = (x - d.window.x0) / (d.window.x1 - d.window.x0), fy = (y - d.window.y0) / (d.window.y1 - d.window.y0);
    return { x: body.x + d.area.x + fx * d.area.w, y: body.y + d.area.y + (yDown ? fy * d.area.h : d.area.h - fy * d.area.h) };
  }

  async function shiftDrag(page: Page, from: { x: number; y: number }, to: { x: number; y: number }, alt = false) {
    await page.keyboard.down('Shift');
    if (alt) await page.keyboard.down('Alt');
    await page.mouse.move(from.x, from.y);
    await page.mouse.down();
    await page.mouse.move(to.x, to.y, { steps: 6 });
    await page.mouse.up();
    if (alt) await page.keyboard.up('Alt');
    await page.keyboard.up('Shift');
  }

  test('a selection of points highlights the same items in the heatmap and on another page, with the details under the view', async ({ page, browser }, info) => {
    prod = await startProducer(['--controls', '--fixed']);
    await open(page, prod.url, 5);
    const other = await second(browser, prod.url, ['p', 'h']);
    try {
      // Every view has drawn: a WebAssembly view is past its start-up frames,
      // over which its windows are laid out, and publishes where it drew.
      await drawn(page);
      await expect.poll(async () => (await state(page, 'p')).window?.x1 ?? null).not.toBeNull();
      // The view name of each element that dispatches caliper-selection.
      const events = await page.evaluateHandle(() => {
        const r: string[] = [];
        document.addEventListener('caliper-selection', (e) => r.push((e.target as HTMLElement).getAttribute('view') ?? ''));
        return r;
      });
      // A rectangle around the right-hand points (rows 3 and 9 of the space "test").
      await shiftDrag(page, await at(page, 'p', 0.8, -0.02), await at(page, 'p', 1.2, 1.02), true);
      await roundTrip.poll(() => prod.events.filter((e) => e.type === 'select').at(-1)?.detail ?? null)
        .toEqual({ indices: [1, 3], row_ids: [3, 9], space: 'test', items: [3, 9] });
      // The producer's SELECTION comes back to every view that maps the space, on both pages.
      const want = { space: 'test', axis: 'points', items: [3, 9] };
      await roundTrip.poll(async () => (await state(page, 'p')).linked).toEqual(want);
      await roundTrip.poll(async () => (await state(page, 'h')).linked).toEqual({ space: 'test', axis: 'rows', items: [3, 9] });
      await roundTrip.poll(async () => (await state(other.page, 'h')).linked).toEqual({ space: 'test', axis: 'rows', items: [3, 9] });
      await roundTrip.poll(async () => (await state(other.page, 'p')).selected).toBe(2);
      expect((await state(page, 'ha')).linked).toBeNull();   // its rows are not in a space
      // The details of the items under the view the selection was made in.
      await roundTrip.poll(async () => (await state(page, 'p')).details).toBe('itemlabelscore330.005910.044');
      expect((await state(page, 'h')).details).toBeNull();
      // Each SELECTION comes as an event from p, where the selection was made, and from h, the other
      // view that maps the space "test". The drag can send more than one select, thus the test
      // compares the views, not a count of events.
      expect(await events.evaluate((r) => [...new Set(r)].sort())).toEqual(['h', 'p']);
      await drawn(page, 'caliper-view[view="p"], caliper-view[view="h"]');
      await page.locator('caliper-view[view="p"]').screenshot({ path: shotPath(info, 'linked-points') });
      await page.locator('caliper-view[view="h"]').screenshot({ path: shotPath(info, 'linked-heatmap') });
      // A shift-click clears it everywhere.
      const c = await at(page, 'p', 0.5, 0.5);
      await shiftDrag(page, c, c);
      await roundTrip.poll(async () => (await state(other.page, 'h')).linked?.items ?? null).toEqual([]);
      await roundTrip.poll(async () => (await state(page, 'p')).details).toBeNull();
    } finally {
      await other.close();
    }
  });

  test('a range of steps puts markers on the series', async ({ page }, info) => {
    prod = await startProducer(['--controls', '--fixed']);
    await open(page, prod.url, 5);
    await drawn(page);
    await expect.poll(async () => (await state(page, 's')).window?.x1 ?? null).not.toBeNull();
    const from = await at(page, 's', 300.2, 500), to = await at(page, 's', 340.8, 500);
    await shiftDrag(page, { x: Math.ceil(from.x), y: from.y }, { x: Math.floor(to.x), y: to.y });
    await roundTrip.poll(async () => (await state(page, 's')).linked?.axis ?? null).toBe('steps');
    const l = (await state(page, 's')).linked!;
    expect(l.space).toBe('step');
    expect(l.items[0]).toBeGreaterThanOrEqual(300);
    expect(l.items[0]).toBeLessThanOrEqual(303);
    expect(l.items.at(-1)).toBeGreaterThanOrEqual(338);
    expect(l.items.at(-1)).toBeLessThanOrEqual(341);
    await page.mouse.move(10, 10);
    await drawn(page, 'caliper-view[view="s"]');
    await page.locator('caliper-view[view="s"]').screenshot({ path: shotPath(info, 'linked-series') });
  });
});
