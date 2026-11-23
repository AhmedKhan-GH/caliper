// Every live view of the element's own, against known data pushed through
// the hand-driven StaticSource: what is drawn, hover, zoom and pan,
// selections, toggles and the data view picker. A stream given through the
// source property is always drawn by the element's own views, which draw with
// Canvas 2D in every project, WebGPU's included.

import type { Page } from '@playwright/test';
import { colormapLut, lutIndex, PALETTE } from '../src/render/colormap.js';
import { bodyPixels, expect, shotPath, test } from './util.js';

type Dbg = Record<string, unknown> & {
  renderer: string; window: { x0: number; x1: number; y0: number; y1: number }; area: { x: number; y: number; w: number; h: number };
};

/** Adds <caliper-view id=name> fed by a StaticSource stored as window.src[name]. */
async function mount(page: Page, name: string, kind: string, sources: unknown[] = [], dflt: unknown = {}, width = 560, height = 360) {
  await page.evaluate(({ name, kind, sources, dflt, width, height }) => {
    const w = window as unknown as { harness: { StaticSource: new () => unknown }; src: Record<string, unknown>; events: unknown[] };
    const src = new w.harness.StaticSource() as { addView(n: string, k: string, s: unknown[], d: unknown): void };
    src.addView(name, kind, sources, dflt);
    const el = document.createElement('caliper-view') as HTMLElement & { source: unknown };
    el.id = name;
    el.setAttribute('view', name);
    el.setAttribute('height', String(height));
    el.style.width = `${width}px`;
    el.style.marginBottom = '16px';
    el.source = src;
    w.events ??= [];
    for (const t of ['caliper-select', 'caliper-dataview']) el.addEventListener(t, (e) => w.events.push([t, (e as CustomEvent).detail]));
    document.getElementById('root')!.append(el);
    (w.src ??= {})[name] = src;
  }, { name, kind, sources, dflt, width, height });
}

async function push(page: Page, name: string, snap: Record<string, unknown>) {
  await page.evaluate(({ name, snap }) => {
    const s = snap as Record<string, unknown> & { data: number[]; aux?: number[]; u8?: boolean };
    const w = window as unknown as { src: Record<string, { snapshot(v: string, s: unknown): void }> };
    const data = s.u8 ? new Uint8Array(s.data) : new Float32Array(s.data.map((v) => (v === null ? NaN : v)));
    w.src[name].snapshot(name, { ...s, data, aux: s.aux ? new Float32Array(s.aux) : undefined });
  }, { name, snap });
  await settled(page, name);
}

async function series(page: Page, name: string, start: number, values: number[] | { n: number; f: string }) {
  await page.evaluate(({ name, start, values }) => {
    const w = window as unknown as { src: Record<string, { series(v: string, s: number, x: Float32Array): void }> };
    let arr: Float32Array;
    if (Array.isArray(values)) arr = new Float32Array(values);
    else {
      const f = new Function('i', `return ${values.f}`) as (i: number) => number;
      arr = new Float32Array(values.n).map((_, k) => f(start + k));
    }
    w.src[name].series(name, start, arr);
  }, { name, start, values });
  await settled(page, name);
}

const settled = (page: Page, name: string) =>
  page.evaluate((n) => (document.getElementById(n) as HTMLElement & { settled(): Promise<void> }).settled(), name);
const dbg = (page: Page, name: string) =>
  page.evaluate((n) => (document.getElementById(n) as HTMLElement & { debugState(): unknown }).debugState(), name) as Promise<Dbg>;
const tip = (page: Page, name: string) =>
  page.evaluate((n) => {
    const t = document.getElementById(n)!.shadowRoot!.querySelector('.tip') as HTMLElement;
    return t.hidden ? null : t.textContent;
  }, name);
const events = (page: Page) => page.evaluate(() => (window as unknown as { events: unknown[] }).events);
const sent = (page: Page, name: string) =>
  page.evaluate((n) => (window as unknown as { src: Record<string, { events: unknown[] }> }).src[n].events, name) as Promise<{ event: { type: string; detail: unknown } }[]>;

async function bodyBox(page: Page, name: string) {
  return (await page.locator(`#${name} .body`).boundingBox())!;
}

/** Page coordinates of a data point in a 2-D view (rows grow down in heatmaps). */
function toPage(d: Dbg, box: { x: number; y: number }, x: number, y: number, yDown: boolean) {
  const a = d.area, w = d.window;
  const px = a.x + ((x - w.x0) / (w.x1 - w.x0)) * a.w;
  const f = (y - w.y0) / (w.y1 - w.y0);
  const py = yDown ? a.y + f * a.h : a.y + a.h - f * a.h;
  return { x: box.x + px, y: box.y + py, bx: px, by: py };
}

async function shiftDrag(page: Page, from: { x: number; y: number }, to: { x: number; y: number }, alt = false, via: { x: number; y: number }[] = []) {
  await page.keyboard.down('Shift');
  if (alt) await page.keyboard.down('Alt');
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  for (const p of via) await page.mouse.move(p.x, p.y, { steps: 2 });
  await page.mouse.move(to.x, to.y, { steps: 4 });
  await page.mouse.up();
  if (alt) await page.keyboard.up('Alt');
  await page.keyboard.up('Shift');
}

const rgb = (lut: Uint8Array, i: number) => [lut[i * 4], lut[i * 4 + 1], lut[i * 4 + 2], 255];

// --- heatmap and image ---------------------------------------------------------

test.describe('heatmap', () => {
  const R = 4, Cn = 8;
  const values = (k: number) => Array.from({ length: R * Cn }, (_, i) => (i === R * Cn - 1 ? null : (i * k) % 31));

  test('draws every cell with the native colour table, and redraws when data changes', async ({ harness: page }, info) => {
    await mount(page, 'h', 'heatmap');
    await push(page, 'h', { data: values(1), shape: [R, Cn], vmin: 0, vmax: 31, colormap: 0 });
    const d = await dbg(page, 'h');
    expect(d.renderer).toBe('canvas2d');
    expect(d.status).toBe('connected');
    const lut = colormapLut('viridis');
    const check = async (k: number) => {
      const { px } = await bodyPixels(page, '#h');
      for (let r = 0; r < R; r++) for (let c = 0; c < Cn; c++) {
        const v = values(k)[r * Cn + c];
        const p = toPage(d, { x: 0, y: 0 }, c + 0.5, r + 0.5, true);
        expect(px.at(p.bx, p.by), `cell ${r},${c} = ${v}`).toEqual(rgb(lut, v === null ? 0 : lutIndex(v, 0, 31)));
      }
    };
    await check(1);
    await page.locator('#h').screenshot({ path: shotPath(info, 'heatmap') });
    await push(page, 'h', { data: values(7), shape: [R, Cn], vmin: 0, vmax: 31, colormap: 0 });
    await check(7);
  });

  test('takes vmin and vmax from stats when the AUTO flags are set, with magma and rdbu', async ({ harness: page }) => {
    await mount(page, 'h', 'heatmap');
    for (const [name, cm] of [['magma', 1], ['rdbu', 2]] as const) {
      await push(page, 'h', { data: [-1, 0, 1, 3], shape: [1, 4], colormap: cm, flags: 3, stats: [-1, 3], vmin: 0, vmax: 1 });
      const d = await dbg(page, 'h');
      const { px } = await bodyPixels(page, '#h');
      const lut = colormapLut(name);
      for (let c = 0; c < 4; c++) {
        const p = toPage(d, { x: 0, y: 0 }, c + 0.5, 0.5, true);
        expect(px.at(p.bx, p.by)).toEqual(rgb(lut, lutIndex([-1, 0, 1, 3][c], -1, 3)));
      }
    }
  });

  test('hover reports the cell, zoom and pan change the visible range, double-click resets', async ({ harness: page }) => {
    await mount(page, 'h', 'heatmap');
    await push(page, 'h', { data: values(1), shape: [R, Cn], vmin: 0, vmax: 31 });
    const d = await dbg(page, 'h');
    const box = await bodyBox(page, 'h');
    const p = toPage(d, box, 5.5, 2.5, true);
    await page.mouse.move(p.x, p.y);
    await expect.poll(() => tip(page, 'h')).toBe('row 2  col 5\n21');
    expect((await sent(page, 'h')).at(-1)).toMatchObject({ event: { type: 'hover', detail: { row: 2, col: 5, value: 21 } } });
    // Wheel up over the plot zooms in about the pointer, on both axes.
    await page.mouse.wheel(0, -400);
    const z = (await dbg(page, 'h')).window;
    expect(z.x1 - z.x0).toBeLessThan(Cn * 0.8);
    expect(z.y1 - z.y0).toBeLessThan(R * 0.8);
    expect(z.x0).toBeLessThan(5.5);
    expect(z.x1).toBeGreaterThan(5.5);
    // Dragging pans: content follows the pointer to the right, so the window moves left.
    await page.mouse.down();
    await page.mouse.move(p.x + 80, p.y, { steps: 4 });
    await page.mouse.up();
    const q = (await dbg(page, 'h')).window;
    expect(q.x0).toBeLessThan(z.x0);
    await page.mouse.dblclick(p.x, p.y);
    await expect.poll(async () => (await dbg(page, 'h')).window).toEqual({ x0: 0, x1: Cn, y0: 0, y1: R });
  });

  test('a drag pans from the first pixel of motion, and a press released within 2 pixels is a pick that puts the window back', async ({ harness: page }) => {
    // The rule of the app's camera press (src/host/live/camera_drag.h), which web/src/views/interact.ts follows.
    await mount(page, 'h', 'heatmap');
    await push(page, 'h', { data: values(1), shape: [R, Cn], vmin: 0, vmax: 31 });
    const d = await dbg(page, 'h');
    const box = await bodyBox(page, 'h');
    const p = toPage(d, box, 5.5, 2.5, true);
    const picks = async () => (await sent(page, 'h')).filter((e) => e.event.type === 'pick');
    const hovers = async () => (await sent(page, 'h')).filter((e) => e.event.type === 'hover').length;
    await page.mouse.move(p.x, p.y);
    await page.mouse.down();
    // One pixel of motion already pans the window by one pixel's width of data.
    await page.mouse.move(p.x + 1, p.y);
    const px = (d.window.x1 - d.window.x0) / d.area.w;
    await expect.poll(async () => (await dbg(page, 'h')).window.x0).toBeCloseTo(d.window.x0 - px, 6);
    // Released 1 pixel from the press: a pick of the cell at the press point, with the window put back.
    await page.mouse.up();
    await expect.poll(async () => (await picks()).length).toBe(1);
    expect((await picks())[0]).toMatchObject({ event: { type: 'pick', detail: { row: 2, col: 5, value: 21 } } });
    await expect.poll(async () => (await dbg(page, 'h')).window).toEqual(d.window);
    // A press that goes 3 pixels out and comes back stays a drag. The same cell does not emit another hover.
    await page.mouse.move(p.x, p.y);
    const h0 = await hovers();
    await page.mouse.down();
    await page.mouse.move(p.x + 3, p.y, { steps: 3 });
    await page.mouse.move(p.x, p.y, { steps: 3 });
    await page.mouse.up();
    await expect.poll(() => tip(page, 'h')).toBe('row 2  col 5\n21');
    expect(await hovers()).toBe(h0);
    expect((await picks()).length).toBe(1);
    const back = (await dbg(page, 'h')).window;
    expect(back.x0).toBeCloseTo(d.window.x0, 6);
    expect(back.x1).toBeCloseTo(d.window.x1, 6);
  });

  test('shift-drag selects a block of cells and reports it', async ({ harness: page }) => {
    await mount(page, 'h', 'heatmap');
    await push(page, 'h', { data: values(1), shape: [R, Cn], vmin: 0, vmax: 31 });
    const d = await dbg(page, 'h');
    const box = await bodyBox(page, 'h');
    await shiftDrag(page, toPage(d, box, 1.5, 1.5, true), toPage(d, box, 4.5, 2.5, true));
    const want = { view: 'h', type: 'select', detail: { rows: [1, 3], cols: [1, 5] } };
    expect(await events(page)).toEqual([['caliper-select', want]]);
    expect((await sent(page, 'h')).filter((e) => e.event.type === 'select').map((e) => e.event)).toEqual([want]);
    expect((await dbg(page, 'h')).selection).toEqual({ r0: 1, r1: 3, c0: 1, c1: 5 });
  });

  test('maps payload cells to source indices through the snapshot axes', async ({ harness: page }) => {
    await mount(page, 'h', 'heatmap');
    // 4 x 8 payload pooled from query rows 0..63 and key columns 10, 12, ..., 40.
    await push(page, 'h', {
      data: values(1), shape: [R, Cn], vmin: 0, vmax: 31,
      axes: [{ dim: 'query', start: 0, step: 1, count: 64, size: 4 }, { dim: 'key', start: 10, step: 2, count: 16, size: 8 }],
    });
    const d = await dbg(page, 'h');
    const box = await bodyBox(page, 'h');
    await page.mouse.move(toPage(d, box, 3.5, 1.5, true).x, toPage(d, box, 3.5, 1.5, true).y);
    await expect.poll(() => tip(page, 'h')).toBe('query 16  key 22\n11');
    await shiftDrag(page, toPage(d, box, 1.5, 1.5, true), toPage(d, box, 2.5, 2.5, true));
    expect((await events(page)).at(-1)).toEqual(['caliper-select', { view: 'h', type: 'select', detail: { rows: [16, 48], cols: [14, 21] } }]);
  });
});

test('image draws RGBA pixels as they are', async ({ harness: page }, info) => {
  await mount(page, 'img', 'image');
  const colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [240, 233, 219], [12, 34, 56], [179, 139, 89]];
  await push(page, 'img', { data: colors.flatMap((c) => [...c, 255]), shape: [2, 3, 4], u8: true });
  const d = await dbg(page, 'img');
  expect([d.kind, d.renderer]).toEqual(['image', 'canvas2d']);
  const { px } = await bodyPixels(page, '#img');
  for (let i = 0; i < 6; i++) {
    const p = toPage(d, { x: 0, y: 0 }, (i % 3) + 0.5, Math.floor(i / 3) + 0.5, true);
    expect(px.at(p.bx, p.by)).toEqual([...colors[i], 255]);
  }
  await page.locator('#img').screenshot({ path: shotPath(info, 'image') });
});

// --- series --------------------------------------------------------------------

test.describe('series', () => {
  test('draws the curve, follows new values, and redraws when they arrive', async ({ harness: page }, info) => {
    await mount(page, 's', 'series');
    await series(page, 's', 0, { n: 1000, f: '1 + Math.sin(i / 40) * Math.exp(-i / 800)' });
    let d = await dbg(page, 's');
    expect(d.renderer).toBe('canvas2d');
    expect([d.window.x0, d.window.x1]).toEqual([0, 999]);
    const before = await bodyPixels(page, '#s');
    // The curve is drawn in parchment somewhere in each tenth of the plot.
    const a = d.area;
    for (let k = 0; k < 10; k++) {
      let found = false;
      const x = Math.floor(a.x + (k + 0.5) * (a.w / 10));
      for (let y = Math.ceil(a.y); y < a.y + a.h && !found; y++) {
        const [r, g, b] = before.px.at(x, y);
        found = r > 200 && g > 190 && b > 170;
      }
      expect(found, `curve at column ${x}`).toBe(true);
    }
    await page.locator('#s').screenshot({ path: shotPath(info, 'series') });
    await series(page, 's', 1000, { n: 500, f: '3' });
    d = await dbg(page, 's');
    expect(d.window.x1).toBe(1499);
    const after = await bodyPixels(page, '#s');
    expect(Buffer.compare(Buffer.from(before.px.data), Buffer.from(after.px.data))).not.toBe(0);
  });

  test('decimates 100,000 values to a few vertices per pixel column', async ({ harness: page }, info) => {
    await mount(page, 's', 'series');
    const t = await page.evaluate(async () => {
      const w = window as unknown as { src: Record<string, { series(v: string, s: number, x: Float32Array): void }> };
      const el = document.getElementById('s') as HTMLElement & { settled(): Promise<void> };
      const v = new Float32Array(100_000).map((_, i) => Math.sin(i / 300) + 0.3 * Math.sin(i * 1.7));
      const t0 = performance.now();
      w.src.s.series('s', 0, v);
      await el.settled();
      return performance.now() - t0;
    });
    const d = await dbg(page, 's');
    expect((d.steps as { hi: number }).hi).toBe(100_000);
    // Decimated: the curve the view drew has at most the first, lowest,
    // highest and last value of each pixel column (M4), not 100,000 vertices.
    expect(d.vertices as number).toBeLessThanOrEqual(4 * Math.round(d.area.w) + 8);
    expect(d.vertices as number).toBeGreaterThan(Math.round(d.area.w));
    console.log(`${info.project.name} series of 100000 values: ${d.vertices} vertices over ${Math.round(d.area.w)} columns, drawn in ${t.toFixed(1)} ms`);
    // The time is a measurement. It bounds the draw only on this machine's
    // GPU and Canvas 2D paths and outside CI: a hosted runner, and SwiftShader
    // anywhere, draws slower with nothing wrong.
    if (!process.env.CI && info.project.name !== 'swiftshader') expect(t).toBeLessThan(250);
  });

  test('hover shows step and value; zoom stops following; double-click follows again', async ({ harness: page }) => {
    await mount(page, 's', 'series');
    await series(page, 's', 0, { n: 1000, f: 'i / 4' });
    const d = await dbg(page, 's');
    const box = await bodyBox(page, 's');
    const p = toPage(d, box, 500, 125, false);
    await page.mouse.move(p.x, p.y);
    await expect.poll(() => tip(page, 's')).toBe('step 500\ns 125');
    expect((await sent(page, 's')).at(-1)).toMatchObject({ event: { type: 'hover', detail: { step: 500, value: 125 } } });
    await page.mouse.wheel(0, -500);
    let z = await dbg(page, 's');
    expect(z.follow).toBe(false);
    expect(z.window.x1 - z.window.x0).toBeLessThan(800);
    await series(page, 's', 1000, { n: 100, f: 'i / 4' });
    z = await dbg(page, 's');
    expect(z.window.x1).toBeLessThan(1000);
    await page.mouse.dblclick(p.x, p.y);
    await expect.poll(async () => (await dbg(page, 's')).window.x1).toBe(1099);
    expect((await dbg(page, 's')).follow).toBe(true);
  });

  test('the log toggle puts y on a log scale', async ({ harness: page }, info) => {
    await mount(page, 's', 'series');
    await series(page, 's', 0, { n: 400, f: 'Math.exp(-i / 50) + 1e-3' });
    await page.locator('#s .tool', { hasText: 'log' }).click();
    await settled(page, 's');
    const d = await dbg(page, 's');
    expect(d.logY).toBe(true);
    expect(d.window.y0).toBeGreaterThan(0);
    expect(Math.log10(d.window.y1 / d.window.y0)).toBeGreaterThan(2.5);
    await page.locator('#s').screenshot({ path: shotPath(info, 'series-log') });
  });

  test('shift-drag selects a range of steps', async ({ harness: page }) => {
    await mount(page, 's', 'series');
    await series(page, 's', 0, { n: 1000, f: 'i' });
    const d = await dbg(page, 's');
    const box = await bodyBox(page, 's');
    await shiftDrag(page, toPage(d, box, 200.2, 500, false), toPage(d, box, 399.8, 500, false));
    const ev = (await events(page)).at(-1) as [string, { detail: { steps: number[] } }];
    expect(ev[0]).toBe('caliper-select');
    expect(ev[1].detail.steps[0]).toBeGreaterThanOrEqual(200);
    expect(ev[1].detail.steps[0]).toBeLessThanOrEqual(202);
    expect(ev[1].detail.steps[1]).toBeGreaterThanOrEqual(399);
    expect(ev[1].detail.steps[1]).toBeLessThanOrEqual(401);
  });
});

// --- points --------------------------------------------------------------------

test.describe('points', () => {
  const corners = { data: [0, 0, 1, 0, 0, 1, 1, 1], aux: [0, 1, 2, 3], shape: [4, 2], points_dims: 2, label_count: 4 };

  test('draws 2-D points in their class colours where they project', async ({ harness: page }, info) => {
    await mount(page, 'p', 'points');
    await push(page, 'p', corners);
    const d = await dbg(page, 'p');
    expect([d.renderer, d.n, d.dims]).toEqual(['canvas2d', 4, 2]);
    const { px } = await bodyPixels(page, '#p');
    for (let i = 0; i < 4; i++) {
      const p = toPage(d, { x: 0, y: 0 }, corners.data[2 * i], corners.data[2 * i + 1], false);
      expect(px.at(p.bx, p.by), `point ${i}`).toEqual([...PALETTE[i], 255]);
    }
    await page.locator('#p').screenshot({ path: shotPath(info, 'points-2d') });
  });

  test('hover picks the nearest point; zoom changes the window', async ({ harness: page }) => {
    await mount(page, 'p', 'points');
    await push(page, 'p', { ...corners, axes: [{ dim: 'sample', start: 100, step: 10, count: 40, size: 4 }] });
    const d = await dbg(page, 'p');
    const box = await bodyBox(page, 'p');
    const p = toPage(d, box, 0, 1, false);
    await page.mouse.move(p.x + 4, p.y + 3);
    await expect.poll(() => tip(page, 'p')).toBe('point 2  row 300\nlabel 2\nx 0  y 1');
    expect((await sent(page, 'p')).at(-1)).toMatchObject({ event: { type: 'hover', detail: { index: 2, row_id: 300 } } });
    await page.mouse.wheel(0, -300);
    const z = (await dbg(page, 'p')).window;
    expect(z.x1 - z.x0).toBeLessThan(d.window.x1 - d.window.x0);
  });

  test('shift-alt-drag selects a rectangle and shift-drag a lasso', async ({ harness: page }) => {
    await mount(page, 'p', 'points');
    await push(page, 'p', { ...corners, axes: [{ dim: 'sample', start: 0, step: 3, count: 4, size: 4 }] });
    const d = await dbg(page, 'p');
    const box = await bodyBox(page, 'p');
    await shiftDrag(page, toPage(d, box, 0.8, -0.04, false), toPage(d, box, 1.2, 1.04, false), true);
    expect((await events(page)).at(-1)).toEqual(['caliper-select', { view: 'p', type: 'select', detail: { indices: [1, 3], row_ids: [3, 9] } }]);
    expect((await dbg(page, 'p')).selected).toBe(2);
    const at = (x: number, y: number) => toPage(d, box, x, y, false);
    await shiftDrag(page, at(-0.2, -0.04), at(-0.2, -0.04), false, [at(0.3, -0.04), at(0.3, 0.3), at(-0.2, 0.3)]);
    expect((await events(page)).at(-1)).toEqual(['caliper-select', { view: 'p', type: 'select', detail: { indices: [0], row_ids: [0] } }]);
    // A shift-click clears the selection with an empty detail.
    await shiftDrag(page, at(0.5, 0.5), at(0.5, 0.5));
    expect((await events(page)).at(-1)).toEqual(['caliper-select', { view: 'p', type: 'select', detail: {} }]);
  });

  test('3-D points orbit with a drag and colour by value through the colormap', async ({ harness: page }, info) => {
    await mount(page, 'p3', 'points');
    const n = 2000;
    const data: number[] = [], aux: number[] = [];
    for (let i = 0; i < n; i++) {
      const t = i / n * Math.PI * 8;
      data.push(Math.cos(t) * (1 + i / n), Math.sin(t) * (1 + i / n), i / n * 2 - 1);
      aux.push(i / n);
    }
    await push(page, 'p3', { data, aux, shape: [n, 3], points_dims: 3, label_count: 0, colormap: 1, vmin: 0, vmax: 1 });
    const d = await dbg(page, 'p3');
    expect([d.renderer, d.dims]).toEqual(['canvas2d', 3]);
    const before = await bodyPixels(page, '#p3');
    // Colours come from magma: some drawn pixel is magma's bright end.
    const lut = colormapLut('magma');
    const bright = rgb(lut, 255);
    let seen = false;
    for (let i = 0; i < before.px.data.length && !seen; i += 4) seen = before.px.data[i] === bright[0] && before.px.data[i + 1] === bright[1] && before.px.data[i + 2] === bright[2];
    expect(seen).toBe(true);
    await page.locator('#p3').screenshot({ path: shotPath(info, 'points-3d') });
    const box = await bodyBox(page, 'p3');
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 + 120, box.y + box.height / 2 + 30, { steps: 5 });
    await page.mouse.up();
    await settled(page, 'p3');
    const cam = (await dbg(page, 'p3')).camera as { az: number; el: number };
    expect(cam.az).toBeGreaterThan((d.camera as { az: number }).az + 0.5);
    const after = await bodyPixels(page, '#p3');
    expect(Buffer.compare(Buffer.from(before.px.data), Buffer.from(after.px.data))).not.toBe(0);
  });

  test('3-D points turn from the first pixel of a drag, and a click within 2 pixels puts the camera back', async ({ harness: page }) => {
    await mount(page, 'p3', 'points');
    const n = 500;
    const data: number[] = [];
    for (let i = 0; i < n; i++) data.push(Math.cos(i), Math.sin(i), i / n * 2 - 1);
    await push(page, 'p3', { data, shape: [n, 3], points_dims: 3, label_count: 0 });
    const cam0 = (await dbg(page, 'p3')).camera as { az: number; el: number; dist: number };
    const box = await bodyBox(page, 'p3');
    const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
    await page.mouse.move(cx, cy);
    await page.mouse.down();
    await page.mouse.move(cx + 1, cy + 1);
    await expect.poll(async () => ((await dbg(page, 'p3')).camera as { az: number }).az).toBeCloseTo(cam0.az + 0.01, 6);
    await page.mouse.up();
    await expect.poll(async () => (await dbg(page, 'p3')).camera).toEqual(cam0);
  });
});

// --- histogram -----------------------------------------------------------------

test('histogram draws bars, hovers a bin, toggles log counts, zooms and selects', async ({ harness: page }, info) => {
  await mount(page, 'hist', 'histogram');
  const counts = [1, 2, 4, 8, 16, 32, 16, 8, 4, 0];
  await push(page, 'hist', { data: counts, shape: [10], stats: [-5, 5] });
  const d = await dbg(page, 'hist');
  expect([d.renderer, d.bins, d.range]).toEqual(['canvas2d', 10, [-5, 5]]);
  const { px } = await bodyPixels(page, '#hist');
  const bar = toPage(d, { x: 0, y: 0 }, 0.5, 1, false);
  expect(px.at(bar.bx, bar.by)).toEqual([0xd0, 0xc9, 0xba, 255]);
  const empty = toPage(d, { x: 0, y: 0 }, 4.5, 1, false);
  expect(px.at(empty.bx, empty.by)).toEqual([0x27, 0x26, 0x22, 255]);
  await page.locator('#hist').screenshot({ path: shotPath(info, 'histogram') });
  const box = await bodyBox(page, 'hist');
  const p = toPage(d, box, -1.5, 3, false);
  await page.mouse.move(p.x, p.y);
  await expect.poll(() => tip(page, 'hist')).toBe('bin 3  [-2, -1)\ncount 8');
  expect((await sent(page, 'hist')).at(-1)).toMatchObject({ event: { type: 'hover', detail: { bin: 3, lo: -2, hi: -1, count: 8 } } });
  await page.locator('#hist .tool', { hasText: 'log' }).click();
  await settled(page, 'hist');
  expect((await dbg(page, 'hist')).logY).toBe(true);
  await shiftDrag(page, toPage(d, box, -2.7, 3, false), toPage(d, box, 0.4, 3, false));
  expect((await events(page)).at(-1)).toEqual(['caliper-select', { view: 'hist', type: 'select', detail: { range: [-3, 1] } }]);
  await page.mouse.move(p.x, p.y);
  await page.mouse.wheel(0, -400);
  const z = (await dbg(page, 'hist')).window;
  expect(z.x1 - z.x0).toBeLessThan(8);
});

// --- the header: picker and status ---------------------------------------------

test('the picker sends the data view chosen in the header', async ({ harness: page }, info) => {
  const dims = [{ name: 'layer', size: 2 }, { name: 'head', size: 4 }, { name: 'query', size: 8 }, { name: 'key', size: 8 }];
  await mount(page, 'a', 'heatmap', [{ name: 'attn', dims }, { name: 'scores', dims }], { source: 'attn', index: { layer: 0, head: 0 } });
  await push(page, 'a', { data: new Array(64).fill(0).map((_, i) => i), shape: [8, 8], vmin: 0, vmax: 63, data_view: { source: 'attn', index: { layer: 0, head: 0 }, reduce: {} } });
  const choose = (sel: string, value: string) => page.evaluate(({ sel, value }) => {
    const s = document.getElementById('a')!.shadowRoot!.querySelector(sel) as HTMLSelectElement;
    s.value = value;
    s.dispatchEvent(new Event('change'));
  }, { sel, value });
  const requests = () => page.evaluate(() => (window as unknown as { src: Record<string, { requests: { dataView: unknown }[] }> }).src.a.requests.map((r) => r.dataView));
  expect(await page.evaluate(() => [...document.getElementById('a')!.shadowRoot!.querySelectorAll('.picker select')].map((s) => `${s.getAttribute('aria-label')}=${(s as HTMLSelectElement).value}`)))
    .toEqual(['source=attn', 'layer=i:0', 'head=i:0', 'query=all', 'key=all']);
  await page.locator('#a').screenshot({ path: shotPath(info, 'picker') });
  await choose('select[data-dim="head"]', 'i:3');
  await choose('select[data-dim="head"]', 'r:mean');
  await choose('select[data-dim="head"]', 'all');
  await choose('select.source', 'scores');
  const want = [
    { source: 'attn', index: { layer: 0, head: 3 }, reduce: {} },
    { source: 'attn', index: { layer: 0 }, reduce: { head: 'mean' } },
    { source: 'attn', index: { layer: 0 }, reduce: {}, tile: 'head' },
    { source: 'scores', index: { layer: 0 }, reduce: {}, tile: 'head' },
  ];
  expect(await requests()).toEqual(want);
  expect((await events(page)).map((e) => (e as [string, unknown])[1])).toEqual(want);
  // The producer's reply is what the picker shows; a refused request is reported.
  await page.evaluate(() => (window as unknown as { src: Record<string, { reply(d: unknown): void }> }).src.a.reply({ source: 'scores', index: { layer: 1, head: 2 }, reduce: {} }));
  expect(await page.evaluate(() => [...document.getElementById('a')!.shadowRoot!.querySelectorAll('.picker select')].map((s) => (s as HTMLSelectElement).value)))
    .toEqual(['scores', 'i:1', 'i:2', 'all', 'all']);
  await choose('select[data-dim="query"]', 'i:0');
  await page.evaluate(() => (window as unknown as { src: Record<string, { reply(d: unknown, e: string): void }> }).src.a.reply(null, 'kind 1 needs 2 kept dims'));
  await expect(page.locator('#a .msg.error')).toHaveText('kind 1 needs 2 kept dims');
  expect(await page.evaluate(() => (document.getElementById('a')!.shadowRoot!.querySelector('select[data-dim="query"]') as HTMLSelectElement).value)).toBe('all');
});

test('the status dot shows waiting, connected and ended', async ({ harness: page }) => {
  await mount(page, 'h', 'heatmap');
  const dot = () => page.evaluate(() => (document.getElementById('h')!.shadowRoot!.querySelector('.dot') as HTMLElement).dataset.state);
  expect(await dot()).toBe('waiting');
  await expect(page.locator('#h .msg')).toHaveText('Waiting for data');
  await push(page, 'h', { data: [0, 1], shape: [1, 2] });
  expect(await dot()).toBe('connected');
  await page.evaluate(() => (window as unknown as { src: Record<string, { close(): void }> }).src.h.close());
  expect(await dot()).toBe('ended');
});
