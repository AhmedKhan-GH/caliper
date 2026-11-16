// The spec page against a mock of the app (spec-mock.ts): the dashboard with
// its layout, titles, info strings and colour key; every primitive drawn from
// the forms panel.data sends; the states in place of data; tabs; the page's
// requests, which carry the token and the viewer origin and send the version
// they have; and the list of applets when the address names none. Headless
// Chromium with Canvas 2D, in one project.

import { test, expect, type Page } from '@playwright/test';
import {
  DASHBOARD, ORIGIN, TOKEN, dashboardData, f32, f64, mockApp, openSpec, pageState, pixel, ready, rng, settled,
  type Json, type MockApp,
} from './spec-mock.js';
import { shotPath } from './util.js';

test.beforeEach(({}, info) => test.skip(info.project.name !== 'swiftshader', 'the spec page draws with Canvas 2D: one project is enough'));

function app(specs: Record<string, Json>, data: Record<string, Record<string, Json>>): MockApp {
  return { specs, data, requests: [] };
}

async function noErrors(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  return errors;
}

const hex = (h: string) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
const near = (a: number[], b: number[], tol = 2) => a.slice(0, 3).every((v, i) => Math.abs(v - b[i]) <= tol);

test('the dashboard lays out as the native runtime does, with its titles, info strings and colour key', async ({ page }, info) => {
  const errors = await noErrors(page);
  const mock = app({ 'run42-dashboard': DASHBOARD }, { 'run42-dashboard': dashboardData() });
  await mockApp(page, mock);
  await page.setViewportSize({ width: 1440, height: 900 });
  await openSpec(page, 'run42-dashboard');
  const s = await settled(page, 3);
  expect(s.title).toBe('Run 42');
  await expect(page.locator('#title')).toHaveText('Run 42');
  await expect(page).toHaveTitle('Run 42 · Caliper');
  expect(s.panels.map((p) => [p.id, p.title, p.info])).toEqual([
    ['curves', 'Loss and accuracy', '4,000 steps  2 lines'],
    ['attention', 'Mean attention of layer 0', '48×48  0 … 1'],
    ['rows', 'run42_metrics', '1,000 of 4,000 rows'],
  ]);
  expect(s.panels.map((p) => p.rect)).toEqual([
    { x: 0, y: 0, w: 720, h: 552 }, { x: 721, y: 0, w: 719, h: 552 }, { x: 0, y: 553, w: 1440, h: 297 }]);
  await expect(page.locator('#status')).toHaveText('connected  ·  applet run42-dashboard  ·  3 panels  ·  2 sources');
  // The rules between the panels, the title strips on the plate, the plot field deep.
  expect(near(await pixel(page, 720, 300), hex('#4b4945'))).toBe(true);
  expect(near(await pixel(page, 300, 552), hex('#4b4945'))).toBe(true);
  expect(near(await pixel(page, 400, 3), hex('#201f1c'))).toBe(true);
  expect(near(await pixel(page, 400, 23), hex('#2c2b28'))).toBe(true);
  // The heatmap's first cell is magma's top (1.0), a cell above the diagonal its bottom (0).
  const heat = s.panels[1].rect;
  const magmaTop = [252, 253, 191], magmaBottom = [0, 0, 4];
  expect(near(await pixel(page, heat.x + 8 + 46 * 13 / 10.5 + 2, 24 + 6 + 16 * 13 / 10.5 + 2), magmaTop, 6)).toBe(true);
  expect(near(await pixel(page, heat.x + heat.w - 120, 60), magmaBottom, 6)).toBe(true);
  // The colour key: the legend's swatches and the table's head carry the house pigments.
  const shot = await page.screenshot({ path: shotPath(info, 'spec-dashboard-1440x900') });
  expect(shot.length).toBeGreaterThan(10_000);
  const keyed = await page.evaluate(() => {
    const c = document.getElementById('canvas') as HTMLCanvasElement;
    const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data;
    let blue = 0, teal = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (Math.abs(d[i] - 0x46) < 3 && Math.abs(d[i + 1] - 0x77) < 3 && Math.abs(d[i + 2] - 0xad) < 3) blue++;
      if (Math.abs(d[i] - 0x3c) < 3 && Math.abs(d[i + 1] - 0xa9) < 3 && Math.abs(d[i + 2] - 0x9c) < 3) teal++;
    }
    return { blue, teal };
  });
  expect(keyed.blue).toBeGreaterThan(500);
  expect(keyed.teal).toBeGreaterThan(500);
  expect(errors).toEqual([]);
});

test('the page asks with the token and the viewer origin, and sends the version it has', async ({ page }) => {
  const mock = app({ 'run42-dashboard': DASHBOARD }, { 'run42-dashboard': dashboardData() });
  await mockApp(page, mock);
  await openSpec(page, 'run42-dashboard');
  await settled(page, 3);
  await expect.poll(() => mock.requests.filter((r) => r.name === 'panel.data' && r.args.known === 'v1').length,
    { timeout: 15_000 }).toBeGreaterThanOrEqual(3);
  for (const r of mock.requests) {
    expect(r.headers['authorization']).toBe(`Bearer ${TOKEN}`);
    expect(r.headers['x-caliper-origin']).toBe('viewer');
    expect(['spec.get', 'panel.data']).toContain(r.name);
    expect(r.args.applet).toBe('run42-dashboard');
  }
  // A new version is drawn: the table's info follows it.
  const d = mock.data['run42-dashboard'];
  d.rows = { ...d.rows, version: 'v2', data: { ...d.rows.data, rows: d.rows.data.rows.slice(0, 10), total_rows: 10 } };
  await expect.poll(async () => (await pageState(page)).panels[2].info, { timeout: 15_000 }).toBe('10 rows');
});

/** A spec of every primitive in two tabs, with data in the forms the app sends. */
function everyPrimitive(): { spec: Json; data: Record<string, Json> } {
  const spec = {
    caliper_spec: 0,
    applet: { id: 'every', title: 'Every primitive' },
    source: [
      { id: 'acts', catalog: 'tensor:acts' }, { id: 'img', catalog: 'tensor:sample_png' },
      { id: 'cloud', catalog: 'tensor:embedding' }, { id: 'labels', catalog: 'tensor:embedding_labels' },
      { id: 'm', catalog: 'table:run42_metrics' }, { id: 'live', catalog: 'live:run/attn' },
    ],
    panel: [
      { id: 's', primitive: 'series', bind: { y: 'acts' }, options: { y_label: 'norm' } },
      { id: 'h', primitive: 'histogram', bind: { values: 'acts' }, options: { bins: 32, y_log: true } },
      { id: 'p2', primitive: 'points', bind: { x: 'm.step', y: 'm.loss', color: 'm.label' } },
      { id: 'p3', primitive: 'points', bind: { positions: 'cloud', color: 'labels' } },
      { id: 'im', primitive: 'image', bind: { values: 'img' } },
      { id: 't', primitive: 'table', bind: { rows: 'm' } },
      { id: 'x', primitive: 'text', title: 'Last loss', bind: { value: 'm.loss' }, options: { text: 'Loss' } },
      { id: 'off', primitive: 'heatmap', bind: { values: 'live' } },
    ],
    layout: { tabs: [
      { split: 'horizontal', title: 'Tensors', children: ['s', 'h', { split: 'vertical', children: ['p2', 'p3'] }] },
      { split: 'horizontal', title: 'Tables', children: ['im', 't', { split: 'vertical', children: ['x', 'off'] }] },
    ] },
  };
  const r = rng(7);
  const n = 300, ys: number[] = [];
  for (let i = 0; i < n; i++) ys.push(1 + Math.sin(i / 20) * 0.5 + r() * 0.1);
  const counts: number[] = [];
  for (let i = 0; i < 32; i++) counts.push(i === 3 ? 1 : Math.round(200 * Math.exp(-((i - 16) ** 2) / 40)));
  const px: number[] = [], py: number[] = [], pc: number[] = [];
  for (let i = 0; i < 500; i++) {
    px.push(i);
    py.push(Math.cos(i / 40) + r() * 0.2);
    pc.push(i % 4);
  }
  const cx: number[] = [], cc: number[] = [];
  for (let i = 0; i < 400; i++) {
    const k = i % 3;
    cx.push(k * 3 + r(), k * 2 + r(), -k + r());
    cc.push(k);
  }
  const H = 24, W = 32, rgb: number[] = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) rgb.push((x * 8) % 256, (y * 10) % 256, 128);
  const rows: (number | string)[][] = [];
  for (let i = 0; i < 40; i++) rows.push([i, 1 / (i + 1), i % 4, `run-${i}`]);
  const data = {
    s: ready('s', 'series', { x_index: true, steps: n, lines: [{ binding: 'acts', row: null, count: n, last: ys[n - 1],
      x: f64(ys.map((_, i) => i)), y: f64(ys) }] }),
    h: ready('h', 'histogram', { lo: -4, hi: 4, total: counts.reduce((a, b) => a + b, 0), outside: 0, counts }),
    p2: ready('p2', 'points', { three_d: false, total: 500, stride: 1, x: f32(px), y: f32(py), z: null, color: f32(pc),
      color_integer: true, size: null }),
    p3: ready('p3', 'points', { three_d: true, total: 400, stride: 1, x: f32(cx.filter((_, i) => i % 3 === 0)),
      y: f32(cx.filter((_, i) => i % 3 === 1)), z: f32(cx.filter((_, i) => i % 3 === 2)), color: f32(cc), color_integer: false, size: null }),
    im: ready('im', 'image', { source_rows: H, source_cols: W, stride: 1, channels: 3, float: false, values: f32(rgb, [H, W, 3]) }),
    t: ready('t', 'table', { columns: [{ name: 'step', type: 'BIGINT' }, { name: 'loss', type: 'DOUBLE' }, { name: 'label', type: 'INTEGER' },
      { name: 'run', type: 'VARCHAR' }], rows, total_rows: 40, truncated: false }),
    x: ready('x', 'text', { value: 0.0123456 }),
    off: { panel: 'off', primitive: 'heatmap', version: 'v1', unchanged: false, state: null, bounds: [], data: null,
      error: { code: 'resolve.unavailable', message: 'No live session is attached.', where: 'live' } },
  };
  return { spec, data };
}

test('every primitive draws from the forms the app sends, with tabs', async ({ page }, info) => {
  const errors = await noErrors(page);
  const { spec, data } = everyPrimitive();
  await mockApp(page, app({ every: spec }, { every: data }));
  await page.setViewportSize({ width: 1440, height: 900 });
  await openSpec(page, 'every');
  let s = await settled(page, 4);
  expect(s.tabs).toEqual([{ path: '', labels: ['Tensors', 'Tables'], active: 0 }]);
  const four = (v: number) => String(Number(v.toPrecision(4)));
  expect(s.panels.map((p) => [p.id, p.title, p.info])).toEqual([
    ['s', 'acts', `300 steps  last ${four(data.s.data.lines[0].last)}`],
    ['h', 'acts', `32 bins  -4 … 4  n ${four(data.h.data.total)}`],
    ['p2', 'loss by step', '500 points  2-D  4 classes'],
    ['p3', 'embedding', '400 points  3-D'],
  ]);
  // A histogram on a log axis shows a bin of one value as a bar.
  await page.screenshot({ path: shotPath(info, 'spec-every-tab0') });
  // The second tab, by a click on its label.
  await page.mouse.click(100, 28 + 12);
  s = await settled(page, 4);
  expect(s.tabs[0].active).toBe(1);
  expect(s.panels.map((p) => [p.id, p.title, p.state ?? p.info])).toEqual([
    ['im', 'sample_png', '24×32 RGB'],
    ['t', 'run42_metrics', '40 rows'],
    ['x', 'Last loss', ''],
    ['off', 'run/attn', 'Source offline'],
  ]);
  await page.screenshot({ path: shotPath(info, 'spec-every-tab1') });
  // The image's first pixel is its first value: rgb(0, 0, 128).
  const im = s.panels[0].rect;
  const area = { x: im.x + 8, y: im.y + 24 + 6, w: im.w - 16, h: im.h - 24 - 12 };
  const k = 13 / 10.5, ml = 46 * k, mt = 16 * k, mb = 32 * k, mr = 10 * k;
  const aw = area.w - ml - mr, ah = area.h - mt - mb, kf = Math.min(aw / 32, ah / 24);
  const ax = Math.round(area.x + ml + Math.floor((aw - 32 * kf) / 2)), ay = Math.round(area.y + mt + Math.floor((ah - 24 * kf) / 2));
  expect(near(await pixel(page, ax + 2, ay + 2), [0, 0, 128], 3)).toBe(true);
  expect(errors).toEqual([]);
});

test('states take the place of data: offline, mismatched, empty, a missing applet and panel', async ({ page }) => {
  const spec = {
    caliper_spec: 0, applet: { id: 'st' },
    source: [{ id: 'live', catalog: 'live:run/attn' }, { id: 'q', sql: 'SELECT step FROM run42_metrics WHERE step > 9' },
      { id: 'w', catalog: 'tensor:attn_weights' }, { id: 'd', from: 'w' }],
    panel: [
      { id: 'a', primitive: 'heatmap', bind: { values: 'live' } },
      { id: 'b', primitive: 'histogram', bind: { values: 'q.step' } },
      { id: 'c', primitive: 'heatmap', bind: { values: 'd' } },
      { id: 'e', primitive: 'table', bind: { rows: 'q' } },
    ],
  };
  const st = (panel: string, extra: Json) => ({ panel, primitive: 'x', version: 'v', unchanged: false, error: null, state: null,
    bounds: [], data: null, ...extra });
  await mockApp(page, app({ st: spec }, { st: {
    a: st('a', { error: { code: 'resolve.unavailable', message: '', where: 'live' } }),
    b: st('b', { state: { kind: 'empty', detail: 'q.step' } }),
    c: st('c', { state: { kind: 'mismatch', detail: 'd' } }),
    e: st('e', { error: { code: 'resolve.query_failed', message: '', where: 'q' } }),
  } }));
  await openSpec(page, 'st');
  const s = await settled(page, 4);
  // Stacked with equal shares, as a spec without a layout is, over the page's 850 px.
  expect(s.panels.map((p) => p.rect.h)).toEqual([212, 212, 211, 212]);
  expect(s.panels.map((p) => [p.title, p.state])).toEqual([
    ['run/attn', 'Source offline'], ['step', 'No data'], ['d', 'Mismatched data'], ['run42_metrics', 'Query error']]);
  await openSpec(page, 'gone');
  await expect.poll(async () => (await pageState(page)).state, { timeout: 15_000 }).toBe('Missing applet');
  expect((await pageState(page)).panels).toEqual([]);
  await expect(page.locator('#status')).toHaveText('connected');
});

test('without an applet the page lists the saved spec applets as links', async ({ page }) => {
  await mockApp(page, app({ 'run42-dashboard': DASHBOARD, every: everyPrimitive().spec }, {}));
  await page.goto(`${ORIGIN}/v0/spec?token=${TOKEN}`);
  const links = page.locator('#list a');
  await expect(links).toHaveCount(2, { timeout: 15_000 });
  await expect(links.nth(0)).toHaveText('Run 42');
  expect(await links.nth(0).getAttribute('href')).toBe(`${ORIGIN}/v0/spec?token=${TOKEN}&applet=run42-dashboard`);
  await expect(page.locator('#canvas')).toBeHidden();
});

test('a refused token draws nothing of the applet', async ({ page }) => {
  await mockApp(page, app({ 'run42-dashboard': DASHBOARD }, { 'run42-dashboard': dashboardData() }));
  await page.goto(`${ORIGIN}/v0/spec?token=nope&applet=run42-dashboard`);
  await expect.poll(async () => (await pageState(page)).state, { timeout: 15_000 }).toBe('Access refused');
  expect((await pageState(page)).panels).toEqual([]);
});
