// The spec page on a dashboard of the MNIST probe dashboard's shape (the layout, sources
// and control that the MNIST dashboard tests build), against the mock of the app: the
// layout fills the window over the row of its control (the native layout's control rows),
// the chart's eight lines take the app's palette in the app's order under their sources'
// names with a legend, and the toggle shows the run's state, disabled, with the reason on
// a hover.

import { test, expect } from '@playwright/test';
import { f64, mockApp, openSpec, pixel, ready, rng, type Json, type MockApp } from './spec-mock.js';
import { CATEGORICAL, HOUSE_PIGMENTS } from '../src/render/palette.gen.js';

test.beforeEach(({}, info) => test.skip(info.project.name !== 'swiftshader', 'the spec page draws with Canvas 2D: one project is enough'));

const LINES = ['digit_0', 'digit_1', 'digit_2', 'digit_3', 'digit_4', 'digit_5', 'digit_6', 'digit_7'];

export const MNIST_DASHBOARD: Json = {
  caliper_spec: 2,
  instrument: { id: 'mnist_dashboard', title: 'MNIST Probe Dashboard' },
  state: [{ id: 'digit', type: 'integer', default: 0, min: 0, max: 9, title: 'Digit' }],
  source: [
    { id: 'loss', catalog: 'live:loss', history: 4096 },
    { id: 'loss_window', from: 'loss', transform: [{ op: 'window', steps: 4096 }] },
    { id: 'acc', catalog: 'live:accuracy', history: 4096 },
    { id: 'acc_window', from: 'acc', transform: [{ op: 'window', steps: 4096 }] },
    ...LINES.flatMap((l, idx) => [{ id: `${l}_raw`, catalog: 'live:probe.logits', history: 4096,
      transform: [{ op: 'slice', axis: 1, index: idx }, { op: 'reduce', axis: 0, fn: 'mean' }] },
      { id: l, from: `${l}_raw`, transform: [{ op: 'window', steps: 4096 }] }]),
  ],
  panel: [
    { id: 'digits', primitive: 'scene', title: 'Digit Accuracy Ring', options: { axes: false, up: '+y', select: { state: 'digit', field: 'digit' } } },
    { id: 'samples', primitive: 'scene', title: 'Probe Digits' },
    { id: 'loss_curve', primitive: 'series', title: 'Loss', bind: { x: 'loss_window.step', y: 'loss_window.value' },
      options: { x_label: 'Step', y_label: 'Loss' } },
    { id: 'acc_curve', primitive: 'series', title: 'Accuracy', bind: { x: 'acc_window.step', y: 'acc_window.value' },
      options: { x_label: 'Step', y_label: 'Accuracy' } },
    { id: 'class_scores', primitive: 'series', title: 'Probe Log-Probability by Digit',
      bind: { x: 'digit_0.step', y: LINES.map((l) => `${l}.value`) }, options: { x_label: 'Step', y_label: 'Mean Log-Prob', legend: true } },
  ],
  control: [{ id: 'ablate', kind: 'toggle', title: 'Ablate Filter conv1#3', bind: 'intervene:conv1#3', options: { value: false } }],
  layout: { split: 'vertical', children: [
    { split: 'horizontal', ratio: 0.5, children: [
      { split: 'vertical', ratio: 0.45, children: ['digits', { split: 'vertical', ratio: 0.62, children: ['class_scores',
        { tabs: ['acc_curve', 'loss_curve'] }] }] }, 'samples'] }, 'ablate'] },
};

/** panel.data of the dashboard: the window sources' lines as the app names them, and the toggle's state. */
export function mnistDashboardData(): Record<string, Json> {
  const r = rng(7), n = 60;
  const step = Array.from({ length: n }, (_, i) => 10 + 9 * i);
  const line = (binding: string, name: string, f: (i: number) => number) => {
    const y = step.map((_, i) => f(i));
    return { binding, name, row: null, count: n, last: y[n - 1], x: f64(step), y: f64(y) };
  };
  const series = (id: string, lines: Json[]) => ready(id, 'series', { x_index: false, x_name: lines[0].name, steps: n, lines });
  return {
    class_scores: series('class_scores', LINES.map((l, k) => line(`${l}.value`, l, (i) => 0.03 + 0.005 * k + (k === 0 ? i / 1500 : 0) + 0.004 * r()))),
    acc_curve: series('acc_curve', [line('acc_window.value', 'acc_window', (i) => 0.2 + 0.7 * Math.tanh(i / 12) + 0.02 * r())]),
    loss_curve: series('loss_curve', [line('loss_window.value', 'loss_window', (i) => 2.3 - 1.8 * Math.tanh(i / 20))]),
    digits: ready('digits', 'scene', { scene: null, programs: [] }),
    samples: ready('samples', 'scene', { scene: null, programs: [] }),
    ablate: { panel: 'ablate', primitive: 'control', version: 'c1', unchanged: false, error: null, state: null, bounds: [],
      data: { id: 'ablate', kind: 'toggle', title: 'Ablate Filter conv1#3', bind: 'intervene:conv1#3', value: true,
        text: 'ablate conv1#3  · since step 45', enabled: false,
        why: 'The browser viewer only reads: use the app to change this control.' } },
  };
}

const rgb = (c: readonly number[]) => `rgb(${c[0]},${c[1]},${c[2]})`;
const near = (a: number[], b: readonly number[], tol = 3) => a.slice(0, 3).every((v, i) => Math.abs(v - b[i]) <= tol);

async function open(page: import('@playwright/test').Page): Promise<MockApp & { state: Json }> {
  const app: MockApp = { specs: { mnist_dashboard: MNIST_DASHBOARD }, data: { mnist_dashboard: mnistDashboardData() }, requests: [] };
  await mockApp(page, app);
  await page.setViewportSize({ width: 1280, height: 800 });
  await openSpec(page, 'mnist_dashboard');
  await page.waitForFunction(() => {
    const s = (window as any).caliperSpec?.debugState();
    return s && s.controls.length === 1 && s.controls[0].text !== '' && Object.keys(s.lines).length === 2;
  }, null, { timeout: 15_000 });
  await page.evaluate(() => new Promise((ok) => requestAnimationFrame(() => requestAnimationFrame(ok))));
  return { ...app, state: await page.evaluate(() => (window as any).caliperSpec.debugState()) };
}

test('the MNIST dashboard fills the window over its control row, and the toggle shows the run state, disabled', async ({ page }) => {
  const { state, requests } = await open(page);
  const area = await page.evaluate(() => {
    const r = document.getElementById('area')!.getBoundingClientRect();
    return { top: r.top, w: r.width, h: r.height };
  });
  const ctl = state.controls[0];
  expect(ctl.id).toBe('ablate');
  expect(ctl.kind).toBe('toggle');
  // One control row (two lines of the 15 px face) at the bottom; the panels take everything above it.
  expect(ctl.rect).toEqual({ x: 0, y: area.h - 30, w: area.w, h: 30 });
  const samples = state.panels.find((p: Json) => p.id === 'samples');
  expect(samples.rect.y + samples.rect.h + 1).toBe(ctl.rect.y);
  const acc = state.panels.find((p: Json) => p.id === 'acc_curve');
  expect(acc.rect.y + acc.rect.h + 1).toBe(ctl.rect.y);
  // The lower half is the panels' ground, not the rule colour of an empty share.
  expect(near(await pixel(page, area.w * 0.75, area.h * 0.7), [24, 23, 21])).toBe(true);
  expect(near(await pixel(page, area.w * 0.75, ctl.rect.y + 3), [24, 23, 21])).toBe(true);
  // The toggle's state, as the run gives it, and disabled: a faint knob on the right of a plate track.
  expect(ctl.value).toBe(true);
  expect(ctl.enabled).toBe(false);
  expect(ctl.text).toContain('ablate conv1#3');
  expect(ctl.why).toContain('only reads');
  // The switch's knob is drawn in the faint colour of a disabled widget.
  const knob = await page.evaluate(([y]) => {
    const c = document.getElementById('canvas') as HTMLCanvasElement, d = window.devicePixelRatio || 1;
    const g = c.getContext('2d')!.getImageData(0, Math.round(y * d), c.width, 1).data;
    const faint = [143, 137, 124];
    const xs: number[] = [];
    for (let x = 0; x < c.width; x++) if ([0, 1, 2].every((k) => Math.abs(g[x * 4 + k] - faint[k]) <= 6)) xs.push(x / d);
    return xs;
  }, [ctl.rect.y + 15]);
  expect(knob.length).toBeGreaterThan(0);
  // A hover over the control says why the viewer cannot change it.
  await page.mouse.move(400, area.top + ctl.rect.y + 15);
  await expect.poll(() => page.evaluate(() => document.getElementById('canvas')!.title)).toContain('only reads');
  // The control's state came from panel.data, a read; nothing else was asked.
  expect(requests.some((r) => r.name === 'panel.data' && r.args.panel === 'ablate')).toBe(true);
  expect(requests.every((r) => r.name === 'panel.data' || r.name === 'spec.get')).toBe(true);
});

test('the chart lines take the app palette in its order, named by their sources, with a legend', async ({ page }) => {
  const { state, requests } = await open(page);
  const scores = state.lines.class_scores as { name: string; color: string }[];
  expect(scores.map((l) => l.name)).toEqual(LINES);
  // The app's key: the binding names first ("value", entry 0), then each line's name at its first draw, in
  // layout order: the eight digit series take entries 1 to 8, the accuracy curve entry 9.
  const palette = [...HOUSE_PIGMENTS, ...CATEGORICAL];
  expect(scores.map((l) => l.color)).toEqual(palette.slice(1, 9).map(rgb));
  expect(state.lines.acc_curve[0]).toEqual({ name: 'acc_window', color: rgb(palette[9]) });
  // Every line has its own colour.
  expect(new Set(scores.map((l) => l.color)).size).toBe(8);
  // The page sent only the viewer's reads.
  expect([...new Set(requests.map((r) => r.name))].sort()).toEqual(['panel.data', 'spec.get']);
  // The legend's swatches: the eight colours appear in the chart's top right.
  const chart = state.panels.find((p: Json) => p.id === 'class_scores').rect;
  const seen = await page.evaluate(([x0, y0, w, h]) => {
    const c = document.getElementById('canvas') as HTMLCanvasElement, d = window.devicePixelRatio || 1;
    const g = c.getContext('2d')!.getImageData(Math.round(x0 * d), Math.round(y0 * d), Math.round(w * d), Math.round(h * d)).data;
    const out = new Set<string>();
    for (let i = 0; i < g.length; i += 4) out.add(`rgb(${g[i]},${g[i + 1]},${g[i + 2]})`);
    return [...out];
  }, [chart.x + chart.w / 2, chart.y, chart.w / 2, chart.h / 2]);
  for (const c of palette.slice(1, 9).map(rgb)) expect(seen).toContain(c);
});

test('a dashboard that the app does not have open says so, and an offline source of an open one says Source offline', async ({ page }) => {
  const offline = { panel: 'class_scores', primitive: 'series', version: '', unchanged: false, bounds: [], state: null, data: null,
    error: { code: 'resolve.unavailable', message: 'The live source live:probe.logits is not attached.', where: 'digit_0' } };
  const data = { ...mnistDashboardData(), class_scores: offline };
  const app: MockApp = { specs: { mnist_dashboard: MNIST_DASHBOARD }, data: { mnist_dashboard: data }, requests: [], openInApp: { mnist_dashboard: false } };
  await mockApp(page, app);
  await page.setViewportSize({ width: 1280, height: 800 });
  await openSpec(page, 'mnist_dashboard');
  const state = (s: Json) => s.panels.find((p: Json) => p.id === 'class_scores')?.state;
  await expect.poll(async () => state(await page.evaluate(() => (window as any).caliperSpec.debugState()))).toBe('Instrument not open in the app');
  let s = await page.evaluate(() => (window as any).caliperSpec.debugState());
  expect(s.notOpen).toBe(true);
  expect(s.banner).toBe('The app does not have this instrument open. It shows data when the instrument is open in the app.');
  await expect(page.locator('#banner')).toBeVisible();
  // The person opens it in the app: the banner goes, and an offline source says so plainly.
  app.openInApp = { mnist_dashboard: true };
  await expect.poll(async () => state(await page.evaluate(() => (window as any).caliperSpec.debugState()))).toBe('Source offline');
  s = await page.evaluate(() => (window as any).caliperSpec.debugState());
  expect(s.notOpen).toBe(false);
  expect(s.banner).toBe('');
  await expect(page.locator('#banner')).toBeHidden();
});
