// The spec page on a dashboard of the Arc Tower's shape (the layout, sources
// and control that tests/programmable_graphics/driver.py builds), against the
// mock of the app: the layout fills the window over the row of its control
// (the native layout's control rows), the chart's eight lines take the app's
// palette in the app's order under their sources' names with a legend, and
// the toggle shows the run's state, disabled, with the reason on a hover.

import { test, expect } from '@playwright/test';
import { f64, mockApp, openSpec, pixel, ready, rng, type Json, type MockApp } from './spec-mock.js';
import { CATEGORICAL, HOUSE_PIGMENTS } from '../src/render/palette.gen.js';

test.beforeEach(({}, info) => test.skip(info.project.name !== 'swiftshader', 'the spec page draws with Canvas 2D: one project is enough'));

const LINES = ['l1_h0', 'l1_h1', 'l1_h2', 'l1_h3', 'l0_h0', 'l0_h1', 'l0_h2', 'l0_h3'];

export const ARC_TOWER: Json = {
  caliper_spec: 1,
  applet: { id: 'arc_tower', title: 'Attention Arc Tower' },
  state: [{ id: 'head', type: 'integer', default: 0, min: 0, max: 3, title: 'Head' }],
  source: [
    { id: 'loss', catalog: 'live:loss', history: 4096 },
    { id: 'loss_window', from: 'loss', transform: [{ op: 'window', steps: 4096 }] },
    { id: 'icl', catalog: 'live:token_loss.act', history: 4096 },
    { id: 'icl_window', from: 'icl', transform: [{ op: 'window', steps: 4096 }] },
    ...LINES.flatMap((l) => [{ id: `${l}_raw`, catalog: `live:induction_${l}`, history: 4096 },
      { id: l, from: `${l}_raw`, transform: [{ op: 'window', steps: 4096 }] }]),
  ],
  panel: [
    { id: 'tower', primitive: 'scene', title: 'Attention Arc Tower', options: { axes: false, up: '+y', select: { state: 'head', field: 'head' } } },
    { id: 'pattern', primitive: 'scene', title: 'Attention Grid' },
    { id: 'loss_curve', primitive: 'series', title: 'Loss', bind: { x: 'loss_window.step', y: 'loss_window.value' },
      options: { x_label: 'Step', y_label: 'Loss' } },
    { id: 'icl_curve', primitive: 'series', title: 'In-Context Loss', bind: { x: 'icl_window.step', y: 'icl_window.value' },
      options: { x_label: 'Step', y_label: 'Loss' } },
    { id: 'head_scores', primitive: 'series', title: 'Induction Score of Each Head',
      bind: { x: 'l1_h0.step', y: LINES.map((l) => `${l}.value`) }, options: { x_label: 'Step', y_label: 'Induction Score', legend: true } },
  ],
  control: [{ id: 'ablate', kind: 'toggle', title: 'Ablate Head 0 of Layer 1', bind: 'intervene:blocks.1.attn#0', options: { value: false } }],
  layout: { split: 'vertical', children: [
    { split: 'horizontal', ratio: 0.5, children: [
      { split: 'vertical', ratio: 0.45, children: ['tower', { split: 'vertical', ratio: 0.62, children: ['head_scores',
        { tabs: ['icl_curve', 'loss_curve'] }] }] }, 'pattern'] }, 'ablate'] },
};

/** panel.data of the dashboard: the window sources' lines as the app names them, and the toggle's state. */
export function arcTowerData(): Record<string, Json> {
  const r = rng(7), n = 60;
  const step = Array.from({ length: n }, (_, i) => 10 + 9 * i);
  const line = (binding: string, name: string, f: (i: number) => number) => {
    const y = step.map((_, i) => f(i));
    return { binding, name, row: null, count: n, last: y[n - 1], x: f64(step), y: f64(y) };
  };
  const series = (id: string, lines: Json[]) => ready(id, 'series', { x_index: false, x_name: lines[0].name, steps: n, lines });
  return {
    head_scores: series('head_scores', LINES.map((l, k) => line(`${l}.value`, l, (i) => 0.03 + 0.005 * k + (k === 0 ? i / 1500 : 0) + 0.004 * r()))),
    icl_curve: series('icl_curve', [line('icl_window.value', 'icl_window', (i) => 4.4 - 0.3 * Math.tanh(i / 12) + 0.02 * r())]),
    loss_curve: series('loss_curve', [line('loss_window.value', 'loss_window', (i) => 3.9 - 0.2 * Math.tanh(i / 20))]),
    tower: ready('tower', 'scene', { scene: null, programs: [] }),
    pattern: ready('pattern', 'scene', { scene: null, programs: [] }),
    ablate: { panel: 'ablate', primitive: 'control', version: 'c1', unchanged: false, error: null, state: null, bounds: [],
      data: { id: 'ablate', kind: 'toggle', title: 'Ablate Head 0 of Layer 1', bind: 'intervene:blocks.1.attn#0', value: true,
        text: 'ablate blocks.1.attn#0  · since step 412', enabled: false,
        why: 'The browser viewer only reads: use the app to change this control.' } },
  };
}

const rgb = (c: readonly number[]) => `rgb(${c[0]},${c[1]},${c[2]})`;
const near = (a: number[], b: readonly number[], tol = 3) => a.slice(0, 3).every((v, i) => Math.abs(v - b[i]) <= tol);

async function open(page: import('@playwright/test').Page): Promise<MockApp & { state: Json }> {
  const app: MockApp = { specs: { arc_tower: ARC_TOWER }, data: { arc_tower: arcTowerData() }, requests: [] };
  await mockApp(page, app);
  await page.setViewportSize({ width: 1280, height: 800 });
  await openSpec(page, 'arc_tower');
  await page.waitForFunction(() => {
    const s = (window as any).caliperSpec?.debugState();
    return s && s.controls.length === 1 && s.controls[0].text !== '' && Object.keys(s.lines).length === 2;
  }, null, { timeout: 15_000 });
  await page.evaluate(() => new Promise((ok) => requestAnimationFrame(() => requestAnimationFrame(ok))));
  return { ...app, state: await page.evaluate(() => (window as any).caliperSpec.debugState()) };
}

test('the Arc Tower fills the window over its control row, and the toggle shows the run state, disabled', async ({ page }) => {
  const { state, requests } = await open(page);
  const area = await page.evaluate(() => { const r = document.getElementById('area')!.getBoundingClientRect(); return { w: r.width, h: r.height }; });
  const ctl = state.controls[0];
  expect(ctl.id).toBe('ablate');
  expect(ctl.kind).toBe('toggle');
  // One control row (two lines of the 15 px face) at the bottom; the panels take everything above it.
  expect(ctl.rect).toEqual({ x: 0, y: area.h - 30, w: area.w, h: 30 });
  const pattern = state.panels.find((p: Json) => p.id === 'pattern');
  expect(pattern.rect.y + pattern.rect.h + 1).toBe(ctl.rect.y);
  const icl = state.panels.find((p: Json) => p.id === 'icl_curve');
  expect(icl.rect.y + icl.rect.h + 1).toBe(ctl.rect.y);
  // The lower half is the panels' ground, not the rule colour of an empty share.
  expect(near(await pixel(page, area.w * 0.75, area.h * 0.7), [24, 23, 21])).toBe(true);
  expect(near(await pixel(page, area.w * 0.75, ctl.rect.y + 3), [24, 23, 21])).toBe(true);
  // The toggle's state, as the run gives it, and disabled: a faint knob on the right of a plate track.
  expect(ctl.value).toBe(true);
  expect(ctl.enabled).toBe(false);
  expect(ctl.text).toContain('ablate blocks.1.attn#0');
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
  await page.mouse.move(400, ctl.rect.y + 15);
  await expect.poll(() => page.evaluate(() => document.getElementById('canvas')!.title)).toContain('only reads');
  // The control's state came from panel.data, a read; nothing else was asked.
  expect(requests.some((r) => r.name === 'panel.data' && r.args.panel === 'ablate')).toBe(true);
  expect(requests.every((r) => r.name === 'panel.data' || r.name === 'spec.get')).toBe(true);
});

test('the chart lines take the app palette in its order, named by their sources, with a legend', async ({ page }) => {
  const { state, requests } = await open(page);
  const head = state.lines.head_scores as { name: string; color: string }[];
  expect(head.map((l) => l.name)).toEqual(LINES);
  // The app's key: the binding names first ("value", entry 0), then each line's name at its first draw, in
  // layout order: the eight heads take entries 1 to 8, the in-context loss entry 9.
  const palette = [...HOUSE_PIGMENTS, ...CATEGORICAL];
  expect(head.map((l) => l.color)).toEqual(palette.slice(1, 9).map(rgb));
  expect(state.lines.icl_curve[0]).toEqual({ name: 'icl_window', color: rgb(palette[9]) });
  // Every line has its own colour.
  expect(new Set(head.map((l) => l.color)).size).toBe(8);
  // The page sent only the viewer's reads.
  expect([...new Set(requests.map((r) => r.name))].sort()).toEqual(['panel.data', 'spec.get']);
  // The legend's swatches: the eight colours appear in the chart's top right.
  const chart = state.panels.find((p: Json) => p.id === 'head_scores').rect;
  const seen = await page.evaluate(([x0, y0, w, h]) => {
    const c = document.getElementById('canvas') as HTMLCanvasElement, d = window.devicePixelRatio || 1;
    const g = c.getContext('2d')!.getImageData(Math.round(x0 * d), Math.round(y0 * d), Math.round(w * d), Math.round(h * d)).data;
    const out = new Set<string>();
    for (let i = 0; i < g.length; i += 4) out.add(`rgb(${g[i]},${g[i + 1]},${g[i + 2]})`);
    return [...out];
  }, [chart.x + chart.w / 2, chart.y, chart.w / 2, chart.h / 2]);
  for (const c of palette.slice(1, 9).map(rgb)) expect(seen).toContain(c);
});
