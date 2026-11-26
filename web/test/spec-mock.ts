// A mock of the app for the spec page's tests: Playwright's routing answers
// GET /v0/spec with the built page (dist/spec.html) and POST
// /v0/commands/{spec.get, applet.list, panel.data} with the forms the app's
// command layer gives (src/builder/session/panel_data.cpp), from specs and
// data that the tests make here. It records every command request.

import type { Page, Route } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { WEB } from './util.js';

export const ORIGIN = 'http://127.0.0.1:7399';
export const TOKEN = 'viewer_token-0123456789abcdefghij';

export type Json = Record<string, any>;

export function f64(values: ArrayLike<number>, shape?: number[]): Json {
  const a = Float64Array.from(values as ArrayLike<number>);
  return { dtype: 'f64', shape: shape ?? [a.length], base64: Buffer.from(a.buffer).toString('base64') };
}

export function f32(values: ArrayLike<number>, shape?: number[]): Json {
  const a = Float32Array.from(values as ArrayLike<number>);
  return { dtype: 'f32', shape: shape ?? [a.length], base64: Buffer.from(a.buffer).toString('base64') };
}

/** A panel.data result that draws. */
export function ready(panel: string, primitive: string, data: Json, version = 'v1'): Json {
  return { panel, primitive, version, unchanged: false, error: null, state: null, bounds: [], data };
}

export interface MockApp {
  specs: Record<string, Json>;                         // applet id -> canonical JSON
  data: Record<string, Record<string, Json>>;          // applet id -> panel id -> panel.data result
  requests: { name: string; args: Json; headers: Record<string, string> }[];
  /** spec.get's open_in_app for an applet: whether the app has it open; absent when not set. */
  openInApp?: Record<string, boolean>;
}

/** A seeded pseudo-random sequence, so captures repeat. */
export function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

/** The dashboard fixture's spec (src/builder/spec/tests/fixtures/dashboard.spec.toml) in its JSON form. */
export const DASHBOARD: Json = {
  caliper_spec: 0,
  applet: { id: 'run42-dashboard', title: 'Run 42', description: 'Loss and accuracy of run 42.' },
  source: [
    { id: 'metrics', catalog: 'table:run42_metrics' },
    { id: 'attn', catalog: 'tensor:attn_weights', transform: [
      { op: 'slice', axis: 0, index: 0 }, { op: 'reduce', axis: 0, fn: 'mean' }, { op: 'normalise', mode: 'minmax' }] },
  ],
  panel: [
    { id: 'curves', primitive: 'series', title: 'Loss and accuracy', bind: { x: 'metrics.step', y: ['metrics.loss', 'metrics.accuracy'] },
      options: { x_label: 'Step', y_log: true } },
    { id: 'attention', primitive: 'heatmap', title: 'Mean attention of layer 0', bind: { values: 'attn' },
      options: { colormap: 'magma', range: [0, 1], x_label: 'Key', y_label: 'Query' } },
    { id: 'rows', primitive: 'table', bind: { rows: 'metrics' }, options: { columns: ['step', 'loss'] } },
  ],
  layout: { split: 'vertical', ratio: 0.65, children: [{ split: 'horizontal', children: ['curves', 'attention'] }, 'rows'] },
};

/** Data like the native captures': 4,000 steps of loss and accuracy, a 48 x 48 causal attention, the metrics table. */
export function dashboardData(): Record<string, Json> {
  const r = rng(42);
  const n = 4000, step: number[] = [], loss: number[] = [], acc: number[] = [];
  for (let i = 0; i < n; i++) {
    step.push(i);
    loss.push(2.6 * Math.exp(-i / 1400) + 0.12 + (r() - 0.5) * 0.08 * Math.exp(-i / 3000));
    acc.push(0.95 / (1 + Math.exp(-(i - 600) / 160)) + 0.03 + (r() - 0.5) * 0.02);
  }
  const T = 48, att: number[] = [];
  for (let q = 0; q < T; q++)
    for (let k = 0; k < T; k++) att.push(k > q ? 0 : k === q ? 0.35 + 0.1 * r() : k === 0 ? 0.25 : (q - k) % 8 === 0 ? 0.12 : 0.02 * r());
  att[0] = 1;
  // The app decimates a line of more than 8,192 points; 4,000 travel whole.
  const lines = [
    { binding: 'metrics.loss', name: 'loss', row: null, count: n, last: loss[n - 1], x: f64(step), y: f64(loss) },
    { binding: 'metrics.accuracy', name: 'accuracy', row: null, count: n, last: acc[n - 1], x: f64(step), y: f64(acc) },
  ];
  const rows: number[][] = [];
  for (let i = 0; i < 1000; i++) rows.push([i, loss[i]]);
  return {
    curves: ready('curves', 'series', { x_index: false, x_name: 'step', steps: n, lines }),
    attention: ready('attention', 'heatmap', { source_rows: T, source_cols: T, stride_rows: 1, stride_cols: 1, values: f32(att, [T, T]) }),
    rows: ready('rows', 'table', { columns: [{ name: 'step', type: 'BIGINT' }, { name: 'loss', type: 'DOUBLE' }], rows,
      total_rows: n, truncated: false }),
  };
}

/** Routes the page and the three commands of the viewer to `app`. */
/** The WebAssembly views' files, which the app serves beside the page (GET /v0/spec/<name>). */
export const VIEWS_FILES = ['caliper-views-loader.js', 'caliper-views.js', 'caliper-views.wasm'];

/**
 * Routes the mock app. With `assets`, the folder of the WebAssembly views, it
 * serves their three files as the app does, with the viewer's token in the
 * query, and the page's CSP as the app sends it.
 */
export async function mockApp(page: Page, app: MockApp, assets?: string): Promise<void> {
  const html = fs.readFileSync(path.join(WEB, 'dist', 'spec.html'), 'utf8');
  await page.route(`${ORIGIN}/**`, async (route: Route) => {
    const req = route.request();
    const url = new URL(req.url());
    if (req.method() === 'GET' && url.pathname === '/v0/spec') {
      await route.fulfill({ status: 200, contentType: 'text/html; charset=utf-8', body: html, headers: {
        'Content-Security-Policy': "default-src 'none'; script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'; " +
          "style-src 'unsafe-inline'; font-src data:; img-src data: blob:; connect-src 'self'; base-uri 'none'; " +
          "form-action 'none'; frame-ancestors 'none'" } });
      return;
    }
    const asset = /^\/v0\/spec\/([a-z-]+\.(?:js|wasm))$/.exec(url.pathname);
    if (req.method() === 'GET' && asset && VIEWS_FILES.includes(asset[1])) {
      app.requests.push({ name: `GET ${asset[1]}`, args: {}, headers: req.headers() });
      if (!assets || url.searchParams.get('token') !== TOKEN || !fs.existsSync(path.join(assets, asset[1]))) {
        await route.fulfill({ status: assets ? 401 : 404, contentType: 'application/json', body: '{"error":"no such file"}' });
        return;
      }
      await route.fulfill({ status: 200, contentType: asset[1].endsWith('.wasm') ? 'application/wasm' : 'text/javascript',
        body: fs.readFileSync(path.join(assets, asset[1])) });
      return;
    }
    const m = /^\/v0\/commands\/([a-z.]+)$/.exec(url.pathname);
    if (req.method() !== 'POST' || !m) {
      await route.fulfill({ status: 404, contentType: 'application/json', body: '{"error":"no such endpoint"}' });
      return;
    }
    const name = m[1], args = JSON.parse(req.postData() ?? '{}') as Json, headers = req.headers();
    app.requests.push({ name, args, headers });
    const ok = (result: Json) => route.fulfill({ status: 200, contentType: 'application/json',
      body: JSON.stringify({ id: `api-${app.requests.length}`, ok: true, result }) });
    const fail = (code: string) => route.fulfill({ status: 200, contentType: 'application/json',
      body: JSON.stringify({ id: `api-${app.requests.length}`, ok: false, error: { code, message: code, where: '' } }) });
    if (headers['authorization'] !== `Bearer ${TOKEN}`) {
      await route.fulfill({ status: 401, contentType: 'application/json', body: '{"error":"the session token is missing or wrong"}' });
      return;
    }
    if (name === 'applet.list') {
      await ok({ applets: Object.entries(app.specs).map(([id, s]) => ({ id, title: s.applet.title ?? id, path: `applets/${id}`, updated_at: null })) });
    } else if (name === 'spec.get') {
      const s = app.specs[args.applet];
      if (!s) await fail('applet.not_found');
      else await ok({ id: s.applet.id, path: `applets/${args.applet}/${args.applet}.spec.toml`, dirty: false, text: JSON.stringify(s), json: s,
        ...(app.openInApp && args.applet in app.openInApp ? { open_in_app: app.openInApp[args.applet] } : {}) });
    } else if (name === 'panel.data') {
      const d = app.data[args.applet]?.[args.panel];
      if (!d) await fail('panel.not_found');
      else if (args.known && args.known === d.version) await ok({ ...d, unchanged: true, data: null });
      else await ok(d);
    } else {
      await route.fulfill({ status: 403, contentType: 'application/json', body: '{"error":"the viewer token opens only the spec page"}' });
    }
  });
}

export async function openSpec(page: Page, applet: string): Promise<void> {
  await page.goto(`${ORIGIN}/v0/spec?token=${TOKEN}&applet=${applet}`);
}

export interface PageState {
  status: string; state: string | null; applet: string; title: string; layoutLocked: boolean;
  panels: { id: string; title: string; primitive: string; drawn: boolean; state: string | null; info: string;
            closable: boolean; closeButton: Json | null; rect: Json }[];
  controls?: { id: string; kind: string; title: string; value: unknown; text: string; enabled: boolean; why: string;
               closable: boolean; rect: Json }[];
  tabs: { path: string; labels: string[]; active: number }[];
  requests: number;
}

export const pageState = (page: Page): Promise<PageState> =>
  page.evaluate(() => (window as unknown as { caliperSpec: { debugState(): PageState } }).caliperSpec.debugState());

/** Waits until every shown panel drew or shows a state. */
export async function settled(page: Page, n: number): Promise<PageState> {
  await page.waitForFunction((n) => {
    const s = (window as unknown as { caliperSpec?: { debugState(): PageState } }).caliperSpec?.debugState();
    return !!s && s.panels.length === n && s.panels.every((p) => p.drawn || p.state !== null);
  }, n, { timeout: 15_000 });
  // One more frame, so the canvas holds what the state says.
  await page.evaluate(() => new Promise((ok) => requestAnimationFrame(() => requestAnimationFrame(ok))));
  return pageState(page);
}

/** The colour of the canvas's pixel at CSS (x, y). */
export const pixel = (page: Page, x: number, y: number): Promise<number[]> => page.evaluate(([x, y]) => {
  const c = document.getElementById('canvas') as HTMLCanvasElement;
  const d = window.devicePixelRatio || 1;
  return [...c.getContext('2d')!.getImageData(Math.round(x * d), Math.round(y * d), 1, 1).data];
}, [x, y]);
