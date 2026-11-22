// The element with Caliper's real WebAssembly views (not the stub): the
// producer, mock_server.py, also serves the three files of web/wasm's build,
// as caliper.serve() does from the Python package. The files come from
// CALIPER_WASM_DIR, else from python/caliper/_web, where scripts/build_wasm.py
// and scripts/fetch_wasm.py put them; without them these tests are skipped,
// and fallback.spec.ts checks the element without them.
//
// CALIPER_VIEWS_COMPAT=1 asks every mount for a compatibility-mode WebGPU
// device (the loader's default, set before the page loads).

import { test as base, type Page } from '@playwright/test';
import { spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { colormapLut, lutIndex } from '../src/render/colormap.js';
import { readColormapCases } from './colormap_cases.js';
import { decodePng, drawn, expect, REPO, roundTrip, serve, shotPath, WEB, wireModule } from './util.js';

const PYTHON = process.env.CALIPER_PYTHON ?? path.join(REPO, '.venv', 'bin', 'python');
const WASM = process.env.CALIPER_WASM_DIR ?? path.join(REPO, 'python', 'caliper', '_web');
const FILES = ['caliper-views-loader.js', 'caliper-views.js', 'caliper-views.wasm'];
const COMPAT = process.env.CALIPER_VIEWS_COMPAT === '1';

/** A producer; events holds every EVENT a viewer sent it, as mock_server.py prints them. */
interface Server { url: string; ws: string; proc: ChildProcess; events: ProducerEvent[] }
interface ProducerEvent { view: string; type: string; detail: Record<string, unknown>; sub?: number; seq?: number }

/** Starts mock_server.py with the WebAssembly views and these options. */
async function startProducer(args: string[] = []): Promise<Server> {
  const proc = spawn(PYTHON, [path.join(WEB, 'test', 'mock_server.py'), '--wire', wireModule(), '--wasm', WASM, ...args],
                     { stdio: ['ignore', 'pipe', 'inherit'] });
  const lines = readline.createInterface({ input: proc.stdout! });
  const events: ProducerEvent[] = [];
  const url = await new Promise<string>((ok, fail) => {
    proc.once('exit', (code) => fail(new Error(`mock_server.py exited with ${code}`)));
    lines.on('line', (line) => {
      if (line.startsWith('http://')) ok(line.trim());
      else if (line.startsWith('{')) events.push((JSON.parse(line) as { event: ProducerEvent }).event);
    });
  });
  const u = new URL(url);
  return { url, ws: `ws://${u.host}/v3/stream?token=${u.searchParams.get('token')}`, proc, events };
}

const test = base.extend<{ producer: Server }>({
  page: async ({ page }, use) => {
    if (COMPAT) await page.addInitScript(() => Object.assign(window, { caliperViewsCompat: true }));
    await use(page);
  },
  producer: async ({}, use) => {
    const p = await startProducer();
    await use(p);
    p.proc.kill();
  },
});

test.skip(!fs.existsSync(PYTHON), 'needs a Python with numpy and websockets (CALIPER_PYTHON)');
test.skip(!FILES.every((f) => fs.existsSync(path.join(WASM, f))), `needs the WebAssembly views in ${WASM}`);

type El = HTMLElement & { debugState(): Record<string, unknown>; dataView: unknown };
const state = (page: Page, view: string) => page.evaluate((v) =>
  (document.querySelector(`caliper-view[view="${v}"]`) as El).debugState(), view);

/** A second producer with its own options (the fixture's has the defaults). */
const producerWith = (args: string[]) => startProducer(args);

async function wasmStats(page: Page, view: string): Promise<Record<string, unknown>> {
  return ((await state(page, view)).wasmStats ?? {}) as Record<string, unknown>;
}

/**
 * The module's self-check of a view ("view:kind:mismatches:maxdiff"), waited on
 * until it matches re. A failure prints what the module reported: its checks,
 * how many frames it drew and how fast, and its adapter.
 */
async function moduleCheck(page: Page, view: string, re: RegExp, timeout = 30_000): Promise<string> {
  const read = async () => {
    const st = await wasmStats(page, view);
    return { checks: String(st.checks ?? ''), frames: st.frames, fps: st.fps, checkState: st.checkState,
      path: st.rendererPath, state: st.state, adapter: (await state(page, view)).adapter };
  };
  // Checked on every animation frame until it matches; past the deadline the
  // expectation below says what the module reported.
  await page.waitForFunction(([v, src]) => new RegExp(src).test(String(((document.querySelector(`caliper-view[view="${v}"]`) as El)
    .debugState().wasmStats as { checks?: string } | null)?.checks ?? '')), [view, re.source] as const, { polling: 'raf', timeout })
    .catch(() => undefined);
  const got = await read();
  expect(got.checks, `${view}: the module's checks (${JSON.stringify(got)})`).toMatch(re);
  return got.checks;
}

async function plates(page: Page, n: number) {
  await page.waitForFunction((n) => {
    const els = [...document.querySelectorAll('caliper-view')] as El[];
    return els.length === n && els.every((e) => e.debugState().status === 'connected' && e.debugState().renderer);
  }, n, { timeout: 30_000 });
}

test('auto draws each view with the WebAssembly views when they list its kind, and with the TypeScript views otherwise', async ({ page, producer }, info) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto(producer.url);
  await plates(page, 6);
  const got = await page.evaluate(() => [...document.querySelectorAll('caliper-view')].map((e) => {
    const s = (e as El).debugState();
    const root = e.shadowRoot!;
    return [s.view, s.drawer, s.renderer, root.querySelector('.name')?.textContent, (root.querySelector('.plate') as HTMLElement).hidden];
  }));
  if (info.project.name === 'canvas2d') {
    // No adapter: the TypeScript views draw everything.
    expect(got.map((g) => g[1])).toEqual(['ts', 'ts', 'ts', 'ts', 'ts', 'ts']);
  } else {
    // The loader lists every kind the stream has, so the WebAssembly views draw them all.
    expect(got).toEqual([
      ['loss', 'wasm', 'wasm', 'loss', false], ['weights', 'wasm', 'wasm', 'weights', false],
      ['attention', 'wasm', 'wasm', 'attention', false], ['embedding', 'wasm', 'wasm', 'embedding', false],
      ['weight_hist', 'wasm', 'wasm', 'weight_hist', false], ['input', 'wasm', 'wasm', 'input', false],
    ]);
    // Each WebAssembly view drew on its own canvas inside the element's body.
    const canvases = await page.evaluate(() => [...document.querySelectorAll('caliper-view')]
      .filter((e) => (e as El).debugState().drawer === 'wasm')
      .map((e) => {
        const c = e.shadowRoot!.querySelector('.body .wasm canvas') as HTMLCanvasElement | null;
        return c ? c.width > 0 && c.height > 0 : false;
      }));
    expect(canvases).toEqual([true, true, true, true, true, true]);
    // Each module read its texture back: the image is exactly the bytes that were sent, and the
    // heatmap exactly the CPU reference colormap (web/wasm/README.md, "The colormap").
    const [, , imageBad, imageMax] = (await moduleCheck(page, 'input', /^input:2:\d+:\d+$/)).split(':').map(Number);
    expect([imageBad, imageMax]).toEqual([0, 0]);
    const [, , heatBad, heatMax] = (await moduleCheck(page, 'weights', /^weights:1:\d+:\d+$/)).split(':').map(Number);
    expect([heatBad, heatMax]).toEqual([0, 0]);
    // The graphics layer drew, on the device asked for, and WebGPU reported no error to it.
    const st = (await state(page, 'weights')).wasmStats as { gfxReady?: number; featureLevel?: string; gfxErrors?: string };
    expect(st.gfxReady).toBe(1);
    expect(st.featureLevel).toBe(COMPAT ? 'compatibility' : 'core');
    expect(st.gfxErrors).toBe('');
    // The histogram holds the producer's bins, once its module has published them.
    const histInfo = async () => String(((await state(page, 'weight_hist')).wasmStats as { viewInfo?: string } | null)?.viewInfo ?? '');
    await expect.poll(histInfo, { timeout: 30_000, message: 'weight_hist: the module\'s view info' })
      .toMatch(/weight_hist:kind=5,updates=\d+,count=[1-9]\d*/);
  }
  await drawn(page);
  await page.screenshot({ path: shotPath(info, 'wasm-page'), fullPage: true });
  expect(errors).toEqual([]);
});

test('the header picker sets the WebAssembly view\'s data view, and the stream\'s reply confirms it', async ({ page, producer }, info) => {
  test.skip(info.project.name === 'canvas2d', 'the WebAssembly views need a WebGPU adapter');
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto(producer.url);
  await plates(page, 6);
  expect((await state(page, 'attention')).drawer).toBe('wasm');
  // The picker is built from what the view offers (the element's metadata connection).
  const dims = await page.evaluate(() => [...document.querySelector('caliper-view[view="attention"]')!.shadowRoot!.querySelectorAll('select[data-dim]')]
    .map((s) => (s as HTMLSelectElement).dataset.dim));
  expect(dims).toEqual(['layer', 'head', 'query', 'key']);
  const before = JSON.stringify((await state(page, 'attention')).dataView);
  await page.evaluate(() => {
    const sel = document.querySelector('caliper-view[view="attention"]')!.shadowRoot!.querySelector('select[data-dim="head"]') as HTMLSelectElement;
    sel.value = 'i:5';
    sel.dispatchEvent(new Event('change'));
  });
  // The module's VIEW_REQUEST is answered; the reply (not the request) sets the element's data view.
  await roundTrip.poll(async () => JSON.stringify((await state(page, 'attention')).dataView)).toContain('"head":5');
  expect(before).not.toContain('"head":5');
  await drawn(page, 'caliper-view[view="attention"]');
  await page.locator('caliper-view[view="attention"]').screenshot({ path: shotPath(info, 'wasm-picker') });
});

test('points draw as quads with their palette colours, and hover finds the point under the mouse', async ({ page, producer }, info) => {
  test.skip(info.project.name === 'canvas2d', 'the WebAssembly views need a WebGPU adapter');
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto(producer.url);
  await plates(page, 6);
  expect((await state(page, 'embedding')).drawer).toBe('wasm');
  // The module read its render target back: at every sampled point no other point covers, the
  // pixel is the palette colour of the point's label ("view:3:mismatches:compared").
  const [, , bad, compared] = (await moduleCheck(page, 'embedding', /^embedding:3:\d+:\d+$/)).split(':').map(Number);
  expect(bad).toBe(0);
  expect(compared).toBeGreaterThan(10);
  // Hover: somewhere around the middle of the cloud the module finds a point.
  const box = (await page.locator('caliper-view[view="embedding"] .body').boundingBox())!;
  const hover = async () => Number(/hover=(-?\d+)/.exec(String((await wasmStats(page, 'embedding')).viewInfo))?.[1] ?? -1);
  let found = -1;
  for (let k = 0; k < 40 && found < 0; ++k) {
    const r = 4 + k * 3, a = k * 0.9;
    await page.mouse.move(box.x + box.width / 2 + r * Math.cos(a), box.y + box.height / 2 + r * Math.sin(a));
    // The module publishes what it hovers at the end of a frame: one that began after the move.
    await drawn(page, 'caliper-view[view="embedding"]', 1);
    found = await hover();
  }
  expect(found).toBeGreaterThanOrEqual(0);
  await page.locator('caliper-view[view="embedding"]').screenshot({ path: shotPath(info, 'wasm-points-hover') });
});

test('100,000 points orbit at an interactive rate on the GPU', async ({ page }, info) => {
  test.skip(info.project.name !== 'webgpu', 'measured on the machine\'s GPU only');
  const big = await producerWith(['--points', '100000']);
  try {
    await page.setViewportSize({ width: 1000, height: 760 });
    await page.goto(big.url);
    await plates(page, 6);
    expect((await state(page, 'embedding')).drawer).toBe('wasm');
    const box = (await page.locator('caliper-view[view="embedding"] .body').boundingBox())!;
    const frames = async () => Number((await wasmStats(page, 'embedding')).frames ?? 0);
    const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
    await page.mouse.move(cx, cy);
    await page.mouse.down();
    // The time the module takes to draw 180 frames while the mouse drags,
    // each move sent once the page has handled the last.
    const f0 = await frames();
    const t0 = Date.now();
    let f = f0;
    for (let k = 0; f < f0 + 180; ++k) {
      await page.mouse.move(cx + 120 * Math.sin(k / 20), cy + 60 * Math.cos(k / 27));
      f = await frames();
    }
    const fps = (f - f0) / ((Date.now() - t0) / 1000);
    await page.mouse.up();
    console.log(`${info.project.name} wasm points 100000: ${fps.toFixed(1)} frames per second while orbiting`);
    expect(fps).toBeGreaterThan(30);
    await page.locator('caliper-view[view="embedding"]').screenshot({ path: shotPath(info, 'wasm-points-100k') });
  } finally {
    big.proc.kill();
  }
});

// --- the graphics layer -------------------------------------------------------------

test('the graphics layer draws an indexed quad and instanced draws, dispatches and copies, and reads every target back exactly', async ({ page }, info) => {
  test.skip(info.project.name === 'canvas2d', 'the WebAssembly views need a WebGPU adapter');
  // The module alone on a blank page: no stream, only the self-test (web/wasm/src/gfx_selftest.cpp).
  const site = await serve(WEB, (name) => (FILES.includes(name.slice(1)) ? path.join(WASM, name.slice(1)) : null));
  const consoleErrors: string[] = [];
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') consoleErrors.push(m.text()); });
  try {
    await page.goto(`${site.origin}/test/blank.html`);
    for (const compat of [false, true]) {
      const got = await page.evaluate(async ({ base, compat }) => {
        type M = { caliperStats?: Record<string, string>; caliperGfxImages?: Record<string, { w: number; h: number; data: Uint8Array }>;
                   _caliper_views_gfx_selftest(): number; _caliper_views_destroy(): void };
        const { preloadCaliperViews } = await import(base + 'caliper-views-loader.js');
        const { factory, wasm } = await preloadCaliperViews(base);
        const canvas = document.createElement('canvas');
        canvas.style.width = '64px';
        canvas.style.height = '64px';
        document.body.append(canvas);
        const m: M = await factory({
          canvas, caliperConfig: { src: '', compat: compat ? 'true' : 'false' },
          instantiateWasm: (imports: WebAssembly.Imports, receive: (i: WebAssembly.Instance, m: WebAssembly.Module) => void) => {
            WebAssembly.instantiate(wasm as WebAssembly.Module, imports).then((i) => receive(i, wasm));
            return {};
          },
          print: () => {}, printErr: () => {},
        });
        const until = async (f: () => boolean) => {
          for (let i = 0; i < 400 && !f(); i++) await new Promise((ok) => setTimeout(ok, 25));
          return f();
        };
        if (!(await until(() => Number(m.caliperStats?.gfxReady) === 1))) return { report: `no layer: ${JSON.stringify(m.caliperStats ?? {})}`, level: '', images: {} };
        m._caliper_views_gfx_selftest();
        await until(() => !!m.caliperStats?.gfxSelftest);
        // Each picture as a PNG, four times its size, nearest.
        const images: Record<string, string> = {};
        for (const [name, im] of Object.entries(m.caliperGfxImages ?? {})) {
          const c = document.createElement('canvas');
          c.width = im.w * 4;
          c.height = im.h * 4;
          const small = document.createElement('canvas');
          small.width = im.w;
          small.height = im.h;
          small.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(im.data), im.w, im.h), 0, 0);
          const g = c.getContext('2d')!;
          g.imageSmoothingEnabled = false;
          g.drawImage(small, 0, 0, c.width, c.height);
          images[name] = c.toDataURL('image/png');
        }
        const out = { report: m.caliperStats!.gfxSelftest, level: m.caliperStats!.featureLevel, images };
        m._caliper_views_destroy();
        canvas.remove();
        return out;
      }, { base: `${site.origin}/`, compat });
      console.log(`${info.project.name} graphics layer self-test (${compat ? 'compatibility' : 'core'} device): ${got.report}`);
      if (!got.report.startsWith('ok')) console.log(consoleErrors.join('\n'));
      for (const [name, url] of Object.entries(got.images))
        fs.writeFileSync(shotPath(info, `${name}-${compat ? 'compat' : 'core'}`), Buffer.from(url.split(',')[1], 'base64'));
      // uniforms=same: the scene core's fixture (src/host/scene/uniform_check.h) compiles here to the
      // arena bytes it compiles to natively, so the browser draws from the same f32 matrices as Metal.
      expect(got.report).toBe('ok color=0 id=0 depth=0 fill=0 copy=0 import=copied-from-host:24 own-target=refused '
                              + `submit-error=reported level=${compat ? 'compatibility' : 'core'} uniforms=same`);
      expect(Object.keys(got.images).sort()).toEqual(['layer-compute', 'layer-copy', 'layer-draws', 'layer-ids']);
    }
  } finally {
    site.close();
  }
});

// The colormap's keys as the module derives them at run time, from the WebAssembly build of the CPU
// reference, against the case file that the native tool writes (CALIPER_COLORMAP_CASES), from
// which the browser rows take their keys and expected colours: the module's scene views use the
// former, so both builds of the reference must agree on every key of every boundary range.
test('the WebAssembly views derive the same colormap keys as the native case file', async ({ page }, info) => {
  test.skip(info.project.name === 'canvas2d', 'the WebAssembly views need a WebGPU adapter');
  const CASES = process.env.CALIPER_COLORMAP_CASES;
  test.skip(!CASES, 'needs CALIPER_COLORMAP_CASES, the file that caliper_colormap_cases writes');
  const cases = readColormapCases(CASES!);
  const site = await serve(WEB, (name) => (FILES.includes(name.slice(1)) ? path.join(WASM, name.slice(1)) : null));
  try {
    await page.goto(`${site.origin}/test/blank.html`);
    // Each range as the bits of its two ends, so that every float reaches the module exactly.
    const ends = cases.map((c) => [c.block.readUInt32LE(0), c.block.readUInt32LE(4)]);
    const derived: number[][] = await page.evaluate(async ({ base, ends }) => {
      type M = { caliperColormapKeys?: Uint32Array; _caliper_views_colormap_keys(vmin: number, vmax: number): number;
                 _caliper_views_destroy(): void };
      const { preloadCaliperViews } = await import(base + 'caliper-views-loader.js');
      const { factory, wasm } = await preloadCaliperViews(base);
      const canvas = document.createElement('canvas');
      document.body.append(canvas);
      const m: M = await factory({
        canvas, caliperConfig: { src: '' },
        instantiateWasm: (imports: WebAssembly.Imports, receive: (i: WebAssembly.Instance, m: WebAssembly.Module) => void) => {
          WebAssembly.instantiate(wasm as WebAssembly.Module, imports).then((i) => receive(i, wasm));
          return {};
        },
        print: () => {}, printErr: () => {},
      });
      const f = new Float32Array(2), u = new Uint32Array(f.buffer);
      const out = ends.map(([a, b]) => {
        u[0] = a;
        u[1] = b;
        m._caliper_views_colormap_keys(f[0], f[1]);
        return Array.from(m.caliperColormapKeys!);
      });
      m._caliper_views_destroy();
      canvas.remove();
      return out;
    }, { base: `${site.origin}/`, ends });
    const bad: string[] = [];
    let compared = 0;
    for (const [i, c] of cases.entries()) {
      const want = [c.block.readUInt32LE(8)];                                   // the flip
      for (let k = 0; k < 256; k++) want.push(c.block.readUInt32LE(16 + 16 * k));
      want.push(c.block.readUInt32LE(16 + 16 * 255 + 4));                      // key[256], the end
      const got = derived[i];
      for (let k = 0; k < want.length; k++) {
        compared++;
        if (got[k] !== want[k] && bad.length < 8)
          bad.push(`${c.name}: ${k === 0 ? 'flip' : `key ${k - 1}`} is 0x${got[k]?.toString(16)} here, 0x${want[k].toString(16)} in the case file`);
      }
    }
    console.log(`${info.project.name} colormap keys: ${compared} values over ${cases.length} ranges compared with the native case file, ${bad.length} differ`);
    expect(bad).toEqual([]);
  } finally {
    site.close();
  }
});

// The fixed views' pictures. Each run saves them (fixed-<view>-<project>.png under the shots
// folder); with CALIPER_VIEWS_REFERENCE naming a folder of the same files, saved from an earlier
// build, every byte of each view's data area (the heatmap's cells, the points' render target, as
// the module reports the area) must equal it. The whole canvas is compared too and its count
// printed: the dimension names "row" and "col" are drawn at unrounded positions, and their edges
// move by a level or two between runs of the same build. The migration steps of the renderer core
// use this to show that a view's pixels did not move.
test('the fixed heatmaps and points draw the same pixels as the reference set', async ({ page }, info) => {
  test.skip(info.project.name === 'canvas2d', 'the WebAssembly views need a WebGPU adapter');
  const srv = await fixedPage(page);
  try {
    for (const v of ['h', 'ha', 'p'])
      await expect.poll(async () => String((await wasmStats(page, v)).checks ?? ''), { timeout: 20_000 }).toMatch(new RegExp(`^${v}:\\d+:0:\\d+$`));
    await page.mouse.move(1, 1);
    // Every view draws 30 more frames once the pointer has left it, so no hover is left in the
    // pictures; a wait on frames, not on time, holds on a slow runner too.
    const views = ['h', 'ha', 'p'];
    const frames = async (v: string) => Number((await wasmStats(page, v)).frames ?? 0);
    const f0 = await Promise.all(views.map(frames));
    for (const [i, v] of views.entries())
      await expect.poll(() => frames(v), { timeout: 20_000 }).toBeGreaterThanOrEqual(f0[i] + 30);
    const ref = process.env.CALIPER_VIEWS_REFERENCE;
    for (const v of ['h', 'ha', 'p']) {
      const shot = await page.locator(`caliper-view[view="${v}"] .body .wasm canvas`).screenshot();
      const file = shotPath(info, `fixed-${v}`);
      fs.writeFileSync(file, shot);
      if (!ref) continue;
      const a = decodePng(fs.readFileSync(path.join(ref, path.basename(file)))), b = decodePng(shot);
      expect([b.width, b.height], `view ${v}`).toEqual([a.width, a.height]);
      const [ax, ay, aw, ah] = (await geometry(page, v)).geo.area;
      let all = 0, inside = 0, compared = 0;
      for (let y = 0; y < a.height; y++)
        for (let x = 0; x < a.width; x++) {
          const i = (y * a.width + x) * 4;
          const d = (a.data[i] !== b.data[i] ? 1 : 0) + (a.data[i + 1] !== b.data[i + 1] ? 1 : 0) +
            (a.data[i + 2] !== b.data[i + 2] ? 1 : 0) + (a.data[i + 3] !== b.data[i + 3] ? 1 : 0);
          all += d;
          if (x >= Math.ceil(ax) && x < Math.floor(ax + aw) && y >= Math.ceil(ay) && y < Math.floor(ay + ah)) {
            inside += d;
            compared += 4;
          }
        }
      console.log(`${info.project.name} ${v}: ${inside} of ${compared} bytes of the data area and ${all} of ${a.data.length} of the canvas differ from the reference`);
      expect(compared, `view ${v}`).toBeGreaterThan(100_000);
      expect(inside, `view ${v}`).toBe(0);
    }
  } finally {
    srv.proc.kill();
  }
});

// --- selections, picks and hovers in the WebAssembly views ------------------------
// The fixed data of views.spec.ts (mock_server.py --fixed), and the same
// expectations as the TypeScript views' tests there: what the element's
// caliper-select event carries and what the producer receives as an EVENT.

interface Geo { area: number[]; window: number[]; yDown: boolean; selected: number }

/** Where view's data is on the page: its geometry from the module, and its canvas. */
async function geometry(page: Page, view: string) {
  // Past the module's 30 start-up frames the windows are laid out, and the
  // module publishes every 250 ms instead of every 15 frames.
  await page.waitForFunction((v) => Number(((document.querySelector(`caliper-view[view="${v}"]`) as El)
    .debugState().wasmStats as { frames?: number } | null)?.frames ?? 0) > 30, view, { polling: 'raf', timeout: 30_000 });
  await expect.poll(async () => JSON.parse(String((await wasmStats(page, view)).geometry ?? '{}'))[view] ?? null,
    { timeout: 10_000 }).not.toBeNull();
  const geo = JSON.parse(String((await wasmStats(page, view)).geometry))[view] as Geo;
  const box = (await page.locator(`caliper-view[view="${view}"] .body .wasm canvas`).boundingBox())!;
  /** Page coordinates of data point (x, y): columns and rows for a heatmap. */
  const at = (x: number, y: number) => {
    const [ax, ay, aw, ah] = geo.area, [x0, x1, y0, y1] = geo.window;
    const fx = (x - x0) / (x1 - x0), fy = (y - y0) / (y1 - y0);
    return { x: box.x + ax + fx * aw, y: box.y + (geo.yDown ? ay + fy * ah : ay + ah - fy * ah) };
  };
  return { geo, at };
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

/**
 * The page of the fixed views, one plate per row so that the series has
 * under two steps per pixel, with every caliper-select event recorded in
 * window.selects.
 */
async function fixedPage(page: Page): Promise<Server> {
  const srv = await producerWith(['--fixed']);
  await page.setViewportSize({ width: 860, height: 1900 });
  await page.goto(srv.url);
  await plates(page, 5);
  await page.evaluate(() => {
    const w = window as unknown as { selects: unknown[] };
    w.selects = [];
    for (const el of document.querySelectorAll('caliper-view')) el.addEventListener('caliper-select', (e) => w.selects.push((e as CustomEvent).detail));
  });
  return srv;
}

const selects = (page: Page) => page.evaluate(() => (window as unknown as { selects: unknown[] }).selects);
/** The producer's last event of a type for a view, once it has arrived. */
async function received(srv: Server, view: string, type: string, count = 1) {
  await roundTrip.poll(() => srv.events.filter((e) => e.view === view && e.type === type).length)
    .toBeGreaterThanOrEqual(count);
  return srv.events.filter((e) => e.view === view && e.type === type).at(-1)!;
}

test.describe('selections in the WebAssembly views', () => {
  test('a heatmap selects a block of cells and reports it in source indices; hover and pick report the cell', async ({ page }, info) => {
    test.skip(info.project.name === 'canvas2d', 'the WebAssembly views need a WebGPU adapter');
    const srv = await fixedPage(page);
    try {
      expect((await state(page, 'h')).drawer).toBe('wasm');
      // The header says what the TypeScript view would; a view whose one source has
      // no more dims than the kind keeps has no picker.
      await expect.poll(async () => (await state(page, 'h')).info).toBe('4×8  0 … 31');
      expect(await page.locator('caliper-view[view="h"] .picker select').count()).toBe(0);
      const h = await geometry(page, 'h');
      await shiftDrag(page, h.at(1.5, 1.5), h.at(4.5, 2.5));
      const want = { view: 'h', type: 'select', detail: { rows: [1, 3], cols: [1, 5] } };
      await expect.poll(() => selects(page)).toEqual([want]);
      expect((await received(srv, 'h', 'select')).detail).toEqual(want.detail);
      expect((await state(page, 'h')).selected).toBe(8);

      // Pooled from query rows 0..63 and key columns 10, 12, ..., 40.
      const a = await geometry(page, 'ha');
      await page.mouse.move(a.at(3.5, 1.5).x, a.at(3.5, 1.5).y);
      await page.mouse.move(a.at(3.5, 1.5).x + 1, a.at(3.5, 1.5).y);
      expect((await received(srv, 'ha', 'hover')).detail).toEqual({ row: 16, col: 22, value: 11 });
      await shiftDrag(page, a.at(1.5, 1.5), a.at(2.5, 2.5));
      await expect.poll(async () => (await selects(page)).at(-1)).toEqual({ view: 'ha', type: 'select', detail: { rows: [16, 48], cols: [14, 21] } });
      expect((await received(srv, 'ha', 'select')).detail).toEqual({ rows: [16, 48], cols: [14, 21] });
      // A click picks the cell; a shift-click clears the selection with an empty detail.
      await page.mouse.click(a.at(0.5, 3.5).x, a.at(0.5, 3.5).y);
      expect((await received(srv, 'ha', 'pick')).detail).toEqual({ row: 48, col: 10, value: 24 });
      await shiftDrag(page, a.at(5.5, 0.5), a.at(5.5, 0.5));
      await expect.poll(async () => (await selects(page)).at(-1)).toEqual({ view: 'ha', type: 'select', detail: {} });
      // Every event names what the viewer was showing.
      const ev = await received(srv, 'ha', 'select', 2);
      expect(ev.sub).toBeGreaterThan(0);
      expect(ev.seq).toBeGreaterThan(0);
      await page.locator('caliper-view[view="h"]').screenshot({ path: shotPath(info, 'wasm-heatmap-selection') });
    } finally {
      srv.proc.kill();
    }
  });

  test('points select a rectangle with shift-alt-drag and a lasso with shift-drag, and keep hover, zoom and pan', async ({ page }, info) => {
    test.skip(info.project.name === 'canvas2d', 'the WebAssembly views need a WebGPU adapter');
    const srv = await fixedPage(page);
    try {
      expect((await state(page, 'p')).drawer).toBe('wasm');
      await expect.poll(async () => (await state(page, 'p')).info).toBe('4 points  2-D  4 classes');
      expect(await page.locator('caliper-view[view="p"] .picker select').count()).toBe(0);
      const p = await geometry(page, 'p');
      // The fit leaves the points 4% inside the area, so the drags of
      // views.spec.ts start at y -0.02 and 1.02 instead of -0.04 and 1.04.
      await shiftDrag(page, p.at(0.8, -0.02), p.at(1.2, 1.02), true);
      const rect = { view: 'p', type: 'select', detail: { indices: [1, 3], row_ids: [3, 9] } };
      await expect.poll(async () => (await selects(page)).at(-1)).toEqual(rect);
      expect((await received(srv, 'p', 'select')).detail).toEqual(rect.detail);
      expect((await state(page, 'p')).selected).toBe(2);
      await drawn(page, 'caliper-view[view="p"]');
      await page.locator('caliper-view[view="p"]').screenshot({ path: shotPath(info, 'wasm-points-selection') });
      await shiftDrag(page, p.at(-0.2, -0.02), p.at(-0.2, -0.02), false, [p.at(0.3, -0.02), p.at(0.3, 0.3), p.at(-0.2, 0.3)]);
      await expect.poll(async () => (await selects(page)).at(-1)).toEqual({ view: 'p', type: 'select', detail: { indices: [0], row_ids: [0] } });
      // A shift-click clears the selection with an empty detail.
      await shiftDrag(page, p.at(0.5, 0.5), p.at(0.5, 0.5));
      await expect.poll(async () => (await selects(page)).at(-1)).toEqual({ view: 'p', type: 'select', detail: {} });
      expect((await state(page, 'p')).selected).toBe(0);
      // Hover finds the point under the mouse and reports its source row.
      await page.mouse.move(p.at(0, 1).x + 2, p.at(0, 1).y + 2);
      await page.mouse.move(p.at(0, 1).x + 3, p.at(0, 1).y + 2);
      expect((await received(srv, 'p', 'hover')).detail).toEqual({ index: 2, row_id: 6 });
      // Wheel zooms; a drag without shift pans.
      await page.mouse.wheel(0, -300);
      await expect.poll(async () => { const g = await geometry(page, 'p'); return g.geo.window[1] - g.geo.window[0]; })
        .toBeLessThan(p.geo.window[1] - p.geo.window[0]);
      const z = (await geometry(page, 'p')).geo.window;
      const c = p.at(0.5, 0.5);
      await page.mouse.move(c.x, c.y);
      await page.mouse.down();
      await page.mouse.move(c.x + 80, c.y, { steps: 4 });
      await page.mouse.up();
      await expect.poll(async () => (await geometry(page, 'p')).geo.window[0]).toBeLessThan(z[0]);
      // A fast drag, one movement and the release right after it, pans by all of it.
      await drawn(page, 'caliper-view[view="p"]', 2);
      const g1 = await geometry(page, 'p');
      const [w0, w1] = g1.geo.window;
      const q = g1.at((w0 + w1) / 2, (g1.geo.window[2] + g1.geo.window[3]) / 2);
      await page.mouse.move(q.x, q.y);
      await page.mouse.down();
      await page.mouse.move(q.x + 80, q.y);
      await page.mouse.up();
      const shift = ((w1 - w0) * 80) / g1.geo.area[2];
      await expect.poll(async () => w0 - (await geometry(page, 'p')).geo.window[0]).toBeGreaterThan(shift * 0.95);
      expect(w0 - (await geometry(page, 'p')).geo.window[0]).toBeLessThan(shift * 1.05);
    } finally {
      srv.proc.kill();
    }
  });

  test('a series selects a range of steps and a histogram a range of bins', async ({ page }, info) => {
    test.skip(info.project.name === 'canvas2d', 'the WebAssembly views need a WebGPU adapter');
    const srv = await fixedPage(page);
    try {
      await expect.poll(async () => (await state(page, 's')).info).toBe('1000 steps  last 999');
      const s = await geometry(page, 's');
      // The browser hands the page whole pixels, so the drag starts on the
      // pixel at or right of step 200.2 and ends on the one at or left of 399.8.
      const [from, to] = [s.at(200.2, 500), s.at(399.8, 500)];
      await shiftDrag(page, { x: Math.ceil(from.x), y: from.y }, { x: Math.floor(to.x), y: to.y });
      await expect.poll(async () => ((await selects(page)).at(-1) as { view?: string } | undefined)?.view).toBe('s');
      const ev = (await selects(page)).at(-1) as { type: string; detail: { steps: number[] } };
      expect(ev.type).toBe('select');
      expect(ev.detail.steps[0]).toBeGreaterThanOrEqual(200);
      expect(ev.detail.steps[0]).toBeLessThanOrEqual(202);
      expect(ev.detail.steps[1]).toBeGreaterThanOrEqual(399);
      expect(ev.detail.steps[1]).toBeLessThanOrEqual(401);
      expect((await received(srv, 's', 'select')).detail).toEqual(ev.detail);
      await drawn(page, 'caliper-view[view="s"]');
      await page.locator('caliper-view[view="s"]').screenshot({ path: shotPath(info, 'wasm-series-selection') });

      await expect.poll(async () => (await state(page, 'hist')).info).toBe('10 bins  -5 … 5  n 91');
      const g = await geometry(page, 'hist');
      await page.mouse.move(g.at(-1.5, 3).x, g.at(-1.5, 3).y);
      await page.mouse.move(g.at(-1.5, 3).x + 1, g.at(-1.5, 3).y);
      expect((await received(srv, 'hist', 'hover')).detail).toEqual({ bin: 3, lo: -2, hi: -1, count: 8 });
      await shiftDrag(page, g.at(-2.7, 3), g.at(0.4, 3));
      await expect.poll(async () => (await selects(page)).at(-1)).toEqual({ view: 'hist', type: 'select', detail: { range: [-3, 1] } });
      expect((await received(srv, 'hist', 'select')).detail).toEqual({ range: [-3, 1] });
      await drawn(page, 'caliper-view[view="hist"]');
      await page.locator('caliper-view[view="hist"]').screenshot({ path: shotPath(info, 'wasm-histogram-selection') });
    } finally {
      srv.proc.kill();
    }
  });
});

test.describe('pan and zoom in the WebAssembly views', () => {
  type Win = { x0: number; x1: number; y0: number; y1: number };
  const win = async (page: Page, view: string) => (await state(page, view)).window as Win;

  test('a heatmap zooms about the pointer, pans with a drag and fits again on a double click, and hover, selection and nearest sampling follow', async ({ page }, info) => {
    test.skip(info.project.name === 'canvas2d', 'the WebAssembly views need a WebGPU adapter');
    const srv = await fixedPage(page);
    try {
      let h = await geometry(page, 'h');
      const fit = await win(page, 'h');
      expect([fit.y0, fit.y1]).toEqual([0, 4]);   // the 4 x 8 cells fit the height, centred
      expect((fit.x0 + fit.x1) / 2).toBeCloseTo(4, 6);
      const p = h.at(5.5, 2.5);
      await page.mouse.move(p.x, p.y);
      await page.mouse.wheel(0, -400);
      await expect.poll(async () => (await win(page, 'h')).x1 - (await win(page, 'h')).x0).toBeLessThan((fit.x1 - fit.x0) * 0.8);
      const z = await win(page, 'h');
      expect(z.y1 - z.y0).toBeLessThan(4 * 0.8);
      expect(z.x0).toBeLessThan(5.5);
      expect(z.x1).toBeGreaterThan(5.5);
      // Hover and pick at this zoom report the cell under the pointer.
      h = await geometry(page, 'h');
      const q = h.at(5.5, 2.5);
      await page.mouse.move(q.x + 1, q.y);
      await expect.poll(async () => (await state(page, 'h')).hover).toEqual({ r: 2, c: 5 });
      expect((await received(srv, 'h', 'hover')).detail).toEqual({ row: 2, col: 5, value: 21 });
      // Magnified, every pixel of a cell has the cell's colour (nearest sampling);
      // the pointer moves off the view first, so that no tooltip covers the cell.
      await page.mouse.move(2, 2);
      await drawn(page, 'caliper-view[view="h"]', 2);
      const box = (await page.locator('caliper-view[view="h"] .body .wasm canvas').boundingBox())!;
      const png = decodePng(await page.screenshot({ clip: box }));
      const lut = colormapLut('viridis');
      const want = [lut[lutIndex(21, 0, 31) * 4], lut[lutIndex(21, 0, 31) * 4 + 1], lut[lutIndex(21, 0, 31) * 4 + 2], 255];
      for (const [cx, cy] of [[5.08, 2.08], [5.5, 2.5], [5.92, 2.92]]) {
        const a = h.at(cx, cy);
        expect(png.at(a.x - box.x, a.y - box.y), `cell 2,5 at ${cx},${cy}`).toEqual(want);
      }
      // A shift-drag at this zoom selects the cells under it.
      await shiftDrag(page, h.at(4.2, 1.2), h.at(5.8, 2.8));
      await expect.poll(async () => (await selects(page)).at(-1)).toEqual({ view: 'h', type: 'select', detail: { rows: [1, 3], cols: [4, 6] } });
      // A drag pans: the content follows the pointer to the right, so the window moves left.
      await page.mouse.move(q.x, q.y);
      await page.mouse.down();
      await page.mouse.move(q.x + 80, q.y, { steps: 4 });
      await page.mouse.up();
      await expect.poll(async () => (await win(page, 'h')).x0).toBeLessThan(z.x0);
      await page.locator('caliper-view[view="h"]').screenshot({ path: shotPath(info, 'wasm-heatmap-zoomed') });
      await page.mouse.dblclick(q.x, q.y);
      await expect.poll(() => win(page, 'h')).toEqual(fit);
    } finally {
      srv.proc.kill();
    }
  });

  test('a series stops following on a pan or a zoom, holds the user\'s view, and follows again on a double click or the follow toggle', async ({ page }, info) => {
    test.skip(info.project.name === 'canvas2d', 'the WebAssembly views need a WebGPU adapter');
    const srv = await fixedPage(page);
    try {
      await geometry(page, 's');
      await geometry(page, 'hist');
      const follow = page.locator('caliper-view[view="s"] .tool', { hasText: 'follow' });
      await expect(follow).toHaveAttribute('aria-pressed', 'true');
      await expect(page.locator('caliper-view[view="s"] .tool', { hasText: 'log' })).toHaveAttribute('aria-pressed', 'false');
      expect((await state(page, 's')).follow).toBe(true);
      const fit = await win(page, 's');
      const s = await geometry(page, 's');
      const c = s.at(500, 500);
      await page.mouse.move(c.x, c.y);
      await page.mouse.wheel(0, -400);
      await expect.poll(async () => (await state(page, 's')).follow).toBe(false);
      await expect(follow).toHaveAttribute('aria-pressed', 'false');
      await expect.poll(async () => { const w = await win(page, 's'); return w.x1 - w.x0; }).toBeLessThan((fit.x1 - fit.x0) * 0.9);
      const z = await win(page, 's');
      await page.mouse.down();
      await page.mouse.move(c.x + 80, c.y, { steps: 4 });
      await page.mouse.up();
      await expect.poll(async () => (await win(page, 's')).x0).toBeLessThan(z.x0);
      // Once the pan has landed, the view holds however many frames follow.
      await drawn(page, 'caliper-view[view="s"]', 2);
      const held = await win(page, 's');
      await drawn(page, 'caliper-view[view="s"]', 5);
      expect(await win(page, 's')).toEqual(held);
      await page.mouse.dblclick(c.x, c.y);
      await expect.poll(async () => (await state(page, 's')).follow).toBe(true);
      await expect(follow).toHaveAttribute('aria-pressed', 'true');
      await expect.poll(() => win(page, 's')).toEqual(fit);
      // The header toggle is the same control.
      await follow.click();
      await expect.poll(async () => (await state(page, 's')).follow).toBe(false);
      await follow.click();
      await expect.poll(async () => (await state(page, 's')).follow).toBe(true);
      await page.locator('caliper-view[view="s"] .tool', { hasText: 'log' }).click();
      await expect.poll(async () => (await state(page, 's')).logY).toBe(true);
      await page.locator('caliper-view[view="s"]').screenshot({ path: shotPath(info, 'wasm-series-log') });

      // A histogram keeps a zoom until a double click fits it again.
      const hf = await win(page, 'hist');
      const g = await geometry(page, 'hist');
      await page.mouse.move(g.at(0, 10).x, g.at(0, 10).y);
      await page.mouse.wheel(0, -400);
      await expect.poll(async () => { const w = await win(page, 'hist'); return w.x1 - w.x0; }).toBeLessThan((hf.x1 - hf.x0) * 0.9);
      await page.mouse.dblclick(g.at(0, 10).x, g.at(0, 10).y);
      await expect.poll(() => win(page, 'hist')).toEqual(hf);
    } finally {
      srv.proc.kill();
    }
  });
});

test('data view changes and resizes do not grow what the module holds', async ({ page, producer }, info) => {
  test.skip(info.project.name !== 'webgpu', 'measured on the machine\'s GPU only');
  test.setTimeout(120_000);
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto(producer.url);
  await plates(page, 6);
  // Each data view below has another payload shape (32 x 32, then 8 x 32), so
  // every change makes the heatmap a new texture; each width makes the points
  // a new render target.
  const views = [{ source: 'attn', index: { layer: 0, head: 0 } }, { source: 'attn', index: { layer: 0, query: 0 } }];
  const shapes = ['32×32 ', '8×32 '];   // the heatmap's rows and columns in its header, for each data view
  const churn = async (n: number) => {
    for (let i = 0; i < n; i++) {
      await page.evaluate(([dv, w]) => {
        (document.querySelector('caliper-view[view="attention"]') as El).dataView = dv;
        (document.querySelector('caliper-view[view="embedding"]') as HTMLElement).style.width = `${w}px`;
      }, [views[i % 2], 560 + (i % 2) * 64] as const);
      // Until the module has made the heatmap's texture of this shape (its
      // header says so), and drawn a frame at the new width.
      await page.waitForFunction((want) => String((document.querySelector('caliper-view[view="attention"]') as El)
        .debugState().info ?? '').startsWith(want), shapes[i % 2], { polling: 'raf', timeout: 10_000 });
    }
    await drawn(page, 'caliper-view[view="attention"], caliper-view[view="embedding"]', 5);
  };
  const held = async () => {
    const a = await wasmStats(page, 'attention'), e = await wasmStats(page, 'embedding');
    return {
      heap: Number(a.heapUsed) + Number(e.heapUsed), textures: Number(a.textures) + Number(e.textures),
      allocations: Number(a.allocations) + Number(e.allocations), retired: Number(a.retiredViews) + Number(e.retiredViews),
      // How often the heatmap's texture and the points' render target were made.
      creates: [a, e].map((s) => Number(/creates=(\d+)/.exec(String(s.viewInfo))?.[1] ?? 0)),
    };
  };
  await churn(20);
  const before = await held();
  await churn(300);
  const after = await held();
  console.log(`${info.project.name} after 300 data view changes and resizes: heap ${before.heap} -> ${after.heap} bytes, textures ${before.textures} -> ${after.textures}, buffers ${before.allocations} -> ${after.allocations}, released views waiting ${after.retired}, textures made ${before.creates} -> ${after.creates}`);
  expect(after.creates[0] - before.creates[0]).toBeGreaterThan(200);
  expect(after.creates[1] - before.creates[1]).toBeGreaterThan(200);
  expect(after.textures).toBe(before.textures);
  expect(after.allocations).toBeLessThanOrEqual(before.allocations + 4);
  expect(after.retired).toBe(0);
  expect(after.heap - before.heap).toBeLessThan(256 * 1024);
});

// A scene's entity transform reaches the scene view intact through the stream (protocol 3.2):
// mock_server.py --scene3 puts five points under the entity "moved" and moves that entity by
// (10, 2, 0), and then sends the points four more times. The module decodes SCENE_TRANSFORMS on the
// socket's callback and applies it at the next frame, after it decoded the frames that follow, so its
// matrices must live in a copy of their frame, as an update's sections do. The view's fit must then
// frame the moved points' bounds, (9, 1, 0) to (11, 3, 0.5), at a finite distance, with a scale bar.
test('a scene\'s entity transform arrives intact through the stream, and the fit frames the moved points with a scale bar', async ({ page }) => {
  const p = await producerWith(['--scene3']);
  try {
    await page.goto(p.url);
    await plates(page, 7);
    expect((await state(page, 'world')).drawer).toBe('wasm');
    const geo = async () => JSON.parse(String((await wasmStats(page, 'world')).geometry ?? '{}')).world ?? null;
    const fit = async () => {
      const g = await geo();
      const o = g?.orbit;
      if (!Array.isArray(o)) return `orbit ${JSON.stringify(o ?? null)}`;
      const [x, y, z, d] = o as number[];
      const near = Math.abs(x - 10) < 1e-3 && Math.abs(y - 2) < 1e-3 && Math.abs(z - 0.25) < 1e-3;
      return near && Number.isFinite(d) && d > 0.5 && d < 50 ? 'framed' : `orbit ${JSON.stringify(o)}`;
    };
    await expect.poll(fit, { timeout: 30_000 }).toBe('framed');
    // Whatever makes a fit fail, a camera or a scale bar that is not finite or not above zero fails here.
    const g = await geo();
    const [x, y, z, d] = g.orbit as number[];
    for (const v of [x, y, z, d]) expect(Number.isFinite(v), `orbit ${JSON.stringify(g.orbit)}`).toBe(true);
    expect(d, 'the orbit\'s distance').toBeGreaterThan(0);
    const band = g.scaleBar as number[] | null;
    expect(band, 'the scale bar is drawn').not.toBeNull();
    for (const v of band!) expect(Number.isFinite(v), `band ${JSON.stringify(band)}`).toBe(true);
    expect(band![2] - band![0]).toBeGreaterThan(0);
    expect(band![3] - band![1]).toBeGreaterThan(0);
    // The bar's length, in metres and in logical pixels, finite and above zero, and its scale the
    // orbit's at the target: 2 · distance · tan(fovy / 2) per picture height, fovy 45 degrees.
    expect(Array.isArray(g.scaleBarLength), `scale bar length ${JSON.stringify(g.scaleBarLength)}`).toBe(true);
    const [metres, pixels] = g.scaleBarLength as number[];
    expect(Number.isFinite(metres) && Number.isFinite(pixels)).toBe(true);
    expect(metres).toBeGreaterThan(0);
    expect(pixels).toBeGreaterThan(0);
    const perPixel = (2 * d * Math.tan(Math.PI / 8)) / g.area[3];
    expect(Math.abs(metres / pixels - perPixel) / perPixel).toBeLessThan(1e-6);
    expect(metres, 'a bar of the moved points\' scale').toBeLessThan(10);
  } finally {
    p.proc.kill();
  }
});
