// The pixel comparison of a whole dashboard between the app and the browser's spec page: the
// Attention Arc Tower as a short seeded run left it, recorded once into tests/draw_parity/dashboard
// (the saved spec and the panel.data answer of every leaf, as the app gave them to a viewer). The
// app's side, tests/draw_parity/dashboard_parity_app.cpp (ctest caliper_dashboard_parity_app),
// draws the recording with the app's spec runtime on Metal at 1280 by 750; here the spec page draws
// the same answers, served by the mock of the app, in a window of 1280 by 800, whose area under the
// page's bar is 1280 by 750. Nothing trains: both sides draw the recorded lines and the recorded
// resource versions of the two programs.
//
// The picture is tied to a frame: once every program of the module reports that it draws the
// recorded step, a callback that runs after every other animation callback of one frame copies the
// page's canvas and each module canvas into one picture, in the same task as the module's frame (a
// WebGPU canvas keeps its texture only until the end of that task).
//
// The rules, as tests/draw_parity uses them, without ID targets, so a pixel's neighborhood tells
// its kind, applied to each panel, control and tab strip on its own (the app's rectangles), and to
// the rest of the area:
//   - an interior pixel, whose 3 by 3 neighborhood is one color (within the levels) in both
//     pictures, differs by at most 2 levels in each channel, everywhere;
//   - an edge pixel may differ, but the edge pixels that differ beyond the levels and take neither
//     side (their color is not within the levels of a color of the other picture's neighborhood)
//     are at most 1 % of the region, as in a program picture; the plot panels have their own bound
//     (PLOT_EDGE_BOUND says why);
//   - text: the app draws its text with ImGui's rasterizer and the page with the browser's, so the
//     comparison masks the box of every string the page draws (globalThis.caliperTextLog, grown by
//     2 pixels) in both pictures, and checks the strings and their places on their own: the
//     layout's rectangles of every panel and control are the app's to the pixel, the tab labels are
//     the app's, and each panel's title, the legend's names and the toggle's text are drawn by the
//     page at the app's places. The labels that the programs' text tables give are drawn by the
//     same ImGui code in both, over each scene; each side reports their strings and boxes (the app's
//     report.json, the module's stats), the test checks the same strings in boxes within a pixel of
//     the app's, and masks those boxes in both pictures, as the page's text.
//
// The picture of each side and their difference are saved as parity-dashboard.png under the shots
// folder and CALIPER_DRAW_PARITY_CAPTURES. Where the app's side had no Metal device
// (dashboard_out/skipped.txt) or did not run, the page still draws the recording and checks its own
// picture, and the comparison is skipped with that reason. The canvas2d project has no WebGPU, so the
// programs cannot draw there, and it is skipped.

import { test, expect, type Page } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { mockApp, openSpec, VIEWS_FILES, type Json, type MockApp } from './spec-mock.js';
import { REPO, shotPath } from './util.js';

const WASM = process.env.CALIPER_WASM_DIR ?? path.join(REPO, 'python', 'caliper', '_web');
const FIXTURE = path.join(REPO, 'tests', 'draw_parity', 'dashboard');
const APP = process.env.CALIPER_DASHBOARD_PARITY_DIR ?? path.join(REPO, 'cmake-build-release', 'tests', 'draw_parity', 'dashboard_out');
const CAPTURES = process.env.CALIPER_DRAW_PARITY_CAPTURES ?? '';
const W = 1280, H = 750;
const LIMIT = 2;
/** The share of a region whose edge pixels may differ and take neither side: the rule of the program pictures. */
const EDGE_BOUND = 0.01;
/**
 * The same share in a plot panel. Its lines come from two rasterizers, ImPlot's anti-aliased polylines (triangle
 * strips with a feathered fringe) in the app and Canvas 2D strokes in the page, whose coverage of a 1.25 px line
 * differs along every line, so most pixels of every line are edge pixels that take neither side. The frame, the
 * grid, the ticks, the legend and the layout keep the rule of every region (interior within the levels), and the
 * names, places and legend entries are checked as strings. Measured on the recording: 3.22 % (WebGPU) and 3.24 %
 * (SwiftShader) in the chart of eight lines, 1.15 % and 1.14 % in the chart of one; the bound is 4 %.
 */
const PLOT_EDGE_BOUND = 0.04;

test.beforeEach(({}, info) => test.skip(info.project.name === 'canvas2d', 'the programs draw with WebGPU, which the canvas2d project disables'));
test.skip(!VIEWS_FILES.every((f) => fs.existsSync(path.join(WASM, f))), `needs the WebAssembly views in ${WASM}`);
test.skip(!fs.existsSync(path.join(FIXTURE, 'spec.json')), 'needs the recording in tests/draw_parity/dashboard');

function encodePng(width: number, height: number, rgba: Uint8Array): Buffer {
  const table = Array.from({ length: 256 }, (_, n) => {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    return c >>> 0;
  });
  const crc = (b: Buffer) => {
    let c = 0xffffffff;
    for (const x of b) c = table[(c ^ x) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  };
  const chunk = (type: string, data: Buffer) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const sum = Buffer.alloc(4);
    sum.writeUInt32BE(crc(body));
    return Buffer.concat([len, body, sum]);
  };
  const head = Buffer.alloc(13);
  head.writeUInt32BE(width, 0);
  head.writeUInt32BE(height, 4);
  head[8] = 8;
  head[9] = 6;
  const raw = Buffer.alloc((width * 4 + 1) * height);
  for (let y = 0; y < height; y++) Buffer.from(rgba.buffer, rgba.byteOffset + y * width * 4, width * 4).copy(raw, y * (width * 4 + 1) + 1);
  return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), chunk('IHDR', head),
    chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

interface TextBox { s: string; x: number; y: number; w: number; h: number }
interface Composite { rgba: string; texts: TextBox[]; state: Json; programs: Json; frame: number; labels: Record<string, TextBox[]>; scenes: Record<string, number[]> }

/** The recording as the mock app's applets and answers. */
function recording(): { app: MockApp; spec: Json; step: number } {
  const spec = JSON.parse(fs.readFileSync(path.join(FIXTURE, 'spec.json'), 'utf8')) as Json;
  const data: Record<string, Json> = {};
  let step = -1;
  for (const f of fs.readdirSync(path.join(FIXTURE, 'panels'))) {
    if (!f.endsWith('.json')) continue;
    const a = JSON.parse(fs.readFileSync(path.join(FIXTURE, 'panels', f), 'utf8')) as Json;
    for (const p of a.data?.programs ?? []) for (const r of p.resources ?? []) step = Math.max(step, r.step);
    data[f.slice(0, -5)] = a;
  }
  return { app: { specs: { [spec.applet.id]: spec }, data: { [spec.applet.id]: data }, requests: [] }, spec, step };
}

/** The page's picture at the first frame at which every program draws `step`, with the text it drew. */
async function composite(page: Page, step: number, programs: number, consoleLines: string[]): Promise<Composite> {
  await page.waitForFunction(([step, n]) => {
    const s = (window as any).caliperSpec?.debugState();
    if (!s || !s.panels.length || !s.panels.every((p: any) => p.drawn || p.primitive === 'scene')) return false;
    const progs = Object.values(s.programs).flatMap((m: any) => m.programs);
    const mods = (window as any).caliperSpec.programs;
    let running = 0;
    for (const m of (mods.mounts as Map<string, any>).values()) {
      const list = m.handle?.stats.programs;
      // Drawing the recorded step; slow software adapters warn on frame time and keep running.
      for (const p of (typeof list === 'string' ? JSON.parse(list) : list ?? []) as any[])
        if (p.state === 'running' && p.step === step && p.frames >= 1) running++;
    }
    return progs.length === n && running === n && s.controls.every((c: any) => c.text || c.kind === 'readout');
  }, [step, programs], { timeout: 90_000, polling: 100 }).catch(async (e) => {
    // Why no frame came: the page's programs, each module's counters and the console.
    const why = await page.evaluate(() => {
      const w = window as any, s = w.caliperSpec?.debugState();
      const mods = w.caliperSpec ? [...(w.caliperSpec.programs.mounts as Map<string, any>).values()].map((m) =>
        ({ panel: m.panel, failed: m.failed, starting: !!m.starting, stats: m.handle?.stats ?? null })) : [];
      return { panels: s?.panels.map((p: any) => [p.id, p.drawn, p.state]), programs: s?.programs, mods };
    }).catch(() => ({}));
    throw new Error(`${e}: ${JSON.stringify(why).slice(0, 4000)}; console: ${consoleLines.slice(-30).join(' | ').slice(0, 3000)}`);
  });
  return page.evaluate(() => new Promise<Composite>((done) => {
    const w = window as any;
    w.caliperSpec.stop();
    w.__dashboard.snaps.clear();
    w.__dashboard.ready = () => {
      for (const m of (w.caliperSpec.programs.mounts as Map<string, any>).values()) {
        const mc = m.handle?.canvas as HTMLCanvasElement | undefined;
        if (!mc || m.box.style.display === 'none') continue;
        if (!w.__dashboard.snaps.has(m.panel)) return false;
      }
      return w.caliperTextLog.length > 0;
    };
    w.__dashboard.compose = () => {
      const area = document.getElementById('area')!.getBoundingClientRect();
      const d = window.devicePixelRatio || 1;
      const c = document.createElement('canvas');
      c.width = Math.round(area.width * d);
      c.height = Math.round(area.height * d);
      const g = c.getContext('2d', { willReadFrequently: true })!;
      g.drawImage(document.getElementById('canvas') as HTMLCanvasElement, 0, 0);
      for (const m of (w.caliperSpec.programs.mounts as Map<string, any>).values()) {
        const mc = m.handle?.canvas as HTMLCanvasElement | undefined;
        if (!mc || m.box.style.display === 'none') continue;
        const snap = w.__dashboard.snaps.get(m.panel) as HTMLCanvasElement | undefined;
        const r = mc.getBoundingClientRect();
        g.drawImage(snap ?? mc, Math.round((r.left - area.left) * d), Math.round((r.top - area.top) * d), Math.round(r.width * d), Math.round(r.height * d));
      }
      const px = g.getImageData(0, 0, c.width, c.height).data;
      let bin = '';
      for (let i = 0; i < px.length; i += 0x8000) bin += String.fromCharCode(...px.subarray(i, i + 0x8000));
      return btoa(bin);
    };
    // The labels that each module drew over its scene (its stats, CSS pixels of its canvas), in the area's pixels,
    // and where each module's canvas lies.
    const modules = () => {
      const area = document.getElementById('area')!.getBoundingClientRect();
      const labels: Record<string, { s: string; x: number; y: number; w: number; h: number }[]> = {};
      const scenes: Record<string, number[]> = {};
      for (const m of (w.caliperSpec.programs.mounts as Map<string, any>).values()) {
        const mc = m.handle?.canvas as HTMLCanvasElement | undefined;
        if (!mc || m.box.style.display === 'none') continue;
        const r = mc.getBoundingClientRect(), x = r.left - area.left, y = r.top - area.top;
        scenes[m.panel] = [x, y, r.width, r.height];
        let list: [string, number, number, number, number][] = [];
        try { list = JSON.parse(String(m.handle.stats.pageLabels ?? '[]')); } catch { /* not yet published */ }
        labels[m.panel] = list.map(([s, x0, y0, x1, y1]) => ({ s, x: x + x0, y: y + y0, w: x1 - x0, h: y1 - y0 }));
      }
      return { labels, scenes };
    };
    w.__dashboard.done = (rgba: string, frame: number) => done({ rgba, frame, texts: [...w.caliperTextLog],
      state: w.caliperSpec.debugState(), programs: w.caliperSpec.programs.debug(), ...modules() });
    // The page draws once more, so the text log holds this picture's text.
    w.caliperTextLog.length = 0;
    w.caliperSpec.redraw();
    requestAnimationFrame(() => { w.__dashboard.armed = true; });
  }));
}

interface Counts { area: number; interior: number; over: number; maxDiff: number; edge: number; edgeDiffer: number; masked: number; first: string }
interface Comparison extends Counts { differShare: number; diff: Uint8Array; regions: Counts[] }

const counts = (): Counts => ({ area: 0, interior: 0, over: 0, maxDiff: 0, edge: 0, edgeDiffer: 0, masked: 0, first: '' });

/** Compares the pictures, counting each pixel in the whole and in its region (region[p], an index into `names`). */
function compare(app: Uint8Array, web: Uint8Array, mask: Uint8Array, region: Uint8Array, names: string[]): Comparison {
  const c: Comparison = { ...counts(), differShare: 0, diff: new Uint8Array(W * H * 4), regions: names.map(counts) };
  const flat = (a: Uint8Array, x: number, y: number) => {
    const p = 4 * (y * W + x);
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        const q = 4 * (ny * W + nx);
        for (let k = 0; k < 3; k++) if (Math.abs(a[p + k] - a[q + k]) > LIMIT) return false;
      }
    return true;
  };
  const tookSide = (a: Uint8Array, b: Uint8Array, x: number, y: number) => {
    const p = 4 * (y * W + x);
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        const q = 4 * (ny * W + nx);
        let d = 0;
        for (let k = 0; k < 3; k++) d = Math.max(d, Math.abs(a[p + k] - b[q + k]));
        if (d <= LIMIT) return true;
      }
    return false;
  };
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const p = y * W + x, o = 4 * p, r = c.regions[region[p]];
      c.diff[o + 3] = 255;
      c.area++;
      r.area++;
      if (mask[p]) {
        c.masked++;
        r.masked++;
        c.diff[o + 2] = 90;   // text, masked: dark blue
        continue;
      }
      let d = 0;
      for (let k = 0; k < 3; k++) d = Math.max(d, Math.abs(app[o + k] - web[o + k]));
      if (!(flat(app, x, y) && flat(web, x, y))) {
        c.edge++;
        r.edge++;
        const differs = d > LIMIT && !(tookSide(web, app, x, y) || tookSide(app, web, x, y));
        if (differs) {
          c.edgeDiffer++;
          r.edgeDiffer++;
          c.diff[o] = 255;
          c.diff[o + 1] = 220;
        } else c.diff[o] = c.diff[o + 1] = c.diff[o + 2] = d > LIMIT ? 110 : 40;
        continue;
      }
      c.interior++;
      r.interior++;
      c.maxDiff = Math.max(c.maxDiff, d);
      r.maxDiff = Math.max(r.maxDiff, d);
      if (d > LIMIT) {
        c.over++;
        r.over++;
        c.diff[o] = 255;
        const at = `(${x}, ${y}) differs by ${d} levels, app ${[...app.slice(o, o + 3)]}, browser ${[...web.slice(o, o + 3)]}`;
        if (!c.first) c.first = at;
        if (!r.first) r.first = at;
      } else c.diff[o] = c.diff[o + 1] = c.diff[o + 2] = Math.min(255, d * 40);
    }
  c.differShare = c.edgeDiffer / (W * H);
  return c;
}

test('the Arc Tower dashboard draws in the browser as in the app, on the same recorded data', async ({ page }, info) => {
  test.setTimeout(240_000);
  const { app, spec, step } = recording();
  const nPrograms = (spec.program ?? []).length;
  expect(step, 'the recording holds resource versions').toBeGreaterThan(0);
  await page.addInitScript(() => {
    const w = window as any;
    w.caliperTextLog = [];
    w.__dashboard = { armed: false, compose: null, done: null, ready: null, snaps: new Map() };
    // Each module's WebGPU canvas keeps its presented surface texture only within the animation
    // frame that rendered it, and a slow software adapter can skip or delay a swapchain texture on
    // one tick; copy each presented canvas synchronously after its callback and finish once all
    // shown modules have presented an opaque frame.
    const original = window.requestAnimationFrame.bind(window), cancel = window.cancelAnimationFrame.bind(window);
    let last = 0, lastBatch = -1, frame = 0;
    let check: HTMLCanvasElement | null = null;
    let cg: CanvasRenderingContext2D | null = null;
    const snapshotModules = () => {
      const mounts = w.caliperSpec?.programs?.mounts as Map<string, any> | undefined;
      if (!mounts) return;
      if (!check) {
        check = document.createElement('canvas');
        check.width = 1;
        check.height = 1;
        cg = check.getContext('2d', { willReadFrequently: true });
      }
      if (!cg) return;
      for (const m of mounts.values()) {
        const mc = m.handle?.canvas as HTMLCanvasElement | undefined;
        if (!mc || m.box.style.display === 'none' || mc.width < 1 || mc.height < 1) continue;
        cg.clearRect(0, 0, 1, 1);
        cg.drawImage(mc, 0, 0, 1, 1, 0, 0, 1, 1);
        if (cg.getImageData(0, 0, 1, 1).data[3] !== 255) continue;
        let probe = w.__dashboard.snaps.get(m.panel) as HTMLCanvasElement | undefined;
        if (!probe) probe = document.createElement('canvas');
        if (probe.width !== mc.width || probe.height !== mc.height) {
          probe.width = mc.width;
          probe.height = mc.height;
        }
        const pg = probe.getContext('2d', { willReadFrequently: true })!;
        pg.clearRect(0, 0, probe.width, probe.height);
        pg.drawImage(mc, 0, 0);
        w.__dashboard.snaps.set(m.panel, probe);
      }
    };
    const capture = (t: number) => {
      last = 0;
      frame++;
      const d = w.__dashboard;
      if (!d.armed || !d.compose) return;
      snapshotModules();
      if (d.ready && !d.ready()) {
        last = original(capture);
        lastBatch = t;
        return;
      }
      d.armed = false;
      d.done(d.compose(), frame);
    };
    window.requestAnimationFrame = (cb: FrameRequestCallback) => original((t) => {
      cb(t);
      if (!w.__dashboard.armed) return;
      snapshotModules();
      if (last && lastBatch === t) cancel(last);
      else if (last) return;   // registered in an earlier frame: it runs in this one, after this callback
      last = original(capture);
      lastBatch = t;
    });
  });
  const consoleLines: string[] = [];
  page.on('console', (m) => consoleLines.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => consoleLines.push(`[pageerror] ${e.message}`));
  await mockApp(page, app, WASM);
  await page.setViewportSize({ width: W, height: 800 });
  await openSpec(page, spec.applet.id);
  const shot = await composite(page, step, nPrograms, consoleLines);
  const web = new Uint8Array(Buffer.from(shot.rgba, 'base64'));
  expect(web.length).toBe(W * H * 4);

  // The page alone: every leaf drawn, both programs drawn by the module, the control's state shown.
  const state = shot.state;
  for (const [id, m] of Object.entries(shot.programs) as [string, Json][]) {
    expect(m.mounted, `${id}: the module is mounted`).toBe(true);
    for (const p of m.programs) expect(p.module, `${id}/${p.id}: ${p.code}`).toBe('running');
  }
  expect(state.controls.length).toBe((spec.control ?? []).length);
  for (const c of state.controls) expect(c.enabled).toBe(false);
  expect(app.requests.every((r) => r.name === 'spec.get' || r.name === 'panel.data' || r.name.startsWith('GET caliper-views'))).toBe(true);

  // The app's side.
  const skipped = path.join(APP, 'skipped.txt'), report = path.join(APP, 'app', 'report.json');
  const write = (name: string, png: Buffer) => {
    fs.writeFileSync(shotPath(info, name), png);
    if (CAPTURES) {
      fs.mkdirSync(CAPTURES, { recursive: true });
      fs.writeFileSync(path.join(CAPTURES, `${name}.png`), png);
    }
  };
  if (fs.existsSync(skipped) || !fs.existsSync(report)) {
    const why = fs.existsSync(skipped) ? fs.readFileSync(skipped, 'utf8').trim() : `no app side in ${APP}`;
    write('parity-dashboard-browser', encodePng(W, H, web));
    console.log(`[${info.project.name}] dashboard: comparison skipped: ${why}`);
    info.annotations.push({ type: 'skip', description: `comparison skipped: ${why}` });
    return;
  }
  const rep = JSON.parse(fs.readFileSync(report, 'utf8')) as Json;
  expect([rep.width, rep.height]).toEqual([W, H]);
  expect(rep.step).toBe(step);
  const appPx = new Uint8Array(fs.readFileSync(path.join(APP, 'app', 'dashboard.rgba')));
  expect(appPx.length).toBe(W * H * 4);

  // The layout: every leaf on the app's rectangle, the same tab strips and labels.
  const rect = (r: Json) => [r.x, r.y, r.w, r.h];
  const pageLeaves = [...state.panels, ...state.controls];
  for (const l of rep.layout as Json[]) {
    const mine = pageLeaves.find((p: Json) => p.id === l.id);
    expect(mine, `the page shows ${l.id}`).toBeDefined();
    expect(rect(mine.rect), `${l.id} on the app's rectangle`).toEqual(rect(l.rect));
  }
  expect(state.tabs.map((t: Json) => t.labels)).toEqual((rep.tabs as Json[]).map((t) => t.labels));
  // The strings: every panel's title at the app's place for it (8 px in, centred in the 24 px strip),
  // the legend's names, the toggle's text.
  const drawn = (s: string) => shot.texts.filter((t) => t.s === s);
  for (const l of rep.layout as Json[]) {
    if (l.control || l.tab) continue;
    const title = state.panels.find((p: Json) => p.id === l.id).title;
    const at = drawn(title).find((t) => Math.abs(t.x - (l.rect.x + 8)) <= 1 && t.y >= l.rect.y && t.y < l.rect.y + 24);
    expect(at, `the title "${title}" of ${l.id} at the app's place`).toBeDefined();
  }
  for (const [panel, lines] of Object.entries(state.lines) as [string, { name: string }[]][])
    if (lines.length > 1) for (const l of lines) expect(drawn(l.name).length, `${panel}: the legend names ${l.name}`).toBeGreaterThan(0);
  for (const c of state.controls) expect(drawn(c.text.split('  ')[0]).length + drawn(c.text).length).toBeGreaterThan(0);
  // The labels that the programs' text tables give, which each module draws over its scene with the app's own
  // placement code: each scene where the app's drew, the same strings, each in a box within a pixel of the app's.
  const appLabels: Record<string, TextBox[]> = {};
  for (const [panel, list] of Object.entries((rep.labels ?? {}) as Record<string, Json[]>)) {
    appLabels[panel] = list.map((l) => ({ s: l.text, x: l.x0, y: l.y0, w: l.x1 - l.x0, h: l.y1 - l.y0 }));
    const at = (rep.scenes as Record<string, number[]>)[panel], mine = shot.scenes[panel];
    expect(mine, `the module of ${panel} draws`).toBeDefined();
    for (let k = 0; k < 4; k++) expect(Math.abs(mine[k] - at[k]), `${panel}: its scene on the app's rectangle`).toBeLessThanOrEqual(0.5);
    const web = shot.labels[panel] ?? [];
    expect(web.map((l) => l.s).sort(), `${panel}: the labels' strings`).toEqual(appLabels[panel].map((l) => l.s).sort());
    for (const a of appLabels[panel]) {
      const near = web.find((b) => b.s === a.s && Math.abs(b.x - a.x) <= 1 && Math.abs(b.y - a.y) <= 1 &&
        Math.abs(b.w - a.w) <= 1 && Math.abs(b.h - a.h) <= 1);
      expect(near, `${panel}: the label "${a.s}" at the app's (${a.x}, ${a.y})`).toBeDefined();
    }
    console.log(`[${info.project.name}] dashboard labels ${panel}: ${appLabels[panel].length} in the app, ${web.length} in the browser, ` +
      'every string in a box within 1 px of the app\'s');
  }
  expect(Object.keys(appLabels).sort(), 'the scene panels whose labels the app reports').toEqual(
    state.panels.filter((p: Json) => p.primitive === 'scene').map((p: Json) => p.id).sort());

  // The pixels, with the text masked: the page's strings and the labels over the scenes, of both sides.
  const mask = new Uint8Array(W * H);
  const masked = [...shot.texts, ...Object.values(appLabels).flat(), ...Object.values(shot.labels).flat()];
  for (const t of masked)
    for (let y = Math.max(0, Math.floor(t.y) - 2); y < Math.min(H, Math.ceil(t.y + t.h) + 2); y++)
      for (let x = Math.max(0, Math.floor(t.x) - 2); x < Math.min(W, Math.ceil(t.x + t.w) + 2); x++) mask[y * W + x] = 1;
  // The regions: each panel and control on the app's rectangle, each tab strip, and the rest (the splitters).
  const names = ['rest'], kinds = ['rest'], region = new Uint8Array(W * H);
  const paint = (r: { x: number; y: number; w: number; h: number }, name: string, kind: string) => {
    const i = names.length;
    names.push(name);
    kinds.push(kind);
    for (let y = Math.max(0, Math.round(r.y)); y < Math.min(H, Math.round(r.y + r.h)); y++)
      for (let x = Math.max(0, Math.round(r.x)); x < Math.min(W, Math.round(r.x + r.w)); x++) region[y * W + x] = i;
  };
  for (const l of rep.layout as Json[]) {
    const primitive = l.control ? 'control' : state.panels.find((p: Json) => p.id === l.id)?.primitive;
    paint(l.rect, l.id, primitive === 'scene' ? 'program' : primitive === 'series' ? 'plot' : String(primitive));
  }
  for (const t of rep.tabs as Json[]) paint(t.bar, `tabs ${t.path}`, 'tabs');
  const c = compare(appPx, web, mask, region, names);
  const side = new Uint8Array(W * 3 * H * 4);
  for (let y = 0; y < H; y++) {
    side.set(appPx.subarray(y * W * 4, (y + 1) * W * 4), (y * 3 * W) * 4);
    side.set(web.subarray(y * W * 4, (y + 1) * W * 4), (y * 3 * W + W) * 4);
    side.set(c.diff.subarray(y * W * 4, (y + 1) * W * 4), (y * 3 * W + 2 * W) * 4);
  }
  write('parity-dashboard', encodePng(3 * W, H, side));
  console.log(`[${info.project.name}] dashboard: interior ${c.interior} px, over ${LIMIT} levels ${c.over} ` +
    `(max ${c.maxDiff}); edge ${c.edge} px, of which ${c.edgeDiffer} took neither side ` +
    `(${(100 * c.differShare).toFixed(3)} % of the picture); text masked ${c.masked} px in ${shot.texts.length} strings; ` +
    `frame ${shot.frame}${c.first ? `; first interior difference ${c.first}` : ''}`);
  // Each region on its own, with its bound.
  const failed: string[] = [];
  c.regions.forEach((r, i) => {
    const bound = kinds[i] === 'plot' ? PLOT_EDGE_BOUND : EDGE_BOUND, share = r.area ? r.edgeDiffer / r.area : 0;
    console.log(`[${info.project.name}] dashboard ${kinds[i]} ${names[i]}: ${r.area} px, interior ${r.interior}, ` +
      `over ${LIMIT} levels ${r.over} (max ${r.maxDiff}), edge ${r.edge}, neither side ${r.edgeDiffer} ` +
      `(${(100 * share).toFixed(3)} %, bound ${(100 * bound).toFixed(1)} %), text masked ${r.masked}`);
    if (r.over) failed.push(`${names[i]}: ${r.over} interior pixels over ${LIMIT} levels, first ${r.first}`);
    if (share > bound) failed.push(`${names[i]}: ${(100 * share).toFixed(3)} % of its edge pixels took neither side, above ${(100 * bound).toFixed(1)} %`);
  });
  expect(failed, 'every region within its rule').toEqual([]);
});
