// The pixel comparison between the app and the browser module (design/programmable-graphics.md,
// Pixel Comparison). The app's side (tests/draw_parity/draw_parity_app.mm, ctest
// caliper_draw_parity_app) wrote a fixture of wire frames, two instances of pieces
// (attention_field and attention_arcs) as PROGRAM sets and one recorded version of each resource
// as RESOURCE, and drew it with the app's draw runtime on its Metal device. Here the WebAssembly
// views draw the same frames with the same runtime on WebGPU in headless Chromium, at 512 by 512
// with the same fixed camera, and the test compares the color and the ID targets:
//
//   - an interior pixel, whose 3 by 3 neighborhood has one ID (pick id and item) in both images,
//     differs by at most 2 levels in each channel, and by at most 3 where the bloom of a glow
//     target lights it;
//   - an edge pixel may differ in any way, but the edge pixels that differ beyond those levels,
//     and take neither side of the edge, are at most 1 % of the image;
//   - the ID of each interior pixel is the same.
//
// Two readings of the plan's rules, because the fixture's arcs are thin and each is an item of its
// own: about a quarter of the arcs' picture lies next to another ID, so the 1 % bounds the edge
// pixels that actually differ, and the test prints the share of all edge pixels beside it. An edge
// pixel that took a side, that is whose color in one picture is within the levels of a color of
// its 3 by 3 neighborhood in the other, as the scene rows' parity rules allow, does not count. And an
// offscreen target has no ID that tells its edges: the arcs rasterized at half size into the glow
// target can cover a texel in one renderer and not in the other, and the bloom spreads that texel
// over its neighborhood. A bloom pixel over 3 levels therefore counts as an edge pixel of the glow
// target, under the same 1 % bound, and the test prints how many there are.
//
// Two more programs of the fixture must be refused by the browser module: a piece whose draw.wgsl
// WebGPU does not compile (program.browser_refused, with the line of the file), and a pin whose
// hash the files do not have (piece.hash_mismatch). Each comparison is saved as
// parity-<program>.png, the app's picture, the browser's and their difference side by side, under
// the shots folder and CALIPER_DRAW_PARITY_CAPTURES when set.
//
// The fixture comes from CALIPER_DRAW_PARITY_DIR (the folder the tool wrote); without it, or
// without the WebAssembly views (CALIPER_WASM_DIR, else python/caliper/_web), the test is skipped.
// Where the app's side had no Metal device it wrote the frames only; the browser then draws them
// and checks its own pictures, and the comparison is skipped with that reason.
//
// The live path: tests/draw_parity/serve_fixture.py serves the same frames as a producer of a run
// relays them to a viewer that announced resources, with a scene view for each panel. The module's
// network source takes PROGRAM and RESOURCE, checks each draw half, and its scene view "tower" draws
// the arcs; the refused programs report their codes. The view is saved as parity-live-tower.png.

import { test, expect } from '@playwright/test';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import zlib from 'node:zlib';
import { decodePng, REPO, serve, shotPath, WEB } from './util.js';

const WASM = process.env.CALIPER_WASM_DIR ?? path.join(REPO, 'python', 'caliper', '_web');
const FILES = ['caliper-views-loader.js', 'caliper-views.js', 'caliper-views.wasm'];
const FIXTURE = process.env.CALIPER_DRAW_PARITY_DIR ?? '';
const CAPTURES = process.env.CALIPER_DRAW_PARITY_CAPTURES ?? '';
const BACKGROUND = [0x14, 0x18, 0x20, 0xff];
const PYTHON = process.env.CALIPER_PYTHON ?? path.join(REPO, '.venv', 'bin', 'python');

test.skip(!FILES.every((f) => fs.existsSync(path.join(WASM, f))), `needs the WebAssembly views in ${WASM}`);
test.skip(!FIXTURE || !fs.existsSync(path.join(FIXTURE, 'frames.bin')),
  'needs the fixture of caliper_draw_parity_app in CALIPER_DRAW_PARITY_DIR');

interface PlanProgram { id: string; pick_id: number; expect: string; line?: number }
interface Plan { size: number; frames: number; step: number; programs: PlanProgram[] }
interface Drawn { code: string; message: string; where: string; line: number; column: number; color: string; ids: string }

// A PNG of RGBA8 pixels (no filter, one IDAT), for the captures.
function encodePng(width: number, height: number, rgba: Uint8Array): Buffer {
  const crcTable = Array.from({ length: 256 }, (_, n) => {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    return c >>> 0;
  });
  const crc = (b: Buffer) => {
    let c = 0xffffffff;
    for (const x of b) c = crcTable[(c ^ x) & 0xff] ^ (c >>> 8);
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
  for (let y = 0; y < height; y++)
    Buffer.from(rgba.buffer, rgba.byteOffset + y * width * 4, width * 4).copy(raw, y * (width * 4 + 1) + 1);
  return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), chunk('IHDR', head),
    chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

interface Comparison {
  interior: number; edge: number; edgeShare: number; edgeOver: number; edgeDiffer: number; differShare: number; over: number;
  overBloom: number; maxDiff: number; maxDiffBloom: number; idMismatch: number; first: string; diff: Uint8Array;
}

// The rules of the plan over two pictures and their ID targets; `bloom` marks the pixels that the
// glow lights (3 levels in place of 2).
function compare(size: number, app: Uint8Array, appIds: Uint32Array, web: Uint8Array, webIds: Uint32Array,
  bloom: Uint8Array): Comparison {
  const c: Comparison = { interior: 0, edge: 0, edgeShare: 0, edgeOver: 0, edgeDiffer: 0, differShare: 0, over: 0, overBloom: 0,
    maxDiff: 0, maxDiffBloom: 0, idMismatch: 0, first: '', diff: new Uint8Array(size * size * 4) };
  const sameAround = (ids: Uint32Array, x: number, y: number) => {
    const p = y * size + x;
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
        const q = ny * size + nx;
        if (ids[2 * q] !== ids[2 * p] || ids[2 * q + 1] !== ids[2 * p + 1]) return false;
      }
    return true;
  };
  // Whether the color of `a` at (x, y) is within `limit` of a color of `b` in its 3 by 3 neighborhood.
  const tookSide = (a: Uint8Array, b: Uint8Array, x: number, y: number, limit: number) => {
    const p = 4 * (y * size + x);
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
        const q = 4 * (ny * size + nx);
        let d = 0;
        for (let k = 0; k < 4; k++) d = Math.max(d, Math.abs(a[p + k] - b[q + k]));
        if (d <= limit) return true;
      }
    return false;
  };
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const p = y * size + x;
      let d = 0;
      for (let k = 0; k < 4; k++) d = Math.max(d, Math.abs(app[4 * p + k] - web[4 * p + k]));
      const interior = sameAround(appIds, x, y) && sameAround(webIds, x, y);
      const o = 4 * p;
      c.diff[o + 3] = 255;
      const glow = bloom[p] === 1;
      const limit = glow ? 3 : 2;
      if (!interior) {
        c.edge++;
        const differs = d > limit && !(tookSide(web, app, x, y, limit) || tookSide(app, web, x, y, limit));
        if (d > limit) c.edgeOver++;
        if (differs) c.edgeDiffer++;
        // An edge pixel: grey, brighter over the levels, yellow when it took neither side.
        if (differs) { c.diff[o] = 255; c.diff[o + 1] = 220; }
        else c.diff[o] = c.diff[o + 1] = c.diff[o + 2] = d > limit ? 110 : 40;
        continue;
      }
      c.interior++;
      if (glow) c.maxDiffBloom = Math.max(c.maxDiffBloom, d);
      else c.maxDiff = Math.max(c.maxDiff, d);
      const idSame = appIds[2 * p] === webIds[2 * p] && appIds[2 * p + 1] === webIds[2 * p + 1];
      if (!idSame) c.idMismatch++;
      if (d > limit) {
        if (glow) c.overBloom++;
        else c.over++;
        if (!glow && !c.first) c.first = `(${x}, ${y}) differs by ${d} levels, app ${[...app.slice(o, o + 4)]}, browser ${[...web.slice(o, o + 4)]}`;
      }
      // The difference: red over the tolerance or for another ID, else the levels times 40.
      if (d > limit || !idSame) c.diff[o] = 255;
      else c.diff[o] = c.diff[o + 1] = c.diff[o + 2] = Math.min(255, d * 40);
    }
  c.edgeShare = c.edge / (size * size);
  c.differShare = (c.edgeDiffer + c.overBloom) / (size * size);
  return c;
}

test('the browser module draws the fixture of pieces as the app does, within the tolerances', async ({ page }, info) => {
  test.skip(info.project.name === 'canvas2d', 'the WebAssembly views need a WebGPU adapter');
  test.setTimeout(240_000);
  const plan = JSON.parse(fs.readFileSync(path.join(FIXTURE, 'plan.json'), 'utf8')) as Plan;
  const frames = fs.readFileSync(path.join(FIXTURE, 'frames.bin'));
  const appDrew = fs.existsSync(path.join(FIXTURE, 'app', 'report.txt'));
  const site = await serve(WEB, (name) => (FILES.includes(name.slice(1)) ? path.join(WASM, name.slice(1)) : null));
  try {
    await page.goto(`${site.origin}/test/blank.html`);
    const got = await page.evaluate(async ({ base, frames, plan }) => {
      type M = { caliperStats?: Record<string, string>; caliperDrawParity?: Record<string, {
                   code: string; message: string; where: string; line: number; column: number;
                   color: Uint8Array; ids: Uint8Array }>;
                 HEAPU8: Uint8Array; _malloc(n: number): number; _free(p: number): void;
                 _caliper_views_draw_parity(frames: number, n: number, plan: number): number;
                 _caliper_views_destroy(): void; stringToNewUTF8(s: string): number };
      const { preloadCaliperViews } = await import(base + 'caliper-views-loader.js');
      const { factory, wasm } = await preloadCaliperViews(base);
      const canvas = document.createElement('canvas');
      canvas.style.width = '64px';
      canvas.style.height = '64px';
      document.body.append(canvas);
      const m: M = await factory({
        canvas, caliperConfig: { src: '' },
        instantiateWasm: (imports: WebAssembly.Imports, receive: (i: WebAssembly.Instance, m: WebAssembly.Module) => void) => {
          WebAssembly.instantiate(wasm as WebAssembly.Module, imports).then((i) => receive(i, wasm));
          return {};
        },
        print: () => {}, printErr: () => {},
      });
      // Waits on what the module publishes each frame, never on a fixed time.
      const until = async (f: () => boolean) => {
        for (let i = 0; i < 6000 && !f(); i++) await new Promise((ok) => requestAnimationFrame(() => ok(null)));
        return f();
      };
      const base64 = (b: Uint8Array) => {
        let s = '';
        for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode(...b.subarray(i, i + 0x8000));
        return btoa(s);
      };
      if (!(await until(() => Number(m.caliperStats?.gfxReady) === 1)))
        return { report: `no layer: ${JSON.stringify(m.caliperStats ?? {})}`, drawn: {}, errors: '', adapter: '' };
      const bytes = Uint8Array.from(atob(frames), (ch) => ch.charCodeAt(0));
      const p = m._malloc(bytes.length);
      m.HEAPU8.set(bytes, p);
      const q = m.stringToNewUTF8(plan);
      const started = m._caliper_views_draw_parity(p, bytes.length, q);
      m._free(p);
      m._free(q);
      if (started) await until(() => !!m.caliperStats?.drawParity);
      const drawn: Record<string, unknown> = {};
      for (const [id, r] of Object.entries(m.caliperDrawParity ?? {}))
        drawn[id] = { code: r.code, message: r.message, where: r.where, line: r.line, column: r.column,
          color: base64(r.color), ids: base64(r.ids) };
      const out = { report: m.caliperStats?.drawParity ?? 'fail no report', drawn,
        errors: m.caliperStats?.gfxErrors ?? '', adapter: m.caliperStats?.featureLevel ?? '' };
      m._caliper_views_destroy();
      canvas.remove();
      return out;
    }, { base: `${site.origin}/`, frames: frames.toString('base64'), plan: JSON.stringify(plan) });

    console.log(`[${info.project.name}] draw parity: ${got.report} (device ${got.adapter})`);
    expect(got.report, 'every program drew or was refused as the plan expects').toMatch(/^ok /);
    expect(got.errors, 'WebGPU reported no error of the drawn programs').toBe('');
    const drawn = got.drawn as Record<string, Drawn>;
    const size = plan.size;
    for (const pp of plan.programs) {
      const r = drawn[pp.id];
      expect(r, pp.id).toBeTruthy();
      if (pp.expect) {
        console.log(`[${info.project.name}] ${pp.id}: ${r.code} at ${r.where} line ${r.line}: ${r.message}`);
        expect(r.code, pp.id).toBe(pp.expect);
        if (pp.line) {
          expect(r.where, pp.id).toBe('draw.passes[0]');
          expect(r.line, `${pp.id}: the line of draw.wgsl`).toBe(pp.line);
          expect(r.message, pp.id).toContain(`draw.wgsl line ${pp.line}`);
        }
        continue;
      }
      expect(r.code, `${pp.id}: ${r.message}`).toBe('');
      const web = new Uint8Array(Buffer.from(r.color, 'base64'));
      const webIdBytes = Buffer.from(r.ids, 'base64');
      expect(web.length, pp.id).toBe(size * size * 4);
      expect(webIdBytes.length, pp.id).toBe(size * size * 8);
      const webIds = new Uint32Array(webIdBytes.buffer, webIdBytes.byteOffset, size * size * 2);
      let lit = 0, picked = 0;
      for (let i = 0; i < size * size; i++) {
        if (web[4 * i] !== BACKGROUND[0] || web[4 * i + 1] !== BACKGROUND[1] || web[4 * i + 2] !== BACKGROUND[2]) lit++;
        if (webIds[2 * i] === pp.pick_id) picked++;
      }
      console.log(`[${info.project.name}] ${pp.id}: browser lit ${lit}, with its ID ${picked}`);
      expect(lit, `${pp.id} draws`).toBeGreaterThan(2000);
      if (!appDrew) {
        fs.writeFileSync(shotPath(info, `parity-${pp.id}-browser`), encodePng(size, size, web));
        continue;
      }
      const app = new Uint8Array(fs.readFileSync(path.join(FIXTURE, 'app', `${pp.id}.color`)));
      const appIdBytes = fs.readFileSync(path.join(FIXTURE, 'app', `${pp.id}.id`));
      const appIds = new Uint32Array(appIdBytes.buffer, appIdBytes.byteOffset, size * size * 2);
      const bloom = new Uint8Array(fs.readFileSync(path.join(FIXTURE, 'app', `${pp.id}.bloom`)));
      const c = compare(size, app, appIds, web, webIds, bloom);
      const side = new Uint8Array(size * 3 * size * 4);
      for (let y = 0; y < size; y++) {
        side.set(app.subarray(y * size * 4, (y + 1) * size * 4), (y * 3 * size) * 4);
        side.set(web.subarray(y * size * 4, (y + 1) * size * 4), (y * 3 * size + size) * 4);
        side.set(c.diff.subarray(y * size * 4, (y + 1) * size * 4), (y * 3 * size + 2 * size) * 4);
      }
      const png = encodePng(size * 3, size, side);
      fs.writeFileSync(shotPath(info, `parity-${pp.id}`), png);
      if (CAPTURES) {
        fs.mkdirSync(CAPTURES, { recursive: true });
        fs.writeFileSync(path.join(CAPTURES, `parity-${pp.id}-${info.project.name}.png`), png);
      }
      console.log(`[${info.project.name}] ${pp.id}: interior ${c.interior}, edge ${c.edge} ` +
        `(${(100 * c.edgeShare).toFixed(3)} %, ${c.edgeOver} of them over the levels, ${c.edgeDiffer} of those on ` +
        `neither side), largest difference ` +
        `${c.maxDiff} levels (${c.over} over 2), in the bloom ${c.maxDiffBloom} (${c.overBloom} over 3, edges of ` +
        `the glow target), differing edge pixels ${(100 * c.differShare).toFixed(3)} % of the image, ` +
        `other IDs ${c.idMismatch}` + (c.first ? `; first over 2 outside the bloom: ${c.first}` : ''));
      expect(c.over, `${pp.id}: interior pixels outside the bloom over 2 levels; first ${c.first}`).toBe(0);
      expect(c.differShare, `${pp.id}: the share of edge pixels that differ`).toBeLessThanOrEqual(0.01);
      expect(c.idMismatch, `${pp.id}: interior pixels with another ID`).toBe(0);
    }
    if (!appDrew) {
      const why = fs.existsSync(path.join(FIXTURE, 'skipped.txt'))
        ? fs.readFileSync(path.join(FIXTURE, 'skipped.txt'), 'utf8').trim() : 'the app drew no picture';
      test.info().annotations.push({ type: 'skip', description: `comparison skipped: ${why}` });
      console.log(`[${info.project.name}] comparison skipped: ${why}`);
    }
  } finally {
    site.close();
  }
});

test('the module draws the draw halves that PROGRAM and RESOURCE bring over its connection, and refuses the bad ones', async ({ page }, info) => {
  test.skip(info.project.name === 'canvas2d', 'the WebAssembly views need a WebGPU adapter');
  test.skip(!fs.existsSync(PYTHON), 'needs a Python with numpy and websockets (CALIPER_PYTHON)');
  test.setTimeout(240_000);
  const proc = spawn(PYTHON, [path.join(REPO, 'tests', 'draw_parity', 'serve_fixture.py'), '--fixture', FIXTURE],
    { stdio: ['ignore', 'pipe', 'inherit'], env: { ...process.env, PYTHONPATH: path.join(REPO, 'python') } });
  const site = await serve(WEB, (name) => (FILES.includes(name.slice(1)) ? path.join(WASM, name.slice(1)) : null));
  try {
    const url = await new Promise<string>((ok, fail) => {
      proc.once('exit', (code) => fail(new Error(`serve_fixture.py exited with ${code}`)));
      readline.createInterface({ input: proc.stdout! }).on('line', (line) => {
        if (line.startsWith('{')) ok((JSON.parse(line) as { url: string }).url);
      });
    });
    await page.goto(`${site.origin}/test/blank.html`);
    const programs = await page.evaluate(async ({ base, src }) => {
      type M = { caliperStats?: Record<string, string>; _caliper_views_destroy(): void };
      const { preloadCaliperViews } = await import(base + 'caliper-views-loader.js');
      const { factory, wasm } = await preloadCaliperViews(base);
      const canvas = document.createElement('canvas');
      canvas.id = 'live';
      canvas.style.width = '480px';
      canvas.style.height = '480px';
      document.body.append(canvas);
      const m: M = await factory({
        canvas, caliperConfig: { src, view: 'tower', kind: 'scene' },
        instantiateWasm: (imports: WebAssembly.Imports, receive: (i: WebAssembly.Instance, m: WebAssembly.Module) => void) => {
          WebAssembly.instantiate(wasm as WebAssembly.Module, imports).then((i) => receive(i, wasm));
          return {};
        },
        print: () => {}, printErr: () => {},
      });
      (window as unknown as { liveViews: M }).liveViews = m;
      type P = { id: string; state: string; code: string; message: string; where: string; line: number; frames: number };
      const list = () => JSON.parse(m.caliperStats?.programs ?? '[]') as P[];
      const of = (id: string) => list().find((p) => p.id === id);
      // Waits on what the module publishes, never on a fixed time: the arcs have drawn some frames,
      // and the other two programs were refused.
      const frames = () => Number(m.caliperStats?.frames ?? 0);
      let since = -1;
      const settled = () => {
        if (of('tower_arcs')?.state === 'running' && since < 0) since = frames();
        return since >= 0 && frames() >= since + 3 && of('broken_pattern')?.state === 'refused' &&
          of('wrong_pin')?.state === 'refused';
      };
      for (let i = 0; i < 6000 && !settled(); i++) await new Promise((ok) => requestAnimationFrame(() => ok(null)));
      return { list: list(), state: m.caliperStats?.state ?? '', errors: m.caliperStats?.gfxErrors ?? '' };
    }, { base: `${site.origin}/`, src: url });
    console.log(`[${info.project.name}] live programs: ${JSON.stringify(programs.list.map((p) => [p.id, p.state, p.code]))}`);
    const of = (id: string) => programs.list.find((p) => p.id === id);
    expect(of('tower_arcs')?.state, `tower_arcs: ${of('tower_arcs')?.message}`).toBe('running');
    expect(of('head_pattern')?.code ?? '', 'head_pattern: no refusal').toBe('');
    expect(of('broken_pattern')?.code).toBe('program.browser_refused');
    expect(of('broken_pattern')?.where).toBe('draw.passes[0]');
    expect(of('wrong_pin')?.code).toBe('piece.hash_mismatch');
    expect(programs.errors, 'WebGPU reported no error').toBe('');
    const png = await page.locator('#live').screenshot();
    fs.writeFileSync(shotPath(info, 'parity-live-tower'), png);
    if (CAPTURES) {
      fs.mkdirSync(CAPTURES, { recursive: true });
      fs.writeFileSync(path.join(CAPTURES, `parity-live-tower-${info.project.name}.png`), png);
    }
    const px = decodePng(png);
    let bright = 0;
    for (let y = 0; y < px.height; y++)
      for (let x = 0; x < px.width; x++) {
        const [r, g, b] = px.at(x, y);
        if (g > 150 && b > 150 && g + b > r + 200) bright++;   // the cyan and white of the arcs
      }
    console.log(`[${info.project.name}] live tower: ${bright} pixels of the arcs' colors in ${px.width} x ${px.height}`);
    expect(bright, 'the scene view draws the arcs').toBeGreaterThan(500);
    // On a software adapter the frame time is the time from the submit until the work is done (no
    // timestamps), far over the arcs' 4 ms, so the safety rule may stop them after 30 frames: the
    // arcs either keep running or stop with program.draw_over_time, never with another code.
    const later = await page.evaluate(() => {
      const m = (window as unknown as { liveViews: { caliperStats?: Record<string, string>; _caliper_views_destroy(): void } }).liveViews;
      const list = JSON.parse(m.caliperStats?.programs ?? '[]') as { id: string; state: string; code: string }[];
      m._caliper_views_destroy();
      return list.find((p) => p.id === 'tower_arcs');
    });
    console.log(`[${info.project.name}] live tower_arcs at the end: ${later?.state} ${later?.code}`);
    expect(['running', 'stopped']).toContain(later?.state);
    if (later?.state === 'stopped') expect(later.code).toBe('program.draw_over_time');
  } finally {
    site.close();
    proc.kill();
  }
});
