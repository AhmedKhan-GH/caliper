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
//
// The scene comparison: the scene fixtures that caliper_scene_parity_app drew with the app's scene
// view and the library pieces into <fixture>/scene, drawn by the module's scene view on WebGPU and
// compared under the same rules, with the counts of the shapes of the same frame and the hits of
// picks (the last test of this file). Each comparison is saved as scene-parity-<fixture>.png.

import { test, expect } from '@playwright/test';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import zlib from 'node:zlib';
import { REPO, serve, shotPath, WEB } from './util.js';

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
  kind: Uint8Array;   // for each pixel: 0 interior, 1 edge, 2 an edge pixel that differs and took neither side
}

// The rules of the plan over two pictures and their ID targets; `bloom` marks the pixels that the
// glow lights (3 levels in place of 2). With `noIdIsEdge`, a pixel without an ID in either picture is
// an edge pixel: in a scene, the ground and the shapes that the built-in draw draws write no ID.
function compare(size: number, app: Uint8Array, appIds: Uint32Array, web: Uint8Array, webIds: Uint32Array,
  bloom: Uint8Array, noIdIsEdge = false): Comparison {
  const c: Comparison = { interior: 0, edge: 0, edgeShare: 0, edgeOver: 0, edgeDiffer: 0, differShare: 0, over: 0, overBloom: 0,
    maxDiff: 0, maxDiffBloom: 0, idMismatch: 0, first: '', diff: new Uint8Array(size * size * 4),
    kind: new Uint8Array(size * size) };
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
      const interior = sameAround(appIds, x, y) && sameAround(webIds, x, y) &&
        !(noIdIsEdge && (appIds[2 * p] === 0 || webIds[2 * p] === 0));
      const o = 4 * p;
      c.diff[o + 3] = 255;
      const glow = bloom[p] === 1;
      const limit = glow ? 3 : 2;
      if (!interior) {
        c.edge++;
        const differs = d > limit && !(tookSide(web, app, x, y, limit) || tookSide(app, web, x, y, limit));
        if (d > limit) c.edgeOver++;
        if (differs) c.edgeDiffer++;
        c.kind[p] = differs ? 2 : 1;
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
    // Opt-in for a check on a slow machine: CALIPER_CPU_THROTTLE=6 slows the page six times.
    const throttle = Number(process.env.CALIPER_CPU_THROTTLE ?? 1);
    if (throttle > 1) {
      const cdp = await page.context().newCDPSession(page);
      await cdp.send('Emulation.setCPUThrottlingRate', { rate: throttle });
    }
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
      // The arcs' pixels are counted inside the page, in the same task as each frame of the module:
      // right after the module's animation callback has drawn, the canvas is copied into a 2D canvas
      // (a WebGPU canvas keeps its texture only until the end of that task). A screenshot from the
      // test process has no tie to a frame, and a slow software adapter may stop the arcs
      // (program.draw_over_time) or sit between presents when it lands. The wait ends when the count
      // passes the bound while the two programs are refused, or when the arcs leave "running", or at
      // the frame cap.
      const probe = document.createElement('canvas');
      const pg = probe.getContext('2d', { willReadFrequently: true })!;
      const count = () => {
        probe.width = canvas.width;   // the module sizes the canvas after its first frames
        probe.height = canvas.height;
        pg.clearRect(0, 0, probe.width, probe.height);
        pg.drawImage(canvas, 0, 0);
        const d = pg.getImageData(0, 0, probe.width, probe.height).data;
        let n = 0;
        for (let i = 0; i < d.length; i += 4)
          if (d[i + 1] > 150 && d[i + 2] > 150 && d[i + 1] + d[i + 2] > d[i] + 200) n++;
        return n;
      };
      let since = -1, largest = 0, largestFrame = -1, stopped = '', ticks = 0;
      const frames = () => Number(m.caliperStats?.frames ?? 0);
      const afterFrame = () => {
        const arcs = of('tower_arcs');
        if (arcs?.state === 'running') {
          if (since < 0) since = frames();
          if (frames() >= since + 1 && largest <= 500) {
            const n = count();
            if (n > largest) { largest = n; largestFrame = frames(); }
          }
        } else if (since >= 0 && !stopped) stopped = `${arcs?.state} ${arcs?.code}`;
      };
      const original = window.requestAnimationFrame.bind(window);
      let hooked = true;
      window.requestAnimationFrame = (cb: FrameRequestCallback) => original((t) => { cb(t); if (hooked) afterFrame(); });
      const ready = () => largest > 500 && of('broken_pattern')?.state === 'refused' && of('wrong_pin')?.state === 'refused';
      for (; ticks < 6000 && !stopped && !ready(); ticks++) await new Promise((ok) => original(() => ok(null)));
      hooked = false;
      window.requestAnimationFrame = original;
      return { list: list(), state: m.caliperStats?.state ?? '', errors: m.caliperStats?.gfxErrors ?? '',
        largest, largestFrame, stopped, ticks, frame: frames(), since };
    }, { base: `${site.origin}/`, src: url });
    console.log(`[${info.project.name}] live programs: ${JSON.stringify(programs.list.map((p) => [p.id, p.state, p.code]))}`);
    const of = (id: string) => programs.list.find((p) => p.id === id);
    console.log(`[${info.project.name}] live tower: largest count ${programs.largest} at module frame ` +
      `${programs.largestFrame} (arcs running from frame ${programs.since}, ended at frame ${programs.frame}, ` +
      `${programs.ticks} animation frames)`);
    if (programs.largest <= 500) {
      const why = programs.stopped ? `the arcs stopped with "${programs.stopped}" after ${programs.frame - programs.since} frames`
        : of('tower_arcs')?.state === 'running' ? 'the frame cap ended the wait while the arcs ran' : 'the arcs never ran';
      expect.soft(programs.largest, `the scene view draws the arcs: ${why}; largest count of the arcs' colors seen ` +
        `${programs.largest} (module frame ${programs.largestFrame})`).toBeGreaterThan(500);
    }
    expect(of('tower_arcs')?.state === 'running' || programs.stopped !== '', `tower_arcs: ${of('tower_arcs')?.message}`).toBe(true);
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
    expect(programs.largest, 'the scene view draws the arcs').toBeGreaterThan(500);
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

// The scene comparison (tests/draw_parity/scene_parity.h). caliper_scene_parity_app drew the scene
// fixtures of the scene rows (tests/scene/primitive_fixtures.h and "through-pieces") through the app's
// scene view, whose shapes the library pieces draw, on Metal into <fixture>/scene. Here the module
// draws each fixture through its own scene view on WebGPU, and the test compares the first frame of
// each: the picture and the pieces' IDs under the rules above, where a pixel without a piece's ID (the
// ground, or a shape that the built-in draw keeps) counts as an edge pixel; the counts of the shapes of
// that frame with the reason of each fallback; the program of each pick id; and the hit of a pick at
// each of the app's points. A module made with caliperConfig.shapes "builtin" draws every shape with
// the built-in draw, as CALIPER_SCENE_SHAPES=builtin does in the app.
const SCENE = path.join(FIXTURE, 'scene');
const SCENE_SIZE = 256;

interface SceneFacts {
  frame: number; draw: string; pieces: number; builtin: number; fallbacks: Record<string, number>;
  programs: Record<string, string>; picks: { at: [number, number]; hit?: string; result?: number; position?: number[] }[];
}
interface SceneRun { report: string; facts: string; color: string; ids: string }

// Runs the module's side of the scene comparison for each config, one after the other, in one module
// made with `moduleConfig`; waits on what the module publishes each frame, never on a fixed time.
async function sceneRuns(page: import('@playwright/test').Page, origin: string, configs: object[],
  moduleConfig: Record<string, string>): Promise<{ runs: SceneRun[]; errors: string; adapter: string; why: string }> {
  return page.evaluate(async ({ base, configs, moduleConfig }) => {
    type M = { caliperStats?: Record<string, string>; caliperSceneParity?: { facts: string; color: Uint8Array; ids: Uint8Array };
               _caliper_views_scene_parity(config: number): number; _caliper_views_destroy(): void;
               _free(p: number): void; stringToNewUTF8(s: string): number };
    const { preloadCaliperViews } = await import(base + 'caliper-views-loader.js');
    const { factory, wasm } = await preloadCaliperViews(base);
    const canvas = document.createElement('canvas');
    canvas.style.width = '64px';
    canvas.style.height = '64px';
    document.body.append(canvas);
    const m: M = await factory({
      canvas, caliperConfig: { src: '', ...moduleConfig },
      instantiateWasm: (imports: WebAssembly.Imports, receive: (i: WebAssembly.Instance, m: WebAssembly.Module) => void) => {
        WebAssembly.instantiate(wasm as WebAssembly.Module, imports).then((i) => receive(i, wasm));
        return {};
      },
      print: () => {}, printErr: () => {},
    });
    const until = async (f: () => boolean) => {
      for (let i = 0; i < 6000 && !f(); i++) await new Promise((ok) => requestAnimationFrame(() => ok(null)));
      return f();
    };
    const base64 = (b: Uint8Array) => {
      let s = '';
      for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode(...b.subarray(i, i + 0x8000));
      return btoa(s);
    };
    const out = { runs: [] as SceneRun[], errors: '', adapter: '', why: '' };
    if (!(await until(() => Number(m.caliperStats?.gfxReady) === 1))) {
      out.why = `no layer: ${JSON.stringify(m.caliperStats ?? {})}`;
      return out;
    }
    for (const config of configs) {
      const q = m.stringToNewUTF8(JSON.stringify(config));
      const started = m._caliper_views_scene_parity(q);
      m._free(q);
      if (started) await until(() => !!m.caliperStats?.sceneParity);
      const r = m.caliperSceneParity;
      out.runs.push({ report: m.caliperStats?.sceneParity ?? 'fail no report', facts: r?.facts ?? '{}',
        color: r ? base64(r.color) : '', ids: r ? base64(r.ids) : '' });
    }
    out.errors = m.caliperStats?.gfxErrors ?? '';
    out.adapter = m.caliperStats?.featureLevel ?? '';
    m._caliper_views_destroy();
    canvas.remove();
    return out;
  }, { base: `${origin}/`, configs, moduleConfig });
}

test('the browser module draws the scene fixtures with the library pieces as the app does, with the same counts and picks', async ({ page }, info) => {
  test.skip(info.project.name === 'canvas2d', 'the WebAssembly views need a WebGPU adapter');
  test.skip(!fs.existsSync(path.join(SCENE, 'report.txt')),
    fs.existsSync(path.join(SCENE, 'skipped.txt')) ? `the app's side skipped: ${fs.readFileSync(path.join(SCENE, 'skipped.txt'), 'utf8')}`
      : 'needs the scene side of caliper_scene_parity_app in CALIPER_DRAW_PARITY_DIR/scene');
  test.setTimeout(300_000);
  const names = fs.readdirSync(SCENE).filter((f) => f.endsWith('.json')).map((f) => f.slice(0, -5)).sort();
  expect(names.length, 'the fixtures that the app drew').toBeGreaterThanOrEqual(8);
  const app: Record<string, SceneFacts> = {};
  for (const n of names) app[n] = JSON.parse(fs.readFileSync(path.join(SCENE, `${n}.json`), 'utf8')) as SceneFacts;
  const site = await serve(WEB, (name) => (FILES.includes(name.slice(1)) ? path.join(WASM, name.slice(1)) : null));
  try {
    await page.goto(`${site.origin}/test/blank.html`);
    const got = await sceneRuns(page, site.origin, names.map((n) => ({ fixture: n, points: app[n].picks.map((p) => p.at) })), {});
    expect(got.why, 'the module started').toBe('');
    console.log(`[${info.project.name}] scene comparison on a ${got.adapter} device`);
    expect(got.errors, 'WebGPU reported no error').toBe('');
    const size = SCENE_SIZE;
    // The interior pixels of each shape's piece, by the primitive of its program, across the fixtures.
    const shapes: Record<string, { interior: number; maxDiff: number; edgeDiffer: number }> = {};
    let asked = 0, agreed = 0, positionGap = 0;
    for (let i = 0; i < names.length; i++) {
      const name = names[i];
      const run = got.runs[i];
      const a = app[name];
      expect(run.report, `${name}: the module's report`).toMatch(/^ok /);
      const web = JSON.parse(run.facts) as SceneFacts;
      // The counts of the same frame.
      expect(web.frame, `${name}: the frame of the facts`).toBe(a.frame);
      expect(web.draw, name).toBe('pieces');
      expect({ pieces: web.pieces, builtin: web.builtin, fallbacks: web.fallbacks }, `${name}: the shapes of frame ${a.frame}`)
        .toEqual({ pieces: a.pieces, builtin: a.builtin, fallbacks: a.fallbacks });
      expect(web.programs, `${name}: the program of each pick id`).toEqual(a.programs);
      // The pictures and the IDs of the pieces.
      const webColor = new Uint8Array(Buffer.from(run.color, 'base64'));
      const webIdBytes = Buffer.from(run.ids, 'base64');
      expect(webColor.length, name).toBe(size * size * 4);
      expect(webIdBytes.length, name).toBe(size * size * 8);
      const webIds = new Uint32Array(webIdBytes.buffer, webIdBytes.byteOffset, size * size * 2);
      const appColor = new Uint8Array(fs.readFileSync(path.join(SCENE, `${name}.color`)));
      const appIdBytes = fs.readFileSync(path.join(SCENE, `${name}.id`));
      const appIds = new Uint32Array(appIdBytes.buffer, appIdBytes.byteOffset, size * size * 2);
      const c = compare(size, appColor, appIds, webColor, webIds, new Uint8Array(size * size), true);
      // By shape: an interior pixel's largest difference, and the differing edge pixels of a piece's ID.
      const shapeOf = (p: number) => (a.programs[String(appIds[2 * p])] ?? 'none').split('#').pop() ?? 'none';
      for (let p = 0; p < size * size; p++) {
        if (appIds[2 * p] === 0) continue;
        const key = shapeOf(p);
        const s = (shapes[key] ??= { interior: 0, maxDiff: 0, edgeDiffer: 0 });
        let d = 0;
        for (let k = 0; k < 4; k++) d = Math.max(d, Math.abs(appColor[4 * p + k] - webColor[4 * p + k]));
        if (c.kind[p] === 2) s.edgeDiffer++;
        else if (c.kind[p] === 0) { s.interior++; s.maxDiff = Math.max(s.maxDiff, d); }
      }
      const side = new Uint8Array(size * 3 * size * 4);
      for (let y = 0; y < size; y++) {
        side.set(appColor.subarray(y * size * 4, (y + 1) * size * 4), (y * 3 * size) * 4);
        side.set(webColor.subarray(y * size * 4, (y + 1) * size * 4), (y * 3 * size + size) * 4);
        side.set(c.diff.subarray(y * size * 4, (y + 1) * size * 4), (y * 3 * size + 2 * size) * 4);
      }
      const png = encodePng(size * 3, size, side);
      fs.writeFileSync(shotPath(info, `scene-parity-${name}`), png);
      if (CAPTURES) {
        fs.mkdirSync(CAPTURES, { recursive: true });
        fs.writeFileSync(path.join(CAPTURES, `scene-parity-${name}-${info.project.name}.png`), png);
      }
      // The picks at the app's points.
      expect(web.picks.length, `${name}: the picks`).toBe(a.picks.length);
      let first = '';
      for (let k = 0; k < a.picks.length; k++) {
        const ap = a.picks[k], wp = web.picks[k];
        asked++;
        const same = ap.hit === wp.hit && ap.result === wp.result;
        if (same) agreed++;
        else if (!first) first = `at (${ap.at}): app ${ap.hit ?? `result ${ap.result}`}, browser ${wp.hit ?? `result ${wp.result}`}`;
        if (ap.position && wp.position)
          for (let j = 0; j < 3; j++) positionGap = Math.max(positionGap, Math.abs(ap.position[j] - wp.position[j]));
      }
      console.log(`[${info.project.name}] scene ${name}: frame ${web.frame}, ${web.pieces} shapes by pieces, ${web.builtin} by the ` +
        `built-in ${JSON.stringify(web.fallbacks)}; interior ${c.interior}, edge ${c.edge} (${c.edgeOver} over the levels, ` +
        `${c.edgeDiffer} of those on neither side), largest interior difference ${c.maxDiff} levels (${c.over} over 2), ` +
        `differing edge pixels ${(100 * c.differShare).toFixed(3)} % of the image, other IDs ${c.idMismatch}; picks ` +
        `${web.picks.filter((p, k) => p.hit === a.picks[k].hit && p.result === a.picks[k].result).length} of ${a.picks.length} agree` +
        (first ? `, first other: ${first}` : '') + (c.first ? `; first over 2: ${c.first}` : ''));
      expect(c.over, `${name}: interior pixels over 2 levels; first ${c.first}`).toBe(0);
      expect(c.differShare, `${name}: the share of edge pixels that differ`).toBeLessThanOrEqual(0.01);
      expect(c.idMismatch, `${name}: interior pixels with another ID`).toBe(0);
      expect(first, `${name}: every pick gives the app's hit`).toBe('');
    }
    for (const [key, s] of Object.entries(shapes).sort())
      console.log(`[${info.project.name}] scene shape ${key}: ${s.interior} interior pixels, largest difference ${s.maxDiff} ` +
        `levels, ${s.edgeDiffer} differing edge pixels`);
    console.log(`[${info.project.name}] scene picks: ${agreed} of ${asked} give the app's hit; largest gap of a picked ` +
      `position ${positionGap.toExponential(3)}`);
    expect(asked, 'picks at the app\'s points').toBeGreaterThan(50);
    expect(agreed).toBe(asked);
    expect(Object.keys(shapes).sort(), 'the shapes that pieces drew')
      .toEqual(expect.arrayContaining(['arrows', 'boxes', 'camera', 'image', 'lines', 'mesh', 'points']));
  } finally {
    site.close();
  }
});
