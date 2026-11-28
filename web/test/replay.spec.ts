// Static replay tests for <caliper-view> and the docs embed component (<caliper-replay>).
//
// Proves real WebAssembly + WebGPU drawing from static files only (no WebSocket,
// no API server) with paths relative to a nested docs-like page on webgpu and
// swiftshader, and verifies the static screenshot + one-line fallback when
// WebGPU or the WebAssembly module is absent (canvas2d and missing-WASM cases).

import { expect, test, type Page } from '@playwright/test';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import zlib from 'node:zlib';
import { decode, encode } from '../src/protocol/codec.js';
import {
  KIND_IDS, PROTOCOL_MINOR_3_2, SNAP_AUTO_VMAX, SNAP_AUTO_VMIN,
  type Message, type ViewEntry,
} from '../src/protocol/messages.js';
import { decodePng, REPO, shotPath, WEB } from './util.js';

const WASM_DIR = process.env.CALIPER_WASM_DIR ?? path.join(REPO, 'python', 'caliper', '_web');
const WASM_FILES = ['caliper-views-loader.js', 'caliper-views.js', 'caliper-views.wasm'];
const MNIST_RECORDING = path.join(REPO, 'examples', 'mnist', 'recording', 'recording.calrec');
const MNIST_FALLBACK = path.join(WEB, 'docs-embed', 'mnist-fallback.png');
const EMBED_BUNDLE = path.join(WEB, 'docs-embed', 'caliper-replay.js');
const VIEW_BUNDLE = path.join(WEB, 'dist', 'caliper-view.js');
const EXTRA_SHOTS = process.env.CALIPER_REPLAY_SHOTS ?? '';

/** Builds a valid non-blank RGBA PNG buffer for fallback image tests when no file is on disk yet. */
function makeFallbackPng(width = 640, height = 360): Buffer {
  const rgba = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * 4;
      const border = x < 2 || y < 2 || x >= width - 2 || y >= height - 2;
      const bar = y < 28;
      const card = x > 24 && x < width - 24 && y > 48 && y < height - 24;
      rgba[o] = border ? 75 : bar ? 32 : card ? 42 : 24;
      rgba[o + 1] = border ? 73 : bar ? 31 : card ? 40 : 23;
      rgba[o + 2] = border ? 69 : bar ? 28 : card ? 36 : 21;
      rgba[o + 3] = 255;
      if (card && Math.abs(y - (height - 60 - Math.round(140 * Math.exp(-(x - 24) / 200)))) <= 2) {
        rgba[o] = 179;
        rgba[o + 1] = 139;
        rgba[o + 2] = 89;
      }
    }
  }
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
  for (let y = 0; y < height; y++) {
    Buffer.from(rgba.buffer, y * width * 4, width * 4).copy(raw, y * (width * 4 + 1) + 1);
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', head),
    chunk('IDAT', zlib.deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

function concatFrames(msgs: Message[]): Buffer {
  return Buffer.concat(msgs.map((m) => Buffer.from(encode(m, PROTOCOL_MINOR_3_2))));
}

/** Builds a compact synthetic .calrec recording with either 1 step or multiple steps. */
function syntheticRecording(steps: number[]): Buffer {
  const views: ViewEntry[] = [
    { id: 1, name: 'loss', kind: KIND_IDS.series, sources: [] },
    { id: 2, name: 'filters', kind: KIND_IDS.heatmap, sources: [] },
    { id: 3, name: 'embedding', kind: KIND_IDS.points, sources: [] },
    { id: 4, name: 'weights_hist', kind: KIND_IDS.histogram, sources: [] },
  ];
  const msgs: Message[] = [
    {
      type: 'welcome', role: 'producer', capabilities: [], minor: PROTOCOL_MINOR_3_2,
      session: { id: 'synth', script: 'synthetic_mnist.py', pid: 0, start_unix_ms: 1700000000000 },
    },
    { type: 'views', views, timelines: [{ name: 'step', kind: 'sequence' }, { name: 'time', kind: 'time' }] },
  ];
  // One loss sample per step from the first to the last snapshot step, so the timeline holds those steps only.
  const firstStep = steps[0];
  const maxStep = steps[steps.length - 1];
  const nSeries = maxStep - firstStep + 1;
  const loss = new Float32Array(nSeries);
  const stamps = new BigInt64Array(nSeries * 2);
  for (let i = 0; i < nSeries; i++) {
    const step = firstStep + i;
    loss[i] = 2.3 * Math.exp(-step / Math.max(10, maxStep * 0.35)) + 0.08;
    stamps[i * 2] = BigInt(step);
    stamps[i * 2 + 1] = BigInt(step * 1000);
  }
  msgs.push({
    type: 'series', sub: 1, view: 'loss', start: 0, total: nSeries, stamp_ns: nSeries * 1000,
    values: loss, stamps, stamps_t: 2,
  });
  steps.forEach((step, idx) => {
    const t = maxStep > 0 ? step / maxStep : 1;
    const h = 16, w = 32;
    const heat = new Float32Array(h * w);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        heat[y * w + x] = Math.sin((x / w) * Math.PI * 2 + t * 2.5) * Math.cos((y / h) * Math.PI * 2) * (0.2 + 0.8 * t);
      }
    }
    msgs.push({
      type: 'snapshot', sub: 2, view: 'filters', seq: idx + 1, stamp_ns: step * 1000, kind: KIND_IDS.heatmap,
      data_view: {}, data: heat, shape: [h, w], colormap: 2, flags: SNAP_AUTO_VMIN | SNAP_AUTO_VMAX,
      vmin: -1, vmax: 1, stats: [-1, 1], axes: [],
      stamp: new BigInt64Array([BigInt(step), BigInt(step * 1000)]),
    });
    const nPts = 200;
    const pts = new Float32Array(nPts * 3);
    const aux = new Float32Array(nPts);
    for (let i = 0; i < nPts; i++) {
      const cls = i % 10;
      const a = (2 * Math.PI * cls) / 10;
      const spread = 1.4 - 1.1 * t;
      const jitterX = Math.sin(i * 12.9898) * spread;
      const jitterY = Math.cos(i * 78.233) * spread;
      const jitterZ = Math.sin(i * 37.719) * spread;
      pts[i * 3] = Math.cos(a) * 3.0 * t + jitterX;
      pts[i * 3 + 1] = Math.sin(a) * 3.0 * t + jitterY;
      pts[i * 3 + 2] = (cls % 2 ? 1 : -1) * 1.2 * t + jitterZ;
      aux[i] = cls;
    }
    msgs.push({
      type: 'snapshot', sub: 3, view: 'embedding', seq: idx + 1, stamp_ns: step * 1000, kind: KIND_IDS.points,
      data_view: {}, data: pts, shape: [nPts, 3], aux, colormap: 0, flags: 0, vmin: 0, vmax: 9, stats: [], axes: [],
      points_dims: 3, label_count: 10, stamp: new BigInt64Array([BigInt(step), BigInt(step * 1000)]),
    });
    const bins = new Float32Array(32);
    for (let b = 0; b < 32; b++) {
      const u = (b - 15.5) / (4 + 4 * (1 - t));
      bins[b] = Math.exp(-0.5 * u * u) * 100;
    }
    msgs.push({
      type: 'snapshot', sub: 4, view: 'weights_hist', seq: idx + 1, stamp_ns: step * 1000, kind: KIND_IDS.histogram,
      data_view: {}, data: bins, shape: [32], colormap: 0, flags: 0, vmin: 0, vmax: 1, stats: [-1.5, 1.5], axes: [],
      stamp: new BigInt64Array([BigInt(step), BigInt(step * 1000)]),
    });
  });
  return concatFrames(msgs);
}

/** Every step that any view of a recording holds data for: snapshot stamps and series sample stamps. */
function recordedSteps(bytes: Buffer): number[] {
  const steps = new Set<number>();
  for (let pos = 0; pos < bytes.length;) {
    const len = bytes.readUInt32LE(pos + 12);
    const msg = decode(new Uint8Array(bytes.buffer, bytes.byteOffset + pos, len));
    if (msg.type === 'snapshot' && msg.stamp) steps.add(Number(msg.stamp[0]));
    if (msg.type === 'series' && msg.stamps && msg.stamps_t) {
      for (let i = 0; i < msg.values.length; i++) steps.add(Number(msg.stamps[i * msg.stamps_t]));
    }
    pos += len;
  }
  return [...steps].sort((a, b) => a - b);
}

interface StaticServer {
  origin: string;
  requests: string[];
  close(): void;
}

/** Pure static HTTP GET server (zero WebSockets, zero API endpoints), simulating GitHub Pages. */
async function startStaticSite(routes: Record<string, { type: string; body: Buffer | string }>): Promise<StaticServer> {
  const requests: string[] = [];
  const srv = http.createServer((req, res) => {
    const pathname = decodeURIComponent(new URL(req.url ?? '/', 'http://127.0.0.1').pathname);
    requests.push(`${req.method} ${pathname}`);
    if (req.method !== 'GET') {
      res.writeHead(405).end();
      return;
    }
    const entry = routes[pathname];
    if (!entry) {
      res.writeHead(404).end();
      return;
    }
    res.writeHead(200, { 'content-type': entry.type, 'cache-control': 'no-store' });
    res.end(entry.body);
  });
  await new Promise<void>((ok) => srv.listen(0, '127.0.0.1', ok));
  const port = (srv.address() as { port: number }).port;
  return { origin: `http://127.0.0.1:${port}`, requests, close: () => srv.close() };
}

function wasmRoutes(prefix: string): Record<string, { type: string; body: Buffer }> {
  const out: Record<string, { type: string; body: Buffer }> = {};
  for (const name of WASM_FILES) {
    out[`${prefix}/${name}`] = {
      type: name.endsWith('.wasm') ? 'application/wasm' : 'text/javascript; charset=utf-8',
      body: fs.readFileSync(path.join(WASM_DIR, name)),
    };
  }
  return out;
}

function saveShot(info: import('@playwright/test').TestInfo, name: string, png: Buffer): void {
  fs.writeFileSync(shotPath(info, name), png);
  if (EXTRA_SHOTS) {
    fs.mkdirSync(EXTRA_SHOTS, { recursive: true });
    fs.writeFileSync(path.join(EXTRA_SHOTS, `${name}-${info.project.name}.png`), png);
  }
}

/** Counts distinct RGB colors inside a PNG to confirm the capture is not a blank box. */
function distinctColors(png: Buffer): number {
  const px = decodePng(png);
  const seen = new Set<number>();
  for (let i = 0; i < px.data.length; i += 4) {
    seen.add((px.data[i] << 16) | (px.data[i + 1] << 8) | px.data[i + 2]);
  }
  return seen.size;
}

/** Counts pixels whose RGB channels differ by more than 4 levels between two same-sized PNGs. */
function differingPixels(a: Buffer, b: Buffer): number {
  const pa = decodePng(a), pb = decodePng(b);
  if (pa.width !== pb.width || pa.height !== pb.height) return -1;
  let diff = 0;
  for (let i = 0; i < pa.data.length; i += 4) {
    if (
      Math.abs(pa.data[i] - pb.data[i]) > 4 ||
      Math.abs(pa.data[i + 1] - pb.data[i + 1]) > 4 ||
      Math.abs(pa.data[i + 2] - pb.data[i + 2]) > 4
    ) diff++;
  }
  return diff;
}

/** Waits until every child <caliper-view> inside <caliper-replay> has drawn at least minFrames WASM frames. */
async function waitReplayLive(page: Page, minFrames = 32, timeout = 90_000): Promise<Record<string, unknown>> {
  await page.waitForFunction((minF) => {
    const el = document.querySelector('caliper-replay') as (HTMLElement & { debugState(): Record<string, unknown> }) | null;
    if (!el) return false;
    const s = el.debugState();
    if (s.mode !== 'live') return false;
    const views = s.views as { drawer?: string; renderer?: string; wasmReady?: boolean; wasmStats?: { frames?: number; updates?: number; error?: string } }[];
    return views.length > 0 && views.every((v) =>
      v.drawer === 'wasm' && v.renderer === 'wasm' && v.wasmReady === true &&
      Number(v.wasmStats?.updates ?? 0) >= 1 && Number(v.wasmStats?.frames ?? 0) >= minF &&
      !v.wasmStats?.error,
    );
  }, minFrames, { timeout, polling: 100 });
  return page.evaluate(() => (document.querySelector('caliper-replay') as HTMLElement & { debugState(): Record<string, unknown> }).debugState());
}

/** Waits until every child WASM view advances by at least one update and draws a fresh frame after a scrub. */
async function waitReplayStepDrawn(page: Page, prevUpdates: number[], prevFrames: number[], timeout = 60_000): Promise<Record<string, unknown>> {
  await page.waitForFunction(([u0, f0]) => {
    const el = document.querySelector('caliper-replay') as HTMLElement & { debugState(): Record<string, unknown> };
    const views = el.debugState().views as { wasmStats?: { updates?: number; frames?: number } }[];
    return views.length === (u0 as number[]).length && views.every((v, i) =>
      Number(v.wasmStats?.updates ?? 0) > (u0 as number[])[i] &&
      Number(v.wasmStats?.frames ?? 0) >= (f0 as number[])[i] + 2,
    );
  }, [prevUpdates, prevFrames] as const, { timeout, polling: 100 });
  return page.evaluate(() => (document.querySelector('caliper-replay') as HTMLElement & { debugState(): Record<string, unknown> }).debugState());
}

test('docs embed replays the recorded MNIST run via static files and WASM on WebGPU, or shows static fallback without WebGPU', async ({ page }, info) => {
  test.setTimeout(180_000);
  test.skip(!WASM_FILES.every((f) => fs.existsSync(path.join(WASM_DIR, f))), `needs WebAssembly views in ${WASM_DIR}`);

  const hasMnist = fs.existsSync(MNIST_RECORDING);
  const recBytes = hasMnist ? fs.readFileSync(MNIST_RECORDING) : syntheticRecording([0, 20, 60, 120]);
  const fallbackBytes = fs.existsSync(MNIST_FALLBACK) ? fs.readFileSync(MNIST_FALLBACK) : makeFallbackPng();

  // Serve from a nested docs path so every asset URL is strictly relative to the page.
  const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>First Look</title>
<style>body { margin: 0; padding: 16px; background: #12110f; }</style>
</head>
<body>
<script type="module" src="../../../assets/caliper-replay/caliper-replay.js"></script>
<caliper-replay
  src="../../../assets/caliper-replay/mnist/recording.calrec"
  wasm-base="../../../assets/caliper-replay/wasm/"
  fallback="../../../assets/caliper-replay/mnist/fallback.png"
  height="260"
></caliper-replay>
</body>
</html>`;

  const site = await startStaticSite({
    '/docs/introduction/first-look/index.html': { type: 'text/html; charset=utf-8', body: html },
    '/assets/caliper-replay/caliper-replay.js': { type: 'text/javascript; charset=utf-8', body: fs.readFileSync(EMBED_BUNDLE) },
    '/assets/caliper-replay/mnist/recording.calrec': { type: 'application/octet-stream', body: recBytes },
    '/assets/caliper-replay/mnist/fallback.png': { type: 'image/png', body: fallbackBytes },
    ...wasmRoutes('/assets/caliper-replay/wasm'),
  });

  try {
    const errors: string[] = [];
    const wsUrls: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('websocket', (ws) => wsUrls.push(ws.url()));
    await page.setViewportSize({ width: 1200, height: 960 });
    await page.goto(`${site.origin}/docs/introduction/first-look/index.html`);

    if (info.project.name === 'canvas2d') {
      // No WebGPU: must show the static screenshot + one line of text, never a blank box, and never fetch WASM.
      await page.waitForFunction(() => {
        const el = document.querySelector('caliper-replay') as (HTMLElement & { debugState(): Record<string, unknown> }) | null;
        return el?.debugState().mode === 'fallback';
      }, null, { timeout: 15_000 });
      const state = await page.evaluate(() => {
        const el = document.querySelector('caliper-replay') as HTMLElement & { debugState(): Record<string, unknown> };
        const img = el.shadowRoot!.querySelector('.fallback-img') as HTMLImageElement;
        const note = el.shadowRoot!.querySelector('.fallback-note') as HTMLElement;
        return {
          debug: el.debugState(),
          imgComplete: img.complete,
          naturalWidth: img.naturalWidth,
          naturalHeight: img.naturalHeight,
          imgBox: img.getBoundingClientRect().toJSON(),
          noteBox: note.getBoundingClientRect().toJSON(),
        };
      });
      expect(state.debug.fallbackShown).toBe(true);
      expect(state.debug.fallbackImgVisible).toBe(true);
      expect(state.debug.fallbackText).toMatch(/WebGPU/);
      expect(state.imgComplete).toBe(true);
      expect(state.naturalWidth).toBeGreaterThan(0);
      expect(state.naturalHeight).toBeGreaterThan(0);
      expect((state.imgBox as { height: number }).height).toBeGreaterThan(100);
      expect((state.noteBox as { height: number }).height).toBeGreaterThan(10);
      expect(site.requests.some((r) => r.includes('caliper-views.wasm'))).toBe(false);
      expect(wsUrls).toEqual([]);
      expect(errors).toEqual([]);

      const shot = await page.locator('caliper-replay').screenshot();
      saveShot(info, 'replay-fallback', shot);
      expect(distinctColors(shot)).toBeGreaterThan(3);
      return;
    }

    // WebGPU (metal) and SwiftShader: real WASM views draw every recorded view with no server and no WebSocket.
    const firstState = await waitReplayLive(page);
    const steps = firstState.steps as number[];
    expect(steps.length).toBeGreaterThan(1);
    // The timeline holds every recorded step of any view, not only the snapshot steps.
    expect(steps).toEqual(recordedSteps(recBytes));
    expect(firstState.scrubberVisible).toBe(true);
    expect(firstState.fallbackShown).toBe(false);
    // On load the replay sits on the first step.
    expect(firstState.stepIndex).toBe(0);
    expect(firstState.currentStep).toBe(steps[0]);
    expect(firstState.playing).toBe(false);
    expect(wsUrls).toEqual([]);
    expect(site.requests.some((r) => r === 'GET /assets/caliper-replay/wasm/caliper-views.wasm')).toBe(true);
    expect(site.requests.some((r) => r === 'GET /assets/caliper-replay/mnist/recording.calrec')).toBe(true);

    type ViewState = { view: string; kind: string; info: string; window?: unknown; drawer: string; renderer: string; wasmStats: { updates: number; frames: number; gfxErrors?: string } };
    const views = firstState.views as ViewState[];
    if (hasMnist) {
      expect(views.map((v) => v.view)).toEqual(['loss', 'accuracy', 'conv1.filters', 'conv1.act', 'embedding', 'fc1.weight.hist']);
    }
    for (const v of views) {
      expect(v.drawer, v.view).toBe('wasm');
      expect(v.renderer, v.view).toBe('wasm');
      expect(v.wasmStats.gfxErrors ?? '', v.view).toBe('');
    }

    // The transport bar has Play, Reset, the range and the step, and Reset looks like Play.
    const transport = await page.evaluate(() => {
      const root = document.querySelector('caliper-replay')!.shadowRoot!;
      const look = (sel: string) => {
        const cs = getComputedStyle(root.querySelector(sel)!);
        return ['font-family', 'font-size', 'color', 'background-color', 'border-top', 'border-radius', 'height', 'padding', 'min-width']
          .map((p) => cs.getPropertyValue(p));
      };
      return {
        order: [...root.querySelector('.transport')!.children].map((c) => c.className),
        resetText: root.querySelector('.reset')!.textContent,
        stepText: root.querySelector('.step')!.textContent,
        playLook: look('.play'),
        resetLook: look('.reset'),
      };
    });
    expect(transport.order).toEqual(['play', 'reset', 'range', 'step']);
    expect(transport.resetText).toBe('Reset');
    expect(transport.stepText).toBe(`Step ${steps[0]} / ${steps[steps.length - 1]}`);
    expect(transport.resetLook).toEqual(transport.playLook);

    const shotFirst = await page.locator('caliper-replay').screenshot();
    saveShot(info, 'replay-mnist-step0', shotFirst);
    expect(distinctColors(shotFirst)).toBeGreaterThan(20);

    // Scrub to the last step via the range slider and confirm every WASM view updates and pixels change.
    const updatesAtFirst = views.map((v) => Number(v.wasmStats.updates));
    const framesAtFirst = views.map((v) => Number(v.wasmStats.frames));
    await page.evaluate(() => {
      const el = document.querySelector('caliper-replay')!;
      const range = el.shadowRoot!.querySelector('.range') as HTMLInputElement;
      range.value = range.max;
      range.dispatchEvent(new Event('input'));
    });
    const lastState = await waitReplayStepDrawn(page, updatesAtFirst, framesAtFirst);
    expect(lastState.currentStep).toBe(steps[steps.length - 1]);
    const lossInfo = (s: Record<string, unknown>) => (s.views as ViewState[]).find((v) => v.view === 'loss')?.info ?? '';
    expect(lossInfo(lastState)).not.toBe(lossInfo(firstState));

    const shotLast = await page.locator('caliper-replay').screenshot();
    saveShot(info, 'replay-mnist-final', shotLast);
    expect(differingPixels(shotFirst, shotLast)).toBeGreaterThan(500);

    // Play advances one recorded step on every display frame. A frame counter that starts in the
    // same task as Play runs before the playback callback on each frame, so the counts match.
    const startIndex = await page.evaluate(() => {
      const w = window as unknown as { frameCount: number; frameId: number };
      w.frameCount = 0;
      const count = () => {
        w.frameCount++;
        w.frameId = requestAnimationFrame(count);
      };
      w.frameId = requestAnimationFrame(count);
      const el = document.querySelector('caliper-replay') as HTMLElement & { debugState(): Record<string, unknown> };
      (el.shadowRoot!.querySelector('.play') as HTMLButtonElement).click();
      return el.debugState().stepIndex;
    });
    // Play from the last step starts again at the first step.
    expect(startIndex).toBe(0);
    await page.waitForFunction(() => (window as unknown as { frameCount: number }).frameCount >= 10, null, { timeout: 30_000 });
    const played = await page.evaluate(() => {
      const w = window as unknown as { frameCount: number; frameId: number };
      const el = document.querySelector('caliper-replay') as HTMLElement & { debugState(): Record<string, unknown> };
      const s = el.debugState();
      cancelAnimationFrame(w.frameId);
      (el.shadowRoot!.querySelector('.play') as HTMLButtonElement).click();
      return { frames: w.frameCount, stepIndex: Number(s.stepIndex), playing: s.playing, paused: !el.debugState().playing };
    });
    expect(played.playing).toBe(true);
    expect(played.stepIndex).toBe(played.frames);
    expect(played.paused).toBe(true);

    // Reset during playback pauses, returns to the first step, and clears the views back to their
    // first-step state: the series hold one value again, and every view reports what it did on load.
    await page.evaluate(() => (document.querySelector('caliper-replay')!.shadowRoot!.querySelector('.play') as HTMLButtonElement).click());
    await page.waitForFunction((target) => {
      const el = document.querySelector('caliper-replay') as HTMLElement & { debugState(): Record<string, unknown> };
      return Number(el.debugState().stepIndex) >= target;
    }, played.stepIndex + 5, { timeout: 30_000 });
    const beforeReset = await page.evaluate(() => (document.querySelector('caliper-replay') as HTMLElement & { debugState(): Record<string, unknown> }).debugState());
    expect(beforeReset.playing).toBe(true);
    expect(Number(beforeReset.stepIndex)).toBeGreaterThan(played.stepIndex);
    await page.evaluate(() => (document.querySelector('caliper-replay')!.shadowRoot!.querySelector('.reset') as HTMLButtonElement).click());
    const updatesBeforeReset = (beforeReset.views as ViewState[]).map((v) => Number(v.wasmStats.updates));
    const framesBeforeReset = (beforeReset.views as ViewState[]).map((v) => Number(v.wasmStats.frames));
    const resetState = await waitReplayStepDrawn(page, updatesBeforeReset, framesBeforeReset);
    expect(resetState.playing).toBe(false);
    expect(resetState.stepIndex).toBe(0);
    expect(resetState.currentStep).toBe(steps[0]);
    const viewFacts = (s: Record<string, unknown>) => (s.views as ViewState[]).map((v) => ({
      view: v.view, info: v.info, window: v.kind === 'series' ? v.window : undefined,
    }));
    await expect.poll(async () => viewFacts(await page.evaluate(() =>
      (document.querySelector('caliper-replay') as HTMLElement & { debugState(): Record<string, unknown> }).debugState())),
    { timeout: 15_000 }).toEqual(viewFacts(firstState));
    if (hasMnist) expect(lossInfo(resetState)).toMatch(/^1 steps /);

    const shotReset = await page.locator('caliper-replay').screenshot();
    saveShot(info, 'replay-mnist-reset', shotReset);
    expect(errors).toEqual([]);
  } finally {
    site.close();
  }
});

test('docs embed falls back to the static screenshot and text when WebAssembly views are missing', async ({ page }, info) => {
  const recBytes = syntheticRecording([0, 10]);
  const fallbackBytes = makeFallbackPng();
  const html = `<!doctype html><html><body style="margin:0;background:#12110f">
<script type="module" src="./caliper-replay.js"></script>
<caliper-replay src="./rec.calrec" wasm-base="./missing-wasm/" fallback="./fallback.png"
  fallback-text="Static fallback capture for browsers without WebAssembly views."></caliper-replay>
</body></html>`;
  const site = await startStaticSite({
    '/index.html': { type: 'text/html; charset=utf-8', body: html },
    '/caliper-replay.js': { type: 'text/javascript; charset=utf-8', body: fs.readFileSync(EMBED_BUNDLE) },
    '/rec.calrec': { type: 'application/octet-stream', body: recBytes },
    '/fallback.png': { type: 'image/png', body: fallbackBytes },
  });
  try {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(`${site.origin}/index.html`);
    await page.waitForFunction(() => {
      const el = document.querySelector('caliper-replay') as (HTMLElement & { debugState(): Record<string, unknown> }) | null;
      return el?.debugState().mode === 'fallback';
    }, null, { timeout: 15_000 });
    const dbg = await page.evaluate(() => (document.querySelector('caliper-replay') as HTMLElement & { debugState(): Record<string, unknown> }).debugState());
    expect(dbg.fallbackShown).toBe(true);
    expect(dbg.fallbackImgVisible).toBe(true);
    expect(dbg.fallbackText).toBe('Static fallback capture for browsers without WebAssembly views.');
    expect(errors).toEqual([]);
    const shot = await page.locator('caliper-replay').screenshot();
    saveShot(info, 'replay-missing-wasm-fallback', shot);
    expect(distinctColors(shot)).toBeGreaterThan(3);
  } finally {
    site.close();
  }
});

test('<caliper-view> takes a static recording src with a scrubber only when the recording has multiple steps', async ({ page }, info) => {
  test.setTimeout(120_000);
  test.skip(info.project.name !== 'canvas2d' && !WASM_FILES.every((f) => fs.existsSync(path.join(WASM_DIR, f))), `needs WebAssembly views in ${WASM_DIR}`);

  const multi = syntheticRecording([0, 10, 30]);
  const single = syntheticRecording([42]);
  const html = `<!doctype html><html><body style="margin:0;padding:12px;background:#181715;display:grid;gap:12px">
<script type="module" src="./caliper-view.js"></script>
<caliper-view id="multi" src="./multi.calrec" wasm-base="./wasm/" view="filters" height="240"></caliper-view>
<caliper-view id="single" src="./single.calrec" wasm-base="./wasm/" view="filters" height="240"></caliper-view>
</body></html>`;

  const site = await startStaticSite({
    '/index.html': { type: 'text/html; charset=utf-8', body: html },
    '/caliper-view.js': { type: 'text/javascript; charset=utf-8', body: fs.readFileSync(VIEW_BUNDLE) },
    '/multi.calrec': { type: 'application/octet-stream', body: multi },
    '/single.calrec': { type: 'application/octet-stream', body: single },
    ...(WASM_FILES.every((f) => fs.existsSync(path.join(WASM_DIR, f))) ? wasmRoutes('/wasm') : {}),
  });
  try {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(`${site.origin}/index.html`);
    await page.waitForFunction(() => {
      const a = document.getElementById('multi') as (HTMLElement & { debugState(): Record<string, unknown> }) | null;
      const b = document.getElementById('single') as (HTMLElement & { debugState(): Record<string, unknown> }) | null;
      return a?.debugState().status === 'connected' && b?.debugState().status === 'connected' &&
        !!a.debugState().renderer && !!b.debugState().renderer;
    }, null, { timeout: 60_000 });

    const res = await page.evaluate(() => {
      const a = document.getElementById('multi') as HTMLElement & { debugState(): Record<string, unknown> };
      const b = document.getElementById('single') as HTMLElement & { debugState(): Record<string, unknown> };
      return {
        multi: {
          drawer: a.debugState().drawer,
          renderer: a.debugState().renderer,
          scrubVisible: !(a.shadowRoot!.querySelector('.scrub') as HTMLElement).hidden,
          scrubText: a.shadowRoot!.querySelector('.scrub-step')?.textContent ?? '',
        },
        single: {
          drawer: b.debugState().drawer,
          renderer: b.debugState().renderer,
          scrubVisible: !(b.shadowRoot!.querySelector('.scrub') as HTMLElement).hidden,
        },
      };
    });

    const wantDrawer = info.project.name === 'canvas2d' ? 'ts' : 'wasm';
    const wantRenderer = info.project.name === 'canvas2d' ? 'canvas2d' : 'wasm';
    expect(res.multi.drawer).toBe(wantDrawer);
    expect(res.multi.renderer).toBe(wantRenderer);
    expect(res.multi.scrubVisible).toBe(true);
    expect(res.multi.scrubText).toBe('step 30 / 30');
    expect(res.single.drawer).toBe(wantDrawer);
    expect(res.single.renderer).toBe(wantRenderer);
    expect(res.single.scrubVisible).toBe(false);

    // Scrub #multi back to step 0 and confirm its readout updates.
    await page.evaluate(() => {
      const a = document.getElementById('multi')!;
      const r = a.shadowRoot!.querySelector('.scrub-range') as HTMLInputElement;
      r.value = '0';
      r.dispatchEvent(new Event('input'));
    });
    await expect.poll(() => page.evaluate(() =>
      document.getElementById('multi')!.shadowRoot!.querySelector('.scrub-step')?.textContent ?? '',
    )).toBe('step 0 / 30');
    expect(errors).toEqual([]);
  } finally {
    site.close();
  }
});
