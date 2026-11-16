// Shared helpers of the Playwright tests: a static server for web/, the
// harness page, golden frames fetched at run time, PNG decoding and the
// screenshot folder.

import { test as base, expect, type Page, type TestInfo } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import zlib from 'node:zlib';

export const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const REPO = path.resolve(WEB, '..');
/** Screenshots go to $CALIPER_SHOTS, or test-results/shots. */
export const SHOTS = process.env.CALIPER_SHOTS ?? path.join(WEB, 'test-results', 'shots');

const TYPES: Record<string, string> = {
  '.html': 'text/html', '.js': 'text/javascript', '.woff': 'font/woff', '.json': 'application/json', '.map': 'application/json',
};

/** Serves the files under root on 127.0.0.1; returns the origin and a close function. */
export async function serve(root: string, extra?: (url: string) => string | null): Promise<{ origin: string; close: () => void }> {
  const srv = http.createServer((req, res) => {
    const name = decodeURIComponent(new URL(req.url ?? '/', 'http://127.0.0.1').pathname);
    const mapped = extra?.(name);
    const file = mapped ?? path.join(root, name);
    if (!file.startsWith(root) && !mapped) return void res.writeHead(403).end();
    // As a stream server must, so that a page on another loopback origin (a
    // notebook) can import the WebAssembly views' loader.
    res.setHeader('access-control-allow-origin', '*');
    if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) return void res.writeHead(404).end();
    res.writeHead(200, { 'content-type': TYPES[path.extname(file)] ?? 'application/octet-stream', 'cache-control': 'no-store' });
    fs.createReadStream(file).pipe(res);
  });
  await new Promise<void>((ok) => srv.listen(0, '127.0.0.1', ok));
  const port = (srv.address() as { port: number }).port;
  return { origin: `http://127.0.0.1:${port}`, close: () => srv.close() };
}

/** The render path a project expects for heatmaps, images and points. */
export function gpuPath(info: TestInfo): 'webgpu' | 'canvas2d' {
  return info.project.name === 'canvas2d' ? 'canvas2d' : 'webgpu';
}

export const test = base.extend<{ harness: Page }, { server: string }>({
  server: [async ({}, use) => {
    const s = await serve(WEB, (name) => (name === '/caliper-views-loader.js' ? path.join(WEB, 'test', 'stub-loader.js') : null));
    await use(s.origin);
    s.close();
  }, { scope: 'worker' }],
  harness: async ({ page, server }, use) => {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('console', (m) => {
      // Elements pointed at a server that is not a stream log failed sockets.
      if (m.type() === 'error' && !/WebSocket connection|Failed to load resource/.test(m.text())) errors.push(m.text());
    });
    await page.goto(`${server}/test/harness.html`);
    await page.waitForFunction(() => 'harness' in window && customElements.get('caliper-view'));
    await use(page);
    expect(errors, 'page errors').toEqual([]);
  },
});
export { expect };

/**
 * expect with a deadline for state that comes back from the producer over the
 * socket: replies, broadcasts and the snapshots that follow a request. On a
 * three-core CI runner a page stalls for seconds while its GPU views start
 * (SwiftShader compiles their pipelines), so a round trip gets 30 s. It still
 * waits on the state and ends as soon as it holds; the suite's default stays.
 */
export const roundTrip = expect.configure({ timeout: 30_000 });

/** Saves a screenshot under SHOTS with the project's name. */
export function shotPath(info: TestInfo, name: string): string {
  fs.mkdirSync(SHOTS, { recursive: true });
  return path.join(SHOTS, `${name}-${info.project.name}.png`);
}

// --- golden frames -------------------------------------------------------------

/**
 * The folder of the wire v3 golden frames: the checkout's own copy when it has
 * one, otherwise a copy taken with git show from the branch that holds them
 * (CALIPER_WIRE_REF, sprint2 or sprint2-proto), never checked in here.
 */
export function goldenFrames(): string {
  const inTree = path.join(REPO, 'tests', 'fixtures', 'wire_v3');
  if (fs.existsSync(path.join(inTree, 'snapshot.bin'))) return inTree;
  return fromGit('tests/fixtures/wire_v3');
}

/** The Python codec _wire.py: the checkout's own, or a copy from git. */
export function wireModule(): string {
  const inTree = path.join(REPO, 'python', 'caliper', '_wire.py');
  if (fs.existsSync(inTree)) return inTree;
  return path.join(fromGit('python/caliper/_wire.py'), '_wire.py');
}

/**
 * Copies a file or folder of another branch into web/.cache/<object id>/
 * (ignored by git). Workers that race write their own copy and rename it into
 * place, so a reader never sees a half-written folder.
 */
function fromGit(what: string): string {
  const refs = [process.env.CALIPER_WIRE_REF, 'sprint2', 'sprint2-proto'].filter(Boolean) as string[];
  const git = (...args: string[]) => execFileSync('git', ['-C', REPO, ...args], { stdio: ['ignore', 'pipe', 'ignore'] });
  for (const ref of refs) {
    let id: string;
    try {
      id = git('rev-parse', `${ref}:${what}`).toString().trim();
    } catch {
      continue;
    }
    const dest = path.join(WEB, '.cache', id);
    if (fs.existsSync(dest)) return dest;
    const files = git('ls-tree', '-r', '--name-only', ref, what).toString().split('\n').filter(Boolean);
    const tmp = `${dest}.${process.pid}`;
    for (const f of files) {
      const rel = f === what ? path.basename(f) : path.relative(what, f);
      const out = path.join(tmp, rel);
      fs.mkdirSync(path.dirname(out), { recursive: true });
      fs.writeFileSync(out, git('show', `${ref}:${f}`));
    }
    try {
      fs.renameSync(tmp, dest);
    } catch {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
    return dest;
  }
  throw new Error(`${what} is neither in the checkout nor on ${refs.join(', ')}`);
}

// --- PNG decoding ---------------------------------------------------------------

export interface Pixels { width: number; height: number; data: Uint8Array; at(x: number, y: number): [number, number, number, number] }

/** Decodes an 8-bit RGB or RGBA PNG (what Chromium's screenshots are). */
export function decodePng(buf: Buffer): Pixels {
  let pos = 8, width = 0, height = 0, type = 0;
  const idat: Buffer[] = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos);
    const kind = buf.toString('ascii', pos + 4, pos + 8);
    const body = buf.subarray(pos + 8, pos + 8 + len);
    if (kind === 'IHDR') {
      width = body.readUInt32BE(0);
      height = body.readUInt32BE(4);
      if (body[8] !== 8) throw new Error('only 8-bit PNGs');
      type = body[9];
    } else if (kind === 'IDAT') idat.push(body);
    pos += 12 + len;
  }
  const ch = type === 6 ? 4 : type === 2 ? 3 : 0;
  if (!ch) throw new Error(`PNG colour type ${type}`);
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const stride = width * ch;
  const out = new Uint8Array(width * height * 4);
  const prev = new Uint8Array(stride), cur = new Uint8Array(stride);
  for (let y = 0; y < height; y++) {
    const f = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let i = 0; i < stride; i++) {
      const a = i >= ch ? cur[i - ch] : 0, b = prev[i], c = i >= ch ? prev[i - ch] : 0;
      let v = line[i];
      if (f === 1) v += a;
      else if (f === 2) v += b;
      else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) {
        const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      cur[i] = v & 255;
    }
    for (let x = 0; x < width; x++) {
      for (let k = 0; k < 3; k++) out[(y * width + x) * 4 + k] = cur[x * ch + k];
      out[(y * width + x) * 4 + 3] = ch === 4 ? cur[x * ch + 3] : 255;
    }
    prev.set(cur);
  }
  return {
    width, height, data: out,
    at: (x, y) => {
      const o = (Math.floor(y) * width + Math.floor(x)) * 4;
      return [out[o], out[o + 1], out[o + 2], out[o + 3]];
    },
  };
}

/** Screenshot of an element's drawing area, decoded. */
export async function bodyPixels(page: Page, selector: string): Promise<{ px: Pixels; box: { x: number; y: number } }> {
  const box = await page.evaluate((sel) => {
    const el = document.querySelector(sel) as HTMLElement;
    const r = el.shadowRoot!.querySelector('.body')!.getBoundingClientRect();
    return { x: r.left, y: r.top, width: r.width, height: r.height };
  }, selector);
  const png = await page.screenshot({ clip: box });
  return { px: decodePng(png), box };
}

// --- waits on what the page has drawn and received --------------------------------
// Tests wait on frames, events or published state, never on a fixed time: a
// software adapter on a CI runner draws a frame in a second or more.

type ViewEl = HTMLElement & { debugState(): Record<string, unknown>; settled(): Promise<void> };

/**
 * Waits until the views matching selector have drawn what they hold. A
 * TypeScript view draws on demand: settled() resolves once no frame is
 * pending and the GPU has finished, and while data keeps arriving faster than
 * a slow machine draws, a frame drawn after this call is enough (its frames
 * count grows). A WebAssembly view's module draws on every
 * animation frame and publishes its frame count at the end of a frame: the
 * count must first change, which a frame that began after this call does, and
 * pass the module's 30 start-up frames, over which the windows are laid out
 * (it publishes every 250 ms after them, every 15 frames before); then it must
 * grow by frames - 1 more.
 */
export async function drawn(page: Page, selector = 'caliper-view', frames = 2, timeout = 30_000): Promise<void> {
  const settled = page.evaluate(async (sel) => {
    const els = [...document.querySelectorAll(sel)] as ViewEl[];
    const wasm = (e: ViewEl) => e.debugState().drawer === 'wasm';
    const frames = (e: ViewEl) => Number(e.debugState().frames ?? NaN);
    const drew = async (e: ViewEl) => {
      const f0 = frames(e);
      let done = false;
      void e.settled().then(() => { done = true; });
      while (!done && !(frames(e) > f0)) await new Promise((ok) => requestAnimationFrame(() => ok(null)));
    };
    await Promise.all(els.filter((e) => !wasm(e)).map(drew));
    return els.map((e) => (wasm(e) ? Number((e.debugState().wasmStats as { frames?: number } | null)?.frames ?? 0) : null));
  }, selector);
  // A deadline, not a wait: a view that keeps drawing (turning, say) never settles.
  let deadline: ReturnType<typeof setTimeout> | undefined;
  const counts = await Promise.race([settled, new Promise<never>((_, no) => {
    deadline = setTimeout(() => no(new Error(`${selector}: still drawing after ${timeout} ms`)), timeout);
  })]).finally(() => clearTimeout(deadline));
  if (counts.every((c) => c === null)) return;
  const grown = (min: (number | null)[]) => page.waitForFunction(([sel, min]) => {
    const now = ([...document.querySelectorAll(sel as string)] as ViewEl[])
      .map((e) => (e.debugState().drawer === 'wasm' ? Number((e.debugState().wasmStats as { frames?: number } | null)?.frames ?? 0) : null));
    return now.every((n, i) => n === null || (min as (number | null)[])[i] === null || n >= (min as number[])[i]) && now;
  }, [selector, min] as const, { polling: 'raf', timeout });
  const first = await (await grown(counts.map((c) => (c === null ? null : Math.max(c + 1, 31))))).jsonValue() as (number | null)[];
  if (frames > 1) await grown(first.map((c) => (c === null ? null : c + frames - 1)));
}

/**
 * Waits until the view has received k more snapshots from its stream (the
 * TypeScript views' seq, the WebAssembly views' count of applied updates as
 * the module publishes it).
 */
export async function snapshotsPass(page: Page, selector: string, k: number, timeout = 30_000): Promise<void> {
  const count = (sel: string) => {
    const s = (document.querySelector(sel) as ViewEl).debugState();
    return Number(s.drawer === 'wasm' ? s.updates : s.seq);
  };
  const n0 = await page.evaluate(count, selector);
  await page.waitForFunction(([sel, n]) => {
    const s = (document.querySelector(sel as string) as ViewEl).debugState();
    return Number(s.drawer === 'wasm' ? s.updates : s.seq) >= (n as number);
  }, [selector, n0 + k] as const, { polling: 'raf', timeout });
}
