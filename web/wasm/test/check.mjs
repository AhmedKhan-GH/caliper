// Headless check of Caliper's views in the browser (web/wasm/README.md).
//
//   node check.mjs --build <build dir> --out <shot.png> [--adapter swiftshader|metal]
//                  [--view NAME] [--kind KIND] [--frames N] [--heat WxH]
//                  [--viewport WxH] [--scale S] [--hover X,Y] [--page demo.html]
//                  [--python PYTHON] [--seconds S] [--dataview JSON]
//                  [--resize WxH] [--input X,Y] [--sample S] [--points N] [--dims 2|3]
//                  [--orbit S] [--rate HZ] [--compat 1]
//
// Starts the synthetic producer (stream_server.py) on 127.0.0.1, which serves
// the build directory over HTTP and the stream on /v3/stream on one port, as
// caliper.serve() does. Opens the page in headless Chromium with WebGPU, waits
// until the module has drawn N frames, applied snapshots and (when a heatmap
// is shown) finished its colormap check, prints the module's report
// (module.caliperStats) as JSON and saves a screenshot. With --hover it then
// moves the mouse there and saves <shot>-hover.png. It exits with 1 when the
// module reports an error or any of its texture or points checks found a
// difference. With --dataview it asks
// for that data view (VIEW_REQUEST) before the screenshot and reports the
// data views the producer resolved. With --resize it then changes the page's
// viewport and saves <shot>-resized.png. With --input it clicks at X,Y, turns
// the wheel and presses keys there, and reports what reached ImGui. With
// --seconds S it keeps streaming S seconds before the report, and with
// --sample T it records the module's heap and counters every T seconds of
// that time (report.samples). With --orbit S it drags across the middle of
// the canvas for S seconds (a points view orbits and redraws every frame) and
// reports the frames the module drew per second meanwhile (report.orbit).
// --compat 1 asks for a compatibility-mode WebGPU device.
// Every
// process it starts
// is stopped before it exits. Headless only: it never opens a visible browser.
//
// Adapters. Playwright's headless Chromium (chromium-headless-shell) gives
// WebGPU a SwiftShader adapter by default but cannot present a WebGPU canvas
// with its default flags: getCurrentTexture() returns an invalid texture and
// the device is lost. Two flag sets work:
//   swiftshader  --use-angle=swiftshader (+ Vulkan,WebGPU features): software
//                rasteriser, the CI-like case.
//   metal        --use-angle=metal: the machine's GPU through Metal.
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const here = path.dirname(fileURLToPath(import.meta.url));
const opt = { adapter: 'swiftshader', frames: '300', viewport: '1280x800', scale: '1', page: 'demo.html',
              python: process.env.CALIPER_PYTHON || 'python3', seconds: '0' };
for (let i = 2; i < process.argv.length; i += 2) opt[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
if (!opt.build || !opt.out) {
  console.error('usage: node check.mjs --build <build dir> --out <shot.png> [options]');
  process.exit(2);
}

export const ADAPTER_FLAGS = {
  swiftshader: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=swiftshader'],
  metal: ['--enable-unsafe-webgpu', '--use-angle=metal'],
};

// The producer: wait for its READY line.
const serverArgs = [path.join(here, 'stream_server.py'), '--site', opt.build];
if (opt.heat) serverArgs.push('--heat', opt.heat);
if (opt.points) serverArgs.push('--points', opt.points);
if (opt.dims) serverArgs.push('--dims', opt.dims);
if (opt.rate) serverArgs.push('--rate', opt.rate);
const server = spawn(opt.python, serverArgs, { stdio: ['ignore', 'pipe', 'inherit'] });
const serverLines = [];
const ready = new Promise((resolve, reject) => {
  let buf = '';
  server.stdout.on('data', (d) => {
    buf += d;
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      serverLines.push(line);
      const m = /^READY (\d+) (\S+)/.exec(line);
      if (m) resolve({ port: Number(m[1]), token: m[2] });
    }
  });
  server.on('exit', (code) => reject(new Error(`stream_server.py exited with ${code}`)));
  setTimeout(() => reject(new Error('stream_server.py did not start')), 20000);
});

// Waits until the module has published a frame count from a frame that began
// after this call (it publishes at the end of a frame), then k - 1 frames more.
// Checks wait on frames drawn, never on a fixed time: a software adapter on a
// CI runner draws a frame in a second or more.
async function framesPass(page, k = 2, timeout = 60000) {
  const f0 = await page.evaluate(() => globalThis.caliperViews.caliperStats.frames);
  const f1 = await (await page.waitForFunction((f) => {
    const n = globalThis.caliperViews.caliperStats.frames;
    return n > f && n;
  }, f0, { polling: 'raf', timeout })).jsonValue();
  if (k > 1) {
    await page.waitForFunction((n) => globalThis.caliperViews.caliperStats.frames >= n, f1 + k - 1,
                               { polling: 'raf', timeout });
  }
}

let browser = null;
let failed = true;
try {
  const { port, token } = await ready;
  const [vw, vh] = opt.viewport.split('x').map(Number);
  browser = await chromium.launch({ headless: true, args: ADAPTER_FLAGS[opt.adapter] });
  const page = await browser.newPage({ viewport: { width: vw, height: vh }, deviceScaleFactor: Number(opt.scale) });
  const log = [];
  page.on('console', (m) => log.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => log.push(`[pageerror] ${e.message}`));
  const q = new URLSearchParams({ token });
  if (opt.view) q.set('view', opt.view);
  if (opt.kind) q.set('kind', opt.kind);
  if (opt.compat) q.set('compat', '1');
  await page.goto(`http://127.0.0.1:${port}/${opt.page}?${q}`);
  const wantFrames = Number(opt.frames);
  let error = null;
  try {
    await page.waitForFunction((n) => {
      const s = globalThis.caliperViews?.caliperStats;
      if (!s) return false;
      if (s.error) return true;
      const heat = /kind=1,/.test(s.viewInfo || '');
      return s.frames >= n && s.updates > 0 && (!heat || s.checkState === 2);
    }, wantFrames, { timeout: 120000, polling: 'raf' });
  } catch (e) {
    error = String(e);
  }
  if (opt.dataview) {
    const replies = await page.evaluate(() => globalThis.caliperDataViews.length);
    const sent = await page.evaluate((json) => {
      const m = globalThis.caliperViews;
      const p = m.stringToNewUTF8(json);
      const ok = m._caliper_views_set_data_view(p);
      m._free(p);
      return ok;
    }, opt.dataview);
    if (!sent) error = 'set_data_view refused';
    // The stream's reply, then two frames that draw what it sends.
    else {
      await page.waitForFunction((n) => globalThis.caliperDataViews.length > n, replies, { polling: 'raf', timeout: 60000 });
      await framesPass(page);
    }
  }
  if (opt.input) {
    const [ix, iy] = opt.input.split(',').map(Number);
    await page.mouse.click(ix, iy);
    await page.mouse.wheel(0, 240);
    await page.keyboard.press('KeyF');
    await page.keyboard.type('ab');
    await framesPass(page);
  }
  if (opt.orbit) {
    const frames = () => page.evaluate(() => globalThis.caliperViews.caliperStats.frames);
    const [vw2, vh2] = [vw / 2, vh / 2];
    await page.mouse.move(vw2, vh2);
    await page.mouse.down();
    const f0 = await frames();
    const t0 = Date.now();
    const end = t0 + Number(opt.orbit) * 1000;
    // A measurement over the given time: each move is sent once the page has handled the last.
    for (let k = 0; Date.now() < end; ++k) {
      await page.mouse.move(vw2 + 120 * Math.sin(k / 20), vh2 + 60 * Math.cos(k / 27));
    }
    const f1 = await frames();
    const secs = (Date.now() - t0) / 1000;
    await page.mouse.up();
    globalThis.orbitReport = { seconds: +secs.toFixed(2), frames: f1 - f0, fps: +((f1 - f0) / secs).toFixed(1) };
  }
  const samples = [];
  // A soak: the counters sampled over the given time, which is what it measures.
  if (Number(opt.seconds) > 0) {
    const end = Date.now() + Number(opt.seconds) * 1000;
    const every = Number(opt.sample || 0) * 1000;
    while (every > 0 && Date.now() + every <= end) {
      await page.waitForTimeout(every);
      samples.push(await page.evaluate(() => {
        const s = globalThis.caliperViews.caliperStats;
        return { t: Math.round(performance.now() / 1000), frames: s.frames, updates: s.updates,
                 messages: s.messages, mbytes: Math.round(s.bytes / 1e6), heapSize: s.heapSize,
                 heapUsed: s.heapUsed, fps: Math.round(s.fps * 10) / 10 };
      }));
    }
    await page.waitForTimeout(Math.max(0, end - Date.now()));
  }
  const report = await page.evaluate(() => ({ ...globalThis.caliperViews?.caliperStats,
                                              dataViews: globalThis.caliperDataViews }));
  if (samples.length) report.samples = samples;
  if (globalThis.orbitReport) report.orbit = globalThis.orbitReport;
  await page.screenshot({ path: opt.out });
  if (opt.resize) {
    const [rw, rh] = opt.resize.split('x').map(Number);
    const was = await page.evaluate(() => [globalThis.caliperViews.caliperStats.cssW, globalThis.caliperViews.caliperStats.cssH]);
    await page.setViewportSize({ width: rw, height: rh });
    // The module has drawn at the new size, then two frames more.
    await page.waitForFunction(([w, h]) => {
      const s = globalThis.caliperViews.caliperStats;
      return s.cssW !== w || s.cssH !== h;
    }, was, { polling: 'raf', timeout: 60000 });
    await framesPass(page);
    await page.screenshot({ path: opt.out.replace(/\.png$/, '') + '-resized.png' });
    report.resized = await page.evaluate(() => {
      const s = globalThis.caliperViews.caliperStats;
      return { cssW: s.cssW, cssH: s.cssH, fbW: s.fbW, fbH: s.fbH };
    });
  }
  if (opt.hover) {
    const [hx, hy] = opt.hover.split(',').map(Number);
    await page.mouse.move(hx, hy);
    await framesPass(page);
    await page.screenshot({ path: opt.out.replace(/\.png$/, '') + '-hover.png' });
  }
  console.log('report:', JSON.stringify(report, null, 2));
  if (error) console.log('error:', error);
  console.log('console:\n' + [...new Set(log)].slice(0, 40).join('\n'));
  // Every texture and points check must have found nothing different.
  const bad = String(report?.checks ?? '').split(';').filter(Boolean)
    .filter((c) => Number(c.split(':')[2]) !== 0);
  if (bad.length) console.log('checks that found differences:', bad.join(' '));
  failed = !!error || !report || !!report.error || bad.length > 0;
} catch (e) {
  console.log('error:', String(e));
} finally {
  if (browser) await browser.close();
  server.kill('SIGTERM');
}
process.exit(failed ? 1 : 0);
