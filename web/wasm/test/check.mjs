// Headless check of Caliper's views in the browser (web/wasm/README.md).
//
//   node check.mjs --build <build dir> --out <shot.png> [--adapter swiftshader|metal]
//                  [--view NAME] [--kind KIND] [--frames N] [--heat WxH]
//                  [--viewport WxH] [--scale S] [--hover X,Y] [--page demo.html]
//                  [--python PYTHON] [--seconds S] [--dataview JSON]
//                  [--resize WxH] [--input X,Y] [--sample S]
//
// Starts the synthetic producer (stream_server.py) on 127.0.0.1, which serves
// the build directory over HTTP and the stream on /v3/stream on one port, as
// caliper.serve() does. Opens the page in headless Chromium with WebGPU, waits
// until the module has drawn N frames, applied snapshots and (when a heatmap
// is shown) finished its colormap check, prints the module's report
// (module.caliperStats) as JSON and saves a screenshot. With --hover it then
// moves the mouse there and saves <shot>-hover.png. With --dataview it asks
// for that data view (VIEW_REQUEST) before the screenshot and reports the
// data views the producer resolved. With --resize it then changes the page's
// viewport and saves <shot>-resized.png. With --input it clicks at X,Y, turns
// the wheel and presses keys there, and reports what reached ImGui. With
// --seconds S it keeps streaming S seconds before the report, and with
// --sample T it records the module's heap and counters every T seconds of
// that time (report.samples). Every
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
    }, wantFrames, { timeout: 120000, polling: 250 });
  } catch (e) {
    error = String(e);
  }
  if (opt.dataview) {
    const sent = await page.evaluate((json) => {
      const m = globalThis.caliperViews;
      const p = m.stringToNewUTF8(json);
      const ok = m._caliper_views_set_data_view(p);
      m._free(p);
      return ok;
    }, opt.dataview);
    if (!sent) error = 'set_data_view refused';
    await page.waitForTimeout(1000);
  }
  if (opt.input) {
    const [ix, iy] = opt.input.split(',').map(Number);
    await page.mouse.click(ix, iy);
    await page.mouse.wheel(0, 240);
    await page.keyboard.press('KeyF');
    await page.keyboard.type('ab');
    await page.waitForTimeout(300);
  }
  const samples = [];
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
  await page.screenshot({ path: opt.out });
  if (opt.resize) {
    const [rw, rh] = opt.resize.split('x').map(Number);
    await page.setViewportSize({ width: rw, height: rh });
    await page.waitForTimeout(600);
    await page.screenshot({ path: opt.out.replace(/\.png$/, '') + '-resized.png' });
    report.resized = await page.evaluate(() => {
      const s = globalThis.caliperViews.caliperStats;
      return { cssW: s.cssW, cssH: s.cssH, fbW: s.fbW, fbH: s.fbH };
    });
  }
  if (opt.hover) {
    const [hx, hy] = opt.hover.split(',').map(Number);
    await page.mouse.move(hx, hy);
    await page.waitForTimeout(500);
    await page.screenshot({ path: opt.out.replace(/\.png$/, '') + '-hover.png' });
  }
  console.log('report:', JSON.stringify(report, null, 2));
  if (error) console.log('error:', error);
  console.log('console:\n' + [...new Set(log)].slice(0, 40).join('\n'));
  failed = !!error || !report || !!report.error;
} catch (e) {
  console.log('error:', String(e));
} finally {
  if (browser) await browser.close();
  server.kill('SIGTERM');
}
process.exit(failed ? 1 : 0);
