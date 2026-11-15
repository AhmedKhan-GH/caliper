// Several Caliper view mounts on one page (test/pages/multi.html), through the
// loader, each inside a shadow root as the <caliper-view> element mounts it.
//
//   node multi.mjs --build <build dir> --out <shot.png> [--adapter swiftshader|metal]
//                  [--views a:kind,b:kind] [--destroy I] [--python PYTHON]
//
// Starts the synthetic producer (stream_server.py), opens the page in headless
// Chromium, waits until every mount drew its first snapshot, and reports per
// mount the time to mount and to the first frame, each instance's heap, and
// the page's JavaScript and ArrayBuffer memory before and after (Chromium's
// Runtime.getHeapUsage). With --destroy it destroys mount I and checks that
// the others keep drawing. Saves a screenshot. Headless only.
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const here = path.dirname(fileURLToPath(import.meta.url));
const opt = { adapter: 'swiftshader', python: process.env.CALIPER_PYTHON || 'python3',
              views: 'activations:heatmap,loss:series,histogram:histogram' };
for (let i = 2; i < process.argv.length; i += 2) opt[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
const FLAGS = {
  swiftshader: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=swiftshader'],
  metal: ['--enable-unsafe-webgpu', '--use-angle=metal'],
};

const server = spawn(opt.python, [path.join(here, 'stream_server.py'), '--site', opt.build],
                     { stdio: ['ignore', 'pipe', 'inherit'] });
const ready = new Promise((resolve, reject) => {
  let buf = '';
  server.stdout.on('data', (d) => {
    buf += d;
    const m = /READY (\d+) (\S+)/.exec(buf);
    if (m) resolve({ port: Number(m[1]), token: m[2] });
  });
  server.on('exit', (c) => reject(new Error(`stream_server.py exited with ${c}`)));
});

let browser = null;
let failed = true;
try {
  const { port, token } = await ready;
  browser = await chromium.launch({ headless: true, args: FLAGS[opt.adapter] });
  const page = await browser.newPage({ viewport: { width: 640, height: 820 } });
  const log = [];
  page.on('console', (m) => log.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => log.push(`[pageerror] ${e.message}`));
  const cdp = await page.context().newCDPSession(page);
  const heap = async () => {
    const u = await cdp.send('Runtime.getHeapUsage');
    return { jsMB: +(u.usedSize / 1e6).toFixed(1), backingMB: +((u.backingStorageSize || 0) / 1e6).toFixed(1) };
  };
  await page.goto(`http://127.0.0.1:${port}/multi.html?${new URLSearchParams({ token, views: opt.views })}`);
  await page.waitForFunction(() => document.readyState === 'complete');
  const before = await heap();
  await page.waitForFunction(() => globalThis.caliperMulti?.done, null, { timeout: 120000, polling: 200 });
  await page.waitForTimeout(1500);
  const after = await heap();
  const report = await page.evaluate(() => ({
    preloadMs: Math.round(globalThis.caliperMulti.preloadMs),
    mounts: globalThis.caliperMulti.mounts.map((m, i) => {
      const s = globalThis.caliperMulti.handles[i].stats;
      return { view: m.view, mountMs: Math.round(m.mountMs), firstFrameMs: Math.round(m.firstFrameMs),
               frames: s.frames, updates: s.updates, heapMB: +(s.heapSize / 1e6).toFixed(1),
               heapUsedMB: +(s.heapUsed / 1e6).toFixed(1), dataViews: m.dataViews.length };
    }),
  }));
  report.before = before;
  report.after = after;
  if (opt.destroy !== undefined) {
    const d = Number(opt.destroy);
    const counts = async () => page.evaluate(() => globalThis.caliperMulti.handles.map((h) => h.stats.frames));
    const c0 = await counts();
    await page.evaluate((i) => globalThis.caliperMulti.handles[i].destroy(), d);
    await page.waitForTimeout(1000);
    const c1 = await counts();
    report.afterDestroy = {
      canvases: await page.evaluate(() => globalThis.caliperMulti.handles.filter((h) => h.canvas.isConnected).length),
      framesBefore: c0, framesAfter: c1,
      othersDrawing: c1.every((n, i) => i === d || n > c0[i]),
      destroyedStopped: c1[d] === c0[d] || c1[d] - c0[d] <= 1,
    };
  }
  await page.screenshot({ path: opt.out, fullPage: true });
  console.log('report:', JSON.stringify(report, null, 2));
  console.log('console:\n' + [...new Set(log)].slice(0, 30).join('\n'));
  failed = report.mounts.length !== opt.views.split(',').length ||
           report.mounts.some((m) => !m.updates) ||
           (report.afterDestroy && !(report.afterDestroy.othersDrawing && report.afterDestroy.destroyedStopped));
} catch (e) {
  console.log('error:', String(e));
} finally {
  if (browser) await browser.close();
  server.kill('SIGTERM');
}
process.exit(failed ? 1 : 0);
