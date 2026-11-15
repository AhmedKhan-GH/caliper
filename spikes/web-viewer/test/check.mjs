// Headless check of the web viewer spike (docs/plans/web-viewer-spike.md).
//
//   node check.mjs <site dir> <screenshot.png> [swiftshader|metal] [frames] [WxH]
//
// Serves <site dir> (index.html, caliper_web_spike.js and .wasm) from a static
// server bound to 127.0.0.1, loads it in headless Chromium with WebGPU enabled,
// waits until the module has drawn `frames` frames and finished its colormap
// check, prints the module's own report (globalThis.caliperSpike) as JSON and
// saves a screenshot. It then moves the mouse over the series plot and saves a
// second screenshot (<screenshot>-hover.png), which shows the plot's hover
// tooltip if input reaches ImGui. WxH sets the heatmap size (default 256x192).
// Headless only: this script never opens a visible browser.
//
// Adapters. Playwright's headless Chromium (chromium-headless-shell) gives
// WebGPU a SwiftShader adapter by default but cannot present a WebGPU canvas
// with its default flags: getCurrentTexture() returns an invalid texture and
// the device is lost. Two flag sets work:
//   swiftshader  --use-angle=swiftshader (+ Vulkan,WebGPU features): software
//                rasteriser, the CI-like case.
//   metal        --use-angle=metal: the machine's GPU through Metal.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright-core';

const [siteDir, shotPath, adapter = 'swiftshader', framesArg = '300', size = ''] = process.argv.slice(2);
if (!siteDir || !shotPath) {
  console.error('usage: node check.mjs <site dir> <screenshot.png> [swiftshader|metal] [frames]');
  process.exit(2);
}
const wantFrames = Number(framesArg);

const types = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm' };
const server = http.createServer((req, res) => {
  const name = decodeURIComponent(new URL(req.url, 'http://127.0.0.1').pathname);
  const file = path.join(siteDir, name === '/' ? 'index.html' : name);
  if (!file.startsWith(path.resolve(siteDir)) || !fs.existsSync(file)) {
    res.writeHead(404).end();
    return;
  }
  res.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});
await new Promise((ok) => server.listen(0, '127.0.0.1', ok));
const [qw, qh] = size ? size.split('x') : [];
const url = `http://127.0.0.1:${server.address().port}/` + (size ? `?w=${qw}&h=${qh}` : '');

const flags = {
  swiftshader: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=swiftshader'],
  metal: ['--enable-unsafe-webgpu', '--use-angle=metal'],
};
const args = flags[adapter];
if (!args) {
  console.error(`unknown adapter '${adapter}' (swiftshader or metal)`);
  process.exit(2);
}

const browser = await chromium.launch({ headless: true, args });
const page = await browser.newPage({ viewport: { width: 1280, height: 830 }, deviceScaleFactor: 1 });
const log = [];
page.on('console', (m) => log.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => log.push(`[pageerror] ${e.message}`));

let report = null;
let error = null;
try {
  await page.goto(url);
  const gpu = await page.evaluate(async () => {
    if (!navigator.gpu) return { present: false };
    const a = await navigator.gpu.requestAdapter();
    if (!a) return { present: true, adapter: null };
    return { present: true, adapter: { ...a.info && { vendor: a.info.vendor, architecture: a.info.architecture,
      device: a.info.device, description: a.info.description } }, fallback: a.isFallbackAdapter ?? null };
  });
  console.log('navigator.gpu:', JSON.stringify(gpu));
  await page.waitForFunction(
    (n) => globalThis.caliperSpike && (globalThis.caliperSpike.error ||
      (globalThis.caliperSpike.frames >= n && globalThis.caliperSpike.checkState === 2)),
    wantFrames, { timeout: 120000, polling: 250 });
  report = await page.evaluate(() => globalThis.caliperSpike);
  await page.screenshot({ path: shotPath });
  // Hover the series plot (the second window, tiled to the right of the heatmap).
  await page.mouse.move(470, 140);
  await page.waitForTimeout(400);
  await page.screenshot({ path: shotPath.replace(/\.png$/, '') + '-hover.png' });
} catch (e) {
  error = String(e);
  report = await page.evaluate(() => globalThis.caliperSpike ?? null).catch(() => null);
  await page.screenshot({ path: shotPath }).catch(() => {});
}
console.log('report:', JSON.stringify(report, null, 2));
if (error) console.log('error:', error);
console.log('console:\n' + [...new Set(log)].slice(0, 60).join('\n'));
await browser.close();
server.close();
process.exit(error || !report || report.error ? 1 : 0);
