// The loader from another origin, as a notebook page uses it: a page served by
// a second server on 127.0.0.1 (another port, so another origin) imports
// caliper-views-loader.js from the producer's server, which also serves the
// module and the stream (stream_server.py, with Access-Control-Allow-Origin on
// its static files), and mounts one view inside a shadow root.
//
//   node cross_origin.mjs --build <build dir> --out <shot.png> [--adapter swiftshader|metal]
//                         [--view loss] [--kind series] [--python PYTHON]
//
// Headless only. Every process it starts is stopped before it exits.
import { spawn } from 'node:child_process';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const here = path.dirname(fileURLToPath(import.meta.url));
const opt = { adapter: 'swiftshader', python: process.env.CALIPER_PYTHON || 'python3', view: 'loss',
              kind: 'series' };
for (let i = 2; i < process.argv.length; i += 2) opt[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
const FLAGS = {
  swiftshader: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=swiftshader'],
  metal: ['--enable-unsafe-webgpu', '--use-angle=metal'],
};

const producer = spawn(opt.python, [path.join(here, 'stream_server.py'), '--site', opt.build],
                       { stdio: ['ignore', 'pipe', 'inherit'] });
const ready = new Promise((resolve, reject) => {
  let buf = '';
  producer.stdout.on('data', (d) => {
    buf += d;
    const m = /READY (\d+) (\S+)/.exec(buf);
    if (m) resolve({ port: Number(m[1]), token: m[2] });
  });
  producer.on('exit', (c) => reject(new Error(`stream_server.py exited with ${c}`)));
});

// The "notebook" origin: one page and nothing else.
const page = (assetBase, src) => `<!doctype html><meta charset="utf-8"><body style="margin:0;background:#181715">
<div id="host" style="width:640px"></div>
<script type="module">
  const host = document.getElementById('host');
  const root = host.attachShadow({ mode: 'open' });
  const container = document.createElement('div');
  root.appendChild(container);
  try {
    const { mountCaliperViews } = await import(${JSON.stringify(assetBase + 'caliper-views-loader.js')});
    const h = await mountCaliperViews(container, { src: ${JSON.stringify(src)}, view: ${JSON.stringify(opt.view)},
      kind: ${JSON.stringify(opt.kind)}, height: 280, assetBase: ${JSON.stringify(assetBase)} });
    await h.ready;
    globalThis.mounted = h;
  } catch (e) {
    globalThis.mountError = String(e);
  }
</script>`;

let browser = null;
let notebook = null;
let failed = true;
try {
  const { port, token } = await ready;
  const assetBase = `http://127.0.0.1:${port}/`;
  const src = `ws://127.0.0.1:${port}/v3/stream?token=${token}`;
  notebook = http.createServer((q, r) => {
    r.writeHead(200, { 'Content-Type': 'text/html' });
    r.end(page(assetBase, src));
  });
  await new Promise((ok) => notebook.listen(0, '127.0.0.1', ok));
  browser = await chromium.launch({ headless: true, args: FLAGS[opt.adapter] });
  const p = await browser.newPage({ viewport: { width: 640, height: 300 } });
  const log = [];
  p.on('console', (m) => log.push(`[${m.type()}] ${m.text()}`));
  p.on('pageerror', (e) => log.push(`[pageerror] ${e.message}`));
  const origin = `http://127.0.0.1:${notebook.address().port}`;
  await p.goto(origin + '/');
  await p.waitForFunction(() => globalThis.mounted || globalThis.mountError, null, { timeout: 60000 });
  // Until the module has published that it applied a snapshot (it publishes
  // every 15 frames at first); past the deadline the check below fails.
  await p.waitForFunction(() => globalThis.mountError || globalThis.mounted.stats?.updates > 0, null,
                          { polling: 'raf', timeout: 60000 }).catch(() => undefined);
  const res = await p.evaluate(() => globalThis.mountError ? { error: globalThis.mountError }
    : { stats: globalThis.mounted.stats });
  await p.screenshot({ path: opt.out });
  console.log(JSON.stringify({ pageOrigin: origin, assetBase, error: res.error,
                               updates: res.stats?.updates, state: res.stats?.state }, null, 2));
  console.log('console:\n' + [...new Set(log)].slice(0, 20).join('\n'));
  failed = !!res.error || !(res.stats?.updates > 0);
} catch (e) {
  console.log('error:', String(e));
} finally {
  if (browser) await browser.close();
  if (notebook) notebook.close();
  producer.kill('SIGTERM');
}
process.exit(failed ? 1 : 0);
