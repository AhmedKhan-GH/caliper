// Headless check of a scene in the browser (protocol 3.2 draft, web/wasm/README.md).
//
//   node scene.mjs --out <shot.png> [--adapter metal|swiftshader] [--python PYTHON]
//
// Starts scene_server.py, which logs demo 0's kind of scene through
// caliper.serve() (Caliper LIB on PYTHONPATH, with the WebAssembly views and
// the browser bundles in its _web folder), opens the page link it prints, the
// viewer page, in headless Chromium with WebGPU, and waits until the page's
// <caliper-view> of the scene has mounted the WebAssembly views and they have
// applied scene messages and drawn frames. It saves a screenshot to <shot.png>
// and checks it: the plate is not blank and shows the marker's red. It then
// asks the script to recolour the surface, saves <shot>-update.png and checks
// that the plate changed. It prints the element's state and the module's
// report and exits with 1 when a check failed. Headless only.
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const here = path.dirname(fileURLToPath(import.meta.url));
const opt = { adapter: 'metal', python: process.env.CALIPER_PYTHON || 'python3', viewport: '1280x900' };
for (let i = 2; i < process.argv.length; i += 2) opt[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
if (!opt.out) {
  console.error('usage: node scene.mjs --out <shot.png> [--adapter metal|swiftshader] [--python PYTHON]');
  process.exit(2);
}
const FLAGS = {
  swiftshader: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=swiftshader'],
  metal: ['--enable-unsafe-webgpu', '--use-angle=metal'],
};

const server = spawn(opt.python, [path.join(here, 'scene_server.py')], { stdio: ['pipe', 'pipe', 'inherit'] });
const ready = new Promise((resolve, reject) => {
  let buf = '';
  server.stdout.on('data', (d) => {
    buf += d;
    const m = /READY (\S+)/.exec(buf);
    if (m) resolve(m[1]);
  });
  server.on('exit', (code) => reject(new Error(`scene_server.py exited with ${code}`)));
  setTimeout(() => reject(new Error('scene_server.py did not start')), 30000);
});

// The plate's pixels from a PNG screenshot, decoded in the page (no image library here).
async function plate(page, png) {
  return page.evaluate(async (b64) => {
    const img = new Image();
    img.src = `data:image/png;base64,${b64}`;
    await img.decode();
    const el = document.querySelector('caliper-view');
    const r = el.getBoundingClientRect();
    const c = document.createElement('canvas');
    c.width = img.width;
    c.height = img.height;
    const g = c.getContext('2d');
    g.drawImage(img, 0, 0);
    const k = img.width / innerWidth;
    const d = g.getImageData(Math.round(r.x * k), Math.round(r.y * k), Math.round(r.width * k), Math.round(r.height * k));
    const colours = new Set();
    let red = 0;
    for (let i = 0; i < d.data.length; i += 4) {
      const [cr, cg, cb] = [d.data[i], d.data[i + 1], d.data[i + 2]];
      colours.add((cr << 16) | (cg << 8) | cb);
      if (Math.abs(cr - 255) <= 2 && Math.abs(cg - 80) <= 2 && Math.abs(cb - 80) <= 2) ++red;
    }
    let sum = 0;
    for (let i = 0; i < d.data.length; i += 4) sum = (sum * 31 + d.data[i] + d.data[i + 1] * 7 + d.data[i + 2] * 13) >>> 0;
    return { colours: colours.size, red, hash: sum, w: d.width, h: d.height };
  }, png.toString('base64'));
}

let browser = null;
let failed = true;
try {
  const url = await ready;
  const [vw, vh] = opt.viewport.split('x').map(Number);
  browser = await chromium.launch({ headless: true, args: FLAGS[opt.adapter] });
  const page = await browser.newPage({ viewport: { width: vw, height: vh } });
  const log = [];
  page.on('console', (m) => log.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => log.push(`[pageerror] ${e.message}`));
  await page.goto(url);
  const state = () => page.evaluate(() => document.querySelector('caliper-view')?.debugState?.() ?? null);
  let error = null;
  try {
    await page.waitForFunction(() => {
      const s = document.querySelector('caliper-view')?.debugState?.();
      const w = s?.wasmStats;
      return !!w && (w.error || (w.updates > 3 && w.frames > 60));
    }, null, { timeout: 120000, polling: 250 });
  } catch (e) {
    error = String(e);
  }
  await page.waitForTimeout(500);
  const shot = await page.screenshot({ path: opt.out });
  const before = await plate(page, shot);
  const st = await state();
  // The surface recoloured: the script shifts its values, the plate changes.
  server.stdin.write('update\n');
  await page.waitForTimeout(2000);
  const shot2 = await page.screenshot({ path: opt.out.replace(/\.png$/, '') + '-update.png' });
  const after = await plate(page, shot2);
  console.log('element:', JSON.stringify({ ...st, wasmStats: undefined }, null, 2));
  console.log('module:', JSON.stringify(st?.wasmStats ?? null, null, 2));
  console.log('plate:', JSON.stringify({ before, after }));
  console.log('console:\n' + [...new Set(log)].slice(0, 40).join('\n'));
  const checks = {
    drawnByWasm: st?.drawer === 'wasm',
    notBlank: before.colours > 50,
    marker: before.red > 0,
    recoloured: after.hash !== before.hash && after.red > 0,
    noModuleError: !st?.wasmStats?.error,
  };
  console.log('checks:', JSON.stringify(checks));
  if (error) console.log('error:', error);
  failed = !!error || Object.values(checks).some((v) => !v);
} catch (e) {
  console.log('error:', String(e));
} finally {
  if (browser) await browser.close();
  server.stdin.end();
  server.kill('SIGTERM');
}
process.exit(failed ? 1 : 0);
