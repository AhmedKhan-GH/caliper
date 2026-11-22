// The browser side of scene_check.py: the viewer page of caliper.serve() in headless Chromium,
// never a visible window. Playwright comes from web/wasm/test or web/ (npm ci there) and its
// Chromium headless shell from Playwright's cache; nothing is downloaded.
//
//   node scene_check_browser.mjs <page link> <view> <png> <adapter: metal | swiftshader>
//
// It waits until the page's <caliper-view> of <view> draws with Caliper's WebAssembly views and
// its updates have stopped for 30 frames, double-clicks the centre of the scene's picture so
// that the view fits its camera to the whole scene, saves the view's drawing area to <png>,
// prints "REFIT", clicks the centre once when a line "go" arrives on its standard input, waits
// 30 more frames, and prints one line "RESULT <json>" with the
// module's geometry of the view (its area and its fitted orbit: target x, y, z and distance).
// Every wait is on the page's published state or its frames.
import path from 'node:path';
import readline from 'node:readline';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const [url, view, png, adapter = 'metal'] = process.argv.slice(2);
const FLAGS = {
  swiftshader: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=swiftshader'],
  metal: ['--enable-unsafe-webgpu', '--use-angle=metal'],
};
const SEL = `caliper-view[view="${view}"]`;

let chromium = null;
for (const dir of [['web', 'wasm', 'test'], ['web']]) {
  try {
    ({ chromium } = createRequire(path.join(here, '..', '..', ...dir, 'package.json'))('playwright-core'));
    break;
  } catch { /* the next place */ }
}
if (!chromium) {
  console.log('FAIL playwright-core is installed in neither web/wasm/test nor web (npm ci there)');
  process.exit(1);
}

const browser = await chromium.launch({ headless: true, args: FLAGS[adapter] });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const log = [];
page.on('console', (m) => log.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => log.push(`[pageerror] ${e.message}`));
const state = () => page.evaluate((sel) => document.querySelector(sel)?.debugState?.() ?? null, SEL);
const geometry = (st) => {
  try { return JSON.parse(String(st?.wasmStats?.geometry ?? '{}'))[view] ?? null; } catch { return null; }
};
const frames = async (n) => {
  const f0 = (await state())?.wasmStats?.frames ?? 0;
  await page.waitForFunction(([sel, want]) => (document.querySelector(sel)?.debugState?.()?.wasmStats?.frames ?? 0) >= want,
    [SEL, f0 + n], { timeout: 60000, polling: 'raf' });
};

let code = 1;
try {
  await page.goto(url);
  await page.waitForFunction((sel) => {
    const s = document.querySelector(sel)?.debugState?.();
    const w = s?.wasmStats;
    return s?.drawer === 'canvas2d' || (!!w && (w.error || (s.drawer === 'wasm' && w.updates > 0 && w.frames > 10)));
  }, SEL, { timeout: 90000, polling: 100 });
  let st = await state();
  if (st?.drawer !== 'wasm' || st?.wasmStats?.error) throw new Error(`the scene is not drawn by the WebAssembly views: ${JSON.stringify(st)}`);
  // Every static batch arrives at the subscription; wait until 30 frames pass with no new update.
  for (let quiet = 0, last = -1, rounds = 0; quiet < 3 && rounds < 200; rounds++) {
    const u = (await state())?.wasmStats?.updates ?? 0;
    quiet = u === last ? quiet + 1 : 0;
    last = u;
    await frames(10);
  }
  const body = await page.evaluate((sel) => {
    const r = document.querySelector(sel).shadowRoot.querySelector('.body').getBoundingClientRect();
    return { x: r.x, y: r.y, width: r.width, height: r.height };
  }, SEL);
  let geo = geometry(await state());
  if (!geo?.area || !geo?.orbit) throw new Error(`the module published no geometry for ${view}: ${(await state())?.wasmStats?.geometry}`);
  // The area is in the canvas's CSS pixels, and the canvas fills the body.
  const [ax, ay, aw, ah] = geo.area;
  const cx = body.x + ax + aw / 2, cy = body.y + ay + ah / 2;
  await page.mouse.move(cx, cy);
  await frames(10);
  await page.mouse.dblclick(cx, cy);
  await frames(30);
  st = await state();
  geo = geometry(st);
  await page.screenshot({ path: png, clip: body });
  // The caller collects the double click's picks, then says when to click once more.
  console.log('REFIT');
  const lines = readline.createInterface({ input: process.stdin });
  for await (const line of lines) if (line.trim() === 'go') break;
  lines.close();
  await page.mouse.click(cx, cy);
  await frames(30);
  console.log(`RESULT ${JSON.stringify({ geometry: geo, adapter: st?.wasmStats?.adapter ?? '', updates: st?.wasmStats?.updates,
    messages: st?.wasmStats?.messages, console: [...new Set(log)].slice(0, 20) })}`);
  code = 0;
} catch (e) {
  console.log(`FAIL ${JSON.stringify({ error: String(e), state: await state().catch(() => null), console: log.slice(0, 20) })}`);
} finally {
  await browser.close();
}
process.exit(code);
