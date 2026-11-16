// The viewer page of caliper.serve() in headless Chromium, for the browser row of
// demo 0 (test_loss_landscape.py, test_capture_browser). Never a visible window,
// and nothing is downloaded: Playwright comes from web/wasm/test/node_modules
// (npm ci there) and its Chromium headless shell from Playwright's cache.
//
//   node viewer_page.mjs <page link> <adapter: swiftshader | metal>
//
// It opens the page at 1280 x 900 CSS pixels and prints "READY" once the page's
// <caliper-view> of the scene "landscape" draws with Caliper's WebAssembly views
// and has applied a scene message, or "SKIP <why>" when Playwright, its Chromium
// or WebGPU is missing. Then, for each line "shot <messages> <png>" on its
// standard input, it waits until the scene's module has received at least
// <messages> messages (what the server sent on its connection), then 30 more of
// its frames, saves the scene's drawing area to <png> and prints "SHOT <json>"
// with the element's state. Every wait is on the page's published state or its
// frames. It ends when its standard input closes.
import path from 'node:path';
import readline from 'node:readline';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const [url, adapter] = process.argv.slice(2);
const FLAGS = {
  swiftshader: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=swiftshader'],
  metal: ['--enable-unsafe-webgpu', '--use-angle=metal'],
};
const SCENE = 'caliper-view[view="landscape"]';

let chromium;
try {
  ({ chromium } = createRequire(path.join(here, '..', '..', 'web', 'wasm', 'test', 'package.json'))('playwright-core'));
} catch (e) {
  console.log(`SKIP playwright-core is not installed in web/wasm/test (npm ci there): ${e.message.split('\n')[0]}`);
  process.exit(0);
}
let browser;
try {
  browser = await chromium.launch({ headless: true, args: FLAGS[adapter] });
} catch (e) {
  console.log(`SKIP Playwright's Chromium headless shell is not in its cache: ${e.message.split('\n')[0]}`);
  process.exit(0);
}
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const log = [];
page.on('console', (m) => log.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => log.push(`[pageerror] ${e.message}`));
const state = () => page.evaluate((sel) => document.querySelector(sel)?.debugState?.() ?? null, SCENE);

try {
  await page.goto(url);
  await page.waitForFunction((sel) => {
    const s = document.querySelector(sel)?.debugState?.();
    const w = s?.wasmStats;
    return s?.drawer === 'canvas2d' || (!!w && (w.error || (s.drawer === 'wasm' && w.updates > 0 && w.frames > 10)));
  }, SCENE, { timeout: 90000, polling: 100 });
  const s = await state();
  if (s.drawer !== 'wasm') {
    console.log(`SKIP the page draws the scene without the WebAssembly views (drawer ${s.drawer})`);
    await browser.close();
    process.exit(0);
  }
  console.log('READY');
} catch (e) {
  console.log(`FAIL ${JSON.stringify({ error: String(e), state: await state(), console: log.slice(0, 20) })}`);
  await browser.close();
  process.exit(1);
}

for await (const line of readline.createInterface({ input: process.stdin })) {
  const [cmd, messages, png] = line.trim().split(' ');
  if (cmd !== 'shot') continue;
  try {
    await page.waitForFunction(([sel, want]) => {
      const w = document.querySelector(sel)?.debugState?.()?.wasmStats;
      return !!w && (w.error || w.messages >= want);
    }, [SCENE, Number(messages)], { timeout: 60000, polling: 50 });
    const f0 = (await state())?.wasmStats?.frames ?? 0;
    await page.waitForFunction(([sel, want]) => {
      const w = document.querySelector(sel)?.debugState?.()?.wasmStats;
      return !!w && w.frames >= want;
    }, [SCENE, f0 + 30], { timeout: 60000, polling: 'raf' });
    const body = await page.evaluate((sel) => {
      const r = document.querySelector(sel).shadowRoot.querySelector('.body').getBoundingClientRect();
      return { x: r.x, y: r.y, width: r.width, height: r.height };
    }, SCENE);
    await page.screenshot({ path: png, clip: body });
    const st = await state();
    const w = st?.wasmStats ?? {};
    console.log(`SHOT ${JSON.stringify({
      drawer: st?.drawer, info: st?.info, messages: w.messages, updates: w.updates, frames: w.frames,
      error: w.error ?? '', gfxErrors: w.gfxErrors ?? '', console: log.slice(0, 20),
    })}`);
  } catch (e) {
    console.log(`SHOT ${JSON.stringify({ error: String(e), console: log.slice(0, 20) })}`);
  }
}
await browser.close();
