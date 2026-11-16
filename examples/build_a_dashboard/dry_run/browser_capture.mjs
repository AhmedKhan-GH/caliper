// browser_capture.mjs: opens a saved spec applet in headless Chromium from the
// running app's link and captures the page; then lays the app's capture and the
// browser's side by side, whole and panel by panel.
//
//   node browser_capture.mjs capture <url> <WxH> <browser.png> <state.json>
//   node browser_capture.mjs compare <app.png> <browser.png> <state.json> <side.png> <panels.png>
//
// <url> is the link the app prints with --api ("Spec applets in a browser at
// ..."), with &applet=<id> added. The capture is the window's size in points
// times 2, as the app's captures are on a Retina display, and the state is the
// page's own account of its panels: title, info line, rectangle and state. The
// comparison puts the app left and the browser right, at half size, and cuts
// each panel's rectangle, as the page lays it out, from both at full size.
//
// Playwright comes from the web package (cd web && npm ci) or from NODE_PATH;
// its Chromium runs headless, and nothing is shown on screen.

import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(here, '..', '..', '..', 'web', 'package.json'));
const { chromium } = require('@playwright/test');

const SCALE = 2;
const [mode, ...args] = process.argv.slice(2);

async function capture(url, size, out, stateFile) {
  const [width, height] = size.split('x').map(Number);
  const browser = await chromium.launch({ headless: true });
  try {
    const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: SCALE });
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(url);
    // Settled: every panel drawn or showing its state, then two more frames.
    await page.waitForFunction(() => {
      const s = window.caliperSpec?.debugState();
      return !!s && s.status === 'connected' && s.panels.length > 0 && s.panels.every((p) => p.drawn || p.state !== null);
    }, null, { timeout: 30_000 });
    await page.evaluate(() => new Promise((ok) => requestAnimationFrame(() => requestAnimationFrame(ok))));
    const state = await page.evaluate(() => {
      const s = window.caliperSpec.debugState();
      const c = document.getElementById('canvas').getBoundingClientRect();
      return { ...s, canvas: { x: c.x, y: c.y, w: c.width, h: c.height },
        status_text: document.getElementById('status')?.textContent ?? '' };
    });
    state.errors = errors;
    await page.screenshot({ path: out });
    fs.writeFileSync(stateFile, JSON.stringify(state, null, 2) + '\n');
    console.log(`browser: "${state.title}", ${state.panels.length} panels, status "${state.status_text}"`);
    for (const p of state.panels) console.log(`  ${p.id} ${p.primitive} drawn=${p.drawn} info="${p.info}" rect=${JSON.stringify(p.rect)}`);
    if (errors.length) console.log(`  page errors: ${errors.join(' | ')}`);
  } finally {
    await browser.close();
  }
}

async function compare(appShot, browserShot, stateFile, sideOut, panelsOut) {
  const state = JSON.parse(fs.readFileSync(stateFile, 'utf8'));
  const uri = (f) => `data:image/png;base64,${fs.readFileSync(f).toString('base64')}`;
  const app = uri(appShot), web = uri(browserShot);
  // The page's size in points, from the browser capture's PNG header.
  const head = fs.readFileSync(browserShot).subarray(0, 24);
  const width = Math.round(head.readUInt32BE(16) / SCALE), height = Math.round(head.readUInt32BE(20) / SCALE);
  const loaded = (p) => p.waitForFunction(() => [...document.images].every((i) => i.complete && i.naturalWidth > 0));
  const browser = await chromium.launch({ headless: true });
  try {
    // Whole pages, side by side at half size.
    const side = await browser.newPage({ viewport: { width: 2 * width + 8, height }, deviceScaleFactor: 1 });
    await side.setContent(`<body style="margin:0;background:#000;display:flex;gap:8px">` +
      `<img src="${app}" style="width:${width}px;height:${height}px"><img src="${web}" style="width:${width}px;height:${height}px"></body>`);
    await loaded(side);
    await side.screenshot({ path: sideOut });
    // Each panel's rectangle, as the page lays it out, cut from both at full size.
    const rows = state.panels.map((p) => {
      const x = Math.round((state.canvas.x + p.rect.x) * SCALE), y = Math.round((state.canvas.y + p.rect.y) * SCALE);
      const w = Math.round(p.rect.w * SCALE), h = Math.round(p.rect.h * SCALE);
      const cut = (src) => `<div style="width:${w}px;height:${h}px;overflow:hidden;position:relative;flex:none">` +
        `<img src="${src}" style="position:absolute;left:-${x}px;top:-${y}px"></div>`;
      return `<div style="display:flex;gap:8px;margin-bottom:8px">${cut(app)}${cut(web)}</div>`;
    }).join('');
    const pw = Math.max(...state.panels.map((p) => Math.round(p.rect.w * SCALE))) * 2 + 8;
    const ph = state.panels.reduce((a, p) => a + Math.round(p.rect.h * SCALE) + 8, 0);
    const cuts = await browser.newPage({ viewport: { width: pw, height: ph }, deviceScaleFactor: 1 });
    await cuts.setContent(`<body style="margin:0;background:#000">${rows}</body>`);
    await loaded(cuts);
    await cuts.screenshot({ path: panelsOut, fullPage: true });
  } finally {
    await browser.close();
  }
}

if (mode === 'capture' && args.length === 4) await capture(...args);
else if (mode === 'compare' && args.length === 5) await compare(...args);
else {
  console.error('usage: node browser_capture.mjs capture <url> <WxH> <browser.png> <state.json>\n' +
                '       node browser_capture.mjs compare <app.png> <browser.png> <state.json> <side.png> <panels.png>');
  process.exit(2);
}
