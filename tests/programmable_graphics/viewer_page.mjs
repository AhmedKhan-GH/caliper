// The browser's side of tests/programmable_graphics/viewer_driver.py: opens the real spec page of the
// hidden app in headless Chromium and prints one JSON line of what the page drew.
//
//   node viewer_page.mjs --web <repo>/web --url <viewer link> --project webgpu|swiftshader --mode programs|not-open
//
// Mode programs: waits until every program of the page's scene panels reports "running" in the module
// (or stops), then, for 10 frames, counts in each module canvas the pixels that differ from the
// canvas's own background, in the same task as the module's frame: each animation callback is
// followed, in its task, by a copy of every module canvas into a 2D canvas, and the largest count of
// each panel is kept (a WebGPU canvas keeps its texture only until the end of the task that drew it).
// Mode not-open: waits for the page's banner and its panels' states. Waits on published state only.
import path from 'node:path';
import { createRequire } from 'node:module';

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const web = arg('web'), url = arg('url'), project = arg('project', 'webgpu'), mode = arg('mode', 'programs');
const { chromium } = createRequire(path.join(web, 'package.json'))('@playwright/test');
const flags = {
  webgpu: ['--enable-unsafe-webgpu', '--use-angle=metal'],
  swiftshader: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=swiftshader'],
}[project];
const browser = await chromium.launch({ headless: true, args: flags });
const out = { project, mode };
let page = null;
const lines = [];
try {
  page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  page.on('console', (m) => lines.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => lines.push(`[pageerror] ${e.message}`));
  await page.addInitScript(() => {
    const w = window;
    w.__viewer = { armed: 0, counts: {}, frames: 0 };
    const original = w.requestAnimationFrame.bind(w);
    const probe = document.createElement('canvas');
    const g = probe.getContext('2d', { willReadFrequently: true });
    const count = () => {
      const v = w.__viewer;
      for (const m of w.caliperSpec?.programs?.mounts?.values() ?? []) {
        const c = m.handle?.canvas;
        if (!c || !c.width || !c.height) continue;
        probe.width = c.width;
        probe.height = c.height;
        g.clearRect(0, 0, c.width, c.height);
        g.drawImage(c, 0, 0);
        const d = g.getImageData(0, 0, c.width, c.height).data;
        // The background: the colour of the top-left pixel, where no program of these pieces draws.
        const b = [d[0], d[1], d[2]];
        let n = 0;
        for (let i = 0; i < d.length; i += 4)
          if (Math.max(Math.abs(d[i] - b[0]), Math.abs(d[i + 1] - b[1]), Math.abs(d[i + 2] - b[2])) > 24) n++;
        v.counts[m.panel] = Math.max(v.counts[m.panel] ?? 0, n);
      }
    };
    w.requestAnimationFrame = (cb) => original((t) => {
      cb(t);
      if (w.__viewer.armed > 0) count();
    });
    // The frames of the count: each one ends with a callback registered after the others.
    w.__viewer.run = (frames) => new Promise((done) => {
      w.__viewer.armed = 1;
      const tick = () => {
        w.__viewer.frames++;
        if (w.__viewer.frames >= frames) {
          w.__viewer.armed = 0;
          done(w.__viewer.counts);
        } else original(tick);
      };
      original(tick);
    });
  });
  await page.goto(url);
  if (mode === 'programs') {
    await page.waitForFunction(() => {
      const s = window.caliperSpec?.debugState();
      const mounts = s ? Object.values(s.programs) : [];
      return mounts.length > 0 && mounts.every((m) => m.failed || (m.mounted && m.programs.length > 0 &&
        m.programs.every((p) => (p.module === 'running' && p.frames >= 1) || p.module === 'stopped' || p.module === 'refused')));
    }, null, { timeout: 120000, polling: 200 });
    out.counts = await page.evaluate(() => window.__viewer.run(10));
    const s = await page.evaluate(() => window.caliperSpec.debugState());
    out.programs = Object.fromEntries(Object.entries(s.programs).map(([id, m]) => [id, {
      mounted: m.mounted, failed: m.failed, programs: m.programs.map((p) => ({ id: p.id, state: p.state, module: p.module, code: p.code, frames: p.frames })),
    }]));
  } else {
    await page.waitForFunction(() => {
      const s = window.caliperSpec?.debugState();
      return s && s.notOpen && s.panels.length > 0 && s.panels.every((p) => p.state !== null);
    }, null, { timeout: 60000, polling: 200 });
    const s = await page.evaluate(() => window.caliperSpec.debugState());
    out.banner = s.banner;
    out.panels = s.panels.map((p) => ({ id: p.id, state: p.state }));
  }
  out.console = lines.slice(-20);
} catch (e) {
  out.error = String(e);
  // What the page shows when the wait ends without its state: its panels, programs and requests.
  out.state = await page?.evaluate(() => {
    const s = window.caliperSpec?.debugState();
    return s ? { status: s.status, state: s.state, notOpen: s.notOpen, requests: s.requests, programs: s.programs,
      panels: s.panels.map((p) => [p.id, p.drawn, p.state]) } : null;
  }).catch((x) => String(x));
  out.console = lines.slice(-20);
} finally {
  await browser.close();
}
console.log(JSON.stringify(out));
