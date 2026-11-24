// Actual fixture resources through the shared browser draw parity entrypoint.
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const { chromium } = createRequire(path.join(root, 'web/package.json'))('@playwright/test');
const out = process.argv[2];
const wasm = process.env.CALIPER_WASM_DIR ?? path.join(root, 'python/caliper/_web');
const files = ['caliper-views-loader.js', 'caliper-views.js', 'caliper-views.wasm'];
for (const file of files) if (!fs.existsSync(path.join(wasm, file))) throw new Error(`Missing ${file}`);
const server = http.createServer((req, res) => {
  const name = new URL(req.url, 'http://127.0.0.1').pathname.slice(1);
  if (!name) return res.end('<!doctype html><html><body></body></html>');
  if (!files.includes(name)) return res.writeHead(404).end();
  res.setHeader('Content-Type', name.endsWith('.js') ? 'text/javascript' : 'application/wasm');
  fs.createReadStream(path.join(wasm, name)).pipe(res);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const site = { origin: `http://127.0.0.1:${server.address().port}` };
let browser;
try {
  browser = await chromium.launch({headless: true, args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=swiftshader']});
  const page = await browser.newPage({ viewport: { width: 512, height: 512 }, deviceScaleFactor: 1 });
  page.on('console', message => console.log(message.text()));
  page.on('pageerror', error => console.error(error));
  await page.goto(site.origin);
  const frames = fs.readFileSync(path.join(out, 'frames.bin')).toString('base64');
  const plan = JSON.parse(fs.readFileSync(path.join(out, 'plan.json'), 'utf8'));
      const result = await page.evaluate(async ({ base, frames, plan }) => {
        const { preloadCaliperViews } = await import(base + 'caliper-views-loader.js');
        const { factory, wasm } = await preloadCaliperViews(base);
        const canvas = document.createElement('canvas');
        document.body.append(canvas);
        const module = await factory({ canvas, caliperConfig: { src: '' },
          instantiateWasm: (imports, receive) => {
            WebAssembly.instantiate(wasm, imports).then((i) => receive(i, wasm));
            return {};
          }, print: (s) => console.log(s), printErr: (s) => console.error(s),
        });
        const until = async (ready) => {
          for (let frame = 0; frame < 6000 && !ready(); frame++)
            await new Promise((resolve) => requestAnimationFrame(resolve));
          if (!ready()) throw new Error('The browser module did not publish completion.');
        };
        const encode = (b) => {
          let s = '';
          for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode(...b.subarray(i, i + 0x8000));
          return btoa(s);
        };
        try {
          await until(() => Number(module.caliperStats?.gfxReady) === 1);
          const bytes = Uint8Array.from(atob(frames), (ch) => ch.charCodeAt(0));
          const p = module._malloc(bytes.length), q = module.stringToNewUTF8(plan);
          module.HEAPU8.set(bytes, p);
          const started = module._caliper_views_draw_parity(p, bytes.length, q);
          module._free(p); module._free(q);
          if (!started) throw new Error('The browser rejected the fixture.');
          await until(() => !!module.caliperStats?.drawParity);
          const drawn = {};
          for (const [id, raw] of Object.entries(module.caliperDrawParity ?? {})) {
            const r = raw;
            drawn[id] = { code: r.code, message: r.message, color: encode(r.color), ids: encode(r.ids) };
          }
          return { report: module.caliperStats.drawParity, errors: module.caliperStats.gfxErrors ?? '', drawn };
        } finally { module._caliper_views_destroy(); canvas.remove(); }
      }, { base: `${site.origin}/`, frames, plan: JSON.stringify(plan) });
  if (!result.report.startsWith('ok ') || result.errors) throw new Error(JSON.stringify(result));
  const graph = result.drawn.graph;
  if (!graph || graph.code) throw new Error(JSON.stringify(graph));
  fs.writeFileSync(path.join(out, 'browser.rgba'), Buffer.from(graph.color, 'base64'));
  fs.writeFileSync(path.join(out, 'browser.id'), Buffer.from(graph.ids, 'base64'));
  fs.writeFileSync(path.join(out, 'browser-result.json'), JSON.stringify({report: result.report, errors: result.errors}));
} finally {
  if (browser) await browser.close();
  await new Promise(resolve => server.close(resolve));
}
