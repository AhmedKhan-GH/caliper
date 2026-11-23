// Each scene fixture draws through library pieces on core and compatibility WebGPU devices.
// The report checks piece count, nonempty pixels, and scene errors. draw_parity.spec.ts compares
// the app and browser color, ID, and depth targets for the same fixtures.
//
// The files come from CALIPER_WASM_DIR, else from python/caliper/_web, where scripts/build_wasm.py
// puts them; without them these tests are skipped.

import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { REPO, serve, shotPath, WEB } from './util.js';

const WASM = process.env.CALIPER_WASM_DIR ?? path.join(REPO, 'python', 'caliper', '_web');
const FILES = ['caliper-views-loader.js', 'caliper-views.js', 'caliper-views.wasm'];
const FIXTURES = ['arrows', 'boxes', 'edges', 'mesh', 'cameras', 'through', 'images'];

test.skip(!FILES.every((f) => fs.existsSync(path.join(WASM, f))), `needs the WebAssembly views in ${WASM}`);

for (const compat of [false, true]) {
  const level = compat ? 'compatibility' : 'core';
  test(`every primitive drawn by the scene view draws with library pieces on a ${level} device`, async ({ page }, info) => {
    test.skip(info.project.name === 'canvas2d', 'the WebAssembly views need a WebGPU adapter');
    const site = await serve(WEB, (name) => (FILES.includes(name.slice(1)) ? path.join(WASM, name.slice(1)) : null));
    try {
      await page.goto(`${site.origin}/test/blank.html`);
      const got = await page.evaluate(async ({ base, compat, fixtures }) => {
        type M = { caliperStats?: Record<string, string>; caliperScenePixels?: Uint8Array;
                   _caliper_views_scene_selftest(name: number): number; _caliper_views_destroy(): void;
                   stringToNewUTF8(s: string): number; _free(p: number): void };
        const { preloadCaliperViews } = await import(base + 'caliper-views-loader.js');
        const { factory, wasm } = await preloadCaliperViews(base);
        const canvas = document.createElement('canvas');
        canvas.style.width = '64px';
        canvas.style.height = '64px';
        document.body.append(canvas);
        const m: M = await factory({
          canvas, caliperConfig: { src: '', compat: compat ? 'true' : 'false' },
          instantiateWasm: (imports: WebAssembly.Imports, receive: (i: WebAssembly.Instance, m: WebAssembly.Module) => void) => {
            WebAssembly.instantiate(wasm as WebAssembly.Module, imports).then((i) => receive(i, wasm));
            return {};
          },
          print: () => {}, printErr: () => {},
        });
        // Waits on what the module publishes each frame, never on a fixed time.
        const until = async (f: () => boolean) => {
          for (let i = 0; i < 2400 && !f(); i++) await new Promise((ok) => requestAnimationFrame(() => ok(null)));
          return f();
        };
        if (!(await until(() => Number(m.caliperStats?.gfxReady) === 1)))
          return { reports: { start: `no layer: ${JSON.stringify(m.caliperStats ?? {})}` }, images: {}, names: '' };
        const reports: Record<string, string> = {};
        const images: Record<string, string> = {};
        for (const name of fixtures) {
          const p = m.stringToNewUTF8(name);
          const started = m._caliper_views_scene_selftest(p);
          m._free(p);
          if (!started) {
            reports[name] = m.caliperStats?.sceneSelftest ?? 'fail not started';
            continue;
          }
          await until(() => !!m.caliperStats?.sceneSelftest);
          reports[name] = m.caliperStats?.sceneSelftest ?? 'fail no report';
          const px = m.caliperScenePixels;
          if (px && px.length === 256 * 256 * 4) {
            const c = document.createElement('canvas');
            c.width = c.height = 256;
            c.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(px), 256, 256), 0, 0);
            images[name] = c.toDataURL('image/png');
          }
        }
        const names = m.caliperStats?.sceneFixtures ?? '';
        m._caliper_views_destroy();
        canvas.remove();
        return { reports, images, names };
      }, { base: `${site.origin}/`, compat, fixtures: FIXTURES });
      expect(got.names.split(','), 'the module knows the same fixtures').toEqual(FIXTURES);
      for (const [name, url] of Object.entries(got.images))
        fs.writeFileSync(shotPath(info, `scene-${name}-${level}`), Buffer.from(url.split(',')[1], 'base64'));
      for (const name of FIXTURES) {
        const r = got.reports[name] ?? 'fail no report';
        console.log(`[${info.project.name}] ${level}: ${r}`);
        expect(r, name).toMatch(/^ok /);
        expect(r, name).toContain('notes=0');
        expect(r, name).toContain(`level=${level}`);
      }
    } finally {
      site.close();
    }
  });
}

// The scene fixture exposes its library pieces and program identities.
test('the scene views draw their shapes with the library pieces', async ({ page }, info) => {
  test.skip(info.project.name === 'canvas2d', 'the WebAssembly views need a WebGPU adapter');
  const site = await serve(WEB, (name) => (FILES.includes(name.slice(1)) ? path.join(WASM, name.slice(1)) : null));
  try {
    await page.goto(`${site.origin}/test/blank.html`);
    const facts = async () => page.evaluate(async ({ base }) => {
      type M = { caliperStats?: Record<string, string>; caliperSceneParity?: { facts: string };
                 _caliper_views_scene_parity(config: number): number; _caliper_views_destroy(): void;
                 _free(p: number): void; stringToNewUTF8(s: string): number };
      const { preloadCaliperViews } = await import(base + 'caliper-views-loader.js');
      const { factory, wasm } = await preloadCaliperViews(base);
      const canvas = document.createElement('canvas');
      canvas.style.width = '64px';
      canvas.style.height = '64px';
      document.body.append(canvas);
      const m: M = await factory({
        canvas, caliperConfig: { src: '' },
        instantiateWasm: (imports: WebAssembly.Imports, receive: (i: WebAssembly.Instance, m: WebAssembly.Module) => void) => {
          WebAssembly.instantiate(wasm as WebAssembly.Module, imports).then((i) => receive(i, wasm));
          return {};
        },
        print: () => {}, printErr: () => {},
      });
      // Waits on what the module publishes each frame, never on a fixed time.
      const until = async (f: () => boolean) => {
        for (let i = 0; i < 6000 && !f(); i++) await new Promise((ok) => requestAnimationFrame(() => ok(null)));
        return f();
      };
      if (!(await until(() => Number(m.caliperStats?.gfxReady) === 1))) return { report: 'fail no layer', facts: '{}' };
      const q = m.stringToNewUTF8(JSON.stringify({ fixture: 'boxes', points: [] }));
      const started = m._caliper_views_scene_parity(q);
      m._free(q);
      if (started) await until(() => !!m.caliperStats?.sceneParity);
      const out = { report: m.caliperStats?.sceneParity ?? 'fail no report', facts: m.caliperSceneParity?.facts ?? '{}' };
      m._caliper_views_destroy();
      canvas.remove();
      return out;
    }, { base: `${site.origin}/` });
    const byDefault = await facts();
    expect(byDefault.report).toMatch(/^ok /);
    const p = JSON.parse(byDefault.facts) as { draw: string; pieces: number; programs: object };
    expect(p.draw).toBe('pieces');
    expect(p.pieces, 'the shapes that pieces drew').toBeGreaterThan(0);
    expect(Object.keys(p.programs).length).toBe(p.pieces);
  } finally {
    site.close();
  }
});
