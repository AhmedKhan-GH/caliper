import fs from 'node:fs';
import path from 'node:path';
import type { Expect } from '../../web/node_modules/@playwright/test/index';
import { REPO, WEB, serve } from '../../web/test/util.js';

// These limits match web/test/draw_parity.spec.ts: two channel levels in interiors,
// at most one percent differing edge pixels, and identical interior IDs. There is no bloom.
function compare(size: number, app: Uint8Array, aIds: Uint32Array, web: Uint8Array, wIds: Uint32Array) {
  const sameAround = (ids: Uint32Array, x: number, y: number) => {
    const p = y * size + x;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
      const q = ny * size + nx;
      if (ids[2 * q] !== ids[2 * p] || ids[2 * q + 1] !== ids[2 * p + 1]) return false;
    }
    return true;
  };
  const tookSide = (a: Uint8Array, b: Uint8Array, x: number, y: number) => {
    const p = 4 * (y * size + x);
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
      const q = 4 * (ny * size + nx);
      let gap = 0;
      for (let k = 0; k < 4; k++) gap = Math.max(gap, Math.abs(a[p + k] - b[q + k]));
      if (gap <= 2) return true;
    }
    return false;
  };
  let interiorOver = 0, edgeDiffer = 0, idMismatch = 0, maxInterior = 0;
  let first = '';
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const p = y * size + x;
    let gap = 0;
    for (let k = 0; k < 4; k++) gap = Math.max(gap, Math.abs(app[4 * p + k] - web[4 * p + k]));
    if (sameAround(aIds, x, y) && sameAround(wIds, x, y)) {
      maxInterior = Math.max(maxInterior, gap);
      if (gap > 2) {
        interiorOver++;
        if (!first) first = `${x},${y}: ${gap} levels`;
      }
      if (aIds[2 * p] !== wIds[2 * p] || aIds[2 * p + 1] !== wIds[2 * p + 1]) idMismatch++;
    } else if (gap > 2 && !tookSide(app, web, x, y) && !tookSide(web, app, x, y)) edgeDiffer++;
  }
  return { interiorOver, idMismatch, maxInterior, edgeShare: edgeDiffer / (size * size), first };
}

export function registerInterpPieceTests(test: typeof import('../../web/node_modules/@playwright/test/index').test, expect: Expect) {
  test('interpretation pieces preserve native colors and picks in the browser', async ({ page }, info) => {
    test.skip(info.project.name === 'canvas2d', 'Piece draws require WebGPU.');
    const fixture = process.env.CALIPER_INTERP_PIECES_OUT ?? path.join(REPO, 'cmake-build-release/tests/interp_pieces/out');
    const wasm = process.env.CALIPER_WASM_DIR ?? path.join(REPO, 'python/caliper/_web');
    const files = ['caliper-views-loader.js', 'caliper-views.js', 'caliper-views.wasm'];
    const hasFixture = fs.existsSync(path.join(fixture, 'frames.bin'));
    const hasViews = files.every((f) => fs.existsSync(path.join(wasm, f)));
    if (process.env.CALIPER_REQUIRE_INTERP_PIECES === '1') {
      expect(hasFixture, 'Metal is available, and CTest must export the interpretation fixture.').toBe(true);
      expect(hasViews, 'The required WebAssembly views must exist.').toBe(true);
    }
    test.skip(!hasFixture, 'The offscreen native fixture requires a Metal backend.');
    test.skip(!hasViews, 'The WebAssembly views are absent.');
    test.setTimeout(180_000);
    const plan = JSON.parse(fs.readFileSync(path.join(fixture, 'plan.json'), 'utf8'));
    const picks = JSON.parse(fs.readFileSync(path.join(fixture, 'picks.json'), 'utf8'));
    const frames = fs.readFileSync(path.join(fixture, 'frames.bin')).toString('base64');
    expect(plan.programs.map((p: { id: string }) => p.id).sort()).toEqual(
      ['examples_0', 'examples_1', 'graph', 'graph_selected', 'logits', 'recovery']);
    page.on('console', (message) => console.log(`browser: ${message.text()}`));
    const site = await serve(WEB, (name) => files.includes(name.slice(1)) ? path.join(wasm, name.slice(1)) : null);
    try {
      await page.goto(`${site.origin}/test/blank.html`);
      const result = await page.evaluate(async ({ base, frames, plan }) => {
        const { preloadCaliperViews } = await import(base + 'caliper-views-loader.js');
        const { factory, wasm } = await preloadCaliperViews(base);
        const canvas = document.createElement('canvas');
        document.body.append(canvas);
        const module = await factory({ canvas, caliperConfig: { src: '' },
          instantiateWasm: (imports: WebAssembly.Imports, receive: (i: WebAssembly.Instance, m: WebAssembly.Module) => void) => {
            WebAssembly.instantiate(wasm as WebAssembly.Module, imports).then((i) => receive(i, wasm));
            return {};
          }, print: (s: string) => console.log(s), printErr: (s: string) => console.error(s),
        });
        const until = async (ready: () => boolean) => {
          for (let frame = 0; frame < 6000 && !ready(); frame++)
            await new Promise((resolve) => requestAnimationFrame(resolve));
          if (!ready()) throw new Error('The browser module did not publish completion.');
        };
        const encode = (b: Uint8Array) => {
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
          const drawn: Record<string, { code: string; message: string; color: string; ids: string }> = {};
          for (const [id, raw] of Object.entries(module.caliperDrawParity ?? {})) {
            const r = raw as { code: string; message: string; color: Uint8Array; ids: Uint8Array };
            drawn[id] = { code: r.code, message: r.message, color: encode(r.color), ids: encode(r.ids) };
          }
          return { report: module.caliperStats.drawParity, errors: module.caliperStats.gfxErrors ?? '', drawn };
        } finally { module._caliper_views_destroy(); canvas.remove(); }
      }, { base: `${site.origin}/`, frames, plan: JSON.stringify(plan) });
      expect(result.report).toMatch(/^ok /);
      expect(result.errors).toBe('');
      for (const pp of plan.programs) {
        const r = result.drawn[pp.id];
        expect(r.code, `${pp.id}: ${r.message}`).toBe('');
        const color = Buffer.from(r.color, 'base64'), ids = Buffer.from(r.ids, 'base64');
        expect(color.length).toBe(plan.size * plan.size * 4);
        expect(ids.length).toBe(plan.size * plan.size * 8);
        const app = fs.readFileSync(path.join(fixture, 'app', pp.id + '.color'));
        const appIds = fs.readFileSync(path.join(fixture, 'app', pp.id + '.id'));
        const webIds = new Uint32Array(ids.buffer, ids.byteOffset, ids.length / 4);
        const comparison = compare(plan.size, app, new Uint32Array(appIds.buffer, appIds.byteOffset, appIds.length / 4), color, webIds);
        console.log(`${pp.id}: ${JSON.stringify(comparison)}`);
        expect.soft(comparison.interiorOver, `${pp.id}: ${comparison.first}`).toBe(0);
        expect.soft(comparison.edgeShare, pp.id).toBeLessThanOrEqual(0.01);
        expect.soft(comparison.idMismatch, pp.id).toBe(0);
        for (const [x, y, item] of picks[pp.id]) {
          const p = 2 * (y * plan.size + x);
          expect(webIds[p], `${pp.id} program at ${x},${y}`).toBe(pp.pick_id);
          expect(webIds[p + 1], `${pp.id} item at ${x},${y}`).toBe(item);
        }
        fs.writeFileSync(info.outputPath(pp.id + '.color'), color);
        fs.writeFileSync(info.outputPath(pp.id + '.id'), ids);
      }
    } finally { site.close(); }
  });
}
