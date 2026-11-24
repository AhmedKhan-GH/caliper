import fs from 'node:fs';
import path from 'node:path';
import type { Expect } from '../../web/node_modules/@playwright/test/index';
import { REPO, WEB, serve } from '../../web/test/util.js';
import { mockApp, openSpec, ready, VIEWS_FILES, type MockApp } from '../../web/test/spec-mock.js';

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

  test('the browser spec page shows pointer hover labels and preserves local picks', async ({ page }, info) => {
    test.skip(info.project.name === 'canvas2d', 'Piece draws require WebGPU.');
    const fixture = process.env.CALIPER_INTERP_PIECES_OUT ?? path.join(REPO, 'cmake-build-release/tests/interp_pieces/out');
    const wasm = process.env.CALIPER_WASM_DIR ?? path.join(REPO, 'python/caliper/_web');
    const available = fs.existsSync(path.join(fixture, 'hover.json')) && VIEWS_FILES.every((f) => fs.existsSync(path.join(wasm, f)));
    if (process.env.CALIPER_REQUIRE_INTERP_PIECES === '1') expect(available, 'The current native hover fixture and WASM views must exist.').toBe(true);
    test.skip(!available, 'The native hover fixture or WASM views are absent.');
    test.setTimeout(120_000);
    const program = JSON.parse(fs.readFileSync(path.join(fixture, 'hover.json'), 'utf8'));
    expect(program.instance.manifest.draw.text.hover).toEqual({ anchor: 33, string_offset: 32 });
    program.params.hovered_node = -1;
    program.instance.params.hovered_node = -1;
    for (const r of program.resources) { r.base64 = Buffer.from(r.bytes).toString('base64'); delete r.bytes; }
    const panel = program.id as string;
    const spec = {
      caliper_spec: 1, applet: { id: 'hover-test', title: 'Hover Test' },
      panel: [{ id: panel, primitive: 'scene', bind: {}, options: { labels: true, select: { state: 'selected_node', field: 'index' } } }],
      program: [{ id: panel, params: { selected_node: { state: 'selected_node', default: 4 } } }], layout: panel,
    };
    const app: MockApp = { specs: { 'hover-test': spec }, data: { 'hover-test': {
      [panel]: ready(panel, 'scene', { scene: null, programs: [program] }),
    } }, requests: [] };
    await mockApp(page, app, wasm);
    await page.setViewportSize({ width: 800, height: 640 });
    await page.mouse.move(1, 1);
    await openSpec(page, 'hover-test');
    await page.waitForFunction((id) => {
      const p = (window as any).caliperSpec?.debugState().programs[id]?.programs[0];
      return p?.module === 'running' && p.frames > 0;
    }, panel, { timeout: 60_000, polling: 'raf' });
    const labels = () => page.evaluate((id) => {
      const raw = (window as any).caliperSpec.programs.mounts.get(id)?.handle?.stats.pageLabels ?? '[]';
      return (JSON.parse(raw) as [string, ...number[]][]).map((label) => label[0]);
    }, panel);
    await expect.poll(labels).toContain('label 36');
    // Feature discs have green pixels. Connected components give actual visible node centers.
    // The later click proves the hovered node's ID without a synthetic hover parameter.
    const centers = await page.evaluate((id) => {
      const canvas = (window as any).caliperSpec.programs.canvasOf(id) as HTMLCanvasElement;
      const probe = document.createElement('canvas'); probe.width = canvas.width; probe.height = canvas.height;
      const context = probe.getContext('2d', { willReadFrequently: true })!;
      context.drawImage(canvas, 0, 0);
      const pixels = context.getImageData(0, 0, probe.width, probe.height).data;
      const mask = new Uint8Array(probe.width * probe.height);
      for (let i = 0; i < mask.length; ++i) {
        const r = pixels[4 * i], g = pixels[4 * i + 1], b = pixels[4 * i + 2];
        if (g > r + 45 && g > b + 20 && g > 100) mask[i] = 1;
      }
      const bounds = canvas.getBoundingClientRect(), found: { x: number; y: number; count: number }[] = [];
      for (let i = 0; i < mask.length; ++i) {
        if (!mask[i]) continue;
        const todo = [i]; mask[i] = 0;
        let x = 0, y = 0, count = 0;
        while (todo.length) {
          const p = todo.pop()!, px = p % probe.width, py = Math.floor(p / probe.width);
          x += px; y += py; ++count;
          for (const [nx, ny] of [[px - 1, py], [px + 1, py], [px, py - 1], [px, py + 1]]) {
            if (nx < 0 || ny < 0 || nx >= probe.width || ny >= probe.height) continue;
            const next = ny * probe.width + nx;
            if (mask[next]) { mask[next] = 0; todo.push(next); }
          }
        }
        if (count >= 4) found.push({ x: bounds.left + (x / count + 0.5) * bounds.width / probe.width,
          y: bounds.top + (y / count + 0.5) * bounds.height / probe.height, count });
      }
      return found.sort((a, b) => b.count - a.count);
    }, panel);
    expect(centers.length).toBeGreaterThanOrEqual(2);
    const hoverLabels = async () => (await labels()).filter((s) => /^label (3[2-9]|4[0-7])$/.test(s) && s !== 'label 36');
    await page.mouse.move(centers[0].x, centers[0].y);
    await expect.poll(hoverLabels).toHaveLength(1);
    const first = (await hoverLabels())[0];
    await page.mouse.move(centers[1].x, centers[1].y);
    await expect.poll(hoverLabels).toHaveLength(1);
    await expect.poll(async () => (await hoverLabels())[0]).not.toBe(first);
    const second = (await hoverLabels())[0];
    expect(await page.evaluate(() => (window as any).caliperSpec.debugState().picks.length)).toBe(0);
    expect(await page.evaluate(() => (window as any).caliperSpec.programs.localState)).toEqual({});
    await page.screenshot({ path: info.outputPath('hover-node.png') });
    await page.mouse.move(1, 1);
    await expect.poll(hoverLabels).toHaveLength(0);
    expect(await labels()).toContain('label 36');
    await page.screenshot({ path: info.outputPath('hover-exit.png') });
    await page.mouse.click(centers[1].x, centers[1].y);
    await expect.poll(() => page.evaluate(() => (window as any).caliperSpec.debugState().picks.length)).toBe(1);
    const picked = await page.evaluate(() => (window as any).caliperSpec.debugState().picks[0].detail);
    expect(second).toBe(`label ${32 + picked.item}`);
    expect(picked.fields.index).toBe(picked.item);
    expect(await page.evaluate(() => (window as any).caliperSpec.programs.localState.selected_node)).toBe(picked.item);
    expect(app.requests.filter((r) => ['state.set', 'selection.set', 'trace.run'].includes(r.name))).toEqual([]);
  });
}
