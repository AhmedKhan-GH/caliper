// 3-D points in both browser renderers, the element's own TypeScript view
// (renderer="ts": WebGPU, or Canvas 2D without an adapter) and Caliper's views
// compiled to WebAssembly (renderer="wasm"), fed the same fixed scene by
// mock_server.py --scene: the box with its ticks, the default camera, the
// depth cues and the rotate toggle. web/wasm/README.md and
// web/src/views/box3d.ts describe them.

import { test, type Page } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { CAMERA } from '../src/views/camera.js';
import { PALETTE } from '../src/render/colormap.js';
import { bodyPixels, expect, shotPath, WEB } from './util.js';
import { HAVE_PYTHON, startProducer, WASM, type Producer } from './producer.js';

test.skip(!HAVE_PYTHON, 'needs a Python with numpy and websockets (CALIPER_PYTHON)');

const HAVE_WASM = ['caliper-views-loader.js', 'caliper-views.js', 'caliper-views.wasm'].every((f) => fs.existsSync(path.join(WASM, f)));
type El = HTMLElement & { debugState(): Record<string, unknown> };
type Renderer = 'ts' | 'wasm';

/** The renderers this project can show: the TypeScript view always, the WebAssembly views with an adapter. */
function renderers(project: string): Renderer[] {
  return project === 'canvas2d' || !HAVE_WASM ? ['ts'] : ['ts', 'wasm'];
}

/** A producer of the scene that also serves test/points3d.html beside the web package's build. */
async function producer(args: string[]): Promise<Producer & { dir: string }> {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'caliper-points3d-'));
  for (const f of fs.readdirSync(path.join(WEB, 'dist'))) fs.copyFileSync(path.join(WEB, 'dist', f), path.join(dir, f));
  fs.copyFileSync(path.join(WEB, 'test', 'points3d.html'), path.join(dir, 'points3d.html'));
  const p = await startProducer(['--dist', dir, '--step-ms', '100000', ...args]);
  return { ...p, dir, stop: () => { p.stop(); fs.rmSync(dir, { recursive: true, force: true }); } };
}

/** The page with one element per renderer, waited on until each has drawn its box (or, in 2-D, its data). */
async function open(page: Page, prod: Producer, which: Renderer[], dims = 3): Promise<void> {
  const u = new URL(prod.url);
  await page.setViewportSize({ width: 1200, height: 960 });
  await page.goto(`${u.origin}/points3d.html?token=${u.searchParams.get('token')}&r=${which.join(',')}`);
  await page.waitForFunction(([ids, dims]) => (ids as string[]).every((id) => {
    const s = (document.getElementById(id) as El | null)?.debugState();
    return !!s && s.status === 'connected' && !!s.renderer && (id !== 'wasm' || s.wasmReady) &&
      s.dims === dims && (dims === 2 || String(s.box ?? '').includes('x='));
  }), [which, dims] as const, { timeout: 30_000 });
  await page.waitForTimeout(300);
}

const state = (page: Page, id: string) => page.evaluate((i) => (document.getElementById(i) as El).debugState(), id);

test('the 3-D view has a box with ticks on x, y and z and the default three-quarter camera, the same in both renderers', async ({ page }, info) => {
  const prod = await producer(['--scene', 'clusters']);
  try {
    const which = renderers(info.project.name);
    await open(page, prod, which);
    const got = await Promise.all(which.map((r) => state(page, r)));
    for (const [k, s] of got.entries()) {
      const cam = s.camera as { az: number; el: number; dist: number };
      expect(cam.az, which[k]).toBeCloseTo(CAMERA.AZ, 5);
      expect(cam.el, which[k]).toBeCloseTo(CAMERA.EL, 5);
      expect(String(s.box), which[k]).toMatch(/^x=[^;]+;y=[^;]+;z=[^;]+$/);
      expect(s.rotate).toBe(false);
    }
    if (got.length === 2) {
      // The same fit and the same labels: the renderers agree.
      const [a, b] = got.map((s) => s.camera as { dist: number });
      expect(b.dist).toBeCloseTo(a.dist, 3);
      expect(got[1].box).toBe(got[0].box);
    }
    for (const r of which) await page.locator(`#${r}`).screenshot({ path: shotPath(info, `points3d-box-${r}`) });
  } finally {
    prod.stop();
  }
});

/** Pixels of exactly rgb, and pixels that are rgb scaled down with its hue kept. */
function census(px: { data: Uint8Array }, rgb: readonly number[]) {
  let exact = 0, dimmed = 0, darkest = 255;
  const m = Math.max(...rgb);
  for (let i = 0; i < px.data.length; i += 4) {
    const p = [px.data[i], px.data[i + 1], px.data[i + 2]];
    if (p[0] === rgb[0] && p[1] === rgb[1] && p[2] === rgb[2]) {
      exact++;
      continue;
    }
    const f = Math.max(...p) / m;
    if (f < 0.45 || f >= 1) continue;
    if (p.every((v, k) => Math.abs(v - rgb[k] * f) <= 2)) {
      dimmed++;
      darkest = Math.min(darkest, Math.max(...p));
    }
  }
  return { exact, dimmed, darkest };
}

test('a point in front of the target is drawn larger than one behind it, which is dimmer with its hue kept', async ({ page }, info) => {
  // Anchors at the box's corners (green), one point in front of the centre (orange), one behind it (red).
  const prod = await producer(['--scene', 'depth']);
  try {
    const which = renderers(info.project.name);
    await open(page, prod, which);
    for (const r of which) {
      const { px } = await bodyPixels(page, `#${r}`);
      const near = census(px, PALETTE[1]), far = census(px, PALETTE[3]);
      console.log(`${info.project.name} ${r}: near ${near.exact} px exact; far ${far.exact} exact, ${far.dimmed} dimmed (brightest channel ${far.darkest})`);
      expect(near.exact, r).toBeGreaterThan(10);
      expect(far.exact, r).toBe(0);
      expect(far.dimmed, r).toBeGreaterThan(5);
      expect(near.exact, r).toBeGreaterThan(far.dimmed);
      expect(far.darkest, r).toBeLessThan(PALETTE[3][0]);
      await page.locator(`#${r}`).screenshot({ path: shotPath(info, `points3d-cues-${r}`) });
    }
  } finally {
    prod.stop();
  }
});

test('the rotate toggle turns the 3-D view until a drag stops it, and a 2-D view has none', async ({ page }, info) => {
  const prod = await producer(['--scene', 'clusters']);
  try {
    const which = renderers(info.project.name);
    await open(page, prod, which);
    for (const r of which) {
      const button = page.locator(`#${r}`).locator('button.tool', { hasText: 'rotate' });
      await expect(button).toHaveAttribute('aria-pressed', 'false');
      const az0 = ((await state(page, r)).camera as { az: number }).az;
      const before = await bodyPixels(page, `#${r}`);
      await button.click();
      await expect(button).toHaveAttribute('aria-pressed', 'true');
      await expect.poll(async () => ((await state(page, r)).camera as { az: number }).az, { timeout: 5000 }).toBeGreaterThan(az0 + 0.05);
      expect((await state(page, r)).rotate).toBe(true);
      const after = await bodyPixels(page, `#${r}`);
      expect(Buffer.compare(Buffer.from(before.px.data), Buffer.from(after.px.data)), r).not.toBe(0);
      // A drag stops it, and the header shows that.
      const box = (await page.locator(`#${r}`).locator('.body').boundingBox())!;
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.mouse.down();
      await page.mouse.move(box.x + box.width / 2 + 40, box.y + box.height / 2, { steps: 4 });
      await page.mouse.up();
      await expect(button).toHaveAttribute('aria-pressed', 'false');
      await expect.poll(async () => (await state(page, r)).rotate, { timeout: 5000 }).toBe(false);
      // The module publishes its state a few times a second: let the drag's own turn arrive first.
      await page.waitForTimeout(500);
      const az1 = ((await state(page, r)).camera as { az: number }).az;
      await page.waitForTimeout(600);
      expect(((await state(page, r)).camera as { az: number }).az, r).toBeCloseTo(az1, 6);
    }
  } finally {
    prod.stop();
  }
  // 2-D points: no rotate toggle and no box.
  const flat = await producer(['--dims', '2']);
  try {
    const which = renderers(info.project.name);
    await open(page, flat, which, 2);
    for (const r of which) {
      await expect(page.locator(`#${r}`).locator('button.tool', { hasText: 'rotate' })).toHaveCount(0);
      expect((await state(page, r)).box ?? '', r).toBe('');
    }
  } finally {
    flat.stop();
  }
});
