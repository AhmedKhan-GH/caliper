// Frame rate of the points view while orbiting: 100,000 points in 3-D on
// every path, and 20,000 on Canvas 2D, the size its fallback is meant for.
// Under the SwiftShader flags every path, Canvas 2D included, is rasterised
// in software, so that project measures the software case.
// The numbers are printed and written to perf-<project>.json beside the
// screenshots; the assertions only guard against a broken path.

import fs from 'node:fs';
import path from 'node:path';
import { expect, SHOTS, test } from './util.js';

async function measure(page: import('@playwright/test').Page, n: number) {
  return page.evaluate(async (n) => {
    const w = window as unknown as { harness: { StaticSource: new () => { addView(a: string, b: string): void; snapshot(v: string, s: unknown): void } } };
    const src = new w.harness.StaticSource();
    src.addView('cloud', 'points');
    const el = document.createElement('caliper-view') as HTMLElement & { source: unknown; settled(): Promise<void>; debugState(): Record<string, unknown> };
    el.setAttribute('view', 'cloud');
    el.setAttribute('height', '600');
    el.style.width = '800px';
    el.source = src;
    document.getElementById('root')!.append(el);
    const data = new Float32Array(n * 3), aux = new Float32Array(n);
    let s = 1;
    const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647) * 2 - 1;
    for (let i = 0; i < n; i++) {
      const c = i % 8;
      data.set([Math.cos(c) * 3 + rnd(), Math.sin(c) * 3 + rnd(), (c % 2 ? 1 : -1) + rnd()], i * 3);
      aux[i] = c;
    }
    src.snapshot('cloud', { data, aux, shape: [n, 3], points_dims: 3, label_count: 8 });
    await el.settled();
    const overlay = el.shadowRoot!.querySelector('canvas.overlay')!;
    const r = overlay.getBoundingClientRect();
    let x = r.left + r.width / 2;
    const y = r.top + r.height / 2;
    const ev = (type: string) => overlay.dispatchEvent(new PointerEvent(type, { clientX: x, clientY: y, bubbles: true, pointerId: 1, button: 0, buttons: 1 }));
    ev('pointerdown');
    // Paced: one drag step per animation frame for 3 seconds.
    const f0 = el.debugState().frames as number;
    const t0 = performance.now();
    while (performance.now() - t0 < 3000) {
      x += 2;
      ev('pointermove');
      await new Promise((ok) => requestAnimationFrame(ok));
    }
    await el.settled();
    const paced = ((el.debugState().frames as number) - f0) / ((performance.now() - t0) / 1000);
    // The frame's own cost: from the animation frame that drew it until the
    // GPU finished it, so without the wait for the next frame.
    const times: number[] = [], cpu: number[] = [];
    for (let i = 0; i < 40; i++) {
      x += 2;
      ev('pointermove');
      await new Promise((ok) => requestAnimationFrame(ok));
      const t = performance.now() - (el.debugState().lastDrawMs as number);
      cpu.push(el.debugState().lastDrawMs as number);
      await el.settled();
      times.push(performance.now() - t);
    }
    ev('pointerup');
    const median = (a: number[]) => Math.round(a.sort((p, q) => p - q)[a.length >> 1] * 10) / 10;
    const st = el.debugState();
    return { n, renderer: st.renderer, adapter: st.adapter, pacedFps: Math.round(paced * 10) / 10, frameMs: median(times), cpuMs: median(cpu) };
  }, n);
}

test('points: frame rate while orbiting', async ({ harness: page }, info) => {
  // Not on swiftshader in CI (playwright.config.ts): a hosted runner without a
  // GPU and with few cores draws 100,000 points in SwiftShader at well under one
  // frame a second (1.6 s a frame on a 18-core Mac), so the measurement says
  // nothing and outlasts any timeout. It runs there locally, and the webgpu
  // project measures the GPU.
  test.setTimeout(120_000);
  const runs = [await measure(page, 100_000)];
  if (info.project.name === 'canvas2d') runs.push(await measure(page, 20_000));
  console.log(info.project.name, JSON.stringify(runs));
  fs.mkdirSync(SHOTS, { recursive: true });
  fs.writeFileSync(path.join(SHOTS, `perf-${info.project.name}.json`), JSON.stringify(runs, null, 2));
  for (const r of runs) expect(r.pacedFps).toBeGreaterThan(0);
});
