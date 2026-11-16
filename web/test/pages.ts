// The pages of the controls and linked-selection specs, against
// mock_server.py --controls: the producer's viewer page, a second page in
// its own browser context, and the views' and the panel's debug state.

import { type Browser, type Page } from '@playwright/test';

export type Dbg = Record<string, unknown> & {
  status: string; area: { x: number; y: number; w: number; h: number }; window: { x0: number; x1: number; y0: number; y1: number };
  linked: { space: string; axis: string; items: number[] } | null; details: string | null; run: string | null; selected?: number;
};
export const state = (page: Page, view: string) => page.evaluate((v) =>
  (document.querySelector(`caliper-view[view="${v}"]`) as HTMLElement & { debugState(): unknown }).debugState(), view) as Promise<Dbg>;
export const panel = (page: Page) => page.locator('caliper-controls');
export const panelState = (page: Page) => page.evaluate(() =>
  (document.querySelector('caliper-controls') as HTMLElement & { debugState(): unknown }).debugState()) as Promise<{
  access: { shown: boolean; allowed: boolean; reason: string } | null;
  controls: { run: { state: string; step?: number }; params: { name: string; value: unknown; pending?: unknown }[]; actions: { name: string; running: boolean; queued: boolean }[] } | null;
}>;
export const param = async (page: Page, name: string) => (await panelState(page)).controls?.params.find((p) => p.name === name);

export async function open(page: Page, url: string, views: number) {
  await page.setViewportSize({ width: 1280, height: 1400 });
  await page.goto(url);
  await connected(page, views);
}

export async function connected(page: Page, views: number) {
  await page.waitForFunction((n) => {
    const els = [...document.querySelectorAll('caliper-view')] as (HTMLElement & { debugState(): Record<string, unknown> })[];
    return els.length === n && els.every((e) => e.debugState().status === 'connected');
  }, views, { timeout: 90_000 });
}

/**
 * A second page in its own browser context: a <caliper-controls> panel and
 * the named views drawn by the element's own views (renderer="ts"), linked,
 * on the same stream. (SwiftShader in headless Chromium hands a second
 * context no WebGPU devices for six more WebAssembly views.)
 */
export async function second(browser: Browser, url: string, views: string[]) {
  const ctx = await browser.newContext({ viewport: { width: 1000, height: 900 } });
  const page = await ctx.newPage();
  const u = new URL(url);
  const ws = `ws://${u.host}/v3/stream?token=${u.searchParams.get('token')}`;
  await page.setContent(`<!doctype html><body style="background:#181715">
    <caliper-controls src="${ws}"></caliper-controls>
    ${views.map((v) => `<caliper-view view="${v}" src="${ws}" renderer="ts" link details height="260"></caliper-view>`).join('')}
    <script type="module">import '${u.origin}/caliper-view.js';</script></body>`);
  await connected(page, views.length);
  return { page, close: () => ctx.close() };
}
