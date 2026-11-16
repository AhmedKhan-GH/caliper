// Wire protocol 3.1 in the browser, against mock_server.py --controls: the run
// controls in the element header and the <caliper-controls> panel, live
// parameters and actions, who may control, a 3.0 producer, and linked
// selection with item details. On every project: with WebGPU the views are
// Caliper's WebAssembly views, in Canvas 2D the element's own; the controls
// and the highlights must look and behave the same.

import { expect, test, type Browser, type Page } from '@playwright/test';
import { MSG } from '../src/protocol/messages.js';
import { HAVE_PYTHON, startProducer, type Producer } from './producer.js';
import { drawn, shotPath, snapshotsPass } from './util.js';

test.skip(!HAVE_PYTHON, 'needs a Python with numpy and websockets (CALIPER_PYTHON)');

type Dbg = Record<string, unknown> & {
  status: string; area: { x: number; y: number; w: number; h: number }; window: { x0: number; x1: number; y0: number; y1: number };
  linked: { space: string; axis: string; items: number[] } | null; details: string | null; run: string | null; selected?: number;
};
const state = (page: Page, view: string) => page.evaluate((v) =>
  (document.querySelector(`caliper-view[view="${v}"]`) as HTMLElement & { debugState(): unknown }).debugState(), view) as Promise<Dbg>;
const panel = (page: Page) => page.locator('caliper-controls');
const panelState = (page: Page) => page.evaluate(() =>
  (document.querySelector('caliper-controls') as HTMLElement & { debugState(): unknown }).debugState()) as Promise<{
  access: { shown: boolean; allowed: boolean; reason: string } | null;
  controls: { run: { state: string; step?: number }; params: { name: string; value: unknown; pending?: unknown }[]; actions: { name: string; running: boolean; queued: boolean }[] } | null;
}>;
const param = async (page: Page, name: string) => (await panelState(page)).controls?.params.find((p) => p.name === name);

async function open(page: Page, url: string, views: number) {
  await page.setViewportSize({ width: 1280, height: 1400 });
  await page.goto(url);
  await connected(page, views);
}

async function connected(page: Page, views: number) {
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
async function second(browser: Browser, url: string, views: string[]) {
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

// Two pages of WebAssembly views on the software adapter are slow to start
// while other tests run.
test.beforeEach(({}, info) => test.slow(info.project.name === 'swiftshader'));

test.describe('run controls and the controls panel', () => {
  let prod: Producer;
  test.afterEach(() => prod?.stop());

  test('pause, step and resume from the panel reach every page, and every header shows the run state', async ({ page, browser }, info) => {
    prod = await startProducer(['--controls']);
    await open(page, prod.url, 6);
    const other = await second(browser, prod.url, ['embedding']);
    try {
      await expect(panel(page)).toBeVisible();
      await expect.poll(async () => (await state(page, 'loss')).run).toBe('running');
      const run = panel(page).locator('.run');
      await expect(run.locator('.badge')).toHaveText('running');
      await run.locator('.run-toggle').click();
      await expect.poll(async () => (await panelState(page)).controls?.run.state).toBe('paused');
      const at = (await panelState(page)).controls!.run.step!;
      await expect(run.locator('.badge')).toHaveText(`paused at ${at}`);
      await expect(run.locator('.run-toggle')).toHaveText('Resume');
      // The other page and every header agree.
      await expect.poll(async () => (await state(other.page, 'embedding')).run).toBe(`paused at ${at}`);
      await expect.poll(async () => (await state(page, 'attention')).run).toBe(`paused at ${at}`);
      // Held: the step stays while the other page receives five more snapshots,
      // one each 50 ms, the time of a training step.
      await snapshotsPass(other.page, 'caliper-view[view="embedding"]', 5);
      expect((await panelState(page)).controls!.run.step).toBe(at);
      await drawn(page);
      await page.screenshot({ path: shotPath(info, 'controls-paused'), fullPage: false });
      await run.locator('.run-step').click();
      await expect.poll(async () => (await panelState(page)).controls?.run.step).toBe(at + 1);
      expect((await panelState(page)).controls!.run.state).toBe('paused');
      // The element's methods do the same, and resolve with the reply.
      const reply = await page.evaluate(() => (document.querySelector('caliper-view[view="loss"]') as HTMLElement & { step(n: number): Promise<unknown> }).step(3));
      expect(reply).toMatchObject({ type: 'control_reply', op: 'step', status: 'ok' });
      await expect.poll(async () => (await panelState(page)).controls?.run.step, { timeout: 10_000 }).toBe(at + 4);
      await other.page.locator('caliper-controls .run-toggle').click();
      await expect.poll(async () => (await panelState(page)).controls?.run.state).toBe('running');
      await expect.poll(async () => (await state(page, 'loss')).run).toBe('running');
      expect(prod.controls.map((c) => c.op)).toEqual(['pause', 'step', 'step', 'resume']);
    } finally {
      await other.close();
    }
  });

  test('a header with run-controls pauses and resumes too', async ({ page }) => {
    prod = await startProducer(['--controls']);
    await open(page, prod.url, 6);
    const loss = page.locator('caliper-view[view="loss"]');
    await expect(loss.locator('.run-toggle')).toBeHidden();
    await loss.evaluate((el) => el.setAttribute('run-controls', ''));
    await expect(loss.locator('.run-toggle')).toHaveText('Pause');
    await loss.locator('.run-toggle').click();
    await expect.poll(async () => (await state(page, 'loss')).run ?? '').toMatch(/^paused at \d+$/);
    await expect(loss.locator('.run-toggle')).toHaveText('Resume');
    await loss.locator('.run-toggle').click();
    await expect.poll(async () => (await state(page, 'loss')).run).toBe('running');
  });

  test('parameters show as a slider, a field, a toggle and a menu; a change shows pending, then applied, on every page', async ({ page, browser }, info) => {
    prod = await startProducer(['--controls', '--step-ms', '3000']);
    await open(page, prod.url, 6);
    const other = await second(browser, prod.url, ['embedding']);
    try {
      const p = panel(page);
      await expect(p.locator('input[type="range"][data-param="lr"]')).toBeVisible();
      await expect(p.locator('input[type="number"][data-param="lr"]')).toHaveValue('0.001');
      await expect(p.locator('input[type="range"][data-param="batch"]')).toBeVisible();
      await expect(p.locator('button[data-param="augment"]')).toHaveText('on');
      await expect(p.locator('select[data-param="optimizer"]')).toHaveValue('adam');
      // Running, a change waits for the script's next checkpoint (a step takes
      // 3 s here): both pages show it pending. Set right after a step.
      const steps = (await state(page, 'loss')).info;
      await expect.poll(async () => (await state(page, 'loss')).info, { timeout: 10_000 }).not.toBe(steps);
      await p.locator('input[type="number"][data-param="lr"]').fill('0.02');
      await p.locator('input[type="number"][data-param="lr"]').press('Enter');
      await expect.poll(async () => (await param(page, 'lr'))?.pending).toBe(0.02);
      await expect.poll(async () => (await param(other.page, 'lr'))?.pending).toBe(0.02);
      await expect(other.page.locator('caliper-controls input[type="number"][data-param="lr"]')).toHaveValue('0.02');
      await expect(other.page.locator('caliper-controls .note').first()).toHaveText('at the next step');
      await page.screenshot({ path: shotPath(info, 'controls-pending') });
      // The next checkpoint applies it.
      await expect.poll(async () => (await param(other.page, 'lr'))?.value, { timeout: 10_000 }).toBe(0.02);
      expect((await param(other.page, 'lr'))?.pending).toBeUndefined();
      await expect(other.page.locator('caliper-controls .note').first()).toHaveText('');
      // Paused, a change applies in the checkpoint the run waits in. The other
      // controls, from the other page; a value out of range is clamped.
      await p.locator('.run-toggle').click();
      await expect.poll(async () => (await panelState(page)).controls?.run.state, { timeout: 10_000 }).toBe('paused');
      const op = panel(other.page);
      const reply = await other.page.evaluate(() => (document.querySelector('caliper-controls') as HTMLElement & { setParam(n: string, v: unknown): Promise<unknown> }).setParam('batch', 1000));
      expect(reply).toMatchObject({ status: 'ok', name: 'batch', value: 512 });
      await op.locator('button[data-param="augment"]').click();
      await op.locator('select[data-param="optimizer"]').selectOption('sgd');
      await expect.poll(async () => [(await param(page, 'batch'))?.value, (await param(page, 'augment'))?.value, (await param(page, 'optimizer'))?.value])
        .toEqual([512, false, 'sgd']);
      await expect(p.locator('button[data-param="augment"]')).toHaveText('off');
      await expect(p.locator('select[data-param="optimizer"]')).toHaveValue('sgd');
      await expect(p.locator('input[type="number"][data-param="batch"]')).toHaveValue('512');
      // Dragging the slider sends at most 30 sets a second and ends with the value where it stops.
      const slider = p.locator('input[type="range"][data-param="lr"]');
      const box = (await slider.boundingBox())!;
      const sent = prod.controls.length;
      await page.mouse.move(box.x + 4, box.y + box.height / 2);
      await page.mouse.down();
      for (let k = 1; k <= 20; k++) await page.mouse.move(box.x + 4 + (box.width - 8) * k / 20, box.y + box.height / 2);
      await page.mouse.up();
      await expect.poll(async () => (await param(page, 'lr'))?.value).toBeCloseTo(0.1, 6);
      const sets = prod.controls.slice(sent).filter((c) => c.op === 'set' && c.name === 'lr');
      expect(sets.at(-1)?.value).toBeCloseTo(0.1, 6);
      await page.screenshot({ path: shotPath(info, 'controls-applied') });
    } finally {
      await other.close();
    }
  });

  test('an action shows queued and running, then its result to the page that ran it', async ({ page }) => {
    prod = await startProducer(['--controls']);
    await open(page, prod.url, 6);
    const replies = await page.evaluateHandle(() => {
      const r: unknown[] = [];
      document.querySelector('caliper-controls')!.addEventListener('caliper-reply', (e) => r.push((e as CustomEvent).detail));
      return r;
    });
    // Queued and running last one training step and 0.2 s, which can be over
    // before the click returns on a busy machine: the page records each state
    // of the button and its note as it changes.
    const seen = await page.evaluateHandle(() => {
      const root = document.querySelector('caliper-controls')!.shadowRoot!;
      const now = () => {
        const b = root.querySelector('button[data-action="save checkpoint"]')!;
        return `${b.getAttribute('aria-busy')} ${b.parentElement!.querySelector('.note')!.textContent}`;
      };
      const r = [now()];
      new MutationObserver(() => { if (r.at(-1) !== now()) r.push(now()); })
        .observe(root, { subtree: true, childList: true, characterData: true, attributes: true });
      return r;
    });
    const btn = panel(page).locator('button[data-action="save checkpoint"]');
    await btn.click();
    await expect(panel(page).locator('.action .note')).toHaveText(/^saved step \d+$/, { timeout: 10_000 });
    await expect(btn).toHaveAttribute('aria-busy', 'false');
    const states = await seen.evaluate((r) => r.map((s) => s.replace(/\d+$/, 'N')));
    expect(states).toEqual(['false ', 'true queued', 'true running', 'false saved step N']);
    const got = await replies.evaluate((r) => r.map((x) => (x as { status: string }).status));
    expect(got).toEqual(['queued', 'ok']);
  });

  test('a viewer the producer refuses shows the controls but cannot use them', async ({ page }, info) => {
    prod = await startProducer(['--controls', '--read-only']);
    await open(page, prod.url, 6);
    await expect(panel(page)).toBeVisible();
    const s = await panelState(page);
    expect(s.access).toEqual({ shown: true, allowed: false, reason: 'This viewer may not control the run' });
    for (const sel of ['.run-toggle', '.run-step', 'input[data-param="lr"]', 'button[data-param="augment"]', 'select[data-param="optimizer"]', 'button[data-action="save checkpoint"]']) {
      await expect(panel(page).locator(sel).first(), sel).toBeDisabled();
    }
    await expect(panel(page).locator('.run-toggle')).toHaveAttribute('title', 'This viewer may not control the run');
    // The calls are refused in the page: they resolve with null and no CONTROL
    // frame leaves it (a frame's type is its bytes 8 and 9), so none reaches
    // the producer either.
    const sent = await page.evaluateHandle(() => {
      const types: number[] = [];
      const send = WebSocket.prototype.send;
      WebSocket.prototype.send = function (d) {
        if (ArrayBuffer.isView(d)) types.push(new DataView(d.buffer, d.byteOffset, d.byteLength).getUint16(8, true));
        else if (d instanceof ArrayBuffer) types.push(new DataView(d).getUint16(8, true));
        return send.call(this, d);
      };
      return types;
    });
    const r = await page.evaluate(() => Promise.all([
      (document.querySelector('caliper-controls') as HTMLElement & { setParam(n: string, v: unknown): Promise<unknown> }).setParam('lr', 0.01),
      (document.querySelector('caliper-view') as HTMLElement & { pause(): Promise<unknown> }).pause(),
    ]));
    expect(r).toEqual([null, null]);
    expect(await sent.evaluate((t, control) => t.filter((x) => x === control).length, MSG.control)).toBe(0);
    expect(prod.controls).toEqual([]);
    await drawn(page);
    await page.screenshot({ path: shotPath(info, 'controls-read-only') });
  });

  test('without a checkpoint the run controls wait for one, and parameters apply at once', async ({ page }) => {
    prod = await startProducer(['--controls', '--no-checkpoint']);
    await open(page, prod.url, 6);
    const p = panel(page);
    await expect(p.locator('.run-toggle')).toBeDisabled();
    await expect(p.locator('.run-toggle')).toHaveAttribute('title', 'The script has no checkpoint');
    expect((await state(page, 'loss')).run).toBeNull();   // no badge for the state none
    await p.locator('select[data-param="optimizer"]').selectOption('lion');
    await expect.poll(async () => (await param(page, 'optimizer'))?.value).toBe('lion');
    await p.locator('button[data-action="save checkpoint"]').click();
    await expect(p.locator('.action .note')).toHaveText('waits for a checkpoint');
  });

  test('a 3.0 producer still works and shows no controls', async ({ page }) => {
    prod = await startProducer(['--v30']);
    await open(page, prod.url, 6);
    await expect(panel(page)).toBeHidden();
    const loss = page.locator('caliper-view[view="loss"]');
    await loss.evaluate((el) => el.setAttribute('run-controls', ''));
    // The stream goes on (two more snapshots) and still no run bar shows.
    await snapshotsPass(page, 'caliper-view[view="embedding"]', 2);
    await expect(loss.locator('.run')).toBeHidden();
    expect((await state(page, 'loss')).run).toBeNull();
    expect((await state(page, 'embedding')).status).toBe('connected');
  });
});

test.describe('linked selection', () => {
  let prod: Producer;
  test.afterEach(() => prod?.stop());

  /** Page coordinates of data point (x, y) of a view, from its debug area and window. */
  async function at(page: Page, view: string, x: number, y: number) {
    const d = await state(page, view);
    const body = (await page.locator(`caliper-view[view="${view}"] .body`).boundingBox())!;
    const yDown = view === 'h' || view === 'ha';
    const fx = (x - d.window.x0) / (d.window.x1 - d.window.x0), fy = (y - d.window.y0) / (d.window.y1 - d.window.y0);
    return { x: body.x + d.area.x + fx * d.area.w, y: body.y + d.area.y + (yDown ? fy * d.area.h : d.area.h - fy * d.area.h) };
  }

  async function shiftDrag(page: Page, from: { x: number; y: number }, to: { x: number; y: number }, alt = false) {
    await page.keyboard.down('Shift');
    if (alt) await page.keyboard.down('Alt');
    await page.mouse.move(from.x, from.y);
    await page.mouse.down();
    await page.mouse.move(to.x, to.y, { steps: 6 });
    await page.mouse.up();
    if (alt) await page.keyboard.up('Alt');
    await page.keyboard.up('Shift');
  }

  test('a selection of points highlights the same items in the heatmap and on another page, with the details under the view', async ({ page, browser }, info) => {
    prod = await startProducer(['--controls', '--fixed']);
    await open(page, prod.url, 5);
    const other = await second(browser, prod.url, ['p', 'h']);
    try {
      await expect.poll(async () => (await state(page, 'p')).window?.x1 ?? null).not.toBeNull();
      const events = await page.evaluateHandle(() => {
        const r: unknown[] = [];
        document.addEventListener('caliper-selection', (e) => r.push((e as CustomEvent).detail));
        return r;
      });
      // A rectangle around the right-hand points (rows 3 and 9 of the space "test").
      await shiftDrag(page, await at(page, 'p', 0.8, -0.02), await at(page, 'p', 1.2, 1.02), true);
      await expect.poll(() => prod.events.filter((e) => e.type === 'select').at(-1)?.detail ?? null)
        .toEqual({ indices: [1, 3], row_ids: [3, 9], space: 'test', items: [3, 9] });
      // The producer's SELECTION comes back to every view that maps the space, on both pages.
      const want = { space: 'test', axis: 'points', items: [3, 9] };
      await expect.poll(async () => (await state(page, 'p')).linked).toEqual(want);
      await expect.poll(async () => (await state(page, 'h')).linked).toEqual({ space: 'test', axis: 'rows', items: [3, 9] });
      await expect.poll(async () => (await state(other.page, 'h')).linked).toEqual({ space: 'test', axis: 'rows', items: [3, 9] });
      await expect.poll(async () => (await state(other.page, 'p')).selected).toBe(2);
      expect((await state(page, 'ha')).linked).toBeNull();   // its rows are not in a space
      // The details of the items under the view the selection was made in.
      await expect.poll(async () => (await state(page, 'p')).details).toBe('itemlabelscore330.005910.044');
      expect((await state(page, 'h')).details).toBeNull();
      expect((await events.evaluate((r) => r.length))).toBeGreaterThanOrEqual(2);
      await drawn(page, 'caliper-view[view="p"], caliper-view[view="h"]');
      await page.locator('caliper-view[view="p"]').screenshot({ path: shotPath(info, 'linked-points') });
      await page.locator('caliper-view[view="h"]').screenshot({ path: shotPath(info, 'linked-heatmap') });
      // A shift-click clears it everywhere.
      const c = await at(page, 'p', 0.5, 0.5);
      await shiftDrag(page, c, c);
      await expect.poll(async () => (await state(other.page, 'h')).linked?.items ?? null).toEqual([]);
      await expect.poll(async () => (await state(page, 'p')).details).toBeNull();
    } finally {
      await other.close();
    }
  });

  test('a range of steps puts markers on the series', async ({ page }, info) => {
    prod = await startProducer(['--controls', '--fixed']);
    await open(page, prod.url, 5);
    await expect.poll(async () => (await state(page, 's')).window?.x1 ?? null).not.toBeNull();
    const from = await at(page, 's', 300.2, 500), to = await at(page, 's', 340.8, 500);
    await shiftDrag(page, { x: Math.ceil(from.x), y: from.y }, { x: Math.floor(to.x), y: to.y });
    await expect.poll(async () => (await state(page, 's')).linked?.axis ?? null).toBe('steps');
    const l = (await state(page, 's')).linked!;
    expect(l.space).toBe('step');
    expect(l.items[0]).toBeGreaterThanOrEqual(300);
    expect(l.items[0]).toBeLessThanOrEqual(303);
    expect(l.items.at(-1)).toBeGreaterThanOrEqual(338);
    expect(l.items.at(-1)).toBeLessThanOrEqual(341);
    await page.mouse.move(10, 10);
    await drawn(page, 'caliper-view[view="s"]');
    await page.locator('caliper-view[view="s"]').screenshot({ path: shotPath(info, 'linked-series') });
  });
});
