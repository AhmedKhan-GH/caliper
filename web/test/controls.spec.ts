// Wire protocol 3.1 in the browser, against mock_server.py --controls: the run
// controls in the element header and the <caliper-controls> panel, live
// parameters and actions, who may control, and a 3.0 producer. On every
// project: with WebGPU the views are Caliper's WebAssembly views, in Canvas 2D
// the element's own; the controls must look and behave the same. In CI it is
// a protocol spec and runs on canvas2d only (playwright.config.ts); linked
// selection, which the views draw, is in linked.spec.ts.

import { expect, test } from '@playwright/test';
import { MSG } from '../src/protocol/messages.js';
import { open, panel, panelState, param, second, state } from './pages.js';
import { HAVE_PYTHON, startProducer, type Producer } from './producer.js';
import { drawn, roundTrip, shotPath, snapshotsPass } from './util.js';

test.skip(!HAVE_PYTHON, 'needs a Python with numpy and websockets (CALIPER_PYTHON)');

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
      await roundTrip(panel(page)).toBeVisible();
      await roundTrip.poll(async () => (await state(page, 'loss')).run).toBe('running');
      const run = panel(page).locator('.run');
      await roundTrip(run.locator('.badge')).toHaveText('running');
      await run.locator('.run-toggle').click();
      await roundTrip.poll(async () => (await panelState(page)).controls?.run.state).toBe('paused');
      const at = (await panelState(page)).controls!.run.step!;
      await roundTrip(run.locator('.badge')).toHaveText(`paused at ${at}`);
      await roundTrip(run.locator('.run-toggle')).toHaveText('Resume');
      // The other page and every header agree.
      await roundTrip.poll(async () => (await state(other.page, 'embedding')).run).toBe(`paused at ${at}`);
      await roundTrip.poll(async () => (await state(page, 'attention')).run).toBe(`paused at ${at}`);
      // Held: the step stays while the other page receives five more snapshots,
      // one each 50 ms, the time of a training step.
      await snapshotsPass(other.page, 'caliper-view[view="embedding"]', 5);
      expect((await panelState(page)).controls!.run.step).toBe(at);
      await drawn(page);
      await page.screenshot({ path: shotPath(info, 'controls-paused'), fullPage: false });
      await run.locator('.run-step').click();
      await roundTrip.poll(async () => (await panelState(page)).controls?.run.step).toBe(at + 1);
      expect((await panelState(page)).controls!.run.state).toBe('paused');
      // The element's methods do the same, and resolve with the reply.
      const reply = await page.evaluate(() => (document.querySelector('caliper-view[view="loss"]') as HTMLElement & { step(n: number): Promise<unknown> }).step(3));
      expect(reply).toMatchObject({ type: 'control_reply', op: 'step', status: 'ok' });
      await roundTrip.poll(async () => (await panelState(page)).controls?.run.step).toBe(at + 4);
      await other.page.locator('caliper-controls .run-toggle').click();
      await roundTrip.poll(async () => (await panelState(page)).controls?.run.state).toBe('running');
      await roundTrip.poll(async () => (await state(page, 'loss')).run).toBe('running');
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
    await roundTrip(loss.locator('.run-toggle')).toHaveText('Pause');
    await loss.locator('.run-toggle').click();
    await roundTrip.poll(async () => (await state(page, 'loss')).run ?? '').toMatch(/^paused at \d+$/);
    await roundTrip(loss.locator('.run-toggle')).toHaveText('Resume');
    await loss.locator('.run-toggle').click();
    await roundTrip.poll(async () => (await state(page, 'loss')).run).toBe('running');
  });

  test('parameters show as a slider, a field, a toggle and a menu; a change shows pending, then applied, on every page', async ({ page, browser }, info) => {
    prod = await startProducer(['--controls', '--step-ms', '3000']);
    await open(page, prod.url, 6);
    const other = await second(browser, prod.url, ['embedding']);
    try {
      const p = panel(page);
      await roundTrip(p.locator('input[type="range"][data-param="lr"]')).toBeVisible();
      await roundTrip(p.locator('input[type="number"][data-param="lr"]')).toHaveValue('0.001');
      await roundTrip(p.locator('input[type="range"][data-param="batch"]')).toBeVisible();
      await roundTrip(p.locator('button[data-param="augment"]')).toHaveText('on');
      await roundTrip(p.locator('select[data-param="optimizer"]')).toHaveValue('adam');
      // Running, a change waits for the script's next checkpoint (a step takes
      // 3 s here): both pages show it pending. Set right after a step.
      const steps = (await state(page, 'loss')).info;
      await roundTrip.poll(async () => (await state(page, 'loss')).info).not.toBe(steps);
      await p.locator('input[type="number"][data-param="lr"]').fill('0.02');
      await p.locator('input[type="number"][data-param="lr"]').press('Enter');
      await roundTrip.poll(async () => (await param(page, 'lr'))?.pending).toBe(0.02);
      await roundTrip.poll(async () => (await param(other.page, 'lr'))?.pending).toBe(0.02);
      await roundTrip(other.page.locator('caliper-controls input[type="number"][data-param="lr"]')).toHaveValue('0.02');
      await roundTrip(other.page.locator('caliper-controls .note').first()).toHaveText('at the next step');
      await page.screenshot({ path: shotPath(info, 'controls-pending') });
      // The next checkpoint applies it.
      await roundTrip.poll(async () => (await param(other.page, 'lr'))?.value).toBe(0.02);
      expect((await param(other.page, 'lr'))?.pending).toBeUndefined();
      await roundTrip(other.page.locator('caliper-controls .note').first()).toHaveText('');
      // Paused, a change applies in the checkpoint the run waits in. The other
      // controls, from the other page; a value out of range is clamped.
      await p.locator('.run-toggle').click();
      await roundTrip.poll(async () => (await panelState(page)).controls?.run.state).toBe('paused');
      const op = panel(other.page);
      const reply = await other.page.evaluate(() => (document.querySelector('caliper-controls') as HTMLElement & { setParam(n: string, v: unknown): Promise<unknown> }).setParam('batch', 1000));
      expect(reply).toMatchObject({ status: 'ok', name: 'batch', value: 512 });
      await op.locator('button[data-param="augment"]').click();
      await op.locator('select[data-param="optimizer"]').selectOption('sgd');
      await roundTrip.poll(async () => [(await param(page, 'batch'))?.value, (await param(page, 'augment'))?.value, (await param(page, 'optimizer'))?.value])
        .toEqual([512, false, 'sgd']);
      await roundTrip(p.locator('button[data-param="augment"]')).toHaveText('off');
      await roundTrip(p.locator('select[data-param="optimizer"]')).toHaveValue('sgd');
      await roundTrip(p.locator('input[type="number"][data-param="batch"]')).toHaveValue('512');
      // Dragging the slider sends at most 30 sets a second and ends with the value where it stops.
      const slider = p.locator('input[type="range"][data-param="lr"]');
      const box = (await slider.boundingBox())!;
      const sent = prod.controls.length;
      await page.mouse.move(box.x + 4, box.y + box.height / 2);
      await page.mouse.down();
      for (let k = 1; k <= 20; k++) await page.mouse.move(box.x + 4 + (box.width - 8) * k / 20, box.y + box.height / 2);
      await page.mouse.up();
      await roundTrip.poll(async () => (await param(page, 'lr'))?.value).toBeCloseTo(0.1, 6);
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
    // The page records each state of the button and its note as it changes.
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
    await roundTrip(panel(page).locator('.action .note')).toHaveText(/^saved step \d+$/);
    await roundTrip(btn).toHaveAttribute('aria-busy', 'false');
    const got = await replies.evaluate((r) => r.map((x) => (x as { status: string }).status));
    expect(got).toEqual(['queued', 'ok']);
    // The drawn states come in this order. A transient one may be missed on a
    // busy machine (the run can start at the checkpoint right after it was
    // queued, before CONTROLS says queued), but one out of order is a failure.
    const order = ['false ', 'true queued', 'true running', 'false saved step N'];
    const drew = await seen.evaluate((r) => r.map((s) => s.replace(/\d+$/, 'N')));
    const at = drew.map((d) => order.indexOf(d));
    expect(at.every((k, i) => k >= 0 && (i === 0 || k > at[i - 1])), drew.join(', ')).toBe(true);
    expect(drew.at(-1)).toBe('false saved step N');
  });

  test('a viewer the producer refuses shows the controls but cannot use them', async ({ page }, info) => {
    prod = await startProducer(['--controls', '--read-only']);
    await open(page, prod.url, 6);
    await roundTrip(panel(page)).toBeVisible();
    const s = await panelState(page);
    expect(s.access).toEqual({ shown: true, allowed: false, reason: 'This viewer may not control the run' });
    for (const sel of ['.run-toggle', '.run-step', 'input[data-param="lr"]', 'button[data-param="augment"]', 'select[data-param="optimizer"]', 'button[data-action="save checkpoint"]']) {
      await roundTrip(panel(page).locator(sel).first(), sel).toBeDisabled();
    }
    await roundTrip(panel(page).locator('.run-toggle')).toHaveAttribute('title', 'This viewer may not control the run');
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
    await roundTrip(p.locator('.run-toggle')).toBeDisabled();
    await roundTrip(p.locator('.run-toggle')).toHaveAttribute('title', 'The script has no checkpoint');
    expect((await state(page, 'loss')).run).toBeNull();   // no badge for the state none
    await p.locator('select[data-param="optimizer"]').selectOption('lion');
    await roundTrip.poll(async () => (await param(page, 'optimizer'))?.value).toBe('lion');
    await p.locator('button[data-action="save checkpoint"]').click();
    await roundTrip(p.locator('.action .note')).toHaveText('waits for a checkpoint');
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
