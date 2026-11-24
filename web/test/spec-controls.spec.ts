import { test, expect } from '@playwright/test';
import { mockApp, openSpec, ready, type Json, type MockApp } from './spec-mock.js';

test.beforeEach(({}, info) => test.skip(info.project.name !== 'swiftshader', 'Canvas 2D needs one project'));

test('spec buttons draw plain labels and retain the viewer restriction', async ({ page }, info) => {
  const controls = [
    { id: 'clamp', kind: 'button', title: 'Apply Clamp', bind: 'action:apply_clamp' },
    { id: 'reset', kind: 'button', title: 'Feature Reset', bind: 'action:reset' },
    { id: 'generate', kind: 'button', title: 'Token Generation', bind: 'action:generate' },
    { id: 'pause', kind: 'button', title: 'Pause', bind: 'run:pause' },
  ];
  const why = 'The browser viewer only reads: use the app to change this control.';
  const data: Record<string, Json> = Object.fromEntries(controls.map((c) => [c.id, ready(c.id, 'control', {
    ...c, value: null, text: c.title, enabled: false, why,
  })]));
  const app: MockApp = {
    specs: { buttons: {
      caliper_spec: 1, applet: { id: 'buttons', title: 'Button Controls' },
      control: controls, layout: { split: 'vertical', children: controls.map((c) => c.id) },
    } },
    data: { buttons: data }, requests: [],
  };
  // Record the text that reaches the canvas, without a replacement renderer.
  await page.addInitScript(() => {
    const labels = new Set<string>();
    (window as any).drawnLabels = labels;
    const fillText = CanvasRenderingContext2D.prototype.fillText;
    CanvasRenderingContext2D.prototype.fillText = function (text, x, y, maxWidth) {
      if (this.canvas.id === 'canvas') labels.add(text);
      if (maxWidth === undefined) fillText.call(this, text, x, y);
      else fillText.call(this, text, x, y, maxWidth);
    };
  });
  await mockApp(page, app);
  await openSpec(page, 'buttons');
  await page.waitForFunction(() => {
    const controls = (window as any).caliperSpec?.debugState().controls;
    return controls?.length === 4 && controls.every((c: Json) => c.text !== '');
  });
  await page.evaluate(() => {
    (window as any).drawnLabels.clear();
    return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  });
  await expect.poll(() => page.evaluate(() => [...(window as any).drawnLabels].sort())).toEqual(
    controls.map((c) => c.title).sort(),
  );
  const state = await page.evaluate(() => (window as any).caliperSpec.debugState());
  expect(state.controls.map((c: Json) => c.enabled)).toEqual(controls.map(() => false));
  const area = await page.locator('#area').boundingBox();
  expect(area).not.toBeNull();
  const last = state.controls.at(-1).rect;
  await page.screenshot({ path: info.outputPath('plain-buttons.png'),
    clip: { x: area!.x, y: area!.y, width: area!.width, height: last.y + last.h } });
  const row = state.controls.find((c: Json) => c.id === 'clamp').rect;
  await page.mouse.click(area!.x + row.x + row.w / 2, area!.y + row.y + row.h / 2);
  await expect(page.locator('#canvas')).toHaveAttribute('title', why);
  expect([...new Set(app.requests.map((r) => r.name))].sort()).toEqual(['panel.data', 'spec.get']);

  // A later state changes the transport label without a binding suffix.
  data.pause = ready('pause', 'control', {
    ...controls[3], value: null, text: 'Resume', enabled: false, why,
  }, 'paused');
  await expect.poll(() => page.evaluate(() => (window as any).drawnLabels.has('Resume'))).toBe(true);
  expect(await page.evaluate(() => [...(window as any).drawnLabels].sort())).toEqual(
    [...controls.map((c) => c.title), 'Resume'].sort(),
  );
});
