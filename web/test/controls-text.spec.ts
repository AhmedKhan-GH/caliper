import { expect, test } from '@playwright/test';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

declare global {
  interface Window { TextControlsFixture: typeof import('./controls-text.fixture.js') }
}

// One controls element, no network, producer, model, renderer, or WebGPU.
test.use({ headless: true });
test.setTimeout(20_000);
test.beforeEach(async ({ page }) => {
  const bundle = await build({
    entryPoints: [fileURLToPath(new URL('./controls-text.fixture.ts', import.meta.url))],
    bundle: true, format: 'iife', globalName: 'TextControlsFixture', write: false,
  });
  await page.setContent('<button id="outside">Outside</button>');
  await page.addScriptTag({ content: bundle.outputFiles[0].text });
});

test('Text Draft Recovery Across Background Frames', async ({ page }) => {
  const fixture = await page.evaluateHandle(() => window.TextControlsFixture.mountTextControls());
  const field = page.locator('caliper-controls textarea');
  const invalid = 'é'.repeat(2049);
  await field.fill(invalid);
  await page.locator('#outside').click();
  await fixture.evaluate(({ source }) => { source.publish('background'); source.status(); });
  await expect(field).toHaveValue(invalid);
  expect(await field.evaluate((el: HTMLTextAreaElement) => el.validity.valid)).toBe(false);
  expect(await fixture.evaluate(({ source }) => source.calls.length)).toBe(0);
  // A change in another row also preserves the draft.
  await fixture.evaluate(({ source }) => source.publish('background', true));
  await expect(field).toHaveValue(invalid);
  await field.fill('é'.repeat(2048));
  await field.press('Enter');
  await page.locator('#outside').click();
  expect(await fixture.evaluate(({ source }) => source.calls.map((c) => c.request.value))).toEqual(['é'.repeat(2048)]);
  await fixture.evaluate(({ source }) => { source.respond(0, 'ok'); source.publish('é'.repeat(2048), true); });
  expect(await field.evaluate((el: HTMLTextAreaElement) => el.validity.valid)).toBe(true);
  await fixture.evaluate(({ source }) => source.publish('remote', true));
  await expect(field).toHaveValue('remote');
});

test('Text Return, Focus Loss, Multiline, and Refusal', async ({ page }) => {
  const fixture = await page.evaluateHandle(() => window.TextControlsFixture.mountTextControls());
  const field = page.locator('caliper-controls textarea');
  await field.fill('first');
  await field.press('Shift+Enter');
  await field.press('End');
  await field.press('x');
  await expect(field).toHaveValue('first\nx');
  expect(await fixture.evaluate(({ source }) => source.calls.length)).toBe(0);
  await fixture.evaluate(({ source }) => { source.publish('background'); source.status(); });
  await expect(field).toHaveValue('first\nx');
  await field.press('Enter');
  await field.press('Enter');
  await page.locator('#outside').click();
  expect(await fixture.evaluate(({ source }) => source.calls.map((c) => c.request.value))).toEqual(['first\nx']);
  await fixture.evaluate(({ source }) => { source.respond(0, 'error'); source.publish('background'); });
  await expect(field).toHaveValue('first\nx');
  await expect(page.locator('caliper-controls .note').first()).toHaveText('Try a shorter prompt');
  await field.fill('revised');
  await page.locator('#outside').click();
  expect(await fixture.evaluate(({ source }) => source.calls.map((c) => c.request.value))).toEqual(['first\nx', 'revised']);
  await fixture.evaluate(({ source }) => { source.respond(1, 'ok'); source.publish('revised'); });
  await field.fill('');
  await field.press('Enter');
  await page.locator('#outside').click();
  expect(await fixture.evaluate(({ source }) => source.calls.map((c) => c.request.value))).toEqual(['first\nx', 'revised', '']);
});

test('Text Composition and Session Reset', async ({ page }) => {
  const fixture = await page.evaluateHandle(() => window.TextControlsFixture.mountTextControls());
  const field = page.locator('caliper-controls textarea');
  await field.fill('composed');
  expect(await field.evaluate((el) => el.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Enter', isComposing: true, bubbles: true, cancelable: true,
  })))).toBe(true);
  expect(await fixture.evaluate(({ source }) => source.calls.length)).toBe(0);
  await field.fill('😀'.repeat(1025));
  await page.locator('#outside').click();
  await fixture.evaluate(({ source }) => { source.drop(); source.publish('new session'); });
  await expect(field).toHaveValue('new session');
});
