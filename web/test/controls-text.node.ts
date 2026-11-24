import assert from 'node:assert/strict';
import test, { after } from 'node:test';

// Minimal DOM events for CPU checks. The browser spec checks native focus,
// textarea edits, and validity on the same controls fixture.
class ElementStub extends EventTarget {
  readonly children: ElementStub[] = [];
  readonly dataset: Record<string, string> = {};
  readonly style: Record<string, string> = {};
  readonly attributes = new Map<string, string>();
  shadowRoot: ElementStub | null = null;
  activeElement: ElementStub | null = null;
  isConnected = false;
  value = '';
  validationMessage = '';
  disabled = false;
  constructor(readonly tagName = 'caliper-controls') { super(); }
  attachShadow(): ElementStub { this.shadowRoot = new ElementStub('shadow'); return this.shadowRoot; }
  append(...children: ElementStub[]): void {
    this.children.push(...children);
    for (const child of children) {
      child.isConnected = this.isConnected;
      if (child.isConnected) (child as ElementStub & { connectedCallback?(): void }).connectedCallback?.();
    }
  }
  replaceChildren(...children: ElementStub[]): void { this.children.length = 0; this.append(...children); }
  setAttribute(key: string, value: string): void { this.attributes.set(key, value); }
  getAttribute(key: string): string | null { return this.attributes.get(key) ?? null; }
  setCustomValidity(message: string): void { this.validationMessage = message; }
  reportValidity(): boolean { return !this.validationMessage; }
  descendants(): ElementStub[] { return this.children.flatMap((child) => [child, ...child.descendants()]); }
  querySelectorAll(): ElementStub[] { return this.descendants().filter((el) => 'param' in el.dataset || 'action' in el.dataset); }
}

const originals = new Map(['HTMLElement', 'document', 'customElements'].map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
const body = new ElementStub('body');
body.isConnected = true;
const registry = new Map();
Object.defineProperties(globalThis, {
  HTMLElement: { configurable: true, value: ElementStub },
  document: { configurable: true, value: { body, createElement: (tag: string) => new ElementStub(tag) } },
  customElements: { configurable: true, value: { get: (name: string) => registry.get(name), define: (name: string, ctor: unknown) => registry.set(name, ctor) } },
});
after(() => {
  for (const [key, descriptor] of originals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else Reflect.deleteProperty(globalThis, key);
  }
});
const { mountTextControls } = await import('./controls-text.fixture.js');

function mounted() {
  const { source, element } = mountTextControls();
  const root = element.shadowRoot as unknown as ElementStub;
  const field = () => root.descendants().find((el) => el.tagName === 'textarea')!;
  const input = (value: string) => {
    root.activeElement = field();
    field().value = value;
    field().dispatchEvent(new Event('input'));
  };
  const enter = (extra = {}) => {
    const event = new Event('keydown', { cancelable: true });
    Object.assign(event, { key: 'Enter', shiftKey: false, isComposing: false, keyCode: 13 }, extra);
    field().dispatchEvent(event);
    return event.defaultPrevented;
  };
  const blur = () => { root.activeElement = null; field().dispatchEvent(new Event('blur')); };
  return { source, element, root, field, input, enter, blur };
}

test('Text Invalid Drafts Survive Focus Loss, Frames, and Row Changes', () => {
  const h = mounted();
  for (const invalid of ['é'.repeat(2049), 'a\0b', '\ud800', '\udc00']) {
    h.input(invalid);
    h.blur();
    h.source.publish('background');
    h.source.status();
    assert.equal(h.field().value, invalid);
    assert.ok(h.field().validationMessage);
    h.source.publish('background', true);
    assert.equal(h.field().value, invalid);
    assert.ok(h.field().validationMessage);
    assert.equal(h.source.calls.length, 0);
  }
  const valid = 'é'.repeat(2048);
  h.input(valid);
  assert.equal(h.enter(), true);
  h.blur();
  assert.deepEqual(h.source.calls.map((c) => c.request.value), [valid]);
  assert.equal(h.field().validationMessage, '');
  h.source.respond(0, 'ok');
  h.source.publish(valid);
  h.source.publish('remote');
  assert.equal(h.field().value, 'remote');
});

test('Text Return and Focus Loss Submit Each Edit Once', () => {
  const h = mounted();
  h.input('first\nsecond');
  h.source.publish('background');
  assert.equal(h.field().value, 'first\nsecond');
  assert.equal(h.enter(), true);
  h.enter();
  h.blur();
  h.source.publish('background');
  assert.equal(h.field().value, 'first\nsecond');
  assert.deepEqual(h.source.calls.map((c) => c.request.value), ['first\nsecond']);
  h.source.respond(0, 'ok');
  h.source.publish('first\nsecond');
  h.input('focus loss');
  h.blur();
  h.input('');
  h.enter();
  h.blur();
  assert.deepEqual(h.source.calls.map((c) => c.request.value), ['first\nsecond', 'focus loss', '']);
});

test('Text Shift Return and Composition Do Not Submit', () => {
  const h = mounted();
  h.input('draft');
  assert.equal(h.enter({ shiftKey: true }), false);
  assert.equal(h.enter({ isComposing: true }), false);
  assert.equal(h.enter({ keyCode: 229 }), false);
  assert.equal(h.source.calls.length, 0);
});

test('Text Refused Drafts Survive Frames and Permit a Revised Submission', async () => {
  const h = mounted();
  h.input('refused');
  h.enter();
  h.blur();
  h.source.respond(0, 'error');
  await Promise.resolve();
  h.source.publish('background', true);
  h.source.status();
  assert.equal(h.field().value, 'refused');
  h.input('revised');
  h.blur();
  assert.deepEqual(h.source.calls.map((c) => c.request.value), ['refused', 'revised']);
});

test('Text Unsent Drafts Survive Disabled Access and Send Failures', async () => {
  const h = mounted();
  h.input('draft');
  h.source.allowed = false;
  h.blur();
  h.source.status();
  assert.equal(h.field().value, 'draft');
  h.source.allowed = true;
  h.source.failSend = true;
  h.enter();
  await Promise.resolve();
  await Promise.resolve();
  h.source.publish('background');
  assert.equal(h.field().value, 'draft');
  h.source.failSend = false;
  h.enter();
  assert.deepEqual(h.source.calls.map((c) => c.request.value), ['draft']);
});

test('Text Drafts End with the Session or Source', () => {
  const h = mounted();
  h.input('é'.repeat(2049));
  h.blur();
  h.source.drop();
  h.source.publish('new session');
  assert.equal(h.field().value, 'new session');
  h.input('é'.repeat(2049));
  h.blur();
  h.element.source = null;
  h.element.source = h.source;
  assert.equal(h.field().value, 'new session');
});
