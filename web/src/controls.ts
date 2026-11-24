// The <caliper-controls> element: the controls of wire protocol 3.1 for the
// session of a stream, while the script runs. It shows the run state with
// Pause or Resume and Step, every parameter the script registered as a slider
// (linear or log), a number or integer field, a toggle or a menu, and every
// action as a button with its state and its last result. The controls travel
// on the stream's own WebSocket (shared with the <caliper-view> elements of
// the same src), so they work while a notebook cell runs.
//
// A change is sent as CONTROL, and the control shows the value it asked for
// until the producer's CONTROLS shows it pending or applied, so every viewer
// shows what the producer holds. Attributes: src, height. Methods:
// setParam(name, value), runAction(name), pause(), resume(), step(n).
// Events: caliper-controls (the state changed; detail: the CONTROLS message or
// null) and caliper-reply (a reply to this element's request).

import { acquire, release } from './client/socket.js';
import type { StreamClient, StreamListener } from './client/stream.js';
import { RunBar, sendControl } from './controls/run.js';
import type { ActionInfo, ControlReplyMsg, ControlsMsg, ParamInfo, ParamValue } from './protocol/messages.js';
import { isParamText } from './protocol/param-text.js';
import { C, FONT_MONO, FONT_TEXT, fmt } from './views/theme.js';

const STYLE = `
:host { display: block; color: ${C.parchment}; font: 13px/1.4 ${FONT_TEXT}; -webkit-font-smoothing: antialiased; }
:host([hidden]) { display: none; }
* { box-sizing: border-box; }
.panel { background: ${C.plate}; border: 1px solid ${C.ruleFaint}; border-radius: 6px; overflow: auto; padding: 6px 10px 8px; }
.row { display: flex; align-items: center; gap: 8px; min-height: 28px; }
.label { color: ${C.faint}; font-size: 12px; min-width: 7em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.section { color: ${C.faint}; font-size: 11px; letter-spacing: .06em; margin: 6px 0 2px; }
.grid { display: grid; grid-template-columns: minmax(6em, max-content) minmax(8em, 1fr) minmax(5em, max-content) max-content; gap: 3px 10px; align-items: center; }
.name { color: ${C.secondary}; font-size: 12.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.value { font: 11.5px ${FONT_MONO}; color: ${C.parchment}; white-space: nowrap; }
.note { font: 11px ${FONT_TEXT}; color: ${C.faint}; white-space: nowrap; }
.note[data-kind="pending"] { color: ${C.bronze}; }
.note[data-kind="error"] { color: ${C.secondary}; }
input[type="range"] { width: 100%; accent-color: ${C.bronze}; margin: 0; }
input[type="number"], select, textarea { font: 12px ${FONT_MONO}; color: ${C.parchment}; background: #1c1b18; border: 1px solid ${C.rule};
  border-radius: 3px; padding: 1px 4px; height: 22px; width: 8em; }
textarea { width: 100%; height: 4em; resize: vertical; }
select { font-family: ${FONT_TEXT}; width: auto; }
.tool { font: 11px ${FONT_TEXT}; letter-spacing: .06em; color: ${C.faint}; background: transparent; border: 1px solid ${C.rule};
  border-radius: 3px; padding: 1px 8px; height: 22px; cursor: pointer; }
.tool:hover { color: ${C.parchment}; }
.tool[aria-pressed="true"] { color: ${C.bronze}; border-color: ${C.bronze}; }
.tool:disabled, input:disabled, select:disabled, textarea:disabled { opacity: .45; cursor: default; }
.tool:disabled:hover { color: ${C.faint}; }
.tool[aria-busy="true"] { color: ${C.bronze}; border-color: ${C.bronze}; }
button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible { outline: 1px solid ${C.bronze}; outline-offset: 1px; }
.run { display: inline-flex; align-items: center; gap: 6px; }
.run[hidden], .run [hidden], [hidden] { display: none !important; }
.badge { font: 11px ${FONT_MONO}; color: ${C.faint}; border: 1px solid ${C.rule}; border-radius: 3px; padding: 1px 6px; white-space: nowrap; }
.badge[data-state="paused"], .badge[data-state="pausing"] { color: ${C.bronze}; border-color: ${C.bronze}; }
.actions { display: flex; flex-wrap: wrap; gap: 6px 14px; }
.action { display: inline-flex; align-items: center; gap: 6px; }
.msg { color: ${C.faint}; font-size: 12.5px; padding: 4px 0; }
`;

const SLIDER = 1000;
const MAX_SLIDER_STEPS = 10000;

/** What a parameter's control shows while a request of this element is open. */
interface Local { value: ParamValue; error?: string }

export class CaliperControls extends HTMLElement {
  static get observedAttributes(): string[] {
    return ['src', 'height'];
  }

  private sourceProp: StreamClient | null = null;
  private client: StreamClient | null = null;
  private acquired: string | null = null;
  private unlisten: (() => void) | null = null;
  private readonly root: ShadowRoot;
  private readonly panel = document.createElement('div');
  private readonly runRow = document.createElement('div');
  private readonly paramsHead = document.createElement('div');
  private readonly params = document.createElement('div');
  private readonly actionsHead = document.createElement('div');
  private readonly actions = document.createElement('div');
  private readonly msg = document.createElement('div');
  private readonly runBar = new RunBar(() => this.client, true, (r) => this.replied(r));
  private readonly local = new Map<string, Local>();
  private readonly results = new Map<string, { text: string; error: boolean }>();
  private readonly dragging = new Set<string>();
  /** The parameter and action lists the rows were built for (rebuilt when they change). */
  private built = '';
  private rows = new Map<string, { update(p: ParamInfo): void }>();
  private actionRows = new Map<string, { update(a: ActionInfo): void }>();

  constructor() {
    super();
    this.root = this.attachShadow({ mode: 'open' });
    const style = document.createElement('style');
    style.textContent = STYLE;
    this.panel.className = 'panel';
    this.runRow.className = 'row';
    const runLabel = document.createElement('span');
    runLabel.className = 'label';
    runLabel.textContent = 'run';
    this.runRow.append(runLabel, this.runBar.el);
    this.paramsHead.className = 'section';
    this.paramsHead.textContent = 'parameters';
    this.params.className = 'grid';
    this.actionsHead.className = 'section';
    this.actionsHead.textContent = 'actions';
    this.actions.className = 'actions';
    this.msg.className = 'msg';
    this.panel.append(this.msg, this.runRow, this.paramsHead, this.params, this.actionsHead, this.actions);
    this.root.append(style, this.panel);
  }

  // --- public interface -------------------------------------------------------

  /** A StreamClient to read from instead of the socket named by src. */
  get source(): StreamClient | null {
    return this.sourceProp;
  }

  set source(c: StreamClient | null) {
    this.sourceProp = c;
    if (this.isConnected) this.attach();
  }

  /** The latest CONTROLS of the session, or null. */
  get controls(): ControlsMsg | null {
    return this.client?.controls ?? null;
  }

  /** Sets a parameter; resolves with the reply, or null when it could not be sent. */
  setParam(name: string, value: ParamValue): Promise<ControlReplyMsg | null> {
    if (!this.client?.controlAccess?.().allowed) return Promise.resolve(null);
    if (this.param(name)?.type === 'text' && !isParamText(value)) return Promise.resolve(null);
    this.local.set(name, { value });
    this.render();
    return sendControl(this.client, { op: 'set', name, value }, (r) => {
      if (r.status === 'ok' && r.value !== undefined) this.local.set(name, { value: r.value });
      else if (r.status === 'error') this.local.set(name, { value, error: r.error?.message || r.error?.code || 'refused' });
      this.replied(r);
      this.render();
    });
  }

  /** Runs an action; resolves with its result (after "queued"), or null when it could not be sent. */
  runAction(name: string): Promise<ControlReplyMsg | null> {
    return sendControl(this.client, { op: 'run', name }, (r) => {
      if (r.status === 'ok') this.results.set(name, { text: r.message || 'done', error: false });
      else if (r.status === 'error') this.results.set(name, { text: r.error?.message || r.error?.code || 'failed', error: true });
      this.replied(r);
      this.render();
    });
  }

  pause(): Promise<ControlReplyMsg | null> {
    return this.runBar.pause();
  }

  resume(): Promise<ControlReplyMsg | null> {
    return this.runBar.resume();
  }

  step(n = 1): Promise<ControlReplyMsg | null> {
    return this.runBar.step(n);
  }

  /** Read-only state for tests and debugging. */
  debugState(): Record<string, unknown> {
    return {
      access: this.client?.controlAccess?.() ?? null, controls: this.client?.controls ?? null,
      local: Object.fromEntries(this.local), results: Object.fromEntries(this.results),
    };
  }

  // --- lifecycle --------------------------------------------------------------

  connectedCallback(): void {
    this.applyHeight();
    this.attach();
  }

  disconnectedCallback(): void {
    this.detach();
  }

  attributeChangedCallback(name: string, old: string | null, value: string | null): void {
    if (!this.isConnected || old === value) return;
    if (name === 'height') this.applyHeight();
    else this.attach();
  }

  private applyHeight(): void {
    const h = Number(this.getAttribute('height'));
    this.panel.style.maxHeight = h > 0 ? `${h}px` : '';
  }

  private attach(): void {
    this.detach();
    let client = this.sourceProp;
    const src = this.getAttribute('src');
    if (!client && src) {
      client = acquire(src);
      this.acquired = src;
    }
    this.client = client;
    if (client) {
      const listener: StreamListener = {
        onControls: (c) => {
          this.render();
          this.dispatchEvent(new CustomEvent('caliper-controls', { detail: c, bubbles: true, composed: true }));
        },
        onStatus: () => this.render(),
        onSession: () => this.render(),
        onViews: () => this.render(),
      };
      this.unlisten = client.listen(listener);
      client.connect();
    }
    this.render();
  }

  private detach(): void {
    this.unlisten?.();
    this.unlisten = null;
    if (this.acquired) release(this.acquired);
    this.acquired = null;
    this.client = null;
    this.built = '';
  }

  private replied(r: ControlReplyMsg): void {
    this.dispatchEvent(new CustomEvent('caliper-reply', { detail: r, bubbles: true, composed: true }));
  }

  // --- drawing ----------------------------------------------------------------

  private render(): void {
    const c = this.client;
    const access = c?.controlAccess?.() ?? { shown: false, allowed: false, reason: '' };
    const ctl = access.shown ? c?.controls ?? null : null;
    this.runBar.update();
    this.msg.hidden = !!ctl;
    this.msg.textContent = !c ? 'Not connected' : c.state === 'connecting' ? 'Connecting' : 'The script has no controls';
    this.runRow.hidden = !ctl;
    const params = ctl?.params ?? [], actions = ctl?.actions ?? [];
    this.paramsHead.hidden = this.params.hidden = !params.length;
    this.actionsHead.hidden = this.actions.hidden = !actions.length;
    // Rebuild the rows when the lists change; otherwise update them in place,
    // so a slider being dragged keeps its pointer.
    const shape = JSON.stringify([params.map((p) => [p.name, p.type, p.min, p.max, p.log, p.choices]), actions.map((a) => a.name)]);
    if (shape !== this.built) {
      this.built = shape;
      this.params.replaceChildren();
      this.actions.replaceChildren();
      this.rows.clear();
      this.actionRows.clear();
      for (const p of params) this.rows.set(p.name, this.paramRow(p));
      for (const a of actions) this.actionRows.set(a.name, this.actionRow(a));
    }
    for (const p of params) {
      // The producer's state confirms what this element asked for: the value
      // then shows as the producer holds it (pending or applied).
      const l = this.local.get(p.name);
      if (l && !l.error && (p.pending === l.value || (p.pending === undefined && p.value === l.value))) this.local.delete(p.name);
      this.rows.get(p.name)?.update(p);
    }
    for (const a of actions) this.actionRows.get(a.name)?.update(a);
    const off = !access.allowed;
    for (const el of this.root.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement | HTMLTextAreaElement>('.grid input, .grid select, .grid button, .grid textarea, .actions button')) {
      el.disabled = off;
      if (off) el.title = access.reason;
    }
  }

  /** The value a parameter's control shows, and the note beside it. */
  private shown(p: ParamInfo): { value: ParamValue; note: string; kind: string } {
    const l = this.local.get(p.name);
    if (l?.error) return { value: p.pending ?? p.value, note: l.error, kind: 'error' };
    if (l) return { value: l.value, note: 'sending', kind: 'pending' };
    if (p.pending !== undefined) {
      const state = this.client?.controls?.run.state;
      return { value: p.pending, note: state === 'none' ? 'pending' : 'at the next step', kind: 'pending' };
    }
    return { value: p.value, note: '', kind: '' };
  }

  private paramRow(p: ParamInfo): { update(p: ParamInfo): void } {
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = p.name;
    name.title = p.help ?? p.name;
    const control = document.createElement('span');
    const value = document.createElement('span');
    value.className = 'value';
    const note = document.createElement('span');
    note.className = 'note';
    this.params.append(name, control, value, note);
    const show = (p2: ParamInfo) => {
      const s = this.shown(p2);
      value.textContent = typeof s.value === 'number' ? fmt(s.value, 4) : String(s.value);
      note.textContent = s.note;
      note.dataset.kind = s.kind;
      return s;
    };
    const send = (v: ParamValue) => void this.setParam(p.name, v);

    if (p.type === 'boolean') {
      const b = document.createElement('button');
      b.className = 'tool';
      b.dataset.param = p.name;
      b.addEventListener('click', () => send(!(this.shown(this.param(p.name) ?? p).value as boolean)));
      control.append(b);
      return {
        update: (p2) => {
          const s = show(p2);
          b.textContent = s.value ? 'on' : 'off';
          b.setAttribute('aria-pressed', String(!!s.value));
          value.textContent = '';
        },
      };
    }
    if (p.type === 'choice') {
      const sel = document.createElement('select');
      sel.dataset.param = p.name;
      sel.setAttribute('aria-label', p.name);
      for (const c of p.choices ?? []) sel.append(new Option(c, c));
      sel.addEventListener('change', () => send(sel.value));
      control.append(sel);
      return {
        update: (p2) => {
          const s = show(p2);
          sel.value = String(s.value);
          value.textContent = '';
        },
      };
    }
    if (p.type === 'text') {
      const field = document.createElement('textarea');
      field.dataset.param = p.name;
      field.setAttribute('aria-label', p.name);
      field.addEventListener('input', () => {
        field.setCustomValidity(isParamText(field.value) ? '' : 'Text accepts at most 4096 UTF-8 bytes without NUL.');
      });
      field.addEventListener('change', () => {
        if (field.reportValidity() && isParamText(field.value)) send(field.value);
      });
      control.append(field);
      return {
        update: (p2) => {
          const s = show(p2);
          if (this.shadowRoot?.activeElement !== field) {
            field.value = String(s.value);
            field.setCustomValidity('');
          }
          value.textContent = '';
        },
      };
    }
    // number and integer: a slider over a finite range (on a log scale with
    // log), and a field for an exact value.
    const int = p.type === 'integer';
    const lo = p.min, hi = p.max;
    const ranged = lo != null && hi != null && hi > lo && (!int || p.log || hi - lo <= MAX_SLIDER_STEPS);
    const log = !!p.log && ranged && lo! > 0;
    const toPos = (v: number) => (log ? (Math.log(v / lo!) / Math.log(hi! / lo!)) * SLIDER : ((v - lo!) / (hi! - lo!)) * SLIDER);
    const fromPos = (x: number) => {
      const t = x / SLIDER;
      const v = log ? lo! * (hi! / lo!) ** t : lo! + t * (hi! - lo!);
      return int ? Math.round(v) : Number(v.toPrecision(6));
    };
    const field = document.createElement('input');
    field.type = 'number';
    field.dataset.param = p.name;
    field.setAttribute('aria-label', p.name);
    if (lo != null) field.min = String(lo);
    if (hi != null) field.max = String(hi);
    field.step = int ? '1' : 'any';
    field.addEventListener('change', () => {
      const v = Number(field.value);
      if (field.value !== '' && Number.isFinite(v)) send(int ? Math.round(v) : v);
    });
    let slider: HTMLInputElement | null = null;
    if (ranged) {
      slider = document.createElement('input');
      slider.type = 'range';
      slider.min = '0';
      slider.max = String(SLIDER);
      slider.step = int && !log ? String(SLIDER / (hi! - lo!)) : '1';
      slider.dataset.param = p.name;
      slider.setAttribute('aria-label', `${p.name} slider`);
      if (p.help) slider.title = p.help;
      const s = slider;
      s.addEventListener('pointerdown', () => this.dragging.add(p.name));
      // While dragging, the value goes out at most 30 times a second (the
      // client coalesces), ending with the value where the drag stops.
      s.addEventListener('input', () => send(fromPos(Number(s.value))));
      s.addEventListener('change', () => {
        this.dragging.delete(p.name);
        send(fromPos(Number(s.value)));
      });
      control.append(s);
      value.replaceWith(field);
      value.className = 'value';
      // The field stands where the value text was; the slider takes the control column.
    } else {
      control.append(field);
    }
    return {
      update: (p2) => {
        const s = show(p2);
        const v = s.value as number;
        if (slider && !this.dragging.has(p.name)) slider.value = String(toPos(v));
        if (this.root.activeElement !== field) field.value = String(v);
      },
    };
  }

  private param(name: string): ParamInfo | undefined {
    return this.client?.controls?.params.find((p) => p.name === name);
  }

  private actionRow(a: ActionInfo): { update(a: ActionInfo): void } {
    const wrap = document.createElement('span');
    wrap.className = 'action';
    const b = document.createElement('button');
    b.className = 'tool';
    b.dataset.action = a.name;
    b.textContent = a.name;
    if (a.help) b.title = a.help;
    b.addEventListener('click', () => void this.runAction(a.name));
    const note = document.createElement('span');
    note.className = 'note';
    wrap.append(b, note);
    this.actions.append(wrap);
    return {
      update: (a2) => {
        const busy = a2.running || a2.queued;
        b.setAttribute('aria-busy', String(busy));
        const r = this.results.get(a2.name);
        const none = this.client?.controls?.run.state === 'none';
        note.textContent = a2.running ? 'running' : a2.queued ? (none ? 'waits for a checkpoint' : 'queued') : r?.text ?? '';
        note.dataset.kind = busy ? 'pending' : r?.error ? 'error' : '';
        if (a2.help && !b.disabled) b.title = a2.help;
      },
    };
  }
}

export function defineControls(): void {
  if (!customElements.get('caliper-controls')) customElements.define('caliper-controls', CaliperControls);
}
