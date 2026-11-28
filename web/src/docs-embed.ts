// Embeddable static replay component (<caliper-replay>) for the docs site.
//
// Loads a recorded Caliper run (.calrec) and Caliper's real WebAssembly views
// (caliper-views-loader.js, caliper-views.js, caliper-views.wasm) from paths
// relative to the page, with no install and no server. When WebGPU or the
// WebAssembly module is unavailable, renders the static fallback screenshot
// under a notice that says why, never a blank box.

import { ReplaySource, resolvePageUrl } from './client/replay.js';
import { CaliperView, define as defineView } from './element.js';
import { kindName } from './protocol/messages.js';
import { C, FONT_MONO, FONT_TEXT } from './views/theme.js';
import { hasAdapter, preloadWasm, wasmKinds } from './wasm.js';

defineView();

const SCREENSHOT_TEXT = 'The image below is a static screenshot of the replay.';

// Hang guard: the visible time from live mode until every plate's WebAssembly view is ready, after
// which a plate that is still not ready falls back. The module is fetched before live mode, so this
// covers the mount on the device and the first frame. A plate gets ready in a requestAnimationFrame
// frame, which a hidden tab does not run, so only the polls of a visible page count. Measured in
// headless Chromium on Apple silicon, five runs each (web/test/replay.spec.ts prints it): ready at the
// first 200 ms poll on Metal WebGPU and at the second, 0.4 s, on SwiftShader. 30 s is 75 times the
// slowest run, generous for slower machines.
const HEALTH_POLL_MS = 200;
const READY_DEADLINE_MS = 30_000;

/** The first sentence of the fallback notice: why the live replay does not show. */
function fallbackCause(reason: string): string {
  if (reason === 'insecure-context') {
    return 'WebGPU is not available because this page is not served over HTTPS or from localhost.';
  }
  if (reason === 'no-webgpu') return 'This browser does not support WebGPU.';
  if (reason === 'wasm-ready-timeout') return 'The live replay did not start in time.';
  if (reason === 'no-adapter' || reason.startsWith('wasm-')) return 'WebGPU did not start in this browser.';
  return 'The recorded run did not load.';
}

const STYLE = `
:host { display: block; color: ${C.parchment}; font: 13px/1.45 ${FONT_TEXT}; -webkit-font-smoothing: antialiased; }
:host([hidden]) { display: none; }
* { box-sizing: border-box; }
.shell { background: ${C.ground}; border: 1px solid ${C.ruleFaint}; border-radius: 8px; overflow: hidden; }
.bar { display: flex; align-items: center; flex-wrap: wrap; gap: 8px 14px; padding: 8px 14px;
  background: ${C.plate}; border-bottom: 1px solid ${C.ruleFaint}; }
.brand { font: 12px/1 ${FONT_TEXT}; letter-spacing: .14em; text-transform: uppercase; color: ${C.parchment}; }
.script { font: 12px ${FONT_MONO}; color: ${C.secondary}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 24em; }
.tag { font: 11px ${FONT_MONO}; color: ${C.bronze}; border: 1px solid ${C.rule}; border-radius: 3px; padding: 1px 6px; white-space: nowrap; }
.transport { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-left: auto; flex: 1 1 280px; max-width: 480px; }
.transport[hidden] { display: none; }
.play, .reset { font: 12px ${FONT_TEXT}; color: ${C.parchment}; background: ${C.deep}; border: 1px solid ${C.rule};
  border-radius: 4px; padding: 2px 10px; height: 24px; cursor: pointer; flex: none; min-width: 56px; }
.play:hover, .reset:hover { border-color: ${C.bronze}; color: ${C.bronze}; }
.play:focus-visible, .reset:focus-visible, .range:focus-visible { outline: 1px solid ${C.bronze}; outline-offset: 2px; }
.range { flex: 1; min-width: 90px; accent-color: ${C.bronze}; cursor: pointer; margin: 0; }
.step { font: 11.5px ${FONT_MONO}; color: ${C.secondary}; white-space: nowrap; flex: none; min-width: 9.5em; text-align: right; margin-left: auto; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 400px), 1fr)); gap: 12px; padding: 12px; }
.loading { padding: 36px 20px; text-align: center; color: ${C.faint}; font: 13px ${FONT_TEXT}; }
.fallback { padding: 12px; background: ${C.ground}; }
.fallback-img { display: block; width: 100%; height: auto; border: 1px solid ${C.ruleFaint}; border-radius: 6px; background: ${C.plate}; }
.fallback-note { margin: 0 0 12px; padding: 12px 16px; color: ${C.parchment}; font: 16px/1.5 ${FONT_TEXT};
  background: ${C.plate}; border: 1px solid ${C.rule}; border-left: 3px solid ${C.bronze}; border-radius: 6px; }
[hidden] { display: none !important; }
`;

export type ReplayMode = 'loading' | 'live' | 'fallback';

function parseInitialStep(raw: string | null): 'first' | 'last' | number {
  if (!raw) return 'first';
  const s = raw.trim().toLowerCase();
  if (s === 'first' || s === 'last') return s;
  const n = Number(s);
  return Number.isFinite(n) ? n : 'first';
}

export class CaliperReplay extends HTMLElement {
  static get observedAttributes(): string[] {
    return ['src', 'wasm-base', 'fallback', 'fallback-text', 'alt', 'views', 'height', 'step'];
  }

  private readonly root: ShadowRoot;
  private built = false;
  private generation = 0;
  private mode: ReplayMode = 'loading';
  private reason = '';
  private replay: ReplaySource | null = null;
  private unlistenTimeline: (() => void) | null = null;
  private healthTimer: ReturnType<typeof setInterval> | null = null;
  private liveSince = 0;
  private visibleWaitMs = 0;
  private readyMs: number | null = null;
  private plates: CaliperView[] = [];
  private readonly parts: {
    shell: HTMLElement;
    bar: HTMLElement;
    script: HTMLElement;
    transport: HTMLElement;
    play: HTMLButtonElement;
    reset: HTMLButtonElement;
    range: HTMLInputElement;
    step: HTMLElement;
    grid: HTMLElement;
    loading: HTMLElement;
    fallback: HTMLElement;
    fallbackImg: HTMLImageElement;
    fallbackNote: HTMLElement;
  };

  constructor() {
    super();
    this.root = this.attachShadow({ mode: 'open' });
    const el = <K extends keyof HTMLElementTagNameMap>(tag: K, cls: string) => {
      const e = document.createElement(tag);
      e.className = cls;
      return e;
    };
    this.parts = {
      shell: el('div', 'shell'),
      bar: el('div', 'bar'),
      script: el('span', 'script'),
      transport: el('div', 'transport'),
      play: el('button', 'play'),
      reset: el('button', 'reset'),
      range: el('input', 'range'),
      step: el('span', 'step'),
      grid: el('div', 'grid'),
      loading: el('div', 'loading'),
      fallback: el('div', 'fallback'),
      fallbackImg: el('img', 'fallback-img'),
      fallbackNote: el('p', 'fallback-note'),
    };
  }

  connectedCallback(): void {
    if (!this.built) this.build();
    void this.start();
  }

  disconnectedCallback(): void {
    this.teardown();
  }

  attributeChangedCallback(_name: string, oldVal: string | null, newVal: string | null): void {
    if (!this.isConnected || oldVal === newVal) return;
    void this.start();
  }

  /** Underlying ReplaySource once loaded in live mode, or null. */
  get source(): ReplaySource | null {
    return this.replay;
  }

  /** Moves the replay cursor to a step index (0..steps.length - 1). */
  seekIndex(index: number): void {
    this.replay?.seekIndex(index);
  }

  /** Moves the replay cursor to the latest recorded step at or before step. */
  seekStep(step: number): void {
    this.replay?.seekStep(step);
  }

  /** Starts playback through recorded steps, one step per display frame. */
  play(): void {
    this.replay?.play();
  }

  /** Pauses playback. */
  pause(): void {
    this.replay?.pause();
  }

  /** Pauses playback and returns every view to its first-step state. */
  reset(): void {
    this.replay?.reset();
  }

  /** Read-only state for automated tests and diagnostics. */
  debugState(): Record<string, unknown> {
    const r = this.replay;
    return {
      mode: this.mode,
      reason: this.reason,
      steps: r ? [...r.steps] : [],
      stepIndex: r ? r.stepIndex : 0,
      currentStep: r ? r.currentStep : 0,
      playing: r ? r.playing : false,
      readyMs: this.readyMs,
      fallbackShown: !this.parts.fallback.hidden,
      fallbackImgVisible: !this.parts.fallback.hidden && !this.parts.fallbackImg.hidden,
      fallbackSrc: this.parts.fallbackImg.getAttribute('src') ?? '',
      fallbackText: this.parts.fallbackNote.textContent ?? '',
      scrubberVisible: !this.parts.bar.hidden && !this.parts.transport.hidden,
      views: this.plates.map((p) => p.debugState()),
    };
  }

  private build(): void {
    this.built = true;
    const style = document.createElement('style');
    style.textContent = STYLE;
    const p = this.parts;

    const brand = document.createElement('span');
    brand.className = 'brand';
    brand.textContent = 'Caliper';
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.textContent = 'WebGPU';

    p.play.type = 'button';
    p.play.textContent = 'Play';
    p.play.addEventListener('click', () => this.replay?.togglePlay());
    p.reset.type = 'button';
    p.reset.textContent = 'Reset';
    p.reset.addEventListener('click', () => this.replay?.reset());
    p.range.type = 'range';
    p.range.min = '0';
    p.range.max = '0';
    p.range.value = '0';
    p.range.setAttribute('aria-label', 'Recorded training step');
    p.range.addEventListener('input', () => {
      if (!this.replay) return;
      this.replay.pause();
      this.replay.seekIndex(Number(p.range.value));
    });
    p.transport.hidden = true;
    p.transport.append(p.play, p.reset, p.range, p.step);

    p.bar.hidden = true;
    p.bar.append(brand, p.script, tag, p.transport);

    p.grid.hidden = true;
    p.loading.textContent = 'WebGPU replay load in progress.';

    p.fallbackImg.decoding = 'async';
    p.fallbackImg.addEventListener('error', () => {
      p.fallbackImg.hidden = true;
      if (this.mode === 'fallback') this.writeNotice();
    });
    p.fallbackNote.setAttribute('role', 'note');
    p.fallback.hidden = true;
    p.fallback.append(p.fallbackNote, p.fallbackImg);

    p.shell.append(p.bar, p.loading, p.grid, p.fallback);
    this.root.append(style, p.shell);
  }

  private teardown(): void {
    this.generation++;
    if (this.healthTimer) {
      clearInterval(this.healthTimer);
      this.healthTimer = null;
    }
    this.readyMs = null;
    this.unlistenTimeline?.();
    this.unlistenTimeline = null;
    for (const plate of this.plates) plate.remove();
    this.plates = [];
    this.replay?.close();
    this.replay = null;
  }

  private resolveWasmDir(): string {
    const attr = this.getAttribute('wasm-base');
    if (attr && attr.trim()) return resolvePageUrl(attr.trim()).replace(/\/?$/, '/');
    return new URL('./', import.meta.url).href;
  }

  private showLoading(): void {
    const p = this.parts;
    this.mode = 'loading';
    this.reason = '';
    p.bar.hidden = true;
    p.grid.hidden = true;
    p.fallback.hidden = true;
    p.loading.hidden = false;
  }

  private showFallback(reason: string): void {
    const p = this.parts;
    if (this.healthTimer) {
      clearInterval(this.healthTimer);
      this.healthTimer = null;
    }
    for (const plate of this.plates) plate.remove();
    this.plates = [];
    this.replay?.close();
    this.replay = null;

    this.mode = 'fallback';
    this.reason = reason;
    p.bar.hidden = true;
    p.grid.hidden = true;
    p.loading.hidden = true;

    const imgAttr = this.getAttribute('fallback');
    if (imgAttr && imgAttr.trim()) {
      p.fallbackImg.src = resolvePageUrl(imgAttr.trim());
      p.fallbackImg.alt = this.getAttribute('alt') ?? 'Recorded Caliper run';
      p.fallbackImg.hidden = false;
    } else {
      p.fallbackImg.removeAttribute('src');
      p.fallbackImg.hidden = true;
    }
    this.writeNotice();
    p.fallback.hidden = false;
  }

  /** The notice above the screenshot: the page's fallback-text, or the cause and what the image is. */
  private writeNotice(): void {
    const p = this.parts;
    const custom = this.getAttribute('fallback-text')?.trim();
    p.fallbackNote.textContent = custom
      || (p.fallbackImg.hidden ? fallbackCause(this.reason) : `${fallbackCause(this.reason)} ${SCREENSHOT_TEXT}`);
  }

  private syncTimeline(): void {
    const p = this.parts;
    const r = this.replay;
    if (!r) return;
    const steps = r.steps;
    if (steps.length <= 1) {
      p.transport.hidden = true;
      return;
    }
    p.transport.hidden = false;
    p.range.max = String(steps.length - 1);
    p.range.value = String(r.stepIndex);
    p.play.textContent = r.playing ? 'Pause' : 'Play';
    const last = steps[steps.length - 1];
    p.step.textContent = `Step ${r.currentStep} / ${last}`;
    // The widest label sets the width, so the range bar stays still while playback adds digits.
    p.step.style.minWidth = `${`Step ${last} / ${last}`.length}ch`;
  }

  private async start(): Promise<void> {
    this.teardown();
    const gen = this.generation;
    const srcAttr = this.getAttribute('src')?.trim() ?? '';
    if (!srcAttr) {
      this.showFallback('No recording path is set.');
      return;
    }
    this.showLoading();

    // Check WebGPU before fetching the WebAssembly module or recording.
    const adapterOk = await hasAdapter();
    if (gen !== this.generation) return;
    if (!adapterOk) {
      // WebGPU exists only in secure contexts, so check the context before navigator.gpu.
      const gpu = (navigator as Navigator & { gpu?: unknown }).gpu;
      this.showFallback(!window.isSecureContext ? 'insecure-context' : gpu ? 'no-adapter' : 'no-webgpu');
      return;
    }

    const wasmBase = this.resolveWasmDir();
    const recUrl = resolvePageUrl(srcAttr);
    const replay = new ReplaySource(recUrl, { initialStep: parseInitialStep(this.getAttribute('step')) });
    this.replay = replay;
    replay.connect();

    // Load the WebAssembly module and the static recording in parallel.
    const [kinds, preloaded] = await Promise.all([
      wasmKinds(wasmBase),
      preloadWasm(wasmBase),
      replay.ready,
    ]);
    if (gen !== this.generation) return;
    if (!kinds || !preloaded) {
      this.showFallback('wasm-unavailable');
      return;
    }
    if (replay.error || !replay.views.length) {
      this.showFallback(replay.error || 'empty-recording');
      return;
    }

    // Select views to render (all declared views, or the requested subset in order).
    const wantAttr = this.getAttribute('views')?.trim();
    const entries = wantAttr
      ? wantAttr.split(',').map((s) => s.trim()).filter(Boolean)
          .map((name) => replay.views.find((v) => v.name === name)).filter((v): v is NonNullable<typeof v> => !!v)
      : [...replay.views];
    const supported = entries.filter((v) => {
      const k = kindName(v.kind);
      return k !== null && kinds.includes(k);
    });
    if (!supported.length) {
      this.showFallback('no-supported-views');
      return;
    }

    const height = Math.max(160, Number(this.getAttribute('height')) || 280);
    const p = this.parts;
    p.script.textContent = replay.session?.script ?? '';
    p.script.title = p.script.textContent;
    p.grid.replaceChildren();
    this.plates = supported.map((v) => {
      const el = document.createElement('caliper-view') as CaliperView;
      el.setAttribute('view', v.name);
      const k = kindName(v.kind);
      if (k) el.setAttribute('kind', k);
      el.setAttribute('height', String(height));
      el.setAttribute('renderer', 'wasm');
      el.setAttribute('wasm-base', wasmBase);
      el.setAttribute('no-scrubber', '');
      el.source = replay;
      p.grid.append(el);
      return el;
    });

    this.unlistenTimeline = replay.onTimeline(() => this.syncTimeline());
    this.mode = 'live';
    this.reason = '';
    p.loading.hidden = true;
    p.fallback.hidden = true;
    p.bar.hidden = false;
    p.grid.hidden = false;
    this.syncTimeline();

    // Monitor WASM device startup: if any plate fails to acquire a WebGPU device or falls off WASM,
    // switch cleanly to the static fallback instead of showing a blank box. A plate that never gets
    // ready and never reports an error also falls back, after READY_DEADLINE_MS.
    this.liveSince = performance.now();
    this.visibleWaitMs = 0;
    this.readyMs = null;
    this.healthTimer = setInterval(() => {
      if (gen !== this.generation || this.mode !== 'live') return;
      let allReady = true;
      for (const plate of this.plates) {
        const st = plate.debugState();
        const wStats = st.wasmStats as { error?: string } | null;
        if (wStats && typeof wStats.error === 'string' && wStats.error.length > 0) {
          this.showFallback(`wasm-error:${wStats.error}`);
          return;
        }
        if (st.drawer === 'ts') {
          this.showFallback('wasm-mount-failed');
          return;
        }
        if (!st.wasmReady) allReady = false;
      }
      if (allReady && this.healthTimer) {
        // The first poll that saw every plate ready: an upper bound at the poll's resolution.
        this.readyMs = performance.now() - this.liveSince;
        clearInterval(this.healthTimer);
        this.healthTimer = null;
      } else if (!allReady && !document.hidden) {
        this.visibleWaitMs += HEALTH_POLL_MS;
        if (this.visibleWaitMs > READY_DEADLINE_MS) this.showFallback('wasm-ready-timeout');
      }
    }, HEALTH_POLL_MS);
  }
}

export function defineReplay(): void {
  if (!customElements.get('caliper-replay')) customElements.define('caliper-replay', CaliperReplay);
}

defineReplay();

export { CaliperView, ReplaySource };
