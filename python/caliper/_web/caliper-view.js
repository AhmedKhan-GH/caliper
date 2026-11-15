// CALIPER PLACEHOLDER ASSET. A stand-in for the build of the web/ package, made from python/tests/fixtures/web_placeholder. It opens the stream socket, counts binary frames and draws one bar per frame; it does not decode the wire protocol. The build of web/ replaces it.
// <caliper-view>: opens the stream socket named by `src` (one socket per src on
// a page), counts the binary frames and draws the last frames as bars on a
// canvas. A click sends a `caliper-select` event.

const COLORS = {
  ground: "#181715", plate: "#272622", rule: "#36342f",
  text: "#f0e9db", faint: "#a49d8f", accent: "#b38b59",
};
const HISTORY = 96;
const streams = new Map();

function connect(src) {
  let s = streams.get(src);
  if (s) return s;
  s = { src, frames: 0, bytes: 0, state: "connecting", bars: [], views: new Set(), ws: null };
  const ws = new WebSocket(src);
  ws.binaryType = "arraybuffer";
  s.ws = ws;
  const notify = () => s.views.forEach((v) => v.requestDraw());
  ws.addEventListener("open", () => { s.state = "open"; notify(); });
  ws.addEventListener("error", () => { s.state = "error"; notify(); });
  ws.addEventListener("close", () => {
    s.state = "closed";
    if (streams.get(src) === s) streams.delete(src);
    notify();
  });
  ws.addEventListener("message", (e) => {
    if (typeof e.data === "string") return;
    const b = new Uint8Array(e.data, 0, Math.min(e.data.byteLength, 64));
    let sum = 0;
    for (const x of b) sum += x;
    s.frames += 1;
    s.bytes += e.data.byteLength;
    s.bars.push(0.55 + 0.35 * Math.sin(s.frames / 7) + 0.1 * ((sum % 256) / 255));
    if (s.bars.length > HISTORY) s.bars.shift();
    notify();
  });
  streams.set(src, s);
  return s;
}

function release(s, view) {
  s.views.delete(view);
  if (s.views.size === 0) {
    if (streams.get(s.src) === s) streams.delete(s.src);
    s.ws.close();
  }
}

class CaliperView extends HTMLElement {
  static get observedAttributes() { return ["src", "view", "kind", "height"]; }

  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this.shadowRoot.innerHTML =
      `<style>:host{display:block;background:${COLORS.ground}}` +
      "canvas{display:block;width:100%;cursor:crosshair}</style><canvas></canvas>";
    this.canvas = this.shadowRoot.querySelector("canvas");
    this.stream = null;
    this.dataView = null;
    this.pending = false;
    this.resize = new ResizeObserver(() => this.requestDraw());
    this.canvas.addEventListener("click", (e) => {
      const detail = {
        view: this.getAttribute("view") || "", type: "select",
        detail: { frame: this.frames, x: e.offsetX, y: e.offsetY },
      };
      this.dispatchEvent(new CustomEvent("caliper-select", { detail, bubbles: true, composed: true }));
    });
  }

  get frames() { return this.stream ? this.stream.frames : 0; }

  connectedCallback() { this.resize.observe(this); this.attach(); }

  disconnectedCallback() {
    this.resize.disconnect();
    if (this.stream) release(this.stream, this);
    this.stream = null;
  }

  attributeChangedCallback(name) {
    if (name === "src" && this.isConnected) this.attach();
    this.requestDraw();
  }

  attach() {
    const src = this.getAttribute("src");
    if (this.stream && this.stream.src === src) return;
    if (this.stream) release(this.stream, this);
    this.stream = src ? connect(src) : null;
    if (this.stream) this.stream.views.add(this);
    this.requestDraw();
  }

  requestDraw() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => { this.pending = false; this.draw(); });
  }

  draw() {
    const h = Number(this.getAttribute("height")) || 320;
    const w = Math.max(this.clientWidth, 1);
    const dpr = window.devicePixelRatio || 1;
    const c = this.canvas;
    c.style.height = `${h}px`;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = COLORS.ground;
    g.fillRect(0, 0, w, h);
    const pad = 12, top = 34, plotH = h - top - pad;
    g.fillStyle = COLORS.plate;
    g.fillRect(pad, top, w - 2 * pad, plotH);
    const s = this.stream;
    const bars = s ? s.bars : [];
    const bw = (w - 2 * pad) / HISTORY;
    g.fillStyle = COLORS.accent;
    bars.forEach((v, i) => {
      const bh = v * (plotH - 8);
      g.fillRect(pad + (HISTORY - bars.length + i) * bw + 1, top + plotH - bh, Math.max(bw - 2, 1), bh);
    });
    g.strokeStyle = COLORS.rule;
    g.strokeRect(pad + 0.5, top + 0.5, w - 2 * pad - 1, plotH - 1);
    g.font = "13px 'IBM Plex Sans Condensed', system-ui, sans-serif";
    g.fillStyle = COLORS.text;
    g.fillText(`${this.getAttribute("view") || "(no view)"}  placeholder`, pad, 22);
    g.font = "12px ui-monospace, 'Source Code Pro', monospace";
    g.fillStyle = COLORS.faint;
    const status = s ? `${s.state}  frames ${s.frames}` : "no src";
    g.fillText(status, w - pad - g.measureText(status).width, 22);
  }
}

if (!customElements.get("caliper-view")) customElements.define("caliper-view", CaliperView);

export { CaliperView, connect };
