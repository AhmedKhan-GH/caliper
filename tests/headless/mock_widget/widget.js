// ESM of MockCanvas. It opens a WebSocket to `src`, draws every binary frame
// it receives on a canvas inside an open shadow root, and reports back over
// the same socket how many frames arrived. Nothing is drawn until the first
// frame arrives, so a non-blank canvas proves that the stream reached it.

const GROUND = [24, 23, 21];
const ACCENT = [179, 139, 89];
const TEXT = "#f0e9db";
const FAINT = "#a49d8f";

async function probeWebGPU() {
  if (!navigator.gpu) return { has_gpu: false };
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { has_gpu: true, adapter: false };
    const device = await adapter.requestDevice();
    const info = adapter.info || {};
    device.destroy();
    return { has_gpu: true, adapter: true, device: true, vendor: info.vendor || "",
             architecture: info.architecture || "" };
  } catch (e) {
    return { has_gpu: true, error: String(e) };
  }
}

function render({ model, el }) {
  const height = model.get("height");
  const width = height * 2;
  const host = document.createElement("div");
  host.className = "mock-view";
  const root = host.attachShadow({ mode: "open" });
  const canvas = document.createElement("canvas");
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  canvas.style.display = "block";
  root.append(canvas);
  el.append(host);

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.fillStyle = "#181715";
  ctx.fillRect(0, 0, width, height);
  const field = document.createElement("canvas");

  let socket = null;
  let received = 0;
  let closed = false;

  function draw(buffer) {
    const header = new DataView(buffer, 0, 16);
    const index = header.getUint32(0, true);
    const w = header.getUint32(4, true);
    const h = header.getUint32(8, true);
    const values = new Uint8Array(buffer, 16, w * h);
    if (field.width !== w || field.height !== h) {
      field.width = w;
      field.height = h;
    }
    const image = new ImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const t = values[i] / 255;
      image.data[4 * i] = GROUND[0] + (ACCENT[0] - GROUND[0]) * t;
      image.data[4 * i + 1] = GROUND[1] + (ACCENT[1] - GROUND[1]) * t;
      image.data[4 * i + 2] = GROUND[2] + (ACCENT[2] - GROUND[2]) * t;
      image.data[4 * i + 3] = 255;
    }
    field.getContext("2d").putImageData(image, 0, 0);
    ctx.fillStyle = "#181715";
    ctx.fillRect(0, 0, width, height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(field, 0, 0, height, height);
    ctx.font = "13px monospace";
    ctx.fillStyle = TEXT;
    ctx.fillText(`frame ${index}`, height + 16, 28);
    ctx.fillStyle = FAINT;
    ctx.fillText(`received ${received}`, height + 16, 48);
  }

  function connect() {
    socket = new WebSocket(model.get("src"));
    socket.binaryType = "arraybuffer";
    socket.onopen = async () => {
      const webgpu = await probeWebGPU();
      if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ webgpu }));
    };
    socket.onmessage = (event) => {
      if (typeof event.data === "string") return;
      received += 1;
      host.dataset.received = String(received);   // for the checks, which wait on frames received
      draw(event.data);
    };
    socket.onclose = () => {
      if (!closed) setTimeout(connect, 1000);
    };
  }

  const timer = setInterval(() => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ received }));
      host.dataset.reported = String(received);   // the count the server has been sent
    }
  }, 1000);

  model.on("change:src", () => socket && socket.close());
  connect();

  return () => {
    closed = true;
    clearInterval(timer);
    if (socket) socket.close();
  };
}

export default { render };
