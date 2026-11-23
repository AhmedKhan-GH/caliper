// caliper-views-loader.js: mounts Caliper's own views (the WebAssembly view
// layer of web/wasm) on a page. One ES module with no imports of its own; it
// fetches the module's two files from `assetBase`:
//
//   import { mountCaliperViews } from '<assetBase>caliper-views-loader.js';
//   const handle = await mountCaliperViews(container, {
//     src,          // 'ws://127.0.0.1:<port>/v3/stream?token=<token>'
//     view,         // the view to show; '' or absent: every view of the session
//     kind,         // 'heatmap' | 'image' | 'points' | 'series' | 'histogram' | 'scene', optional
//     height,       // CSS pixels, optional (default: fill the container)
//     header,       // false: the page shows the view's name itself, so the view has no title bar
//     assetBase,    // where caliper-views.js and .wasm are; default: beside this file
//     onSelect,     // (detail, view) => {}, every selection; {} clears it
//     onEvent,      // ({ view, type, detail }) => {}, every select, pick and hover
//     onInfo,       // (facts, view) => {}, what a header says about the view, as it changes
//     onToggle,     // (name, on, view) => {}, a view changed a header toggle itself ('follow')
//     augment,      // (detail, view) => detail, adds to a selection's detail before it is sent (3.1 item space)
//     onDataView,   // (dataView) => {}, every data view the producer resolved
//     compat,       // true: a WebGPU device in compatibility mode; default globalThis.caliperViewsCompat
//     shapes,       // 'builtin': a scene's shapes drawn by the built-in draw in place of the library
//                   // pieces, for a comparison; default globalThis.caliperViewsShapes, else 'pieces'
//   });
//   handle.setDataView({ index: { layer: 2 } });   // a VIEW_REQUEST
//   handle.setToggle('follow', false);             // a header toggle of the view
//   handle.setLink('points', [3, 9]);              // a linked selection to draw (3.1)
//   handle.destroy();
//
// A selection, pick or hover goes to the producer as an EVENT on the mount's
// own connection, with the same detail the <caliper-view> element's
// TypeScript views send (wire protocol v3, "Events"), and to these callbacks.
//
// Each mount is its own module instance with its own canvas, WebGPU device,
// heap and WebSocket; the compiled WebAssembly module is shared by every mount
// on the page, so the second mount does not fetch or compile again. Every
// listener sits on the mount's own canvas, which is created inside `container`
// (never looked up in the document), so a container inside a shadow root works.

// The live view kinds these views draw. A page draws the other kinds itself
// (the <caliper-view> element uses its TypeScript views for them). A scene
// (the protocol 3.2 draft) is drawn by these views only.
export const CALIPER_VIEWS_KINDS = Object.freeze(['heatmap', 'image', 'points', 'series', 'histogram', 'scene']);

const compiled = new Map();   // assetBase -> Promise<{ factory, wasm }>

function base(assetBase) {
  if (assetBase) return new URL(assetBase, location.href).href.replace(/\/?$/, '/');
  return new URL('.', import.meta.url).href;
}

async function compileWasm(url) {
  const response = fetch(url, { mode: 'cors', credentials: 'omit' });
  if (WebAssembly.compileStreaming) {
    try {
      return await WebAssembly.compileStreaming(response);
    } catch (e) {
      // A server without the application/wasm type: compile from the bytes.
    }
  }
  const r = await fetch(url, { mode: 'cors', credentials: 'omit' });
  if (!r.ok) throw new Error(`caliper-views: ${url}: HTTP ${r.status}`);
  return WebAssembly.compile(await r.arrayBuffer());
}

// Fetches and compiles the module once per assetBase. Optional: a mount does
// it anyway, but a page may start it early.
export function preloadCaliperViews(assetBase) {
  const b = base(assetBase);
  if (!compiled.has(b)) {
    const p = Promise.all([import(b + 'caliper-views.js'), compileWasm(b + 'caliper-views.wasm')])
      .then(([mod, wasm]) => ({ factory: mod.default, wasm }));
    p.catch(() => compiled.delete(b));   // a failed load may be retried
    compiled.set(b, p);
  }
  return compiled.get(b);
}

export async function mountCaliperViews(container, options = {}) {
  const { src = '', view = '', kind = '', height, header = true, assetBase, onSelect, onEvent, onInfo, onToggle, augment, onDataView,
          compat = globalThis.caliperViewsCompat ?? false, shapes = globalThis.caliperViewsShapes ?? 'pieces' } = options;
  if (!container) throw new Error('caliper-views: no container element');
  if (!navigator.gpu) throw new Error('caliper-views: this browser has no WebGPU');

  const canvas = document.createElement('canvas');
  canvas.tabIndex = 0;
  canvas.style.display = 'block';
  canvas.style.width = '100%';
  canvas.style.height = height ? `${height}px` : '100%';
  container.appendChild(canvas);

  const { factory, wasm } = await preloadCaliperViews(assetBase);
  let ready;
  const readyPromise = new Promise((resolve) => { ready = resolve; });
  let module;
  try {
    module = await factory({
      canvas,
      caliperConfig: { src, view, kind, header: header ? 'true' : 'false', compat: compat ? 'true' : 'false', shapes: String(shapes) },
      instantiateWasm: (imports, receive) => {
        WebAssembly.instantiate(wasm, imports).then((instance) => receive(instance, wasm));
        return {};
      },
      onReady: () => ready(),
      onDataView: (v, dataView) => { if (onDataView && (!view || v === view)) onDataView(dataView, v); },
      onEvent: (v, type, detail) => {
        if (onEvent) onEvent({ view: v, type, detail });
        if (type === 'select' && onSelect) onSelect(detail, v);
      },
      onInfo: (v, facts) => { if (onInfo && (!view || v === view)) onInfo(facts, v); },
      onToggle: (v, name, on) => { if (onToggle && (!view || v === view)) onToggle(name, on, v); },
      onAugment: (v, detail) => (augment ? augment(detail, v) : null),
      print: () => {},
      printErr: (t) => console.warn('[caliper-views]', t),
    });
  } catch (e) {
    canvas.remove();
    throw e;
  }

  let destroyed = false;
  return {
    canvas,
    module,
    // Resolves once the first snapshot has been drawn.
    ready: readyPromise,
    // The module's counters (frames, updates, bytes, heap, the connection state).
    get stats() { return { ...(module.caliperStats || {}) }; },
    // Asks the producer for another data view of the shown view (VIEW_REQUEST);
    // onDataView hears the one it resolves. False when it could not be sent.
    setDataView(dataView) {
      if (destroyed) return false;
      const p = module.stringToNewUTF8(JSON.stringify(dataView ?? {}));
      const ok = module._caliper_views_set_data_view(p) === 1;
      module._free(p);
      return ok;
    },
    // Sets a header toggle of the view: 'follow' (series: the axes fit the
    // data each frame) or 'log' (series and histogram: y on a log scale).
    setToggle(name, on) {
      if (destroyed) return false;
      const p = module.stringToNewUTF8(String(name));
      const ok = module._caliper_views_set_toggle(p, on ? 1 : 0) === 1;
      module._free(p);
      return ok;
    },
    // Draws a linked selection (3.1): the axis of the view that carries the
    // selection's item space ('points', 'rows', 'columns' or 'steps', or '' to
    // clear) and the selected item ids.
    setLink(axis, items) {
      if (destroyed) return false;
      const a = module.stringToNewUTF8(String(axis || ''));
      const j = module.stringToNewUTF8(JSON.stringify(Array.from(items || [])));
      const ok = module._caliper_views_set_link(a, j) === 1;
      module._free(a);
      module._free(j);
      return ok;
    },
    // Says goodbye to the producer, releases the GPU and removes the canvas.
    destroy() {
      if (destroyed) return;
      destroyed = true;
      module._caliper_views_destroy();
      canvas.remove();
    },
  };
}
