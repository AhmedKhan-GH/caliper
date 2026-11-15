// A stand-in for the WebAssembly views' loader, caliper-views-loader.js, with
// the interface the element codes against. The tests serve it at the stream
// server's origin and read what it was given from globalThis.
export function mountCaliperViews(container, options) {
  if (globalThis.caliperStubFail) throw new Error('the stub refuses to mount');
  const { onSelect, onDataView, ...given } = options;
  const mark = document.createElement('div');
  mark.className = 'stub';
  mark.textContent = `wasm views: ${options.view}`;
  container.append(mark);
  const handle = {
    dataViews: [],
    destroyed: false,
    onSelect: null,
    onDataView: null,
    destroy() {
      handle.destroyed = true;
      mark.remove();
    },
    setDataView(dv) {
      handle.dataViews.push(dv);
    },
  };
  globalThis.caliperStub = { given, handle, container, hasCallbacks: typeof onSelect === 'function' && typeof onDataView === 'function' };
  return handle;
}
