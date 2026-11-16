// The anywidget module: one <caliper-view> per widget view, built from the
// traits src, view, kind and height; selection and data_view go back to Python.

function render({ model, el }) {
  const view = document.createElement("caliper-view");
  const sync = () => {
    const src = model.get("src");
    const kind = model.get("kind");
    if (src) view.setAttribute("src", src); else view.removeAttribute("src");
    view.setAttribute("view", model.get("view") || "");
    if (kind) view.setAttribute("kind", kind); else view.removeAttribute("kind");
    view.setAttribute("height", String(model.get("height") || 320));
    view.dataView = model.get("data_view") || {};
  };
  const traits = ["src", "view", "kind", "height", "data_view"];
  traits.forEach((t) => model.on(`change:${t}`, sync));
  const onSelect = (e) => { model.set("selection", e.detail); model.save_changes(); };
  const onDataView = (e) => { model.set("data_view", e.detail); model.save_changes(); };
  view.addEventListener("caliper-select", onSelect);
  view.addEventListener("caliper-dataview", onDataView);
  sync();
  el.appendChild(view);
  return () => {
    traits.forEach((t) => model.off(`change:${t}`, sync));
    view.removeEventListener("caliper-select", onSelect);
    view.removeEventListener("caliper-dataview", onDataView);
    view.remove();
  };
}

export default { render };
