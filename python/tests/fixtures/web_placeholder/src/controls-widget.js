// The anywidget module of the controls panel: a placeholder that shows the
// stream it would control, from the traits src and height.

function render({ model, el }) {
  const box = document.createElement("div");
  const sync = () => {
    box.textContent = `controls of ${model.get("src") || "no stream"}`;
    box.style.maxHeight = `${model.get("height") || 160}px`;
  };
  ["src", "height"].forEach((t) => model.on(`change:${t}`, sync));
  sync();
  el.append(box);
  return () => box.remove();
}

export default { render };
