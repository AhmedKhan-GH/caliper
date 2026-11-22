/* The window shared by the roadmap graph and the candidate navigator: the same frame, rail, search box,
   pillar links and tooltip placement. Plain ES2017, no build step. */
(function () {
  "use strict";

  var NARROW = window.matchMedia("(max-width: 44.9375em)");

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  /* The window fills the browser window below its top edge, so that the page itself does not scroll while the
     canvas is used; the rail scrolls inside it when it is taller. On a narrow screen the page flows instead. */
  function fit(mount) {
    if (NARROW.matches) { mount.style.height = ""; return; }
    var top = mount.getBoundingClientRect().top + window.scrollY;
    mount.style.height = Math.max(420, window.innerHeight - top - 16) + "px";
  }

  function searchBox(placeholder, label) {
    var box = el("label", "roadmap-search");
    var input = el("input");
    input.type = "search";
    input.placeholder = placeholder;
    input.setAttribute("aria-label", label);
    box.appendChild(input);
    return { box: box, input: input, status: el("div", "roadmap-search-status") };
  }

  function railLink(href, name, note) {
    var a = el("a", "roadmap-rail-link");
    a.href = href;
    a.append(el("span", "roadmap-pillar-name", name), el("span", "roadmap-pillar-counts", note));
    return a;
  }

  /* A pillar entry. On a wide screen a plain click is handed to onPick; elsewhere the link is followed. */
  function pillarLink(opts) {
    var a = el("a", "roadmap-pillar");
    a.href = opts.href;
    a.setAttribute("data-pillar", opts.prefix);
    a.append(el("span", "roadmap-pillar-name", opts.name), el("span", "roadmap-pillar-counts", opts.counts));
    a.addEventListener("click", function (e) {
      if (NARROW.matches || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      e.preventDefault();
      opts.onPick(opts.prefix);
    });
    return a;
  }

  /* Puts the open tooltip right of the anchor, or left of it when there is no room, 8 px inside the frame. */
  function placeTip(tip, anchorRect, frame) {
    var c = frame.getBoundingClientRect(), r = anchorRect;
    var tw = tip.offsetWidth, th = tip.offsetHeight;
    var x = r.right - c.left + 8;
    if (x + tw > c.width - 8) x = r.left - c.left - tw - 8;
    x = Math.max(8, Math.min(x, c.width - tw - 8));
    var y = Math.max(8, Math.min(r.top - c.top, c.height - th - 8));
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }

  /* Zoom buttons in the canvas's top-right corner: out, in, and one named action such as Fit or Reset. */
  function zoomControls(onOut, onIn, actionName, onAction) {
    var box = el("div", "roadmap-zoom");
    [["\u2212", "Zoom out", onOut], ["+", "Zoom in", onIn], [actionName, actionName, onAction]].forEach(function (b) {
      var btn = el("button", "roadmap-zoom__btn", b[0]);
      btn.type = "button";
      btn.setAttribute("aria-label", b[1]);
      btn.title = b[1];
      btn.addEventListener("click", function (e) { e.stopPropagation(); b[2](); });
      btn.addEventListener("pointerdown", function (e) { e.stopPropagation(); });
      box.appendChild(btn);
    });
    return box;
  }

  window.RoadmapWindow = { NARROW: NARROW, el: el, fit: fit, searchBox: searchBox, pillarLink: pillarLink,
    railLink: railLink, placeTip: placeTip, zoomControls: zoomControls };
})();
