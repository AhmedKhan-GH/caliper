/* The page-heading list under the H1: a compact bar that shows the section in view, and a toggle for the full list.
   Material marks the active link (navigation tracking); this script only copies its text into the bar and opens
   or closes the list. It runs again after each page change when instant navigation is on (document$). */
(function () {
  function header() {
    var h = document.querySelector(".md-header");
    if (h) document.documentElement.style.setProperty("--sb-header", h.getBoundingClientRect().height + "px");
  }
  window.addEventListener("resize", header);
  function init() {
    header();
    var root = document.querySelector("[data-sb-toc]");
    if (!root || root.__sbInit) return;
    root.__sbInit = true;
    var bar = root.querySelector(".sb-toc__bar");
    var label = root.querySelector(".sb-toc__label");
    var panel = root.querySelector(".sb-toc__panel");
    var title = root.getAttribute("data-sb-first") || "";
    var inline = document.querySelector("[data-sb-toc-inline]");

    function current() {
      var act = root.querySelectorAll(".md-nav__link--active");
      var a = act.length ? act[act.length - 1] : null;
      var text = a ? a.textContent.trim() : title;
      if (label.textContent !== text) label.textContent = text;
    }
    function setOpen(open) {
      bar.setAttribute("aria-expanded", open ? "true" : "false");
      if (open) panel.removeAttribute("hidden"); else panel.setAttribute("hidden", "");
      root.classList.toggle("sb-toc--open", open);
      if (open) {
        var a = panel.querySelector(".md-nav__link--active");
        if (a) panel.scrollTop = Math.max(0, a.offsetTop - panel.clientHeight / 2);
      }
    }
    bar.addEventListener("click", function () { setOpen(bar.getAttribute("aria-expanded") !== "true"); });
    panel.addEventListener("click", function (e) { if (e.target.closest("a")) setOpen(false); });
    function outside(e) { if (!root.isConnected) return document.removeEventListener("click", outside); if (!root.contains(e.target)) setOpen(false); }
    document.addEventListener("click", outside);
    document.addEventListener("keydown", function (e) { if (e.key === "Escape" && root.isConnected) { setOpen(false); bar.focus(); } });

    new MutationObserver(current).observe(root.querySelector("[data-md-component=toc]"), { attributes: true, attributeFilter: ["class"], subtree: true });
    current();

    /* The bar shows (on desktop) once the inline list has scrolled out from under the header. */
    if (inline && "IntersectionObserver" in window) {
      var io;
      var watch = function () {
        if (io) io.disconnect();
        var h = document.querySelector(".md-header");
        io = new IntersectionObserver(function (es) {
          var pinned = !es[es.length - 1].isIntersecting;
          root.classList.toggle("sb-toc--pinned", pinned);
          if (!pinned) setOpen(false);
        }, { rootMargin: "-" + Math.round(h ? h.getBoundingClientRect().height : 0) + "px 0px 0px 0px" });
        io.observe(inline);
      };
      watch();
      window.addEventListener("resize", function () { if (root.isConnected) watch(); });
    } else root.classList.add("sb-toc--pinned");
  }
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(init);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
