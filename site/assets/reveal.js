/* Optional progressive enhancement: scroll-reveal.
 * The page is fully complete and readable without this file. It only adds a
 * subtle fade-up as sections enter the viewport. The `.reveal-on` class is
 * added here so that, with JS disabled, the reveal CSS never applies and all
 * content renders visible by default.
 */
(function () {
  "use strict";
  var els = document.querySelectorAll("[data-reveal]");
  if (!els.length) return;

  var reduce = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce || !("IntersectionObserver" in window)) return;

  document.documentElement.classList.add("reveal-on");

  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        io.unobserve(entry.target);
      }
    });
  }, { rootMargin: "0px 0px -10% 0px", threshold: 0.08 });

  els.forEach(function (el) { io.observe(el); });
})();
