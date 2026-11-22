/* The candidate navigator: every candidate of every pillar as a card in one scrolling canvas, with a rail of
   pillars and a search box beside it. It reads candidates.json and shares its window with the roadmap graph.
   Plain ES2017, no build step. */
(function () {
  "use strict";

  var el = window.RoadmapWindow && RoadmapWindow.el;
  var BADGE_LABELS = { shipped: "Shipped", built: "Built", progress: "In progress", planned: "Planned",
    idea: "Idea", decision: "Decision" };
  var SIDE_PAD = 24, SEARCH_DELAY = 120;

  function plural(n, one, many) { return n + " " + (n === 1 ? one : many); }
  function mark(origin) { return el("i", "cn-mark cn-mark--" + origin); }

  function start(mount) {
    var data = null, jsonUrl = null;
    var cards = [];          // every card in canvas order: { key, cand, pillar, el, hay }
    var byKey = {}, sections = {}, pillarEls = {}, originTitle = {};
    var selected = null, matches = [], current = -1, timer = null, lastQuery = "";
    var resolveReady;
    var api = { ready: new Promise(function (r) { resolveReady = r; }), data: null };
    mount.__navigator = api;

    /* Rail: legend, search and the pillar list. */
    var rail = el("div", "roadmap-rail");
    var legend = el("div", "roadmap-legend");
    var sb = RoadmapWindow.searchBox("Find a candidate by key or word",
      "Find candidates by key, title, summary or serves, then press Enter for the next match");
    var input = sb.input;
    var pillarList = el("div", "roadmap-pillars");
    rail.append(legend, sb.box, sb.status, pillarList);

    var canvas = el("div", "roadmap-canvas cn-canvas");
    var scroll = el("div", "cn-scroll");
    var tip = el("div", "roadmap-tooltip");
    tip.setAttribute("role", "tooltip");
    var panel = el("aside", "cn-panel");
    panel.setAttribute("aria-label", "Candidate details");
    canvas.append(scroll, tip, panel);
    mount.textContent = "";
    mount.append(rail, canvas);

    function sizeWindow() {
      RoadmapWindow.fit(mount);
      var last = scroll.lastElementChild;
      if (!last || !last.offsetHeight) return;
      // Room below the last pillar so that its heading can reach the top like every other.
      scroll.style.paddingBottom = Math.max(SIDE_PAD, scroll.clientHeight - last.offsetHeight - SIDE_PAD) + "px";
    }

    /* ---- Zoom ---- */

    /* The cards scale with CSS zoom on each pillar section, so the scroller keeps its size and position. */
    var ZOOM_MIN = 0.3, ZOOM_MAX = 1.5, ZOOM_STEP = 1.25, zoom = 1;
    function setZoom(k) {
      zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, k));
      scroll.style.setProperty("--cn-zoom", String(zoom));
      hideTip();
      sizeWindow();
    }
    canvas.appendChild(RoadmapWindow.zoomControls(function () { setZoom(zoom / ZOOM_STEP); },
      function () { setZoom(zoom * ZOOM_STEP); }, "Reset", function () { setZoom(1); scroll.scrollTop = 0; }));
    scroll.addEventListener("wheel", function (e) {
      if (!e.ctrlKey) return;
      e.preventDefault();
      setZoom(zoom * Math.exp(-e.deltaY * 0.01));
    }, { passive: false });

    /* ---- Tooltip ---- */

    function hideTip() { tip.classList.remove("is-open"); }
    function showTip(card) {
      var c = byKey[card.getAttribute("data-key")];
      if (!c || panel.matches(":hover")) return;
      tip.textContent = "";
      var head = el("div", "roadmap-tip-head");
      head.append(el("span", "roadmap-id", c.cand.key), el("span", null, c.cand.title));
      tip.append(head, el("div", "roadmap-tip-meta", originTitle[c.cand.origin] + " · " + c.pillar.title));
      if (c.cand.summary) tip.append(el("div", "roadmap-tip-summary", c.cand.summary));
      if (c.cand.serves) tip.append(el("div", "roadmap-tip-note", "Serves: " + c.cand.serves));
      tip.classList.add("is-open");
      RoadmapWindow.placeTip(tip, card.getBoundingClientRect(), canvas);
    }
    scroll.addEventListener("mouseover", function (e) {
      var a = e.target.closest && e.target.closest("a.cn-card");
      if (a) showTip(a);
    });
    scroll.addEventListener("mouseout", function (e) {
      var a = e.target.closest && e.target.closest("a.cn-card");
      if (a && !(e.relatedTarget && a.contains(e.relatedTarget))) hideTip();
    });
    scroll.addEventListener("focusin", function (e) {
      var a = e.target.closest && e.target.closest("a.cn-card");
      if (a) showTip(a);
    });
    scroll.addEventListener("focusout", hideTip);
    scroll.addEventListener("scroll", hideTip);

    /* ---- Scrolling ---- */

    /* Scrolls the card to the middle of the scroller. Only the scroller moves, never the page. */
    function centre(card) {
      var s = scroll.getBoundingClientRect(), r = card.getBoundingClientRect();
      scroll.scrollTop += r.top - s.top - (scroll.clientHeight - r.height) / 2;
    }
    function goPillar(prefix) {
      var sec = sections[prefix];
      if (!sec) return;
      hideTip();
      var pad = parseFloat(getComputedStyle(scroll).paddingTop) || 0;
      scroll.scrollTop += sec.getBoundingClientRect().top - scroll.getBoundingClientRect().top - pad;
    }

    /* ---- Panel ---- */

    function link(cls, href, text) {
      var a = el("a", cls, text);
      a.href = href;
      return a;
    }
    function row(label, value) {
      var r = el("div", "cn-panel-row");
      r.append(el("div", "cn-panel-label", label), value);
      return r;
    }
    function depItem(id) {
      var f = data.features[id], li = el("li", "cn-dep");
      li.setAttribute("data-id", id);
      li.appendChild(el("span", "roadmap-id", id));
      if (!f) return li;
      li.appendChild(link("cn-dep-page", new URL(f.url, jsonUrl).href, f.title));
      li.appendChild(el("span", "roadmap-badge roadmap-badge--" + f.status, BADGE_LABELS[f.status] || f.status));
      var g = link("cn-dep-graph", new URL(f.graph, jsonUrl).href, "Graph");
      g.setAttribute("aria-label", id + " in the roadmap graph");
      li.appendChild(g);
      return li;
    }
    function open(key) {
      var c = byKey[key];
      if (!c) return false;
      if (selected) selected.el.classList.remove("is-selected");
      selected = c;
      c.el.classList.add("is-selected");
      panel.textContent = "";
      var close = el("button", "cn-panel-close", "×");
      close.type = "button";
      close.setAttribute("aria-label", "Close");
      close.addEventListener("click", closePanel);
      var cand = c.cand;
      var pillar = link("cn-panel-pillar", new URL(c.pillar.url, jsonUrl).href, c.pillar.title);
      var origin = el("div", "cn-panel-value");
      origin.append(mark(cand.origin), document.createTextNode(originTitle[cand.origin]));
      var deps;
      if (cand.depends_on.length) {
        deps = el("ul", "cn-deps");
        cand.depends_on.forEach(function (id) { deps.appendChild(depItem(id)); });
      } else {
        deps = el("div", "cn-panel-value", "No registry features.");
      }
      panel.append(close, el("div", "cn-panel-title", cand.title), el("div", "cn-panel-key roadmap-id", cand.key),
        row("Pillar", pillar), row("Origin", origin),
        row("Summary", el("div", "cn-panel-value", cand.summary)), row("Serves", el("div", "cn-panel-value", cand.serves)),
        row("Depends On", deps));
      panel.scrollTop = 0;
      panel.classList.add("is-open");
      hideTip();
      if (history.replaceState) history.replaceState(null, "", "#" + key);
      return true;
    }
    function closePanel() {
      if (selected) selected.el.classList.remove("is-selected");
      selected = null;
      panel.classList.remove("is-open");
      if (history.replaceState) history.replaceState(null, "", location.pathname + location.search);
    }
    scroll.addEventListener("click", function (e) {
      var a = e.target.closest && e.target.closest("a.cn-card");
      if (!a || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      e.preventDefault();
      open(a.getAttribute("data-key"));
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && selected) closePanel();
    });

    /* ---- Search ---- */

    function updateCounts(per) {
      data.pillars.forEach(function (p) {
        var n = per ? (per[p.prefix] || 0) : null;
        var total = plural(p.candidates.length, "candidate", "candidates");
        pillarEls[p.prefix].querySelector(".roadmap-pillar-counts").textContent =
          n == null ? total : total + ", " + (n ? n + " match" : "none match");
      });
    }
    function setCurrent(i) {
      if (current >= 0 && matches[current]) matches[current].el.classList.remove("is-current");
      current = i;
      if (i < 0) return;
      matches[i].el.classList.add("is-current");
      centre(matches[i].el);
      sb.status.textContent = matches.length === 1 ? "1 match" : matches.length + " matches, " + (i + 1) + " of " + matches.length;
    }
    function clearSearch() {
      cards.forEach(function (c) { c.el.classList.remove("is-found", "is-current"); });
      scroll.classList.remove("is-searching");
      input.classList.remove("is-missing");
      sb.status.textContent = "";
      matches = []; current = -1;
      updateCounts(null);
    }
    function search(query) {
      lastQuery = query.trim().toLowerCase();
      var words = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
      if (!words.length) { clearSearch(); return []; }
      clearSearch();
      var per = {};
      matches = cards.filter(function (c) {
        var ok = words.every(function (w) { return c.hay.indexOf(w) >= 0; });
        c.el.classList.toggle("is-found", ok);
        if (ok) per[c.pillar.prefix] = (per[c.pillar.prefix] || 0) + 1;
        return ok;
      });
      scroll.classList.add("is-searching");
      updateCounts(per);
      input.classList.toggle("is-missing", !matches.length);
      if (!matches.length) { sb.status.textContent = "No match"; return []; }
      var q = query.trim().toLowerCase(), exact = -1;
      matches.forEach(function (c, i) { if (exact < 0 && c.cand.key.toLowerCase() === q) exact = i; });
      setCurrent(exact >= 0 ? exact : 0);
      return matches.map(function (c) { return c.cand.key; });
    }
    function step(d) {
      if (!matches.length) return null;
      setCurrent((current + d + matches.length) % matches.length);
      return matches[current].cand.key;
    }
    input.addEventListener("input", function () {
      clearTimeout(timer);
      timer = setTimeout(function () { search(input.value); }, SEARCH_DELAY);
    });
    input.addEventListener("keydown", function (e) {
      if (e.key !== "Enter") return;
      e.preventDefault();
      clearTimeout(timer);
      if (input.value.trim().toLowerCase() !== lastQuery) search(input.value);
      else step(e.shiftKey ? -1 : 1);
    });

    /* ---- Drawing ---- */

    function draw(d) {
      d.origins.forEach(function (o) {
        originTitle[o.origin] = o.title;
        var item = el("span");
        item.setAttribute("data-origin", o.origin);
        item.append(mark(o.origin), document.createTextNode(o.title));
        legend.appendChild(item);
      });
      var firstProposed = true;
      d.pillars.forEach(function (p) {
        var count = plural(p.candidates.length, "candidate", "candidates");
        if (p.proposed && firstProposed) { pillarList.appendChild(el("div", "roadmap-group", "Proposed Pillars")); firstProposed = false; }
        var a = RoadmapWindow.pillarLink({ href: new URL(p.url, jsonUrl).href, prefix: p.prefix, name: p.title,
          counts: count, onPick: goPillar });
        pillarEls[p.prefix] = a;
        pillarList.appendChild(a);

        var sec = el("section", "cn-pillar");
        sec.setAttribute("data-pillar", p.prefix);
        var head = el("div", "cn-pillar-head");
        head.setAttribute("role", "heading");
        head.setAttribute("aria-level", "2");
        head.append(el("span", "cn-pillar-title", p.title), el("span", "cn-pillar-count", count));
        if (p.proposed) head.appendChild(el("span", "cn-proposed", "Proposed Pillar"));
        sec.appendChild(head);
        if (p.proposed && p.description) sec.appendChild(el("p", "cn-pillar-desc", p.description));
        if (p.proposed && p.alternative) sec.appendChild(el("p", "cn-pillar-alt", "Alternative: " + p.alternative));
        d.origins.forEach(function (o) {
          var list = p.candidates.filter(function (c) { return c.origin === o.origin; });
          if (!list.length) return;
          var group = el("div", "cn-group");
          group.setAttribute("data-origin", o.origin);
          var label = el("div", "cn-group-label");
          label.append(mark(o.origin), document.createTextNode(o.title + " · " + list.length));
          var grid = el("div", "cn-grid");
          list.forEach(function (c) {
            var card = el("a", "cn-card");
            card.href = "#" + c.key;
            card.setAttribute("data-key", c.key);
            card.setAttribute("data-origin", c.origin);
            var k = el("span", "cn-card-key");
            k.append(mark(c.origin), document.createTextNode(c.key));
            card.append(k, el("span", "cn-card-title", c.title));
            grid.appendChild(card);
            var entry = { key: c.key, cand: c, pillar: p, el: card,
              hay: [c.key, c.title, c.summary, c.serves].join(" ").toLowerCase() };
            cards.push(entry);
            byKey[c.key] = entry;
          });
          group.append(label, grid);
          sec.appendChild(group);
        });
        sections[p.prefix] = sec;
        scroll.appendChild(sec);
      });
    }

    function fromHash() {
      var key = decodeURIComponent((location.hash || "").slice(1));
      if (key && byKey[key] && open(key)) centre(byKey[key].el);
    }
    window.addEventListener("hashchange", fromHash);
    window.addEventListener("resize", function () { if (data) sizeWindow(); });

    api.search = search;
    api.next = function () { return step(1); };
    api.select = open;
    api.close = closePanel;
    api.goPillar = goPillar;

    jsonUrl = new URL(mount.getAttribute("data-candidates") || "candidates.json", location.href).href;
    var fonts = document.fonts ? Promise.all([document.fonts.load("12px 'Plex Condensed'"),
      document.fonts.load("10px 'Source Code Pro'")]).catch(function () {}) : Promise.resolve();
    fonts.then(function () { return fetch(jsonUrl); })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(function (d) {
        data = d;
        api.data = d;
        draw(d);
        sizeWindow();
        fromHash();
        resolveReady();
      })
      .catch(function (err) {
        canvas.textContent = "";
        canvas.appendChild(el("p", "roadmap-error", "The candidates could not be loaded (" + err.message + ")."));
        resolveReady();
      });
  }

  function init() {
    var mount = document.getElementById("candidate-navigator");
    if (!mount || typeof RoadmapWindow === "undefined") return;
    start(mount);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
