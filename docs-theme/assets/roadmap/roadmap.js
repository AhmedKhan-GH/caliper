/* The roadmap graph: every feature as a node, laid out left to right by ELK's layered algorithm (shipped work on the left),
   then moved into one horizontal lane per pillar and drawn with Cytoscape.js. Plain ES2017, no build step. */
(function () {
  "use strict";

  var NODE_W = 200, NODE_H = 44, ROW = 52, LANE_PAD = 14, COLLAPSED_H = 100, MIN_LANE_H = 110, MAX_ROWS = 4, COL_GAP = 16, LAYER_GAP = 56, FIT_PAD = 24;
  var STATUS_LABEL = { shipped: "Shipped on main", built: "Built on a branch", progress: "In progress on a branch", planned: "Planned", idea: "Idea", decision: "Decision to settle" };
  var STATUS_RANK = { shipped: 0, built: 1, planned: 2, idea: 2, decision: 2 };
  var MEDIA = "not all and (max-width: 44.9375em)";  // the complement of the narrow-screen rule in roadmap.css

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function start(mount) {
    var graphUrl = new URL(mount.getAttribute("data-graph") || "graph.json", location.href);
    var started = false;
    var instance = null;
    var mq = window.matchMedia(MEDIA);

    function maybeStart() {
      if (!mq.matches) return;
      if (started) { if (instance) instance.resize(); return; }
      started = true;
      fetch(graphUrl).then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      }).then(function (data) {
        var fonts = document.fonts && document.fonts.load ? Promise.all([
          document.fonts.load('12px "Plex Condensed"'), document.fonts.load('11px "Source Code Pro"')
        ]) : Promise.resolve();
        return fonts.catch(function () {}).then(function () { return data; });
      }).then(function (data) {
        instance = build(mount, data);
      }).catch(function (err) {
        if (window.console) console.warn("Roadmap graph: " + err);
        mount.textContent = "";
        mount.appendChild(el("p", "roadmap-error", "The graph could not be loaded. The pillar list below links to every feature."));
      });
    }
    if (mq.addEventListener) mq.addEventListener("change", maybeStart); else mq.addListener(maybeStart);
    maybeStart();
  }

  function build(mount, data) {
    cytoscape.use(cytoscapeElk);
    mount.textContent = "";

    /* ---- Structure ---- */
    var toolbar = el("div", "roadmap-toolbar");
    var stage = el("div", "roadmap-stage");
    var bandsEl = el("div", "roadmap-bands");
    var cyEl = el("div", "roadmap-cy");
    var labelsEl = el("div", "roadmap-labels");
    var tip = el("div", "roadmap-tooltip");
    tip.setAttribute("role", "tooltip");
    stage.append(bandsEl, cyEl, labelsEl, tip);
    mount.append(toolbar, stage);

    var pillars = data.pillars.slice().sort(function (a, b) { return a.order - b.order; });
    var lanes = pillars.map(function (p) {
      return { pillar: p, prefix: p.prefix, collapsed: false, top: 0, height: 0, count: 0, nodes: [], slots: 1, band: null, label: null, toggle: null };
    });
    var laneOf = {};
    lanes.forEach(function (l) { laneOf[l.prefix] = l; });

    var nodeById = {};
    data.nodes.forEach(function (n) { nodeById[n.id] = n; });

    /* ---- Toolbar ---- */
    var search = el("div", "roadmap-search");
    var input = el("input");
    input.type = "search";
    input.placeholder = "Find a feature by id or title";
    input.setAttribute("aria-label", "Find a feature by id or title");
    input.setAttribute("list", "roadmap-suggestions");
    input.autocomplete = "off";
    var list = el("datalist");
    list.id = "roadmap-suggestions";
    data.nodes.forEach(function (n) {
      var o = el("option");
      o.value = n.id + " " + n.title;
      list.appendChild(o);
    });
    search.append(input, list);
    var zoomBox = el("div", "roadmap-zoom");
    var bIn = el("button", null, "+"), bOut = el("button", null, "−"), bFit = el("button", null, "Fit");
    bIn.type = bOut.type = bFit.type = "button";
    bIn.setAttribute("aria-label", "Zoom in");
    bOut.setAttribute("aria-label", "Zoom out");
    bFit.setAttribute("aria-label", "Fit the whole graph");
    zoomBox.append(bIn, bOut, bFit);
    var legend = el("div", "roadmap-legend");
    ["shipped", "built", "progress", "planned", "idea", "decision"].forEach(function (s) {
      var item = el("span");
      item.append(el("i", "roadmap-sw roadmap-sw--" + s), document.createTextNode(STATUS_LABEL[s]));
      legend.appendChild(item);
    });
    [["roadmap-ln", "Draft dependency"], ["roadmap-ln roadmap-ln--dep", "Depends on"], ["roadmap-ln roadmap-ln--unblocks", "Unblocks"]].forEach(function (d) {
      var item = el("span");
      item.append(el("i", d[0]), document.createTextNode(d[1]));
      legend.appendChild(item);
    });
    toolbar.append(search, zoomBox, legend);

    /* ---- Cytoscape elements ---- */
    var elements = [];
    data.nodes.forEach(function (n) {
      elements.push({ group: "nodes", classes: "feat s-" + n.status + (n.in_progress ? " wip" : ""), data: { id: n.id, title: n.title, status: n.status, pillar: n.pillar, label: n.title } });
      elements.push({ group: "nodes", classes: "ghost", data: { id: "g:" + n.id, label: n.id, of: n.id } });
    });
    data.edges.forEach(function (e) {
      elements.push({ group: "edges", classes: e.draft ? "draft" : "", data: { id: e.id, source: e.source, target: e.target } });
    });

    var cy = cytoscape({
      container: cyEl, elements: elements, style: styleFor(mount), minZoom: 0.03, maxZoom: 2.5,
      autoungrabify: true, boxSelectionEnabled: false, selectionType: "single", autounselectify: true
    });
    var feats = cy.nodes(".feat");
    var edges = cy.edges();
    function ghostOf(n) { return cy.getElementById("g:" + n.id()); }

    // Initial positions for the layout engine: any, the nodes just need a size.
    feats.forEach(function (n, i) { n.position({ x: 0, y: i * 10 }); });
    cy.nodes(".ghost").forEach(function (g) { g.position({ x: 0, y: 0 }); });

    var elkRun = feats.union(edges).layout({
      name: "elk", fit: false, animate: false, nodeDimensionsIncludeLabels: false,
      elk: {
        algorithm: "layered", "elk.direction": "RIGHT", "elk.partitioning.activate": "true", "elk.hierarchyHandling": "INCLUDE_CHILDREN",
        "elk.layered.spacing.nodeNodeBetweenLayers": "70", "elk.spacing.nodeNode": "14",
        "elk.spacing.edgeNode": "10", "elk.layered.spacing.edgeNodeBetweenLayers": "14"
      },
      nodeLayoutOptions: function (n) {
        return { "elk.partitioning.partition": String(STATUS_RANK[n.data("status")]) };
      }
    });
    var done = elkRun.promiseOn("layoutstop");
    elkRun.run();
    done.then(function () {
      arrange(true);
      bind();
      api.fit(false);
      cy.style().update();
    });

    /* ---- Lanes ---- */
    // Keep each node's x (its layer) and give it a slot inside its pillar's lane, in ELK's vertical order.
    function arrange(initial) {
      if (initial) {
        lanes.forEach(function (l) { l.nodes = []; l.count = 0; });
        var sorted = feats.toArray().sort(function (a, b) { return a.position("y") - b.position("y") || a.id().localeCompare(b.id()); });
        // Group by ELK layer (its x) inside each lane, then wrap each lane's layer into sub-columns of at most MAX_ROWS nodes.
        var layerXs = [];
        sorted.forEach(function (n) { var k = Math.round(n.position("x")); if (layerXs.indexOf(k) < 0) layerXs.push(k); });
        layerXs.sort(function (p, q) { return p - q; });
        var groups = {};
        sorted.forEach(function (n) {
          var lane = laneOf[n.data("pillar")];
          if (!lane) return;
          var li = layerXs.indexOf(Math.round(n.position("x")));
          var key = lane.prefix + "|" + li;
          (groups[key] = groups[key] || []).push(n);
          lane.nodes.push(n);
        });
        var widest = layerXs.map(function () { return 1; });
        Object.keys(groups).forEach(function (key) {
          var li = +key.split("|")[1];
          widest[li] = Math.max(widest[li], Math.ceil(groups[key].length / MAX_ROWS));
        });
        var layerStart = [], xx = 0;
        widest.forEach(function (w, i) { layerStart[i] = xx; xx += w * (NODE_W + COL_GAP) - COL_GAP + LAYER_GAP; });
        lanes.forEach(function (l) { l.slots = 1; });
        Object.keys(groups).forEach(function (key) {
          var parts = key.split("|"), lane = laneOf[parts[0]], li = +parts[1];
          groups[key].forEach(function (n, i) {
            var slot = i % MAX_ROWS, col = Math.floor(i / MAX_ROWS);
            n.scratch("rm", { lane: lane, slot: slot, x: layerStart[li] + NODE_W / 2 + col * (NODE_W + COL_GAP) });
            lane.slots = Math.max(lane.slots, slot + 1);
          });
        });
        lanes.forEach(function (l) { l.count = l.nodes.length; });
      }
      var y = 0;
      cy.batch(function () {
        lanes.forEach(function (l) {
          l.top = y;
          l.height = l.collapsed ? COLLAPSED_H : Math.max(MIN_LANE_H, LANE_PAD * 2 + l.slots * ROW - (ROW - NODE_H));
          l.nodes.forEach(function (n) {
            var s = n.scratch("rm");
            var py = l.top + LANE_PAD + s.slot * ROW + NODE_H / 2;
            n.position({ x: s.x, y: py });
            ghostOf(n).position({ x: s.x, y: py - 10 });
            var show = !l.collapsed;
            n.style("display", show ? "element" : "none");
            ghostOf(n).style("display", show ? "element" : "none");
          });
          y += l.height;
        });
      });
      renderLanes();
    }

    function renderLanes() {
      var z = cy.zoom(), pan = cy.pan(), H = stage.clientHeight;
      lanes.forEach(function (l) {
        if (!l.band) {
          l.band = el("div", "roadmap-band");
          bandsEl.appendChild(l.band);
          l.label = el("div", "roadmap-lane-label");
          l.label.setAttribute("data-lane", l.prefix);
          l.label.title = l.pillar.title;
          l.toggle = el("button", "roadmap-lane-toggle");
          l.toggle.type = "button";
          l.inner = el("div", "roadmap-lane-inner");
          var txt = el("div", "roadmap-lane-text");
          l.count_el = el("span", "roadmap-lane-count", l.count + (l.count === 1 ? " feature" : " features"));
          txt.append(el("span", "roadmap-lane-name", l.pillar.title), l.count_el);
          l.inner.append(l.toggle, txt);
          l.label.appendChild(l.inner);
          l.toggle.addEventListener("click", function () { setCollapsed(l, !l.collapsed); });
          labelsEl.appendChild(l.label);
        }
        var top = l.top * z + pan.y, h = l.height * z;
        l.band.style.top = top + "px";
        l.band.style.height = h + "px";
        // The label is the band's own left cell: same top and height, clipped, and it adapts to the height.
        l.label.style.top = top + "px";
        l.label.style.height = Math.max(0, h - 1) + "px";
        l.label.classList.toggle("is-collapsed", l.collapsed);
        l.label.classList.toggle("is-tall", h >= 56);
        l.label.classList.toggle("is-mid", h >= 30 && h < 56);
        l.label.classList.toggle("is-short", h >= 12 && h < 30);
        l.label.classList.toggle("is-tiny", h < 12);
        l.label.style.visibility = (top > H || top + h < 0) ? "hidden" : "visible";
        l.toggle.textContent = l.collapsed ? "+" : "\u2212";
        var verb = l.collapsed ? "Expand " : "Collapse ";
        l.toggle.setAttribute("aria-label", verb + l.pillar.title);
        l.toggle.setAttribute("aria-expanded", l.collapsed ? "false" : "true");
        l.toggle.title = verb + "the lane";
        // While the band's top is above the view, its content slides down to stay readable, never leaving the band.
        var shift = Math.max(0, Math.min(-top, h - 1 - l.inner.offsetHeight));
        l.inner.style.transform = shift > 0 ? "translateY(" + shift + "px)" : "";
      });
    }

    // Width the lane-label column covers on screen, plus a margin: the left edge of the usable view.
    function labelPad() { return lanes[0].label.offsetWidth + 16; }

    function setCollapsed(lane, value) {
      if (lane.collapsed === value) return;
      lane.collapsed = value;
      clearFocusIfHidden();
      hideTip();
      arrange(false);
    }

    /* ---- Focus ---- */
    var focused = null;
    function setFocus(n) {
      clearFocus();
      focused = n;
      var pred = n.predecessors(), succ = n.successors();
      var keepNodes = pred.nodes().union(succ.nodes()).union(n);
      cy.batch(function () {
        feats.forEach(function (m) {
          var keep = keepNodes.contains(m);
          m.toggleClass("faded", !keep);
          ghostOf(m).toggleClass("faded", !keep);
        });
        edges.forEach(function (e) { e.addClass("faded"); });
        pred.edges().removeClass("faded").addClass("dep");
        succ.edges().removeClass("faded").addClass("unblocks");
        n.addClass("focus");
      });
    }
    function clearFocus() {
      focused = null;
      cy.batch(function () {
        cy.elements().removeClass("faded dep unblocks focus");
      });
    }
    function clearFocusIfHidden() {
      if (focused && laneOf[focused.data("pillar")].collapsed) clearFocus();
    }

    /* ---- Tooltip ---- */
    var tipTimer = null, tipFor = null;
    function showTip(n) {
      clearTimeout(tipTimer);
      var d = nodeById[n.id()];
      if (!d) return;
      tipFor = n;
      tip.textContent = "";
      var head = el("div", "roadmap-tip-head");
      head.append(el("span", "roadmap-id", d.id), el("span", null, d.title));
      var meta = STATUS_LABEL[d.in_progress ? "progress" : d.status] + (d.sprint != null ? " · Sprint " + d.sprint : "");
      tip.append(head, el("div", "roadmap-tip-meta", meta));
      if (d.summary) tip.append(el("div", "roadmap-tip-summary", d.summary));
      if (d.draft) tip.append(el("p", "roadmap-tip-draft", "Draft. The dependencies of this feature are not settled."));
      var a = el("a", null, "Open the feature page");
      a.href = new URL(d.url, location.href).href;
      tip.append(a);
      tip.classList.add("is-open");
      var bb = n.renderedBoundingBox();
      var W = stage.clientWidth, H = stage.clientHeight, tw = tip.offsetWidth, th = tip.offsetHeight;
      var x = bb.x2 + 6;
      if (x + tw > W - 8) x = bb.x1 - tw - 6;
      x = Math.max(8, Math.min(x, W - tw - 8));
      var y = Math.max(8, Math.min(bb.y1, H - th - 8));
      tip.style.left = x + "px";
      tip.style.top = y + "px";
    }
    function hideTip() { clearTimeout(tipTimer); tipFor = null; tip.classList.remove("is-open"); }
    function hideTipSoon() { clearTimeout(tipTimer); tipTimer = setTimeout(hideTip, 220); }
    tip.addEventListener("mouseenter", function () { clearTimeout(tipTimer); });
    tip.addEventListener("mouseleave", hideTipSoon);

    /* ---- View ---- */
    function visibleExtent() {
      var minx = Infinity, maxx = -Infinity;
      feats.forEach(function (n) { var p = n.position("x"); minx = Math.min(minx, p - NODE_W / 2); maxx = Math.max(maxx, p + NODE_W / 2); });
      var last = lanes[lanes.length - 1];
      return { x1: minx, x2: maxx, y1: 0, y2: last.top + last.height };
    }
    var api = {
      cy: cy, lanes: lanes, setCollapsed: setCollapsed,
      fit: function (animate) {
        var b = visibleExtent(), W = stage.clientWidth, H = stage.clientHeight;
        var bw = b.x2 - b.x1, bh = b.y2 - b.y1;
        var lp = labelPad(), availW = W - lp - FIT_PAD, availH = H - FIT_PAD * 2 - 30;
        var z = Math.max(cy.minZoom(), Math.min(availW / bw, availH / bh, 1));
        var pan = { x: lp + (availW - bw * z) / 2 - b.x1 * z, y: FIT_PAD - b.y1 * z };
        if (animate) cy.animate({ zoom: z, pan: pan }, { duration: 250 }); else { cy.zoom(z); cy.pan(pan); }
      },
      // Bring a set of nodes into view at a readable zoom (at least 1): fit the set if it fits at that zoom, else centre the focus node.
      show: function (set, n) {
        var W = stage.clientWidth, H = stage.clientHeight, bb = set.boundingBox();
        var lp = labelPad(), availW = W - lp - FIT_PAD, availH = H - FIT_PAD * 2;
        var zf = Math.min(availW / (bb.w + 40), availH / (bb.h + 40), 1.3);
        var z, cx, cyy;
        if (zf >= 1) { z = zf; cx = (bb.x1 + bb.x2) / 2; cyy = (bb.y1 + bb.y2) / 2; }
        else { z = Math.min(cy.maxZoom(), Math.max(cy.zoom(), 1)); var p = n.position(); cx = p.x; cyy = p.y; }
        cy.animate({ zoom: z, pan: { x: lp + availW / 2 - cx * z, y: H / 2 - cyy * z } }, { duration: 300 });
      },
      resize: function () { cy.resize(); renderLanes(); },
      focusId: function (id) {
        var n = cy.getElementById(id);
        if (!n.nonempty() || !n.hasClass("feat")) return;
        var lane = laneOf[n.data("pillar")];
        if (lane.collapsed) setCollapsed(lane, false);
        setFocus(n);
        api.show(n.predecessors().nodes().union(n.successors().nodes()).union(n).filter(".feat"), n);
      }
    };
    mount.__roadmap = api;

    /* ---- Events ---- */
    function bind() {
      var raf = 0;
      cy.on("viewport", function () {
        hideTip();
        if (!raf) raf = requestAnimationFrame(function () { raf = 0; renderLanes(); });
      });
      cy.on("mouseover", "node.feat", function (e) { stage.style.cursor = "pointer"; showTip(e.target); });
      cy.on("mouseout", "node.feat", function () { stage.style.cursor = ""; hideTipSoon(); });
      // A tap on a feature focuses it; a tap anywhere else, an edge included, clears the focus. The
      // camera moves to the focus only once a double-click can no longer follow, so that the second
      // click lands on the same feature, and a double-click opens the feature of its first click.
      var DBL_MS = 400, lastTap = null, showTimer = 0;
      cy.on("tap", function (e) {
        var t = e.target, now = Date.now(), p = e.renderedPosition || { x: 0, y: 0 };
        if (lastTap && now - lastTap.time < DBL_MS && Math.abs(p.x - lastTap.x) < 8 && Math.abs(p.y - lastTap.y) < 8) return;
        clearTimeout(showTimer);
        if (t !== cy && t.isNode() && t.hasClass("feat")) {
          lastTap = { node: t, time: now, x: p.x, y: p.y };
          setFocus(t);
          showTimer = setTimeout(function () {
            api.show(t.predecessors().nodes().union(t.successors().nodes()).union(t).filter(".feat"), t);
          }, DBL_MS);
        } else {
          lastTap = null;
          clearFocus();
        }
      });
      cy.on("dbltap", function (e) {
        var t = lastTap && Date.now() - lastTap.time < 2 * DBL_MS ? lastTap.node : e.target;
        if (t === cy || !t.isNode() || !t.hasClass("feat")) return;
        clearTimeout(showTimer);
        var d = nodeById[t.id()];
        if (d) location.assign(new URL(d.url, location.href).href);
      });
      var zoomBy = function (f) {
        cy.animate({ zoom: { level: Math.max(cy.minZoom(), Math.min(cy.maxZoom(), cy.zoom() * f)), renderedPosition: { x: (stage.clientWidth + labelPad()) / 2, y: stage.clientHeight / 2 } } }, { duration: 150 });
      };
      bIn.addEventListener("click", function () { zoomBy(1.35); });
      bOut.addEventListener("click", function () { zoomBy(1 / 1.35); });
      bFit.addEventListener("click", function () { api.fit(true); });

      function find(q) {
        q = q.trim();
        if (!q) return null;
        var low = q.toLowerCase(), i;
        for (i = 0; i < data.nodes.length; i++) if (data.nodes[i].id.toLowerCase() === low) return data.nodes[i];
        var first = low.split(/\s+/)[0];
        for (i = 0; i < data.nodes.length; i++) if ((data.nodes[i].id + " " + data.nodes[i].title).toLowerCase() === low) return data.nodes[i];
        for (i = 0; i < data.nodes.length; i++) if (data.nodes[i].id.toLowerCase() === first && low.indexOf(" ") > 0 && low.indexOf(data.nodes[i].title.toLowerCase()) > 0) return data.nodes[i];
        for (i = 0; i < data.nodes.length; i++) if (data.nodes[i].title.toLowerCase().indexOf(low) >= 0) return data.nodes[i];
        return null;
      }
      var lastQuery = "";
      function go() {
        if (input.value === lastQuery) return;
        lastQuery = input.value;
        var n = find(input.value);
        input.setCustomValidity(n || !input.value.trim() ? "" : "No feature matches.");
        if (n) api.focusId(n.id);
      }
      input.addEventListener("keydown", function (e) { if (e.key === "Enter") { e.preventDefault(); lastQuery = ""; go(); } });
      input.addEventListener("input", function (e) {
        // A suggestion chosen from the list fills in the full "ID Title" text.
        var v = input.value;
        if (!e.inputType || e.inputType === "insertReplacementText") { if (find(v) && data.nodes.some(function (n) { return n.id + " " + n.title === v; })) go(); }
      });
      input.addEventListener("change", go);

      new MutationObserver(function () { cy.style().fromJson(styleFor(mount)).update(); })
        .observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
      new MutationObserver(function () { cy.style().fromJson(styleFor(mount)).update(); })
        .observe(document.documentElement, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
      window.addEventListener("resize", function () { cy.resize(); renderLanes(); });
      if (window.ResizeObserver) new ResizeObserver(function () { cy.resize(); renderLanes(); }).observe(stage);
    }
    return api;
  }

  /* ---- Style, read from the custom properties of the current scheme ---- */
  function styleFor(mount) {
    var cs = getComputedStyle(mount);
    function v(name) { return cs.getPropertyValue(name).trim(); }
    var c = {
      shipped: v("--rm-shipped"), built: v("--rm-built"), planned: v("--rm-planned"), idea: v("--rm-idea"), decision: v("--rm-decision"),
      dep: v("--rm-dep"), unb: v("--rm-unblocks"), text: v("--rm-text"), dim: v("--rm-text-dim"), edge: v("--rm-edge")
    };
    var text = '"Plex Condensed", sans-serif', code = '"Source Code Pro", monospace';
    // Opaque fills (the status colour mixed into the lane colour) so that no edge shows through a node's text.
    var lane = v("--rm-lane-b");
    function mix(col, over, t) {
      var m = /^#([0-9a-f]{6})$/i, a = m.exec(col), b = m.exec(over);
      if (!a || !b) return col;
      var out = "#";
      for (var i = 0; i < 3; i++) {
        var x = parseInt(a[1].substr(i * 2, 2), 16), y = parseInt(b[1].substr(i * 2, 2), 16);
        out += ("0" + Math.round(x * t + y * (1 - t)).toString(16)).slice(-2);
      }
      return out;
    }
    function tinted(col) { return { "background-color": mix(col, lane, 0.3), "background-opacity": 1, "border-color": col, "border-width": 2, "border-style": "solid" }; }
    return [
      { selector: "node.feat", style: {
        shape: "round-rectangle", width: NODE_W, height: NODE_H, label: "data(title)", color: c.text, "font-family": text, "font-size": 12,
        "text-valign": "center", "text-halign": "center", "text-margin-y": 8, "text-wrap": "ellipsis", "text-max-width": NODE_W - 16,
        "min-zoomed-font-size": 6, "border-opacity": 1, "z-index-compare": "manual", "z-index": 20
      } },
      { selector: "node.s-shipped", style: tinted(c.shipped) },
      { selector: "node.s-built", style: tinted(c.built) },
      // Built work whose branches are unfinished: the built colour, a paler fill and a dashed outline.
      { selector: "node.s-built.wip", style: { "background-color": mix(c.built, lane, 0.12), "border-style": "dashed", "border-width": 2.5 } },
      { selector: "node.s-planned", style: tinted(c.planned) },
      { selector: "node.s-idea", style: { "background-color": lane, "background-opacity": 1, "border-color": c.idea, "border-width": 1.5, "border-style": "dashed" } },
      { selector: "node.s-decision", style: { shape: "cut-rectangle", "background-color": lane, "background-opacity": 1, "border-color": c.decision, "border-width": 2, "border-style": "solid" } },
      { selector: "node.ghost", style: {
        width: 1, height: 1, "background-opacity": 0, "border-width": 0, events: "no", "z-index-compare": "manual", "z-index": 21, label: "data(label)", color: c.dim, "font-family": code,
        "font-size": 11, "text-valign": "center", "text-halign": "center", "min-zoomed-font-size": 6
      } },
      { selector: "edge", style: {
        width: 1.2, "line-color": c.edge, "target-arrow-color": c.edge, "target-arrow-shape": "triangle", "arrow-scale": 0.8,
        "z-index-compare": "manual", "z-index": 1, "curve-style": "taxi", "taxi-direction": "rightward", "taxi-turn": 24, "taxi-turn-min-distance": 6, opacity: 0.75
      } },
      { selector: "edge.draft", style: { "line-style": "dotted", width: 1.6 } },
      { selector: "edge.dep", style: { "line-color": c.dep, "target-arrow-color": c.dep, width: 2.2, opacity: 1, "z-index": 10 } },
      { selector: "edge.unblocks", style: { "line-color": c.unb, "target-arrow-color": c.unb, width: 2.2, opacity: 1, "z-index": 10 } },
      { selector: "node.focus", style: { "border-width": 3.5 } },
      { selector: "node.faded", style: { opacity: 0.25 } },
      { selector: "edge.faded", style: { opacity: 0.15 } }
    ];
  }

  function init() {
    var mount = document.getElementById("roadmap-graph");
    if (!mount || typeof cytoscape === "undefined" || typeof cytoscapeElk === "undefined") return;
    start(mount);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
