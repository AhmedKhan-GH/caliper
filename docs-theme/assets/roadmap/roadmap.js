/* The roadmap graph: nine pillar graphs stacked on one canvas, each laid out by ELK's layered
   algorithm on its own and drawn in SVG with ELK's own edge routes. A dependency on a feature of
   another pillar is drawn as a reference node inside the dependent's pillar, so no line leaves its
   pillar. Plain ES2017 and elkjs, no build step. */
(function () {
  "use strict";

  var NODE_W = 184, REF_W = 160, PAD_X = 10, PAD_TOP = 7, ID_LINE = 13, ID_GAP = 2, TITLE_LINE = 15,
    PAD_BOTTOM = 8, MARGIN = 24, HEADING_H = 38, PILLAR_GAP = 72, CORNER = 6;
  var ZOOM_MIN = 0.75, ZOOM_MAX = 1.5;
  var TEXT_FONT = '"Plex Condensed", sans-serif', CODE_FONT = '"Source Code Pro", monospace';
  var ELK_OPTIONS = {
    "elk.algorithm": "layered",
    "elk.direction": "RIGHT",
    "elk.edgeRouting": "ORTHOGONAL",
    "elk.layered.layering.strategy": "NETWORK_SIMPLEX",
    "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
    "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
    // More crossing-minimisation sweeps than the default 7: the Rendering pillar, with its many
    // reference nodes, needs them to stay under 0.6 crossings per edge.
    "elk.layered.thoroughness": "50",
    // One layered graph per pillar, unconnected features included, so that a small component is not
    // packed into a row of its own beside the main one and the pillar opens on its own features.
    "elk.separateConnectedComponents": "false",
    "elk.spacing.nodeNode": "12",
    "elk.layered.spacing.nodeNodeBetweenLayers": "56",
    "elk.layered.spacing.edgeEdgeBetweenLayers": "10",
    "elk.layered.spacing.edgeNodeBetweenLayers": "16",
    "elk.padding": "[top=0,left=0,bottom=0,right=0]"
  };
  var STATUS_LABEL = { shipped: "Shipped on main", built: "Built on a branch", progress: "In progress on a branch",
    planned: "Planned", idea: "Idea", decision: "Decision to settle" };
  var SVG = "http://www.w3.org/2000/svg";

  var el = window.RoadmapWindow && RoadmapWindow.el;
  function sv(tag, attrs) {
    var e = document.createElementNS(SVG, tag);
    for (var k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    return e;
  }
  function idKey(id) {
    var parts = id.split("-");
    return [parts[0], parseInt(parts[1], 10) || 0];
  }
  function byId(a, b) {
    var x = idKey(a), y = idKey(b);
    return x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : x[1] - y[1];
  }
  function shown(n) { return n.status === "built" && n.in_progress ? "progress" : n.status; }

  /* ---- Measuring labels ---- */

  var measureCtx = null;
  function textWidth(text, font) {
    if (!measureCtx) measureCtx = document.createElement("canvas").getContext("2d");
    measureCtx.font = font;
    return measureCtx.measureText(text).width;
  }
  /* Greedy word wrap into lines no wider than max; a word wider than a line is split by characters. */
  function wrap(text, max) {
    var font = "12px " + TEXT_FONT, lines = [], line = "";
    text.split(/\s+/).forEach(function (word) {
      var next = line ? line + " " + word : word;
      if (textWidth(next, font) <= max) { line = next; return; }
      if (line) lines.push(line);
      line = "";
      while (textWidth(word, font) > max) {
        var cut = word.length - 1;
        while (cut > 1 && textWidth(word.slice(0, cut), font) > max) cut--;
        lines.push(word.slice(0, cut));
        word = word.slice(cut);
      }
      line = word;
    });
    if (line) lines.push(line);
    return lines;
  }
  function nodeSize(title, width) {
    var lines = wrap(title, width - 2 * PAD_X);
    return { lines: lines, width: width, height: PAD_TOP + ID_LINE + ID_GAP + lines.length * TITLE_LINE + PAD_BOTTOM };
  }

  /* ---- Layout ---- */

  /* The ELK graph of one pillar: its features, one reference node per foreign feature that one of
     them depends on, and the edges that end at its features. Input is sorted by id, which with
     considerModelOrder keeps the layout stable as features are added. */
  function pillarGraph(pillar, data, byNode) {
    var mine = data.nodes.filter(function (n) { return n.pillar === pillar.prefix; })
      .sort(function (a, b) { return byId(a.id, b.id); });
    var inPillar = {};
    mine.forEach(function (n) { inPillar[n.id] = true; });
    var edges = data.edges.filter(function (e) { return inPillar[e.target]; })
      .sort(function (a, b) { return byId(a.target, b.target) || byId(a.source, b.source); });
    var refIds = [];
    edges.forEach(function (e) { if (!inPillar[e.source] && refIds.indexOf(e.source) < 0) refIds.push(e.source); });
    refIds.sort(byId);
    var nodes = [], children = [];
    refIds.forEach(function (id) {
      var real = byNode[id];
      var size = nodeSize(real.title, REF_W);
      var key = "ref:" + pillar.prefix + ":" + id;
      nodes.push({ key: key, id: id, ref: true, of: id, pillar: pillar.prefix, lines: size.lines, width: size.width, height: size.height });
      children.push({ id: key, width: size.width, height: size.height });
    });
    mine.forEach(function (n) {
      var size = nodeSize(n.title, NODE_W);
      nodes.push({ key: n.id, id: n.id, ref: false, of: null, pillar: pillar.prefix, lines: size.lines, width: size.width, height: size.height });
      children.push({ id: n.id, width: size.width, height: size.height });
    });
    var drawn = edges.map(function (e) {
      var from = inPillar[e.source] ? e.source : "ref:" + pillar.prefix + ":" + e.source;
      return { id: e.id, source: e.source, target: e.target, from: from, to: e.target, pillar: pillar.prefix, draft: !!e.draft };
    });
    return {
      nodes: nodes, edges: drawn,
      elk: { id: "pillar-" + pillar.prefix, layoutOptions: ELK_OPTIONS, children: children,
        edges: drawn.map(function (e) { return { id: e.id, sources: [e.from], targets: [e.to] }; }) }
    };
  }

  /* Lays out every pillar on its own and stacks the sections vertically, left-aligned. */
  function layoutGraph(data) {
    var elk = new ELK();
    var byNode = {};
    data.nodes.forEach(function (n) { byNode[n.id] = n; });
    var parts = data.pillars.map(function (p) { return pillarGraph(p, data, byNode); });
    return Promise.all(parts.map(function (part) { return elk.layout(part.elk); })).then(function (results) {
      var layout = { pillars: [], nodes: [], edges: [], width: 0, height: 0 };
      var y = MARGIN;
      results.forEach(function (res, i) {
        var pillar = data.pillars[i], part = parts[i];
        var ox = MARGIN, oy = y + HEADING_H;
        var pos = {};
        res.children.forEach(function (c) { pos[c.id] = c; });
        part.nodes.forEach(function (n) {
          var c = pos[n.key];
          n.x = ox + c.x; n.y = oy + c.y; n.width = c.width; n.height = c.height;
          layout.nodes.push(n);
        });
        var routes = {};
        (res.edges || []).forEach(function (e) { routes[e.id] = e; });
        part.edges.forEach(function (e) {
          var pts = [];
          (routes[e.id].sections || []).forEach(function (s) {
            [s.startPoint].concat(s.bendPoints || [], [s.endPoint]).forEach(function (p) {
              var q = [ox + p.x, oy + p.y], last = pts[pts.length - 1];
              if (!last || last[0] !== q[0] || last[1] !== q[1]) pts.push(q);
            });
          });
          e.points = pts;
          layout.edges.push(e);
        });
        var headingW = textWidth(pillar.title, "15px " + TEXT_FONT) + 8;
        var w = Math.max(res.width || 0, headingW), h = HEADING_H + (res.height || 0);
        layout.pillars.push({ prefix: pillar.prefix, title: pillar.title, x: ox, y: y, width: w, height: h });
        layout.width = Math.max(layout.width, ox + w + MARGIN);
        y += h + PILLAR_GAP;
      });
      layout.height = y - PILLAR_GAP + MARGIN;
      return layout;
    });
  }

  /* ---- Drawing ---- */

  /* An orthogonal polyline with rounded corners of at most CORNER px. */
  function roundedPath(pts) {
    var d = "M" + pts[0][0] + " " + pts[0][1];
    for (var i = 1; i < pts.length - 1; i++) {
      var p = pts[i - 1], q = pts[i], r = pts[i + 1];
      var l1 = Math.hypot(q[0] - p[0], q[1] - p[1]), l2 = Math.hypot(r[0] - q[0], r[1] - q[1]);
      var c = Math.min(CORNER, l1 / 2, l2 / 2);
      if (c < 0.5) { d += " L" + q[0] + " " + q[1]; continue; }
      var a = [q[0] - (q[0] - p[0]) / l1 * c, q[1] - (q[1] - p[1]) / l1 * c];
      var b = [q[0] + (r[0] - q[0]) / l2 * c, q[1] + (r[1] - q[1]) / l2 * c];
      d += " L" + a[0] + " " + a[1] + " Q" + q[0] + " " + q[1] + " " + b[0] + " " + b[1];
    }
    var last = pts[pts.length - 1];
    return d + " L" + last[0] + " " + last[1];
  }

  function shapeFor(n, status) {
    return sv("rect", { "class": "rm-shape", x: n.x, y: n.y, width: n.width, height: n.height, rx: 4, ry: 4 });
  }

  function drawLabel(group, n) {
    var id = sv("text", { "class": "rm-id", x: n.x + PAD_X, y: n.y + PAD_TOP + 10 });
    id.textContent = n.id;
    var title = sv("text", { "class": "rm-title", x: n.x + PAD_X, y: n.y + PAD_TOP + ID_LINE + ID_GAP + 11 });
    n.lines.forEach(function (line, i) {
      var t = sv("tspan", { x: n.x + PAD_X, dy: i ? TITLE_LINE : 0 });
      t.textContent = line;
      title.appendChild(t);
    });
    group.appendChild(id);
    group.appendChild(title);
  }

  function draw(svg, data, layout) {
    var byNode = {};
    data.nodes.forEach(function (n) { byNode[n.id] = n; });
    var defs = sv("defs", {});
    var marker = sv("marker", { id: "rm-arrow", viewBox: "0 0 8 8", refX: 7.5, refY: 4, markerWidth: 7, markerHeight: 7,
      markerUnits: "userSpaceOnUse", orient: "auto" });
    marker.appendChild(sv("path", { d: "M0 0.8 L7.5 4 L0 7.2 z", "class": "rm-arrowhead" }));
    defs.appendChild(marker);
    var stripes = sv("pattern", { id: "rm-stripes", width: 8, height: 8, patternUnits: "userSpaceOnUse",
      patternTransform: "rotate(45)" });
    stripes.appendChild(sv("rect", { "class": "rm-stripe-base", width: 8, height: 8 }));
    stripes.appendChild(sv("rect", { "class": "rm-stripe", width: 3.5, height: 8 }));
    defs.appendChild(stripes);
    svg.appendChild(defs);
    var world = sv("g", { "class": "rm-world" });
    svg.appendChild(world);
    var nodeEls = {}, edgeEls = {};
    layout.pillars.forEach(function (sec) {
      var g = sv("g", { "class": "rm-pillar", "data-pillar": sec.prefix });
      var heading = sv("text", { "class": "rm-heading", x: sec.x, y: sec.y + 18 });
      heading.textContent = sec.title;
      g.appendChild(heading);
      var edgesG = sv("g", { "class": "rm-edges" }), nodesG = sv("g", { "class": "rm-nodes" });
      layout.edges.forEach(function (e) {
        if (e.pillar !== sec.prefix) return;
        var path = sv("path", { "class": "rm-edge" + (e.draft ? " is-draft" : ""), d: roundedPath(e.points),
          "marker-end": "url(#rm-arrow)", "data-id": e.id, "data-source": e.source, "data-target": e.target,
          "data-from": e.from, "data-to": e.to });
        edgesG.appendChild(path);
        edgeEls[e.id] = path;
      });
      layout.nodes.forEach(function (n) {
        if (n.pillar !== sec.prefix) return;
        var real = byNode[n.id], a;
        if (n.ref) {
          a = sv("a", { "class": "rm-ref", href: "#" + n.id, "data-of": n.id, "data-key": n.key });
          a.appendChild(sv("rect", { "class": "rm-shape", x: n.x, y: n.y, width: n.width, height: n.height, rx: 4, ry: 4 }));
          a.setAttribute("aria-label", n.id + " " + real.title + ", in another pillar");
        } else {
          var status = shown(real);
          a = sv("a", { "class": "rm-node s-" + status, href: real.url, "data-id": n.id });
          a.appendChild(shapeFor(n, status));
        }
        drawLabel(a, Object.assign({}, n, { title: real.title }));
        nodesG.appendChild(a);
        nodeEls[n.key] = a;
      });
      g.appendChild(edgesG);
      g.appendChild(nodesG);
      world.appendChild(g);
    });
    return { world: world, nodeEls: nodeEls, edgeEls: edgeEls };
  }

  /* ---- The page ---- */

  function counts(data, prefix) {
    var c = { shipped: 0, built: 0, progress: 0, planned: 0, idea: 0, decision: 0 };
    data.nodes.forEach(function (n) { if (n.pillar === prefix) c[shown(n)]++; });
    var parts = [];
    if (c.shipped) parts.push(c.shipped + " shipped");
    if (c.built) parts.push(c.built + " built");
    if (c.progress) parts.push(c.progress + " in progress");
    if (c.planned) parts.push(c.planned + " planned");
    if (c.idea) parts.push(c.idea + (c.idea === 1 ? " idea" : " ideas"));
    if (c.decision) parts.push(c.decision + (c.decision === 1 ? " decision" : " decisions"));
    return parts.join(", ");
  }

  function start(mount) {
    var data = null, layout = null, els = null;
    var view = { x: 0, y: 0, k: 1 };
    var api = { zoomRange: [ZOOM_MIN, ZOOM_MAX], layoutGraph: layoutGraph };
    var resolveReady;
    api.ready = new Promise(function (r) { resolveReady = r; });
    mount.__roadmap = api;

    /* Rail: legend, search and the pillar list. */
    var rail = el("div", "roadmap-rail");
    var legend = el("div", "roadmap-legend");
    ["shipped", "built", "progress", "planned", "idea", "decision"].forEach(function (s) {
      var item = el("span"), sw = el("i", "roadmap-sw roadmap-sw--" + s);
      item.append(sw, document.createTextNode(STATUS_LABEL[s]));
      legend.appendChild(item);
    });
    [["roadmap-sw roadmap-sw--ref", "Feature in another pillar"], ["roadmap-ln", "Depends on"],
      ["roadmap-ln roadmap-ln--draft", "Draft dependency"]].forEach(function (d) {
      var item = el("span");
      item.append(el("i", d[0]), document.createTextNode(d[1]));
      legend.appendChild(item);
    });
    var sb = RoadmapWindow.searchBox("Find a feature by id or title", "Find a feature by id or title, then press Enter");
    var input = sb.input;
    var pillarList = el("div", "roadmap-pillars");
    /* Candidates: ideas kept outside the registry, on their own pages under candidates/. */
    var cand = RoadmapWindow.railLink("candidates/", "Candidates", "Ideas outside the registry, by pillar");
    rail.append(legend, sb.box, sb.status, pillarList, cand);

    var canvas = el("div", "roadmap-canvas");
    var svg = sv("svg", { "class": "roadmap-svg", role: "img", "aria-label": "Roadmap graph of every feature by pillar" });
    var tip = el("div", "roadmap-tooltip");
    tip.setAttribute("role", "tooltip");
    canvas.append(svg, tip);
    mount.textContent = "";
    mount.append(rail, canvas);

    function sizeCanvas() { RoadmapWindow.fit(mount); }

    function apply() {
      els.world.setAttribute("transform", "translate(" + view.x + " " + view.y + ") scale(" + view.k + ")");
    }
    /* Keeps the graph in sight: its top-left corner stays at or above the canvas's, and its far edges
       may come in as far as the middle of the canvas, so that every pillar can reach the top. */
    function clampView() {
      var cw = canvas.clientWidth, ch = canvas.clientHeight;
      var w = layout.width * view.k, h = layout.height * view.k;
      view.x = Math.min(0, Math.max(view.x, Math.min(0, cw / 2 - w)));
      view.y = Math.min(0, Math.max(view.y, Math.min(0, ch / 2 - h)));
    }
    function setView(x, y, k) {
      view.k = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, k));
      view.x = x; view.y = y;
      clampView();
      apply();
    }
    api.setView = setView;
    api.view = function () { return { x: view.x, y: view.y, k: view.k }; };

    var found = null;
    function focus(id) {
      var n = layout && layout.nodes.filter(function (m) { return m.key === id; })[0];
      if (!n) return false;
      if (found) found.classList.remove("is-found");
      found = els.nodeEls[n.key];
      found.classList.add("is-found");
      var cw = canvas.clientWidth, ch = canvas.clientHeight;
      setView(cw / 2 - view.k * (n.x + n.width / 2), ch / 2 - view.k * (n.y + n.height / 2), view.k);
      return true;
    }
    api.focus = focus;

    function goPillar(prefix) {
      var sec = layout.pillars.filter(function (p) { return p.prefix === prefix; })[0];
      if (sec) setView(MARGIN - view.k * sec.x, MARGIN - view.k * sec.y, view.k);
    }

    /* Search: an exact id, then the first title in registry order that holds the text. */
    input.addEventListener("keydown", function (e) {
      if (e.key !== "Enter") return;
      e.preventDefault();
      var q = input.value.trim().toLowerCase();
      if (!q || !data) return;
      var hit = data.nodes.filter(function (n) { return n.id.toLowerCase() === q; })[0] ||
        data.nodes.filter(function (n) { return n.title.toLowerCase().indexOf(q) >= 0; })[0];
      input.classList.toggle("is-missing", !hit);
      if (hit) focus(hit.id);
    });

    /* Pan with the wheel and by dragging; zoom with ctrl-wheel or a pinch, around the pointer. */
    canvas.addEventListener("wheel", function (e) {
      if (!layout) return;
      e.preventDefault();
      var unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? canvas.clientHeight : 1;
      if (e.ctrlKey) {
        var r = canvas.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
        var k = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, view.k * Math.exp(-e.deltaY * unit * 0.01)));
        setView(px - (px - view.x) * k / view.k, py - (py - view.y) * k / view.k, k);
      } else {
        var dx = e.deltaX * unit, dy = e.deltaY * unit;
        if (e.shiftKey && !dx) { dx = dy; dy = 0; }
        setView(view.x - dx, view.y - dy, view.k);
      }
      hideTip();
    }, { passive: false });
    var drag = null, dragged = false;
    canvas.addEventListener("pointerdown", function (e) {
      if (e.button !== 0 || !layout) return;
      drag = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, id: e.pointerId };
      dragged = false;
    });
    canvas.addEventListener("pointermove", function (e) {
      if (!drag || e.pointerId !== drag.id) return;
      var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (!dragged && Math.abs(dx) + Math.abs(dy) > 4) {
        dragged = true;
        canvas.setPointerCapture(e.pointerId);
        canvas.classList.add("is-dragging");
        hideTip();
      }
      if (dragged) setView(drag.vx + dx, drag.vy + dy, view.k);
    });
    function endDrag(e) {
      if (!drag || e.pointerId !== drag.id) return;
      drag = null;
      canvas.classList.remove("is-dragging");
    }
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
    /* A drag that ends on a node is not a click. */
    svg.addEventListener("click", function (e) {
      if (dragged) { e.preventDefault(); e.stopPropagation(); dragged = false; return; }
      var ref = e.target.closest && e.target.closest("a.rm-ref");
      if (ref) {
        e.preventDefault();
        var id = ref.getAttribute("data-of");
        focus(id);
        if (history.replaceState) history.replaceState(null, "", "#" + id);
      }
    }, true);

    /* Hover: the feature's own lines thicken and the tooltip shows its summary, sprint and branches. */
    var hot = [];
    function hideTip() {
      tip.classList.remove("is-open");
      hot.forEach(function (p) { p.classList.remove("is-hot"); });
      hot = [];
    }
    function showTip(a) {
      hideTip();
      var key = a.getAttribute("data-id") || a.getAttribute("data-key");
      var id = a.getAttribute("data-id") || a.getAttribute("data-of");
      var n = data.nodes.filter(function (m) { return m.id === id; })[0];
      if (!n) return;
      layout.edges.forEach(function (e) {
        if (e.from === key || e.to === key) { var p = els.edgeEls[e.id]; p.classList.add("is-hot"); hot.push(p); }
      });
      tip.textContent = "";
      var head = el("div", "roadmap-tip-head");
      head.append(el("span", "roadmap-id", n.id), el("span", null, n.title));
      var meta = STATUS_LABEL[shown(n)] + (n.sprint != null ? " · Sprint " + n.sprint : "");
      tip.append(head, el("div", "roadmap-tip-meta", meta));
      if (n.summary) tip.append(el("div", "roadmap-tip-summary", n.summary));
      if (n.branches && n.branches.length) tip.append(el("div", "roadmap-tip-branches", "Branches: " + n.branches.join(", ")));
      if (a.classList.contains("rm-ref")) tip.append(el("div", "roadmap-tip-note", "In another pillar. Click to go to it."));
      tip.classList.add("is-open");
      RoadmapWindow.placeTip(tip, a.getBoundingClientRect(), canvas);
    }
    svg.addEventListener("mouseover", function (e) {
      if (drag && dragged) return;
      var a = e.target.closest && e.target.closest("a.rm-node, a.rm-ref");
      if (a) showTip(a);
    });
    svg.addEventListener("mouseout", function (e) {
      var a = e.target.closest && e.target.closest("a.rm-node, a.rm-ref");
      if (a && !(e.relatedTarget && a.contains(e.relatedTarget))) hideTip();
    });

    function fromHash() {
      var id = decodeURIComponent((location.hash || "").slice(1)).toUpperCase();
      if (id && /^[A-Z]{3}-\d+$/.test(id)) focus(id);
    }
    window.addEventListener("hashchange", fromHash);
    window.addEventListener("resize", function () { if (layout) { sizeCanvas(); setView(view.x, view.y, view.k); } });

    var url = new URL(mount.getAttribute("data-graph") || "graph.json", location.href).href;
    var fonts = document.fonts ? Promise.all([document.fonts.load("12px " + TEXT_FONT), document.fonts.load("10px " + CODE_FONT),
      document.fonts.load("15px " + TEXT_FONT)]).catch(function () {}) : Promise.resolve();
    fonts.then(function () { return fetch(url); })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(function (d) {
        data = d;
        data.pillars.forEach(function (p) {
          pillarList.appendChild(RoadmapWindow.pillarLink({ href: p.url, prefix: p.prefix, name: p.title,
            counts: counts(data, p.prefix), onPick: goPillar }));
        });
        return layoutGraph(data);
      })
      .then(function (l) {
        layout = l;
        api.layout = l;
        els = draw(svg, data, layout);
        sizeCanvas();
        setView(0, 0, 1);
        fromHash();
        resolveReady();
      })
      .catch(function (err) {
        canvas.textContent = "";
        canvas.appendChild(el("p", "roadmap-error", "The roadmap graph could not be drawn (" + err.message + ")."));
        resolveReady();
      });
  }

  function init() {
    var mount = document.getElementById("roadmap-graph");
    if (!mount || typeof ELK === "undefined" || typeof RoadmapWindow === "undefined") return;
    start(mount);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
