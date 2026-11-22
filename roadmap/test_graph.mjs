// Acceptance test of the roadmap graph page (the home of the built docs site, /roadmap/).
//
// It serves a built site folder on 127.0.0.1, opens /roadmap/ in headless Chromium at
// 1440 by 900 in dark mode, and runs the twelve acceptance checks of the graph design
// (line overlaps, lines through nodes, node and label boxes, font sizes, first view,
// registry match, pillar bounds, crossings, links and search, titles, determinism and
// speed, and the phone fallback at 390 by 844). It relies only on the DOM and the
// `__roadmap` object of the page contract.
// It prints one line per check (PASS or FAIL), lists console and page errors, and
// exits 1 if anything failed.
//
// Run:  node roadmap/test_graph.mjs --site <built site folder> [--captures <folder>]
// With --captures it also saves the first view, each pillar, a hover, a search result,
// a feature page and the phone view as PNG files in that folder.
// Playwright comes from the folder named by PLAYWRIGHT_CORE (a playwright-core package)
// when that variable is set, and otherwise from the installed `playwright-core`.

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const args = process.argv.slice(2);
const arg = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
};
const siteDir = arg("--site");
const capDir = arg("--captures");
if (!siteDir || !fs.existsSync(siteDir)) {
  console.error("usage: node test_graph.mjs --site <built site folder> [--captures <folder>]");
  process.exit(2);
}
if (capDir) fs.mkdirSync(capDir, { recursive: true });

const require = createRequire(import.meta.url);
let pw;
if (process.env.PLAYWRIGHT_CORE) pw = require(path.resolve(process.env.PLAYWRIGHT_CORE));
else {
  try { pw = require("playwright-core"); } catch { pw = await import("playwright-core"); }
}
const chromium = pw.chromium ?? pw.default?.chromium;

// ---------- static server ----------
const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml", ".woff2": "font/woff2", ".png": "image/png",
};
const root = path.resolve(siteDir);
const server = http.createServer((req, res) => {
  try {
    let p = decodeURIComponent(new URL(req.url, "http://x").pathname);
    let f = path.join(root, p);
    if (!f.startsWith(root)) { res.writeHead(403).end(); return; }
    if (fs.existsSync(f) && fs.statSync(f).isDirectory()) f = path.join(f, "index.html");
    if (!fs.existsSync(f)) { res.writeHead(404).end("not found"); return; }
    res.writeHead(200, { "Content-Type": TYPES[path.extname(f).toLowerCase()] || "application/octet-stream" });
    fs.createReadStream(f).pipe(res);
  } catch { res.writeHead(500).end(); }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}`;

// ---------- reporting ----------
let failed = false;
const short = (list, n = 4) => list.slice(0, n).join("; ") + (list.length > n ? ` (+${list.length - n} more)` : "");
async function check(n, what, fn) {
  try {
    const bad = await fn(); // returns "" / undefined / [] on success, else a detail
    const detail = Array.isArray(bad) ? short(bad) : bad;
    if (detail) { failed = true; console.log(`FAIL ${n}. ${what}: ${detail}`); }
    else console.log(`PASS ${n}. ${what}`);
  } catch (e) {
    failed = true;
    console.log(`FAIL ${n}. ${what}: ${String(e && e.message || e).split("\n")[0]}`);
  }
}

// ---------- geometry helpers ----------
const EPS = 0.5;
const f1 = (v) => Math.round(v * 10) / 10;
function segments(edge) {
  const out = [];
  const p = edge.points || [];
  for (let i = 0; i + 1 < p.length; i++) {
    const [x1, y1] = p[i], [x2, y2] = p[i + 1];
    if (Math.hypot(x2 - x1, y2 - y1) < 1e-6) continue;
    const h = Math.abs(y1 - y2) < 0.01, v = Math.abs(x1 - x2) < 0.01;
    out.push({ e: edge, x1, y1, x2, y2, h: h && !v, v: v && !h });
  }
  return out;
}
// Does the segment cross the interior of the rectangle (Liang-Barsky, strict)?
function segHitsRect(s, r) {
  let t0 = 0, t1 = 1;
  const dx = s.x2 - s.x1, dy = s.y2 - s.y1;
  const tests = [[-dx, s.x1 - r.x0], [dx, r.x1 - s.x1], [-dy, s.y1 - r.y0], [dy, r.y1 - s.y1]];
  for (const [p, q] of tests) {
    if (Math.abs(p) < 1e-12) { if (q <= 0) return false; continue; }
    const t = q / p;
    if (p < 0) { if (t > t1) return false; if (t > t0) t0 = t; }
    else { if (t < t0) return false; if (t < t1) t1 = t; }
  }
  return t1 - t0 > 1e-9;
}
const rectsOverlap = (a, b) =>
  a.x < b.x + b.width - 0.01 && b.x < a.x + a.width - 0.01 &&
  a.y < b.y + b.height - 0.01 && b.y < a.y + a.height - 0.01;

// ---------- synthetic registry ----------
function synthetic(realGraph) {
  let seed = 20260101;
  const rnd = () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const int = (n) => Math.floor(rnd() * n);
  const prefixes = ["COR", "REN", "STR", "AGT", "BLD", "EMB", "INT", "PLT", "RUN"];
  const pillars = prefixes.map((prefix, i) => ({
    prefix, title: `Pillar ${prefix}`, slug: prefix.toLowerCase(), order: i, url: `${prefix.toLowerCase()}/`,
  }));
  const rp = realGraph.pillars?.[0], rn = realGraph.nodes?.[0];
  const key = (p) => (rp && rn && rn.pillar === rp.slug && rp.slug !== rp.prefix ? p.slug : p.prefix);
  const words = ("graph layout render shader buffer stream token agent cache index probe trace kernel tensor "
    + "scheduler pipeline compiler runtime export import device memory queue vector matrix lens report "
    + "session worker bridge format parser filter window panel viewer loader planner solver").split(" ");
  const statuses = ["shipped", "built", "planned", "idea", "decision"];
  const counters = {};
  const nodes = [];
  for (let i = 0; i < 400; i++) {
    const p = pillars[i % 9];
    counters[p.prefix] = (counters[p.prefix] || 0) + 1;
    const id = `${p.prefix}-${String(counters[p.prefix]).padStart(2, "0")}`;
    const nw = 2 + int(5);
    const title = Array.from({ length: nw }, () => words[int(words.length)]).join(" ");
    const status = statuses[int(5)];
    nodes.push({ id, title, status, in_progress: status === "built" && rnd() < 0.3, pillar: key(p),
      summary: `Synthetic feature ${id}.`, sprint: null, branches: [], draft: false, url: `${p.slug}/${id.toLowerCase()}/` });
  }
  const seen = new Set(), edges = [];
  while (edges.length < 600) {
    const i = int(399);
    let j;
    if (rnd() < 0.2) { j = i + 1 + int(399 - i); if (j % 9 === i % 9) continue; }
    else { const k = 1 + int(Math.max(1, Math.floor((399 - i) / 9))); j = i + 9 * k; if (j > 399) continue; }
    const id = `${nodes[i].id}>${nodes[j].id}`;
    if (seen.has(id)) continue;
    seen.add(id);
    edges.push({ id, source: nodes[i].id, target: nodes[j].id, draft: false });
  }
  return { pillars, nodes, edges };
}

// ---------- browser ----------
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: "dark" });
const page = await context.newPage();
const errors = [];
page.on("console", (m) => {
  if (m.type() !== "error") return;
  if (/favicon\.ico/.test(m.location().url || "")) return;
  errors.push(`console: ${m.text()}`);
});
page.on("pageerror", (e) => errors.push(`page error: ${e.message}`));

const G = "document.getElementById('roadmap-graph').__roadmap";
const load = async (p = page) => {
  await p.goto(`${base}/roadmap/`, { waitUntil: "load" });
  await p.evaluate(`${G}.ready`);
};
const shot = async (p, name, full = false) => {
  if (capDir) await p.screenshot({ path: path.join(capDir, name), fullPage: full });
};

const graph = await (await fetch(`${base}/roadmap/graph.json`)).json();

try {
  await load();
  await page.waitForTimeout(400);
  await shot(page, "first-view.png");
  const layout = await page.evaluate(`JSON.parse(JSON.stringify(${G}.layout))`);
  const firstLayoutJson = JSON.stringify(layout);
  const byKey = new Map(layout.nodes.map((n) => [n.key, n]));
  const allSegs = layout.edges.flatMap(segments);

  // 1
  await check(1, "no overlapping parallel segments from different sources within 6 px", () => {
    const bad = [];
    for (let i = 0; i < allSegs.length; i++) for (let j = i + 1; j < allSegs.length; j++) {
      const a = allSegs[i], b = allSegs[j];
      if (a.e.source === b.e.source) continue;
      let gap, ov;
      if (a.h && b.h) { gap = Math.abs(a.y1 - b.y1); ov = Math.min(Math.max(a.x1, a.x2), Math.max(b.x1, b.x2)) - Math.max(Math.min(a.x1, a.x2), Math.min(b.x1, b.x2)); }
      else if (a.v && b.v) { gap = Math.abs(a.x1 - b.x1); ov = Math.min(Math.max(a.y1, a.y2), Math.max(b.y1, b.y2)) - Math.max(Math.min(a.y1, a.y2), Math.min(b.y1, b.y2)); }
      else continue;
      if (gap < 6 && ov > EPS) bad.push(`${a.e.id} / ${b.e.id} gap ${f1(gap)} overlap ${f1(ov)}`);
    }
    return bad;
  });

  // 2
  await check(2, "no segment passes through a node other than its two ends", () => {
    const bad = [];
    const rects = layout.nodes.map((n) => ({ key: n.key, x0: n.x + 1, y0: n.y + 1, x1: n.x + n.width - 1, y1: n.y + n.height - 1 }));
    for (const s of allSegs) for (const r of rects) {
      if (r.key === s.e.from || r.key === s.e.to) continue;
      if (segHitsRect(s, r)) { bad.push(`${s.e.id} through ${r.key}`); break; }
    }
    return bad;
  });

  // 3
  await check(3, "node rectangles disjoint, labels inside their node, no ellipsis", async () => {
    const bad = [];
    const ns = layout.nodes;
    for (let i = 0; i < ns.length; i++) for (let j = i + 1; j < ns.length; j++)
      if (rectsOverlap(ns[i], ns[j])) bad.push(`${ns[i].key} overlaps ${ns[j].key}`);
    const labels = await page.evaluate(() => {
      const world = document.querySelector("g.rm-world");
      const inv = world.getScreenCTM().inverse();
      const out = [];
      for (const a of document.querySelectorAll("a.rm-node, a.rm-ref")) {
        const pillar = a.closest("g.rm-pillar")?.dataset.pillar;
        const key = a.classList.contains("rm-ref") ? `ref:${pillar}:${a.dataset.of}` : a.dataset.id;
        for (const cls of ["rm-id", "rm-title"]) {
          const t = a.querySelector(`text.${cls}`);
          if (!t) { out.push({ key, cls, missing: true }); continue; }
          const b = t.getBBox(), m = inv.multiply(t.getScreenCTM());
          const pts = [[b.x, b.y], [b.x + b.width, b.y], [b.x, b.y + b.height], [b.x + b.width, b.y + b.height]]
            .map(([x, y]) => new DOMPoint(x, y).matrixTransform(m));
          out.push({ key, cls, text: t.textContent,
            x0: Math.min(...pts.map((p) => p.x)), x1: Math.max(...pts.map((p) => p.x)),
            y0: Math.min(...pts.map((p) => p.y)), y1: Math.max(...pts.map((p) => p.y)) });
        }
      }
      return out;
    });
    for (const l of labels) {
      const n = byKey.get(l.key);
      if (!n) { bad.push(`${l.key} has no layout node`); continue; }
      if (l.missing) { bad.push(`${l.key} lacks ${l.cls}`); continue; }
      if (l.x0 < n.x - EPS || l.y0 < n.y - EPS || l.x1 > n.x + n.width + EPS || l.y1 > n.y + n.height + EPS)
        bad.push(`${l.key} ${l.cls} outside its node`);
      if (/…|\.\.\./.test(l.text)) bad.push(`${l.key} ${l.cls} has an ellipsis`);
    }
    return bad;
  });

  // 4
  await check(4, "titles at least 12 px at first view", async () => {
    const measure = () => page.evaluate(`(() => {
      const k = ${G}.view().k;
      const out = [];
      for (const t of document.querySelectorAll("text.rm-title"))
        out.push([t.closest("a")?.dataset.id || t.closest("a")?.dataset.of, parseFloat(getComputedStyle(t).fontSize) * k]);
      return out;
    })()`);
    const bad = [];
    const first = await measure();
    if (!first.length) bad.push("no titles rendered");
    for (const [id, px] of first) if (px < 12 - 0.01) bad.push(`${id} ${f1(px)} px at first view`);
    return bad;
  });

  // 5
  await check(5, "first view shows a pillar heading, the legend and at least 12 complete features", async () => {
    const r = await page.evaluate(() => {
      const c = document.querySelector(".roadmap-canvas").getBoundingClientRect();
      const inC = (b) => b.left >= c.left - 0.5 && b.top >= c.top - 0.5 && b.right <= c.right + 0.5 && b.bottom <= c.bottom + 0.5;
      const inV = (b) => b.left >= 0 && b.top >= 0 && b.right <= innerWidth && b.bottom <= innerHeight;
      const complete = [...document.querySelectorAll("a.rm-node")].filter((a) => { const b = a.getBoundingClientRect(); return inC(b) && inV(b); }).length;
      const heading = [...document.querySelectorAll("text.rm-heading")].some((h) => inV(h.getBoundingClientRect()));
      const lg = document.querySelector(".roadmap-legend");
      return { complete, heading, legend: !!lg && inV(lg.getBoundingClientRect()) };
    });
    const bad = [];
    if (!r.heading) bad.push("no pillar heading in the viewport");
    if (!r.legend) bad.push("legend not in the viewport");
    if (r.complete < 12) bad.push(`only ${r.complete} complete features`);
    return bad;
  });

  // 6
  await check(6, "drawn features and lines equal the registry", async () => {
    const d = await page.evaluate(() => ({
      nodes: [...document.querySelectorAll("a.rm-node")].map((a) => a.dataset.id),
      edges: [...document.querySelectorAll("path.rm-edge")].map((p) => p.dataset.id),
    }));
    const bad = [];
    const cmp = (what, drawn, reg) => {
      const dup = drawn.filter((x, i) => drawn.indexOf(x) !== i);
      if (dup.length) bad.push(`${what} duplicated: ${short([...new Set(dup)], 3)}`);
      const ds = new Set(drawn), rs = new Set(reg);
      const missing = reg.filter((x) => !ds.has(x)), extra = [...ds].filter((x) => !rs.has(x));
      if (missing.length) bad.push(`${what} missing: ${short(missing, 3)}`);
      if (extra.length) bad.push(`${what} extra: ${short(extra, 3)}`);
    };
    cmp("nodes", d.nodes, graph.nodes.map((n) => n.id));
    cmp("edges", d.edges, graph.edges.map((e) => e.id));
    return bad;
  });

  // 7
  await check(7, "every line's bounding box lies inside its pillar section", async () => {
    const bad = [];
    const dom = await page.evaluate(() => {
      const world = document.querySelector("g.rm-world");
      const inv = world.getScreenCTM().inverse();
      const out = {};
      for (const p of document.querySelectorAll("path.rm-edge")) {
        const b = p.getBBox(), m = inv.multiply(p.getScreenCTM());
        const pts = [[b.x, b.y], [b.x + b.width, b.y + b.height]].map(([x, y]) => new DOMPoint(x, y).matrixTransform(m));
        out[p.dataset.id] = [Math.min(pts[0].x, pts[1].x), Math.min(pts[0].y, pts[1].y), Math.max(pts[0].x, pts[1].x), Math.max(pts[0].y, pts[1].y)];
      }
      return out;
    });
    for (const e of layout.edges) {
      const pl = layout.pillars.find((p) => p.prefix === e.pillar);
      if (!pl) { bad.push(`${e.id} has unknown pillar ${e.pillar}`); continue; }
      const xs = e.points.map((p) => p[0]), ys = e.points.map((p) => p[1]);
      const boxes = [[Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]];
      if (dom[e.id]) boxes.push(dom[e.id]);
      for (const b of boxes)
        if (b[0] < pl.x - EPS || b[1] < pl.y - EPS || b[2] > pl.x + pl.width + EPS || b[3] > pl.y + pl.height + EPS) { bad.push(`${e.id} outside ${e.pillar}`); break; }
    }
    return bad;
  });

  // 8
  await check(8, "crossings per edge at most 0.6 in every pillar", () => {
    let worst = { r: 0, p: "" };
    for (const pl of layout.pillars) {
      const es = layout.edges.filter((e) => e.pillar === pl.prefix);
      if (!es.length) continue;
      const ss = es.flatMap(segments);
      const ends = (e) => [e.points[0], e.points[e.points.length - 1]];
      let n = 0;
      for (let i = 0; i < ss.length; i++) for (let j = i + 1; j < ss.length; j++) {
        const a = ss[i], b = ss[j];
        if (a.e === b.e || a.e.source === b.e.source) continue;
        const h = a.h && b.v ? a : b.h && a.v ? b : null, v = h === a ? b : h === b ? a : null;
        if (!h) continue;
        const px = v.x1, py = h.y1, e = 0.01;
        if (!(px > Math.min(h.x1, h.x2) + e && px < Math.max(h.x1, h.x2) - e)) continue;
        if (!(py > Math.min(v.y1, v.y2) + e && py < Math.max(v.y1, v.y2) - e)) continue;
        if ([...ends(a.e), ...ends(b.e)].some(([x, y]) => Math.abs(x - px) < EPS && Math.abs(y - py) < EPS)) continue;
        n++;
      }
      const r = n / es.length;
      if (r > worst.r) worst = { r, p: pl.prefix };
    }
    return worst.r > 0.6 ? `worst pillar ${worst.p} at ${worst.r.toFixed(2)}` : "";
  });

  // 9
  await check(9, "every feature link resolves and search brings each feature into view", async () => {
    const bad = [];
    const hrefs = await page.evaluate(() => [...document.querySelectorAll("a.rm-node")].map((a) => [a.dataset.id, a.getAttribute("href")]));
    const pageUrl = page.url();
    for (let i = 0; i < hrefs.length; i += 20) {
      await Promise.all(hrefs.slice(i, i + 20).map(async ([id, h]) => {
        try {
          const r = await page.request.get(new URL(h, pageUrl).href);
          if (r.status() !== 200) bad.push(`${id} link ${r.status()}`);
        } catch (e) { bad.push(`${id} link error`); }
      }));
    }
    const input = page.locator(".roadmap-search input");
    for (const [id] of hrefs) {
      await input.fill(id);
      await input.press("Enter");
      const ok = await page.evaluate(async (id) => {
        const inside = () => {
          const a = document.querySelector(`a.rm-node[data-id="${id}"]`);
          if (!a) return false;
          const b = a.getBoundingClientRect(), c = document.querySelector(".roadmap-canvas").getBoundingClientRect();
          return b.left >= c.left - 0.5 && b.top >= c.top - 0.5 && b.right <= c.right + 0.5 && b.bottom <= c.bottom + 0.5;
        };
        for (let t = 0; t < 40; t++) { if (inside()) return true; await new Promise((r) => setTimeout(r, 50)); }
        return false;
      }, id);
      if (!ok) bad.push(`${id} not in view after search`);
    }
    return bad;
  });

  // Beyond the eleven: a URL hash centres and outlines its feature, as the feature pages' links
  // back to the graph use it, and clicking a reference node moves to the real feature.
  await check("9b", "a URL hash and a reference node bring their feature into view", async () => {
    const bad = [];
    const inView = (id) => page.evaluate((id) => {
      const a = document.querySelector(`a.rm-node[data-id="${id}"]`);
      const b = a.getBoundingClientRect(), c = document.querySelector(".roadmap-canvas").getBoundingClientRect();
      return a.classList.contains("is-found") && b.left >= c.left - 0.5 && b.top >= c.top - 0.5 && b.right <= c.right + 0.5 && b.bottom <= c.bottom + 0.5;
    }, id);
    await page.goto(`${base}/roadmap/#REN-15`, { waitUntil: "load" });
    await page.evaluate(`${G}.ready`);
    if (!(await inView("REN-15"))) bad.push("#REN-15 does not centre REN-15");
    const ref = layout.nodes.find((n) => n.ref && n.pillar !== layout.nodes.find((m) => m.key === n.of)?.pillar);
    if (!ref) return bad.concat("no reference node");
    await page.evaluate(`${G}.focus(${JSON.stringify(ref.of)})`);
    const sel = `g.rm-pillar[data-pillar="${ref.pillar}"] a.rm-ref[data-of="${ref.of}"]`;
    await page.evaluate((s) => {
      const lay = document.getElementById("roadmap-graph").__roadmap;
      const n = lay.layout.nodes.find((m) => m.key === s), c = document.querySelector(".roadmap-canvas");
      const v = lay.view();
      lay.setView(c.clientWidth / 2 - v.k * (n.x + n.width / 2), c.clientHeight / 2 - v.k * (n.y + n.height / 2), v.k);
    }, ref.key);
    await page.click(sel);
    if (!(await inView(ref.of))) bad.push(`clicking the reference ${ref.key} does not bring ${ref.of} into view`);
    return bad;
  });

  // 10
  await check(10, 'no title starts with "The", "A" or "An"', () => {
    const re = /^(the|a|an) /i;
    return [...graph.nodes.map((n) => [n.id, n.title]), ...graph.pillars.map((p) => [p.prefix, p.title])]
      .filter(([, t]) => re.test(t)).map(([id, t]) => `${id} "${t}"`);
  });

  // 11
  await check(11, "two loads give identical positions and 400 features lay out in under 1,000 ms", async () => {
    const bad = [];
    await load();
    const again = await page.evaluate(`JSON.stringify(${G}.layout)`);
    if (again !== firstLayoutJson) bad.push("layout differs between loads");
    const syn = synthetic(graph);
    const ms = await page.evaluate(async (syn) => {
      const rm = document.getElementById("roadmap-graph").__roadmap;
      await rm.layoutGraph(syn);
      const t0 = performance.now();
      await rm.layoutGraph(syn);
      return performance.now() - t0;
    }, syn);
    console.log(`     synthetic layout (400 features, 600 edges): ${Math.round(ms)} ms`);
    if (!(ms < 1000)) bad.push(`took ${Math.round(ms)} ms`);
    return bad;
  });

  // 12
  await check(12, "on a phone the rail's pillar links take the graph's place, with no Candidates link (it has its own tab)", async () => {
    const bad = [];
    const phoneCtx = await browser.newContext({ viewport: { width: 390, height: 844 }, colorScheme: "dark" });
    try {
      const ph = await phoneCtx.newPage();
      ph.on("pageerror", (e) => errors.push(`page error (phone): ${e.message}`));
      await ph.goto(`${base}/roadmap/`, { waitUntil: "load" });
      await ph.evaluate(`${G}.ready`);
      const s = await ph.evaluate(() => {
        const shows = (e) => !!e && getComputedStyle(e).display !== "none" && e.getBoundingClientRect().width > 0;
        const canvas = document.querySelector("#roadmap-graph .roadmap-canvas");
        const links = [...document.querySelectorAll("#roadmap-graph .roadmap-pillars a[data-pillar]")];
        const cand = document.querySelector("#roadmap-graph .roadmap-rail-link");
        return {
          canvas: shows(canvas), pillars: links.filter(shows).map((a) => [a.dataset.pillar, new URL(a.href).pathname]),
          cand: shows(cand) ? new URL(cand.href).pathname : null,
          wide: document.documentElement.scrollWidth > innerWidth,
        };
      });
      if (s.canvas) bad.push("the canvas shows");
      const want = graph.pillars.map((p) => [p.prefix, `/roadmap/${p.url}`]);
      if (JSON.stringify(s.pillars) !== JSON.stringify(want)) bad.push(`pillar links ${JSON.stringify(s.pillars)}`);
      for (const [, href] of s.pillars) {
        const r = await fetch(`${base}${href}`);
        if (r.status !== 200) bad.push(`${href} returns ${r.status}`);
      }
      if (s.cand) bad.push(`unexpected Candidates rail link ${s.cand}`);
      if (s.wide) bad.push("the page scrolls sideways");
      await ph.click(`.roadmap-pillars a[data-pillar="${graph.pillars[0].prefix}"]`);
      await ph.waitForURL(`**/roadmap/${graph.pillars[0].url}`, { timeout: 5000 }).catch(() => bad.push("a tap on a pillar link does not open its page"));
    } finally {
      await phoneCtx.close();
    }
    return bad;
  });

  // ---------- captures ----------
  if (capDir) {
    try {
      await load();
      for (const p of graph.pillars) {
        await page.click(`.roadmap-pillars a[data-pillar="${p.prefix}"]`);
        await page.waitForTimeout(700);
        await shot(page, `pillar-${p.prefix}.png`);
      }
      await load();
      const hasDeps = (id) => graph.edges.some((e) => e.source === id || e.target === id);
      const hid = graph.nodes.some((n) => n.id === "REN-12") ? "REN-12" : (graph.nodes.find((n) => hasDeps(n.id)) || graph.nodes[0]).id;
      await page.evaluate(`${G}.focus(${JSON.stringify(hid)})`);
      await page.waitForTimeout(800);
      const r = await page.evaluate((id) => {
        const b = document.querySelector(`a.rm-node[data-id="${id}"]`).getBoundingClientRect();
        return [b.left + b.width / 2, b.top + b.height / 2];
      }, hid);
      await page.mouse.move(r[0] - 10, r[1] - 10);
      await page.mouse.move(r[0], r[1], { steps: 5 });
      await page.waitForSelector(".roadmap-tooltip.is-open", { timeout: 3000 }).catch(() => console.log("INFO hover: tooltip did not open"));
      await page.waitForTimeout(300);
      await shot(page, "hover.png");
      await page.mouse.move(720, 890);
      await load();
      const input = page.locator(".roadmap-search input");
      await input.fill("STR-12");
      await input.press("Enter");
      await page.waitForTimeout(900);
      await shot(page, "search.png");

      const fp = await context.newPage();
      const fnode = graph.nodes.find((n) => n.id === "REN-15") || graph.nodes[0];
      await fp.goto(new URL(fnode.url, `${base}/roadmap/`).href, { waitUntil: "load" });
      await shot(fp, "feature-page.png", true);
      await fp.close();

      const phoneCtx = await browser.newContext({ viewport: { width: 390, height: 844 }, colorScheme: "dark" });
      const ph = await phoneCtx.newPage();
      await ph.goto(`${base}/roadmap/`, { waitUntil: "load" });
      await ph.evaluate(`${G}.ready`);
      await shot(ph, "phone.png", true);
      await phoneCtx.close();
    } catch (e) {
      failed = true;
      console.log(`FAIL captures: ${String(e && e.message || e).split("\n")[0]}`);
    }
  }
} catch (e) {
  failed = true;
  console.log(`FAIL setup: ${String(e && e.message || e).split("\n")[0]}`);
}

if (errors.length) { failed = true; console.log(`FAIL console: ${short(errors, 5)}`); }
else console.log("PASS console: no console or page errors");

await browser.close();
await new Promise((r) => server.close(r));
process.exit(failed ? 1 : 0);
