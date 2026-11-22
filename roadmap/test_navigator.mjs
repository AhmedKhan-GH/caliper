// Acceptance test of the candidate navigator (/roadmap/candidates/ of the built docs site).
//
// It serves a built site folder on 127.0.0.1 and opens the navigator in headless Chromium at
// 1440 by 900 in dark mode. It relies only on the DOM, `candidates.json` and the
// `__navigator` object of the page contract (and `__roadmap` of the graph page). The eight
// checks: the shared window (same frame as the graph page), every candidate drawn once,
// search (by key and by a word of the summary), the pillar links, the detail panel and its
// links, nothing overlapping or cut off, the phone fallback on both pages, and a clean
// console with no request to another origin. It prints one line per check (PASS or FAIL),
// each failure with a short detail, and exits 1 if anything failed.
//
// Run:  node roadmap/test_navigator.mjs --site <built site folder> [--captures <folder>]
// With --captures it also saves the first view, a proposed pillar, a hover, the detail
// panel, a search result and both phone pages as PNG files in that folder.
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
  console.error("usage: node test_navigator.mjs --site <built site folder> [--captures <folder>]");
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
const firstLine = (e) => String((e && e.message) || e).split("\n")[0];
async function check(n, what, fn) {
  let timer;
  try {
    const guard = new Promise((_, rej) => { timer = setTimeout(() => rej(new Error("timed out after 240 s")), 240000); });
    const bad = await Promise.race([fn(), guard]); // "" / undefined / [] on success, else a detail
    const detail = Array.isArray(bad) ? short(bad) : bad;
    if (detail) { failed = true; console.log(`FAIL ${n}. ${what}: ${detail}`); }
    else console.log(`PASS ${n}. ${what}`);
  } catch (e) {
    failed = true;
    console.log(`FAIL ${n}. ${what}: ${firstLine(e)}`);
  } finally { clearTimeout(timer); }
}

// ---------- browser ----------
const browser = await chromium.launch({ headless: true });
const errors = [];
const foreign = new Set();
async function newContext(opts) {
  const ctx = await browser.newContext({ colorScheme: "dark", ...opts });
  ctx.setDefaultTimeout(15000);
  ctx.on("request", (r) => {
    const u = r.url();
    if (/^(data|blob|about):/.test(u)) return;
    if (new URL(u).origin !== base) foreign.add(u);
  });
  ctx.on("page", (p) => {
    p.on("console", (m) => {
      if (m.type() !== "error") return;
      if (/favicon\.ico/.test(m.location().url || "")) return;
      errors.push(`console: ${m.text()}`);
    });
    p.on("pageerror", (e) => errors.push(`page error: ${e.message}`));
  });
  return ctx;
}
const context = await newContext({ viewport: { width: 1440, height: 900 } });
await context.addInitScript(() => {
  // Geometry helpers the page-side checks share.
  window.__t = {
    inBox: (b, c) => b.left >= c.left - 0.5 && b.right <= c.right + 0.5 && b.top >= c.top - 0.5 && b.bottom <= c.bottom + 0.5,
    visBox: (sc) => { const r = sc.getBoundingClientRect(); return { left: r.left, top: r.top, right: r.left + sc.clientWidth, bottom: r.top + sc.clientHeight }; },
  };
});
const page = await context.newPage();

const NAV = "document.getElementById('candidate-navigator').__navigator";
const GRAPH = "document.getElementById('roadmap-graph').__roadmap";
const ROUTE = `${base}/roadmap/`;
const CAND = `${base}/roadmap/candidates/`;

// Open the navigator in a fresh document (a hash-only change would not reload) and wait for it.
async function loadNav(p = page, hash = "") {
  await p.goto("about:blank");
  await p.goto(CAND + hash, { waitUntil: "load" });
  await p.waitForFunction(`!!(${NAV})`, null, { timeout: 10000 }).catch(() => { throw new Error("the navigator did not start (no __navigator on #candidate-navigator)"); });
  await p.evaluate(`${NAV}.ready`);
}
async function loadGraph(p = page, hash = "") {
  await p.goto("about:blank");
  await p.goto(ROUTE + hash, { waitUntil: "load" });
  await p.waitForFunction(`!!(${GRAPH})`, null, { timeout: 10000 }).catch(() => { throw new Error("the graph did not start (no __roadmap on #roadmap-graph)"); });
  await p.evaluate(`${GRAPH}.ready`);
}
const shot = async (p, name, full = false) => {
  if (capDir) await p.screenshot({ path: path.join(capDir, name), fullPage: full });
};
async function status200(urls, bad, label) {
  for (let i = 0; i < urls.length; i += 20) {
    await Promise.all(urls.slice(i, i + 20).map(async (u) => {
      try {
        const r = await page.request.get(u);
        if (r.status() !== 200) bad.push(`${label} ${u.replace(base, "")} gives ${r.status()}`);
      } catch { bad.push(`${label} ${u.replace(base, "")} request failed`); }
    }));
  }
}

// ---------- data ----------
let data = null, graph = null, dataError = "";
try {
  data = await (await fetch(`${base}/roadmap/candidates.json`)).json();
  graph = await (await fetch(`${base}/roadmap/graph.json`)).json();
} catch (e) { dataError = `cannot read candidates.json or graph.json (${firstLine(e)})`; }

const all = []; // every candidate in canvas order: pillar, then origin group, then file order
const byKey = new Map();
let dupJsonKeys = [];
if (data) {
  const order = data.origins.map((o) => o.origin);
  for (const p of data.pillars) {
    const sorted = p.candidates.map((c, i) => [c, i]).sort((a, b) => order.indexOf(a[0].origin) - order.indexOf(b[0].origin) || a[1] - b[1]);
    for (const [c] of sorted) {
      const rec = { ...c, pillar: p, hay: [c.key, c.title, c.summary, c.serves].join("\n").toLowerCase() };
      if (byKey.has(c.key)) dupJsonKeys.push(c.key);
      byKey.set(c.key, rec);
      all.push(rec);
    }
  }
}
const originTitle = (o) => (data.origins.find((x) => x.origin === o) || {}).title;
const filterKeys = (query) => {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  return all.filter((c) => words.every((w) => c.hay.includes(w))).map((c) => c.key);
};
const needData = () => { if (!data) throw new Error(dataError); };

// Pick a word of 6+ letters from the summary that is not in the title, with few matches.
function summaryWord(c) {
  const title = c.title.toLowerCase(), key = c.key.toLowerCase();
  let best = null;
  for (const w of new Set(c.summary.toLowerCase().split(/[^a-z]+/))) {
    if (w.length < 6 || title.includes(w) || key.includes(w)) continue;
    const n = filterKeys(w).length;
    if (n >= 1 && n <= 30 && (!best || n < best.n)) best = { w, n };
  }
  return best;
}

try {
  // ---------- 1. window ----------
  await check(1, "the navigator fills the window like the graph page", async () => {
    const measure = async (loader, mountSel) => {
      await loader();
      await page.waitForTimeout(250);
      return page.evaluate((sel) => {
        const m = document.querySelector(sel), rail = m.querySelector(".roadmap-rail"), cv = m.querySelector(".roadmap-canvas");
        const r = (e) => { const b = e.getBoundingClientRect(); return { left: b.left, right: b.right, top: b.top, bottom: b.bottom, width: b.width, height: b.height }; };
        const de = document.documentElement;
        return { m: r(m), rail: r(rail), cv: r(cv), innerHeight, scrollW: de.scrollWidth, clientW: de.clientWidth,
          font: getComputedStyle(rail).fontFamily, railBg: getComputedStyle(rail).backgroundColor, cvBg: getComputedStyle(cv).backgroundColor,
          isWindow: m.classList.contains("roadmap-window") };
      }, mountSel);
    };
    const g = await measure(() => loadGraph(), "#roadmap-graph");
    const n = await measure(() => loadNav(), "#candidate-navigator");
    const bad = [];
    if (!n.isWindow) bad.push("mount lacks class roadmap-window");
    const eq = (what, a, b) => { if (Math.abs(a - b) > 1) bad.push(`${what} ${a.toFixed(1)} vs graph ${b.toFixed(1)}`); };
    eq("mount left", n.m.left, g.m.left); eq("mount right", n.m.right, g.m.right);
    eq("mount top", n.m.top, g.m.top); eq("mount height", n.m.height, g.m.height);
    eq("rail width", n.rail.width, g.rail.width);
    eq("canvas left", n.cv.left, g.cv.left); eq("canvas width", n.cv.width, g.cv.width);
    if (n.font !== g.font) bad.push(`rail font ${n.font} vs graph ${g.font}`);
    if (n.railBg !== g.railBg) bad.push(`rail background ${n.railBg} vs graph ${g.railBg}`);
    if (n.cvBg !== g.cvBg) bad.push(`canvas background ${n.cvBg} vs graph ${g.cvBg}`);
    if (Math.abs(n.m.bottom - (n.innerHeight - 16)) > 1) bad.push(`mount bottom ${n.m.bottom.toFixed(1)}, expected ${n.innerHeight - 16}`);
    if (n.scrollW > n.clientW + 1) bad.push(`page scrolls sideways (${n.scrollW} > ${n.clientW})`);
    return bad;
  });

  // ---------- 2. every candidate once ----------
  await check(2, `every candidate appears exactly once, under its pillar and origin`, async () => {
    needData();
    await loadNav();
    const bad = [];
    console.log(`     candidates.json: ${all.length} candidates in ${data.pillars.length} pillars`);
    if (dupJsonKeys.length) bad.push(`keys repeated in candidates.json: ${short(dupJsonKeys, 3)}`);
    const dom = await page.evaluate(() => ({
      cards: [...document.querySelectorAll(".cn-card")].map((a) => ({
        key: a.dataset.key,
        pillar: a.closest("section.cn-pillar[data-pillar]")?.dataset.pillar ?? null,
        origin: a.closest(".cn-group[data-origin]")?.dataset.origin ?? null,
        groupPillar: a.closest(".cn-group")?.closest("section.cn-pillar")?.dataset.pillar ?? null,
      })),
      sections: [...document.querySelectorAll("section.cn-pillar[data-pillar]")].map((s) => s.dataset.pillar),
    }));
    if (dom.cards.length !== all.length) bad.push(`${dom.cards.length} cards, expected ${all.length}`);
    const seen = new Set(), dup = [];
    for (const c of dom.cards) { if (seen.has(c.key)) dup.push(c.key); seen.add(c.key); }
    if (dup.length) bad.push(`keys drawn twice: ${short([...new Set(dup)], 3)}`);
    const missing = all.filter((c) => !seen.has(c.key)).map((c) => c.key);
    const extra = [...seen].filter((k) => !byKey.has(k));
    if (missing.length) bad.push(`keys missing: ${short(missing, 3)}`);
    if (extra.length) bad.push(`keys not in the data: ${short(extra, 3)}`);
    const wrong = [];
    for (const c of dom.cards) {
      const r = byKey.get(c.key);
      if (!r) continue;
      if (c.pillar !== r.pillar.prefix || c.groupPillar !== r.pillar.prefix) wrong.push(`${c.key} in pillar ${c.pillar}, expected ${r.pillar.prefix}`);
      else if (c.origin !== r.origin) wrong.push(`${c.key} in origin ${c.origin}, expected ${r.origin}`);
    }
    if (wrong.length) bad.push(`misplaced: ${short(wrong, 3)}`);
    const want = data.pillars.map((p) => p.prefix).join(",");
    if (dom.sections.join(",") !== want) bad.push(`pillar order ${dom.sections.join(",")} differs from the data order`);
    return bad;
  });

  // ---------- 3. search ----------
  await check(3, "search finds candidates by key and by a word of the summary", async () => {
    needData();
    await loadNav();
    const bad = [];

    // Keys of a spread sample: first, middle and last of every pillar.
    const sample = [];
    for (const p of data.pillars) {
      const cs = p.candidates;
      for (const i of new Set([0, Math.floor(cs.length / 2), cs.length - 1])) if (cs[i]) sample.push(cs[i].key);
    }
    const keyRes = await page.evaluate(async (keys) => {
      const { inBox, visBox } = window.__t;
      const nav = document.getElementById("candidate-navigator").__navigator;
      const sc = document.querySelector(".cn-scroll");
      const out = [];
      for (const k of keys) {
        nav.search(k);
        await new Promise((r) => requestAnimationFrame(r));
        const cur = [...document.querySelectorAll(".cn-card.is-current")];
        out.push({ k, cur: cur.map((c) => c.dataset.key), inView: cur.length === 1 && inBox(cur[0].getBoundingClientRect(), visBox(sc)) });
      }
      return out;
    }, sample);
    const keyBad = keyRes.filter((r) => r.cur.length !== 1 || r.cur[0] !== r.k || !r.inView);
    if (keyBad.length) bad.push(`by key ${keyBad.length} of ${sample.length} fail: ${short(keyBad.map((r) => `${r.k} (current ${r.cur.join("+") || "none"}${r.cur.length === 1 && r.cur[0] === r.k ? ", not in view" : ""})`), 3)}`);

    // A summary word that is not in the title, for at least 10 candidates spread over the data.
    const eligible = [];
    for (const c of all) { const w = summaryWord(c); if (w) eligible.push({ key: c.key, word: w.w }); }
    const step = Math.max(1, Math.floor(eligible.length / 12));
    const picks = eligible.filter((_, i) => i % step === 0).slice(0, 12);
    if (picks.length < 10) bad.push(`only ${picks.length} candidates have a usable summary word`);
    const wordRes = await page.evaluate(async (picks) => {
      const { inBox, visBox } = window.__t;
      const nav = document.getElementById("candidate-navigator").__navigator;
      const sc = document.querySelector(".cn-scroll");
      const cur = () => document.querySelector(".cn-card.is-current")?.dataset.key ?? null;
      const out = [];
      for (const p of picks) {
        const keys = nav.search(p.word);
        let c = cur(), steps = 0;
        while (c !== p.key && steps < keys.length) { c = nav.next(); steps++; }
        await new Promise((r) => requestAnimationFrame(r));
        const el = document.querySelector(".cn-card.is-current");
        out.push({ ...p, keys, found: c === p.key && el?.dataset.key === p.key, inView: !!el && inBox(el.getBoundingClientRect(), visBox(sc)) });
      }
      return out;
    }, picks);
    for (const r of wordRes) {
      if (!r.found) bad.push(`word "${r.word}" does not reach ${r.key}`);
      else if (!r.inView) bad.push(`word "${r.word}": ${r.key} is current but not in view`);
      const want = filterKeys(r.word);
      if (r.keys.join() !== want.join()) bad.push(`word "${r.word}" returns ${r.keys.length} keys, the filter gives ${want.length}`);
    }

    // The returned keys equal a filter computed here, for several queries.
    const queries = ["render", "RENDER", "gpu memory", "stream data", "applet", "model"];
    const qRes = await page.evaluate((qs) => {
      const nav = document.getElementById("candidate-navigator").__navigator;
      return qs.map((q) => [q, nav.search(q)]);
    }, queries);
    for (const [q, keys] of qRes) {
      const want = filterKeys(q);
      if (keys.join() !== want.join()) {
        const ks = new Set(keys), ws = new Set(want);
        bad.push(`query "${q}" returns ${keys.length} keys, expected ${want.length} (missing ${short(want.filter((k) => !ks.has(k)), 2) || "none"}, extra ${short(keys.filter((k) => !ws.has(k)), 2) || "none"})`);
      }
    }

    // Gibberish finds nothing, marks the input and says so.
    const nope = await page.evaluate(() => {
      const nav = document.getElementById("candidate-navigator").__navigator;
      const keys = nav.search("zzqxjvkwy");
      const input = document.querySelector(".roadmap-search input");
      const out = { n: keys.length, missing: input.classList.contains("is-missing"),
        status: document.querySelector(".roadmap-search-status")?.textContent.trim(),
        found: document.querySelectorAll(".cn-card.is-found, .cn-card.is-current").length };
      nav.search("");
      out.cleared = !input.classList.contains("is-missing") && !document.querySelector(".cn-card.is-found, .cn-card.is-current, .cn-scroll.is-searching")
        && !document.querySelector(".roadmap-search-status")?.textContent.trim();
      return out;
    });
    if (nope.n !== 0 || nope.found) bad.push(`gibberish matched ${nope.n} candidates`);
    if (!nope.missing) bad.push("gibberish: input lacks .is-missing");
    if (nope.status !== "No match") bad.push(`gibberish: status reads "${nope.status}", expected "No match"`);
    if (!nope.cleared) bad.push("an empty query does not clear the classes and the status");

    // Real typing: a key that matches only itself, debounce, then Enter.
    const solo = all.find((c, i) => i > all.length / 3 && filterKeys(c.key).length === 1) || all[Math.floor(all.length / 2)];
    const input = page.locator(".roadmap-search input");
    await input.fill("");
    await input.pressSequentially(solo.key, { delay: 15 });
    await page.waitForFunction((k) => document.querySelector(".cn-card.is-current")?.dataset.key === k, solo.key, { timeout: 3000 })
      .catch(() => bad.push(`typing "${solo.key}" did not make it current within the debounce`));
    await input.press("Enter");
    const typed = await page.evaluate(async (k) => {
      const { inBox, visBox } = window.__t;
      await new Promise((r) => requestAnimationFrame(r));
      const el = document.querySelector(".cn-card.is-current");
      return { cur: el?.dataset.key, inView: !!el && inBox(el.getBoundingClientRect(), visBox(document.querySelector(".cn-scroll"))) };
    }, solo.key);
    if (typed.cur !== solo.key) bad.push(`after typing and Enter, current is ${typed.cur}, expected ${solo.key}`);
    else if (!typed.inView) bad.push(`after typing and Enter, ${solo.key} is not in view`);
    return bad;
  });

  // ---------- 4. pillar links ----------
  await check(4, "every pillar link scrolls to its pillar and resolves", async () => {
    needData();
    await loadNav();
    const bad = [];
    const links = await page.evaluate(() => [...document.querySelectorAll(".roadmap-pillars a[data-pillar]")].map((a) => [a.dataset.pillar, a.href]));
    if (links.length !== data.pillars.length) bad.push(`${links.length} pillar links, expected ${data.pillars.length}`);
    const urls = [];
    for (const p of data.pillars) {
      const l = links.find((x) => x[0] === p.prefix);
      if (!l) { bad.push(`no pillar link for ${p.prefix}`); continue; }
      const want = `/roadmap/candidates/${p.slug}/`;
      if (new URL(l[1]).pathname !== want) bad.push(`${p.prefix} href ${new URL(l[1]).pathname}, expected ${want}`);
      else urls.push(l[1]);
      await page.locator(`.roadmap-pillars a[data-pillar="${p.prefix}"]`).click();
      const d = await page.evaluate(async (prefix) => {
        const { inBox, visBox } = window.__t;
        await new Promise((r) => requestAnimationFrame(r));
        const sc = document.querySelector(".cn-scroll"), s = document.querySelector(`section.cn-pillar[data-pillar="${prefix}"]`);
        const want = sc.getBoundingClientRect().top + parseFloat(getComputedStyle(sc).paddingTop) + sc.clientTop;
        return { diff: s.getBoundingClientRect().top - want, url: location.pathname };
      }, p.prefix);
      if (Math.abs(d.diff) > 2) bad.push(`${p.prefix} section top is ${d.diff.toFixed(1)} px from the scroller's padding line`);
      if (d.url !== "/roadmap/candidates/") bad.push(`${p.prefix} click navigated to ${d.url}`);
    }
    await status200(urls, bad, "pillar page");
    return bad;
  });

  // ---------- 5. detail panel ----------
  await check(5, "the detail panel opens, its links resolve, and hashes open cards", async () => {
    needData();
    await loadNav();
    const bad = [];
    const three = all.find((c) => c.depends_on.length === 3) || all.find((c) => c.depends_on.length > 0);
    const card = page.locator(`.cn-card[data-key="${three.key}"]`);
    await card.scrollIntoViewIfNeeded();
    await card.click();
    await page.waitForSelector("aside.cn-panel.is-open", { timeout: 3000 }).catch(() => bad.push("a click does not open aside.cn-panel.is-open"));
    const panel = await page.evaluate(() => {
      const p = document.querySelector("aside.cn-panel.is-open");
      if (!p) return null;
      return { text: p.textContent.replace(/\s+/g, " "), inCanvas: !!p.closest(".roadmap-canvas"), hash: location.hash,
        deps: [...p.querySelectorAll("li.cn-dep")].map((l) => l.dataset.id), selected: document.querySelectorAll(".cn-card.is-selected").length };
    });
    if (panel) {
      if (!panel.inCanvas) bad.push("panel is not inside the canvas");
      if (panel.hash !== `#${three.key}`) bad.push(`hash is "${panel.hash}", expected #${three.key}`);
      if (panel.selected !== 1) bad.push(`${panel.selected} cards are .is-selected, expected 1`);
      for (const [what, v] of [["title", three.title], ["key", three.key], ["pillar title", three.pillar.title],
        ["origin title", originTitle(three.origin)], ["summary", three.summary], ["serves", three.serves]]) {
        if (!panel.text.includes(String(v).replace(/\s+/g, " "))) bad.push(`panel lacks the ${what} "${String(v).slice(0, 30)}"`);
      }
      if (panel.deps.join() !== three.depends_on.join()) bad.push(`panel lists dependencies ${panel.deps.join(",")}, expected ${three.depends_on.join(",")}`);
      await page.keyboard.press("Escape");
      const after = await page.evaluate(() => ({ open: !!document.querySelector("aside.cn-panel.is-open"), hash: location.hash, sel: document.querySelectorAll(".cn-card.is-selected").length }));
      if (after.open) bad.push("Escape does not close the panel");
      if (after.hash) bad.push(`Escape leaves the hash "${after.hash}"`);
      if (after.sel) bad.push("Escape leaves a card .is-selected");
    }

    // Every link in every panel, by select().
    const harvest = await page.evaluate(async () => {
      const nav = document.getElementById("candidate-navigator").__navigator;
      const out = { links: [], counts: [] };
      for (const p of nav.data.pillars) for (const c of p.candidates) {
        nav.select(c.key);
        const panel = document.querySelector("aside.cn-panel");
        for (const a of panel.querySelectorAll("a.cn-dep-page, a.cn-dep-graph, a.cn-panel-pillar"))
          out.links.push({ cls: a.className.split(/\s+/).find((x) => /^cn-(dep-page|dep-graph|panel-pillar)$/.test(x)), href: a.href, key: c.key, pillar: p.slug,
            id: a.closest("li.cn-dep")?.dataset.id });
        out.counts.push([c.key, panel.querySelectorAll("li.cn-dep").length, c.depends_on.length]);
      }
      nav.close();
      return out;
    });
    const badCount = harvest.counts.filter(([, n, m]) => n !== m);
    if (badCount.length) bad.push(`dependency rows differ from the data for ${badCount.length} candidates (${badCount[0][0]})`);
    const jsonUrl = new URL("candidates.json", ROUTE).href;
    const wrongHref = [];
    for (const l of harvest.links) {
      let want;
      if (l.cls === "cn-panel-pillar") want = `${base}/roadmap/candidates/${l.pillar}/`;
      else { const f = data.features[l.id]; if (!f) { wrongHref.push(`${l.key} lists ${l.id}, not in features`); continue; } want = new URL(l.cls === "cn-dep-page" ? f.url : f.graph, jsonUrl).href; }
      if (l.href !== want) wrongHref.push(`${l.key} ${l.cls} ${l.href.replace(base, "")}, expected ${want.replace(base, "")}`);
    }
    if (wrongHref.length) bad.push(`link targets: ${short(wrongHref, 3)}`);
    const unique = (cls) => [...new Set(harvest.links.filter((l) => l.cls === cls).map((l) => l.href.split("#")[0]))];
    for (const cls of ["cn-dep-page", "cn-dep-graph", "cn-panel-pillar"]) {
      const urls = unique(cls);
      if (!urls.length) bad.push(`no ${cls} links found`);
      await status200(urls, bad, cls);
    }

    // Three graph links open their node in the roadmap graph.
    const graphLinks = [...new Set(harvest.links.filter((l) => l.cls === "cn-dep-graph").map((l) => l.href))];
    const gl = graphLinks.length >= 3 ? [graphLinks[0], graphLinks[Math.floor(graphLinks.length / 2)], graphLinks[graphLinks.length - 1]] : graphLinks;
    for (const href of gl) {
      const id = decodeURIComponent(new URL(href).hash.slice(1));
      await page.goto("about:blank");
      await page.goto(href, { waitUntil: "load" });
      await page.waitForFunction(`!!(${GRAPH})`, null, { timeout: 10000 });
      await page.evaluate(`${GRAPH}.ready`);
      const ok = await page.evaluate(async (id) => {
        const { inBox, visBox } = window.__t;
        await new Promise((r) => setTimeout(r, 150));
        const a = document.querySelector(`a.rm-node[data-id="${id}"]`);
        return !!a && a.classList.contains("is-found") && inBox(a.getBoundingClientRect(), document.querySelector(".roadmap-canvas").getBoundingClientRect());
      }, id);
      if (!ok) bad.push(`graph link #${id} does not bring the node into view and .is-found`);
    }

    // A page loaded with #<key> opens that card's panel, with the card in view.
    const deep = all[all.length - 1];
    await loadNav(page, `#${deep.key}`);
    const h = await page.evaluate(async (k) => {
      const { inBox, visBox } = window.__t;
      await new Promise((r) => setTimeout(r, 200));
      const p = document.querySelector("aside.cn-panel.is-open"), c = document.querySelector(`.cn-card[data-key="${k}"]`);
      return { open: !!p, text: p ? p.textContent.includes(k) : false, inView: !!c && inBox(c.getBoundingClientRect(), visBox(document.querySelector(".cn-scroll"))) };
    }, deep.key);
    if (!h.open || !h.text) bad.push(`loading #${deep.key} does not open its panel`);
    if (!h.inView) bad.push(`loading #${deep.key} leaves the card out of view`);
    return bad;
  });

  // ---------- 6. nothing overlaps or is cut off ----------
  await check(6, "nothing overlaps or is cut off", async () => {
    needData();
    await loadNav();
    const bad = [];
    const r = await page.evaluate(() => {
      const { inBox, visBox } = window.__t;
      const out = { cards: [], overlap: [], heads: [], rail: [] };
      const box = (e) => e.getBoundingClientRect();
      const content = (e) => { const g = document.createRange(); g.selectNodeContents(e); return g.getBoundingClientRect(); };
      const cards = [...document.querySelectorAll(".cn-card")];
      const rects = cards.map(box);
      cards.forEach((c, i) => {
        const cb = rects[i];
        for (const cls of ["cn-card-key", "cn-card-title"]) {
          const s = c.querySelector("." + cls);
          if (!s) { out.cards.push(`${c.dataset.key} lacks ${cls}`); continue; }
          const sb = content(s);
          if (!inBox(sb, cb)) out.cards.push(`${c.dataset.key} ${cls} outside the card`);
          else if (s.clientWidth && s.scrollWidth > s.clientWidth + 1) out.cards.push(`${c.dataset.key} ${cls} overflows sideways`);
          if (getComputedStyle(s).textOverflow === "ellipsis") out.cards.push(`${c.dataset.key} ${cls} is truncated with an ellipsis`);
        }
        if (c.scrollWidth > c.clientWidth + 1) out.cards.push(`${c.dataset.key} overflows its own box`);
      });
      for (let i = 0; i < cards.length; i++) for (let j = i + 1; j < cards.length; j++) {
        const a = rects[i], b = rects[j];
        if (a.left < b.right - 0.5 && b.left < a.right - 0.5 && a.top < b.bottom - 0.5 && b.top < a.bottom - 0.5) out.overlap.push(`${cards[i].dataset.key} / ${cards[j].dataset.key}`);
      }
      for (const s of document.querySelectorAll("section.cn-pillar")) {
        const sb = box(s), id = s.dataset.pillar;
        for (const sel of [".cn-pillar-head", ".cn-pillar-title", ".cn-pillar-count", ".cn-proposed", ".cn-pillar-desc", ".cn-pillar-alt"]) {
          const e = s.querySelector(sel);
          if (!e) continue;
          if (!inBox(box(e), sb)) out.heads.push(`${id} ${sel} outside its section`);
          if (e.clientWidth && e.scrollWidth > e.clientWidth + 1) out.heads.push(`${id} ${sel} overflows sideways`);
          if (!inBox(content(e), sb)) out.heads.push(`${id} ${sel} text outside its section`);
        }
      }
      const rail = document.querySelector(".roadmap-rail"), rb = box(rail);
      if (rail.scrollWidth > rail.clientWidth + 1) out.rail.push(`rail scrolls sideways (${rail.scrollWidth} > ${rail.clientWidth})`);
      for (const e of rail.querySelectorAll("*")) {
        const b = box(e);
        if (!b.width && !b.height) continue;
        if (b.left < rb.left - 0.5 || b.right > rb.right + 0.5) out.rail.push(`${e.tagName.toLowerCase()}.${String(e.className).split(" ")[0]} sticks out of the rail`);
      }
      return out;
    });
    if (r.cards.length) bad.push(`cards: ${short(r.cards, 3)}`);
    if (r.overlap.length) bad.push(`cards overlap: ${short(r.overlap, 3)}`);
    if (r.heads.length) bad.push(`pillar heads: ${short(r.heads, 3)}`);
    if (r.rail.length) bad.push(`rail: ${short(r.rail, 3)}`);

    // The open panel and its contents, for a three-dependency candidate, the most dependencies
    // and the longest text.
    const most = all.reduce((a, c) => (c.depends_on.length > a.depends_on.length ? c : a), all[0]);
    const longest = all.reduce((a, c) => (c.title.length + c.summary.length + c.serves.length > a.title.length + a.summary.length + a.serves.length ? c : a), all[0]);
    const three = all.find((c) => c.depends_on.length === 3) || most;
    for (const c of new Set([three, most, longest])) {
      const p = await page.evaluate(async (k) => {
        const { inBox, visBox } = window.__t;
        document.getElementById("candidate-navigator").__navigator.select(k);
        await new Promise((r) => requestAnimationFrame(r));
        const panel = document.querySelector("aside.cn-panel.is-open");
        if (!panel) return { none: true };
        const pb = panel.getBoundingClientRect(), cb = document.querySelector(".roadmap-canvas").getBoundingClientRect();
        const out = { none: false, inCanvas: inBox(pb, cb), wide: panel.scrollWidth > panel.clientWidth + 1, outside: [] };
        for (const e of panel.querySelectorAll("*")) {
          const b = e.getBoundingClientRect();
          if (!b.width && !b.height) continue;
          if (b.left < pb.left - 0.5 || b.right > pb.right + 0.5) out.outside.push(`${e.tagName.toLowerCase()}.${String(e.className).split(" ")[0]}`);
        }
        document.getElementById("candidate-navigator").__navigator.close();
        return out;
      }, c.key);
      if (p.none) { bad.push(`select(${c.key}) opens no panel`); continue; }
      if (!p.inCanvas) bad.push(`panel of ${c.key} leaves the canvas`);
      if (p.wide) bad.push(`panel of ${c.key} scrolls sideways`);
      if (p.outside.length) bad.push(`panel of ${c.key}: ${short(p.outside, 2)} outside the panel`);
    }

    // The tooltip, from a card at the right edge, the left edge and near the bottom.
    const cases = await page.evaluate(async () => {
      const { inBox, visBox } = window.__t;
      const sc = document.querySelector(".cn-scroll");
      sc.scrollTop = 0;
      await new Promise((r) => requestAnimationFrame(r));
      const vb = visBox(sc);
      const vis = [...document.querySelectorAll(".cn-card")].filter((c) => inBox(c.getBoundingClientRect(), vb));
      const pick = (f) => vis.reduce((a, c) => (f(c.getBoundingClientRect()) > f(a.getBoundingClientRect()) ? c : a), vis[0]);
      const right = pick((b) => b.right).dataset.key, left = pick((b) => -b.left).dataset.key;
      // Bring a card from lower down so its bottom edge sits just above the canvas bottom.
      const all = [...document.querySelectorAll(".cn-card")];
      const low = all[Math.floor(all.length / 2)];
      sc.scrollTop += low.getBoundingClientRect().bottom - (vb.bottom - 4);
      await new Promise((r) => requestAnimationFrame(r));
      return { right, left, bottom: low.dataset.key };
    });
    for (const [where, key] of [["right edge", cases.right], ["left edge", cases.left], ["bottom", cases.bottom]]) {
      await page.evaluate(([k, w]) => {
        const sc = document.querySelector(".cn-scroll"), c = document.querySelector(`.cn-card[data-key="${k}"]`);
        sc.scrollTop = 0; c.scrollIntoView({ block: "nearest" });
        if (w === "bottom") {
          // Put this card's bottom edge just above the canvas bottom.
          const r = sc.getBoundingClientRect();
          sc.scrollTop += c.getBoundingClientRect().bottom - (r.top + sc.clientHeight - 4);
        }
      }, [key, where]);
      const box = await page.locator(`.cn-card[data-key="${key}"]`).boundingBox();
      await page.mouse.move(2, 2);
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 4 });
      const open = await page.waitForSelector(".roadmap-tooltip.is-open", { timeout: 3000 }).then(() => true).catch(() => false);
      if (!open) { bad.push(`tooltip does not open for ${key} (${where})`); continue; }
      const t = await page.evaluate(() => {
        const { inBox, visBox } = window.__t;
        const tip = document.querySelector(".roadmap-tooltip.is-open");
        return { inCanvas: inBox(tip.getBoundingClientRect(), document.querySelector(".roadmap-canvas").getBoundingClientRect()) };
      }, );
      if (!t.inCanvas) bad.push(`tooltip for ${key} (${where}) leaves the canvas`);
      await page.mouse.move(2, 2);
    }
    return bad;
  });

  // ---------- 7. phone ----------
  await check(7, "the phone fallback works on both pages", async () => {
    needData();
    const bad = [];
    const ctx = await newContext({ viewport: { width: 390, height: 844 }, hasTouch: true });
    try {
      const ph = await ctx.newPage();
      const settle = async (sel) => {
        await ph.waitForFunction((s) => { const m = document.querySelector(s); return m && (m.__navigator || m.__roadmap); }, sel, { timeout: 8000 }).catch(() => {});
        await ph.evaluate(async (s) => { const m = document.querySelector(s); const p = (m.__navigator || m.__roadmap || {}).ready; if (p) await Promise.race([p, new Promise((r) => setTimeout(r, 6000))]); }, sel).catch(() => {});
        await ph.waitForTimeout(300);
      };
      const state = (mountSel) => ph.evaluate((sel) => {
        const shown = (e) => !!e && getComputedStyle(e).display !== "none" && e.getBoundingClientRect().width > 0 && e.getBoundingClientRect().height > 0;
        const de = document.documentElement;
        return {
          canvas: document.querySelector(sel + " .roadmap-canvas") ? shown(document.querySelector(sel + " .roadmap-canvas")) : "missing",
          legend: document.querySelector(sel + " .roadmap-legend") ? shown(document.querySelector(sel + " .roadmap-legend")) : "missing",
          search: document.querySelector(sel + " .roadmap-search") ? shown(document.querySelector(sel + " .roadmap-search")) : "missing",
          links: [...document.querySelectorAll(sel + " .roadmap-pillars a[data-pillar]")].map((a) => ({ href: a.href, shown: shown(a) })),
          group: [...document.querySelectorAll(sel + " .roadmap-group")].map((g) => ({ text: g.textContent.trim(), shown: shown(g) })),
          railLinks: [...document.querySelectorAll(sel + " .roadmap-rail-link")].map((a) => ({ href: a.href, shown: shown(a) })),
          headings: [...document.querySelectorAll("article h1, article h2, article h3, article h4, article h5, article h6")].map((h) => h.textContent.replace(/¶/g, "").trim()),
          scrollW: de.scrollWidth, clientW: de.clientWidth,
        };
      }, mountSel);

      // Navigator
      await ph.goto(CAND, { waitUntil: "load" });
      await settle("#candidate-navigator");
      let s = await state("#candidate-navigator");
      for (const k of ["canvas", "legend", "search"]) {
        if (s[k] === "missing") bad.push(`navigator: no ${k} element`);
        else if (s[k]) bad.push(`navigator: ${k} is visible on a phone`);
      }
      const shownLinks = s.links.filter((l) => l.shown);
      if (shownLinks.length !== data.pillars.length) bad.push(`navigator: ${shownLinks.length} pillar links shown, expected ${data.pillars.length}`);
      if (!s.group.some((g) => g.text === "Proposed Pillars" && g.shown)) bad.push('navigator: the "Proposed Pillars" group is not shown');
      const want = data.pillars.map((p) => `/roadmap/candidates/${p.slug}/`);
      const got = s.links.map((l) => new URL(l.href).pathname);
      if (got.join() !== want.join()) bad.push(`navigator: pillar hrefs differ from the data (${short(got.filter((g, i) => g !== want[i]), 2)})`);
      await status200(s.links.map((l) => l.href), bad, "navigator link");
      if (s.scrollW > s.clientW + 1) bad.push(`navigator scrolls sideways (${s.scrollW} > ${s.clientW})`);
      await shot(ph, "navigator-phone.png", true);
      const first = data.pillars[0];
      await ph.locator(`.roadmap-pillars a[data-pillar="${first.prefix}"]`).tap();
      await ph.waitForURL(`**/roadmap/candidates/${first.slug}/`, { timeout: 8000 }).catch(() => bad.push(`navigator: tapping ${first.prefix} did not open its page (at ${new URL(ph.url()).pathname})`));

      // Graph page
      await ph.goto(ROUTE, { waitUntil: "load" });
      await settle("#roadmap-graph");
      s = await state("#roadmap-graph");
      if (s.canvas === "missing") bad.push("graph: no canvas element");
      else if (s.canvas) bad.push("graph: the canvas is visible on a phone");
      const gShown = s.links.filter((l) => l.shown);
      if (gShown.length !== graph.pillars.length) bad.push(`graph: ${gShown.length} pillar links shown, expected ${graph.pillars.length}`);
      const gWant = graph.pillars.map((p) => new URL(p.url, ROUTE).pathname);
      const gGot = s.links.map((l) => new URL(l.href).pathname);
      if (gGot.join() !== gWant.join() || gGot.some((g) => !/^\/roadmap\/[a-z0-9-]+\/$/.test(g))) bad.push(`graph: pillar hrefs ${short(gGot, 3)} differ from /roadmap/<slug>/ of graph.json`);
      await status200(s.links.map((l) => l.href), bad, "graph link");
      const cl = s.railLinks.find((l) => new URL(l.href).pathname === "/roadmap/candidates/");
      if (!cl) bad.push("graph: no Candidates rail link to /roadmap/candidates/");
      else if (!cl.shown) bad.push("graph: the Candidates rail link is hidden");
      if (s.headings.some((h) => h === "Pillars")) bad.push('graph: the article still has a "Pillars" heading');
      if (s.scrollW > s.clientW + 1) bad.push(`graph page scrolls sideways (${s.scrollW} > ${s.clientW})`);
      await shot(ph, "roadmap-phone.png", true);
      const gp = graph.pillars[0];
      await ph.locator(`.roadmap-pillars a[data-pillar="${gp.prefix}"]`).tap();
      await ph.waitForURL(`**${new URL(gp.url, ROUTE).pathname}`, { timeout: 8000 }).catch(() => bad.push(`graph: tapping ${gp.prefix} did not open its page (at ${new URL(ph.url()).pathname})`));
    } finally { await ctx.close(); }
    return bad;
  });

  // ---------- captures ----------
  if (capDir) {
    const cap = async (what, fn) => {
      try { await fn(); } catch (e) { failed = true; console.log(`FAIL capture ${what}: ${firstLine(e)}`); }
    };
    await cap("first view", async () => { await loadNav(); await page.waitForTimeout(300); await shot(page, "navigator-first-view.png"); });
    await cap("proposed pillar", async () => {
      await loadNav();
      await page.evaluate(`${NAV}.goPillar("DAT")`);
      await page.waitForTimeout(300);
      await shot(page, "navigator-proposed-pillar.png");
    });
    await cap("hover", async () => {
      await loadNav();
      const box = await page.locator(".cn-card").nth(4).boundingBox();
      await page.mouse.move(2, 2);
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 4 });
      await page.waitForSelector(".roadmap-tooltip.is-open", { timeout: 3000 }).catch(() => console.log("INFO hover: tooltip did not open"));
      await page.waitForTimeout(250);
      await shot(page, "navigator-hover.png");
      await page.mouse.move(2, 2);
    });
    await cap("detail panel", async () => {
      await loadNav();
      const c = all.find((x) => x.depends_on.length === 3 && x.summary.length > 80) || all.find((x) => x.depends_on.length === 3) || all[0];
      const card = page.locator(`.cn-card[data-key="${c.key}"]`);
      await card.scrollIntoViewIfNeeded();
      await card.click();
      await page.mouse.move(2, 2);
      await page.waitForTimeout(300);
      await shot(page, "navigator-detail-panel.png");
    });
    await cap("search", async () => {
      await loadNav();
      let pick = null;
      for (const c of all) { const w = summaryWord(c); if (w && w.n >= 3 && w.n <= 8) { pick = w.w; break; } }
      if (!pick) pick = "render";
      const input = page.locator(".roadmap-search input");
      await input.fill(pick);
      await page.waitForTimeout(400);
      await input.press("Enter");
      await page.waitForTimeout(400);
      await shot(page, "navigator-search.png");
    });
  }
} catch (e) {
  failed = true;
  console.log(`FAIL setup: ${firstLine(e)}`);
}

// ---------- 8. console and network ----------
await check(8, "no console errors, page errors or requests to another origin", () => {
  const bad = [];
  if (errors.length) bad.push(`errors: ${short([...new Set(errors)], 3)}`);
  if (foreign.size) bad.push(`foreign requests: ${short([...foreign], 3)}`);
  return bad;
});

await browser.close();
await new Promise((r) => server.close(r));
process.exit(failed ? 1 : 0);
