# Caliper Public Landing Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a single hand-designed static case-study page, deployed from `site/` to `https://ahmedkhan-gh.github.io/caliper/` via GitHub Actions.

**Architecture:** Hand-authored HTML + CSS in `site/` (no framework, no bundler). Copy is distilled from the repo's own docs (WHITEPAPER.md, ZEROCOPY.md, GEOMETRY.md, PLATFORM.md) into a copy deck, then assembled into one page styled by a small design-token CSS system. A GitHub Actions workflow uploads `site/` to Pages on push to `main`.

**Tech Stack:** Static HTML5/CSS3, optional minimal vanilla JS, GitHub Actions (`configure-pages` / `upload-pages-artifact` / `deploy-pages`), Python `http.server` for local preview, Playwright (via design skill) for screenshots.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-11-caliper-pages-site-design.md` — binding.
- All asset/page URLs **relative** (`assets/style.css`, never `/assets/...`) — page must work under the `/caliper/` base path.
- Hero claim verbatim: "GPU tensors become pixels without ever touching the CPU — byte-exact on Metal and Vulkan."
- Copy voice: engineering copy with confidence; every claim paired with its proving artifact; no marketing superlatives ("blazing", "revolutionary" are failures).
- Placeholder slots are first-class: designed frames, honestly labeled ("demo recording coming"), drop-in replaceable by swapping one asset file / one `src`.
- Net-new files only: `site/**`, `.github/workflows/pages.yml`, this plan. Never touch `CMakeLists.txt`, `src/**`, `README.md`. Never run cmake (concurrent libcaliper agent).
- Commit after every task, on branch `worktree-pages-site`.
- The design skill's render → screenshot → critique loop is mandatory before the page is called done (desktop 1440px and mobile 390px).

## File Structure

```
site/
  index.html              # the whole page
  assets/
    style.css             # design tokens + all styling
    placeholder-hero.svg  # 1200×675 (16:9) hero demo slot
    placeholder-applet-1.svg  # 800×500 gallery slots (mesh_scope)
    placeholder-applet-2.svg  # (instance_scope)
    placeholder-applet-3.svg  # (repnet_demo Training Lab)
.github/workflows/pages.yml
docs/superpowers/plans/2026-07-11-caliper-pages-site-copydeck.md  # Task 2 output, consumed by Task 4
```

Section IDs in `index.html` (fixed interface): `#hero`, `#problem`, `#hard-part`, `#built`, `#proof`, `#gallery`, `#links` (footer).

---

### Task 1: Deploy workflow + site scaffold

**Files:**
- Create: `.github/workflows/pages.yml`
- Create: `site/index.html` (skeleton)
- Create: `site/assets/style.css` (empty stub with token block comment)

**Interfaces:**
- Produces: the seven section IDs above; `site/assets/style.css` path; workflow deploying `site/` on push to `main`.

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/pages.yml
name: Deploy Pages
on:
  push:
    branches: [main]
    paths: ['site/**', '.github/workflows/pages.yml']
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: pages
  cancel-in-progress: true
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: site
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Write the HTML skeleton**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Caliper — zero-copy ML instruments, tensors to pixels</title>
  <meta name="description" content="Caliper is a native ML instrument platform: GPU tensors become pixels without ever touching the CPU — byte-exact on Metal and Vulkan.">
  <link rel="stylesheet" href="assets/style.css">
</head>
<body>
  <main>
    <section id="hero"></section>
    <section id="problem"></section>
    <section id="hard-part"></section>
    <section id="built"></section>
    <section id="proof"></section>
    <section id="gallery"></section>
  </main>
  <footer id="links"></footer>
</body>
</html>
```

- [ ] **Step 3: Verify locally**

Run: `python3 -m http.server 8765 --directory site &` then `curl -s http://localhost:8765/ | head -5`
Expected: the doctype and `<html lang="en">` echoed back; no 404.

- [ ] **Step 4: Validate workflow YAML**

Run: `python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/pages.yml')); print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/pages.yml site/
git commit -m "feat(site): Pages deploy workflow + page scaffold"
```

---

### Task 2: Copy deck (token-heavy — Opus worker)

**Files:**
- Create: `docs/superpowers/plans/2026-07-11-caliper-pages-site-copydeck.md`
- Read (sources): `WHITEPAPER.md`, `ZEROCOPY.md`, `GEOMETRY.md` (§11 and the R0–R3 ladder), `PLATFORM.md` (services list, D-decisions), `APPLETS.md`, `docs/wiki/index.md`

**Interfaces:**
- Produces: one markdown file with a `## <section-id>` heading per page section (`hero`, `problem`, `hard-part`, `built`, `proof`, `gallery`, `links`), containing FINAL copy: headings, body paragraphs, card titles+one-liners, image captions, alt text, and exact link hrefs. Task 4 pastes from this file verbatim.

- [ ] **Step 1: Read the six source docs and distill.** Requirements:
  - `hero`: the verbatim hero claim; a one-line descriptor of Caliper; one supporting sub-line; CTA labels ("View on GitHub", "Read the whitepaper").
  - `problem`: ≤120 words — why ML tooling round-trips GPU→CPU→GPU and what that caps.
  - `hard-part`: ≤150 words + 2 side-by-side platform notes (MPS unified memory vs CUDA external-memory import with ~2 MiB VMM padding), and byte-exact verification as the bar.
  - `built`: 5–6 cards, each title + ≤15-word one-liner: applet contract (C ABI, epochs), 8 host-neutral services (name them), HostRenderer (Metal/Vulkan), tensor bridge, geometry ladder R0–R3, golden-applet wall / test suites.
  - `proof`: intro line + the byte-exact matrix as a markdown table (backend × geometry rows, values from GEOMETRY.md — real row names, no invented data) + one provenance line quoted from the docs.
  - `gallery`: 3 captions + alt text for mesh_scope, instance_scope, repnet_demo Training Lab — say what each shows, one line each.
  - `links`: hrefs — `https://github.com/AhmedKhan-GH/caliper`, `https://github.com/AhmedKhan-GH/caliper/blob/main/WHITEPAPER.md`, `https://github.com/AhmedKhan-GH/caliper/tree/main/docs/wiki` (label: "Reference docs"), plus a `mailto:emailahmedebadkhan@gmail.com` contact.
  - Every factual claim must be traceable to a source doc; if a number can't be found, omit it rather than invent it.

- [ ] **Step 2: Self-check** — grep the deck for banned marketing words (`blazing|revolutionary|cutting-edge|world-class|game-chang`), confirm every `## <section-id>` present.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-07-11-caliper-pages-site-copydeck.md
git commit -m "docs(site): copy deck for landing page"
```

---

### Task 3: Placeholder assets (Opus worker, parallel with Task 2)

**Files:**
- Create: `site/assets/placeholder-hero.svg` (viewBox `0 0 1200 675`)
- Create: `site/assets/placeholder-applet-1.svg`, `-2.svg`, `-3.svg` (viewBox `0 0 800 500`)

**Interfaces:**
- Produces: the four SVG files at those exact paths; dark-ground frames that read as intentional design, each labeled honestly. Colors must use only these literals so they harmonize with Task 4's tokens: bg `#0B0E14`, surface `#131826`, line `#2A3245`, text `#8B94A7`, accent `#5EEAD4`.

- [ ] **Step 1: Author the four SVGs.** Each: dark ground, subtle 40px grid or corner-tick motif suggesting an instrument/oscilloscope face, centered label. Hero label: `DEMO RECORDING — COMING SOON` over sub-label `zero-copy tensors → pixels, captured live`. Applet labels: `mesh_scope`, `instance_scope`, `repnet_demo — training lab`, each over `recording coming soon`. Pattern (hero shown; applets same recipe at 800×500):

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 675" role="img" aria-label="Placeholder: demo recording coming soon">
  <rect width="1200" height="675" fill="#0B0E14"/>
  <!-- grid, frame, ticks in #2A3245 / surface panels in #131826 -->
  <!-- labels: text #8B94A7, small accent details only in #5EEAD4 -->
</svg>
```

- [ ] **Step 2: Verify they render**

Run: `for f in site/assets/placeholder-*.svg; do python3 -c "import xml.dom.minidom,sys; xml.dom.minidom.parse('$f'); print('$f OK')"; done`
Expected: four `OK` lines. Then screenshot one in a browser (design-skill loop applies at Task 5 anyway).

- [ ] **Step 3: Commit**

```bash
git add site/assets/placeholder-*.svg
git commit -m "feat(site): designed placeholder frames for demo recordings"
```

---

### Task 4: Page build — HTML + design-system CSS (token-heavy — Opus worker, after Tasks 1–3)

**Files:**
- Modify: `site/index.html` (fill all sections from the copy deck, verbatim)
- Modify: `site/assets/style.css` (full design system)

**Interfaces:**
- Consumes: copy deck (`## <section-id>` blocks), placeholder SVG paths, section IDs from Task 1.
- Produces: the finished page.

- [ ] **Step 1: Invoke the design skill** (mandatory) and follow its knowledge-graph grounding for direction. Design constraints, non-negotiable:
  - Dark engineering aesthetic; CSS custom properties on `:root` exactly: `--bg:#0B0E14; --surface:#131826; --line:#2A3245; --text:#E6EAF2; --muted:#8B94A7; --accent:#5EEAD4;` (matches Task 3 SVGs).
  - Type: `Inter` (text) + `JetBrains Mono` (code/labels/matrix) via Google Fonts `<link>`; system-stack fallbacks.
  - Max content width 72rem; generous vertical rhythm; proof matrix rendered as a real `<table>` styled monospace with green checks in `--accent`.
  - Hero: claim as the `<h1>`, placeholder-hero.svg in a framed `<figure>` with visible caption.
  - `#built` cards as a responsive grid (`repeat(auto-fit, minmax(16rem, 1fr))`).
  - Fully responsive at 390px; no horizontal scroll; JS optional and only for scroll-reveal (page must be complete with JS disabled).
- [ ] **Step 2: Assemble HTML** — paste copy deck text verbatim into the section skeleton; all links from the deck's `links` block; `alt` text from the deck.
- [ ] **Step 3: Local render check**

Run: `python3 -m http.server 8765 --directory site` and screenshot at 1440px and 390px widths (design skill loop: render → screenshot → critique → fix, minimum one full cycle, repeat until no critique-blocking issues).

- [ ] **Step 4: Link + asset check**

Run: `python3 - <<'EOF'
import re,os
html=open('site/index.html').read()
for m in re.findall(r'(?:src|href)="([^"#][^"]*)"',html):
    if m.startswith('http') or m.startswith('mailto'): continue
    assert os.path.exists(os.path.join('site',m)), f'MISSING {m}'
print('local refs OK')
EOF`
Expected: `local refs OK`

- [ ] **Step 5: Commit**

```bash
git add site/
git commit -m "feat(site): full landing page — copy, design system, proof matrix"
```

---

### Task 5: Final verification + ship

**Files:**
- Modify: whatever the critique loop demands (site/ only)

- [ ] **Step 1: Fresh-eyes critique pass** — re-screenshot both widths; check the spec's acceptance list one by one (claim lands ≤10s, story ≤2min, placeholders look intentional, matrix real, links live).
- [ ] **Step 2: External-link check**

Run: `for u in https://github.com/AhmedKhan-GH/caliper https://github.com/AhmedKhan-GH/caliper/blob/main/WHITEPAPER.md https://github.com/AhmedKhan-GH/caliper/tree/main/docs/wiki; do curl -s -o /dev/null -w "%{http_code} $u\n" "$u"; done`
Expected: three `200` lines.

- [ ] **Step 3: Collision audit**

Run: `git diff --name-only $(git merge-base HEAD origin/main)..HEAD | grep -vE '^(site/|\.github/workflows/pages\.yml|docs/superpowers/)' || echo "CLEAN"`
Expected: `CLEAN`

- [ ] **Step 4: Push branch + open PR to `main`**

```bash
git push -u origin worktree-pages-site
gh pr create --base main --title "feat(site): public landing page — GitHub Pages case study" --body "$(cat <<'EOF'
Ships the Caliper public case-study page per docs/superpowers/specs/2026-07-11-caliper-pages-site-design.md.
Net-new files only (site/, pages workflow); zero overlap with feat/libcaliper.
After merge: flip Settings → Pages → Source: GitHub Actions.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: Hand the owner the one manual step** — Settings → Pages → Source: "GitHub Actions" (or approve `gh api repos/AhmedKhan-GH/caliper/pages -X POST -f build_type=workflow`).
