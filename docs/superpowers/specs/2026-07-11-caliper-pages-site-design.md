# Caliper public landing page — GitHub Pages case study

**Date:** 2026-07-11
**Status:** DESIGN — approved in conversation 2026-07-11; this doc records it.
**Goal hierarchy:** B (portfolio narrative on a public page) primary, A (a link a
recruiter can absorb in 30 seconds) secondary. C (drive framework adoption) is
explicitly NOT this artifact's job — the mkdocs wiki serves C and is linked as
evidence, not merged into this page.

---

## 1. One paragraph

Ship a single hand-designed static case-study page at
`https://ahmedkhan-gh.github.io/caliper/`, deployed from a new `site/` folder
in this repo via GitHub Actions. The page tells the Caliper story — GPU
tensors become pixels without touching the CPU, proven byte-exact on Metal and
Vulkan — as an engineering narrative with visual proof slots, and links out to
the repo, whitepaper, and (when published) the mkdocs reference wiki. It ships
this week with clearly-designed placeholder slots for demo GIFs/recordings the
owner will capture later.

## 2. What it is / is not

| Is | Is not |
|---|---|
| One scrolling case-study/landing page, hand-designed HTML+CSS | The mkdocs "Caliper Platform" docs site (separate artifact, linked to) |
| The story: problem → hard part → what was built → proof | A spec sheet, feature matrix, or README mirror |
| Static assets only; zero build-time dependencies beyond the Actions workflow | A JS app, WASM demo, or anything requiring the C++ toolchain |
| Personal-domain-ready (CNAME swap later, zero rework) | Tied to github.io permanently |

## 3. Page content (v1)

1. **Hero** — the claim in one sentence: *"GPU tensors become pixels without
   ever touching the CPU — byte-exact on Metal and Vulkan."* Name + one-line
   descriptor of Caliper (native ML instrument platform). Primary demo slot:
   a designed placeholder sized for the future zero-copy GIF, visibly labeled
   as "demo recording coming" (honest-provenance house style applies to the
   page itself).
2. **The problem** — why ML tooling round-trips through the CPU and why that
   caps what an instrument can show.
3. **The hard part** — zero-copy across two GPU ecosystems: MPS unified
   memory vs CUDA external-memory import; byte-exact verification as the bar.
4. **What I built** — the platform in supporting-cast order: applet contract
   (C ABI, versioned epochs), 8 host-neutral services, HostRenderer
   (Metal/Vulkan), geometry ladder R0–R3, the golden-applet wall. Compact
   visual treatment (cards/strip), not prose walls.
5. **Proof strip** — the byte-exact matrix rendered as a visual (backends ×
   rows, all green), test-suite count, both-ecosystem provenance lines.
6. **Applet gallery** — 2–4 applet screenshots/GIF slots (mesh_scope,
   instance_scope, repnet_demo Training Lab) with one-line captions;
   placeholders in v1.
7. **Footer / links out** — GitHub repo, WHITEPAPER.md, mkdocs wiki (once
   published), contact/personal site.

Copy voice: engineering copy with confidence — claims paired with the
artifact that proves them; no marketing superlatives.

## 4. Mechanics

- **Files:** `site/index.html` + `site/assets/` (css, images, future gifs).
  No framework, no bundler; hand-authored HTML/CSS (+ minimal vanilla JS only
  if the design needs it, e.g. scroll reveal — optional).
- **Deploy:** `.github/workflows/pages.yml` — on push to `main` (path-filtered
  to `site/**` and the workflow itself), upload `site/` via
  `actions/upload-pages-artifact` → `actions/deploy-pages`. Modern Actions
  mechanism; leaves room to later deploy the mkdocs wiki under `/docs/`.
- **One-time manual step:** repo Settings → Pages → Source: "GitHub Actions"
  (owner clicks, or `gh api` with owner's approval).
- **Design process:** the design skill's render → screenshot → critique loop
  is mandatory before delivery; the page itself is evidence of finishing to a
  standard.

## 5. Constraints (active this week)

- **Concurrent-agent collision policy:** another agent is actively building
  libcaliper on `feat/libcaliper`. This work happens in an isolated worktree
  (`worktree-pages-site`, based on `origin/main` = c93eaf2), adds only
  net-new files (`site/`, `.github/workflows/pages.yml`, this spec), never
  edits `CMakeLists.txt`/`src/`, and never invokes cmake. Merge order with
  libcaliper is irrelevant; the Pages workflow fires only on push to `main`.
- **Placeholders are first-class:** the page ships before recordings exist.
  Slots must look intentional (designed frames with captions), not broken.

## 6. Out of scope (v1)

Custom domain/CNAME (later, one file + DNS); publishing the mkdocs wiki
(separate task; the link slot ships pointing at the repo docs until then);
analytics; blog; interactive/WASM demos (likely permanently — libtorch/Metal
does not compile to WASM); any change to README or other repo docs.

## 7. Acceptance

- Page live at `ahmedkhan-gh.github.io/caliper` after merge + Pages source
  flip.
- Reads as a story: a stranger gets the claim in ≤10 seconds and the
  narrative in ≤2 minutes.
- Passed the design skill's screenshot-critique loop (desktop + mobile
  widths).
- Zero collisions: `git log feat/libcaliper..worktree-pages-site` touches no
  shared files.
- GIF slots drop-in ready: replacing a placeholder image with a real
  recording requires editing nothing but the asset file (same filename) or a
  single `src`.
