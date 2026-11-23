// The spec page's layout (src/spec/layout.ts) against the cases of the
// native spec runtime's tests (src/host/spec_runtime/tests/test_spec_layout.cpp),
// so that the browser and the app place every panel on the same pixels. The
// fixtures are the spec library's TOML files, read with Python's tomllib, whose
// tables are the canonical JSON form. Plain Node: it runs in one project only.

import { test, expect } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { computeLayout, effectiveLayout, layoutFixedHeight, layoutMinSize, layoutShares, metricsForTextScale, panelTitle,
  bindingName, findPanel, type Rect, type SpecLayout, type TabChoice } from '../src/spec/layout.js';
import type { Spec, SplitNode } from '../src/spec/types.js';
import { REPO } from './util.js';

test.beforeEach(({}, info) => test.skip(info.project.name !== 'swiftshader', 'pure layout arithmetic: one project is enough'));

const FIXTURES = path.join(REPO, 'src', 'builder', 'spec', 'tests', 'fixtures');

// tomllib needs Python 3.11 or newer: CALIPER_PYTHON, else python3.
const PYTHON = process.env.CALIPER_PYTHON ?? 'python3';

function fixture(name: string): Spec {
  return toml(path.join(FIXTURES, name), true);
}

/** A spec from TOML, a file or its text, through Python's tomllib. */
function toml(source: string, file = false): Spec {
  let text: Buffer;
  try {
    text = execFileSync(PYTHON, ['-I', '-c', file
      ? 'import json, sys, tomllib; print(json.dumps(tomllib.load(open(sys.argv[1], "rb"))))'
      : 'import json, sys, tomllib; print(json.dumps(tomllib.loads(sys.stdin.read())))', ...(file ? [source] : [])],
    file ? {} : { input: source });
  } catch {
    test.skip(true, 'needs a Python with tomllib (3.11 or newer) in CALIPER_PYTHON or as python3');
    throw new Error('unreachable');
  }
  return JSON.parse(text.toString()) as Spec;
}

function lay(s: Spec, w: number, h: number, scale = 1, tabs: TabChoice = {}): SpecLayout {
  const root = effectiveLayout(s);
  expect(root).not.toBeNull();
  return computeLayout(root!, s, { x: 0, y: 0, w, h }, metricsForTextScale(scale), tabs);
}

const rectOf = (l: SpecLayout, id: string): Rect | undefined => l.panels.find((p) => p.id === id)?.rect;
const same = (r: Rect | undefined, x: number, y: number, w: number, h: number) => expect(r).toEqual({ x, y, w, h });

// Every panel inside the area, on whole pixels, and no two overlapping.
function checkTiling(l: SpecLayout, w: number, h: number) {
  for (const p of l.panels) {
    expect(p.rect.x).toBeGreaterThanOrEqual(0);
    expect(p.rect.y).toBeGreaterThanOrEqual(0);
    expect(p.rect.x + p.rect.w).toBeLessThanOrEqual(w);
    expect(p.rect.y + p.rect.h).toBeLessThanOrEqual(h);
    for (const v of [p.rect.x, p.rect.y, p.rect.w, p.rect.h]) expect(Number.isInteger(v)).toBe(true);
  }
  for (let i = 0; i < l.panels.length; i++)
    for (let j = i + 1; j < l.panels.length; j++) {
      const a = l.panels[i].rect, b = l.panels[j].rect;
      const apart = a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y;
      expect(apart, `${l.panels[i].id} overlaps ${l.panels[j].id}`).toBe(true);
    }
}

test('the dashboard at 2560x1440 and 1440x900', () => {
  const s = fixture('dashboard.spec.toml');
  let l = lay(s, 2560, 1440);
  expect(l.panels.length).toBe(3);
  expect(l.tabs).toEqual([]);
  same(rectOf(l, 'curves'), 0, 0, 1280, 935);
  same(rectOf(l, 'attention'), 1281, 0, 1279, 935);
  same(rectOf(l, 'rows'), 0, 936, 2560, 504);
  expect(l.panels.find((p) => p.id === 'curves')!.path).toBe('0.0');
  expect(l.panels.find((p) => p.id === 'rows')!.path).toBe('1');
  checkTiling(l, 2560, 1440);
  l = lay(s, 1440, 900);
  same(rectOf(l, 'curves'), 0, 0, 720, 584);
  same(rectOf(l, 'attention'), 721, 0, 719, 584);
  same(rectOf(l, 'rows'), 0, 585, 1440, 315);
  checkTiling(l, 1440, 900);
  // An area that does not start at the origin moves every rectangle with it.
  l = computeLayout(effectiveLayout(s)!, s, { x: 10, y: 50, w: 1440, h: 900 }, metricsForTextScale(1));
  same(rectOf(l, 'curves'), 10, 50, 720, 584);
  same(rectOf(l, 'rows'), 10, 635, 1440, 315);
});

test('minimums hold while the space allows and grow with the text scale', () => {
  const s = fixture('dashboard.spec.toml');
  const m1 = metricsForTextScale(1), m2 = metricsForTextScale(2);
  expect([m1.minPanelW, m1.minPanelH, m2.minPanelW, m2.minPanelH]).toEqual([180, 120, 360, 240]);
  expect(m2.tabBarH).toBeGreaterThan(m1.tabBarH);
  expect([m1.gap, m2.gap]).toEqual([1, 1]);
  // At 400x300 the ratio 0.65 would leave the table 104 px; it keeps its 120.
  let l = lay(s, 400, 300);
  expect(rectOf(l, 'rows')!.h).toBe(120);
  expect(rectOf(l, 'curves')!.h).toBe(179);
  checkTiling(l, 400, 300);
  // At twice the text scale the minimum is 240, in an 800x600 area.
  l = lay(s, 800, 600, 2);
  expect(rectOf(l, 'rows')!.h).toBe(240);
  expect(rectOf(l, 'curves')!.h).toBe(359);
  expect(rectOf(l, 'curves')!.w).toBeGreaterThanOrEqual(360);
  expect(rectOf(l, 'attention')!.w).toBeGreaterThanOrEqual(360);
  checkTiling(l, 800, 600);
  // At scale 1 the same area keeps the ratio.
  l = lay(s, 800, 600, 1);
  expect(rectOf(l, 'curves')!.h).toBe(Math.round(Math.fround(0.65 * 599)));
  // Below every minimum the shares follow the minimums, and nothing leaves the area.
  l = lay(s, 200, 100);
  checkTiling(l, 200, 100);
  expect(rectOf(l, 'curves')!.h).toBeGreaterThan(0);
  expect(rectOf(l, 'rows')!.h).toBeGreaterThan(0);
  expect(rectOf(l, 'curves')!.h + 1 + rectOf(l, 'rows')!.h).toBe(100);
});

test('the shares of a split', () => {
  expect(layoutShares(100, [10, 10], 0.25)).toEqual([25, 75]);
  expect(layoutShares(100, [10, 80], 0.5)).toEqual([20, 80]);
  expect(layoutShares(100, [60, 10], 0.1)).toEqual([60, 40]);
  expect(layoutShares(300, [10, 10, 10], 0.5)).toEqual([100, 100, 100]);
  expect(layoutShares(300, [150, 10, 10], 0.5)).toEqual([150, 75, 75]);
  const odd = layoutShares(1438, [1, 1, 1], 0.5);
  expect(odd.reduce((a, b) => a + b, 0)).toBe(1438);
  for (const v of odd) expect(Number.isInteger(v)).toBe(true);
  expect(layoutShares(100, [100, 300], 0.5)).toEqual([25, 75]);
  expect(layoutShares(0, [10, 10], 0.5)).toEqual([0, 0]);
});

test('tabs show one child under a strip of labels', () => {
  const s = fixture('every-primitive.spec.toml');
  const tab = metricsForTextScale(1).tabBarH;
  let l = lay(s, 1440, 900);
  expect(l.tabs.length).toBe(1);
  expect(l.tabs[0].path).toBe('');
  expect(l.tabs[0].active).toBe(0);
  expect(l.tabs[0].labels).toEqual(['Tensors', 'Tables', 'Text']);
  expect(l.tabs[0].bar).toEqual({ x: 0, y: 0, w: 1440, h: tab });
  // The first tab: a horizontal split of three equal shares.
  expect(l.panels.length).toBe(3);
  same(rectOf(l, 's'), 0, tab, 479, 900 - tab);
  same(rectOf(l, 'h'), 480, tab, 480, 900 - tab);
  same(rectOf(l, 'p3'), 961, tab, 479, 900 - tab);
  expect(rectOf(l, 'img')).toBeUndefined();
  expect(l.panels.find((p) => p.id === 's')!.tab).toBe(false);
  checkTiling(l, 1440, 900);
  // The second tab holds a nested tabs node; its first tab shows.
  l = lay(s, 2560, 1440, 1, { '': 1 });
  expect(l.tabs.length).toBe(2);
  expect(l.tabs[1].path).toBe('1.1');
  expect(l.tabs[1].labels).toEqual(['sample_png', 'run42_metrics', 'gpu']);
  expect(l.panels.length).toBe(2);
  const p2 = l.panels.find((p) => p.id === 'p2')!, img = l.panels.find((p) => p.id === 'img')!;
  expect(p2.rect.y).toBe(tab);
  expect(p2.rect.h).toBe(Math.round(Math.fround(0.3 * (1440 - tab - 1))));
  expect(l.tabs[1].bar.y).toBe(p2.rect.y + p2.rect.h + 1);
  expect(img.rect.y).toBe(l.tabs[1].bar.y + l.tabs[1].bar.h);
  expect(img.rect.y + img.rect.h).toBe(1440);
  expect(img.tab).toBe(true);
  expect(p2.tab).toBe(false);
  checkTiling(l, 2560, 1440);
  l = lay(s, 1440, 900, 1, { '': 1, '1.1': 2 });
  expect(rectOf(l, 'gpu')).toBeDefined();
  expect(rectOf(l, 'img')).toBeUndefined();
  expect(l.tabs[1].active).toBe(2);
  // A choice out of range shows the first tab.
  l = lay(s, 1440, 900, 1, { '': 9 });
  expect(l.tabs[0].active).toBe(0);
  expect(rectOf(l, 's')).toBeDefined();
  l = lay(s, 1440, 900, 1, { '': 2 });
  expect(rectOf(l, 'note')).toBeDefined();
  expect(l.tabs.length).toBe(2);
  expect(l.tabs[1].labels).toEqual(['Text', 'Last value']);
});

test('a spec without a layout stacks its panels', () => {
  const s = fixture('dashboard.spec.toml');
  delete s.layout;
  const l = lay(s, 1000, 902);
  expect(l.panels.length).toBe(3);
  same(l.panels[0].rect, 0, 0, 1000, 300);
  same(l.panels[1].rect, 0, 301, 1000, 300);
  same(l.panels[2].rect, 0, 602, 1000, 300);
  expect(l.panels[0].id).toBe('curves');
  s.panel = s.panel!.slice(0, 1);
  const one = lay(s, 640, 480);
  expect(one.panels.length).toBe(1);
  same(one.panels[0].rect, 0, 0, 640, 480);
});

test('a panel without a title goes by what it draws, never its id', () => {
  const d = fixture('dashboard.spec.toml');
  expect(panelTitle(d, findPanel(d, 'curves')!)).toBe('Loss and accuracy');
  expect(panelTitle(d, findPanel(d, 'rows')!)).toBe('run42_metrics');
  const e = fixture('every-primitive.spec.toml');
  expect(panelTitle(e, findPanel(e, 'h')!)).toBe('acts');
  expect(panelTitle(e, findPanel(e, 'p3')!)).toBe('embedding');
  expect(panelTitle(e, findPanel(e, 'p2')!)).toBe('value by step');
  expect(panelTitle(e, findPanel(e, 'img')!)).toBe('sample_png');
  expect(panelTitle(e, findPanel(e, 't')!)).toBe('run42_metrics');
  expect(panelTitle(e, findPanel(e, 'gpu')!)).toBe('gpu');
  expect(panelTitle(e, findPanel(e, 'note')!)).toBe('Text');
  expect(bindingName(e, 'probs')).toBe('probs');
  expect(bindingName(e, 'metrics.value')).toBe('value');
});

// ---- spec version 1: control leaves, shares and leaf minimums ---------------

test('a row of controls takes one control row under the panels', () => {
  const s = fixture('instruments.spec.toml');
  expect(s.control!.length).toBe(2);
  const m = metricsForTextScale(1);
  const l = lay(s, 1200, 800);
  expect(l.panels.length).toBe(3);
  const world = l.panels.find((p) => p.id === 'world')!, lr = l.panels.find((p) => p.id === 'lr')!;
  const pause = l.panels.find((p) => p.id === 'pause')!;
  expect([world.control, lr.control, pause.control]).toEqual([false, true, true]);
  expect(lr.rect.h).toBe(m.controlRowH);
  expect(pause.rect.h).toBe(m.controlRowH);
  expect(lr.rect.y).toBe(pause.rect.y);
  expect(lr.rect.w + pause.rect.w + m.gap).toBe(1200);
  // The scene panel takes the rest of the height.
  expect(world.rect.h + m.gap + lr.rect.h).toBe(800);
  expect(lr.rect.y + lr.rect.h).toBe(800);
  checkTiling(l, 1200, 800);
  const root = effectiveLayout(s)!;
  expect(layoutFixedHeight(root, s, m)).toBeNull();
  expect(layoutFixedHeight((root as SplitNode).children[1], s, m)).toBe(m.controlRowH);
});

test('the default layout of a spec with controls and its minimums', () => {
  const s = fixture('instruments.spec.toml');
  delete s.layout;
  const m = metricsForTextScale(1);
  const l = lay(s, 900, 600);
  expect(l.panels.length).toBe(3);
  expect(rectOf(l, 'lr')!.h).toBe(m.controlRowH);
  expect(rectOf(l, 'world')!.y).toBe(0);
  const root = effectiveLayout(s)!;
  const min = layoutMinSize(root, m, s);
  expect(min.h).toBe(m.minPanelH + m.gap + m.controlRowH);
  expect(min.w).toBe(Math.max(m.minPanelW, 2 * m.minControlW + m.gap));
  // Without the spec, every leaf counts as a panel.
  expect(layoutMinSize(root, m).h).toBe(2 * m.minPanelH + m.gap);
  // Only controls: the rows stack from the top and the rest stays empty.
  const only: Spec = { ...s, panel: [] };
  const lc = lay(only, 900, 600);
  expect(lc.panels.length).toBe(2);
  expect(lc.panels[0].rect.h).toBe(m.controlRowH);
  expect(lc.panels[0].rect.y).toBe(0);
});

test('a scene panel goes by its scene when it has no title', () => {
  const s = fixture('instruments.spec.toml');
  const p = { ...findPanel(s, 'world')! };
  expect(panelTitle(s, p)).toBe('Model');
  delete p.title;
  expect(panelTitle(s, p)).toBe('model');
});

const SHARES = `caliper_spec = 1
[applet]
id = "shares"
[[source]]
id = "w"
catalog = "tensor:w"
[[panel]]
id = "world"
primitive = "heatmap"
bind = { values = "w" }
[[panel]]
id = "detail"
primitive = "heatmap"
bind = { values = "w" }
[[panel]]
id = "loss"
primitive = "series"
bind = { y = "w" }
[[control]]
id = "lr"
kind = "slider"
bind = "value"
options = { min = 0.0, max = 1.0 }
[[control]]
id = "go"
kind = "button"
bind = "action:go"
[layout]
split = "vertical"
shares = [6, 3, 1]
children = [
  { split = "horizontal", shares = [3, 2], children = [{ panel = "world", min = [48, 20] }, "detail"] },
  "loss",
  { split = "horizontal", children = ["lr", "go"] },
]
`;

test('shares split the space by weight, after the row of controls', () => {
  const s = toml(SHARES);
  const m = metricsForTextScale(1);
  for (const [w, h] of [[2560, 1440], [1440, 900]]) {
    const l = lay(s, w, h);
    checkTiling(l, w, h);
    const world = rectOf(l, 'world')!, detail = rectOf(l, 'detail')!, loss = rectOf(l, 'loss')!, lr = rectOf(l, 'lr')!;
    expect(Math.abs(world.h / loss.h - 2)).toBeLessThan(0.02);
    expect(Math.abs(world.w / (world.w + detail.w + 1) - 0.6)).toBeLessThan(0.01);
    expect(world.w).toBeGreaterThanOrEqual(Math.round(48 * m.charW));
    expect(world.h).toBeGreaterThanOrEqual(Math.round(20 * m.lineH));
    expect(lr.h).toBe(m.controlRowH);
  }
});

test('a share below its minimum keeps the minimum, and the others share the rest by weight', () => {
  const s = layoutShares(1000, [50, 50, 200], 0.5, [6, 3, 1]);
  expect(s[2]).toBe(200);
  expect(Math.abs(s[0] - 533)).toBeLessThanOrEqual(533 * 0.01);
  expect(s[0] + s[1] + s[2]).toBe(1000);
  expect(layoutShares(900, [10, 10], 0.5, [1, 2])).toEqual([300, 600]);
  expect(layoutShares(300, [200, 400], 0.5, [5, 1])).toEqual([100, 200]);
  expect(layoutShares(300, [10, 10, 10], 0.5, [])).toEqual(layoutShares(300, [10, 10, 10], 0.5));
});

test('a leaf minimum is in characters and lines of the text scale', () => {
  const one = metricsForTextScale(1), two = metricsForTextScale(2);
  const a = layoutMinSize({ panel: 'world', min: [48, 20] }, one), b = layoutMinSize({ panel: 'world', min: [48, 20] }, two);
  expect(a).toEqual({ w: Math.round(48 * one.charW), h: Math.round(20 * one.lineH) });
  expect(b.w).toBeGreaterThan(1.9 * a.w);
  expect(b.h).toBeGreaterThan(1.9 * a.h);
});

test('the Arc Tower fills the window over its row of controls', () => {
  // The layout that the acceptance driver builds (tests/programmable_graphics/driver.py, step 6), in canonical JSON.
  const s: Spec = {
    caliper_spec: 1, applet: { id: 'arc_tower', title: 'Attention Arc Tower' },
    panel: ['tower', 'pattern', 'head_scores', 'icl_curve', 'loss_curve'].map((id) => ({ id, primitive: 'series' as const })),
    control: [{ id: 'ablate', kind: 'toggle', bind: 'intervene:blocks.1.attn#0' }],
    layout: { split: 'vertical', children: [
      { split: 'horizontal', ratio: 0.5, children: [
        { split: 'vertical', ratio: 0.45, children: ['tower', { split: 'vertical', ratio: 0.62, children: ['head_scores',
          { tabs: ['icl_curve', 'loss_curve'] }] }] }, 'pattern'] }, 'ablate'] },
  };
  const m = metricsForTextScale(1);
  const l = lay(s, 1280, 750);
  checkTiling(l, 1280, 750);
  const ablate = l.panels.find((p) => p.id === 'ablate')!;
  expect(ablate.control).toBe(true);
  same(ablate.rect, 0, 750 - m.controlRowH, 1280, m.controlRowH);
  // The panels fill everything above the row: the grid reaches the rule over the control.
  const pattern = rectOf(l, 'pattern')!;
  expect(pattern.y).toBe(0);
  expect(pattern.y + pattern.h + m.gap).toBe(ablate.rect.y);
});
