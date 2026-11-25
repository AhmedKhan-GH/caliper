// Drawing a spec's panels on a canvas as the native spec runtime draws them
// (draw_spec_panel, draw_tabs and the primitives in
// src/host/spec_runtime/spec_runtime.cpp): the title strip with the title and
// the info string, the content under it, a state in place of missing data, a
// series and a histogram as ImPlot plots, a heatmap and an image over index
// axes with a heatmap's colorbar, points as squares with their colour key, the
// house table and a text panel. All positions are CSS pixels.

import { fmt4 } from './format.js';
import type { LayoutTabs, Rect } from './layout.js';
import { COL, ColorKey, ROW, SIZE, classColor, fill, fitText, outline, pointSize, text, textWidth, trackedText, trackedWidth } from './look.js';
import { Plot2D, fitLimits, logTicks, type Tick } from './plot.js';
import { lut, type PanelState, type PreparedPanel } from './prepare.js';

const AXES = SIZE.axes;               // the live views' axes in a window: mono at 13
const K = AXES / 10.5;

/** The state of a panel: a short noun phrase and, under it in mono, its detail, centred (draw_state). */
export function drawState(ctx: CanvasRenderingContext2D, st: PanelState, c: Rect): void {
  const label = fitText(ctx, 'ui', SIZE.ui, st.label, c.w);
  const detail = fitText(ctx, 'mono', SIZE.mono - 1, st.detail, c.w);
  const aw = textWidth(ctx, 'ui', SIZE.ui, label), bw = textWidth(ctx, 'mono', SIZE.mono - 1, detail);
  const gap = detail ? 4 : 0;
  const top = c.y + Math.max(0, (c.h - SIZE.ui - gap - (detail ? SIZE.mono - 1 : 0)) / 2);
  const cx = c.x + c.w / 2;
  text(ctx, 'ui', SIZE.ui, cx - aw / 2, top, COL.secondary, label);
  if (detail) text(ctx, 'mono', SIZE.mono - 1, cx - bw / 2, top + SIZE.ui + gap, COL.faint, detail);
}

/** What draw_spec_panel returns: the header and the content's rectangles. */
export interface PanelFrame { header: Rect; content: Rect }

/** Per panel state that outlives a frame: a table's scroll. */
export interface PanelView { scroll: number }

export function drawPanel(ctx: CanvasRenderingContext2D, p: PreparedPanel | null, title: string, rect: Rect,
                          showTitle: boolean, key: ColorKey, view: PanelView): PanelFrame {
  fill(ctx, rect.x, rect.y, rect.w, rect.h, COL.ground);
  const hh = showTitle ? Math.min(rect.h, Math.round(SIZE.ui + 9)) : 0;
  const header = { x: rect.x, y: rect.y, w: rect.w, h: hh };
  const pad = 8;
  if (hh > 0) {
    fill(ctx, rect.x, rect.y, rect.w, hh, COL.plate);
    fill(ctx, rect.x, rect.y + hh - 1, rect.w, 1, COL.ruleFaint);
  }
  const avail = rect.w - 2 * pad;
  const t = hh > 0 ? fitText(ctx, 'ui', SIZE.ui, p?.title ?? title, avail) : '';
  const tw = textWidth(ctx, 'ui', SIZE.ui, t);
  if (t) text(ctx, 'ui', SIZE.ui, rect.x + pad, rect.y + (hh - SIZE.ui) / 2, COL.text, t);
  if (hh > 0 && p && !p.state && p.info) {
    const left = rect.x + pad + tw + 14, right = rect.x + rect.w - pad;
    const info = fitText(ctx, 'mono', SIZE.mono, p.info, right - left);
    if (info) text(ctx, 'mono', SIZE.mono, right - textWidth(ctx, 'mono', SIZE.mono, info), rect.y + (hh - SIZE.mono) / 2, COL.faint, info);
  }
  const py = 6;
  let content: Rect = { x: rect.x + pad, y: rect.y + hh + py, w: rect.w - 2 * pad, h: rect.h - hh - 2 * py };
  // A table runs to the panel's bottom edge, under the title's margin.
  if (p?.primitive === 'table') content = { x: rect.x + pad, y: rect.y + hh + 1, w: rect.w - 2 * pad, h: rect.h - hh - 1 };
  const frame = { header, content };
  if (content.w < 8 || content.h < 8) return frame;
  if (!p) return frame;   // not resolved yet: the panel waits, empty
  if (p.state) {
    drawState(ctx, p.state, content);
    return frame;
  }
  ctx.save();
  ctx.beginPath();
  ctx.rect(content.x, content.y, content.w, content.h);
  ctx.clip();
  switch (p.primitive) {
    case 'series': drawSeries(ctx, p, content, key); break;
    case 'histogram': drawHistogram(ctx, p, content); break;
    case 'heatmap': drawGrid(ctx, p, content, false); break;
    case 'image': drawGrid(ctx, p, content, true); break;
    case 'points': drawPoints(ctx, p, content); break;
    case 'table': drawTable(ctx, p, content, key, view); break;
    case 'text': drawText(ctx, p, content); break;
  }
  ctx.restore();
  return frame;
}

// ---- series ----------------------------------------------------------------------

function drawSeries(ctx: CanvasRenderingContext2D, p: PreparedPanel, c: Rect, key: ColorKey): void {
  const S = p.series!;
  const xs: number[] = [], ys: number[] = [];
  for (const l of S.lines) {
    for (const v of l.x) xs.push(v);
    for (const v of l.y) ys.push(v);
  }
  const xExt = fitLimits(xs, p.xLog, 0) ?? [0, 1];
  const yExt = fitLimits(ys, p.yLog, 0.05) ?? [0, 1];
  // A fixed range on a log axis whose end is not above zero takes that end from the data.
  const xr = p.xRange ? [...p.xRange] : null, yr = p.yRange ? [...p.yRange] : null;
  if (p.xLog && xr && xr[0] <= 0) xr[0] = xExt[0] > 0 ? xExt[0] : xr[1] * 1e-3;
  if (p.yLog && yr && yr[0] <= 0) yr[0] = yExt[0] > 0 ? yExt[0] : yr[1] * 1e-3;
  const xFixed = !!xr && xr[1] > xr[0] && (!p.xLog || xr[0] > 0);
  const yFixed = !!yr && yr[1] > yr[0] && (!p.yLog || yr[0] > 0);
  const xLim = xFixed ? xr! : xExt, yLim = yFixed ? yr! : yExt;
  const plot = new Plot2D(ctx, c,
    { lo: xLim[0], hi: xLim[1], log: p.xLog, label: p.xLabel, ticks: p.xLog ? logTicks(xLim[0], xLim[1], fmt4) : undefined },
    { lo: yLim[0], hi: yLim[1], log: p.yLog, label: p.yLabel, ticks: p.yLog ? logTicks(yLim[0], yLim[1], fmt4) : undefined });
  plot.frame();
  plot.clip();
  ctx.lineWidth = 1.25;
  ctx.lineJoin = 'round';
  S.lines.forEach((l, i) => {
    if (!l.y.length) return;
    ctx.strokeStyle = key.color(l.name) ?? '';
    void i;
    ctx.beginPath();
    for (let k = 0; k < l.y.length; k++) {
      const X = plot.px(l.x[k]), Y = plot.py(l.y[k]);
      if (k) ctx.lineTo(X, Y);
      else ctx.moveTo(X, Y);
    }
    ctx.stroke();
  });
  ctx.restore();
  plot.finish(S.lines.length > 1 ? S.lines.map((l) => ({ name: l.name, color: key.color(l.name) })) : []);
}

// ---- histogram -------------------------------------------------------------------

function drawHistogram(ctx: CanvasRenderingContext2D, p: PreparedPanel, c: Rect): void {
  const H = p.histogram!;
  const B = H.counts.length, w = B ? (H.hi - H.lo) / B : 1;
  let top = p.yLog ? 1 : 0;
  for (const y of H.counts) top = Math.max(top, y);
  let ylo = 0, yhi = top > 0 ? top * 1.05 : 1;
  let ticks: Tick[] | undefined;
  if (p.yLog) {
    // Counts on a log axis start under 1, so that a bin of one value shows as a bar; the ticks start at 1.
    ylo = 0.5;
    yhi = 10 ** (Math.log10(top) + Math.max(0.05, (Math.log10(top) - Math.log10(ylo)) * 0.05));
    ticks = logTicks(1, yhi, fmt4);
  }
  const plot = new Plot2D(ctx, c, { lo: H.lo, hi: H.hi, log: false, label: p.xLabel },
    { lo: ylo, hi: yhi, log: p.yLog, label: p.yLabel, ticks });
  plot.frame();
  plot.clip();
  ctx.fillStyle = COL.text;
  const r = plot.rect;
  for (let i = 0; i < B; i++) {
    const y = H.counts[i];
    if (!(y > 0)) continue;
    const cx = H.lo + (i + 0.5) * w, half = (w * 0.92) / 2;
    const x0 = plot.px(cx - half), x1 = plot.px(cx + half);
    const y0 = p.yLog ? r.y + r.h : plot.py(0), y1 = plot.py(y);
    ctx.fillRect(x0, Math.min(y0, y1), x1 - x0, Math.abs(y0 - y1));
  }
  ctx.restore();
  plot.finish();
}

// ---- heatmap and image -----------------------------------------------------------

/** Evenly spaced round integer ticks in [lo, hi], about n of them (index_ticks). */
function indexTicks(lo: number, hi: number, n: number): number[] {
  const out: number[] = [];
  if (!(hi >= lo) || !Number.isFinite(lo) || !Number.isFinite(hi)) return out;
  const raw = Math.max(hi - lo, 1e-9) / Math.max(1, n);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const r = raw / mag;
  const step = Math.max(1, Math.round((r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag));
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9 && out.length < 64; t += step) out.push(Math.round(t));
  return out;
}

/** About n round ticks in [lo, hi] (nice_ticks). */
function niceTicks(lo: number, hi: number, n: number): number[] {
  const out: number[] = [];
  if (!(hi > lo) || !Number.isFinite(lo) || !Number.isFinite(hi)) return out;
  const raw = (hi - lo) / Math.max(1, n);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const r = raw / mag;
  const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
  for (let t = Math.ceil(lo / step - 1e-9) * step; t <= hi + step * 1e-9 && out.length < 32; t += step)
    out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
  return out;
}

const gridCanvas = new WeakMap<object, HTMLCanvasElement>();

function gridImage(p: PreparedPanel): HTMLCanvasElement {
  const G = p.grid!;
  let cv = gridCanvas.get(G);
  if (!cv) {
    cv = document.createElement('canvas');
    cv.width = G.cols;
    cv.height = G.rows;
    cv.getContext('2d')!.putImageData(new ImageData(G.rgba as Uint8ClampedArray<ArrayBuffer>, G.cols, G.rows), 0, 0);
    gridCanvas.set(G, cv);
  }
  return cv;
}

function drawGrid(ctx: CanvasRenderingContext2D, p: PreparedPanel, c: Rect, image: boolean): void {
  const G = p.grid!;
  let barTicks: number[] = [], barLabels: string[] = [], barLabelW = 0;
  if (!image) {
    const estH = c.h - 48 * K;
    barTicks = niceTicks(G.vmin, G.vmax, Math.max(2, Math.trunc(estH / (48 * K))));
    barLabels = barTicks.map((t) => fmt4(t));
    for (const l of barLabels) barLabelW = Math.max(barLabelW, textWidth(ctx, 'mono', AXES, l));
  }
  const ml = 46 * K, mt = 16 * K, mb = 32 * K;
  const mr = image ? 10 * K : 26 * K + (barLabels.length ? 6 * K + barLabelW : 0);
  let ax = c.x + ml, ay = c.y + mt, aw = c.w - ml - mr, ah = c.h - mt - mb;
  if (aw < 4 || ah < 4 || G.rows <= 0 || G.cols <= 0) return;
  const W = G.sourceCols, Hs = G.sourceRows;
  if (image) {
    // Equal units, centred in the area.
    const kf = Math.min(aw / W, ah / Hs);
    const iw = W * kf, ih = Hs * kf;
    ax += Math.floor((aw - iw) * 0.5);
    ay += Math.floor((ah - ih) * 0.5);
    aw = iw;
    ah = ih;
  }
  ax = Math.round(ax);
  ay = Math.round(ay);
  aw = Math.round(aw);
  ah = Math.round(ah);
  const magnify = Math.min(aw / G.cols, ah / G.rows) * (window.devicePixelRatio || 1) >= 1;
  ctx.imageSmoothingEnabled = !magnify;
  ctx.drawImage(gridImage(p), ax, ay, aw, ah);
  ctx.imageSmoothingEnabled = true;
  // The frame on the pixels around the area; ticks at cell centres, labelled with source indices.
  outline(ctx, ax - 1, ay - 1, ax + aw + 1, ay + ah + 1, COL.rule);
  const sx = aw / W, sy = ah / Hs;
  for (const col of indexTicks(0, W - 1, Math.max(2, Math.trunc(aw / (70 * K))))) {
    const x = Math.round(ax + (col + 0.5) * sx);
    fill(ctx, x, ay + ah, 1, 4, COL.rule);
    const t = String(col);
    text(ctx, 'mono', AXES, x + 0.5 - textWidth(ctx, 'mono', AXES, t) / 2, ay + ah + 6 * K, COL.faint, t);
  }
  for (const row of indexTicks(0, Hs - 1, Math.max(2, Math.trunc(ah / (36 * K))))) {
    const y = Math.round(ay + (row + 0.5) * sy);
    fill(ctx, ax - 4, y, 4, 1, COL.rule);
    const t = String(row);
    text(ctx, 'mono', AXES, ax - 6 * K - textWidth(ctx, 'mono', AXES, t), y + 0.5 - AXES / 2, COL.faint, t);
  }
  if (p.xLabel) text(ctx, 'mono', AXES, ax + aw - textWidth(ctx, 'mono', AXES, p.xLabel), ay + ah + 19 * K, COL.faint, p.xLabel);
  if (p.yLabel)
    text(ctx, 'mono', AXES, Math.max(c.x, ax - 6 * K - textWidth(ctx, 'mono', AXES, p.yLabel)), ay - 4 * K - AXES, COL.faint, p.yLabel);
  if (!image) {
    // The colorbar: the colormap from its top (index 255) to its bottom.
    const table = lut(G.colormap);
    const bx = Math.round(ax + aw + 8 * K), bw = Math.round(8 * K);
    const n = Math.max(2, Math.floor(ah));
    for (let i = 0; i < n; i++) {
      const idx = Math.round(255 * (1 - i / (n - 1))) * 4;
      fill(ctx, bx, ay + (i * ah) / n, bw, ah / n + 0.5, `rgb(${table[idx]},${table[idx + 1]},${table[idx + 2]})`);
    }
    outline(ctx, bx - 1, ay - 1, bx + bw + 1, ay + ah + 1, COL.rule);
    const span = G.vmax - G.vmin;
    barTicks.forEach((t, i) => {
      if (!(span > 0)) return;
      const y = Math.round(ay + ah * (1 - (t - G.vmin) / span));
      fill(ctx, bx + bw, y, 3 * K, 1, COL.rule);
      const ty = Math.min(Math.max(y + 0.5 - AXES / 2, ay - AXES / 2), ay + ah - AXES / 2);
      text(ctx, 'mono', AXES, bx + bw + 6 * K, ty, COL.faint, barLabels[i]);
    });
  }
}

// ---- points ----------------------------------------------------------------------

function drawPoints(ctx: CanvasRenderingContext2D, p: PreparedPanel, c: Rect): void {
  const P = p.points!;
  const keyW = P.rgba && !P.classes.length ? Math.round(Math.max(60, textWidth(ctx, 'mono', AXES, '-0.000e-0') + 26)) : 0;
  const area = { x: c.x, y: c.y, w: c.w - keyW, h: c.h };
  const base = pointSize(P.x.length);
  if (P.threeD) drawPoints3d(ctx, p, area, base);
  else {
    const xl = fitLimits(P.x, false, 0.04), yl = fitLimits(P.y, false, 0.04);
    const plot = new Plot2D(ctx, area, { lo: xl?.[0] ?? 0, hi: xl?.[1] ?? 1, log: false, label: p.xLabel },
      { lo: yl?.[0] ?? 0, hi: yl?.[1] ?? 1, log: false, label: p.yLabel });
    plot.frame();
    plot.clip();
    for (let i = 0; i < P.x.length; i++) {
      const h = 0.5 * base * (P.scale ? P.scale[i] : 1);
      ctx.fillStyle = P.rgba ? `rgb(${P.rgba[i * 4]},${P.rgba[i * 4 + 1]},${P.rgba[i * 4 + 2]})` : COL.flatPoint;
      ctx.fillRect(plot.px(P.x[i]) - h, plot.py(P.y[i]) - h, 2 * h, 2 * h);
    }
    ctx.restore();
    plot.finish();
  }
  if (keyW > 0) drawColorScale(ctx, P.colorName, P.cmin, P.cmax, P.colormap, { x: area.x + area.w + 4, y: c.y, w: keyW - 4, h: c.h });
  if (P.classes.length) drawClassLegend(ctx, P.colorName, P.classes, c);
}

/** ImPlot3D's initial rotation of a new plot, as a quaternion (x, y, z, w). */
const ROT = [-0.513269, -0.212596, -0.318184, 0.76819];

function rotate(p: [number, number, number]): [number, number, number] {
  const [qx, qy, qz, w] = ROT;
  const uv = [qy * p[2] - qz * p[1], qz * p[0] - qx * p[2], qx * p[1] - qy * p[0]];
  const uuv = [qy * uv[2] - qz * uv[1], qz * uv[0] - qx * uv[2], qx * uv[1] - qy * uv[0]];
  return [p[0] + 2 * (w * uv[0] + uuv[0]), p[1] + 2 * (w * uv[1] + uuv[1]), p[2] + 2 * (w * uv[2] + uuv[2])];
}

/** A 3-D cloud in its box as ImPlot3D turns a new plot, the box filling 0.84 of the panel; the far walls on the plot's ground. */
function drawPoints3d(ctx: CanvasRenderingContext2D, p: PreparedPanel, c: Rect, base: number): void {
  const P = p.points!;
  const lim = [fitLimits(P.x, false, 0.02), fitLimits(P.y, false, 0.02), fitLimits(P.z!, false, 0.02)];
  if (lim.some((l) => !l)) return;
  const corner = (i: number): [number, number, number] => [i & 1 ? 0.5 : -0.5, i & 2 ? 0.5 : -0.5, i & 4 ? 0.5 : -0.5];
  let mx = 0, my = 0;
  for (let i = 0; i < 8; i++) {
    const q = rotate(corner(i));
    mx = Math.max(mx, Math.abs(q[0]));
    my = Math.max(my, Math.abs(q[1]));
  }
  const s = Math.min((0.84 * c.w) / (2 * mx), (0.84 * c.h) / (2 * my));
  const cx = c.x + c.w / 2, cy = c.y + c.h / 2;
  const at = (v: [number, number, number]): [number, number] => {
    const q = rotate(v);
    return [cx + q[0] * s, cy - q[1] * s];
  };
  // The far wall of each axis: the face whose centre turns away from the viewer.
  const faces: number[][] = [];
  for (let axis = 0; axis < 3; axis++) {
    const centre: [number, number, number] = [0, 0, 0];
    centre[axis] = -0.5;
    const far = rotate(centre)[2] < 0 ? 0 : 1;
    faces.push([0, 1, 2, 3, 4, 5, 6, 7].filter((i) => ((i >> axis) & 1) === far));
  }
  for (const f of faces) {
    const [a, b, d, e] = f;   // corners of a face in bit order: a, b, d, e with b^d diagonal
    const pts = [at(corner(a)), at(corner(b)), at(corner(e)), at(corner(d))];
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.fillStyle = COL.deep;
    ctx.fill();
    ctx.strokeStyle = COL.rule;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  const norm = (i: number, a: number) => (a - lim[i]![0]) / (lim[i]![1] - lim[i]![0]) - 0.5;
  for (let i = 0; i < P.x.length; i++) {
    const [X, Y] = at([norm(0, P.x[i]), norm(1, P.y[i]), norm(2, P.z![i])]);
    // ImPlot3D's square marker of size 0.5 * base reaches 0.5 * base / sqrt(2) each way.
    const h = Math.SQRT1_2 * 0.5 * base * (P.scale ? P.scale[i] : 1);
    ctx.fillStyle = P.rgba ? `rgb(${P.rgba[i * 4]},${P.rgba[i * 4 + 1]},${P.rgba[i * 4 + 2]})` : COL.flatPoint;
    ctx.fillRect(X - h, Y - h, 2 * h, 2 * h);
  }
}

/** The colormap's scale of a points panel, named by its colour binding (ImPlot's ColormapScale). */
function drawColorScale(ctx: CanvasRenderingContext2D, name: string, lo: number, hi: number, cmap: string, r: Rect): void {
  const table = lut(cmap);
  const top = r.y + 6, bottom = r.y + r.h - 6;
  const bx = Math.round(r.x + (name ? AXES + 5 : 0)), bw = 12;
  const n = Math.max(2, Math.floor(bottom - top));
  for (let i = 0; i < n; i++) {
    const idx = Math.round(255 * (1 - i / (n - 1))) * 4;
    fill(ctx, bx, top + ((bottom - top) * i) / n, bw, (bottom - top) / n + 0.5, `rgb(${table[idx]},${table[idx + 1]},${table[idx + 2]})`);
  }
  outline(ctx, bx, top, bx + bw, bottom, COL.rule);
  for (const t of niceTicks(lo, hi, Math.max(2, Math.trunc((bottom - top) / 40)))) {
    const y = Math.round(bottom - ((t - lo) / (hi - lo)) * (bottom - top));
    fill(ctx, bx + bw, y, 4, 1, COL.rule);
    text(ctx, 'mono', AXES, bx + bw + 6, y - AXES / 2, COL.faint, fmt4(t));
  }
  if (name) {
    ctx.save();
    ctx.translate(Math.round(r.x), Math.round(r.y + r.h / 2 + textWidth(ctx, 'mono', AXES, name) / 2));
    ctx.rotate(-Math.PI / 2);
    text(ctx, 'mono', AXES, 0, 0, COL.faint, name);
    ctx.restore();
  }
}

/** The legend of the classes over the top right corner (draw_class_legend). */
function drawClassLegend(ctx: CanvasRenderingContext2D, name: string, classes: number[], r: Rect): void {
  const sw = 7, gap = 5, pad = 6, line = Math.round(SIZE.mono + 5);
  let w = textWidth(ctx, 'mono', AXES, name);
  for (const c of classes) w = Math.max(w, sw + gap + textWidth(ctx, 'mono', SIZE.mono, String(c)));
  const rows = Math.min(classes.length, Math.max(0, Math.trunc((r.h - 2 * pad) / line - 1)));
  const x1 = r.x + r.w - 8, y0 = r.y + 8, x0 = x1 - Math.round(w + 2 * pad);
  const y1 = y0 + Math.round(2 * pad + line * (rows + 1));
  ctx.globalAlpha = 230 / 255;
  fill(ctx, x0, y0, x1 - x0, y1 - y0, COL.ground);
  ctx.globalAlpha = 1;
  outline(ctx, x0, y0, x1, y1, COL.rule);
  let y = y0 + pad;
  text(ctx, 'mono', AXES, x0 + pad, y, COL.faint, name);
  y += line;
  for (let i = 0; i < rows; i++) {
    const cy = Math.round(y + SIZE.mono / 2);
    fill(ctx, x0 + pad, cy - sw / 2, sw, sw, classColor(classes[i]));
    text(ctx, 'mono', SIZE.mono, x0 + pad + sw + gap, y, COL.secondary, String(classes[i]));
    y += line;
  }
}

// ---- table -----------------------------------------------------------------------

const CELL_PAD_X = 6, CELL_PAD_Y = 3, TRACKING = 0.08, SCROLLBAR = 6;

function drawTable(ctx: CanvasRenderingContext2D, p: PreparedPanel, c: Rect, key: ColorKey, view: PanelView): void {
  const T = p.table!;
  const ncols = Math.min(T.columns.length, 64);
  if (ncols <= 0) return;
  const heads = T.columns.slice(0, ncols).map((n) => n.toUpperCase());
  // Each column as wide as its head and its widest cell among the first rows,
  // the columns together at the left.
  const widths: number[] = [];
  let total = 0;
  for (let i = 0; i < ncols; i++) {
    let w = textWidth(ctx, 'ui', SIZE.small, T.columns[i]) * 1.25 + 10;
    for (let r = 0; r < Math.min(T.cells.length, 200); r++) w = Math.max(w, textWidth(ctx, 'mono', SIZE.mono, T.cells[r][i]));
    widths.push(Math.round(w + 18));
    total += widths[i] + 2 * CELL_PAD_X;
  }
  // Whole rows: the head and as many rows as fit.
  const fit = Math.max(0, Math.floor((c.h - ROW) / ROW));
  const scrolls = fit < T.cells.length;
  const height = Math.min(c.h, ROW * (1 + Math.min(fit, T.cells.length)) + 1);
  const width = Math.min(c.w, total + (scrolls ? SCROLLBAR : 0));
  const first = Math.max(0, Math.min(view.scroll, T.cells.length - fit));
  view.scroll = first;
  // Column edges: no outer padding on the left, the cell padding between columns.
  const x0 = c.x;
  const lefts: number[] = [];
  let at = x0;
  for (let i = 0; i < ncols; i++) {
    lefts.push(i === 0 ? at : at + CELL_PAD_X);
    at += (i === 0 ? 0 : CELL_PAD_X) + widths[i] + CELL_PAD_X;
  }
  const right = Math.min(x0 + width - (scrolls ? SCROLLBAR : 0), at);
  // The head: tracked capitals in the small UI face, a faint top rule and a midrule.
  const y0 = c.y;
  const tracking = TRACKING * SIZE.small;
  for (let i = 0; i < ncols; i++) {
    const hw = trackedWidth(ctx, 'ui', SIZE.small, heads[i], tracking);
    const cellRight = lefts[i] + widths[i];
    const lx = T.numeric[i] ? Math.max(lefts[i], Math.floor(cellRight - hw)) : lefts[i];
    trackedText(ctx, 'ui', SIZE.small, lx, y0 + CELL_PAD_Y, COL.secondary, heads[i], tracking);
    // A column that a series also draws carries its colour key.
    const swatch = key.find(T.columns[i]);
    if (swatch) {
      const sw = 7, cy = Math.round(y0 + ROW / 2);
      const sx = T.numeric[i] ? cellRight - hw - sw - 6 : lefts[i] + hw + 6;
      fill(ctx, sx, cy - sw / 2, sw, sw, swatch);
    }
  }
  fill(ctx, x0, y0, right - x0, 1, COL.faint);
  fill(ctx, x0, y0 + ROW - 1, right - x0, 1, COL.rule);
  // The rows, in mono, figures right-aligned.
  const shown = Math.min(fit, T.cells.length - first);
  for (let r = 0; r < shown; r++) {
    const row = T.cells[first + r];
    const top = y0 + ROW * (r + 1) + CELL_PAD_Y;
    for (let i = 0; i < ncols; i++) {
      const s = row[i] ?? '';
      const x = T.numeric[i] ? lefts[i] + widths[i] - textWidth(ctx, 'mono', SIZE.mono, s) : lefts[i];
      text(ctx, 'mono', SIZE.mono, x, top, COL.text, s);
    }
  }
  // The bottom rule under the last row, or along the bottom of the area.
  fill(ctx, x0, Math.min(y0 + ROW * (shown + 1), y0 + height - 1), right - x0, 1, COL.faint);
  if (scrolls && T.cells.length) {
    const trackTop = y0 + ROW, trackH = height - ROW - 1;
    const thumbH = Math.max(8, (trackH * fit) / T.cells.length);
    const thumbY = trackTop + ((trackH - thumbH) * first) / Math.max(1, T.cells.length - fit);
    fill(ctx, x0 + width - SCROLLBAR + 1, thumbY, SCROLLBAR - 2, thumbH, COL.rule);
  }
}

// ---- text ------------------------------------------------------------------------

function drawText(ctx: CanvasRenderingContext2D, p: PreparedPanel, c: Rect): void {
  const X = p.text!;
  let y = c.y;
  if (X.text) {
    const size = Math.round(SIZE.ui * (X.hasValue ? 1 : 1.25));
    const wrap = Math.min(c.w, textWidth(ctx, 'ui', size, '0') * 80);
    for (const line of wrapLines(ctx, size, X.text, wrap)) {
      text(ctx, 'ui', size, c.x, y, X.hasValue ? COL.secondary : COL.text, line);
      y += size;
    }
    y += 8;
  }
  if (X.hasValue) {
    const big = Math.round(SIZE.mono * 2.6);
    text(ctx, 'mono', big, c.x, y, COL.text, fitText(ctx, 'mono', big, X.value, c.w));
  }
}

function wrapLines(ctx: CanvasRenderingContext2D, size: number, s: string, w: number): string[] {
  const out: string[] = [];
  for (const para of s.split('\n')) {
    let line = '';
    for (const word of para.split(' ')) {
      const next = line ? `${line} ${word}` : word;
      if (line && textWidth(ctx, 'ui', size, next) > w) {
        out.push(line);
        line = word;
      } else line = next;
    }
    out.push(line);
  }
  return out;
}

// ---- tabs ------------------------------------------------------------------------

/** A tabs node's strip of labels (draw_tabs); returns each label's hit rectangle. */
export function drawTabs(ctx: CanvasRenderingContext2D, t: LayoutTabs, info: string): Rect[] {
  const a = t.bar;
  const hits: Rect[] = [];
  fill(ctx, a.x, a.y, a.w, a.h, COL.ground);
  fill(ctx, a.x, a.y + a.h - 1, a.w, 1, COL.rule);
  const pad = 12;
  let x = a.x;
  for (let i = 0; i < t.labels.length; i++) {
    const label = fitText(ctx, 'ui', SIZE.ui, t.labels[i], Math.max(0, a.x + a.w - x - 2 * pad));
    if (!label) break;
    const w = textWidth(ctx, 'ui', SIZE.ui, label) + 2 * pad;
    const active = i === t.active;
    if (active) {
      fill(ctx, x, a.y, w, a.h, COL.plate);
      fill(ctx, x, a.y, w, 1, COL.accent);
    }
    fill(ctx, x + w, a.y, 1, a.h, COL.rule);
    text(ctx, 'ui', SIZE.ui, x + pad, a.y + (a.h - SIZE.ui) / 2, active ? COL.text : COL.secondary, label);
    hits.push({ x, y: a.y, w, h: a.h });
    x += w + 1;
  }
  if (info) {
    const right = a.x + a.w - 8, left = x + 14;
    const s = fitText(ctx, 'mono', SIZE.mono, info, right - left);
    if (s) text(ctx, 'mono', SIZE.mono, right - textWidth(ctx, 'mono', SIZE.mono, s), a.y + (a.h - SIZE.mono) / 2, COL.faint, s);
  }
  return hits;
}
