// The controls of a spec in the browser's spec page, as the app's spec
// runtime draws them (src/host/spec_runtime/spec_controls.cpp): one row, its
// title on the left in the secondary colour and its widget on the right, one
// frame high in the middle of the row, on the panel ground.
//
// The page reads each control's state from panel.data and never changes it:
// every control acts on the run (intervene:, param:, action:, run:pause) or
// on the app's values (value, state:, the cursor), which the viewer origin
// may not change (design/command-layer.md, Origins). So each control shows
// its state with the look of a disabled widget, and a hover shows why; a
// readout only reads, and shows its text as the app does.

import type { Rect } from './layout.js';
import { COL, SIZE, fill, fitText, text, textWidth } from './look.js';

/** panel.data's `data` for a control (panel_data.h, control_data_result). */
export interface ControlData {
  id: string;
  kind: string;
  title: string;
  bind: string;
  value: unknown;
  text: string;
  enabled: boolean;
  why: string;
  min?: number;
  max?: number;
  choices?: { value: unknown; label: string | null }[];
}

/** The app's frame: the UI face at 15 px with a frame padding of 4 (sternbild_theme.cpp, kFramePadding). */
const FRAME_H = SIZE.ui + 2 * 4;
/** ImGui's style.DisabledAlpha in the house style. */
const DISABLED_ALPHA = 0.56;

/** The height of a control's row (spec_control_height): two lines of the UI face. */
export const controlHeight = (): number => Math.round(SIZE.ui * 2);

interface Row { title: Rect; widget: Rect }

/** row_of: the title's part, then the widget's, one frame high, centred in the row. */
function rowOf(ctx: CanvasRenderingContext2D, rect: Rect, title: string, withTitle: boolean): Row {
  const pad = 6;
  const y = Math.round(rect.y + Math.max(0, (rect.h - FRAME_H) * 0.5));
  let x = rect.x + pad;
  let t: Rect = { x, y, w: 0, h: FRAME_H };
  if (withTitle) {
    const want = textWidth(ctx, 'ui', SIZE.ui, title) + 2 * pad;
    const w = Math.min(want, Math.max(0, rect.w * 0.35));
    t = { x, y, w, h: FRAME_H };
    x += w;
  }
  return { title: t, widget: { x, y, w: Math.max(1, rect.x + rect.w - pad - x), h: FRAME_H } };
}

/** A rounded rectangle path. */
function rounded(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
}

/** The switch of draw_toggle: a track with a knob at its left (off) or right (on), then the text. */
function drawToggle(ctx: CanvasRenderingContext2D, row: Row, on: boolean, label: string, enabled: boolean): void {
  const h = Math.round(FRAME_H * 0.7), w = Math.round(h * 1.8);
  const ax = row.widget.x, ay = Math.round(row.widget.y + row.widget.h / 2 - h * 0.5);
  ctx.fillStyle = !enabled ? COL.plate : on ? COL.accent : COL.plate;
  rounded(ctx, ax, ay, w, h, h * 0.5);
  ctx.fill();
  ctx.strokeStyle = COL.ruleFaint;
  ctx.lineWidth = 1;
  rounded(ctx, ax + 0.5, ay + 0.5, w - 1, h - 1, h * 0.5);
  ctx.stroke();
  ctx.fillStyle = enabled ? COL.text : COL.faint;
  ctx.beginPath();
  ctx.arc(on ? ax + w - h * 0.5 : ax + h * 0.5, ay + h * 0.5, h * 0.5 - 2, 0, Math.PI * 2);
  ctx.fill();
  const tx = ax + w + 6;
  const shown = fitText(ctx, 'ui', SIZE.ui, label, Math.max(0, row.widget.x + row.widget.w - tx));
  text(ctx, 'ui', SIZE.ui, tx, row.widget.y + row.widget.h / 2 - SIZE.ui / 2, enabled ? COL.secondary : COL.faint, shown);
}

/** An ImGui frame: the plate with the house's 1 px border. */
function frame(ctx: CanvasRenderingContext2D, r: Rect, color: string = COL.plate): void {
  fill(ctx, r.x, r.y, r.w, r.h, color);
  ctx.strokeStyle = COL.ruleFaint;
  ctx.lineWidth = 1;
  ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
}

/** Text centred in a rectangle, cut to fit. */
function centred(ctx: CanvasRenderingContext2D, r: Rect, s: string, color: string): void {
  const shown = fitText(ctx, 'ui', SIZE.ui, s, r.w - 8);
  const w = textWidth(ctx, 'ui', SIZE.ui, shown);
  text(ctx, 'ui', SIZE.ui, r.x + (r.w - w) / 2, r.y + (r.h - SIZE.ui) / 2, color, shown);
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const label = (v: unknown): string =>
  v === null || v === undefined ? '' : typeof v === 'number' ? String(Math.round(v * 1e6) / 1e6) : typeof v === 'string' ? v : JSON.stringify(v);

/** One control in its row, from its panel.data, or its title alone before the data comes. */
export function drawControl(ctx: CanvasRenderingContext2D, d: ControlData | null, title: string, kind: string, rect: Rect): void {
  fill(ctx, rect.x, rect.y, rect.w, rect.h, COL.ground);
  const button = kind === 'button';
  const row = rowOf(ctx, rect, title, !button);
  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.x, rect.y, rect.w, rect.h);
  ctx.clip();
  if (!button && row.title.w > 0) {
    const shown = fitText(ctx, 'ui', SIZE.ui, title, Math.max(0, row.title.w - 6));
    text(ctx, 'ui', SIZE.ui, row.title.x, row.title.y + (row.title.h - SIZE.ui) / 2, COL.secondary, shown);
  }
  if (d) {
    const enabled = d.enabled;
    const w = row.widget;
    if (kind === 'toggle') {
      drawToggle(ctx, row, d.value === true, d.text || (d.value === true ? 'on' : 'off'), enabled);
    } else if (kind === 'readout') {
      const shown = fitText(ctx, 'ui', SIZE.ui, d.text || '—', w.w);
      text(ctx, 'ui', SIZE.ui, w.x, w.y + (w.h - SIZE.ui) / 2, d.text ? COL.text : COL.faint, shown);
    } else {
      // The other kinds as their ImGui frames, at the disabled alpha.
      ctx.globalAlpha = enabled ? 1 : DISABLED_ALPHA;
      if (kind === 'button') {
        frame(ctx, w);
        centred(ctx, w, d.text || title, COL.text);
      } else if (kind === 'slider' || kind === 'range') {
        frame(ctx, w);
        const lo = num(d.min) ?? 0, hi = num(d.max) ?? 1;
        const at = (v: number) => w.x + 2 + Math.max(0, Math.min(1, hi > lo ? (v - lo) / (hi - lo) : 0)) * (w.w - 4 - 8);
        const vals = kind === 'range' && Array.isArray(d.value) ? (d.value as unknown[]).map(num) : [num(d.value)];
        for (const v of vals) if (v !== null) fill(ctx, at(v), w.y + 2, 8, w.h - 4, COL.accent);
        centred(ctx, w, vals.map((v) => (v === null ? '' : label(v))).join(' – '), COL.text);
      } else if (kind === 'segmented') {
        const choices = d.choices ?? [];
        const on = new Set((Array.isArray(d.value) ? d.value : [d.value]).map((v) => JSON.stringify(v)));
        let x = w.x;
        for (const ch of choices) {
          const b = { x, y: w.y, w: w.h, h: w.h };
          frame(ctx, b, on.has(JSON.stringify(ch.value)) ? COL.accent : COL.plate);
          centred(ctx, b, (ch.label ?? label(ch.value)).slice(0, 2), COL.text);
          x += w.h + 4;
        }
      } else if (kind === 'picker' || kind === 'text') {
        frame(ctx, w);
        const ch = d.choices?.find((c) => JSON.stringify(c.value) === JSON.stringify(d.value));
        const shown = fitText(ctx, 'ui', SIZE.ui, ch?.label ?? label(d.value), w.w - 12);
        text(ctx, 'ui', SIZE.ui, w.x + 6, w.y + (w.h - SIZE.ui) / 2, COL.text, shown);
      } else if (kind === 'transport') {
        const v = d.value as { step?: number; live?: boolean } | null;
        const s = v && typeof v.step === 'number' ? `step ${v.step}${v.live ? '  live' : ''}` : '';
        text(ctx, 'ui', SIZE.ui, w.x, w.y + (w.h - SIZE.ui) / 2, COL.text, s);
      }
      ctx.globalAlpha = 1;
    }
  }
  ctx.restore();
}
