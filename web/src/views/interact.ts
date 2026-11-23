// Pointer gestures shared by every live view: hover, drag (pan or orbit),
// shift-drag (selection), wheel (zoom), click (pick) and double-click (reset).
// Positions are CSS pixels relative to the element.
//
// A press follows the rule of the app's camera press (src/host/live/camera_drag.h):
// the drag moves the view from the first pointer move, with no dead zone, and a
// release within CLICK_TOLERANCE of the press point, all through the press, is a
// click at the press point, which first puts back what the press's motion changed
// (Gestures.press). A selection (shift-drag) starts past CLICK_SLOP, as the
// selections of the app's live views do.

export interface Mods { shift: boolean; alt: boolean; ctrl: boolean; button: number }

export interface Gestures {
  hover?(x: number, y: number): void;
  leave?(): void;
  drag?(dx: number, dy: number, x: number, y: number, mods: Mods): void;
  dragEnd?(): void;
  /** At a press that can drag: a function that puts back the view state that the drag of this press changes, which a press that ends as a click calls. */
  press?(): () => void;
  select?(phase: 'start' | 'move' | 'end', x: number, y: number, mods: Mods): void;
  wheel?(x: number, y: number, factor: number, mods: Mods): void;
  click?(x: number, y: number, mods: Mods): void;
  dblclick?(x: number, y: number): void;
}

const CLICK_TOLERANCE = 2;   // the app's kCameraClickTolerance
const CLICK_SLOP = 3;        // the app's kClickSlop of a selection

export function attachGestures(el: HTMLElement, g: Gestures): () => void {
  let down: {
    x: number; y: number; lx: number; ly: number; mods: Mods; moved: boolean; dragged: boolean; selecting: boolean; id: number;
    restore: (() => void) | null;
  } | null = null;
  const pos = (e: MouseEvent) => {
    const r = el.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };
  const mods = (e: MouseEvent): Mods => ({ shift: e.shiftKey, alt: e.altKey, ctrl: e.ctrlKey || e.metaKey, button: e.button });

  const onDown = (e: PointerEvent) => {
    if (e.button > 2) return;
    const p = pos(e);
    const m = mods(e);
    const selecting = m.shift && e.button === 0;
    down = { x: p.x, y: p.y, lx: p.x, ly: p.y, mods: m, moved: false, dragged: false, selecting, id: e.pointerId, restore: null };
    if (!selecting) down.restore = g.press?.() ?? null;
    try {
      el.setPointerCapture(e.pointerId);
    } catch {
      /* synthetic pointers cannot always be captured */
    }
    if (down.selecting) g.select?.('start', p.x, p.y, m);
    e.preventDefault();
  };
  const onMove = (e: PointerEvent) => {
    const p = pos(e);
    if (!down) {
      g.hover?.(p.x, p.y);
      return;
    }
    motion(down, p);
  };
  // The pointer at p during a press: a selection moves on, and a drag moves the view by the
  // motion since the last point from the first move; `moved` marks a press that left the click
  // tolerance (the slop of a selection) at any point.
  const motion = (d: NonNullable<typeof down>, p: { x: number; y: number }) => {
    const away = Math.hypot(p.x - d.x, p.y - d.y);
    if (away > (d.selecting ? CLICK_SLOP : CLICK_TOLERANCE)) d.moved = true;
    if (d.selecting) g.select?.('move', p.x, p.y, d.mods);
    else if (p.x !== d.lx || p.y !== d.ly) {
      d.dragged = true;
      g.drag?.(p.x - d.lx, p.y - d.ly, p.x, p.y, d.mods);
    }
    d.lx = p.x;
    d.ly = p.y;
  };
  const onUp = (e: PointerEvent) => {
    if (!down) return;
    const p = pos(e);
    const d = down;
    down = null;
    try {
      el.releasePointerCapture(e.pointerId);
    } catch {
      /* not captured */
    }
    if (d.selecting) {
      if (Math.hypot(p.x - d.x, p.y - d.y) > CLICK_SLOP) d.moved = true;
      if (d.moved) g.select?.('end', p.x, p.y, d.mods);
      else {
        g.select?.('end', NaN, NaN, d.mods);
        g.click?.(p.x, p.y, d.mods);
      }
    } else {
      if (e.type === 'pointerup') motion(d, p);   // the release's own point, as the app's release frame
      if (d.moved) g.dragEnd?.();
      else {
        if (d.dragged) d.restore?.();
        g.click?.(d.x, d.y, d.mods);
      }
    }
    g.hover?.(p.x, p.y);
  };
  const onLeave = () => {
    if (!down) g.leave?.();
  };
  const onWheel = (e: WheelEvent) => {
    if (!g.wheel) return;
    e.preventDefault();
    const p = pos(e);
    const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
    g.wheel(p.x, p.y, Math.exp(-dy * 0.0015), mods(e));
  };
  const onDbl = (e: MouseEvent) => {
    const p = pos(e);
    g.dblclick?.(p.x, p.y);
  };
  const onContext = (e: Event) => e.preventDefault();
  el.addEventListener('pointerdown', onDown);
  el.addEventListener('pointermove', onMove);
  el.addEventListener('pointerup', onUp);
  el.addEventListener('pointercancel', onUp);
  el.addEventListener('pointerleave', onLeave);
  el.addEventListener('wheel', onWheel, { passive: false });
  el.addEventListener('dblclick', onDbl);
  el.addEventListener('contextmenu', onContext);
  return () => {
    el.removeEventListener('pointerdown', onDown);
    el.removeEventListener('pointermove', onMove);
    el.removeEventListener('pointerup', onUp);
    el.removeEventListener('pointercancel', onUp);
    el.removeEventListener('pointerleave', onLeave);
    el.removeEventListener('wheel', onWheel);
    el.removeEventListener('dblclick', onDbl);
    el.removeEventListener('contextmenu', onContext);
  };
}
