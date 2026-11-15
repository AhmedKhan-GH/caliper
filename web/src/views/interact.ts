// Pointer gestures shared by every live view: hover, drag (pan or orbit),
// shift-drag (selection), wheel (zoom), click (pick) and double-click (reset).
// Positions are CSS pixels relative to the element.

export interface Mods { shift: boolean; alt: boolean; ctrl: boolean; button: number }

export interface Gestures {
  hover?(x: number, y: number): void;
  leave?(): void;
  drag?(dx: number, dy: number, x: number, y: number, mods: Mods): void;
  dragEnd?(): void;
  select?(phase: 'start' | 'move' | 'end', x: number, y: number, mods: Mods): void;
  wheel?(x: number, y: number, factor: number, mods: Mods): void;
  click?(x: number, y: number, mods: Mods): void;
  dblclick?(x: number, y: number): void;
}

const CLICK_SLOP = 3;

export function attachGestures(el: HTMLElement, g: Gestures): () => void {
  let down: { x: number; y: number; lx: number; ly: number; mods: Mods; moved: boolean; selecting: boolean; id: number } | null = null;
  const pos = (e: MouseEvent) => {
    const r = el.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };
  const mods = (e: MouseEvent): Mods => ({ shift: e.shiftKey, alt: e.altKey, ctrl: e.ctrlKey || e.metaKey, button: e.button });

  const onDown = (e: PointerEvent) => {
    if (e.button > 2) return;
    const p = pos(e);
    const m = mods(e);
    down = { x: p.x, y: p.y, lx: p.x, ly: p.y, mods: m, moved: false, selecting: m.shift && e.button === 0, id: e.pointerId };
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
    if (!down.moved && Math.hypot(p.x - down.x, p.y - down.y) > CLICK_SLOP) down.moved = true;
    if (down.selecting) g.select?.('move', p.x, p.y, down.mods);
    else if (down.moved) g.drag?.(p.x - down.lx, p.y - down.ly, p.x, p.y, down.mods);
    down.lx = p.x;
    down.ly = p.y;
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
      if (d.moved) g.select?.('end', p.x, p.y, d.mods);
      else {
        g.select?.('end', NaN, NaN, d.mods);
        g.click?.(p.x, p.y, d.mods);
      }
    } else if (d.moved) g.dragEnd?.();
    else g.click?.(p.x, p.y, d.mods);
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
