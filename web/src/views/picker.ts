// The data view picker in a view's header, built from what the view offers:
// a source menu when there are several sources, and one control per dimension
// that chooses an index, "all" (the dimension is kept) or a reduction over it.
// A source with no more dimensions than the view's kind must keep has nothing
// to choose but "all", so it gets no dimension controls, and a view with one
// such source gets no picker.

import { KIND_IDS, REDUCTIONS, type DataView, type Reduction, type SourceInfo, type ViewEntry } from '../protocol/messages.js';

type Choice = { kind: 'all' } | { kind: 'index'; i: number } | { kind: 'reduce'; op: Reduction };

const MAX_OPTIONS = 256;

/** The dimensions a data view of each kind keeps at the least (wire protocol v3, "Resolving a data view"). */
const MIN_KEPT: Record<number, number> = { [KIND_IDS.heatmap]: 2, [KIND_IDS.image]: 2, [KIND_IDS.points]: 1 };

export class Picker {
  readonly el = document.createElement('span');
  private entry: ViewEntry | null = null;
  private source: SourceInfo | null = null;
  private choices = new Map<string, Choice>();

  constructor(private readonly onPick: (dv: DataView) => void) {
    this.el.className = 'picker';
  }

  /** Builds the controls for what a view offers (nothing for a view without dims). */
  offer(entry: ViewEntry | undefined): void {
    const same = entry && this.entry && JSON.stringify(entry.sources) === JSON.stringify(this.entry.sources);
    this.entry = entry ?? null;
    if (same) return;
    this.source = entry?.sources[0] ?? null;
    if (entry?.default) this.apply(entry.default, false);
    this.render();
  }

  /** Shows a resolved data view (from a view_reply or a snapshot). */
  show(dv: DataView): void {
    this.apply(dv, true);
  }

  private apply(dv: DataView, render: boolean): void {
    if (!this.entry) return;
    const src = this.entry.sources.find((s) => s.name === dv.source) ?? this.source ?? this.entry.sources[0];
    if (!src) return;
    const next = new Map<string, Choice>();
    for (const d of src.dims) {
      const ix = dv.index?.[d.name];
      const op = dv.reduce?.[d.name];
      if (typeof ix === 'number') next.set(d.name, { kind: 'index', i: ix });
      else if (op) next.set(d.name, { kind: 'reduce', op });
      else next.set(d.name, { kind: 'all' });
    }
    const changed = src !== this.source || [...next].some(([k, v]) => JSON.stringify(this.choices.get(k)) !== JSON.stringify(v));
    this.source = src;
    this.choices = next;
    if (render && changed) this.render();
  }

  /** The data view the controls describe. */
  value(): DataView {
    const dv: DataView = {};
    if (!this.source) return dv;
    dv.source = this.source.name;
    const index: Record<string, number> = {};
    const reduce: Record<string, Reduction> = {};
    const kept: string[] = [];
    for (const d of this.source.dims) {
      const c = this.choices.get(d.name) ?? { kind: 'all' };
      if (c.kind === 'index') index[d.name] = c.i;
      else if (c.kind === 'reduce') reduce[d.name] = c.op;
      else kept.push(d.name);
    }
    dv.index = index;
    dv.reduce = reduce;
    // A heatmap shows a third kept dimension as a mosaic of tiles.
    if (this.entry?.kind === KIND_IDS.heatmap && kept.length === 3) dv.tile = kept[0];
    return dv;
  }

  private render(): void {
    this.el.replaceChildren();
    const e = this.entry;
    const choosable = !!this.source && this.source.dims.length > (MIN_KEPT[e?.kind ?? 0] ?? 0);
    if (!e || !this.source || (e.sources.length < 2 && !choosable)) return;
    if (e.sources.length > 1) {
      const sel = document.createElement('select');
      sel.className = 'source';
      sel.setAttribute('aria-label', 'source');
      for (const s of e.sources) sel.append(new Option(s.name, s.name, false, s === this.source));
      sel.addEventListener('change', () => {
        const src = e.sources.find((s) => s.name === sel.value);
        if (!src) return;
        const old = this.choices;
        this.source = src;
        this.choices = new Map(src.dims.map((d) => [d.name, old.get(d.name) ?? { kind: 'all' }]));
        this.render();
        this.onPick(this.value());
      });
      this.el.append(sel);
    }
    for (const d of choosable ? this.source.dims : []) {
      const wrap = document.createElement('label');
      wrap.className = 'dim';
      const name = document.createElement('span');
      name.textContent = d.name;
      const sel = document.createElement('select');
      sel.dataset.dim = d.name;
      sel.setAttribute('aria-label', d.name);
      sel.append(new Option('all', 'all'));
      const red = document.createElement('optgroup');
      red.label = 'reduce';
      for (const op of REDUCTIONS) red.append(new Option(op, `r:${op}`));
      sel.append(red);
      let num: HTMLInputElement | null = null;
      if (d.size <= MAX_OPTIONS) {
        const idx = document.createElement('optgroup');
        idx.label = 'index';
        for (let i = 0; i < d.size; i++) idx.append(new Option(String(i), `i:${i}`));
        sel.append(idx);
      } else {
        sel.append(new Option('index', 'i:'));
        num = document.createElement('input');
        num.type = 'number';
        num.min = '0';
        num.max = String(d.size - 1);
        num.setAttribute('aria-label', `${d.name} index`);
      }
      const c = this.choices.get(d.name) ?? { kind: 'all' };
      sel.value = c.kind === 'all' ? 'all' : c.kind === 'reduce' ? `r:${c.op}` : num ? 'i:' : `i:${c.i}`;
      if (num) {
        num.hidden = c.kind !== 'index';
        if (c.kind === 'index') num.value = String(c.i);
      }
      const commit = () => {
        const v = sel.value;
        let choice: Choice;
        if (v === 'all') choice = { kind: 'all' };
        else if (v.startsWith('r:')) choice = { kind: 'reduce', op: v.slice(2) as Reduction };
        else {
          const i = num ? Number(num.value || 0) : Number(v.slice(2));
          choice = { kind: 'index', i: Math.max(0, Math.min(d.size - 1, Math.floor(i))) };
          if (num) num.value = String(choice.i);
        }
        if (num) num.hidden = choice.kind !== 'index';
        this.choices.set(d.name, choice);
        this.onPick(this.value());
      };
      sel.addEventListener('change', commit);
      num?.addEventListener('change', commit);
      wrap.append(name, sel);
      if (num) wrap.append(num);
      this.el.append(wrap);
    }
  }
}
