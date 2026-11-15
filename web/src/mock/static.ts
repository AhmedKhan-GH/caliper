// A StreamClient that tests drive by hand: they declare views, push snapshots
// and series deltas with known data, and read back what the viewer sent.

import { emit, type ConnectionState, type StreamClient, type StreamListener } from '../client/stream.js';
import { KIND_IDS, type DataView, type Kind, type SnapshotMsg, type ViewEntry, type ViewerEvent } from '../protocol/messages.js';

export class StaticSource implements StreamClient {
  state: ConnectionState = 'connecting';
  session = { id: 'static', script: 'static.py', pid: 1, start_unix_ms: 0 };
  views: ViewEntry[] = [];
  readonly subs = new Map<number, { view: string; dataView?: DataView }>();
  readonly requests: { sub: number; view: string; dataView: DataView }[] = [];
  readonly events: { event: ViewerEvent; sub?: number; seq?: number; dataView?: DataView }[] = [];
  private readonly listeners = new Set<StreamListener>();
  private nextSub = 1;
  private nextReq = 1;
  private seq = 0;

  connect(): void {
    if (this.state !== 'connecting') return;
    this.state = 'open';
    emit(this.listeners, 'onStatus', 'open');
    emit(this.listeners, 'onSession', this.session);
    emit(this.listeners, 'onViews', this.views);
  }

  close(): void {
    this.state = 'ended';
    emit(this.listeners, 'onStatus', 'ended');
  }

  listen(l: StreamListener): () => void {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  }

  subscribe(view: string, dataView?: DataView): number {
    const sub = this.nextSub++;
    this.subs.set(sub, { view, dataView });
    return sub;
  }

  unsubscribe(sub: number): void {
    this.subs.delete(sub);
  }

  requestDataView(sub: number, dataView: DataView): number {
    const s = this.subs.get(sub);
    this.requests.push({ sub, view: s?.view ?? '', dataView });
    return this.nextReq++;
  }

  sendEvent(event: ViewerEvent, sub?: number, seq?: number, dataView?: DataView): void {
    this.events.push({ event, sub, seq, dataView });
  }

  /** Declares a view (and resends the view list). */
  addView(name: string, kind: Kind, sources: ViewEntry['sources'] = [], dflt: DataView = {}): void {
    this.views = [...this.views.filter((v) => v.name !== name), { id: this.views.length + 1, name, kind: KIND_IDS[kind], sources, default: dflt }];
    emit(this.listeners, 'onViews', this.views);
  }

  private subOf(view: string): number[] {
    return [...this.subs].filter(([, s]) => s.view === view).map(([k]) => k);
  }

  /** Sends a snapshot to every subscription of the view. */
  snapshot(view: string, s: Partial<SnapshotMsg> & { data: Float32Array | Uint8Array; shape: number[] }): void {
    const entry = this.views.find((v) => v.name === view);
    for (const sub of this.subOf(view)) {
      emit(this.listeners, 'onSnapshot', {
        type: 'snapshot', sub, view, seq: ++this.seq, stamp_ns: 0, kind: entry?.kind ?? KIND_IDS.heatmap, data_view: {},
        colormap: 0, flags: 0, vmin: 0, vmax: 1, stats: [], axes: [], ...s,
      } as SnapshotMsg);
    }
  }

  /** Sends series values start .. start + values.length. */
  series(view: string, start: number, values: Float32Array): void {
    for (const sub of this.subOf(view)) {
      emit(this.listeners, 'onSeries', { type: 'series', sub, view, start, total: start + values.length, stamp_ns: 0, values });
    }
  }

  /** Answers the latest data view request. */
  reply(dataView: DataView | null, error?: string): void {
    const r = this.requests[this.requests.length - 1];
    if (!r) return;
    emit(this.listeners, 'onReply', {
      type: 'view_reply', request_id: this.nextReq - 1, sub: r.sub, view: r.view,
      ...(error ? { error: { code: 'bad_data_view', message: error } } : { data_view: dataView ?? r.dataView }),
    });
  }
}
