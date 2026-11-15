// The interface every live view talks to. The WebSocket client (socket.ts)
// implements it over the wire; the mock source (mock/) and tests implement it
// in the page. Everything here is decoded messages, never bytes.

import type {
  DataView, ErrorMsg, SeriesMsg, SessionIdentity, SnapshotMsg, ViewEntry, ViewReplyMsg, ViewerEvent,
} from '../protocol/messages.js';

/** connecting: no session yet; open: attached to a live session; ended: done. */
export type ConnectionState = 'connecting' | 'open' | 'ended';

export interface StreamListener {
  onStatus?(state: ConnectionState): void;
  onSession?(session: SessionIdentity | null): void;
  onViews?(views: readonly ViewEntry[]): void;
  onSnapshot?(snapshot: SnapshotMsg): void;
  onSeries?(delta: SeriesMsg): void;
  onReply?(reply: ViewReplyMsg): void;
  onError?(error: ErrorMsg): void;
}

export interface StreamClient {
  readonly state: ConnectionState;
  readonly session: SessionIdentity | null;
  readonly views: readonly ViewEntry[];
  /** Opens the stream; calling it again is harmless. */
  connect(): void;
  /** Closes the stream for good. */
  close(): void;
  /** Adds a listener and returns the function that removes it. */
  listen(listener: StreamListener): () => void;
  /** Subscribes to a view under a data view (the view's default when omitted); returns the sub id. */
  subscribe(view: string, dataView?: DataView): number;
  unsubscribe(sub: number): void;
  /** Asks for a new data view of an open subscription; returns the request id. */
  requestDataView(sub: number, dataView: DataView): number;
  /** Sends a selection, pick or hover to the producer. */
  sendEvent(event: ViewerEvent, sub?: number, seq?: number, dataView?: DataView): void;
}

/** Calls fn on every listener that defines it, isolating listener errors. */
export function emit<K extends keyof StreamListener>(
  listeners: Iterable<StreamListener>, key: K, ...args: Parameters<NonNullable<StreamListener[K]>>
): void {
  for (const l of listeners) {
    const fn = l[key] as ((...a: unknown[]) => void) | undefined;
    if (!fn) continue;
    try {
      fn.apply(l, args);
    } catch (e) {
      console.error('caliper-view listener', e);
    }
  }
}
