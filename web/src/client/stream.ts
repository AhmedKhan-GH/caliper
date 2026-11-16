// The interface every live view talks to. The WebSocket client (socket.ts)
// implements it over the wire; the mock source (mock/) and tests implement it
// in the page. Everything here is decoded messages, never bytes.

import type {
  ControlOp, ControlReplyMsg, ControlsMsg, DataView, ErrorMsg, ParamValue, SelectionMsg, SeriesMsg, SessionIdentity,
  SnapshotMsg, ViewEntry, ViewReplyMsg, ViewerEvent,
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
  /** 3.1: the run state, parameters and actions changed (null: the session has none, or another session). */
  onControls?(controls: ControlsMsg | null): void;
  /** 3.1: the selection of an item space changed. */
  onSelection?(selection: SelectionMsg): void;
}

/** What a viewer may do with the controls (wire protocol 3.1, "Who may control"). */
export interface ControlAccess {
  /** The server and the session have controls: show them. */
  shown: boolean;
  /** CONTROL may be sent. */
  allowed: boolean;
  /** Why not, for a tooltip, when shown but not allowed. */
  reason: string;
}

/** A CONTROL to send; the client adds the request id. */
export interface ControlRequest { op: ControlOp; name?: string; value?: ParamValue; count?: number }

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

  // --- 3.1, on clients that implement it ---------------------------------------
  /** The server's capabilities from WELCOME. */
  readonly capabilities?: readonly string[];
  /** The latest CONTROLS of the attached session, or null. */
  readonly controls?: ControlsMsg | null;
  /** The latest SELECTION of each item space of the attached session. */
  readonly selections?: ReadonlyMap<string, SelectionMsg>;
  /** Whether this viewer may control the run, and why not. */
  controlAccess?(): ControlAccess;
  /**
   * Sends a CONTROL, or returns null when it may not be sent. onReply hears
   * its CONTROL_REPLY, twice for a run (queued, then the result). A set of a
   * parameter goes out at most 30 times a second; a set inside that interval
   * replaces the waiting one, which is sent at the end of it.
   */
  control?(req: ControlRequest, onReply?: (reply: ControlReplyMsg) => void): number | null;
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
