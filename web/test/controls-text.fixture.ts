import { CaliperControls, defineControls } from '../src/controls.js';
import { emit, type ControlRequest, type StreamClient, type StreamListener } from '../src/client/stream.js';
import type { ControlReplyMsg, ControlsMsg } from '../src/protocol/messages.js';

export class TextControlsSource implements StreamClient {
  state = 'open' as const;
  session = null;
  views = [];
  controls: ControlsMsg | null = {
    type: 'controls', run: { state: 'running' }, params: [{ name: 'prompt', type: 'text', value: 'initial' }], actions: [],
  };
  allowed = true;
  failSend = false;
  readonly calls: { request: ControlRequest; reply?: (r: ControlReplyMsg) => void }[] = [];
  private readonly listeners = new Set<StreamListener>();

  connect(): void {}
  close(): void {}
  subscribe(): number { return 1; }
  unsubscribe(): void {}
  requestDataView(): number { return 1; }
  sendEvent(): void {}
  listen(listener: StreamListener): () => void {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  }
  controlAccess() { return { shown: true, allowed: this.allowed, reason: this.allowed ? '' : 'Read only' }; }
  control(request: ControlRequest, reply?: (r: ControlReplyMsg) => void): number | null {
    if (!this.allowed) return null;
    if (this.failSend) throw new Error('Send refused');
    this.calls.push({ request, reply });
    return this.calls.length;
  }
  respond(index: number, status: 'ok' | 'error'): void {
    const call = this.calls[index];
    call.reply?.({ type: 'control_reply', request_id: index + 1, op: call.request.op, name: 'prompt', status,
      ...(status === 'ok' ? { value: call.request.value } : { error: { code: 'refused', message: 'Try a shorter prompt' } }),
    });
  }
  publish(value: string, extra = false): void {
    this.controls = { type: 'controls', run: { state: 'running' }, actions: [], params: [
      { name: 'prompt', type: 'text', value },
      ...(extra ? [{ name: 'other', type: 'boolean' as const, value: true }] : []),
    ] };
    emit(this.listeners, 'onControls', this.controls);
  }
  status(): void { emit(this.listeners, 'onStatus', this.state); }
  drop(): void { this.controls = null; emit(this.listeners, 'onControls', null); }
}

export function mountTextControls() {
  defineControls();
  const source = new TextControlsSource();
  const element = new CaliperControls();
  element.source = source;
  document.body.append(element);
  return { source, element };
}
