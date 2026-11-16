// The run controls of wire protocol 3.1, shared by the <caliper-view> header
// and the <caliper-controls> panel: a badge with the run state, and Pause or
// Resume and Step buttons. Both renderers of a view use the same element
// header, so the controls look and behave the same whichever draws the view.

import type { ControlRequest, StreamClient } from '../client/stream.js';
import type { ControlReplyMsg, RunInfo } from '../protocol/messages.js';

/** The badge's text for a run state, or '' for none. */
export function runText(run: RunInfo | undefined | null): string {
  if (!run || run.state === 'none') return '';
  if (run.state === 'paused') return run.step != null ? `paused at ${run.step}` : 'paused';
  if (run.state === 'stepping') return run.steps_left != null ? `stepping, ${run.steps_left} left` : 'stepping';
  return run.state;
}

/**
 * Sends a CONTROL through the client and resolves with its final reply (for a
 * run, the result after "queued"), or with null when it could not be sent.
 * onReply hears every reply, the queued one included.
 */
export function sendControl(client: StreamClient | null, req: ControlRequest,
                            onReply?: (r: ControlReplyMsg) => void): Promise<ControlReplyMsg | null> {
  return new Promise((resolve) => {
    if (!client?.control) return resolve(null);
    const id = client.control(req, (r) => {
      onReply?.(r);
      if (r.status !== 'queued') resolve(r);
    });
    if (id === null) resolve(null);
    // A set coalesced into a later one (id 0) is answered through that one.
  });
}

export class RunBar {
  readonly el = document.createElement('span');
  private readonly badge = document.createElement('span');
  private readonly toggleBtn = document.createElement('button');
  private readonly stepBtn = document.createElement('button');

  /**
   * client gives the current client; buttons: show Pause or Resume and Step
   * (the badge shows whenever the run state is not none); onReply hears the
   * replies to this bar's requests.
   */
  constructor(private readonly client: () => StreamClient | null, private buttons: boolean,
              private readonly onReply: (r: ControlReplyMsg) => void) {
    this.el.className = 'run';
    this.badge.className = 'badge';
    this.badge.title = 'The run state of the script';
    this.toggleBtn.className = 'tool run-toggle';
    this.stepBtn.className = 'tool run-step';
    this.stepBtn.textContent = 'Step';
    this.toggleBtn.addEventListener('click', () => {
      const run = this.client()?.controls?.run;
      void (run && (run.state === 'paused' || run.state === 'pausing') ? this.resume() : this.pause());
    });
    this.stepBtn.addEventListener('click', () => void this.step(1));
    this.el.append(this.badge, this.toggleBtn, this.stepBtn);
    this.update();
  }

  setButtons(on: boolean): void {
    this.buttons = on;
    this.update();
  }

  pause(): Promise<ControlReplyMsg | null> {
    return sendControl(this.client(), { op: 'pause' }, this.onReply);
  }

  resume(): Promise<ControlReplyMsg | null> {
    return sendControl(this.client(), { op: 'resume' }, this.onReply);
  }

  step(n = 1): Promise<ControlReplyMsg | null> {
    return sendControl(this.client(), { op: 'step', count: Math.max(1, Math.floor(n)) }, this.onReply);
  }

  /** Redraws from the client's controls and access. */
  update(): void {
    const c = this.client();
    const access = c?.controlAccess?.();
    const run = access?.shown ? c?.controls?.run : null;
    const text = runText(run);
    this.badge.textContent = text;
    this.badge.hidden = !text;
    this.badge.dataset.state = run?.state ?? 'none';
    const show = this.buttons && !!access?.shown;
    this.toggleBtn.hidden = this.stepBtn.hidden = !show;
    if (!show || !access || !run) {
      this.el.hidden = !text;
      return;
    }
    this.el.hidden = false;
    const paused = run.state === 'paused' || run.state === 'pausing';
    this.toggleBtn.textContent = paused ? 'Resume' : 'Pause';
    const why = !access.allowed ? access.reason : run.state === 'none' ? 'The script has no checkpoint' : '';
    for (const b of [this.toggleBtn, this.stepBtn]) {
      b.disabled = !!why;
      b.title = why || (b === this.stepBtn ? 'Run one step, then pause' : paused ? 'Resume the run' : 'Pause the run at its next checkpoint');
    }
  }
}
