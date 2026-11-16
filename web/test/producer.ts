// The mock producer (mock_server.py) for the browser tests, with the
// WebAssembly views' files served beside the page: started per test with its
// own options, its stdout read for the URL and for every EVENT and CONTROL it
// receives.

import { spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { REPO, WEB, wireModule } from './util.js';

export const PYTHON = process.env.CALIPER_PYTHON ?? path.join(REPO, '.venv', 'bin', 'python');
// The WebAssembly views' three files: CALIPER_WASM_DIR, else the package's
// _web folder, where scripts/build_wasm.py and scripts/fetch_wasm.py put them.
export const WASM = process.env.CALIPER_WASM_DIR ?? path.join(REPO, 'python', 'caliper', '_web');
export const HAVE_PYTHON = fs.existsSync(PYTHON);

export interface ProducerEvent { view: string; type: string; detail: Record<string, unknown>; sub?: number; seq?: number }
export interface Producer {
  url: string;
  proc: ChildProcess;
  events: ProducerEvent[];
  controls: Record<string, unknown>[];
  /** Every other line of JSON it printed, such as --edges' values. */
  printed: Record<string, unknown>[];
  stop(): void;
}

/** Starts mock_server.py with these options, serving the WebAssembly views from wasm. */
export async function startProducer(args: string[] = [], wasm: string = WASM): Promise<Producer> {
  const proc = spawn(PYTHON, [path.join(WEB, 'test', 'mock_server.py'), '--wire', wireModule(), '--wasm', wasm, ...args],
                     { stdio: ['ignore', 'pipe', 'inherit'] });
  const lines = readline.createInterface({ input: proc.stdout! });
  const events: ProducerEvent[] = [];
  const controls: Record<string, unknown>[] = [];
  const printed: Record<string, unknown>[] = [];
  const url = await new Promise<string>((ok, fail) => {
    proc.once('exit', (code) => fail(new Error(`mock_server.py exited with ${code}`)));
    lines.on('line', (line) => {
      if (line.startsWith('http://')) ok(line.trim());
      else if (line.startsWith('{')) {
        const j = JSON.parse(line) as { event?: ProducerEvent; control?: Record<string, unknown> };
        if (j.event) events.push(j.event);
        else if (j.control) controls.push(j.control);
        else printed.push(j);
      }
    });
  });
  return { url, proc, events, controls, printed, stop: () => proc.kill() };
}
