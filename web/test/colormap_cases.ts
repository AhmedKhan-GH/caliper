// The colormap's boundary cases as the C++ writes them (caliper_colormap_cases, from
// tests/shaders/colormap_cases.h): for each range its name, the 2,080-byte Cmap block with the
// keys of the shared derivation (src/host/gfx/colormap_keys.h) and the test table, the inputs as
// f32 bits, and the RGBA that the reference map_f32_to_rgba8 gives each.

import fs from 'node:fs';

export type ColormapCase = {
  name: string;
  vmin: number;
  vmax: number;
  block: Buffer;
  /** The block's 256-entry table, packed r | g << 8 | b << 16 | a << 24. */
  table: Uint32Array;
  inputs: Buffer;
  expected: Uint32Array;
};

export function readColormapCases(file: string): ColormapCase[] {
  const b = fs.readFileSync(file);
  if (b.toString('latin1', 0, 8) !== 'CMAPCAS2') throw new Error(`${file} is not a colormap case file`);
  const n = b.readUInt32LE(8);
  const out: ColormapCase[] = [];
  let at = 12;
  const words = (from: number, count: number) => new Uint32Array(b.buffer.slice(b.byteOffset + from, b.byteOffset + from + 4 * count));
  for (let r = 0; r < n; r++) {
    const len = b.readUInt32LE(at); at += 4;
    const name = b.toString('utf8', at, at + len); at += len;
    const block = b.subarray(at, at + 2080);
    const table = words(at + 1056, 256);
    at += 2080;
    const count = b.readUInt32LE(at); at += 4;
    const inputs = b.subarray(at, at + 4 * count); at += 4 * count;
    const expected = words(at, count); at += 4 * count;
    out.push({ name, vmin: block.readFloatLE(0), vmax: block.readFloatLE(4), block, table, inputs, expected });
  }
  if (at !== b.length) throw new Error(`${file} has ${b.length - at} bytes after its cases`);
  return out;
}
