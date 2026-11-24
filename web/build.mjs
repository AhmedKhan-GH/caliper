// Builds the five files of the browser package into dist/:
//   caliper-view.js              the <caliper-view> and <caliper-controls> elements and connect(src)
//   caliper-widget.js            the anywidget module of caliper.widget()
//   caliper-controls-widget.js   the anywidget module of the controls panel (caliper.controls())
//   index.html                   the viewer page, with its fonts and script inlined
//   spec.html                    the browser viewer of spec applets, one file with its fonts and
//                                script inlined, which Caliper API serves at /v0/spec (the app
//                                embeds the checked-in copy, python/caliper/_web/spec.html)
// and, with --test, the test harness bundle into test-dist/.
//
// The fonts are WOFF 1.0 wrappers of the unmodified IBM Plex Sans Condensed
// and Source Code Pro files of site/assets/fonts (SIL OFL 1.1; the licences
// are in fonts/ beside them). Sternbild Display is not distributed with the
// repository, so the page sets the name "Caliper" in the text face.

import * as esbuild from 'esbuild';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import zlib from 'node:zlib';
import { execFileSync } from 'node:child_process';

const here = path.dirname(fileURLToPath(import.meta.url));
const at = (...p) => path.join(here, ...p);
const dist = at('dist');
fs.mkdirSync(dist, { recursive: true });

// Read the exact design token revision selected by the native build.
async function designTokens() {
  const pin = fs.readFileSync(at('..', 'cmake', 'SternbildTokens.cmake'), 'utf8');
  const commit = /set\(STERNBILD_TOKENS_COMMIT ([0-9a-f]{40})\)/.exec(pin)?.[1];
  const repository = /set\(STERNBILD_TOKENS_REPOSITORY (https:\/\/github\.com\/[^\s)]+)\)/.exec(pin)?.[1];
  if (!commit || !repository) throw new Error('palette: design token pin is incomplete');
  const sources = [process.env.STERNBILD_TOKENS_SOURCE,
    at('..', 'cmake-build-release', '_deps', 'sternbild_tokens-src'),
    at('wasm', 'build', '_deps', 'sternbild_tokens-src')].filter(Boolean);
  for (const source of sources) {
    if (!fs.existsSync(source)) continue;
    try {
      return JSON.parse(execFileSync('git', ['-C', source, 'show', `${commit}:design/tokens.json`],
        { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }));
    } catch { /* This checkout does not contain the pinned revision. */ }
  }
  const cache = at('node_modules', '.cache', 'sternbild-tokens', `${commit}.json`);
  if (fs.existsSync(cache)) return JSON.parse(fs.readFileSync(cache, 'utf8'));
  const source = repository.replace('https://github.com/', 'https://raw.githubusercontent.com/').replace(/\.git$/, '');
  const response = await fetch(`${source}/${commit}/design/tokens.json`, { signal: AbortSignal.timeout(30000) });
  if (!response.ok) throw new Error(`palette: design token download failed (${response.status})`);
  const text = await response.text();
  const tokens = JSON.parse(text);
  fs.mkdirSync(path.dirname(cache), { recursive: true });
  fs.writeFileSync(cache, text);
  return tokens;
}

// The app's colours, read from the C++ that the app is built from, so that the
// page and the app share one source: the house's colour tokens and class
// pigments (src/host/ui/sternbild_theme.h) and the live views' categorical
// palette (src/host/scene/scene_tree.cpp, kPalette). The table is written into
// src/render/palette.gen.ts only when it changes; it is checked in, so that
// the type check and the tests that import the sources directly find it.
async function writePalette() {
  const design = await designTokens();
  const theme = fs.readFileSync(at('..', 'src', 'host', 'ui', 'sternbild_theme.h'), 'utf8');
  const tree = fs.readFileSync(at('..', 'src', 'host', 'scene', 'scene_tree.cpp'), 'utf8');
  const ns = /namespace col \{([\s\S]*?)\}\s*\/\/ namespace col/.exec(theme);
  if (!ns) throw new Error('palette: no namespace col in sternbild_theme.h');
  const tokens = {};
  const color = (name, active = new Set()) => {
    if (active.has(name)) throw new Error(`palette: cyclic design token ${name}`);
    active.add(name);
    const token = name.split('.').reduce((node, key) => node?.[key], design);
    const value = token?.$value;
    if (typeof value === 'string' && /^\{[^}]+\}$/.test(value)) return color(value.slice(1, -1), active);
    const hex = value?.hex;
    if (!/^#[0-9a-f]{6}$/i.test(hex ?? '')) throw new Error(`palette: invalid color token ${name}`);
    return [1, 3, 5].map((offset) => parseInt(hex.slice(offset, offset + 2), 16));
  };
  for (const m of ns[1].matchAll(/constexpr ImU32 (\w+)\s*=\s*tokens::(\w+)::(\w+)\.imu32;/g))
    tokens[m[1]] = color(`${m[2]}.${m[3].replaceAll('_', '-')}`);
  const house = ['vehicle', 'dog', 'bird', 'cat', 'other'].map((n) => {
    if (!tokens[n]) throw new Error(`palette: sternbild_theme.h has no class pigment ${n}`);
    return tokens[n];
  });
  const table = /kPalette\[16\] = \{([\s\S]*?)\};/.exec(tree);
  if (!table) throw new Error('palette: no kPalette[16] in scene_tree.cpp');
  const categorical = [...table[1].matchAll(/SCENE_RGB\((\d+),\s*(\d+),\s*(\d+)\)/g)].map((m) => [+m[1], +m[2], +m[3]]);
  if (categorical.length !== 16) throw new Error(`palette: kPalette holds ${categorical.length} colours, not 16`);
  const rgb = (c) => `[${c.join(', ')}]`;
  const text = [
    '// Generated by web/build.mjs from src/host/ui/sternbild_theme.h and src/host/scene/scene_tree.cpp;',
    '// do not edit. The app draws with those constants, and the page with this table.',
    '',
    'export type Rgb = readonly [number, number, number];',
    '',
    '/** The house\'s colour tokens (sternbild::col), by their C++ names. */',
    'export const TOKENS: Readonly<Record<string, Rgb>> = {',
    ...Object.entries(tokens).map(([k, v]) => `  ${k}: ${rgb(v)},`),
    '};',
    '',
    '/** The class pigments in their fixed order: vehicle, dog, bird, cat, other. */',
    `export const HOUSE_PIGMENTS: readonly Rgb[] = [${house.map(rgb).join(', ')}];`,
    '',
    '/** The live views\' categorical palette (caliper_scene::kPalette), 16 colours. */',
    'export const CATEGORICAL: readonly Rgb[] = [',
    ...[0, 4, 8, 12].map((i) => `  ${categorical.slice(i, i + 4).map(rgb).join(', ')},`),
    '];',
    '',
  ].join('\n');
  const file = at('src', 'render', 'palette.gen.ts');
  if (!fs.existsSync(file) || fs.readFileSync(file, 'utf8') !== text) fs.writeFileSync(file, text);
}
await writePalette();

const banner = { js: '/* @caliper/view, Apache License 2.0 */' };
const common = { bundle: true, format: 'esm', target: 'es2022', minify: true, legalComments: 'none', logLevel: 'warning', banner };

await esbuild.build({ ...common, entryPoints: [at('src/index.ts')], outfile: path.join(dist, 'caliper-view.js') });
await esbuild.build({ ...common, entryPoints: [at('src/widget.ts')], outfile: path.join(dist, 'caliper-widget.js') });
await esbuild.build({ ...common, entryPoints: [at('src/controls-widget.ts')], outfile: path.join(dist, 'caliper-controls-widget.js') });

const page = await esbuild.build({
  ...common, banner: {}, entryPoints: [at('src/page/main.ts')], write: false, external: ['./caliper-view.js'],
});
const script = page.outputFiles[0].text.replace(/<\/(script)/gi, '<\\/$1').trim();
const font = (family, file) => {
  const b64 = fs.readFileSync(at('fonts', file)).toString('base64');
  return `  @font-face { font-family: "${family}"; src: url("data:font/woff;base64,${b64}") format("woff"); font-display: swap; }`;
};
const fonts = [
  '  /* IBM Plex Sans Condensed and Source Code Pro: SIL Open Font License 1.1. */',
  font('Plex Condensed', 'IBMPlexSansCondensed-Regular.woff'),
  font('Source Code Pro', 'SourceCodePro-Regular.woff'),
].join('\n');
const template = fs.readFileSync(at('src/page/index.html'), 'utf8');
const html = template.replace('/*@FONTS@*/', () => fonts).replace('/*@SCRIPT@*/', () => script);
fs.writeFileSync(path.join(dist, 'index.html'), html);

// The spec page needs no element module: everything it runs is in the page.
const specPage = await esbuild.build({ ...common, banner: {}, entryPoints: [at('src/spec/main.ts')], write: false });
const specScript = specPage.outputFiles[0].text.replace(/<\/(script)/gi, '<\\/$1').trim();
const specTemplate = fs.readFileSync(at('src/spec/index.html'), 'utf8');
fs.writeFileSync(path.join(dist, 'spec.html'),
  specTemplate.replace('/*@FONTS@*/', () => fonts).replace('/*@SCRIPT@*/', () => specScript));

if (process.argv.includes('--test')) {
  const protocolTest = at('test-dist/text-params.mjs');
  await esbuild.build({
    bundle: true, platform: 'node', format: 'esm', logLevel: 'warning',
    entryPoints: [at('test/text-params.node.ts')], outfile: protocolTest,
  });
  execFileSync(process.execPath, ['--test', protocolTest], { stdio: 'inherit' });
  await esbuild.build({
    bundle: true, format: 'esm', target: 'es2022', logLevel: 'warning', sourcemap: 'inline',
    entryPoints: [at('test/harness.ts')], outfile: at('test-dist/harness.js'),
  });
}

const sizes = ['caliper-view.js', 'caliper-widget.js', 'caliper-controls-widget.js', 'index.html', 'spec.html'].map((f) => {
  const buf = fs.readFileSync(path.join(dist, f));
  return `${f.padEnd(27)} ${String(buf.length).padStart(8)} bytes  ${String(zlib.gzipSync(buf, { level: 9 }).length).padStart(7)} gzipped`;
});
console.log(sizes.join('\n'));
