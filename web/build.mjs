// Builds the three files of the browser package into dist/:
//   caliper-view.js    the <caliper-view> element and connect(src)
//   caliper-widget.js  the anywidget module of caliper.widget()
//   index.html         the viewer page, with its fonts and script inlined
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

const here = path.dirname(fileURLToPath(import.meta.url));
const at = (...p) => path.join(here, ...p);
const dist = at('dist');
fs.mkdirSync(dist, { recursive: true });

const banner = { js: '/* @caliper/view, Apache License 2.0 */' };
const common = { bundle: true, format: 'esm', target: 'es2022', minify: true, legalComments: 'none', logLevel: 'warning', banner };

await esbuild.build({ ...common, entryPoints: [at('src/index.ts')], outfile: path.join(dist, 'caliper-view.js') });
await esbuild.build({ ...common, entryPoints: [at('src/widget.ts')], outfile: path.join(dist, 'caliper-widget.js') });

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

if (process.argv.includes('--test')) {
  await esbuild.build({
    bundle: true, format: 'esm', target: 'es2022', logLevel: 'warning', sourcemap: 'inline',
    entryPoints: [at('test/harness.ts')], outfile: at('test-dist/harness.js'),
  });
}

const sizes = ['caliper-view.js', 'caliper-widget.js', 'index.html'].map((f) => {
  const buf = fs.readFileSync(path.join(dist, f));
  return `${f.padEnd(18)} ${String(buf.length).padStart(8)} bytes  ${String(zlib.gzipSync(buf, { level: 9 }).length).padStart(7)} gzipped`;
});
console.log(sizes.join('\n'));
