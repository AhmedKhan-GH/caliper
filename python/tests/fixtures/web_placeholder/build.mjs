// Builds the placeholder browser assets into dist/, the same four files the
// web/ package's build writes. scripts/build_web.py runs it like the real build:
//
//     python scripts/build_web.py --web python/tests/fixtures/web_placeholder
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";

const here = new URL(".", import.meta.url);
const read = (p) => readFileSync(new URL(p, here), "utf8");
const write = (p, s) => writeFileSync(new URL(p, here), s);

const note =
  "CALIPER PLACEHOLDER ASSET. A stand-in for the build of the web/ package, " +
  "made from python/tests/fixtures/web_placeholder. It opens the stream socket, " +
  "counts binary frames and draws one bar per frame; it does not decode the wire " +
  "protocol. The build of web/ replaces it.";

const view = read("src/view.js");
const widget = read("src/widget.js");
const controls = read("src/controls-widget.js");
const page = read("src/index.html");

rmSync(new URL("dist", here), { recursive: true, force: true });
mkdirSync(new URL("dist", here));
write("dist/caliper-view.js", `// ${note}\n${view}\nexport { CaliperView, connect };\n`);
write("dist/caliper-widget.js", `// ${note}\n${view}\n${widget}`);
write("dist/caliper-controls-widget.js", `// ${note}\n${controls}`);
write("dist/index.html", page.replace("<!-- NOTE -->", `<!-- ${note} -->`));
