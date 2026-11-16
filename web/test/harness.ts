// The test page's helpers: the mock producer, the hand-driven source and the
// viewer page's logic, exposed on window for Playwright. The element itself
// comes from dist/caliper-view.js, loaded by harness.html.

import { MockSource, rng } from '../src/mock/mock.js';
import { StaticSource } from '../src/mock/static.js';
import { mountViewer } from '../src/page/viewer.js';
import { colormapLut, paletteLut } from '../src/render/colormap.js';
import { decode, encode } from '../src/protocol/codec.js';

Object.assign(window, { harness: { MockSource, StaticSource, mountViewer, rng, colormapLut, paletteLut, decode, encode } });
