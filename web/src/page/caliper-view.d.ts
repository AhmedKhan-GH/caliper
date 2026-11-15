// The page imports the element module that sits beside index.html in dist/.
import type { StreamClient } from '../client/stream.js';

export function connect(src: string): StreamClient;
