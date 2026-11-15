// The browser package's main module: importing it registers <caliper-view>.

import { CaliperView, define } from './element.js';
import { connect } from './client/socket.js';

define();

export { CaliperView, connect };
export type { ConnectionState, StreamClient, StreamListener } from './client/stream.js';
export type * from './protocol/messages.js';
