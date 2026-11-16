// The browser package's main module: importing it registers <caliper-view>
// and <caliper-controls>.

import { CaliperView, define } from './element.js';
import { CaliperControls, defineControls } from './controls.js';
import { connect } from './client/socket.js';

define();
defineControls();

export { CaliperControls, CaliperView, connect };
export type { ConnectionState, ControlAccess, StreamClient, StreamListener } from './client/stream.js';
export type * from './protocol/messages.js';
