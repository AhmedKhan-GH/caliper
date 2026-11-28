// The browser package's main module: importing it registers <caliper-view>
// and <caliper-controls>.

import { CaliperView, define } from './element.js';
import { CaliperControls, defineControls } from './controls.js';
import { connect } from './client/socket.js';
import { parseRecording, ReplaySource } from './client/replay.js';

define();
defineControls();

export { CaliperControls, CaliperView, connect, parseRecording, ReplaySource };
export type { ParsedRecording, ReplayOptions } from './client/replay.js';
export type { ConnectionState, ControlAccess, StreamClient, StreamListener } from './client/stream.js';
export type * from './protocol/messages.js';
