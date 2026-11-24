import { PARAM_TEXT_MAX } from './messages.js';

const utf8 = new TextEncoder();
const strictUtf8 = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });

/** Free-text values preserve valid Unicode and exclude NUL. */
export function isParamText(value: unknown): value is string {
  if (typeof value !== 'string' || value.includes('\0')) return false;
  const bytes = utf8.encode(value);
  return bytes.length <= PARAM_TEXT_MAX && strictUtf8.decode(bytes) === value;
}
