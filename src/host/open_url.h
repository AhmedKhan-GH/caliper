#pragma once
// ============================================================================
// open_url.h: hands a link of Caliper API's browser viewer to the person's
// default browser, for the spec page's ⌘O Open in browser (src/main.cpp).
//
// Only a link of this machine's loopback API is opened:
// http://127.0.0.1:<port>/v0/spec?..., of the characters a link of the
// viewer holds, so nothing else can reach the browser through it. The
// browser is started as the system starts it for a link (`open` on macOS,
// ShellExecute on Windows, `xdg-open` elsewhere), from an argument vector and
// never through a shell, and the call does not wait for the browser.
// ============================================================================
#include <string>

namespace caliper_host {

// True for a link of the browser viewer on loopback (see above): its query
// is the token, of [A-Za-z0-9_-], and at most one applet, which
// viewer_applet_id() takes, and nothing else.
bool viewer_link_openable(const std::string& url);

// True for an id a link of the viewer may name, by the workspace's rule for
// an applet id (the spec format's): a lower-case letter, then at most 63
// lower-case letters, digits, _ or -. So never "." or "..".
bool viewer_applet_id(const std::string& id);

// Opens `url` in the default browser; false, with the reason, when the link
// is not one viewer_link_openable() takes or the browser cannot be started.
bool open_in_default_browser(const std::string& url, std::string* why);

}  // namespace caliper_host
