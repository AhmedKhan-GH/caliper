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

// True for a link of the browser viewer on loopback (see above).
bool viewer_link_openable(const std::string& url);

// Opens `url` in the default browser; false, with the reason, when the link
// is not one viewer_link_openable() takes or the browser cannot be started.
bool open_in_default_browser(const std::string& url, std::string* why);

}  // namespace caliper_host
