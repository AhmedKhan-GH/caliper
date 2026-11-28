#pragma once
// ============================================================================
// spec_page.h: the browser viewer of spec-based instruments, the page that
// GET /v0/spec answers (api_server.h, Options::spec_page_html). It is the web
// package's spec page (web/src/spec/, built by web/build.mjs) as the checked-in
// copy python/caliper/_web/spec.html holds it, embedded at build time by
// cmake/embed_spec_page.cmake, so the app serves it without Node and without
// reading a file. Empty in a tree without that copy.
// ============================================================================
#include <string>

namespace caliper_api {

const std::string& spec_page_html();

}  // namespace caliper_api
