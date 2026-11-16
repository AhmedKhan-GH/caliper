#pragma once
// The API file of Caliper API: where caliper-mcp learns the port and the
// session token of the running app (docs/references/api.md, "Starting it").
//
// The app writes `api.json` in its app-data folder while it serves the API and
// removes it when it exits. default_api_file_path() finds that file as the app
// does: in CALIPER_DATA_DIR when that is set and not empty, otherwise in the
// app-data folder of the OS. read_api_file() reads and checks it strictly: the
// file is at most 64 KiB, `version` is 1, `port` is 1 to 65535, `url` is
// exactly http://127.0.0.1:<port>/v0 (so the client never talks to another
// host), and `token` is 16 to 256 characters of A-Z a-z 0-9 _ -, so it can go
// into a header unescaped. Unknown keys are ignored.
//
// The token is a secret. No Error made here holds it or any part of the file's
// text. No function here throws.
//
// Error codes: mcp.no_api_file, mcp.bad_api_file.
#include <caliper_builder/error.h>

#include <cstdint>
#include <string>

namespace caliper_builder::mcp {

struct ApiEndpoint {
    int port = 0;
    std::string token;        // never logged, printed or returned to a client
    std::int64_t pid = 0;     // the app's process id, 0 when the file has none
    std::string url;          // http://127.0.0.1:<port>/v0
    std::string stream_url;   // the stream's address without the token, or empty

    bool same_as(const ApiEndpoint& other) const { return port == other.port && token == other.token; }
};

struct ApiFileRead {
    ApiEndpoint endpoint;
    Error error;
};

// The API file the app writes: $CALIPER_DATA_DIR/api.json, or api.json in the
// app-data folder of the OS (macOS: ~/Library/Application Support/Caliper;
// Windows: %APPDATA%\Caliper; Linux: $XDG_DATA_HOME/caliper or
// ~/.local/share/caliper). Empty when no folder can be found.
std::string default_api_file_path();

// Parses the text of an API file. `where` names the file in errors.
ApiFileRead parse_api_file(const std::string& text, const std::string& where);

// Reads and parses the file at `path`.
ApiFileRead read_api_file(const std::string& path);

}  // namespace caliper_builder::mcp
