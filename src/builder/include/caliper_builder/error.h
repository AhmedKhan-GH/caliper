#pragma once
// caliper_builder::Error: the one error value of the builder libraries. No
// function of a public header throws; a function that can fail returns a value
// that carries an Error, and an empty `code` means success.
//
// `code` is a stable, lower-case, dotted identifier that a program can match on
// (for example "spec.unknown_primitive" or "loader.pickle_refused"). `message`
// is one plain sentence for a person. `where` locates the fault when that helps:
// a path inside a document ("panel[2].bind.y"), a line and column ("12:7") or a
// file name.
#include <string>
#include <utility>

namespace caliper_builder {

struct Error {
    std::string code;
    std::string message;
    std::string where;

    Error() = default;
    Error(std::string c, std::string m, std::string w = {})
        : code(std::move(c)), message(std::move(m)), where(std::move(w)) {}

    bool ok() const { return code.empty(); }
    explicit operator bool() const { return !code.empty(); }
};

}  // namespace caliper_builder
