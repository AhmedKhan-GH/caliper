#pragma once
// The refusal errors that several readers share. Internal to the loaders module.
#include <caliper_builder/error.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace caliper_builder {
namespace detail {

// loader.pickle_refused, with the message of the contracts: the safe routes are
// a conversion to safetensors or the Python loader, which reads torch files
// weights-only. `what` says what was found ("an object dtype").
inline Error pickle_refused(const std::string& path, const std::string& what) {
    return Error("loader.pickle_refused",
                 "The file holds pickle data (" + what +
                     "), which Caliper does not load because loading a pickle can run code; convert it to "
                     "safetensors, or use the Python loader, which reads torch files weights-only "
                     "(torch.load(weights_only=True)) and marks the catalog entry with a warning.",
                 path);
}

// Whether bytes start with the marker of pickle protocols 2 to 5: 0x80 and the
// protocol number. Protocols 0 and 1 have no marker.
inline bool looks_like_pickle(const uint8_t* p, std::size_t n) {
    return n >= 2 && p[0] == 0x80 && p[1] >= 2 && p[1] <= 5;
}

}  // namespace detail
}  // namespace caliper_builder
