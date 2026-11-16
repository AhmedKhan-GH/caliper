#pragma once
// The pickle refusal. Loading a pickle runs code that the file chooses, so no
// native reader ever loads one; the loaders refuse such a file with
// "loader.pickle_refused", whose message names the safe routes (convert the
// file to safetensors, or use the Python loader, which reads torch files
// weights-only and marks the catalog entry with a warning).
//
// A file is refused when any of these holds:
//   - its extension is .pkl, .pickle, .pt, .pth, .ckpt or .joblib, whatever
//     its bytes are;
//   - its bytes start with the marker of pickle protocols 2 to 5 (0x80 and the
//     protocol number), as a legacy torch .bin does;
//   - its extension has no reader and its bytes look like a pickle of
//     protocol 0 or 1 (a first opcode of "(", "]", "}" or "c" and a final
//     STOP opcode ".");
//   - it is a zip archive with a member named data.pkl or ending in
//     /data.pkl (the layout of torch.save), or a member that is a pickle by
//     its name or its bytes;
//   - it is an .npy file, or an .npz member, with the object dtype.
// The check reads at most the first and last bytes, a zip directory and the
// .npy headers inside it. No function here throws.
#include <caliper_builder/error.h>
#include <caliper_builder/loaders/tensor_readers.h>

#include <string>

namespace caliper_builder {

// Whether the file name has an extension that is always a pickle.
bool has_pickle_extension(const std::string& path);

// Returns loader.pickle_refused when the file is a pickle by any rule above,
// and success otherwise, including when the file cannot be read (the reader
// that opens it then reports that).
Error check_not_pickle(const std::string& path, const ReaderLimits& limits = {});

}  // namespace caliper_builder
