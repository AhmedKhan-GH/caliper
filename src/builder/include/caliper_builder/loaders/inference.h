#pragma once
// Inference of the reader of a dropped file, and the default loader for it.
//
// The reader comes from the extension first, then from the first bytes:
//   - a pickle or a torch archive is refused first, whatever the extension
//     says ("loader.pickle_refused", see pickle.h);
//   - when the extension names a reader and the bytes identify a different
//     binary format, inference fails with "loader.reader_mismatch", naming
//     both; text formats (csv, json, jsonl) have no signature, so they clash
//     only with bytes that identify a binary format;
//   - when the extension names no reader, the bytes decide among the binary
//     formats, and a file that matches none fails with "loader.no_reader",
//     which points to the Python loader.
// No function here throws.
#include <caliper_builder/error.h>
#include <caliper_builder/loaders/loader.h>
#include <caliper_builder/loaders/readers.h>
#include <caliper_builder/loaders/tensor_readers.h>

#include <string>
#include <vector>

namespace caliper_builder {

struct ReaderInference {
    Error error;
    ReaderKind reader = ReaderKind::csv;
    bool from_extension = false;
    std::string format;  // what the bytes identified (sniff_format), or ""
};

ReaderInference infer_reader(const std::string& path, const ReaderLimits& limits = {});

struct LoaderInference {
    Error error;
    Loader loader;
};

// The default loader for one file (a canonical path): its id comes from the
// file's stem, made unique against `taken_ids`; its input path is relative to
// the workspace when the file lies inside it; its reader is named explicitly;
// a tabular file gets one view named after the stem, a single-tensor file one
// tensor named after the stem, and a multi-tensor file `tensor = "*"` with the
// prefix "<stem>_".
LoaderInference infer_loader(const std::string& path, const std::string& workspace,
                             const std::vector<std::string>& taken_ids, const ReaderLimits& limits = {});

}  // namespace caliper_builder
