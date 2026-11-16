#pragma once
// The reader names of the loader format, their extensions, and the formats
// that a file's first bytes identify.
//
// A reader is one of csv, parquet, json, jsonl, arrow, npy, npz, safetensors,
// hdf5, image, wav and python. The first five produce tables through DuckDB;
// the others produce tensors, except python, the Python loader, which runs a
// script out of process and makes both (python_loader.h). Extensions compare
// without case. No function here throws.
#include <caliper_builder/error.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace caliper_builder {

enum class ReaderKind : uint8_t { csv, parquet, json, jsonl, arrow, npy, npz, safetensors, hdf5, image, wav, python };

const char* reader_name(ReaderKind reader);
bool reader_from_name(const std::string& name, ReaderKind* out);
// Every reader name, in the order above.
std::vector<std::string> reader_names();

// Whether the reader makes tables (csv, parquet, json, jsonl, arrow).
bool reader_is_tabular(ReaderKind reader);
// Whether a file of this reader holds exactly one tensor (npy, image, wav).
bool reader_is_single_tensor(ReaderKind reader);

// The reader for a path's extension; false when the extension has none.
bool reader_for_extension(const std::string& path, ReaderKind* out);

// The format that a file's first bytes identify: "npy", "zip", "parquet",
// "arrow", "hdf5", "png", "jpeg", "wav", "safetensors", "pickle", or "" when
// they identify none. Text formats have no signature and give "". `file_size`
// bounds the safetensors header length check.
std::string sniff_format(const uint8_t* bytes, std::size_t n, uint64_t file_size);

// The reader that a sniffed binary format calls for; false for "", "pickle"
// and anything else.
bool reader_for_format(const std::string& format, ReaderKind* out);

// Checks the options of a reader (the [reader] table of a loader, as a JSON
// object): every key must be one the reader takes, with a value of the right
// type and range. Fails with "loader.unknown_key" or "loader.bad_value", with
// `where` set to "reader.<key>".
//   csv:     header (bool), delimiter, quote, escape (one printable character;
//            a tab is allowed as the delimiter), skip (0 to 1,000,000),
//            null (a string of at most 256 characters), union_by_name (bool)
//   parquet: union_by_name;  json: format ("auto", "array", "unstructured"),
//   union_by_name;  jsonl: union_by_name;  hdf5: dataset (a string);
//   image: channels (0 to 4);  python: script (required), python,
//   timeout_s (see python_loader.h); every other reader takes no options.
Error validate_reader_options(ReaderKind reader, const nlohmann::json& options);

}  // namespace caliper_builder
