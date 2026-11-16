#pragma once
// The native tensor readers: .npy, .npz, safetensors, PNG and JPEG images, WAV
// audio, and HDF5 (a refusal unless the build has libhdf5). Each reads a file
// that may be hostile: every size, count and offset in a header is checked
// against the file's real size and against ReaderLimits before anything is
// allocated or mapped, and every multiplication is checked for overflow.
//
// A reader returns its tensors in a TensorReadResult. A tensor whose bytes lie
// in the file in the catalog's layout (row-major, little-endian, a catalog
// dtype) is mapped where it lies; any other tensor is converted into owned
// memory (a "derived" tensor), within ReaderLimits::max_derived_bytes. A filter
// limits the work to the tensors a loader asked for; tensors it rejects are not
// read or inflated.
//
// Pickle never loads: an object dtype in an .npy file, or a pickle or torch
// archive inside an .npz file, fails with "loader.pickle_refused".
//
// Errors use the codes "loader.io", "loader.truncated", "loader.malformed",
// "loader.unsupported", "loader.too_large", "loader.pickle_refused",
// "loader.hdf5_not_available", and the "tensor.*" codes of check_shape. Their
// `where` is the file's path. No function here throws.
#include <caliper_builder/error.h>
#include <caliper_builder/loaders/tensor.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace caliper_builder {

struct ReaderLimits {
    uint64_t max_derived_bytes = 4ULL << 30;        // one converted, inflated or decoded tensor
    uint64_t max_npy_header = 1ULL << 20;
    uint64_t max_safetensors_header = 100ULL << 20;
    int max_json_depth = 64;
    uint64_t max_zip_members = 65536;
    uint64_t max_zip_directory = 64ULL << 20;       // bytes of an archive's central directory
    uint64_t max_tensors = 100000;                  // tensors in one file
    uint32_t max_image_side = 32768;
    uint64_t max_image_pixels = 1ULL << 28;
    uint64_t max_image_file = 512ULL << 20;
    uint32_t max_wav_channels = 1024;
};

struct ReadTensor {
    std::string name;              // the tensor's name in the file
    TensorData data;
    std::optional<double> rate_hz;  // the sample rate of an audio tensor
    std::vector<std::string> warnings;
    bool derived = false;          // converted into memory rather than mapped from the file
};

struct TensorReadResult {
    Error error;
    std::vector<ReadTensor> tensors;
    std::vector<std::string> warnings;  // about the file as a whole, such as skipped members
};

// Chooses tensors by their name in the file; an empty filter takes them all.
using TensorFilter = std::function<bool(const std::string& name)>;

// One tensor, named after the file's stem. Versions 1 to 3, C or Fortran order,
// either byte order.
TensorReadResult read_npy(const std::string& path, const ReaderLimits& limits = {});

// The .npy members of a zip archive, stored or deflated, with zip64 records.
// Member "<name>.npy" is tensor "<name>"; other members are skipped with a warning.
TensorReadResult read_npz(const std::string& path, const TensorFilter& filter = {},
                          const ReaderLimits& limits = {});

// Every tensor of a safetensors file, mapped. BF16 becomes a derived f32 tensor
// with a warning; F8 and unknown dtypes are skipped with a warning.
TensorReadResult read_safetensors(const std::string& path, const TensorFilter& filter = {},
                                  const ReaderLimits& limits = {});

// A PNG or JPEG as u8 [H, W, C]. `channels` is 0 (as stored) or 1 to 4.
TensorReadResult read_image(const std::string& path, int channels = 0, const ReaderLimits& limits = {});

// A WAV file as [frames, channels] with rate_hz.
TensorReadResult read_wav(const std::string& path, const ReaderLimits& limits = {});

// Whether this build reads HDF5. When it does not, read_hdf5 fails with
// "loader.hdf5_not_available".
bool hdf5_available();
TensorReadResult read_hdf5(const std::string& path, const std::string& dataset = {},
                           const TensorFilter& filter = {}, const ReaderLimits& limits = {});

// Writes a tensor as a version 1 .npy file whose data starts on a 64-byte
// boundary, and returns that offset in `data_offset`.
Error write_npy(const std::string& path, const TensorData& tensor, uint64_t* data_offset = nullptr);

}  // namespace caliper_builder
