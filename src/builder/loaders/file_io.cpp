// InputFile: bounds-checked reads for the tensor readers.
#include "file_io.h"

#include <filesystem>
#include <limits>

namespace caliper_builder {
namespace detail {

namespace fs = std::filesystem;

Error InputFile::open(const std::string& path) {
    path_ = path;
    std::error_code ec;
    if (!fs::is_regular_file(fs::u8path(path), ec)) {
        return Error("loader.io", "The path is not a readable regular file.", path);
    }
    const uintmax_t size = fs::file_size(fs::u8path(path), ec);
    if (ec) return Error("loader.io", "The size of the file could not be read.", path);
    size_ = static_cast<uint64_t>(size);
    in_.open(fs::u8path(path), std::ios::binary);
    if (!in_) return Error("loader.io", "The file could not be opened.", path);
    return {};
}

Error InputFile::read(uint64_t offset, uint64_t n, void* out) {
    if (!fits(offset, n, size_)) {
        return Error("loader.truncated", "The file ends before the data its header describes.", path_);
    }
    if (n == 0) return {};
    if (offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
        n > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        return Error("loader.too_large", "The read is too large.", path_);
    }
    in_.clear();
    in_.seekg(static_cast<std::streamoff>(offset));
    if (!in_.read(static_cast<char*>(out), static_cast<std::streamsize>(n))) {
        return Error("loader.io", "The file could not be read.", path_);
    }
    return {};
}

Error InputFile::read_vec(uint64_t offset, uint64_t n, std::vector<uint8_t>* out, uint64_t max) {
    if (n > max) return Error("loader.too_large", "A block of the file exceeds the reader's limit.", path_);
    if (!fits(offset, n, size_)) {
        return Error("loader.truncated", "The file ends before the data its header describes.", path_);
    }
    try {
        out->resize(static_cast<std::size_t>(n));
    } catch (...) {
        return Error("loader.too_large", "Memory for the data could not be allocated.", path_);
    }
    return read(offset, n, out->data());
}

std::string file_stem(const std::string& path) {
    return fs::u8path(path).stem().u8string();
}

}  // namespace detail
}  // namespace caliper_builder
