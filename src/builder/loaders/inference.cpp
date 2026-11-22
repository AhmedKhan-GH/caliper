// Reader inference and default loaders for dropped files.
#include <caliper_builder/loaders/inference.h>
#include <caliper_builder/loaders/paths.h>
#include <caliper_builder/loaders/pickle.h>

#include "file_io.h"

#include <algorithm>
#include <filesystem>

namespace caliper_builder {

ReaderInference infer_reader(const std::string& path, const ReaderLimits& limits) {
    ReaderInference r;
    if ((r.error = check_not_pickle(path, limits))) return r;
    detail::InputFile f;
    if ((r.error = f.open(path))) return r;
    uint8_t head[16] = {};
    const uint64_t n = std::min<uint64_t>(f.size(), sizeof head);
    if ((r.error = f.read(0, n, head))) return r;
    r.format = sniff_format(head, static_cast<std::size_t>(n), f.size());
    ReaderKind by_ext, by_bytes;
    const bool has_ext = reader_for_extension(path, &by_ext);
    const bool has_bytes = reader_for_format(r.format, &by_bytes);
    if (has_ext) {
        if (has_bytes && by_bytes != by_ext) {
            r.error = Error("loader.reader_mismatch",
                            std::string("The file's extension says ") + reader_name(by_ext) + ", but its bytes are " +
                                r.format + "; rename the file or name the reader in the loader.",
                            path);
            return r;
        }
        r.reader = by_ext;
        r.from_extension = true;
        return r;
    }
    if (has_bytes) {
        r.reader = by_bytes;
        return r;
    }
    r.error = Error("loader.no_reader",
                    "No native reader knows this file; convert it to a supported format, or use the Python loader.",
                    path);
    return r;
}

LoaderInference infer_loader(const std::string& path, const std::string& workspace,
                             const std::vector<std::string>& taken_ids, const ReaderLimits& limits) {
    LoaderInference r;
    ReaderInference inferred = infer_reader(path, limits);
    if ((r.error = inferred.error)) return r;
    const std::filesystem::path p = std::filesystem::u8path(path);
    const std::string stem = p.stem().u8string();
    std::string id = make_id(stem);
    if (std::find(taken_ids.begin(), taken_ids.end(), id) != taken_ids.end()) {
        const std::string base = id.substr(0, 58);
        for (int k = 2; k < 100000; ++k) {
            const std::string candidate = base + "-" + std::to_string(k);
            if (std::find(taken_ids.begin(), taken_ids.end(), candidate) == taken_ids.end()) {
                id = candidate;
                break;
            }
        }
    }
    Loader& l = r.loader;
    l.id = id;
    l.title = p.filename().u8string();
    l.path = display_path(path, workspace);
    l.reader = inferred.reader;
    LoaderOutput out;
    std::string name = make_id(stem, true);
    if (reader_is_tabular(inferred.reader) || reader_is_single_tensor(inferred.reader)) {
        out.name = name;
    } else {
        if (name.size() > 60) name = name.substr(0, 60);
        out.name = name + "_";
        out.tensor = "*";
    }
    l.outputs.push_back(std::move(out));
    return r;
}

}  // namespace caliper_builder
