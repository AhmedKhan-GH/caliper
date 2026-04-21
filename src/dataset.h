#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

// ============================================================================
// ECG dataset abstraction.
//
// Loaders describe the on-disk *format*, not a specific dataset. Users point
// the app at an arbitrary directory and either let auto-detect pick a format
// or override it manually.
//
// Supported formats:
//
//   SingleFilePerSample   — directory of .csv files; each file is one
//                           recording. First row is a header of lead names
//                           (I, II, III, aVR, aVL, aVF, V1..V6). Subsequent
//                           rows are time samples.
//
//   LeadFilesRowPerSample — directory containing 12 CSVs, one per lead, named
//                           with a `LEAD_<name>` pattern (e.g.
//                           MDC_ECG_LEAD_I.csv, MDC_ECG_LEAD_V6.csv). Each
//                           row in a lead file is one full recording; columns
//                           are time samples. Optional `rwma-outcomes.csv`
//                           (or similar) provides per-row sample IDs.
// ============================================================================

static constexpr int NUM_LEADS = 12;
extern const char* LEAD_NAMES[NUM_LEADS];

struct ECGSample {
    std::string file_id;
    std::string filepath;   // opaque to the UI; loaders encode whatever they need
    std::vector<std::vector<float>> raw;
    std::vector<std::vector<float>> processed;
    float sampling_rate = 0.0f;
    int num_samples = 0;
    bool loaded = false;
    bool processed_valid = false;

    struct LeadStats { float mean=0, stddev=0, min_val=0, max_val=0; };
    std::vector<LeadStats> stats;
};

enum class DatasetFormat {
    Auto,                    // caller wants detect_format() to choose
    SingleFilePerSample,
    LeadFilesRowPerSample,
};

const char* format_display_name(DatasetFormat fmt);

// Inspect a directory and pick the most likely format. Returns
// `SingleFilePerSample` as a conservative default when nothing matches.
DatasetFormat detect_format(const std::string& dir);

class IDatasetLoader {
public:
    virtual ~IDatasetLoader() = default;

    // Scan the backing directory and produce a list of samples (metadata only;
    // raw data is loaded lazily via load()). Returns false if the directory
    // layout doesn't match the loader's expected format.
    virtual bool scan(const std::string& dir, std::vector<ECGSample>& out) = 0;

    // Populate sample.raw, sampling_rate, num_samples for a sample produced by
    // this loader's scan(). Safe to call concurrently from a worker thread —
    // loaders guard any shared state internally.
    virtual bool load(ECGSample& sample) = 0;

    virtual DatasetFormat format() const = 0;
};

// Resolves `Auto` by calling detect_format(dir) when dir is supplied; callers
// that pass a concrete format get that loader directly.
std::unique_ptr<IDatasetLoader> make_dataset_loader(DatasetFormat fmt);
