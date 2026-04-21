#include "dataset.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>

namespace fs = std::filesystem;

const char* LEAD_NAMES[NUM_LEADS] = {
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
};

// Uppercase suffix used in LeadFilesRowPerSample filenames.
static const char* kLeadFileSuffix[NUM_LEADS] = {
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
};

const char* format_display_name(DatasetFormat fmt) {
    switch (fmt) {
        case DatasetFormat::Auto:                  return "Auto-detect";
        case DatasetFormat::SingleFilePerSample:   return "Single CSV per sample";
        case DatasetFormat::LeadFilesRowPerSample: return "Lead files (row per sample)";
    }
    return "Unknown";
}

// ============================================================================
// Format detection
// ============================================================================
//
// LeadFilesRowPerSample is recognized when the directory contains files whose
// names case-insensitively contain `LEAD_<name>.csv` for each of the 12 leads.
// This deliberately does not lock to the `MDC_ECG_LEAD_` prefix so other
// variants (e.g. `LEAD_I.csv`, `recording_LEAD_V1.csv`) match too.
// ============================================================================

static std::string to_upper(std::string s) {
    for (auto& c : s) c = (char)std::toupper((unsigned char)c);
    return s;
}

DatasetFormat detect_format(const std::string& dir) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        return DatasetFormat::SingleFilePerSample;
    }

    bool has_all_leads = true;
    for (int i = 0; i < NUM_LEADS; i++) {
        std::string needle = std::string("LEAD_") + kLeadFileSuffix[i] + ".CSV";
        bool found = false;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            std::string name = to_upper(entry.path().filename().string());
            // needle must appear as the file's stem+ext suffix, so require
            // the name ends with `LEAD_<name>.CSV` or contains `LEAD_<name>.CSV` only once.
            if (name.size() >= needle.size() &&
                name.compare(name.size() - needle.size(), needle.size(), needle) == 0) {
                found = true;
                break;
            }
        }
        if (!found) { has_all_leads = false; break; }
    }

    if (has_all_leads) return DatasetFormat::LeadFilesRowPerSample;

    return DatasetFormat::SingleFilePerSample;
}

// ============================================================================
// SingleFilePerSample — one CSV per sample
// ============================================================================

namespace {

class SingleFilePerSampleLoader : public IDatasetLoader {
public:
    DatasetFormat format() const override { return DatasetFormat::SingleFilePerSample; }

    bool scan(const std::string& dir, std::vector<ECGSample>& out) override {
        out.clear();
        if (!fs::exists(dir) || !fs::is_directory(dir)) return false;

        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() != ".csv") continue;
            ECGSample s;
            s.filepath = entry.path().string();
            s.file_id = entry.path().stem().string();
            out.push_back(std::move(s));
        }
        std::sort(out.begin(), out.end(),
            [](const ECGSample& a, const ECGSample& b) { return a.file_id < b.file_id; });
        return !out.empty();
    }

    bool load(ECGSample& sample) override {
        std::ifstream file(sample.filepath);
        if (!file.is_open()) return false;

        std::string line;
        if (!std::getline(file, line)) return false; // header

        sample.raw.assign(NUM_LEADS, {});

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string val;
            int col = 0;
            while (std::getline(ss, val, ',') && col < NUM_LEADS) {
                size_t start = val.find_first_not_of(" \t\r\n");
                if (start == std::string::npos) { col++; continue; }
                val = val.substr(start);
                try {
                    sample.raw[col].push_back(std::stof(val));
                } catch (...) {}
                col++;
            }
        }

        if (sample.raw[0].empty()) return false;

        sample.num_samples = (int)sample.raw[0].size();
        // Infer sampling rate: 2500 → 250 Hz, 5000 → 500 Hz
        sample.sampling_rate = (sample.num_samples <= 2500) ? 250.0f : 500.0f;
        sample.loaded = true;
        sample.processed_valid = false;
        return true;
    }
};

// ============================================================================
// LeadFilesRowPerSample — 12 lead files, one row per recording
// ============================================================================
//
// Scan:
//   1. For each of 12 leads, find the file whose name ends with
//      `LEAD_<name>.csv` (case-insensitive).
//   2. Build a per-file row-offset index (byte offsets differ between files
//      because row lengths differ — can't share one index across leads).
//   3. Read a per-row ID file (`rwma-outcomes.csv` or similar) if present.
//
// Load:
//   Seek into each of the 12 lead files at that row's offset and parse the
//   line. std::ifstream is not thread-safe, so we serialize with a mutex.
// ============================================================================

class LeadFilesLoader : public IDatasetLoader {
public:
    DatasetFormat format() const override { return DatasetFormat::LeadFilesRowPerSample; }

    bool scan(const std::string& dir, std::vector<ECGSample>& out) override {
        out.clear();
        if (!fs::exists(dir) || !fs::is_directory(dir)) return false;

        // Find each lead file by case-insensitive suffix match.
        for (int i = 0; i < NUM_LEADS; i++) {
            std::string needle = std::string("LEAD_") + kLeadFileSuffix[i] + ".CSV";
            lead_paths_[i].clear();
            for (const auto& entry : fs::directory_iterator(dir)) {
                if (!entry.is_regular_file()) continue;
                std::string name = to_upper(entry.path().filename().string());
                if (name.size() >= needle.size() &&
                    name.compare(name.size() - needle.size(), needle.size(), needle) == 0) {
                    lead_paths_[i] = entry.path().string();
                    break;
                }
            }
            if (lead_paths_[i].empty()) {
                std::cerr << "[dataset] LeadFiles: missing lead file for "
                          << kLeadFileSuffix[i] << " in " << dir << std::endl;
                return false;
            }
        }

        // Per-file row-offset index.
        for (int i = 0; i < NUM_LEADS; i++) {
            if (!build_row_index(lead_paths_[i], lead_offsets_[i])) {
                std::cerr << "[dataset] Failed to index " << lead_paths_[i] << std::endl;
                return false;
            }
            if (i > 0 && lead_offsets_[i].size() != lead_offsets_[0].size()) {
                std::cerr << "[dataset] Row-count mismatch: " << lead_paths_[i]
                          << " has " << lead_offsets_[i].size() << " rows, expected "
                          << lead_offsets_[0].size() << std::endl;
                return false;
            }
        }

        std::vector<std::string> ids = read_id_sidecar(dir);

        int n = (int)lead_offsets_[0].size();
        out.reserve(n);
        for (int i = 0; i < n; i++) {
            ECGSample s;
            s.filepath = lead_paths_[0] + ":row=" + std::to_string(i);
            if (i < (int)ids.size() && !ids[i].empty()) {
                s.file_id = ids[i];
            } else {
                char buf[32]; std::snprintf(buf, sizeof(buf), "row%05d", i);
                s.file_id = buf;
            }
            out.push_back(std::move(s));
        }
        return !out.empty();
    }

    bool load(ECGSample& sample) override {
        auto pos = sample.filepath.rfind(":row=");
        if (pos == std::string::npos) return false;
        int row;
        try { row = std::stoi(sample.filepath.substr(pos + 5)); }
        catch (...) { return false; }
        if (row < 0 || row >= (int)lead_offsets_[0].size()) return false;

        std::lock_guard<std::mutex> lk(mtx_);

        sample.raw.assign(NUM_LEADS, {});
        int num_cols = -1;

        for (int lead = 0; lead < NUM_LEADS; lead++) {
            std::ifstream f(lead_paths_[lead]);
            if (!f.is_open()) return false;
            f.seekg(lead_offsets_[lead][row]);

            std::string line;
            if (!std::getline(f, line)) return false;

            auto& dst = sample.raw[lead];
            dst.reserve(num_cols > 0 ? num_cols : 6000);
            const char* p = line.c_str();
            const char* end = p + line.size();
            while (p < end) {
                while (p < end && (*p == ' ' || *p == '\t')) ++p;
                if (p >= end) break;
                char* next = nullptr;
                float v = std::strtof(p, &next);
                if (next == p) break;
                dst.push_back(v);
                p = next;
                while (p < end && *p != ',') ++p;
                if (p < end) ++p; // skip comma
            }

            if (num_cols < 0) num_cols = (int)dst.size();
            else if ((int)dst.size() != num_cols) {
                std::cerr << "[dataset] LeadFiles: column mismatch at row " << row
                          << " lead " << lead << ": got " << dst.size()
                          << " expected " << num_cols << std::endl;
                return false;
            }
        }

        if (num_cols <= 0) return false;

        sample.num_samples = num_cols;
        // Common sampling rate for row-per-sample datasets is 500 Hz.
        sample.sampling_rate = 500.0f;
        sample.loaded = true;
        sample.processed_valid = false;
        return true;
    }

private:
    static bool build_row_index(const std::string& path,
                                std::vector<std::streampos>& out) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return false;

        out.clear();
        std::string line;
        if (!std::getline(f, line)) return false; // header

        while (true) {
            std::streampos pos = f.tellg();
            if (pos == std::streampos(-1)) break;
            if (!std::getline(f, line)) break;
            if (line.empty() || (line.size() == 1 && line[0] == '\r')) continue;
            out.push_back(pos);
        }
        return !out.empty();
    }

    // Look for a sidecar CSV that supplies per-row IDs. We accept a few common
    // names; the first column of each data row is taken as the ID.
    static std::vector<std::string> read_id_sidecar(const std::string& dir) {
        static const char* kCandidates[] = {
            "rwma-outcomes.csv",
            "outcomes.csv",
            "labels.csv",
            "ids.csv",
        };
        std::vector<std::string> ids;
        for (const char* name : kCandidates) {
            fs::path p = fs::path(dir) / name;
            std::ifstream f(p);
            if (!f.is_open()) continue;
            std::string line;
            if (!std::getline(f, line)) continue; // header
            while (std::getline(f, line)) {
                if (line.empty()) continue;
                auto comma = line.find(',');
                ids.push_back(comma == std::string::npos ? line : line.substr(0, comma));
            }
            if (!ids.empty()) return ids;
        }
        return ids;
    }

    std::string lead_paths_[NUM_LEADS];
    std::vector<std::streampos> lead_offsets_[NUM_LEADS];
    std::mutex mtx_;
};

} // namespace

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<IDatasetLoader> make_dataset_loader(DatasetFormat fmt) {
    switch (fmt) {
        case DatasetFormat::Auto:
        case DatasetFormat::SingleFilePerSample:
            return std::make_unique<SingleFilePerSampleLoader>();
        case DatasetFormat::LeadFilesRowPerSample:
            return std::make_unique<LeadFilesLoader>();
    }
    return nullptr;
}
