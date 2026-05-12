#include "dataset.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

namespace fs = std::filesystem;

const char* LEAD_NAMES[NUM_LEADS] = {
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
};

namespace {

void load_metadata(const std::string& dir, std::vector<ECGSample>& samples) {
    static const char* kCandidates[] = {
        "metadata_with_ekg.csv", "metadata.csv", "labels.csv",
    };
    for (const char* name : kCandidates) {
        fs::path p = fs::path(dir) / name;
        std::ifstream f(p);
        if (!f.is_open()) continue;

        std::string header;
        if (!std::getline(f, header)) continue;

        std::stringstream hss(header);
        std::string col;
        int id_col = -1, label_col = -1, ci = 0;
        while (std::getline(hss, col, ',')) {
            size_t s = col.find_first_not_of(" \t\r\n");
            if (s != std::string::npos) col = col.substr(s);
            size_t e = col.find_last_not_of(" \t\r\n");
            if (e != std::string::npos) col = col.substr(0, e + 1);
            if (col == "ECGTestID") id_col = ci;
            else if (col == "PatLabel") label_col = ci;
            ci++;
        }
        if (id_col < 0 || label_col < 0) continue;

        std::unordered_map<std::string, std::string> labels;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string val;
            int c = 0;
            std::string id, lbl;
            while (std::getline(ss, val, ',')) {
                if (c == id_col) {
                    auto dot = val.find('.');
                    id = (dot != std::string::npos) ? val.substr(0, dot) : val;
                    size_t s = id.find_first_not_of(" \t");
                    if (s != std::string::npos) id = id.substr(s);
                }
                if (c == label_col) {
                    lbl = val;
                    size_t s = lbl.find_first_not_of(" \t\r\n");
                    if (s != std::string::npos) lbl = lbl.substr(s);
                    size_t e = lbl.find_last_not_of(" \t\r\n");
                    if (e != std::string::npos) lbl = lbl.substr(0, e + 1);
                }
                c++;
            }
            if (!id.empty() && !lbl.empty()) labels[id] = lbl;
        }

        for (auto& samp : samples) {
            auto it = labels.find(samp.file_id);
            if (it != labels.end()) samp.label = it->second;
        }
        return;
    }
}

class UCDHLoader : public IDatasetLoader {
public:
    bool scan(const std::string& dir, std::vector<ECGSample>& out) override {
        out.clear();
        if (!fs::exists(dir) || !fs::is_directory(dir)) return false;

        for (const auto& entry : fs::recursive_directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() != ".csv") continue;
            const auto stem = entry.path().stem().string();
            if (stem.empty() || !std::isdigit((unsigned char)stem[0])) continue;
            ECGSample s;
            s.filepath = entry.path().string();
            s.file_id = stem;
            out.push_back(std::move(s));
        }
        std::sort(out.begin(), out.end(),
            [](const ECGSample& a, const ECGSample& b) { return a.file_id < b.file_id; });

        load_metadata(dir, out);
        return !out.empty();
    }

    bool load(ECGSample& sample) override {
        std::ifstream file(sample.filepath);
        if (!file.is_open()) return false;

        std::string line;
        if (!std::getline(file, line)) return false;

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

        sample.original_num_samples = (int)sample.raw[0].size();

        if (sample.original_num_samples > 2500) {
            sample.downsampled = true;
            for (auto& lead : sample.raw) {
                std::vector<float> ds;
                ds.reserve(lead.size() / 2);
                for (size_t i = 0; i < lead.size(); i += 2)
                    ds.push_back(lead[i]);
                lead = std::move(ds);
            }
        }

        sample.num_samples = (int)sample.raw[0].size();
        sample.sampling_rate = 250.0f;
        sample.loaded = true;
        sample.processed_valid = false;
        return true;
    }
};

} // namespace

std::unique_ptr<IDatasetLoader> make_dataset_loader() {
    return std::make_unique<UCDHLoader>();
}
