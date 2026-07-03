#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

static constexpr int NUM_LEADS = 12;
extern const char* LEAD_NAMES[NUM_LEADS];

struct ECGSample {
    std::string file_id;
    std::string filepath;
    std::string label;
    std::vector<std::vector<float>> raw;
    std::vector<std::vector<float>> processed;
    float sampling_rate = 0.0f;
    int num_samples = 0;
    int original_num_samples = 0;
    bool downsampled = false;
    bool loaded = false;
    bool processed_valid = false;

    struct LeadStats { float mean=0, stddev=0, min_val=0, max_val=0; };
    std::vector<LeadStats> stats;
};

class IDatasetLoader {
public:
    virtual ~IDatasetLoader() = default;
    virtual bool scan(const std::string& dir, std::vector<ECGSample>& out) = 0;
    virtual bool load(ECGSample& sample) = 0;
};

std::unique_ptr<IDatasetLoader> make_dataset_loader();
