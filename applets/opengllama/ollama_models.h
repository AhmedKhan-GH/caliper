#pragma once

#include <string>
#include <vector>
#include <cstdint>

struct OllamaModel {
    std::string name;
    std::string tag;
    std::string blob_path;
    uint64_t size_bytes;
};

class OllamaModelStore {
public:
    OllamaModelStore();

    void set_ollama_path(const std::string& path);
    const std::string& ollama_path() const { return ollama_path_; }

    void refresh();
    const std::vector<OllamaModel>& models() const { return models_; }

private:
    void load_config();
    void save_config();
    void scan_manifests();

    std::string ollama_path_;
    std::vector<OllamaModel> models_;
};
