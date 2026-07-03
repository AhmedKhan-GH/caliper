#include "ollama_models.h"
#include "app_paths.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdlib>

namespace fs = std::filesystem;

namespace {

std::string default_ollama_path() {
#ifdef __APPLE__
    const char* home = std::getenv("HOME");
    if (home) return std::string(home) + "/.ollama/models";
#elif defined(_WIN32)
    const char* userprofile = std::getenv("USERPROFILE");
    if (userprofile) return std::string(userprofile) + "/.ollama/models";
#else
    const char* home = std::getenv("HOME");
    if (home) return std::string(home) + "/.ollama/models";
#endif
    return "";
}

std::string config_path() {
    return caliper::app_data_path("opengllama_config.txt");
}

std::string extract_json_string(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\":\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) {
        needle = "\"" + key + "\": \"";
        pos = json.find(needle);
        if (pos == std::string::npos) return "";
    }
    pos += needle.size();
    auto end = json.find('"', pos);
    if (end == std::string::npos) return "";
    return json.substr(pos, end - pos);
}

uint64_t extract_json_uint(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\":";
    auto pos = json.find(needle);
    if (pos == std::string::npos) {
        needle = "\"" + key + "\": ";
        pos = json.find(needle);
        if (pos == std::string::npos) return 0;
    }
    pos += needle.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    uint64_t val = 0;
    while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
        val = val * 10 + (json[pos] - '0');
        ++pos;
    }
    return val;
}

} // anonymous namespace

OllamaModelStore::OllamaModelStore() {
    load_config();
    if (ollama_path_.empty()) {
        ollama_path_ = default_ollama_path();
    }
    refresh();
}

void OllamaModelStore::set_ollama_path(const std::string& path) {
    ollama_path_ = path;
    save_config();
    refresh();
}

void OllamaModelStore::refresh() {
    models_.clear();
    scan_manifests();
}

void OllamaModelStore::load_config() {
    std::ifstream f(config_path());
    if (!f.is_open()) return;

    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("ollama_path=", 0) == 0) {
            ollama_path_ = line.substr(12);
        }
    }
}

void OllamaModelStore::save_config() {
    std::ofstream f(config_path());
    if (!f.is_open()) return;
    f << "ollama_path=" << ollama_path_ << "\n";
}

void OllamaModelStore::scan_manifests() {
    fs::path manifests_dir = fs::path(ollama_path_) / "manifests" / "registry.ollama.ai" / "library";
    fs::path blobs_dir = fs::path(ollama_path_) / "blobs";

    if (!fs::is_directory(manifests_dir)) return;

    for (const auto& model_entry : fs::directory_iterator(manifests_dir)) {
        if (!model_entry.is_directory()) continue;
        std::string model_name = model_entry.path().filename().string();

        for (const auto& tag_entry : fs::directory_iterator(model_entry.path())) {
            if (!tag_entry.is_regular_file()) continue;
            std::string tag_name = tag_entry.path().filename().string();

            std::ifstream mf(tag_entry.path());
            if (!mf.is_open()) continue;

            std::ostringstream ss;
            ss << mf.rdbuf();
            std::string manifest = ss.str();

            // Find the model layer (mediaType contains "image.model")
            std::string model_marker = "application/vnd.ollama.image.model";
            auto marker_pos = manifest.find(model_marker);
            if (marker_pos == std::string::npos) continue;

            // Extract the layer object containing the model marker
            // Find the enclosing { } for this layer
            auto obj_start = manifest.rfind('{', marker_pos);
            auto obj_end = manifest.find('}', marker_pos);
            if (obj_start == std::string::npos || obj_end == std::string::npos) continue;

            std::string layer_json = manifest.substr(obj_start, obj_end - obj_start + 1);

            std::string digest = extract_json_string(layer_json, "digest");
            uint64_t size = extract_json_uint(layer_json, "size");

            if (digest.empty()) continue;

            // Convert digest "sha256:abc..." to blob filename "sha256-abc..."
            std::string blob_filename = digest;
            auto colon_pos = blob_filename.find(':');
            if (colon_pos != std::string::npos) {
                blob_filename[colon_pos] = '-';
            }

            fs::path blob_path = blobs_dir / blob_filename;
            if (!fs::exists(blob_path)) continue;

            OllamaModel m;
            m.name = model_name;
            m.tag = tag_name;
            m.blob_path = blob_path.string();
            m.size_bytes = size;
            models_.push_back(std::move(m));
        }
    }
}
