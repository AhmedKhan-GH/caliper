#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

struct DesignEntry {
    std::string name;
    std::string path;
    int num_gates = 0;
    int num_connections = 0;
    float total_power = 0;
    bool has_features = false;
    bool has_netlist = false;
};

struct QueryResult {
    std::vector<std::string> columns;
    std::vector<std::vector<std::string>> rows;
    std::string error;
    bool ok = false;
};

class CircuitDB {
public:
    CircuitDB();
    ~CircuitDB();

    bool open();
    bool ingest_dataset(const std::string& dataset_dir,
                        const std::function<void(int, int)>& progress = nullptr);
    QueryResult query(const std::string& sql);
    std::vector<DesignEntry> get_designs(int limit = 500, int offset = 0);
    int design_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
