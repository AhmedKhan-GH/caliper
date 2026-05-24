#include "circuit_db.h"
#include "verilog_parser.h"

#include <duckdb.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>

namespace fs = std::filesystem;

struct CircuitDB::Impl {
    std::unique_ptr<duckdb::DuckDB> db;
    std::unique_ptr<duckdb::Connection> con;
    int total_designs = 0;
};

CircuitDB::CircuitDB() : impl_(std::make_unique<Impl>()) {}
CircuitDB::~CircuitDB() = default;

bool CircuitDB::open() {
    impl_->db = std::make_unique<duckdb::DuckDB>(nullptr);
    impl_->con = std::make_unique<duckdb::Connection>(*impl_->db);

    impl_->con->Query(R"(
        CREATE TABLE IF NOT EXISTS designs (
            id INTEGER PRIMARY KEY,
            name VARCHAR,
            path VARCHAR,
            num_gates INTEGER,
            num_connections INTEGER,
            total_power FLOAT,
            has_features BOOLEAN,
            has_netlist BOOLEAN,
            source VARCHAR
        )
    )");

    impl_->con->Query(R"(
        CREATE TABLE IF NOT EXISTS gates (
            design_id INTEGER,
            gate_idx INTEGER,
            inst_name VARCHAR,
            cell_type VARCHAR,
            drive_strength INTEGER,
            fanout_load FLOAT,
            fanout_resistance FLOAT,
            fanout_number INTEGER,
            input_slew FLOAT,
            output_slew FLOAT,
            delay FLOAT
        )
    )");

    return true;
}

bool CircuitDB::ingest_dataset(const std::string& dataset_dir,
                               const std::function<void(int, int)>& progress) {
    if (!impl_->con) return false;

    // Scan for design directories (each has feature.json + final_netlist.v + power_summary.txt)
    std::vector<fs::path> design_dirs;

    auto scan_final = [&](const fs::path& final_dir) {
        if (!fs::is_directory(final_dir)) return;
        for (auto& entry : fs::directory_iterator(final_dir)) {
            if (entry.is_directory()) design_dirs.push_back(entry.path());
        }
    };

    fs::path root(dataset_dir);

    if (fs::exists(root / "dataset" / "Final")) {
        // Root is the top-level dir (e.g. circuitNetv3/)
        scan_final(root / "dataset" / "Final");
        scan_final(root / "dataset_augment" / "Final");
    } else if (fs::exists(root / "Final")) {
        // Root is dataset/ itself — scan it and check for sibling augmented dir
        scan_final(root / "Final");
        scan_final(root.parent_path() / "dataset_augment" / "Final");
    }

    int total = (int)design_dirs.size();
    if (total == 0) return false;

    // Use a prepared statement for bulk insert
    auto appender_designs = impl_->con->Query("DELETE FROM designs");
    auto appender_gates = impl_->con->Query("DELETE FROM gates");

    impl_->con->Query("BEGIN TRANSACTION");

    for (int i = 0; i < total; i++) {
        if (progress) progress(i, total);

        auto& dir = design_dirs[i];
        std::string name = dir.filename().string();
        std::string path = dir.string();

        fs::path netlist_path = dir / "final_netlist.v";
        fs::path feature_path = dir / "feature.json";
        fs::path power_path = dir / "power_summary.txt";

        bool has_netlist = fs::exists(netlist_path);
        bool has_features = fs::exists(feature_path);

        float power = 0;
        if (fs::exists(power_path)) {
            power = read_power(power_path.string());
        }

        int num_gates = 0, num_connections = 0;

        // Quick gate count from netlist without full parse
        if (has_netlist) {
            std::ifstream f(netlist_path);
            std::string line;
            while (std::getline(f, line)) {
                // Count lines with instance patterns (rough)
                if (line.find('(') != std::string::npos &&
                    line.find('.') != std::string::npos &&
                    line.find("module") == std::string::npos &&
                    line.find("input") == std::string::npos &&
                    line.find("output") == std::string::npos) {
                    num_gates++;
                }
            }
        }

        std::string source = (path.find("augment") != std::string::npos) ? "augmented" : "original";

        std::ostringstream sql;
        sql << "INSERT INTO designs VALUES ("
            << i << ", '" << name << "', '" << path << "', "
            << num_gates << ", " << num_connections << ", "
            << power << ", " << (has_features ? "true" : "false") << ", "
            << (has_netlist ? "true" : "false") << ", '"
            << source << "')";
        impl_->con->Query(sql.str());
    }

    impl_->con->Query("COMMIT");
    impl_->total_designs = total;
    return true;
}

QueryResult CircuitDB::query(const std::string& sql) {
    QueryResult result;
    if (!impl_->con) {
        result.error = "Database not open";
        return result;
    }

    auto res = impl_->con->Query(sql);
    if (res->HasError()) {
        result.error = res->GetError();
        return result;
    }

    // Extract column names
    for (size_t i = 0; i < res->ColumnCount(); i++) {
        result.columns.push_back(res->ColumnName(i));
    }

    // Extract rows
    auto chunk = res->Fetch();
    while (chunk) {
        for (size_t row = 0; row < chunk->size(); row++) {
            std::vector<std::string> row_data;
            for (size_t col = 0; col < chunk->ColumnCount(); col++) {
                row_data.push_back(chunk->GetValue(col, row).ToString());
            }
            result.rows.push_back(std::move(row_data));
        }
        chunk = res->Fetch();
    }

    result.ok = true;
    return result;
}

std::vector<DesignEntry> CircuitDB::get_designs(int limit, int offset) {
    std::vector<DesignEntry> out;
    std::ostringstream sql;
    sql << "SELECT name, path, num_gates, num_connections, total_power, has_features, has_netlist "
        << "FROM designs ORDER BY name LIMIT " << limit << " OFFSET " << offset;

    auto res = query(sql.str());
    if (!res.ok) return out;

    for (auto& row : res.rows) {
        DesignEntry e;
        e.name = row[0];
        e.path = row[1];
        try { e.num_gates = std::stoi(row[2]); } catch (...) {}
        try { e.num_connections = std::stoi(row[3]); } catch (...) {}
        try { e.total_power = std::stof(row[4]); } catch (...) {}
        e.has_features = (row[5] == "true");
        e.has_netlist = (row[6] == "true");
        out.push_back(std::move(e));
    }
    return out;
}

int CircuitDB::design_count() const {
    return impl_->total_designs;
}
