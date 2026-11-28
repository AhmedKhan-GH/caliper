#pragma once
// A workspace for the tests that need real loader outputs: a Parquet file
// written through DuckDB and a safetensors file, imported and run through the
// session's commands, and the dashboard of dashboard.spec.toml built command
// by command, as the session's acceptance test builds it.
#include <caliper_builder/commands/command.h>
#include <caliper_builder/session/session.h>
#include <caliper_builder/session/spec_edit.h>
#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>
#include <duckdb.hpp>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace spec_test {

namespace fs = std::filesystem;

// A new, empty folder <build>/test-tmp/<name>, as a canonical path.
inline std::string fresh_dir(const std::string& name) {
    const fs::path dir = fs::path(CALIPER_SPEC_TEST_TMP) / name;
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return fs::canonical(dir).string();
}

inline std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

inline void write_parquet(const std::string& path, const std::string& select) {
    fs::create_directories(fs::path(path).parent_path());
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    std::string quoted = "'";
    for (char c : path) quoted += c == '\'' ? std::string("''") : std::string(1, c);
    quoted += "'";
    auto r = con.Query("COPY (" + select + ") TO " + quoted + " (FORMAT parquet)");
    REQUIRE_MESSAGE(!r->HasError(), r->GetError());
}

inline void write_safetensors_f32(const std::string& path, const std::string& name, const std::vector<int64_t>& shape,
                                  const std::vector<float>& values) {
    nlohmann::json header;
    header[name] = {{"dtype", "F32"}, {"shape", shape}, {"data_offsets", {0, values.size() * 4}}};
    std::string h = header.dump();
    while (h.size() % 8) h += ' ';
    std::vector<char> bytes(8 + h.size() + values.size() * 4);
    const std::uint64_t n = h.size();
    for (int i = 0; i < 8; ++i) bytes[i] = static_cast<char>((n >> (8 * i)) & 0xFF);
    std::memcpy(bytes.data() + 8, h.data(), h.size());
    if (!values.empty()) std::memcpy(bytes.data() + 8 + h.size(), values.data(), values.size() * 4);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

inline nlohmann::json ok(caliper_builder::Session& s, const std::string& name, nlohmann::json args = nlohmann::json::object()) {
    caliper_builder::CommandInvocation inv;
    inv.name = name;
    inv.args = std::move(args);
    inv.origin = caliper_builder::CommandOrigin::ui;
    auto r = s.invoke(std::move(inv));
    INFO(name << ": " << r.error.code << " at " << r.error.where << ": " << r.error.message);
    REQUIRE(r.ok);
    return r.result;
}

inline caliper_builder::spec::Spec dashboard_fixture() {
    auto parsed = caliper_builder::spec::parse_toml(read_text(std::string(CALIPER_SPEC_FIXTURES) + "/dashboard.spec.toml"));
    REQUIRE(parsed.ok());
    return parsed.spec;
}

// Writes run42_metrics.parquet (200 steps of step, loss and accuracy) and
// attn.safetensors (weights [2, 3, 8, 8], a diagonal in every head) into
// `dir`, opens a session over `ws`, imports and runs both, and builds the
// dashboard command by command. Returns the open session.
inline std::unique_ptr<caliper_builder::Session> dashboard_workspace(const std::string& ws, const std::string& dir) {
    write_parquet(dir + "/run42_metrics.parquet",
                  "SELECT range AS step, 0.1 + 2.0 * exp(-range / 50.0) AS loss, "
                  "0.1 + 0.8 * (1 - exp(-range / 40.0)) AS accuracy FROM range(200)");
    std::vector<float> w(2 * 3 * 8 * 8, 0.0f);
    for (int l = 0; l < 2; ++l)
        for (int h = 0; h < 3; ++h)
            for (int q = 0; q < 8; ++q) w[((l * 3 + h) * 8 + q) * 8 + q] = 1.0f + (float)h;
    write_safetensors_f32(dir + "/attn.safetensors", "weights", {2, 3, 8, 8}, w);

    caliper_builder::SessionOptions options;
    options.registry.check_results = true;
    auto opened = caliper_builder::Session::open(ws, options);
    REQUIRE_MESSAGE(opened.error.ok(), opened.error.message);
    auto s = std::move(opened.session);
    const nlohmann::json imported =
        ok(*s, "data.import", {{"paths", {dir + "/run42_metrics.parquet", dir + "/attn.safetensors"}}, {"run", true}});
    REQUIRE(imported["refused"].empty());

    const caliper_builder::spec::Spec d = dashboard_fixture();
    nlohmann::json instrument{{"id", d.instrument.id}};
    if (d.instrument.title) instrument["title"] = *d.instrument.title;
    if (d.instrument.description) instrument["description"] = *d.instrument.description;
    ok(*s, "spec.new", instrument);
    for (const auto& src : d.sources) ok(*s, "source.add", caliper_builder::source_command_args(src));
    for (const auto& p : d.panels) ok(*s, "panel.create", caliper_builder::panel_command_args(p));
    ok(*s, "layout.set", {{"layout", caliper_builder::layout_to_flat(*d.layout, d)}});
    return s;
}

}  // namespace spec_test
