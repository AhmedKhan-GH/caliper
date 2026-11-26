#pragma once

#include "net/wire.h"

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/commands/schema.h>
#include <caliper_builder/session/run_control.h>
#include <caliper_builder/session/session.h>
#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include <string>
#include <utility>
#include <vector>

namespace caliper_test {

struct InterventionRecord {
    nlohmann::json record;
    std::string op;

    nlohmann::json listed() const {
        auto result = record;
        result["op"] = op;
        if (result.contains("kind")) result["kind"] = op;
        return result;
    }

    nlohmann::json request() const {
        auto result = listed();
        for (const auto* key : {"since_step", "state", "error", "error_code"}) result.erase(key);
        return result;
    }
};

inline std::vector<InterventionRecord> intervention_records() {
    using nlohmann::json;
    std::vector<InterventionRecord> result;
    const std::vector<std::pair<std::string, json>> operations{
        {"ablate", json::object()},
        {"add", {{"vector", {0.25, -0.5}}, {"coefficient", 0.75}, {"positions", {0, 2}}}},
        {"scale", {{"feature", 2}, {"multiplier", 1.5}, {"dictionary", "fixture"}}},
        {"clamp", {{"feature", 3}, {"value", -0.5}, {"positions", {1}}}},
        {"patch", {{"source_prompt", "A small fixture"}, {"source_position", 1}, {"positions", {2}}}},
    };
    for (const auto& [op, fields] : operations) {
        for (const bool null_op : {false, true}) {
            json record = fields;
            record.update(json{{"point", "layers." + std::to_string(result.size()) + ".mlp_out"},
                               {"kind", op}, {"from_step", nullptr}, {"until_step", 100},
                               {"since_step", 12}, {"state", "active"}});
            if (null_op) record["op"] = nullptr;
            if (op == "patch") {
                record.update(json{{"state", "ended"}, {"error", "The fixture ended."},
                                   {"error_code", "intervene.hook_failed"}});
            }
            result.push_back({std::move(record), op});
        }
    }
    result.push_back({{{"point", "legacy"}, {"op", "ablate"}, {"from_step", 20},
                       {"until_step", nullptr}, {"since_step", nullptr}, {"state", "pending"}}, "ablate"});
    auto nullable = result.back();
    nullable.record["point"] = "nullable-kind";
    nullable.record["kind"] = nullptr;
    result.push_back(std::move(nullable));
    auto matching = result[6];
    matching.record["point"] = "matching-kind";
    matching.record["op"] = "clamp";
    result.push_back(std::move(matching));
    return result;
}

inline caliper_net::wire::Controls intervention_controls(const std::vector<InterventionRecord>& records) {
    using nlohmann::json;
    json active = json::array();
    for (const auto& entry : records) active.push_back(entry.record);
    const auto text = json{{"points", json::array()}, {"active", active}}.dump();
    caliper_net::wire::Controls controls;
    controls.run = caliper_net::Json::object();
    controls.run.set("state", "running");
    controls.run.set("step", 12);
    std::string error;
    REQUIRE(caliper_net::Json::parse(text.data(), text.size(), &controls.interventions, &error));
    std::vector<std::uint8_t> bytes;
    REQUIRE(caliper_net::wire::encode(controls, &bytes));
    caliper_net::wire::Message decoded;
    REQUIRE(caliper_net::wire::decode(bytes.data(), bytes.size(), &decoded).ok());
    return std::get<caliper_net::wire::Controls>(decoded);
}

inline std::vector<caliper_builder::Intervention> check_intervention_records(
    caliper_builder::RunControl& control, caliper_builder::Session& session,
    const std::vector<InterventionRecord>& records) {
    std::vector<InterventionRecord> ordered;
    for (const bool pending : {true, false})
        for (const auto& entry : records)
            if ((entry.record.at("state") == "pending") == pending) ordered.push_back(entry);
    const auto interventions = control.interventions();
    REQUIRE(interventions.size() == ordered.size());
    nlohmann::json expected = nlohmann::json::array();
    for (std::size_t index = 0; index < ordered.size(); ++index) {
        CAPTURE(index);
        CHECK(interventions[index].op == ordered[index].op);
        CHECK(interventions[index].specification == ordered[index].record);
        CHECK(caliper_builder::intervention_set_spec(interventions[index]) == ordered[index].request());
        expected.push_back(ordered[index].listed());
    }
    caliper_builder::CommandInvocation command;
    command.name = "intervene.list";
    command.args = nlohmann::json::object();
    command.origin = caliper_builder::CommandOrigin::mcp;
    const auto reply = session.invoke(command);
    REQUIRE_MESSAGE(reply.ok, reply.error.message);
    const auto* descriptor = caliper_builder::find_command_descriptor(command.name);
    REQUIRE(descriptor);
    CHECK(caliper_builder::validate_command_value(descriptor->result, reply.result).ok());
    CHECK(reply.result.at("interventions") == expected);
    // The list output must remain a valid lifecycle record on the existing wire route.
    auto listed = ordered;
    for (std::size_t index = 0; index < listed.size(); ++index)
        listed[index].record = reply.result.at("interventions").at(index);
    intervention_controls(listed);
    return interventions;
}

}  // namespace caliper_test
