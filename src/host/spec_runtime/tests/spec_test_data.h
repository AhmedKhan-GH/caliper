#pragma once
// Resolved panel data made by hand for the spec runtime's tests: a spec from
// TOML text, and the slots of PanelData as Session::resolve() gives them.
#include <caliper_builder/session/resolve.h>
#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include <cstring>
#include <string>
#include <vector>

namespace spec_test {

namespace spec = caliper_builder::spec;
using caliper_builder::PanelData;
using caliper_builder::SlotData;
using nlohmann::json;

inline spec::Spec parse(const std::string& toml) {
    auto parsed = spec::parse_toml(toml);
    REQUIRE_MESSAGE(parsed.ok(), (parsed.diagnostics.empty() ? "" : parsed.diagnostics[0].message));
    return parsed.spec;
}

// A spec of one panel `p` over the sources given, from TOML lines.
inline spec::Spec one_panel(const std::string& sources, const std::string& panel) {
    return parse("caliper_spec = 2\n[instrument]\nid = \"t\"\n" + sources + "\n[[panel]]\nid = \"p\"\n" + panel + "\n");
}

inline SlotData column(const std::string& slot, const std::string& binding, json values, size_t index = 0) {
    SlotData s;
    s.slot = slot;
    s.index = index;
    s.binding = binding;
    const size_t dot = binding.find('.');
    s.source = binding.substr(0, dot);
    s.column = binding.substr(dot + 1);
    s.kind = SlotData::Kind::column;
    s.columns = {{s.column, "DOUBLE"}};
    s.values = std::move(values);
    return s;
}

inline SlotData table(const std::string& slot, const std::string& binding,
                      std::vector<caliper_builder::ColumnInfo> columns, json rows) {
    SlotData s;
    s.slot = slot;
    s.binding = binding;
    s.source = binding;
    s.kind = SlotData::Kind::table;
    s.columns = std::move(columns);
    s.values = std::move(rows);
    return s;
}

template <typename T>
SlotData tensor(const std::string& slot, const std::string& binding, caliper_builder::DType dtype,
                std::vector<int64_t> shape, const std::vector<T>& values, size_t index = 0) {
    SlotData s;
    s.slot = slot;
    s.index = index;
    s.binding = binding;
    s.source = binding;
    s.kind = SlotData::Kind::tensor;
    std::vector<uint8_t> bytes(values.size() * sizeof(T));
    if (!values.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
    auto made = caliper_builder::TensorData::from_bytes(dtype, std::move(shape), std::move(bytes));
    REQUIRE_MESSAGE(made.error.ok(), made.error.message);
    s.tensor = made.tensor;
    return s;
}

inline SlotData f32(const std::string& slot, const std::string& binding, std::vector<int64_t> shape,
                    const std::vector<float>& values, size_t index = 0) {
    return tensor<float>(slot, binding, caliper_builder::DType::f32, std::move(shape), values, index);
}

inline PanelData data(std::vector<SlotData> slots, spec::Primitive primitive, const std::string& panel = "p") {
    PanelData d;
    d.panel = panel;
    d.primitive = primitive;
    d.slots = std::move(slots);
    return d;
}

inline PanelData failed(const std::string& code, const std::string& where, const std::string& panel = "p") {
    PanelData d;
    d.panel = panel;
    d.error = caliper_builder::Error(code, "test", where);
    return d;
}

}  // namespace spec_test
