// Tests of the kept versions of published resources (history.h,
// design/programmable-graphics.md, Kept Versions) and of program.read
// (resource_commands.h, Program Commands): the even thinning, the epochs,
// the pins, the cursor over versions, and the elements that program.read
// reads inside each version's bytes.
#include "helpers.h"

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/session/history.h>
#include <caliper_builder/session/resource_commands.h>
#include <caliper_builder/session/session.h>

#include <doctest/doctest.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace caliper_builder;
using namespace session_test;

namespace {

const std::string kPin = "sha256:" + std::string(64, 'a');

// A spec with one instance of a piece, `id`, pinned to `hash`; inline pieces
// are added by the caller.
spec::Spec instance_spec(const std::string& id, const std::string& hash = kPin) {
    spec::Spec s;
    spec::Program p;
    p.id = id;
    spec::PieceRef ref;
    ref.name = "arcs";
    ref.version = 1;
    ref.hash = hash;
    p.piece = ref;
    s.programs.push_back(p);
    return s;
}

// A version of `count` elements of `stride` bytes at `step`, each 32-bit word
// the f32 value of step + word index.
ResourceVersion version_at(std::int64_t step, std::uint32_t count = 2, std::uint32_t stride = 8) {
    auto bytes = std::make_shared<std::vector<std::uint8_t>>((std::size_t)count * stride);
    for (std::size_t w = 0; w * 4 < bytes->size(); ++w) {
        const float f = float(step) + float(w);
        std::memcpy(bytes->data() + 4 * w, &f, 4);
    }
    ResourceVersion v;
    v.step = step;
    v.count = count;
    v.stride = stride;
    v.bytes = bytes;
    return v;
}

std::vector<std::int64_t> kept_steps_of(const StepHistory& h, const std::string& key) {
    std::vector<std::int64_t> out;
    for (std::int64_t s : h.kept_steps())
        if (auto v = h.resource_at(key, s); v && v->step == s) out.push_back(s);
    return out;
}

CommandOutput read(const spec::Spec& s, const StepHistory& h, nlohmann::json args) {
    const CommandDescriptor* d = find_command_descriptor("program.read");
    REQUIRE(d);
    CommandInvocation inv;
    inv.name = "program.read";
    inv.args = std::move(args);
    return resource_command(CommandContext{inv, *d}, s, h);
}

}  // namespace

TEST_CASE("resources: program.read is bound, reads only, and names a missing program") {
    auto s = open_session(fresh_dir("resource_commands_bound") + "/workspace");
    const CommandDescriptor* d = find_command_descriptor("program.read");
    REQUIRE(d);
    CHECK(!d->mutates);
    const auto r = invoke(*s, "program.read", d->examples.front());
    CHECK(!r.ok);
    CHECK(r.error.code == "program.not_found");
}

TEST_CASE("resources: a resource keeps every version until its ring is full, then thins evenly over the run") {
    StepHistory h;
    h.configure(instance_spec("tower"));
    for (std::int64_t step = 0; step <= 70; step += 10) h.append_resource("tower", "arcs", 8, {}, version_at(step));
    CHECK(kept_steps_of(h, "tower/arcs") == std::vector<std::int64_t>{0, 10, 20, 30, 40, 50, 60, 70});
    // The ninth version fills the ring: every second goes, the stride doubles to 20 steps.
    h.append_resource("tower", "arcs", 8, {}, version_at(80));
    CHECK(kept_steps_of(h, "tower/arcs") == std::vector<std::int64_t>{0, 20, 40, 60, 80});
    KeptResource k = h.kept_resources().at(0);
    CHECK(k.key == "tower/arcs");
    CHECK(k.stride == 20);
    // A version before the stride is kept only until a later one comes, so that the newest is always there.
    h.append_resource("tower", "arcs", 8, {}, version_at(90));
    CHECK(h.resource_at("tower/arcs", 1000)->step == 90);
    h.append_resource("tower", "arcs", 8, {}, version_at(100));
    CHECK(kept_steps_of(h, "tower/arcs") == std::vector<std::int64_t>{0, 20, 40, 60, 80, 100});
    for (std::int64_t step = 110; step <= 400; step += 10) h.append_resource("tower", "arcs", 8, {}, version_at(step));
    const std::vector<std::int64_t> kept = kept_steps_of(h, "tower/arcs");
    REQUIRE(kept.size() <= 9);
    CHECK(kept.front() == 0);
    CHECK(kept.back() == 400);
    // The kept versions span the run evenly: no gap is more than twice another, the newest aside.
    std::int64_t lo = 1 << 30, hi = 0;
    for (std::size_t i = 1; i + 1 < kept.size(); ++i) {
        lo = std::min(lo, kept[i] - kept[i - 1]);
        hi = std::max(hi, kept[i] - kept[i - 1]);
    }
    CHECK(hi <= 2 * lo);
    // The cursor moves over the versions, and a version is the newest at or before the cursor.
    CHECK(h.cursor().latest == 400);
    CHECK(h.cursor().first == 0);
    const CursorState back = h.step_cursor(-1);
    CHECK(back.step == kept[kept.size() - 2]);
    CHECK(h.resource_at("tower/arcs", back.step + 1)->step == back.step);
}

TEST_CASE("resources: a step below the newest starts an epoch; another pin or a removed program drops the versions") {
    StepHistory h;
    h.configure(instance_spec("tower"));
    h.append_resource("tower", "arcs", 0, {}, version_at(100));
    h.append_resource("tower", "arcs", 0, {}, version_at(200));
    h.append_resource("tower", "heads", 0, {}, version_at(200));
    // A fork from step 150: the versions after it belong to the run before the fork.
    h.append_resource("tower", "arcs", 0, {}, version_at(150));
    CHECK(kept_steps_of(h, "tower/arcs") == std::vector<std::int64_t>{100, 150});
    CHECK(kept_steps_of(h, "tower/heads").empty());
    CHECK(h.cursor().latest == 150);
    // A version of a program that is not an instance of the open spec is not kept.
    h.append_resource("other", "arcs", 0, {}, version_at(160));
    CHECK(!h.resource_at("other/arcs", 1000));
    // The same program pinned to another piece starts its rings over; a removed program drops them.
    h.configure(instance_spec("tower", "sha256:" + std::string(64, 'b')));
    CHECK(!h.resource_at("tower/arcs", 1000));
    h.append_resource("tower", "arcs", 0, {}, version_at(170));
    CHECK(h.resource_at("tower/arcs", 1000)->step == 170);
    h.configure(spec::Spec{});
    CHECK(h.kept_resources().empty());
    CHECK(h.kept_steps().empty());
    // A new attachment clears the versions.
    h.configure(instance_spec("tower"));
    h.append_resource("tower", "arcs", 0, {}, version_at(5));
    h.clear();
    CHECK(!h.resource_at("tower/arcs", 1000));
}

TEST_CASE("resources: one resource keeps at most a source's share of bytes, and a keep of 1 keeps the newest") {
    StepHistory h;
    h.configure(instance_spec("tower"));
    // Versions of 48 MiB share one buffer: the history counts their bytes, the test allocates one.
    auto big = std::make_shared<std::vector<std::uint8_t>>(std::size_t(48) << 20);
    for (std::int64_t step = 1; step <= 6; ++step) {
        ResourceVersion v;
        v.step = step;
        v.count = (std::uint32_t)(big->size() / 4);
        v.stride = 4;
        v.bytes = big;
        h.append_resource("tower", "arcs", 0, {}, v);
    }
    const KeptResource k = h.kept_resources().at(0);
    CHECK(k.versions <= 4);   // three for good and the provisional newest
    CHECK(k.bytes <= 4 * big->size());
    CHECK(k.last == 6);
    h.append_resource("tower", "heads", 1, {}, version_at(10));
    h.append_resource("tower", "heads", 1, {}, version_at(20));
    CHECK(kept_steps_of(h, "tower/heads") == std::vector<std::int64_t>{20});
}

TEST_CASE("resources: a version that replaces another at the same step has a new serial") {
    StepHistory h;
    h.configure(instance_spec("tower"));
    h.append_resource("tower", "arcs", 0, {}, version_at(100));
    const std::uint64_t first = h.resource_at("tower/arcs", 100)->serial;
    CHECK(first != 0);
    h.append_resource("tower", "arcs", 0, {}, version_at(100, 3));
    const std::optional<ResourceVersion> again = h.resource_at("tower/arcs", 100);
    REQUIRE(again);
    CHECK(again->count == 3);
    CHECK(again->serial != first);
}

TEST_CASE("resources: program.read reads each field of the version at the step, inside its bytes") {
    StepHistory h;
    spec::Spec s = instance_spec("tower");
    h.configure(s);
    const std::vector<ResourceField> fields = {{"from_token", "u32", 0}, {"weight", "f32", 4}};
    for (std::int64_t step : {100, 200}) {
        ResourceVersion v = version_at(step, 3, 8);
        auto bytes = std::make_shared<std::vector<std::uint8_t>>(*v.bytes);
        for (std::uint32_t i = 0; i < 3; ++i) {
            const std::uint32_t token = i + (std::uint32_t)step;
            std::memcpy(bytes->data() + 8 * i, &token, 4);
        }
        v.bytes = bytes;
        h.append_resource("tower", "arcs", 0, fields, v);
    }
    auto r = read(s, h, {{"id", "tower"}, {"resource", "arcs"}, {"step", 150}});
    REQUIRE(r.error.ok());
    CHECK(r.result["step"] == 100);
    CHECK(r.result["count"] == 3);
    REQUIRE(r.result["elements"].size() == 3);
    CHECK(r.result["elements"][1] == nlohmann::json::array({101, 103.0}));
    CHECK(r.result["kept"] == nlohmann::json({{"versions", 2}, {"first", 100}, {"last", 200}}));
    // The cursor's step when absent; first and count stay inside the valid elements.
    r = read(s, h, {{"id", "tower"}, {"resource", "arcs"}, {"first", 2}, {"count", 1024}});
    CHECK(r.result["step"] == 200);
    REQUIRE(r.result["elements"].size() == 1);
    CHECK(r.result["elements"][0][0] == 202);
    r = read(s, h, {{"id", "tower"}, {"resource", "arcs"}, {"first", 4000000000ull}});
    CHECK(r.result["elements"].empty());
    // Before any version: no step, nothing kept.
    r = read(s, h, {{"id", "tower"}, {"resource", "heads"}});
    REQUIRE(r.error.ok());
    CHECK(r.result["step"].is_null());
    CHECK(r.result["kept"]["versions"] == 0);
    // A field that would end past its element is not read.
    h.append_resource("tower", "bad", 0, {{"wide", "vec4<f32>", 4}}, version_at(300, 1, 8));
    r = read(s, h, {{"id", "tower"}, {"resource", "bad"}});
    CHECK(r.result["elements"][0].empty());
}

TEST_CASE("resources: program.read refuses a program that is no instance, and a resource an inline piece does not publish") {
    StepHistory h;
    spec::Spec s = instance_spec("tower");
    s.programs[0].piece->from = spec::PieceFrom::inline_piece;
    spec::Piece piece;
    piece.name = "arcs";
    piece.version = 1;
    spec::Resource res;
    res.id = "arcs";
    piece.resources.push_back(res);
    s.pieces.push_back(piece);
    spec::Program plain;
    plain.id = "plain";
    s.programs.push_back(plain);
    auto r = read(s, h, {{"id", "nope"}, {"resource", "arcs"}});
    CHECK(r.error.code == "program.not_found");
    r = read(s, h, {{"id", "plain"}, {"resource", "arcs"}});
    CHECK(r.error.code == "program.unknown_resource");
    r = read(s, h, {{"id", "tower"}, {"resource", "heads"}});
    CHECK(r.error.code == "program.unknown_resource");
    r = read(s, h, {{"id", "tower"}, {"resource", "arcs"}});
    CHECK(r.error.ok());
}
