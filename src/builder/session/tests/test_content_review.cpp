// Tests of approvals bound to content: an agent's overwrite of a saved instrument
// waits with the diff of the file against the open spec it would write; when
// the open spec changes while the card waits, the approval is void: a held
// call gets a new card for the new content, and a direct approval answers
// history.conflict, so nothing is written that the person did not see; and an
// agent's spec.new or instrument.open over a spec with unsaved changes waits for
// review with the diff of the document it would replace.
#include <caliper_builder/session/session.h>

#include <caliper_builder/commands/journal.h>

#include <doctest/doctest.h>

#include <chrono>
#include <future>

#include "helpers.h"

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

namespace {

const CommandOrigin kUi = CommandOrigin::ui;
const CommandOrigin kAgent = CommandOrigin::agent;

// A workspace whose open spec is saved as the instrument dash.
struct Fixture {
    std::string ws;
    std::unique_ptr<Session> s;

    explicit Fixture(const std::string& name) {
        ws = fresh_dir(name) + "/workspace";
        write_text(ws + "/data/metrics.csv", "step,loss\n0,1.0\n1,0.5\n2,0.25\n");
        s = open_session(ws);
        ok(*s, "data.import", {{"paths", {"data/metrics.csv"}}});
        ok(*s, "spec.new", {{"id", "dash"}});
        ok(*s, "source.add", {{"id", "m"}, {"catalog", "table:metrics"}});
        ok(*s, "panel.create", {{"id", "loss"}, {"primitive", "series"}, {"bind", {{"x", "m.step"}, {"y", "m.loss"}}}});
        ok(*s, "instrument.save");
    }

    std::string saved_file() const { return read_text(ws + "/instruments/dash/dash.spec.toml"); }
    std::string open_text() const { return s->open_spec()->text; }
    void retitle(const std::string& title) {
        ok(*s, "panel.patch", {{"id", "loss"}, {"patch", {{"title", title}}}});
    }
    PendingApproval only_pending() const {
        const auto pending = s->registry().pending_approvals();
        REQUIRE(pending.size() == 1);
        return pending[0];
    }
};

}  // namespace

TEST_CASE("an agent's overwrite of a saved instrument shows the content it would write") {
    Fixture f("content-review-save");
    f.retitle("Changed");
    const std::string before = f.saved_file();
    const CommandResult r = invoke(*f.s, "instrument.save", {{"overwrite", true}}, kAgent);
    REQUIRE(r.error.code == "command.needs_approval");
    const PendingApproval asked = f.only_pending();
    REQUIRE(asked.review.is_object());
    const std::string diff = asked.review["diff"].get<std::string>();
    CHECK(diff.rfind("--- a/instruments/dash/dash.spec.toml\n", 0) == 0);
    CHECK(diff.find("+title = \"Changed\"") != std::string::npos);
    CHECK(f.saved_file() == before);
    REQUIRE(f.s->registry().approve(r.approval).ok);
    CHECK(f.saved_file() == f.open_text());
}

TEST_CASE("an overwrite approved after the open spec changed writes nothing the person did not see") {
    Fixture f("content-review-race");
    f.retitle("Shown");
    const std::string before = f.saved_file();

    // A direct approval after the open spec was replaced: refused, nothing written.
    const CommandResult r = invoke(*f.s, "instrument.save", {{"overwrite", true}}, kAgent);
    REQUIRE(r.error.code == "command.needs_approval");
    ok(*f.s, "instrument.save", {{"overwrite", true}});  // the person saves, so the open spec is clean
    ok(*f.s, "spec.new", {{"id", "dash"}, {"title", "Not shown"}}, kAgent);
    const CommandResult approved = f.s->registry().approve(r.approval);
    CHECK(approved.error.code == "history.conflict");
    CHECK(f.saved_file().find("Not shown") == std::string::npos);
    const JournalRead read = read_journal(f.ws + "/journal/commands.jsonl");
    CHECK(read.events.back().name == "instrument.save");
    CHECK(read.events.back().error == "history.conflict");

    // A held call: the card is reissued for the new content, and the call
    // completes when the person approves the new card.
    ok(*f.s, "history.undo");  // back to the spec that was saved
    f.retitle("First");
    CommandInvocation save;
    save.name = "instrument.save";
    save.args = {{"overwrite", true}};
    save.origin = kAgent;
    std::future<CommandResult> held = f.s->queue().submit(save);
    f.s->queue().drain();
    const PendingApproval first = f.only_pending();
    CHECK(first.review["diff"].get<std::string>().find("+title = \"First\"") != std::string::npos);
    f.retitle("Second");
    f.s->queue().approve(first.approval);
    f.s->queue().drain();
    CHECK(held.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout);
    CHECK(f.saved_file().find("Second") == std::string::npos);
    const PendingApproval second = f.only_pending();
    CHECK(second.approval != first.approval);
    CHECK(second.review["diff"].get<std::string>().find("+title = \"Second\"") != std::string::npos);
    f.s->queue().approve(second.approval);
    f.s->queue().drain();
    REQUIRE(held.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready);
    CHECK(held.get().ok);
    CHECK(f.saved_file().find("title = \"Second\"") != std::string::npos);
}

TEST_CASE("an agent's spec.new or instrument.open over a spec with unsaved changes waits for review") {
    Fixture f("content-review-replace");
    f.retitle("Unsaved");
    const std::string before = f.open_text();
    const CommandResult made = invoke(*f.s, "spec.new", {{"id", "other"}}, kAgent);
    REQUIRE(made.error.code == "command.needs_approval");
    const PendingApproval asked = f.only_pending();
    REQUIRE(asked.review.is_object());
    CHECK(asked.review["diff"].get<std::string>().find("-title = \"Unsaved\"") != std::string::npos);
    CHECK(f.s->registry().deny(made.approval).error.code == "command.denied");
    CHECK(f.open_text() == before);
    const CommandResult opened = invoke(*f.s, "instrument.open", {{"id", "dash"}}, kAgent);
    REQUIRE(opened.error.code == "command.needs_approval");
    CHECK(f.s->registry().deny(opened.approval).error.code == "command.denied");

    // A saved spec is replaced at once: its file and the undo history keep it.
    ok(*f.s, "instrument.save", {{"overwrite", true}});
    ok(*f.s, "spec.new", {{"id", "other"}}, kAgent);
    CHECK(f.s->open_spec()->spec.instrument.id == "other");
}
