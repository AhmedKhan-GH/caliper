// The approval cards without a window: their logic over a real builder
// Session with review on, as the app's frame loop drives it (the queue
// drained, then the cards synced), on a hand-moved clock for the queue's time
// limits; and the strip's layout as plain rectangles at two window sizes with
// one, three and eight cards, at two text scales.
#include "builder/approval_cards.h"

#include "spec_test_workspace.h"

#include <caliper_builder/commands/queue.h>
#include <caliper_builder/commands/registry.h>

#include <doctest/doctest.h>

#include <array>
#include <chrono>
#include <fstream>
#include <future>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace caliper_host;
using namespace spec_test;
using caliper_builder::CommandInvocation;
using caliper_builder::CommandOrigin;
using caliper_builder::CommandResult;
using nlohmann::json;
using namespace std::chrono_literals;

namespace {

bool ready(std::future<CommandResult>& f) { return f.wait_for(0s) == std::future_status::ready; }

// A workspace with one table and an open spec of one series panel, a
// session whose queue runs on a hand-moved clock, and the cards over it.
struct Rig {
    std::string root, ws;
    std::chrono::steady_clock::time_point clock{};
    std::unique_ptr<caliper_builder::Session> s;
    std::vector<std::array<std::string, 3>> activity;
    std::ostringstream err;
    std::unique_ptr<ApprovalCards> cards;
    double now = 0.0;

    explicit Rig(const std::string& name) {
        root = fresh_dir("cards-" + name);
        ws = root + "/ws";
        write_parquet(root + "/data/m.parquet",
                      "SELECT range AS step, 0.1 + 2.0 * exp(-range / 50.0) AS loss, "
                      "0.1 + 0.8 * (1 - exp(-range / 40.0)) AS accuracy FROM range(100)");
        caliper_builder::SessionOptions o;
        o.registry.check_results = true;
        o.queue.now = [this] { return clock; };
        auto opened = caliper_builder::Session::open(ws, o);
        REQUIRE_MESSAGE(opened.error.ok(), opened.error.message);
        s = std::move(opened.session);
        ok(*s, "data.import", {{"paths", {root + "/data/m.parquet"}}, {"run", true}});
        ok(*s, "spec.new", {{"id", "board"}, {"title", "Board"}});
        ok(*s, "source.add", {{"id", "m"}, {"catalog", "table:m"}});
        ok(*s, "panel.create",
           {{"id", "curves"}, {"primitive", "series"}, {"title", "Loss and accuracy"},
            {"bind", {{"x", "m.step"}, {"y", {"m.loss", "m.accuracy"}}}}, {"options", {{"y_log", true}}}});
        ApprovalCards::Hooks h;
        h.log = [this](const std::string& e, const std::string& o, const std::string& d) {
            activity.push_back({e, o, d});
        };
        h.err = &err;
        cards = std::make_unique<ApprovalCards>(h);
    }
    ~Rig() {
        cards.reset();
        s->close();
    }

    // A call as Caliper API submits one from mcp or agent: held for the
    // person's decision for up to 120 seconds.
    std::future<CommandResult> submit(const std::string& name, json args, CommandOrigin origin = CommandOrigin::mcp,
                                      std::string preview = {}) {
        CommandInvocation inv;
        inv.name = name;
        inv.args = std::move(args);
        inv.origin = origin;
        inv.preview = std::move(preview);
        caliper_builder::SubmitOptions so;
        so.approval_timeout = 120s;
        return s->queue().submit(std::move(inv), so);
    }
    // One frame of the app: the queue drained, then the cards synced.
    void frame(double dt = 0.2, bool busy = false) {
        now += dt;
        if (!busy) s->queue().drain();
        cards->sync(s.get(), busy, now);
    }
    json open_json() { return caliper_builder::spec::to_json(s->open_spec()->spec); }
    json y_log() {
        const std::optional<caliper_builder::OpenSpec> open = s->open_spec();
        REQUIRE(open);
        const auto* p = open->spec.find_panel("curves");
        REQUIRE(p);
        return caliper_builder::spec::to_json(*p)["options"]["y_log"];
    }
    const ApprovalCard& only() {
        REQUIRE(cards->cards().size() == 1);
        return cards->cards().front();
    }
};

json y_log_patch(bool on) { return {{"id", "curves"}, {"patch", {{"options", {{"y_log", on}}}}}}; }

const WidgetDiffLine* line_for(const ApprovalCard& c, const std::string& key) {
    for (const WidgetDiffLine& l : c.diff)
        if (l.key == key) return &l;
    return nullptr;
}

}  // namespace

TEST_CASE("approval cards: an mcp spec edit waits as a card with its diff, and Approve applies it") {
    Rig r("approve");
    auto f = r.submit("panel.patch", y_log_patch(false));
    r.frame();
    const ApprovalCard& c = r.only();
    CHECK(c.state == ApprovalState::waiting);
    CHECK(c.who == "MCP agent");
    CHECK(c.origin == "mcp");
    CHECK(c.what == "panel.patch \xC2\xB7 Loss and accuracy");
    CHECK(c.review);
    CHECK(c.loss.empty());
    const WidgetDiffLine* l = line_for(c, "y_log");
    REQUIRE(l);
    CHECK(l->before == "true");
    CHECK(l->after == "false");
    INFO(c.full_diff);
    CHECK(c.full_diff.find("y_log = true") != std::string::npos);
    CHECK(c.full_diff.find("y_log = false") != std::string::npos);
    CHECK(c.panels == std::vector<std::string>{"curves"});
    REQUIRE(c.time_left);
    CHECK(*c.time_left == 120s);
    CHECK(r.cards->card_for_panel("curves") == &c);
    CHECK(r.cards->waiting() == 1);
    REQUIRE(r.activity.size() == 1);
    CHECK(r.activity[0] == std::array<std::string, 3>{"mcp approval", "panel.patch \xC2\xB7 Loss and accuracy", "waiting"});
    CHECK(r.err.str().find("[approval] ap-1 panel.patch (mcp) waiting") != std::string::npos);
    CHECK_FALSE(ready(f));

    // The time left follows the queue's clock.
    r.clock += 30s;
    r.frame();
    CHECK(*r.only().time_left == 90s);
    CHECK(r.activity.size() == 1);

    // Approve goes through the queue; the decision runs at the next drain.
    CHECK(r.cards->approve(c.id));
    CHECK(r.only().state == ApprovalState::deciding);
    CHECK_FALSE(r.cards->approve(c.id));
    CHECK_FALSE(r.cards->deny(c.id));
    CHECK(r.y_log() == true);
    r.frame();
    REQUIRE(ready(f));
    const CommandResult res = f.get();
    CHECK_MESSAGE(res.ok, res.error.code);
    CHECK(r.y_log() == false);
    CHECK(r.only().state == ApprovalState::approved);
    CHECK(r.cards->waiting() == 0);
    CHECK(r.cards->card_for_panel("curves") == nullptr);
    REQUIRE(r.activity.size() == 2);
    CHECK(r.activity[1] == std::array<std::string, 3>{"mcp approval", "panel.patch \xC2\xB7 Loss and accuracy", "approved"});
    // The decided card stays a moment with its outcome, then goes.
    r.frame(1.0);
    CHECK(r.cards->cards().size() == 1);
    r.frame(1.1);
    CHECK(r.cards->cards().empty());
}

TEST_CASE("approval cards: a governed agent's card offers its rules; Always allow saves one and approves") {
    Rig r("rules");
    // An mcp client's edit: the card offers the three answers.
    auto f = r.submit("panel.patch", y_log_patch(false));
    r.frame();
    CHECK(r.only().rules);
    CHECK(std::string(approval_approve_label(true)) == "Allow once");
    CHECK(approval_always_label("panel.patch") == "Always allow panel.patch");
    CHECK(r.cards->allow_always(r.only().id, true));
    r.frame();
    REQUIRE(ready(f));
    CHECK(f.get().ok);
    CHECK(r.s->saved_agent_rules() == std::vector<caliper_builder::AgentRule>{{"mcp", "panel.patch"}});
    // The next one runs without a card.
    auto g = r.submit("panel.patch", y_log_patch(true));
    r.frame(1.5);
    REQUIRE(ready(g));
    CHECK(g.get().ok);
    // The widget agent's card (an edit into nothing but the open spec, client
    // widget) offers none, and its rule answer is refused.
    CommandInvocation w;
    w.name = "panel.patch";
    w.args = y_log_patch(false);
    w.origin = CommandOrigin::agent;
    w.client = "widget";
    caliper_builder::SubmitOptions so;
    so.approval_timeout = 120s;
    auto h = r.s->queue().submit(std::move(w), so);
    r.frame(1.5);
    REQUIRE(r.cards->waiting() == 1);
    const ApprovalCard* wc = nullptr;
    for (const ApprovalCard& c : r.cards->cards())
        if (c.state == ApprovalState::waiting) wc = &c;
    REQUIRE(wc);
    CHECK_FALSE(wc->rules);
    CHECK_FALSE(r.cards->allow_always(wc->id, false));
    CHECK(r.cards->deny(wc->id));
    r.frame();
    CHECK(h.get().error.code == "command.denied");
    // The layout gives the rule answers a row of their own above the buttons.
    ApprovalLayoutInput in;
    in.width = 1440;
    in.window_h = 900;
    ApprovalCardShape shape;
    shape.who = "MCP agent";
    shape.rules = true;
    shape.what_command = "panel.patch";
    in.cards = {shape};
    const WidgetMeasure measure = [](const std::string& t, WidgetFont) { return 7.0f * (float)t.size(); };
    const ApprovalStripLayout lay = layout_approval_strip(in, ApprovalLayoutMetrics{}, measure);
    REQUIRE(lay.cards.size() == 1);
    const ApprovalCardLayout& cl = lay.cards[0];
    CHECK_FALSE(cl.session.empty());
    CHECK_FALSE(cl.always.empty());
    CHECK(cl.session.bottom() <= cl.approve.y);
    CHECK(cl.always.x >= cl.session.right());
    CHECK(cl.always.right() <= cl.card.right());
}

TEST_CASE("approval cards: Deny answers command.denied and leaves the spec") {
    Rig r("deny");
    auto f = r.submit("panel.patch", y_log_patch(false));
    r.frame();
    CHECK(r.cards->deny(r.only().id));
    r.frame();
    REQUIRE(ready(f));
    CHECK(f.get().error.code == "command.denied");
    CHECK(r.y_log() == true);
    CHECK(r.only().state == ApprovalState::denied);
    CHECK(r.activity.back()[2] == "denied");
    CHECK(std::string(approval_state_tag(r.only().state)) == "DENIED");
}

TEST_CASE("approval cards: a request that times out leaves the strip") {
    Rig r("timeout");
    auto f = r.submit("panel.patch", y_log_patch(false));
    r.frame();
    REQUIRE(r.cards->waiting() == 1);
    r.clock += 119s;
    r.frame();
    CHECK(r.cards->waiting() == 1);
    CHECK(*r.only().time_left == 1s);
    r.clock += 2s;
    r.frame();   // the drain withdraws it, then the card settles
    REQUIRE(ready(f));
    CHECK(f.get().error.code == "command.approval_timeout");
    CHECK(r.cards->waiting() == 0);
    CHECK(r.only().state == ApprovalState::timed_out);
    CHECK(r.activity.back()[2] == "timed out");
    r.frame(2.1);
    CHECK(r.cards->cards().empty());
    CHECK(r.y_log() == true);
}

TEST_CASE("approval cards: an edit that conflicts after the spec changed underneath answers history.conflict") {
    Rig r("conflict");
    auto f = r.submit("panel.patch", y_log_patch(false));
    r.frame();
    const std::string id = r.only().id;
    // The person makes the same change first, so the shown change no longer
    // applies as it was shown.
    ok(*r.s, "panel.patch", y_log_patch(false));
    CHECK(r.cards->approve(id));
    r.frame();
    REQUIRE(ready(f));
    CHECK(f.get().error.code == "history.conflict");
    CHECK(r.only().state == ApprovalState::conflict);
    CHECK(r.only().error == "history.conflict");
    CHECK(r.activity.back()[2] == "history.conflict");
    CHECK(std::string(approval_state_tag(ApprovalState::conflict)) == "CONFLICT");
    r.frame(2.1);
    CHECK(r.cards->cards().empty());
}

TEST_CASE("approval cards: a destructive command's card says what is lost, and an api call has no time limit") {
    Rig r("destructive");
    ok(*r.s, "applet.save");
    auto f = r.submit("applet.save", {{"overwrite", true}});
    // loader.delete from the api origin answers at once and stays pending.
    CommandInvocation del;
    del.name = "loader.delete";
    del.args = {{"id", "m"}};
    del.origin = CommandOrigin::api;
    caliper_builder::SubmitOptions now;
    now.wait_for_approval = false;
    auto g = r.s->queue().submit(del, now);
    r.frame();
    REQUIRE(ready(g));
    CHECK(g.get().error.code == "command.needs_approval");
    REQUIRE(r.cards->cards().size() == 2);
    const ApprovalCard& save = r.cards->cards()[0];
    const ApprovalCard& gone = r.cards->cards()[1];
    CHECK(save.what == "applet.save \xC2\xB7 board");
    // An overwrite is approved by its content: the card shows the diff of the
    // file it would write (empty here, the same content), not a loss line.
    CHECK(save.review);
    CHECK(save.loss.empty());
    CHECK(save.diff.empty());
    CHECK(save.time_left);
    CHECK(gone.who == "API");
    CHECK(gone.what == "loader.delete \xC2\xB7 m");
    CHECK(gone.loss == "Loader m and its tables deleted");
    CHECK_FALSE(gone.time_left);
    CHECK(r.cards->waiting() == 2);
    CHECK(r.cards->approve(save.id));
    CHECK(r.cards->deny(gone.id));
    r.frame();
    REQUIRE(ready(f));
    CHECK(f.get().ok);
    CHECK(r.cards->waiting() == 0);
    CHECK(r.s->queue().approval_outcome(r.cards->cards()[1].id)->error.code == "command.denied");
    CHECK(r.activity.size() == 4);
}

TEST_CASE("approval cards: an approval made void by a change of what the card showed is reissued, not failed") {
    Rig r("reissued");
    ok(*r.s, "applet.save");
    // An outside agent overwrites the saved applet: held, with the content.
    auto f = r.submit("applet.save", {{"overwrite", true}});
    r.frame();
    const std::string first = r.only().id;
    // The open spec changes while the card waits; the person approves the card.
    ok(*r.s, "panel.patch", {{"id", "curves"}, {"patch", {{"title", "Changed"}}}});
    CHECK(r.cards->approve(first));
    r.frame();
    // The old card is gone without an outcome; a new card waits for the new
    // content, and the call still waits.
    REQUIRE(r.cards->cards().size() == 1);
    const ApprovalCard& second = r.only();
    CHECK(second.id != first);
    CHECK(second.state == ApprovalState::waiting);
    CHECK(second.name == "applet.save");
    CHECK(r.cards->waiting() == 1);
    CHECK_FALSE(ready(f));
    CHECK(r.err.str().find("[approval] " + first + " applet.save (mcp) reissued as " + second.id) != std::string::npos);
    CHECK(r.err.str().find("failed") == std::string::npos);
    CHECK(r.err.str().find("history.conflict") == std::string::npos);
    // Approving the new card completes the call.
    const std::string second_id = second.id;
    CHECK(r.cards->approve(second_id));
    r.frame();
    REQUIRE(ready(f));
    CHECK(f.get().ok);
    CHECK(r.cards->waiting() == 0);
}

TEST_CASE("approval cards: a new panel's card lists its keys, and the origins read as nouns") {
    Rig r("create");
    auto f = r.submit("panel.create", {{"id", "rows"}, {"primitive", "table"}, {"title", "Rows"},
                                       {"bind", {{"rows", "m"}}}});
    r.frame();
    const ApprovalCard& c = r.only();
    CHECK(c.what == "panel.create \xC2\xB7 Rows");
    const WidgetDiffLine* p = line_for(c, "primitive");
    REQUIRE(p);
    CHECK(p->kind == WidgetDiffLine::Kind::added);
    CHECK(p->after == "\"table\"");
    CHECK(line_for(c, "bind.rows"));
    CHECK(line_for(c, "panel rows") == nullptr);
    CHECK(c.panels.empty());   // no panel of the open spec changes
    r.cards->approve(c.id);
    r.frame();
    REQUIRE(ready(f));
    CHECK(f.get().ok);
    CHECK(r.open_json().dump().find("\"rows\"") != std::string::npos);

    CommandInvocation inv;
    inv.origin = CommandOrigin::agent;
    CHECK(approval_who(inv) == "Side chat");
    inv.preview = "widget-1";
    CHECK(approval_who(inv) == "Widget agent");
    inv.origin = CommandOrigin::api;
    CHECK(approval_who(inv) == "API");
    inv.name = "history.undo";
    inv.args = {{"steps", 2}};
    CHECK(approval_what(inv, nullptr) == "history.undo \xC2\xB7 2 steps");
    CHECK(approval_loss(inv) == "2 steps undone");
    inv.name = "loader.run";
    inv.args = {{"id", "rows"}};
    CHECK(approval_loss(inv) == "Tables of loader rows replaced");
}

TEST_CASE("approval cards: nothing is read while a long command holds the registry") {
    Rig r("busy");
    auto f = r.submit("panel.patch", y_log_patch(false));
    r.s->queue().drain();
    r.frame(0.2, true);
    CHECK(r.cards->cards().empty());
    r.frame(0.2, false);
    CHECK(r.cards->waiting() == 1);
    // The time left still moves while busy, and a decision still queues.
    r.clock += 10s;
    r.frame(0.2, true);
    CHECK(*r.only().time_left == 110s);
    CHECK(r.cards->approve(r.only().id));
    r.frame(0.2, true);
    CHECK(r.only().state == ApprovalState::deciding);
    r.frame(0.2, false);
    CHECK(r.only().state == ApprovalState::approved);
    REQUIRE(ready(f));
    // Without a session the cards go, silently.
    const size_t lines = r.activity.size();
    r.cards->sync(nullptr, false, r.now + 1.0);
    CHECK(r.cards->cards().empty());
    CHECK(r.activity.size() == lines);
}

TEST_CASE("approval cards: the test decisions file decides the oldest matching card") {
    Rig r("decisions");
    const std::string path = r.root + "/decisions.txt";
    { std::ofstream(path) << "approve panel.patch mcp\nbogus line\n"; }
    r.cards->set_decisions(path);
    auto a = r.submit("panel.patch", y_log_patch(false));
    r.frame();   // the card appears, the decision applies
    CHECK(r.only().state == ApprovalState::deciding);
    r.frame();
    REQUIRE(ready(a));
    CHECK(a.get().ok);
    CHECK(r.err.str().find("decision line ignored: bogus line") != std::string::npos);
    // A decision for another origin waits for a card of that origin.
    auto b = r.submit("panel.patch", y_log_patch(true));
    { std::ofstream(path, std::ios::app) << "deny panel.patch agent\ndeny panel.patch\n"; }
    r.frame();
    r.frame();
    r.frame();
    REQUIRE(ready(b));
    CHECK(b.get().error.code == "command.denied");
}

TEST_CASE("approval cards: the compact lines of a unified diff pair a removed and an added key") {
    const std::string diff = "--- a/x.spec.toml\n+++ b/x.spec.toml\n@@ -3,2 +3,3 @@\n [panels.options]\n"
                             "-y_log = true\n+y_log = false\n+x_label = \"Step\"\n-title = \"Old\"\n";
    const std::vector<WidgetDiffLine> lines = approval_diff_from_unified(diff);
    REQUIRE(lines.size() == 3);
    CHECK(lines[0].kind == WidgetDiffLine::Kind::changed);
    CHECK(lines[0].key == "y_log");
    CHECK(lines[0].before == "true");
    CHECK(lines[0].after == "false");
    CHECK(lines[1].kind == WidgetDiffLine::Kind::added);
    CHECK(lines[1].key == "x_label");
    CHECK(lines[2].kind == WidgetDiffLine::Kind::removed);
    CHECK(lines[2].before == "\"Old\"");
}

// ---- the layout ---------------------------------------------------------------------------

namespace {

// Fixed advances: 7 px a character in the UI face and in mono, 8 in tags.
WidgetMeasure measure(float scale = 1.0f) {
    return [scale](const std::string& s, WidgetFont f) {
        size_t n = 0;
        for (unsigned char c : s) n += (c & 0xC0) != 0x80;
        return (float)n * (f == WidgetFont::tag ? 8.0f : 7.0f) * scale;
    };
}

ApprovalCardShape review_shape(int lines = 3, bool warnings = true) {
    ApprovalCardShape s;
    s.who = "MCP agent";
    s.review = true;
    s.diff_lines = lines;
    s.warnings = warnings;
    return s;
}

ApprovalCardShape loss_shape() {
    ApprovalCardShape s;
    s.who = "API";
    return s;
}

ApprovalLayoutInput input(float w, float h, int n) {
    ApprovalLayoutInput in;
    in.width = w;
    in.window_h = h;
    in.waiting = n;
    for (int i = 0; i < n; ++i) in.cards.push_back(i % 3 == 2 ? loss_shape() : review_shape());
    return in;
}

void check_strip(const ApprovalStripLayout& L, const ApprovalLayoutInput& in, const ApprovalLayoutMetrics& m) {
    const WidgetRect strip{0, 0, in.width, L.height};
    CHECK(L.height > 0.0f);
    CHECK(L.height <= std::floor(in.window_h * m.max_share));
    CHECK(L.cards.size() + (size_t)L.hidden == in.cards.size());
    CHECK(!L.cards.empty());
    CHECK((L.hidden == 0) == L.more.empty());
    if (!L.more.empty()) {
        CHECK(strip.contains(L.more));
        CHECK(L.more.x >= L.cards.back().card.right());
    }
    if (!L.label.empty()) {
        CHECK(strip.contains(L.label));
        CHECK(L.label.right() <= L.cards.front().card.x);
    }
    // The count stands under the label, when several wait.
    if (!L.label.empty()) {
        CHECK((in.waiting > 1) == !L.count.empty());
    } else {
        CHECK(L.count.empty());
    }
    for (size_t i = 0; i < L.cards.size(); ++i) {
        const ApprovalCardLayout& c = L.cards[i];
        INFO("card " << i << " at " << c.card.x << "," << c.card.y << " " << c.card.w << "x" << c.card.h);
        CHECK(strip.contains(c.card));
        CHECK(c.card.w <= m.max_card);
        if (i + 1 < L.cards.size()) {
            CHECK_FALSE(c.card.overlaps(L.cards[i + 1].card));
            CHECK(L.cards[i + 1].card.x > c.card.x);
        }
        std::vector<WidgetRect> parts = {c.who, c.what, c.time, c.body, c.warnings, c.approve, c.deny, c.diff};
        for (const WidgetRect& p : parts) {
            if (p.empty()) continue;
            INFO("part at " << p.x << "," << p.y << " " << p.w << "x" << p.h);
            CHECK(c.card.contains(p));
        }
        std::vector<WidgetRect> row = {c.who, c.what, c.time};
        std::vector<WidgetRect> buttons = {c.approve, c.deny, c.diff};
        for (auto* group : {&row, &buttons})
            for (size_t a = 0; a < group->size(); ++a)
                for (size_t b = a + 1; b < group->size(); ++b) CHECK_FALSE((*group)[a].overlaps((*group)[b]));
        CHECK(c.body.y >= c.what.bottom());
        if (!c.warnings.empty()) CHECK(c.warnings.y >= c.body.bottom());
        if (!c.approve.empty()) CHECK(c.approve.y >= c.body.bottom());
    }
}

}  // namespace

TEST_CASE("approval cards: the strip's layout at two window sizes with one, three and eight cards") {
    const ApprovalLayoutMetrics m = ApprovalLayoutMetrics::for_text_scale(1.0f);
    struct Case {
        float w, h;
        int n, shown;
        float card_w;
    };
    // The label takes 72 px and its margins; a card is at least 340 px wide
    // and at most 620.
    const Case cases[] = {
        {2560, 1440, 1, 1, 620}, {2560, 1440, 3, 3, 620}, {2560, 1440, 8, 7, 0},
        {1440, 900, 1, 1, 620},  {1440, 900, 3, 3, 441},  {1440, 900, 8, 3, 0},
    };
    for (const Case& k : cases) {
        INFO(k.w << "x" << k.h << " with " << k.n);
        const ApprovalLayoutInput in = input(k.w, k.h, k.n);
        const ApprovalStripLayout L = layout_approval_strip(in, m, measure());
        check_strip(L, in, m);
        CHECK((int)L.cards.size() == k.shown);
        CHECK(L.hidden == k.n - k.shown);
        if (k.card_w > 0) CHECK(L.cards[0].card.w == doctest::Approx(k.card_w));
        CHECK(L.cards[0].card.w >= m.min_card);
        // A review of three changed keys with warnings: the header, three
        // lines, the warnings and the buttons.
        CHECK(L.height == 2 * m.pad + m.gap + m.line + 3 * m.line + m.line + m.gap + m.button + m.gap);
        CHECK(L.cards[0].diff_shown == 3);
        CHECK_FALSE(L.cards[0].diff_folded);
        CHECK(!L.cards[0].warnings.empty());
        CHECK(!L.cards[0].time.empty());
        CHECK(!L.cards[0].diff.empty());
        CHECK(!L.label.empty());
        if (k.n >= 3) {
            // The third card is a destructive one: one line, no Diff button.
            CHECK(L.cards[2].diff.empty());
            CHECK(L.cards[2].body.h == m.line);
        }
    }
}

TEST_CASE("approval cards: what collapses as the window shrinks, and the minimums grow with the text scale") {
    const ApprovalLayoutMetrics m = ApprovalLayoutMetrics::for_text_scale(1.0f);
    // A short window: the strip keeps to its share; the compact lines past
    // the first fold into "+N", then the warnings row goes.
    {
        ApprovalLayoutInput in = input(1440, 300, 1);
        in.cards[0] = review_shape(6, true);
        const ApprovalStripLayout L = layout_approval_strip(in, m, measure());
        check_strip(L, in, m);
        CHECK(L.cards[0].diff_shown == 1);
        CHECK(L.cards[0].diff_folded);
        CHECK(L.cards[0].warnings.empty());
    }
    // More lines than the most: four rows, three lines and "+3".
    {
        ApprovalLayoutInput in = input(1440, 900, 1);
        in.cards[0] = review_shape(6, false);
        const ApprovalStripLayout L = layout_approval_strip(in, m, measure());
        CHECK(L.cards[0].diff_shown == 3);
        CHECK(L.cards[0].diff_folded);
        CHECK(L.cards[0].body.h == 4 * m.line);
    }
    // A narrow window: the label goes, one card shows and the rest fold.
    {
        const ApprovalLayoutInput in = input(300, 900, 3);
        const ApprovalStripLayout L = layout_approval_strip(in, m, measure());
        check_strip(L, in, m);
        CHECK(L.label.empty());
        CHECK(L.cards.size() == 1);
        CHECK(L.hidden == 2);
    }
    // At text scale 2 every minimum doubles: at 1440 one card fits beside the label.
    {
        const ApprovalLayoutMetrics m2 = ApprovalLayoutMetrics::for_text_scale(2.0f);
        CHECK(m2.min_card == 2 * m.min_card);
        CHECK(m2.line == 2 * m.line);
        const ApprovalLayoutInput in = input(1440, 900, 3);
        const ApprovalStripLayout L = layout_approval_strip(in, m2, measure(2.0f));
        check_strip(L, in, m2);
        CHECK(L.cards.size() == 1);
        CHECK(L.hidden == 2);
        CHECK(L.cards[0].card.w >= m2.min_card);
        CHECK(L.height == doctest::Approx(2 * layout_approval_strip(input(1440, 900, 3), m, measure()).height));
    }
    // A decided card: its outcome on one line and no buttons.
    {
        ApprovalLayoutInput in = input(1440, 900, 1);
        in.cards[0].decided = true;
        in.waiting = 0;
        const ApprovalStripLayout L = layout_approval_strip(in, m, measure());
        check_strip(L, in, m);
        CHECK(L.cards[0].approve.empty());
        CHECK(L.cards[0].deny.empty());
        CHECK(L.cards[0].diff.empty());
        CHECK(L.cards[0].body.h == m.line);
    }
    // No card, no strip.
    ApprovalLayoutInput none;
    none.width = 1440;
    none.window_h = 900;
    CHECK(layout_approval_strip(none, m, measure()).height == 0.0f);
}

TEST_CASE("approval cards: the strip's height changes while open only after the strips' delays") {
    ApprovalStripHeight h;
    CHECK(h.update(0.0, false, 100) == 0);
    CHECK(h.update(1.0, true, 100) == 100);   // opening takes it at once
    CHECK(h.update(1.1, true, 160) == 100);   // taller: after the open delay
    CHECK(h.update(1.5, true, 160) == 100);
    CHECK(h.update(1.61, true, 160) == 160);
    CHECK(h.update(2.0, true, 100) == 160);   // lower: after the close delay
    CHECK(h.update(3.9, true, 100) == 160);
    CHECK(h.update(4.01, true, 100) == 100);
    CHECK(h.update(4.1, false, 100) == 0);
}

TEST_CASE("approval cards: a run command's card states the change and the script, and its rules name the target") {
    CommandInvocation set;
    set.name = "run.set_param";
    set.args = {{"name", "lr"}, {"value", 0.001}};
    set.origin = CommandOrigin::mcp;
    CHECK(approval_what(set, nullptr) == "run.set_param \xC2\xB7 lr");
    CHECK(approval_run_text(set, "train.py") == "Set parameter lr to 0.001 in train.py. The change applies at the next step.");
    CommandInvocation act;
    act.name = "run.action";
    act.args = {{"name", "save"}};
    CHECK(approval_run_text(act, "train.py") == "Run action save in train.py at the next step.");
    CommandInvocation p;
    p.name = "run.pause";
    p.args = {{"op", "pause"}};
    CHECK(approval_run_text(p, "train.py") == "Pause train.py at the next step.");
    p.args = {{"op", "resume"}};
    CHECK(approval_run_text(p, "") == "Resume the attached script.");
    p.args = {{"op", "step"}, {"count", 5}};
    CHECK(approval_run_text(p, "train.py") == "Run 5 steps of train.py, then pause.");
    CHECK(approval_run_text(act, "x").size() > 0);
    CHECK(approval_always_label("run.set_param:lr") == "Always allow changes of lr");
    CHECK(approval_session_label("run.set_param:lr") == "Allow changes of lr for this session");
    CHECK(approval_always_label("run.action:save") == "Always allow action save");
    CHECK(approval_always_label("run.pause") == "Always allow run.pause");
    CHECK(approval_session_label("panel.patch") == "Allow always for this session");
}
