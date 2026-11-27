// session_preview.cpp: spec-edit review and stacked preview command handlers
// of Session::Impl (session_internal.h).
#include "session_internal.h"

#include <algorithm>

namespace caliper_builder {

namespace {

using nlohmann::json;
using namespace session_detail;
using namespace session_internal;

}  // namespace

// ---- review of spec edits --------------------------------------------------

bool Session::Impl::review_for(CommandOrigin o) const { return review_on[static_cast<std::size_t>(o)].load(); }

// The lines a change removes and adds, in order, without the line numbers
// and the context of a unified diff, so that the same change made on a
// spec that changed elsewhere compares equal.
std::string Session::Impl::change_lines(const std::string& before, const std::string& after) {
    const std::string diff = unified_line_diff(before, after, "a", "b", 0);
    std::string out;
    std::size_t start = 0;
    int line = 0;
    while (start < diff.size()) {
        std::size_t end = diff.find('\n', start);
        if (end == std::string::npos) end = diff.size();
        const std::string_view l(diff.data() + start, end - start);
        if (++line > 2 && l.rfind("@@", 0) != 0) {
            out.append(l);
            out += '\n';
        }
        start = end + 1;
    }
    return out;
}

// The review of an invocation, or null when it needs none: review is off
// for its origin, it is not an edit of the open spec, it is a dry run or
// names a preview, or it would fail or change nothing. A review is kept
// for the handler, which runs the edit only as it was shown.
json Session::Impl::review_of(const CommandInvocation& inv) {
    if (!is_spec_edit_command(inv.name) || !review_for(inv.origin) || !inv.preview.empty() ||
        bool_arg(inv.args, "dry_run", false))
        return nullptr;
    Reviewed kept;
    json review;
    {
        std::lock_guard<std::mutex> lock(doc_mutex);
        if (!doc) return nullptr;
        json args = inv.args;
        SpecEdit e = edit_spec(doc->spec, inv.name, args);
        if (e.error || e.spec == doc->spec) return nullptr;
        const std::string before = spec::to_toml(doc->spec);
        const std::string after = spec::to_toml(e.spec);
        const std::string label = spec_label();
        review = {{"diff", unified_line_diff(before, after, "a/" + label, "b/" + label)},
                  {"warnings", warnings_json(e.warnings)}};
        kept = Reviewed{invocation_key(inv), doc->id, change_lines(before, after)};
    }
    std::lock_guard<std::mutex> lock(review_mutex);
    reviewed.push_back(std::move(kept));
    while (reviewed.size() > 256) reviewed.pop_front();
    last_review = {invocation_key(inv), review};
    return review;
}

// What the registry keeps with a parked invocation: the review the policy
// just computed for it.
json Session::Impl::parked_review(const CommandInvocation& inv) {
    std::lock_guard<std::mutex> lock(review_mutex);
    if (last_review.first != invocation_key(inv)) return nullptr;
    json out = std::move(last_review.second);
    last_review = {};
    return out;
}

std::optional<Session::Impl::Reviewed> Session::Impl::take_reviewed(const CommandInvocation& inv) {
    const std::string key = invocation_key(inv);
    std::lock_guard<std::mutex> lock(review_mutex);
    const auto it = std::find_if(reviewed.begin(), reviewed.end(), [&](const Reviewed& r) { return r.key == key; });
    if (it == reviewed.end()) return std::nullopt;
    Reviewed out = std::move(*it);
    reviewed.erase(it);
    return out;
}

CommandOutput Session::Impl::review_status(const CommandContext& ctx) {
    const CommandOrigin o = ctx.invocation.origin;
    return CommandOutput::ok(
        {{"origin", command_origin_name(o)}, {"spec_edits", review_for(o)}, {"destructive", !user_origin(o)}});
}

// ---- stacked previews (all under doc_mutex) --------------------------------

std::chrono::steady_clock::time_point Session::Impl::preview_now() const {
    return options.preview.now ? options.preview.now() : std::chrono::steady_clock::now();
}

// Drops the previews left unused for longer than the idle limit.
void Session::Impl::sweep_previews() {
    const auto now = preview_now();
    for (auto it = previews.begin(); it != previews.end();) {
        if (now - it->second.used > options.preview.idle) {
            expired.push_back(it->first);
            while (expired.size() > 64) expired.pop_front();
            it = previews.erase(it);
        } else {
            ++it;
        }
    }
}

// Finds a preview, after dropping the expired ones, and marks it used.
Error Session::Impl::find_preview(const std::string& id, const std::string& where, Preview** out) {
    sweep_previews();
    const auto it = previews.find(id);
    if (it == previews.end()) {
        if (std::find(expired.begin(), expired.end(), id) != expired.end()) {
            const auto minutes = std::chrono::duration_cast<std::chrono::minutes>(options.preview.idle).count();
            return Error("preview.expired",
                         "The preview " + id + " was dropped after " + std::to_string(minutes) +
                             " minutes without use; preview.begin starts a new one.",
                         where);
        }
        return Error("preview.not_found", "No preview has the id " + id + "; preview.begin starts one.", where);
    }
    it->second.used = preview_now();
    *out = &it->second;
    return {};
}

// Makes the preview's shadow the open spec with its stack applied again,
// when the open spec changed since the shadow was computed.
Error Session::Impl::refresh_preview(Preview& p) {
    if (!doc || doc->id != p.doc)
        return Error("history.conflict", "The preview " + p.id +
                                             " was made on a spec that is no longer open; discard it and begin a "
                                             "new one on the open spec.");
    if (doc->spec == p.base) return {};
    spec::Spec s = doc->spec;
    for (const CommandInvocation& inv : p.stack) {
        json args = inv.args;
        SpecEdit e = edit_spec(s, inv.name, args);
        if (e.error)
            return Error("history.conflict",
                         "The open spec changed since the preview's edits were made, and its " + inv.name +
                             " no longer applies: " + e.error.message + " (" + e.error.code + ")",
                         e.error.where);
        s = std::move(e.spec);
    }
    p.base = doc->spec;
    p.shadow = std::move(s);
    return {};
}

Error Session::Impl::use_preview(const std::string& id, const std::string& where, Preview** out) {
    if (Error e = find_preview(id, where, out)) return e;
    return refresh_preview(**out);
}

// The rule of a scoped preview: it takes new sources and panels, and
// changes only its target, the sources a target panel binds (following
// `from`) and the items it added itself.
Error Session::Impl::check_scope(const Preview& p, const std::string& name, const json& args) const {
    if (p.panel.empty() && p.source.empty()) return {};
    const std::string target = p.panel.empty() ? "source " + p.source : "panel " + p.panel;
    if (name == "spec.set_text" || name == "layout.set" || name == "layout.lock")
        return Error("preview.out_of_scope",
                     "A preview of the " + target + " cannot replace the whole spec or the whole layout; change "
                                                    "the target with the other edits.",
                     "name");
    if (name == "source.add" || name == "panel.create") return {};
    std::set<std::string> panels;
    std::set<std::string> sources;
    if (!p.panel.empty()) panels.insert(p.panel);
    if (!p.source.empty()) sources.insert(p.source);
    for (const CommandInvocation& inv : p.stack) {
        if (inv.name == "panel.create") panels.insert(str_arg(inv.args, "id"));
        if (inv.name == "source.add") sources.insert(str_arg(inv.args, "id"));
    }
    if (const spec::Panel* panel = p.panel.empty() ? nullptr : p.shadow.find_panel(p.panel)) {
        std::vector<std::string> todo;
        for (const auto& slot : panel->bind)
            for (const std::string& b : slot.second.bindings) todo.push_back(b.substr(0, b.find('.')));
        while (!todo.empty() && sources.size() < 4096) {
            const std::string id = todo.back();
            todo.pop_back();
            if (!sources.insert(id).second) continue;
            const spec::Source* src = p.shadow.find_source(id);
            if (src && src->from) todo.push_back(*src->from);
        }
    }
    const bool panel_edit = name == "panel.patch" || name == "panel.remove" || name == "layout.place";
    const std::string id = str_arg(args, name == "layout.place" ? "panel" : "id");
    if (panel_edit ? panels.count(id) : sources.count(id)) return {};
    return Error("preview.out_of_scope",
                 "This preview is about the " + target + ", so it changes only that target, the sources it "
                 "binds and what the preview added; " + id + " is outside it.",
                 name == "layout.place" ? "args.panel" : "args.id");
}

// An edit that names a preview: it applies to the shadow and reports no
// effect, so the registry neither journals it nor files a step.
CommandOutput Session::Impl::preview_edit(const CommandContext& ctx, bool dry) {
    auto quiet = [](CommandOutput out) {
        out.no_effect = true;
        return out;
    };
    if (!doc) return quiet(no_open_spec());
    Preview* p = nullptr;
    if (Error e = use_preview(ctx.invocation.preview, "preview", &p)) return quiet(CommandOutput::fail(e));
    const std::string& name = ctx.invocation.name;
    const json& args = ctx.invocation.args;
    if (Error e = check_scope(*p, name, args)) return quiet(CommandOutput::fail(e));
    json completed = args;
    SpecEdit e = edit_spec(p->shadow, name, completed);
    if (e.error) return quiet(CommandOutput::fail(e.error));
    const std::string label = spec_label();
    json result = std::move(e.result);
    result["warnings"] = warnings_json(e.warnings);
    result["diff"] = unified_line_diff(spec::to_toml(p->shadow), spec::to_toml(e.spec), "a/" + label, "b/" + label);
    result["applied"] = !dry;
    if (dry || e.spec == p->shadow) return CommandOutput::unchanged(std::move(result));
    CommandInvocation kept;
    kept.id = ctx.invocation.id;
    kept.name = name;
    kept.args = args;
    kept.args.erase("dry_run");
    kept.origin = ctx.invocation.origin;
    kept.merge_key = ctx.invocation.merge_key;
    const std::size_t bytes = kept.args.dump(-1, ' ', false, json::error_handler_t::replace).size() + name.size();
    // A run with one merge key sets the same thing each time, so its
    // newest edit takes the place of the one before.
    const bool merges = !kept.merge_key.empty() && !p->stack.empty() && p->stack.back().merge_key == kept.merge_key &&
                        p->stack.back().name == name;
    const std::size_t replaced = merges ? p->stack.back().args.dump(-1, ' ', false, json::error_handler_t::replace).size() +
                                              name.size()
                                        : 0;
    if (!merges && p->stack.size() >= options.preview.max_edits)
        return quiet(fail("preview.too_many_edits",
                          "The preview holds " + std::to_string(p->stack.size()) +
                              " edits, the most it takes; it can be committed or discarded, or a new one begun.",
                          "preview"));
    if (p->bytes - replaced + bytes > options.preview.max_bytes)
        return quiet(fail("preview.too_large",
                          "The preview's edits would hold more than " +
                              std::to_string(options.preview.max_bytes >> 10) + " KiB of arguments.",
                          "preview"));
    if (merges) {
        p->bytes -= replaced;
        p->stack.back() = std::move(kept);
    } else {
        p->stack.push_back(std::move(kept));
    }
    p->bytes += bytes;
    p->shadow = std::move(e.spec);
    return CommandOutput::unchanged(std::move(result));
}

json Session::Impl::scope_json(const std::string& id) { return id.empty() ? json(nullptr) : json(id); }

CommandOutput Session::Impl::preview_begin(const CommandContext& ctx) {
    std::lock_guard<std::mutex> lock(doc_mutex);
    if (!doc) return no_open_spec();
    sweep_previews();
    const json& a = ctx.invocation.args;
    std::string id = str_arg(a, "id");
    const std::string panel = str_arg(a, "panel");
    const std::string source = str_arg(a, "source");
    if (!panel.empty() && !source.empty())
        return fail("preview.bad_scope", "A preview is about a panel or a source, not both.", "args.source");
    if (!panel.empty() && !doc->spec.find_panel(panel))
        return fail("panel.not_found", "The open spec has no panel " + panel + "; spec.get lists them.", "args.panel");
    if (!source.empty() && !doc->spec.find_source(source))
        return fail("source.not_found", "The open spec has no source " + source + "; spec.get lists them.",
                    "args.source");
    if (!id.empty() && previews.count(id))
        return fail("preview.exists", "A preview " + id + " is open already; preview.discard ends it.", "args.id");
    if (previews.size() >= options.preview.max_previews)
        return fail("preview.too_many", std::to_string(previews.size()) +
                                            " previews are open, the most a workspace keeps; commit or discard "
                                            "one first.");
    while (id.empty() || previews.count(id)) id = "preview-" + std::to_string(next_preview++);
    expired.erase(std::remove(expired.begin(), expired.end(), id), expired.end());
    Preview p;
    p.id = id;
    p.doc = doc->id;
    p.base = doc->spec;
    p.shadow = doc->spec;
    p.panel = panel;
    p.source = source;
    p.used = preview_now();
    previews.emplace(id, std::move(p));
    return CommandOutput::ok({{"id", id}, {"panel", scope_json(panel)}, {"source", scope_json(source)}});
}

CommandOutput Session::Impl::preview_diff(const CommandContext& ctx) {
    std::lock_guard<std::mutex> lock(doc_mutex);
    if (!doc) return no_open_spec();
    Preview* p = nullptr;
    if (Error e = use_preview(str_arg(ctx.invocation.args, "id"), "args.id", &p)) return CommandOutput::fail(e);
    const std::string label = spec_label();
    return CommandOutput::ok(
        {{"id", p->id},
         {"edits", p->stack.size()},
         {"diff", unified_line_diff(spec::to_toml(doc->spec), spec::to_toml(p->shadow), "a/" + label, "b/" + label)},
         {"warnings", warnings_json(spec::validate(p->shadow, resolver()).diagnostics)},
         {"panel", scope_json(p->panel)},
         {"source", scope_json(p->source)}});
}

// Accept: the person's act, so only ui and builder commit. The edits run
// as the registry's follow-ups, after being applied again to the open
// spec here, so a stack that no longer applies changes nothing.
CommandOutput Session::Impl::preview_commit(const CommandContext& ctx) {
    if (!user_origin(ctx.invocation.origin))
        return fail("preview.not_allowed",
                    "Only the person commits a preview, with Accept in Caliper; it stays open for them to accept "
                    "or reject.");
    std::lock_guard<std::mutex> lock(doc_mutex);
    if (!doc) return no_open_spec();
    Preview* p = nullptr;
    if (Error e = use_preview(str_arg(ctx.invocation.args, "id"), "args.id", &p)) return CommandOutput::fail(e);
    const std::string label = spec_label();
    json result{
        {"id", p->id},
        {"edits", p->stack.size()},
        {"diff", unified_line_diff(spec::to_toml(doc->spec), spec::to_toml(p->shadow), "a/" + label, "b/" + label)},
        {"warnings", warnings_json(spec::validate(p->shadow, resolver()).diagnostics)}};
    std::vector<CommandInvocation> edits = std::move(p->stack);
    const std::string group = p->id;
    previews.erase(group);
    if (edits.empty()) return CommandOutput::unchanged(std::move(result));
    for (CommandInvocation& inv : edits) {
        inv.id.clear();
        inv.group = group;
        inv.merge_key.clear();
        inv.preview.clear();
    }
    CommandOutput out = CommandOutput::ok(std::move(result));
    out.follow_ups = std::move(edits);
    return out;
}

CommandOutput Session::Impl::preview_discard(const CommandContext& ctx) {
    std::lock_guard<std::mutex> lock(doc_mutex);
    Preview* p = nullptr;
    if (Error e = find_preview(str_arg(ctx.invocation.args, "id"), "args.id", &p)) return CommandOutput::fail(e);
    const std::string id = p->id;
    json result{{"id", id}, {"edits", p->stack.size()}};
    previews.erase(id);
    return CommandOutput::ok(std::move(result));
}

void Session::set_review(CommandOrigin origin, bool on) {
    impl_->review_on[static_cast<std::size_t>(origin)].store(on);
}

ReviewSettings Session::review() const {
    ReviewSettings out;
    for (CommandOrigin o : {CommandOrigin::ui, CommandOrigin::builder, CommandOrigin::agent, CommandOrigin::mcp,
                            CommandOrigin::api})
        out.set(o, impl_->review_for(o));
    return out;
}

}  // namespace caliper_builder
