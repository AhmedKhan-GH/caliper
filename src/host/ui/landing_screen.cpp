// ============================================================================
// The landing screen (see landing_screen.h for the layout). A workstation
// home in the density of a Bloomberg terminal or Foundry: a command bar, four
// docked panes of dense tables, a status bar. Everything is drawn with the
// Sternbild primitives (sternbild_theme.h); this file only arranges them.
//
// Patterns, named where they are applied below:
//   status bar (VS Code / JetBrains)       backend, GPU, memory, applet count
//   command bar of key buttons             Bloomberg command line and function
//                                          keys: each button holds its key
//   dense tables                           Bloomberg, W&B run tables: tabular
//                                          figures, ↑↓ + Enter, one accent
//   docked panes with headers and keys     Bloomberg Launchpad, VS Code panel
//                                          focus: ⌘1–⌘4, bronze when focused
//   numbered menu of ways in               Bloomberg menus, VS Code Welcome
//   log with relative times                VS Code Output, Datadog logs
//   overview first, details on demand      Shneiderman: Instruments, Inspector
//   Fitts / Hick                           one primary action, full-row hits
//   pre-attentive colour                   bronze marks focus and selection
//   error prevention and honesty           disabled actions say why
// ============================================================================

#include "ui/landing_screen.h"

#include "ui/sternbild_fonts.h"
#include "ui/sternbild_theme.h"
#include "applet_loader.h"
#include "device_query.h"
#include "host_services.h"
#include "job_system.h"

#include <imgui.h>
#include <imgui_internal.h>   // DockBuilder*, SetKeyOwner, GetActiveID

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#if defined(__APPLE__)
  #include <sys/sysctl.h>
#elif defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <windows.h>
#elif defined(__linux__)
  #include <sys/sysinfo.h>
#endif

namespace {

using caliper_host::AppletEntry;
using caliper_host::AppletLoader;
using caliper_host::AppletStatus;
namespace col = sternbild::col;
namespace metric = sternbild::metric;
namespace font_size = sternbild::font_size;

// ---- Key caps ---------------------------------------------------------------
// One table, so a face without a glyph is a one-line swap. On macOS the fonts
// module merges Apple Symbols behind the UI and mono faces, so ⌘ and ⏎ render;
// elsewhere they would be missing-glyph boxes, so the ASCII form is compiled
// in. Even on macOS a symbol is only used when the mono face (the face
// key buttons draw keys in) really carries it, in case Apple Symbols was absent.
// Strings are UTF-8 byte escapes so no compiler code page can mangle them.
struct KeyCap {
    const char* symbol;   // UTF-8
    const char* ascii;
    ImWchar probe;        // codepoint of `symbol` to test; 0 = plain ASCII
};
constexpr KeyCap kKeyUp      = {"\xe2\x86\x91", "UP", 0x2191};                 // ↑
constexpr KeyCap kKeyDown    = {"\xe2\x86\x93", "DN", 0x2193};                 // ↓
#ifdef __APPLE__
constexpr KeyCap kKeyEnter   = {"\xe2\x8f\x8e", "ENTER", 0x23ce};              // ⏎
constexpr KeyCap kKeyCommand = {"\xe2\x8c\x98K", "Cmd K", 0x2318};             // ⌘K
constexpr KeyCap kKeyPane[4] = {{"\xe2\x8c\x98" "1", "Cmd 1", 0x2318},         // ⌘1
                                {"\xe2\x8c\x98" "2", "Cmd 2", 0x2318},
                                {"\xe2\x8c\x98" "3", "Cmd 3", 0x2318},
                                {"\xe2\x8c\x98" "4", "Cmd 4", 0x2318}};
constexpr KeyCap kKeyRestart = {"\xe2\x8c\x98R", "Cmd R", 0x2318};             // ⌘R
#else
constexpr KeyCap kKeyEnter   = {"ENTER", "ENTER", 0};
constexpr KeyCap kKeyCommand = {"Ctrl K", "Ctrl K", 0};
constexpr KeyCap kKeyRestart = {"Ctrl R", "Ctrl R", 0};
constexpr KeyCap kKeyPane[4] = {{"Ctrl 1", "Ctrl 1", 0}, {"Ctrl 2", "Ctrl 2", 0},
                                {"Ctrl 3", "Ctrl 3", 0}, {"Ctrl 4", "Ctrl 4", 0}};
#endif
constexpr KeyCap kKeyEscape  = {"esc", "esc", 0};
// The command-line prompt (bronze, like the design's), not a key, but kept
// here with the other glyphs.
constexpr const char* kPrompt = "\xe2\x80\xba";                               // ›

const char* key(const KeyCap& k) {
    if (k.probe == 0) return k.symbol;
    ImFont* face = sternbild::fonts().mono ? sternbild::fonts().mono
                                           : ImGui::GetFont();
    return face && face->IsGlyphInFont(k.probe) ? k.symbol : k.ascii;
}

// ---- Small helpers ----------------------------------------------------------
// std::string-backed InputText: the resize callback from imgui_stdlib.cpp,
// which is not compiled into the imgui target.
int grow_string(ImGuiInputTextCallbackData* data) {
    if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
        auto* s = static_cast<std::string*>(data->UserData);
        s->resize(static_cast<size_t>(data->BufTextLen));
        data->Buf = s->data();
    }
    return 0;
}

std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

std::string upper(std::string s) {
    for (char& c : s) c = (char)std::toupper((unsigned char)c);
    return s;
}

// Case-insensitive compare that orders digit runs by value, so version
// 0.10.0 sorts after 0.9.0 and "Scope 2" before "Scope 10".
int natural_compare(const std::string& a, const std::string& b) {
    auto digit = [](char c) { return std::isdigit((unsigned char)c) != 0; };
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (digit(a[i]) && digit(b[j])) {
            size_t ie = i, je = j;
            while (ie < a.size() && digit(a[ie])) ++ie;
            while (je < b.size() && digit(b[je])) ++je;
            while (i + 1 < ie && a[i] == '0') ++i;   // value, not spelling
            while (j + 1 < je && b[j] == '0') ++j;
            if (ie - i != je - j) return ie - i < je - j ? -1 : 1;
            if (int c = a.compare(i, ie - i, b, j, je - j)) return c < 0 ? -1 : 1;
            i = ie;
            j = je;
            continue;
        }
        const int ca = std::tolower((unsigned char)a[i]);
        const int cb = std::tolower((unsigned char)b[j]);
        if (ca != cb) return ca < cb ? -1 : 1;
        ++i;
        ++j;
    }
    if (i < a.size()) return 1;
    if (j < b.size()) return -1;
    return 0;
}

// Instruments columns; the value is the TableSetupColumn user id.
enum Column { kName = 0, kId = 1, kVersion = 2, kTag = 3, kStatus = 4 };

int status_rank(AppletStatus s) {
    switch (s) {
        case AppletStatus::Active:      return 0;
        case AppletStatus::Ready:       return 1;
        case AppletStatus::Refused:     return 2;
        case AppletStatus::Failed:      return 3;
        case AppletStatus::Quarantined: return 4;
    }
    return 5;
}

const char* status_word(AppletStatus s) {
    switch (s) {
        case AppletStatus::Ready:       return "READY";
        case AppletStatus::Active:      return "ACTIVE";
        case AppletStatus::Refused:     return "REFUSED";
        case AppletStatus::Failed:      return "FAILED";
        case AppletStatus::Quarantined: return "QUARANTINED";
    }
    return "UNKNOWN";
}

// Pre-attentive colour: the running instrument is the one bronze status;
// launchable is secondary; anything that cannot launch recedes to faint.
// No red, no green (SPEC §9).
ImU32 status_colour(AppletStatus s) {
    switch (s) {
        case AppletStatus::Ready:  return col::secondary;
        case AppletStatus::Active: return col::bronze;
        default:                   return col::faint;
    }
}

bool launchable(const AppletEntry& e) { return e.status == AppletStatus::Ready; }

bool any_popup_open() {
    return ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId |
                                           ImGuiPopupFlags_AnyPopupLevel);
}

// Text in one face, size and colour at the cell cursor. The theme's cell
// padding already centres one 15 px line in the 22 px row, and the header
// labels sit at the same point, so cells and heads share a top line.
// Optionally flush right, so figures line up on their last digit.
void cell_text(const char* text, ImFont* face, float size, ImU32 colour,
               bool right_align = false) {
    ImGui::PushFont(face, size);   // null face keeps the current font
    if (right_align) {
        const float w = ImGui::CalcTextSize(text).x;
        const float avail = ImGui::GetContentRegionAvail().x;
        if (avail > w) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + avail - w);
    }
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

// Same, wrapped at the cell or window edge (paths, reasons, prose).
void cell_wrapped(const char* text, ImFont* face, float size, ImU32 colour) {
    ImGui::PushFont(face, size);
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::TextWrapped("%s", text);
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

// A faint single line: empty states and placeholders.
void faint_line(const char* text) {
    ImGui::PushStyleColor(ImGuiCol_Text, col::faint);
    ImGui::TextWrapped("%s", text);
    ImGui::PopStyleColor();
}

// Distance from the line top to the baseline for a face at a size, so runs
// at different sizes on one row can share a baseline.
float ascent(ImFont* face, float size) {
    ImGui::PushFont(face, size);
    const float a = ImGui::GetFontBaked()->Ascent;
    ImGui::PopFont();
    return a;
}

// An empty table keeps its header and says so in a single body row; a
// header with no body would put the bottomrule right on the midrule.
constexpr ImVec2 kEmptyTableSize(0.0f, metric::row * 2.0f + 2.0f);

void empty_row(const char* text) {
    ImGui::TableNextRow(ImGuiTableRowFlags_None, metric::row);
    ImGui::TableNextColumn();
    ImGui::PushStyleColor(ImGuiCol_Text, col::faint);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}

// Total physical memory in bytes, 0 when the platform will not say.
uint64_t physical_memory_bytes() {
#if defined(__APPLE__)
    uint64_t bytes = 0;
    size_t len = sizeof bytes;
    if (sysctlbyname("hw.memsize", &bytes, &len, nullptr, 0) == 0) return bytes;
    return 0;
#elif defined(_WIN32)
    MEMORYSTATUSEX st{};
    st.dwLength = sizeof st;
    if (GlobalMemoryStatusEx(&st)) return (uint64_t)st.ullTotalPhys;
    return 0;
#elif defined(__linux__)
    struct sysinfo si {};
    if (sysinfo(&si) == 0) return (uint64_t)si.totalram * si.mem_unit;
    return 0;
#else
    return 0;
#endif
}

std::string gigabytes(uint64_t bytes) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f GB",
                  (double)bytes / (1024.0 * 1024.0 * 1024.0));
    return buf;
}


int running_jobs() {
    int n = 0;
    for (const auto& v : caliper_host::host_job_system().views())
        n += v.running ? 1 : 0;
    return n;
}

constexpr size_t kLogCap = 500;   // Activity keeps the latest lines only

// Pane titles double as dock-by-title keys and imgui.ini window names.
constexpr const char* kProjects    = "Projects";
constexpr const char* kInstruments = "Instruments";
constexpr const char* kInspector   = "Inspector";
constexpr const char* kActivity    = "Activity";
// ⌘1–⌘4 in this order (kKeyPane).
constexpr const char* kPanes[4] = {kProjects, kInstruments, kInspector, kActivity};

// Pane window. The style's WindowPadding stays: pane_header() pins its strip
// to the window top regardless and starts the content one padding below the
// strip, so a table's toprule never sits on the strip's rule. The pane never
// scrolls as a whole (its tables scroll inside), so nothing slides under the
// header.
bool begin_pane(const char* title, ImGuiWindowFlags extra = 0) {
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse |
                             ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoScrollWithMouse;
    return ImGui::Begin(title, nullptr, flags | extra);
}

// Pane header (Bloomberg Launchpad, VS Code panels): title, subtitle and the
// pane's own shortcut on a strip; bronze while the pane has focus. Must be
// the pane's first call. Content is then held below the strip whatever
// cursor the header leaves.
void pane_header(const char* title, const char* subtitle, const KeyCap& shortcut) {
    const bool focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
    sternbild::pane_header(title, subtitle, key(shortcut), focused);
    const float content_y = ImGui::GetWindowPos().y + metric::pane_header;
    const ImVec2 cursor = ImGui::GetCursorScreenPos();
    if (cursor.y < content_y) ImGui::SetCursorScreenPos(ImVec2(cursor.x, content_y));
}

// Activity WHEN column: relative, coarse, recomputed every frame, so it never
// needs a clock and never shows a time zone (VS Code Output, Datadog logs).
std::string ago(double seconds) {
    char buf[32];
    if (seconds < 5.0) return "just now";
    if (seconds < 60.0)
        std::snprintf(buf, sizeof buf, "%d s ago", (int)seconds);
    else if (seconds < 3600.0)
        std::snprintf(buf, sizeof buf, "%d min ago", (int)(seconds / 60.0));
    else if (seconds < 86400.0)
        std::snprintf(buf, sizeof buf, "%d h ago", (int)(seconds / 3600.0));
    else
        std::snprintf(buf, sizeof buf, "%d d ago", (int)(seconds / 86400.0));
    return buf;
}

double now_epoch() {
    using clock = std::chrono::system_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

}  // namespace

// ============================================================================
// Frame
// ============================================================================

void LandingScreen::draw(const Context& ctx) {
    // The landing (re)appears on the first frame and after every Home. Its
    // panes "just appeared", so ImGui would hand focus to whichever was begun
    // last; the overview table takes it instead, so ↑↓ and Enter work without
    // a click. Asked on two frames: a pane docked on the first frame can take
    // focus back on the next.
    const int frame = ImGui::GetFrameCount();
    if (last_frame_ < 0 || frame - last_frame_ > 1) focus_instruments_ = 2;
    last_frame_ = frame;

    if (selected_ >= (ctx.loader ? ctx.loader->count() : 0)) selected_ = -1;

    ImGui::PushFont(sternbild::fonts().ui, font_size::ui);

    // Chrome first: both bars are viewport side bars, so they are submitted
    // before the dock space that fills the work area they leave.
    draw_command_bar();
    draw_status_bar(ctx);

    // Overview first, details on demand, keyboard first: a row is always
    // selected while any is visible, so the Inspector has something to show
    // and Enter has something to launch. With nothing selected yet, the first
    // launchable row takes the selection; when typing hides the selected row,
    // the top match takes it, as in the VS Code palette, so the bronze row is
    // always what Enter will launch.
    if (ctx.loader) rebuild_rows(*ctx.loader);
    else rows_.clear();
    const bool visible =
        std::find(rows_.begin(), rows_.end(), selected_) != rows_.end();
    if (!visible && !rows_.empty()) {
        int pick = rows_.front();
        if (selected_ < 0) {
            for (const int i : rows_) {
                if (launchable(ctx.loader->at(i))) {
                    pick = i;
                    break;
                }
            }
        }
        selected_ = pick;
        scroll_to_selected_ = true;
    }

    // ⌘1–⌘4 focus a pane (VS Code panel focus, Bloomberg panel keys); the
    // same Cmd/Ctrl chord logic as ⌘K. Applied after the frame, like the
    // other focus requests, so a pane appearing this frame cannot take it.
    static const ImGuiKey pane_keys[4] = {ImGuiKey_1, ImGuiKey_2, ImGuiKey_3,
                                          ImGuiKey_4};
    for (int i = 0; i < 4; ++i)
        if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | pane_keys[i]))
            focus_pane_ = kPanes[i];
    // ⌘R: the Inspector's Restart, while the selected row offers it.
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_R) && ctx.loader &&
        ctx.loader->restartable(selected_))
        pending_restart_ = selected_;

    draw_dock_space();
    if (begin_pane(kProjects)) draw_projects();
    ImGui::End();
    // The table owns the arrows and Enter (SetKeyOwner in draw_instruments);
    // NoNavInputs keeps ImGui's nav cursor from also moving in this pane.
    if (begin_pane(kInstruments, ImGuiWindowFlags_NoNavInputs)) draw_instruments(ctx);
    ImGui::End();
    if (begin_pane(kInspector)) draw_inspector(ctx);
    ImGui::End();
    if (begin_pane(kActivity)) draw_activity();
    ImGui::End();
    select_step_ = 0;   // spent by draw_instruments, or moot with no rows

    ImGui::PopFont();

    // Enter launches the selected row, or the top row when nothing is
    // selected; a row that cannot launch does nothing (the Inspector already
    // says why). From the command line, focus then moves to the table with
    // the filter kept, so ↑↓ can pick another row without the mouse.
    if (enter_requested_) {
        const bool visible =
            std::find(rows_.begin(), rows_.end(), selected_) != rows_.end();
        const int target = visible ? selected_
                                   : (rows_.empty() ? -1 : rows_.front());
        if (target >= 0 && launchable(ctx.loader->at(target))) {
            selected_ = target;
            pending_launch_ = target;
        } else if (enter_from_command_) {
            focus_instruments_ = std::max(focus_instruments_, 1);
        }
        enter_requested_ = enter_from_command_ = false;
    }

    // Launch after every window of this frame has ended, so the applet's
    // initialize() never runs inside a half-built landing frame. A restart
    // is a launch too (it ends in initialize()), so it waits the same way.
    if (pending_launch_ >= 0) {
        const int i = pending_launch_;
        pending_launch_ = -1;
        if (ctx.loader && ctx.launch && i < ctx.loader->count() &&
            launchable(ctx.loader->at(i)))
            ctx.launch(i);
    }
    if (pending_restart_ >= 0) {
        const int i = pending_restart_;
        pending_restart_ = -1;
        if (ctx.loader && ctx.restart && ctx.loader->restartable(i))
            ctx.restart(i);
    }

    if (focus_pane_) {
        ImGui::SetWindowFocus(focus_pane_);
        focus_pane_ = nullptr;
        focus_instruments_ = 0;   // an explicit ⌘n wins over the default
    } else if (focus_instruments_ > 0) {
        ImGui::SetWindowFocus(kInstruments);
        --focus_instruments_;
    }
}

void LandingScreen::log(std::string event, std::string object,
                        std::string detail) {
    log_.push_back({now_epoch(), std::move(event), std::move(object),
                    std::move(detail)});
    if (log_.size() > kLogCap)
        log_.erase(log_.begin(),
                   log_.begin() + (std::ptrdiff_t)(log_.size() - kLogCap));
}

// ============================================================================
// Command bar
// ============================================================================

// Command bar of key buttons (Bloomberg command line and function keys): the
// line filters the Instruments table as you type and Enter launches, and each
// action that drives it is a button holding its key, so nothing has to be
// remembered (recognition over recall) and no label repeats a button. The
// bar holds the actions; the status bar holds only facts about the machine.
void LandingScreen::draw_command_bar() {
    ImFont* mono = sternbild::fonts().mono;

    // ⌘K on macOS, Ctrl+K elsewhere: ImGui swaps Cmd and Ctrl at the event
    // level when io.ConfigMacOSXBehaviors is set (the Apple default), so
    // ImGuiMod_Ctrl is the Command key there and one chord serves both.
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_K))
        focus_command_ = true;

    sternbild::command_bar_begin();
    sternbild::masthead();   // ends on the same line, metric::gap after

    // The prompt: a mono › in bronze, the accent marking where keys go.
    ImGui::PushFont(mono, font_size::mono);
    ImGui::AlignTextToFramePadding();
    ImGui::PushStyleColor(ImGuiCol_Text, col::bronze);
    ImGui::TextUnformatted(kPrompt);
    ImGui::PopStyleColor();
    ImGui::SameLine(0.0f, metric::pad * 0.75f);

    // The field is a ruled fill-in line (SPEC §6.4): no box, no fill, only a
    // 1 px rule under the text that turns parchment while editing.
    const ImGuiID command_id = ImGui::GetID("##command");
    const bool was_active = ImGui::GetActiveID() == command_id;
    const bool esc = ImGui::IsKeyPressed(ImGuiKey_Escape, false);
    if (focus_command_) {
        ImGui::SetKeyboardFocusHere();
        focus_command_ = false;
    }
    ImGui::PushStyleColor(ImGuiCol_FrameBg, IM_COL32(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, IM_COL32(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, IM_COL32(0, 0, 0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,
                        ImVec2(0.0f, ImGui::GetStyle().FramePadding.y));
    ImGui::SetNextItemWidth(320.0f);
    const bool entered = ImGui::InputTextWithHint(
        "##command", "Filter instruments",
        command_.data(), command_.capacity() + 1,
        ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackResize,
        grow_string, &command_);
    command_active_ = ImGui::IsItemActive();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(3);
    {
        const ImVec2 mn = ImGui::GetItemRectMin();
        const ImVec2 mx = ImGui::GetItemRectMax();
        sternbild::rule_h(ImGui::GetWindowDrawList(), std::floor(mn.x),
                          std::floor(mx.x), std::floor(mx.y) - 1.0f,
                          command_active_ ? col::parchment : col::rule);
    }
    ImGui::PopFont();

    if (entered) {
        enter_requested_ = true;
        enter_from_command_ = true;
    }
    // Esc cancels the line in one press: clear the filter and leave the field
    // for the table (ImGui's own Esc only reverts the edit). With the field
    // idle, Esc still clears a standing filter, so one key always returns to
    // the full list.
    if (esc && was_active) {
        command_.clear();
        command_active_ = false;
        focus_instruments_ = std::max(focus_instruments_, 1);
    } else if (esc && !command_.empty() && !ImGui::GetIO().WantTextInput &&
               !any_popup_open()) {
        command_.clear();
    }

    // Right-aligned: the live-run indicator (SPEC §6.5, reduced) and the
    // command line's key buttons, each doing exactly what its key does.
    // ImGui has no right-to-left flow and the theme's widths are its own, so
    // the block is placed by its width last frame. The ↑ ↓ pair joins only
    // when it fits; a click on a row selects it all the same.
    ImGui::SameLine(0.0f, metric::gap);
    const float avail = ImGui::GetContentRegionAvail().x;
    const bool updown_fits =
        command_right_w_ > 0.0f && avail >= command_right_w_ + updown_w_;
    {
        const float block_w = command_right_w_ + (updown_fits ? updown_w_ : 0.0f);
        if (command_right_w_ > 0.0f && avail > block_w)
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + avail - block_w);
    }
    const float right_x0 = ImGui::GetCursorScreenPos().x;
    if (const int jobs = running_jobs(); jobs > 0) {
        // The live diamond, filled while work runs. Its opacity breathes
        // 100% -> 45% -> 100% over 2.4 s (SPEC §6.5), so liveness reads
        // before the count is read.
        ImGui::Dummy(ImVec2(8.0f, ImGui::GetFrameHeight()));
        const ImVec2 mn = ImGui::GetItemRectMin();
        const ImVec2 mx = ImGui::GetItemRectMax();
        const float phase = (float)std::fmod(ImGui::GetTime(), 2.4) / 2.4f;
        const float alpha = 0.725f + 0.275f * std::cos(phase * 6.2831853f);
        sternbild::diamond(ImGui::GetWindowDrawList(),
                           ImVec2(std::floor((mn.x + mx.x) * 0.5f),
                                  std::floor((mn.y + mx.y) * 0.5f)),
                           4.0f,
                           sternbild::with_alpha(col::parchment, (int)(255.0f * alpha)),
                           true);
        ImGui::SameLine(0.0f, metric::pad);
        char text[48];
        std::snprintf(text, sizeof text, jobs == 1 ? "%d job running"
                                                   : "%d jobs running", jobs);
        ImGui::AlignTextToFramePadding();
        ImGui::PushStyleColor(ImGuiCol_Text, col::secondary);
        ImGui::TextUnformatted(text);
        ImGui::PopStyleColor();
        ImGui::SameLine(0.0f, metric::gap);
    }
    // ⌘K: focus the command line (applied next frame, like the chord).
    if (sternbild::key_button(key(kKeyCommand), "Command")) focus_command_ = true;
    // ↑ ↓: move the selection one row, as the arrows do; focus then goes to
    // the table so the arrow keys carry on from the click. Drawn when they
    // fit, and once at start-up so their width is known.
    const bool updown_drawn = updown_fits || updown_w_ == 0.0f;
    if (updown_drawn) {
        const float x = ImGui::GetItemRectMax().x;
        const bool any = !rows_.empty();
        const char* none = "no instruments to select";
        ImGui::SameLine(0.0f, metric::pad);
        if (sternbild::key_button(key(kKeyUp), nullptr, any, none)) select_step_ -= 1;
        if (any) ImGui::SetItemTooltip("Previous instrument");
        ImGui::SameLine(0.0f, metric::pad * 0.5f);
        if (sternbild::key_button(key(kKeyDown), nullptr, any, none)) select_step_ += 1;
        if (any) ImGui::SetItemTooltip("Next instrument");
        updown_w_ = ImGui::GetItemRectMax().x - x;   // gap included
        if (select_step_ != 0) focus_instruments_ = std::max(focus_instruments_, 1);
    }
    // esc: clear the filter and hand the keys to the table, as Esc does from
    // the line. With no filter there is nothing to clear, and it says so.
    ImGui::SameLine(0.0f, metric::pad);
    if (sternbild::key_button(key(kKeyEscape), "Clear", !command_.empty(),
                              "no filter to clear")) {
        command_.clear();
        focus_instruments_ = std::max(focus_instruments_, 1);
    }
    command_right_w_ = ImGui::GetItemRectMax().x - right_x0 -
                       (updown_drawn ? updown_w_ : 0.0f);

    sternbild::command_bar_end();
}

// ============================================================================
// Rows
// ============================================================================

void LandingScreen::rebuild_rows(const AppletLoader& loader) {
    // Space-separated terms, all of which must appear (case-insensitive) in
    // the name, id or tag: "embed train" narrows the way a terminal filter
    // does; a single term is a plain substring match.
    std::vector<std::string> terms;
    {
        const std::string needle = lower(command_);
        size_t i = 0;
        while (i < needle.size()) {
            while (i < needle.size() && std::isspace((unsigned char)needle[i])) ++i;
            size_t j = i;
            while (j < needle.size() && !std::isspace((unsigned char)needle[j])) ++j;
            if (j > i) terms.push_back(needle.substr(i, j - i));
            i = j;
        }
    }

    rows_.clear();
    for (int i = 0; i < loader.count(); ++i) {
        const auto& m = loader.at(i).manifest;
        if (!terms.empty()) {
            const std::string hay =
                lower(m.name) + '\n' + lower(m.id) + '\n' + lower(m.tag);
            bool all = true;
            for (const auto& t : terms) all = all && hay.find(t) != std::string::npos;
            if (!all) continue;
        }
        rows_.push_back(i);
    }

    // The table's sort column first, then tag, then name; stable, so equal
    // rows keep the loader's order and nothing reshuffles between frames.
    const int column = sort_column_;
    const bool descending = sort_descending_;
    auto key_compare = [&](const AppletEntry& a, const AppletEntry& b) {
        switch (column) {
            case kName:    return natural_compare(a.manifest.name, b.manifest.name);
            case kId:      return natural_compare(a.manifest.id, b.manifest.id);
            case kVersion:
                return natural_compare(a.manifest.version, b.manifest.version);
            case kStatus:  return status_rank(a.status) - status_rank(b.status);
            default:       return natural_compare(a.manifest.tag, b.manifest.tag);
        }
    };
    std::stable_sort(rows_.begin(), rows_.end(), [&](int ia, int ib) {
        const AppletEntry& a = loader.at(ia);
        const AppletEntry& b = loader.at(ib);
        if (int c = key_compare(a, b)) return descending ? c > 0 : c < 0;
        if (int c = natural_compare(a.manifest.tag, b.manifest.tag)) return c < 0;
        return natural_compare(a.manifest.name, b.manifest.name) < 0;
    });
}

// ============================================================================
// Dock space
// ============================================================================

// Panes via ImGui docking (Bloomberg Launchpad, Blender areas): one
// undecorated host between the two fixed bars, four panes tiled on 1 px
// rules, each opened by its own header strip (pane_header) instead of a
// docking tab. The tab bar is removed outright (NoTabBar): AutoHideTabBar
// would leave ImGui's "unhide" triangle in every pane's corner, on top of
// the header title. Undocking and the per-node window menu are off: the four
// panes are the landing, not a desktop to rearrange, so the only edit is
// moving a splitter, and that persists in the app-data imgui.ini like the
// applet layout does.
void LandingScreen::draw_dock_space() {
    // The command and status bars are viewport side bars: they reserve their
    // height from the work area (from the next NewFrame on), so the host is
    // simply the work area, with no bar heights to subtract.
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float height = std::max(1.0f, vp->WorkSize.y);
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(ImVec2(vp->WorkSize.x, height));
    ImGui::SetNextWindowViewport(vp->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::Begin("##sternbild_landing", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoBringToFrontOnFocus |
                     ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoDocking);
    ImGui::PopStyleVar(3);

    const ImGuiID dock_id = ImGui::GetID("sternbild_landing_dockspace");

    // First-run layout, attempted once per process, with the same guard as
    // the applet page in main.cpp:
    //  - fresh dockspace (no imgui.ini history): build the four-pane tiling,
    //    left 0.62 (Projects 0.22 over Instruments: Projects is empty until
    //    the project model lands), right 0.38 (Inspector over Activity).
    //  - restored dockspace: dock ONLY panes that have no saved settings yet
    //    (a pane added after the ini was written); the user's splits win.
    if (!layout_built_) {
        static const char* const panes[] = {kProjects, kInstruments, kInspector,
                                            kActivity};
        auto unseen = [](const char* w) {
            return ImGui::FindWindowSettingsByID(ImHashStr(w)) == nullptr;
        };
        ImGuiDockNode* node = ImGui::DockBuilderGetNode(dock_id);
        if (node == nullptr || node->IsEmpty()) {
            ImGui::DockBuilderRemoveNode(dock_id);
            ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
            ImGui::DockBuilderSetNodeSize(dock_id, ImVec2(vp->WorkSize.x, height));
            ImGuiID left = 0, right = 0;
            ImGui::DockBuilderSplitNode(dock_id, ImGuiDir_Left, 0.62f, &left, &right);
            ImGuiID projects = 0, instruments = 0;
            ImGui::DockBuilderSplitNode(left, ImGuiDir_Up, 0.22f, &projects,
                                        &instruments);
            ImGuiID inspector = 0, activity = 0;
            ImGui::DockBuilderSplitNode(right, ImGuiDir_Up, 0.55f, &inspector,
                                        &activity);
            ImGui::DockBuilderDockWindow(kProjects, projects);
            ImGui::DockBuilderDockWindow(kInstruments, instruments);
            ImGui::DockBuilderDockWindow(kInspector, inspector);
            ImGui::DockBuilderDockWindow(kActivity, activity);
            ImGui::DockBuilderFinish(dock_id);
        } else {
            bool any_unseen = false;
            for (const char* w : panes) any_unseen |= unseen(w);
            if (any_unseen) {
                ImGuiDockNode* central = ImGui::DockBuilderGetCentralNode(dock_id);
                const ImGuiID target = central ? central->ID : dock_id;
                for (const char* w : panes)
                    if (unseen(w)) ImGui::DockBuilderDockWindow(w, target);
                ImGui::DockBuilderFinish(dock_id);
            }
        }
        layout_built_ = true;
    }

    ImGuiDockNodeFlags dock_flags = ImGuiDockNodeFlags_NoUndocking;
    dock_flags |= ImGuiDockNodeFlags_NoWindowMenuButton;
    dock_flags |= ImGuiDockNodeFlags_NoTabBar;
    ImGui::DockSpace(dock_id, ImVec2(0.0f, 0.0f), dock_flags);
    ImGui::End();
}

// ============================================================================
// Panes
// ============================================================================

// Projects: with no projects, the empty table becomes the numbered menu of
// ways in (Bloomberg numbered menus, VS Code Welcome; landing/final): number,
// the way in as a key button holding its function key, and the command-line
// words, so the eventual paths are learnable now (progressive disclosure,
// Hick: two choices). None works yet, so each button is disabled, faint, and
// says why on hover (honesty: never a silent dead control), and the words
// are faint too.
void LandingScreen::draw_projects() {
    pane_header(kProjects, "none yet", kKeyPane[0]);

    struct WayIn {
        const char* n;
        const char* key;
        const char* action;
        const char* command;
    };
    static const WayIn ways[] = {
        {"1", "F2", "Open folder", "open <folder>"},
        {"2", "F4", "Attach to process", "attach <pid>"},
    };
    ImFont* mono = sternbild::fonts().mono;
    // A way-in row is its key button and the cell padding around it. The
    // button reports its text line to the row, so the words after it share
    // that line; the number before it is dropped by the same inset by hand.
    const float way_row = metric::key_button + 2.0f * ImGui::GetStyle().CellPadding.y;
    const float text_dy = std::floor((metric::key_button - font_size::ui) * 0.5f);
    const float height = std::max(
        1.0f, std::min(metric::row + way_row * 2.0f + 2.0f,
                       ImGui::GetContentRegionAvail().y));
    if (sternbild::begin_table("##projects", 3, ImVec2(0.0f, height),
                               ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Way in", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Command", ImGuiTableColumnFlags_WidthStretch);
        sternbild::table_header_row();
        for (const WayIn& w : ways) {
            ImGui::TableNextRow(ImGuiTableRowFlags_None, way_row);
            ImGui::TableNextColumn();
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + text_dy);
            cell_text(w.n, mono, font_size::mono, col::faint, true);
            ImGui::TableNextColumn();
            sternbild::key_button(w.key, w.action, false, "not yet available");
            ImGui::TableNextColumn();
            cell_text(w.command, mono, font_size::mono, col::faint);
        }
        sternbild::end_table();
    }
}

// Instruments: the overview (Shneiderman). Dense table (Bloomberg, W&B run
// tables): tabular mono figures for ids and versions, a sortable header, row
// hover, the selected row in bronze, full-row hit targets (Fitts), ↑↓ to move
// and Enter or a double click to launch.
void LandingScreen::draw_instruments(const Context& ctx) {
    const AppletLoader* loader = ctx.loader;
    {
        const int n = loader ? loader->count() : 0;
        int unavailable = 0;
        for (int i = 0; i < n; ++i) unavailable += !launchable(loader->at(i)) &&
                                    loader->at(i).status != AppletStatus::Active;
        char subtitle[64];
        if (unavailable > 0)
            std::snprintf(subtitle, sizeof subtitle,
                          "%d installed \xc2\xb7 %d unavailable", n, unavailable);
        else
            std::snprintf(subtitle, sizeof subtitle, "%d installed", n);
        pane_header(kInstruments, subtitle, kKeyPane[1]);
    }
    if (!loader || loader->count() == 0) {
        faint_line("No instruments found. Drop a dylib + <name>.caliper.toml "
                   "into the applets folder.");
        return;
    }
    ImFont* ui = sternbild::fonts().ui;
    ImFont* mono = sternbild::fonts().mono;

    // Keyboard: arrows move the selection while this pane has focus, or while
    // the command line is being edited (Bloomberg: type, arrow, Enter, never
    // leaving the line). The arrows are claimed for the pane every frame so
    // ImGui's own keyboard nav does not also walk a nav cursor through the
    // rows; Enter is claimed only when the command line is not editing, since
    // the field reads Enter itself. The command bar's ↑ ↓ buttons add their
    // step to the keys'.
    const bool focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
    int step = select_step_;
    if ((focused || command_active_) && !any_popup_open()) {
        const ImGuiID owner = ImGui::GetID("##instrument_keys");
        ImGui::SetKeyOwner(ImGuiKey_UpArrow, owner);
        ImGui::SetKeyOwner(ImGuiKey_DownArrow, owner);
        if (!command_active_) {
            ImGui::SetKeyOwner(ImGuiKey_Enter, owner);
            ImGui::SetKeyOwner(ImGuiKey_KeypadEnter, owner);
        }
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow, true)) step += 1;
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow, true)) step -= 1;
        if (!command_active_ && (ImGui::IsKeyPressed(ImGuiKey_Enter, false) ||
                                 ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false)))
            enter_requested_ = true;
    }
    if (step != 0 && !rows_.empty()) {
        const auto it = std::find(rows_.begin(), rows_.end(), selected_);
        const int last = (int)rows_.size() - 1;
        const int pos = it == rows_.end()
                            ? (step > 0 ? 0 : last)
                            : std::clamp((int)(it - rows_.begin()) + step, 0, last);
        selected_ = rows_[(size_t)pos];
        scroll_to_selected_ = true;
    }

    // An empty result keeps the header (the columns stay legible) and says
    // so in the body, instead of a blank pane.
    const ImVec2 size = rows_.empty() ? kEmptyTableSize : ImVec2(0.0f, 0.0f);
    // Columns sit tight on their content, terminal style; the last column
    // takes the slack so the eye never crosses a gap to reach the next value.
    // ID auto-fits; VERSION leaves room for the sort chevron. No saved
    // settings: only the sort column was persisted, which reordered the
    // columns after a restart; the column order and the tag sort are fixed.
    if (sternbild::begin_table("##instruments", 5, size,
                               ImGuiTableFlags_Sortable |
                                   ImGuiTableFlags_SizingFixedFit |
                                   ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Name",
                                ImGuiTableColumnFlags_WidthFixed |
                                    ImGuiTableColumnFlags_NoHide,
                                150.0f, kName);
        ImGui::TableSetupColumn("Id", ImGuiTableColumnFlags_WidthFixed, 0.0f, kId);
        // '>' marks a numeric column: the theme right-aligns its head over
        // the right-aligned figures and does not draw the marker.
        ImGui::TableSetupColumn(">Version", ImGuiTableColumnFlags_WidthFixed, 64.0f,
                                kVersion);
        ImGui::TableSetupColumn("Tag",
                                ImGuiTableColumnFlags_WidthFixed |
                                    ImGuiTableColumnFlags_DefaultSort,
                                80.0f, kTag);
        ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthStretch, 1.0f,
                                kStatus);

        // A header that looks sortable sorts: clicking a column re-sorts the
        // rows (tag, then name, break ties) for the session.
        if (ImGuiTableSortSpecs* specs = ImGui::TableGetSortSpecs()) {
            if (specs->SpecsCount > 0) {
                const ImGuiTableColumnSortSpecs& s = specs->Specs[0];
                const int column = (int)s.ColumnUserID;
                const bool descending =
                    s.SortDirection == ImGuiSortDirection_Descending;
                if (column != sort_column_ || descending != sort_descending_) {
                    sort_column_ = column;
                    sort_descending_ = descending;
                    rebuild_rows(*loader);
                }
            }
            specs->SpecsDirty = false;
        }
        sternbild::table_header_row();

        for (const int idx : rows_) {
            const AppletEntry& e = loader->at(idx);
            const bool selected = idx == selected_;
            ImGui::PushID(idx);
            ImGui::TableNextRow(ImGuiTableRowFlags_None, metric::row);

            ImGui::TableNextColumn();
            const float cell_x = ImGui::GetCursorPosX();
            bool double_clicked = false;
            if (sternbild::table_row_selectable("##row", selected, &double_clicked))
                selected_ = idx;
            if (double_clicked && launchable(e)) pending_launch_ = idx;

            // Keep a keyboard-moved selection in view. The cell clip rect of an
            // unfrozen row starts below the frozen header, so a row hidden
            // under the header counts as out of view too; centring it avoids
            // landing it right back under the header.
            if (selected && scroll_to_selected_) {
                const ImVec2 clip_min = ImGui::GetWindowDrawList()->GetClipRectMin();
                const ImVec2 clip_max = ImGui::GetWindowDrawList()->GetClipRectMax();
                if (ImGui::GetItemRectMin().y < clip_min.y ||
                    ImGui::GetItemRectMax().y > clip_max.y)
                    ImGui::SetScrollHereY(0.5f);
            }

            // The row hit target spans the row; the cell text starts at the
            // cell's own left edge, aligned with its header.
            ImGui::SameLine();
            ImGui::SetCursorPosX(cell_x);
            cell_text(e.manifest.name.c_str(), ui, font_size::ui,
                      sternbild::row_text(selected, col::parchment));
            ImGui::TableNextColumn();
            cell_text(e.manifest.id.c_str(), mono, font_size::mono,
                      sternbild::row_text(selected, col::secondary));
            ImGui::TableNextColumn();
            cell_text(e.manifest.version.c_str(), mono, font_size::mono,
                      sternbild::row_text(selected, col::parchment), true);
            ImGui::TableNextColumn();
            cell_text(e.manifest.tag.c_str(), ui, font_size::ui,
                      sternbild::row_text(selected, col::secondary));
            ImGui::TableNextColumn();
            cell_text(status_word(e.status), ui, font_size::ui,
                      sternbild::row_text(selected, status_colour(e.status)));
            ImGui::PopID();
        }
        if (rows_.empty())
            empty_row(("No instruments match \"" + command_ + "\".").c_str());
        sternbild::end_table();
    }
    scroll_to_selected_ = false;
}

// Inspector: details on demand (Shneiderman) for the selected row, as a
// two-column definition list that scrolls on its own, over a fixed footer
// holding the one primary action (Fitts: a single target per pane, always in
// the same place and never scrolled out of reach by a long summary).
void LandingScreen::draw_inspector(const Context& ctx) {
    const AppletLoader* loader = ctx.loader;
    const bool any = loader && selected_ >= 0 && selected_ < loader->count();
    pane_header(kInspector,
                any ? loader->at(selected_).manifest.name.c_str() : "none selected",
                kKeyPane[2]);
    if (!any) {
        faint_line("Select an instrument.");
        return;
    }
    const AppletEntry& e = loader->at(selected_);

    // The footer is the plate and a padding above and below it; the list
    // takes the rest of the pane.
    constexpr float kPlate = 24.0f;   // primary_button's documented height
    const float footer_h = kPlate + 2.0f * metric::pad;
    const float table_y = ImGui::GetCursorPosY();
    const float table_h =
        std::max(metric::row, ImGui::GetContentRegionAvail().y - footer_h);
    if (sternbild::begin_table("##inspector", 2, ImVec2(0.0f, table_h),
                               ImGuiTableFlags_SizingFixedFit)) {
        draw_inspector_rows(e);
        sternbild::end_table();
    }

    // Footer: the one ⏎ Launch plate, its key inside it, since Enter on the
    // table or the command line does the same. Disabled when the row cannot
    // launch, with the reason beside it in faint (honesty: a disabled control
    // says why); one line, cut with an ellipsis, since the STATUS row above
    // carries the full text. A quarantined applet says instead what brings it
    // back: after a C++ exception (the stack unwound, nothing is corrupt) a
    // ⌘R Restart key button that unloads and relaunches it; after a signal
    // (memory may be corrupt) only a restart of Caliper itself.
    ImGui::SetCursorPosY(table_y + table_h + metric::pad);
    const bool ready = launchable(e);
    const float button_y = ImGui::GetCursorPosY();
    ImGui::BeginDisabled(!ready);
    if (sternbild::primary_button("Launch", ImVec2(0.0f, kPlate), key(kKeyEnter)))
        pending_launch_ = selected_;
    ImGui::EndDisabled();
    if (!ready && loader->restartable(selected_)) {
        // key_button sits on the plate's text baseline, so the two share a
        // line without any offset here.
        ImGui::SameLine(0.0f, metric::gap);
        if (sternbild::key_button(key(kKeyRestart), "Restart"))
            pending_restart_ = selected_;
    } else if (!ready) {
        ImGui::SameLine(0.0f, metric::gap);
        const float text_h = ImGui::GetTextLineHeight();
        ImGui::SetCursorPosY(button_y + std::max(0.0f, (kPlate - text_h) * 0.5f));
        const char* reason = e.status == AppletStatus::Active ? "running"
                             : e.status == AppletStatus::Quarantined
                                 ? "restart Caliper to reload"
                             : e.status_text.empty() ? "not available"
                                                     : e.status_text.c_str();
        const ImVec2 at = ImGui::GetCursorScreenPos();
        const float w = std::max(1.0f, ImGui::GetContentRegionAvail().x);
        ImGui::PushStyleColor(ImGuiCol_Text, col::faint);
        ImGui::RenderTextEllipsis(ImGui::GetWindowDrawList(), at,
                                  ImVec2(at.x + w, at.y + text_h), at.x + w, reason,
                                  nullptr, nullptr);
        ImGui::PopStyleColor();
        ImGui::Dummy(ImVec2(w, text_h));
    }
}

// The Inspector's definition rows: label (small uppercase secondary) left,
// value right, one row per manifest field, in manifest order.
void LandingScreen::draw_inspector_rows(const caliper_host::AppletEntry& e) {
    const auto& m = e.manifest;
    ImFont* ui = sternbild::fonts().ui;
    ImFont* mono = sternbild::fonts().mono;

    ImGui::TableSetupColumn("Field", ImGuiTableColumnFlags_WidthFixed, 110.0f);
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

    // A definition row: label (small uppercase secondary) left, value right.
    // The label is a size smaller than the value, so it is lowered onto the
    // value's baseline rather than sharing its top.
    const float label_dy = std::max(
        0.0f, ascent(ui, font_size::ui) - ascent(ui, font_size::ui_small));
    auto field = [label_dy](const char* name) {
        ImGui::TableNextRow(ImGuiTableRowFlags_None, metric::row);
        ImGui::TableNextColumn();
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + label_dy);
        sternbild::label(name);
        ImGui::TableNextColumn();
    };
    // Service lists: one id per line in mono, each line its own item so the
    // row grows with the list; an empty list says "none" in faint.
    auto list = [&](const std::vector<std::string>& items) {
        if (items.empty()) cell_text("none", mono, font_size::mono, col::faint);
        for (const auto& s : items)
            cell_text(s.c_str(), mono, font_size::mono, col::parchment);
    };
    // Optional strings: absent values are said, not left blank.
    auto mono_value = [&](const std::string& s, const char* absent) {
        if (s.empty()) cell_text(absent, mono, font_size::mono, col::faint);
        else cell_text(s.c_str(), mono, font_size::mono, col::parchment);
    };

    field("Name");
    cell_wrapped(m.name.c_str(), ui, font_size::ui, col::parchment);
    field("Id");
    mono_value(m.id, "unknown");
    field("Version");
    mono_value(m.version, "unknown");
    field("Tag");
    if (m.tag.empty()) cell_text("none", ui, font_size::ui, col::faint);
    else cell_text(m.tag.c_str(), ui, font_size::ui, col::parchment);
    field("ABI epoch");
    cell_text(std::to_string(m.abi_epoch).c_str(), mono, font_size::mono,
              col::parchment);
    field("Min host");
    mono_value(m.min_host, "none");
    field("Required");
    list(m.required_services);
    field("Optional");
    list(m.optional_services);
    field("Binary");
    cell_wrapped(e.dylib_path.empty() ? "missing" : e.dylib_path.c_str(), mono,
                 font_size::mono, col::faint);
    field("Status");
    cell_text(status_word(e.status), ui, font_size::ui, status_colour(e.status));
    if (!launchable(e) && !e.status_text.empty())
        cell_wrapped(e.status_text.c_str(), ui, font_size::ui, col::faint);
    field("Summary");
    if (m.summary.empty()) cell_text("none", ui, font_size::ui, col::faint);
    else cell_wrapped(m.summary.c_str(), ui, font_size::ui, col::parchment);
}

// Activity: what the machine is doing (running jobs, live) above what it did
// (the host log, newest last). Mono throughout so times and percentages sit
// in fixed columns.
void LandingScreen::draw_activity() {
    pane_header(kActivity, "this machine \xc2\xb7 newest first", kKeyPane[3]);
    ImFont* ui = sternbild::fonts().ui;
    ImFont* mono = sternbild::fonts().mono;
    ImGui::PushFont(mono, font_size::mono);

    bool any_running = false;
    for (const auto& v : caliper_host::host_job_system().views()) {
        if (!v.running) continue;
        any_running = true;
        char pct[8];
        std::snprintf(pct, sizeof pct, "%3d%%", (int)std::lround(v.progress * 100.0f));
        ImGui::PushStyleColor(ImGuiCol_Text, col::parchment);
        ImGui::TextUnformatted(pct);
        ImGui::SameLine(0.0f, metric::pad);
        ImGui::TextUnformatted(v.label.c_str());
        ImGui::PopStyleColor();
        if (!v.message.empty()) {
            ImGui::SameLine(0.0f, metric::pad);
            ImGui::PushStyleColor(ImGuiCol_Text, col::faint);
            ImGui::TextUnformatted(v.message.c_str());
            ImGui::PopStyleColor();
        }
    }
    if (any_running) ImGui::Separator();
    ImGui::PopFont();

    // The host log as a table (VS Code Output, Datadog log explorer), newest
    // first so the latest event is where the eye lands, with a relative WHEN
    // instead of a clock. Objects are paths and ids: never wrapped, so the
    // table scrolls sideways (ScrollX) rather than cutting them at the edge.
    if (log_.empty()) {
        faint_line("No activity.");
        return;
    }
    if (!sternbild::begin_table("##activity", 4, ImVec2(0.0f, 0.0f),
                                ImGuiTableFlags_SizingFixedFit |
                                    ImGuiTableFlags_ScrollX))
        return;
    // WHEN is as wide as its widest value ("59 min ago"), so the column never
    // jitters as the times tick over.
    ImGui::PushFont(mono, font_size::mono);
    const float when_w = ImGui::CalcTextSize("59 min ago").x;
    ImGui::PopFont();
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("When", ImGuiTableColumnFlags_WidthFixed, when_w);
    ImGui::TableSetupColumn("Event", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("Object", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("Detail", ImGuiTableColumnFlags_WidthFixed);
    sternbild::table_header_row();

    const double now = now_epoch();
    ImGuiListClipper clipper;
    clipper.Begin((int)log_.size(), metric::row);
    while (clipper.Step()) {
        for (int r = clipper.DisplayStart; r < clipper.DisplayEnd; ++r) {
            const LogLine& line = log_[log_.size() - 1 - (size_t)r];   // newest first
            ImGui::TableNextRow(ImGuiTableRowFlags_None, metric::row);
            ImGui::TableNextColumn();
            cell_text(ago(std::max(0.0, now - line.time_sec)).c_str(), mono,
                      font_size::mono, col::faint);
            ImGui::TableNextColumn();
            cell_text(line.event.c_str(), ui, font_size::ui, col::parchment);
            ImGui::TableNextColumn();
            cell_text(line.object.c_str(), mono, font_size::mono, col::parchment);
            ImGui::TableNextColumn();
            cell_text(line.detail.c_str(), mono, font_size::mono, col::secondary);
        }
    }
    sternbild::end_table();
}

// ============================================================================
// Status bar
// ============================================================================

void draw_host_status_items(const char* renderer_name) {
    sternbild::status_item(upper(renderer_name ? renderer_name : "").c_str());
    sternbild::status_item(caliper_host::device_info().name.c_str());
}

// Status bar (VS Code / JetBrains): the machine at a glance, always in the
// same place: backend, device, memory, what is installed. Facts only: every
// action is a key button in the command bar or a pane, so none appears twice
// (the panes' ⌘1–⌘4 are their header strips, which focus the pane on click).
void LandingScreen::draw_status_bar(const Context& ctx) {
    const auto& dev = caliper_host::device_info();
    static const uint64_t total_memory = physical_memory_bytes();

    sternbild::status_bar_begin();
    // status_item() places itself after the previous item, dot-separated.
    draw_host_status_items(ctx.renderer_name);
    if (total_memory > 0) {
        sternbild::status_item((gigabytes(total_memory) + " RAM").c_str());
    }
    // free_memory_hint is the device's working-set budget (Metal reports its
    // recommendedMaxWorkingSetSize), not free memory, so it is named that.
    if (dev.free_memory_hint > 0) {
        const std::string budget = gigabytes(dev.free_memory_hint) + " GPU budget";
        sternbild::status_item(budget.c_str());
    }
    if (ctx.loader) {
        int unavailable = 0;
        for (int i = 0; i < ctx.loader->count(); ++i) {
            const AppletStatus s = ctx.loader->at(i).status;
            unavailable += (s != AppletStatus::Ready && s != AppletStatus::Active);
        }
        const int n = ctx.loader->count();
        char text[64];
        std::snprintf(text, sizeof text,
                      n == 1 ? "%d instrument" : "%d instruments", n);
        sternbild::status_item(text);
        if (unavailable > 0) {
            std::snprintf(text, sizeof text, "%d unavailable", unavailable);
            sternbild::status_item(text);
        }
    }

    sternbild::status_bar_end();
}
