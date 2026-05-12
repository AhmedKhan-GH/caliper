// ============================================================================
// UCDH PreE — Preliminary Exploration applet.
//
// Minimal dataset explorer over an in-memory DuckDB connection. The user
// picks a folder, we list supported files (csv / parquet / json), and on
// selection we run a schema probe and a small preview query. DuckDB is
// happy to read these files directly via the path-as-table form
// (`SELECT * FROM 'path.csv'`), so there's no copy / register step.
//
// Designed to be extended: an applet-level DuckDB connection survives the
// session, so future EDA features (filters, ad-hoc SQL, joins between
// open files via VIEWs, plot bindings) can layer on without rebuilding
// state. A SQL editor pane and column-stats sidebar are obvious next steps.
// ============================================================================

#include "ucdh_pree_applet.h"
#include "app_paths.h"

#include <imgui.h>
#include <ImGuiFileDialog.h>
#include <duckdb.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Recognized data formats. Anything DuckDB can read directly via
// `SELECT * FROM 'path'` belongs here.
struct FormatInfo {
    const char* ext;     // lowercase, with dot
    const char* label;   // short tag for UI
};

const FormatInfo kFormats[] = {
    {".csv",     "CSV"},
    {".tsv",     "TSV"},
    {".parquet", "PARQ"},
    {".pq",      "PARQ"},
    {".json",    "JSON"},
    {".jsonl",   "JSONL"},
    {".ndjson",  "JSONL"},
};

const FormatInfo* match_format(const std::string& path) {
    auto pos = path.find_last_of('.');
    if (pos == std::string::npos) return nullptr;
    std::string ext = path.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    for (const auto& f : kFormats) {
        if (ext == f.ext) return &f;
    }
    return nullptr;
}

// Escape single quotes in a path for safe inlining into a SQL string. DuckDB
// uses standard SQL escaping (double up the quote). We don't bind via prepared
// statements because the path is part of the table reference, not a parameter.
std::string sql_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        if (c == '\'') out += "''";
        else           out += c;
    }
    return out;
}

struct DiscoveredFile {
    std::string path;          // absolute
    std::string display_name;  // basename
    const FormatInfo* fmt = nullptr;
    uintmax_t  size_bytes = 0;
};

void scan_folder(const std::string& dir, std::vector<DiscoveredFile>& out) {
    out.clear();
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return;

    for (auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        const auto p = entry.path().string();
        const FormatInfo* fmt = match_format(p);
        if (!fmt) continue;

        DiscoveredFile f;
        f.path         = p;
        f.display_name = entry.path().filename().string();
        f.fmt          = fmt;
        std::error_code se;
        f.size_bytes   = fs::file_size(entry.path(), se);
        out.push_back(std::move(f));
    }

    std::sort(out.begin(), out.end(),
        [](const DiscoveredFile& a, const DiscoveredFile& b) {
            return a.display_name < b.display_name;
        });
}

std::string format_size(uintmax_t bytes) {
    char buf[32];
    if (bytes < 1024) {
        std::snprintf(buf, sizeof(buf), "%llu B", (unsigned long long)bytes);
    } else if (bytes < 1024ULL * 1024) {
        std::snprintf(buf, sizeof(buf), "%.1f KB", bytes / 1024.0);
    } else if (bytes < 1024ULL * 1024 * 1024) {
        std::snprintf(buf, sizeof(buf), "%.1f MB", bytes / (1024.0 * 1024));
    } else {
        std::snprintf(buf, sizeof(buf), "%.2f GB", bytes / (1024.0 * 1024 * 1024));
    }
    return buf;
}

// Snapshot of a preview query — column names + types and the values for the
// first N rows, all stringified. Living in this struct lets us run the query
// once on selection and re-render the table cheaply each frame.
struct PreviewSnapshot {
    bool                                   ready = false;
    std::string                            error;
    std::vector<std::string>               col_names;
    std::vector<std::string>               col_types;
    std::vector<std::vector<std::string>>  rows;     // [row][col]
    size_t                                 row_count = 0;  // rows actually fetched
};

} // anonymous namespace

// ── State ────────────────────────────────────────────────────────────────

struct UCDHPreEApplet::State {
    std::unique_ptr<duckdb::DuckDB>     db;
    std::unique_ptr<duckdb::Connection> con;

    std::string                  current_dir;
    std::vector<DiscoveredFile>  files;

    int                          selected = -1;
    PreviewSnapshot              preview;

    int                          preview_limit = 100;

    // Column pagination — wide datasets (hundreds+ cols) overwhelm both the
    // ImGui table widget (hard cap 512) and the user's eyes. We render a
    // window of `col_page` columns starting at `col_offset`. col_offset is
    // reset to 0 whenever a new file is selected (see select_file_for_preview).
    int                          col_offset    = 0;
    int                          col_page      = 50;

    // Persisted between sessions via a tiny key=value file in the OS app-data
    // directory. We save on cleanup and on folder open; on init, we restore
    // last_dir / preview_limit / col_page. col_offset is intentionally NOT
    // persisted — it depends on which file you've selected.
    static constexpr const char* kSettingsFile = "ucdh_pree.ini";
    void load_settings();
    void save_settings() const;

    // Runs `SELECT * FROM '<path>' LIMIT preview_limit` against the open
    // DuckDB connection and stringifies the result into `preview`. Defined
    // as a method (rather than a free function in the anonymous namespace)
    // because the State struct is private to UCDHPreEApplet.
    void run_preview(const DiscoveredFile& f) {
        preview = PreviewSnapshot{};

        if (!con) {
            preview.error = "DuckDB connection not initialized";
            return;
        }

        const std::string p = sql_escape(f.path);
        char qbuf[1024];
        std::snprintf(qbuf, sizeof(qbuf),
            "SELECT * FROM '%s' LIMIT %d", p.c_str(), preview_limit);

        auto result = con->Query(qbuf);
        if (result->HasError()) {
            preview.error = result->GetError();
            return;
        }

        const size_t ncols = result->ColumnCount();
        preview.col_names.reserve(ncols);
        preview.col_types.reserve(ncols);
        for (size_t c = 0; c < ncols; c++) {
            preview.col_names.push_back(result->ColumnName(c));
            preview.col_types.push_back(result->types[c].ToString());
        }

        // Stream chunks. We cap at preview_limit rows total (LIMIT enforces
        // it, but we also stop early defensively).
        while (auto chunk = result->Fetch()) {
            const size_t nrows = chunk->size();
            for (size_t r = 0; r < nrows; r++) {
                std::vector<std::string> row;
                row.reserve(ncols);
                for (size_t c = 0; c < ncols; c++) {
                    duckdb::Value v = chunk->GetValue(c, r);
                    row.push_back(v.IsNull() ? "NULL" : v.ToString());
                }
                preview.rows.push_back(std::move(row));
            }
            if (preview.rows.size() >= (size_t)preview_limit) break;
        }
        preview.row_count = preview.rows.size();
        preview.ready = true;
    }
};

// ── Persistence ───────────────────────────────────────────────────────────

void UCDHPreEApplet::State::load_settings() {
    const std::string path = caliper::app_data_path(kSettingsFile);
    std::ifstream f(path);
    if (!f) return;     // first run / file missing — keep defaults

    std::string line;
    while (std::getline(f, line)) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq);
        const std::string val = line.substr(eq + 1);

        if (key == "last_dir") {
            current_dir = val;
        } else if (key == "preview_limit") {
            try { preview_limit = std::stoi(val); } catch (...) {}
        } else if (key == "col_page") {
            try { col_page = std::stoi(val); } catch (...) {}
        }
    }
}

void UCDHPreEApplet::State::save_settings() const {
    const std::string path = caliper::app_data_path(kSettingsFile);
    std::ofstream f(path, std::ios::trunc);
    if (!f) return;

    f << "last_dir=" << current_dir << '\n';
    f << "preview_limit=" << preview_limit << '\n';
    f << "col_page=" << col_page << '\n';
}

// ── Lifecycle ─────────────────────────────────────────────────────────────

bool UCDHPreEApplet::initialize() {
    s_ = new State();
    try {
        s_->db  = std::make_unique<duckdb::DuckDB>(nullptr);   // :memory:
        s_->con = std::make_unique<duckdb::Connection>(*s_->db);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ucdh_pree] DuckDB init failed: %s\n", e.what());
        delete s_;
        s_ = nullptr;
        return false;
    }

    // Restore prior session: last folder, page sizes. The folder gets
    // re-scanned lazily on first draw so we don't block init on disk I/O.
    s_->load_settings();
    if (!s_->current_dir.empty()) {
        scan_folder(s_->current_dir, s_->files);
    }
    return true;
}

void UCDHPreEApplet::cleanup() {
    if (!s_) return;
    s_->save_settings();
    s_->con.reset();
    s_->db.reset();
    delete s_;
    s_ = nullptr;
}

// ── Draw ──────────────────────────────────────────────────────────────────

void UCDHPreEApplet::draw_ui(int /*win_w*/, int /*win_h*/) {
    if (!s_) return;

    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::Begin("##UCDHPreERoot", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar);

    // ── Toolbar ──
    if (ImGui::Button("<< Back to Menu", ImVec2(140, 28))) {
        exit_requested_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Open Folder...", ImVec2(140, 28))) {
        IGFD::FileDialogConfig cfg;
        cfg.path = s_->current_dir.empty() ? "." : s_->current_dir;
        cfg.flags = ImGuiFileDialogFlags_Modal;
        // Empty filter → directory selection mode in ImGuiFileDialog.
        ImGuiFileDialog::Instance()->OpenDialog(
            "UCDHPreEOpenFolder", "Choose dataset folder", nullptr, cfg);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    if (s_->current_dir.empty()) {
        ImGui::TextDisabled("(no folder selected)");
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f),
            "%s", s_->current_dir.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("— %d file(s)", (int)s_->files.size());
    }

    // Render the file dialog itself (modal).
    ImVec2 min_sz(600, 400), max_sz(FLT_MAX, FLT_MAX);
    if (ImGuiFileDialog::Instance()->Display("UCDHPreEOpenFolder",
            ImGuiWindowFlags_NoCollapse, min_sz, max_sz)) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            s_->current_dir = ImGuiFileDialog::Instance()->GetCurrentPath();
            scan_folder(s_->current_dir, s_->files);
            s_->selected = -1;
            s_->preview  = PreviewSnapshot{};
            s_->save_settings();    // persist folder choice eagerly
        }
        ImGuiFileDialog::Instance()->Close();
    }

    ImGui::Separator();

    // ── Body: left = file list, right = schema + preview ──
    const float panel_w = 320.0f;
    const float avail_h = ImGui::GetContentRegionAvail().y;

    ImGui::BeginChild("##ucdh_files", ImVec2(panel_w, avail_h), true);
    ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "FILES");
    ImGui::Separator();

    if (s_->files.empty()) {
        ImGui::TextDisabled(
            s_->current_dir.empty()
                ? "Click \"Open Folder...\" to begin."
                : "No supported files in this folder.");
        ImGui::Spacing();
        ImGui::TextDisabled("Supported: .csv .tsv .parquet .json .jsonl");
    } else {
        for (int i = 0; i < (int)s_->files.size(); i++) {
            const auto& f = s_->files[i];
            const bool sel = (i == s_->selected);

            char label[512];
            std::snprintf(label, sizeof(label), "[%s] %s  (%s)##%d",
                f.fmt ? f.fmt->label : "?",
                f.display_name.c_str(),
                format_size(f.size_bytes).c_str(),
                i);

            if (ImGui::Selectable(label, sel)) {
                s_->selected   = i;
                s_->col_offset = 0;       // new file → reset column window
                s_->run_preview(f);
            }
        }
    }

    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("##ucdh_preview", ImVec2(0, avail_h), false);

    if (s_->selected < 0 || s_->selected >= (int)s_->files.size()) {
        ImGui::TextDisabled("Select a file to preview its schema and rows.");
    } else {
        const auto& f = s_->files[s_->selected];
        ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "FILE");
        ImGui::Separator();
        ImGui::TextWrapped("%s", f.path.c_str());
        ImGui::TextDisabled("%s · %s",
            f.fmt ? f.fmt->label : "?",
            format_size(f.size_bytes).c_str());

        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "PREVIEW");
        ImGui::Separator();

        // Preview row-cap control. Re-runs the query when changed.
        ImGui::SetNextItemWidth(180);
        if (ImGui::SliderInt("rows", &s_->preview_limit, 10, 5000, "%d")) {
            s_->run_preview(f);
        }
        ImGui::SameLine();
        if (ImGui::Button("Refresh")) {
            s_->run_preview(f);
        }

        ImGui::Spacing();

        if (!s_->preview.ready) {
            if (!s_->preview.error.empty()) {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                    "Query failed:");
                ImGui::TextWrapped("%s", s_->preview.error.c_str());
            } else {
                ImGui::TextDisabled("(loading...)");
            }
        } else {
            // ── Column pagination ──
            // Wide CSVs (hundreds of cols) need windowed rendering. We show
            // [col_offset, col_offset + col_page). col_page is hard-capped
            // safely below ImGui's 512-column table widget limit.
            const int  total_cols = (int)s_->preview.col_names.size();
            const int  kMaxPage   = 256;
            int&       col_page   = s_->col_page;
            int&       col_off    = s_->col_offset;

            if (col_page < 1)        col_page = 1;
            if (col_page > kMaxPage) col_page = kMaxPage;
            const int max_off = std::max(0, total_cols - col_page);
            if (col_off < 0)         col_off = 0;
            if (col_off > max_off)   col_off = max_off;

            const int ncols     = std::min(col_page, total_cols - col_off);
            const int last_col  = col_off + ncols;

            ImGui::TextDisabled("%zu rows × %d cols total — viewing cols %d–%d",
                s_->preview.row_count, total_cols, col_off + 1, last_col);

            // Pagination controls only appear when the data is wider than
            // a single page can show.
            if (total_cols > col_page) {
                ImGui::BeginDisabled(col_off == 0);
                if (ImGui::Button("<<", ImVec2(36, 0))) col_off = 0;
                ImGui::SameLine();
                if (ImGui::Button("<",  ImVec2(36, 0))) col_off = std::max(0, col_off - col_page);
                ImGui::EndDisabled();

                ImGui::SameLine();
                ImGui::SetNextItemWidth(220);
                ImGui::SliderInt("##col_off", &col_off, 0, max_off, "first col %d");

                ImGui::SameLine();
                ImGui::BeginDisabled(col_off >= max_off);
                if (ImGui::Button(">",  ImVec2(36, 0))) col_off = std::min(max_off, col_off + col_page);
                ImGui::SameLine();
                if (ImGui::Button(">>", ImVec2(36, 0))) col_off = max_off;
                ImGui::EndDisabled();

                ImGui::SameLine();
                ImGui::SetNextItemWidth(140);
                ImGui::SliderInt("cols/page", &col_page, 1, kMaxPage, "%d");
            }

            if (ncols > 0 && ImGui::BeginTable("##preview_tbl", ncols,
                    ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                    ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollX |
                    ImGuiTableFlags_ScrollY,
                    ImVec2(0, ImGui::GetContentRegionAvail().y - 6))) {

                ImGui::TableSetupScrollFreeze(0, 1);
                for (int c = 0; c < ncols; c++) {
                    const int gc = col_off + c;        // global col index
                    ImGui::TableSetupColumn(s_->preview.col_names[gc].c_str(),
                        ImGuiTableColumnFlags_WidthFixed, 140.0f);
                }
                ImGui::TableHeadersRow();

                // Type row
                ImGui::TableNextRow();
                for (int c = 0; c < ncols; c++) {
                    const int gc = col_off + c;
                    ImGui::TableSetColumnIndex(c);
                    ImGui::TextColored(ImVec4(0.65f, 0.85f, 0.65f, 1.0f),
                        "%s", s_->preview.col_types[gc].c_str());
                }

                // Data rows — index into the same global slice.
                for (const auto& row : s_->preview.rows) {
                    ImGui::TableNextRow();
                    for (int c = 0; c < ncols; c++) {
                        const int gc = col_off + c;
                        if (gc >= (int)row.size()) break;
                        ImGui::TableSetColumnIndex(c);
                        ImGui::TextUnformatted(row[gc].c_str());
                    }
                }

                ImGui::EndTable();
            }
        }
    }

    ImGui::EndChild();

    ImGui::End();
}
