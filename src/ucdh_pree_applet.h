#pragma once

// ============================================================================
// UCDH PreE — Preliminary Exploration applet.
//
// A basic dataset explorer powered by DuckDB. Lets you point at a folder,
// see the supported data files inside (CSV / Parquet / JSON), pick one,
// and inspect its schema + a row preview. Designed to grow into a richer
// interactive EDA surface (filters, plots, ad-hoc SQL, joins) over time.
//
// Frame lifecycle (driven by main.cpp):
//   initialize()            spins up an in-memory DuckDB connection
//   draw_ui(win_w, win_h)   full-window layout + ImGuiFileDialog
//   cleanup()               tears down the DuckDB connection
// ============================================================================

class UCDHPreEApplet {
public:
    bool initialize();
    void draw_ui(int win_w, int win_h);
    void cleanup();

    bool should_exit()      const { return exit_requested_; }
    void reset_exit_flag()        { exit_requested_ = false; }

private:
    struct State;
    State* s_ = nullptr;
    bool   exit_requested_ = false;
};
