#pragma once

#include <memory>

struct GLFWwindow;

class UCDHWorkbenchApplet {
public:
    UCDHWorkbenchApplet();
    ~UCDHWorkbenchApplet();

    bool initialize();
    void draw_ui(int win_w, int win_h);
    void cleanup();

    bool should_exit()      const { return exit_requested_; }
    void reset_exit_flag()        { exit_requested_ = false; }

private:
    void draw_panel();
    void draw_leads();
    void draw_vcg_3d();
    void draw_raw_browser();
    void draw_model_tab();
    void draw_model_architecture();
    void draw_model_inference();

    void open_dataset(const std::string& dir);
    void select_sample(int idx);
    void on_params_changed();
    void ensure_vcg_cached();

    struct State;
    std::unique_ptr<State> s_;
    bool exit_requested_ = false;
};
