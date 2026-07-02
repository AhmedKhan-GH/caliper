#pragma once

#include <memory>

#include <caliper/caliper.hpp>

struct GLFWwindow;

class RepNetDemoApplet {
public:
    RepNetDemoApplet();
    ~RepNetDemoApplet();

    bool initialize(caliper::Bridge bridge);
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
    void draw_activation_detail();
    void draw_weight_view();
    void draw_statistics_tab();

    void open_dataset(const std::string& dir);
    void select_sample(int idx);
    void on_params_changed();
    void ensure_vcg_cached();

    struct State;
    std::unique_ptr<State> s_;
    caliper::Bridge bridge_;   // caliper.tensor_bridge.v1 (viz heatmaps)
    bool exit_requested_ = false;
};
