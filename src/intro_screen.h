#pragma once

#include <string>
#include <vector>

struct GLFWwindow;

struct AppletCard {
    std::string name;
    std::string tagline;
    std::string description;
    std::string tag;
};

class IntroScreen {
public:
    bool initialize();
    void update(GLFWwindow* window);
    void render_3d(int fb_w, int fb_h);
    void draw_ui(int win_w, int win_h);
    void cleanup();

    void set_applets(std::vector<AppletCard> cards);

    bool should_launch()     const { return launch_requested_; }
    void reset_launch_flag()       { launch_requested_ = false; }
    int  selected_index()    const;

private:
    struct State;
    State* s_ = nullptr;

    bool launch_requested_ = false;
};
