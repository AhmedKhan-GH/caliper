#pragma once
// ============================================================================
// Docking by role on the applet page (caliper.ui.v1_1 dock_hint, see
// docs/references/service-ui-v1-1.md). The host never names an applet window:
// an applet declares the role of each window beside its Begin(), and this
// unit places each window, once, the first time it appears, in the node for
// its role:
//
//   CALIPER_DOCK_CENTRAL       the central node (also every unknown role)
//   CALIPER_DOCK_SIDE_TOP      the side column, upper part
//   CALIPER_DOCK_SIDE_BOTTOM   the side column, lower part
//
// A window the applet gives no hint is placed in the centre. Only a window
// with no saved layout (no imgui.ini entry) is placed, so a layout the user
// arranged always wins, and a window is considered once per process, so
// moving it afterwards is never undone.
//
// The side column is split off the central node when a window first needs
// it, and on a dock space with no saved layout at all it is built before the
// first frame of the applet page, so the first windows dock in the frame
// they first appear. Otherwise a hint that arrives after its window's first
// Begin() takes effect on the next frame.
//
// Header-only and free of any renderer: the unit test drives it on a
// headless ImGui context.
// ============================================================================
#include <caliper/services/ui_v1_1.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace caliper_host {

class DockLayout {
public:
    // The side column takes this share of a dock space that has no saved
    // layout (the share the column has always had on a first run), and
    // kSideShareOfCentral of the central node when it is added to a restored
    // layout. The lower part takes kSideBottomShare of the column.
    static constexpr float kSideShareFresh = 0.204f;
    static constexpr float kSideShareOfCentral = 0.30f;
    static constexpr float kSideBottomShare = 0.40f;

    // caliper.ui.v1_1 dock_hint, on the UI thread. The first hint for a title
    // counts; every later one returns after a hash lookup.
    void hint(const char* window_title, uint32_t role) {
        if (!window_title || !*window_title || !ImGui::GetCurrentContext()) return;
        const ImGuiID id = ImHashStr(window_title);
        if (!known_.emplace(id, role).second) return;
        place(window_title, id, normal_role(role));
    }

    // Once per frame on the applet page, after DockSpaceOverViewport() and
    // before the applet's frame: places the windows that could not be placed
    // when their hint arrived.
    void apply(ImGuiID dockspace_id) {
        ImGuiDockNode* root = ImGui::DockBuilderGetNode(dockspace_id);
        if (!root) return;
        if (dockspace_id != dockspace_) {
            dockspace_ = dockspace_id;
            side_top_ = side_bottom_ = 0;
            if (root->IsEmpty()) {
                ImGui::DockBuilderSetNodeSize(dockspace_id,
                                              ImGui::GetMainViewport()->WorkSize);
                build_side(kSideShareFresh);
                ImGui::DockBuilderFinish(dockspace_id);
            }
        }
        if (pending_.empty()) return;
        for (const Pending& p : pending_)
            ImGui::DockBuilderDockWindow(p.title.c_str(), node_for(p.role, true));
        ImGui::DockBuilderFinish(dockspace_id);
        pending_.clear();
    }

    // Bracket the applet's frame. Every window the applet began in between
    // without a hint is placed as CALIPER_DOCK_CENTRAL. Child windows, popups,
    // tooltips and windows that opt out of docking or of saved settings are
    // left alone.
    void begin_applet_frame() {
        first_begin_ = ImGui::GetCurrentContext()
                           ? ImGui::GetCurrentContext()->WindowsActiveCount
                           : 0;
    }
    void end_applet_frame() {
        ImGuiContext* g = ImGui::GetCurrentContext();
        if (!g) return;
        constexpr ImGuiWindowFlags kSkip =
            ImGuiWindowFlags_ChildWindow | ImGuiWindowFlags_Popup |
            ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_NoDocking |
            ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_DockNodeHost;
        for (ImGuiWindow* w : g->Windows) {
            if (!w->Active || w->IsFallbackWindow || (w->Flags & kSkip) ||
                w->BeginOrderWithinContext < first_begin_)
                continue;
            if (!known_.emplace(w->ID, CALIPER_DOCK_CENTRAL).second) continue;
            place(w->Name, w->ID, CALIPER_DOCK_CENTRAL);
        }
    }

    // The node a window of `role` docks into now, or 0 when the side column
    // does not exist yet. For tests and diagnostics.
    ImGuiID node_for_role(uint32_t role) { return node_for(normal_role(role), false); }

private:
    struct Pending {
        std::string title;
        uint32_t role;
    };

    static uint32_t normal_role(uint32_t role) {
        return role == CALIPER_DOCK_SIDE_TOP || role == CALIPER_DOCK_SIDE_BOTTOM
                   ? role
                   : (uint32_t)CALIPER_DOCK_CENTRAL;
    }

    // Docks a window met for the first time, unless it has a saved layout or
    // is docked already. A window that does not exist yet is docked through
    // its settings, so its first Begin() creates it in place; the rest wait
    // for apply().
    void place(const char* title, ImGuiID id, uint32_t role) {
        if (ImGui::FindWindowSettingsByID(id)) return;   // the user's layout wins
        ImGuiWindow* w = ImGui::FindWindowByID(id);
        if (w && w->DockId != 0) return;
        if (!w && dockspace_ != 0) {
            if (const ImGuiID node = node_for(role, false)) {
                ImGui::DockBuilderDockWindow(title, node);
                return;
            }
        }
        pending_.push_back({title, role});
    }

    // A leaf node of this dock space, still alive.
    bool alive(ImGuiID id) const {
        if (!id) return false;
        ImGuiDockNode* n = ImGui::DockBuilderGetNode(id);
        if (!n || !n->IsLeafNode()) return false;
        while (n->ParentNode) n = n->ParentNode;
        return n->ID == dockspace_;
    }

    ImGuiID central() const {
        ImGuiDockNode* c = ImGui::DockBuilderGetCentralNode(dockspace_);
        return c ? c->ID : dockspace_;
    }

    ImGuiID node_for(uint32_t role, bool create) {
        if (role == CALIPER_DOCK_CENTRAL) return central();
        if (!alive(side_top_) || !alive(side_bottom_)) {
            if (!create) return 0;
            build_side(kSideShareOfCentral);
        }
        return role == CALIPER_DOCK_SIDE_TOP ? side_top_ : side_bottom_;
    }

    // Splits the side column off the central node, or restores the half of
    // it that the user emptied (an emptied node removes itself).
    void build_side(float share) {
        const bool top = alive(side_top_), bottom = alive(side_bottom_);
        if (top && bottom) return;
        if (top) {
            side_bottom_ = ImGui::DockBuilderSplitNode(side_top_, ImGuiDir_Down,
                                                       kSideBottomShare, nullptr,
                                                       &side_top_);
        } else if (bottom) {
            side_top_ = ImGui::DockBuilderSplitNode(side_bottom_, ImGuiDir_Up,
                                                    1.0f - kSideBottomShare,
                                                    nullptr, &side_bottom_);
        } else {
            ImGuiID centre = central();
            const ImGuiID side = ImGui::DockBuilderSplitNode(
                centre, ImGuiDir_Right, share, nullptr, &centre);
            side_bottom_ = ImGui::DockBuilderSplitNode(side, ImGuiDir_Down,
                                                       kSideBottomShare, nullptr,
                                                       &side_top_);
        }
    }

    std::unordered_map<ImGuiID, uint32_t> known_;   // window id -> first role
    std::vector<Pending> pending_;
    ImGuiID dockspace_ = 0;
    ImGuiID side_top_ = 0;
    ImGuiID side_bottom_ = 0;
    int first_begin_ = 0;
};

} // namespace caliper_host
