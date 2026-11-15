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
// Windows of one role share one node, as tabs. The node of a side role is the
// one this unit made for it, or else the node where a window of that role is
// docked already, such as one restored from imgui.ini; only when neither
// exists is a side column split off the central node.
//
// A window the applet gives no hint is placed in the centre when it is an
// ordinary window: not a child window, popup, tooltip or modal, and without
// ImGuiWindowFlags_NoDocking, NoSavedSettings, NoMove, NoResize, NoTitleBar
// or AlwaysAutoResize. Any other window keeps the placement ImGui gives it.
// Only a window with no saved layout (no imgui.ini entry) is placed, so a
// layout the user arranged always wins, and a window is considered once per
// process, so moving it afterwards is never undone. Nothing is written to
// imgui.ini for a window with NoSavedSettings.
//
// On a dock space with no saved layout at all the side column is built
// before the first frame of the applet page, so the first windows dock in
// the frame they first appear. Otherwise a window that needs a new node, or
// whose hint arrives after its first Begin(), is placed on the next frame.
//
// Header-only and free of any renderer: the unit test drives it on a
// headless ImGui context.
// ============================================================================
#include <caliper/services/ui_v1_1.h>
#include "ui/work_area.h"
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

    // The flags that keep a window without a hint where ImGui puts it.
    static constexpr ImGuiWindowFlags kNotOrdinary =
        ImGuiWindowFlags_ChildWindow | ImGuiWindowFlags_Popup |
        ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_Modal |
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_DockNodeHost;

    // caliper.ui.v1_1 dock_hint, on the UI thread. The first hint for a title
    // counts; every later one returns after a hash lookup.
    void hint(const char* window_title, uint32_t role) {
        if (!window_title || !*window_title || !ImGui::GetCurrentContext()) return;
        const ImGuiID id = ImHashStr(window_title);
        role = normal_role(role);
        if (!known_.emplace(id, role).second) return;
        place(window_title, id, role);
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
                // The work area this frame's bars leave, which the dock
                // space fills (work_area.h).
                ImGui::DockBuilderSetNodeSize(dockspace_id, work_area().GetSize());
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

    // Bracket the applet's frame. Every ordinary window (see the header) the
    // applet began in between without a hint is placed as
    // CALIPER_DOCK_CENTRAL. A hinted window that turned out to have
    // NoSavedSettings loses the imgui.ini entry its hint made, and is docked
    // on the next frame instead.
    void begin_applet_frame() {
        first_begin_ = ImGui::GetCurrentContext()
                           ? ImGui::GetCurrentContext()->WindowsActiveCount
                           : 0;
    }
    void end_applet_frame() {
        ImGuiContext* g = ImGui::GetCurrentContext();
        if (!g) return;
        for (size_t i = 0; i < via_settings_.size();) {
            ImGuiWindow* w = ImGui::FindWindowByID(via_settings_[i]);
            if (!w) {   // not begun yet: look again after a later frame
                i++;
                continue;
            }
            if (w->Flags & ImGuiWindowFlags_NoSavedSettings) {
                if (ImGuiWindowSettings* s = ImGui::FindWindowSettingsByID(w->ID))
                    s->WantDelete = true;
                pending_.push_back({w->Name, known_[w->ID]});
            }
            via_settings_.erase(via_settings_.begin() + (std::ptrdiff_t)i);
        }
        for (ImGuiWindow* w : g->Windows) {
            if (!w->Active || w->IsFallbackWindow || (w->Flags & kNotOrdinary) ||
                w->BeginOrderWithinContext < first_begin_)
                continue;
            if (!known_.emplace(w->ID, (uint32_t)CALIPER_DOCK_CENTRAL).second) continue;
            place(w->Name, w->ID, CALIPER_DOCK_CENTRAL);
        }
    }

    // The node a window of `role` docks into now, or 0 when there is none
    // yet. For tests and diagnostics.
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
                via_settings_.push_back(id);
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

    // The node of a window of `role` that is docked in this dock space now,
    // outside the central node (a side window the user moved to the centre
    // does not take the side role's node with it), or 0.
    ImGuiID docked_node_of(uint32_t role) const {
        const ImGuiID centre = central();
        for (const auto& [id, r] : known_) {
            if (r != role) continue;
            if (ImGuiWindow* w = ImGui::FindWindowByID(id);
                w && w->DockId != centre && alive(w->DockId))
                return w->DockId;
        }
        return 0;
    }

    // The node made for a side role if it is still there, else the node of a
    // window of that role that is docked already.
    void adopt(uint32_t role, ImGuiID& node) const {
        if (!alive(node)) node = docked_node_of(role);
    }

    ImGuiID node_for(uint32_t role, bool create) {
        if (role == CALIPER_DOCK_CENTRAL) return central();
        adopt(CALIPER_DOCK_SIDE_TOP, side_top_);
        adopt(CALIPER_DOCK_SIDE_BOTTOM, side_bottom_);
        ImGuiID& node = role == CALIPER_DOCK_SIDE_TOP ? side_top_ : side_bottom_;
        if (!alive(node)) {
            if (!create) return 0;
            build_side(kSideShareOfCentral);
        }
        return node;
    }

    // Splits the side column off the central node, or the missing half of
    // it off the half that exists (an emptied node removes itself).
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

    std::unordered_map<ImGuiID, uint32_t> known_;   // window id -> its role
    std::vector<Pending> pending_;
    std::vector<ImGuiID> via_settings_;   // docked through settings, not yet begun
    ImGuiID dockspace_ = 0;
    ImGuiID side_top_ = 0;
    ImGuiID side_bottom_ = 0;
    int first_begin_ = 0;
};

} // namespace caliper_host
