// lib module pick@1 (design/programmable-graphics.md, Lib Modules): the ID
// that a pick pass writes and the linked selection. A piece that imports it
// calls pick::id, pick::selected and pick::selected_index. Released; never
// changes.

// The ID that a pass with pick = true writes at location 1: the program's
// pick id and the item, an element index of the item map's resource.
fn pick__id(item: u32) -> vec2<u32> {
    return vec2<u32>(caliper_view.frame.z, item);
}

// The count of valid entries of caliper_selection.
fn pick__selection_count() -> u32 {
    return min(caliper_view.frame.w, min(arrayLength(&caliper_selection), 256u));
}

// Whether an item of this program is in the linked selection.
fn pick__selected(item: u32) -> bool {
    let n = pick__selection_count();
    for (var i = 0u; i < n; i = i + 1u) {
        let s = caliper_selection[i];
        if (s.pick == caliper_view.frame.z && s.item == item) {
            return true;
        }
    }
    return false;
}

// Whether a selected item of any program has these tensor indices; an index
// below 0 matches every index.
fn pick__selected_index(index: vec4<i32>) -> bool {
    let n = pick__selection_count();
    for (var i = 0u; i < n; i = i + 1u) {
        let s = caliper_selection[i].index;
        if (all((index == s) | (index < vec4<i32>(0)))) {
            return true;
        }
    }
    return false;
}
