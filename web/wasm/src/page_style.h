// page_style.h: the colours of the views on a web page, after the native
// Sternbild style (sternbild::apply_style) has set fonts, metrics and the rest.
#pragma once

namespace caliper_web {

// The view windows take the plate colour of the page's own views, and ImPlot
// takes the page palette (the series line in parchment, hover marks in bronze,
// rules and tick labels as the TypeScript views draw them), so the two
// renderers of the <caliper-view> element look alike. `bare` (the element
// draws the header) also drops the window's border and most of its padding.
void apply_page_style(bool bare);

}  // namespace caliper_web
