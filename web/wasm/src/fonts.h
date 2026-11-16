// fonts.h: the faces embedded in the module (fonts.cpp).
#pragma once

#include <cstddef>

namespace caliper_web {

struct EmbeddedFont {
    const unsigned char* data;
    size_t size;
};

EmbeddedFont ui_font();     // IBM Plex Sans Condensed Regular
EmbeddedFont mono_font();   // Source Code Pro Regular

}  // namespace caliper_web
