// uniform_blocks.h: the uniform blocks the scene core fills (Frame, PrimParams, Cmap,
// ClampParams, GatherParams) and the programs it draws with, as the shader generator emits them
// from the WGSL sources in src/host/gfx/shaders/. The scene core includes this one file, so the
// generated headers are its single source for the block fields and the program descriptions.
#pragma once

#include "gfx/shaders/generated/programs.h"
#include "gfx/shaders/generated/uniforms.h"
