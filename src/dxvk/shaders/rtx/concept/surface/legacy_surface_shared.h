/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#pragma once

// The layout of the 80 B per-surface legacy (fixed-function) side record ("Legacy Surface Buffer",
// rtx.coreMaterial.legacyRecord), parallel to the surface index. Every field is a verbatim
// copy of the words the Surface record (surface.h) carries, so both decode alike. The host writer
// (rtx_fork_core_material.cpp) and the Slang LegacySurface module (legacy_surface.slangh) share it.

#define LEGACY_SURFACE_SIZE                           80
#define LEGACY_SURFACE_ROWS                           5  // uint4 rows

#define LEGACY_SURFACE_OFFSET_TRANSFORM               0  // 32 B: texture transform rows, or eye basis U / V (Surface data11-data12)
#define LEGACY_SURFACE_OFFSET_CLIP_PLANE              32 // 16 B: Surface data14
#define LEGACY_SURFACE_OFFSET_TFACTOR                 48 // u32: Surface data13.y
#define LEGACY_SURFACE_OFFSET_TEXTURE_FLAGS           52 // u32: Surface data13.z bits 0-18 verbatim, plus bits 19-22 below
#define LEGACY_SURFACE_OFFSET_SPRITESHEET             56 // u8 rows, u8 cols, u8 fps (Surface data13.x bits 0-23); byte 59 zero
#define LEGACY_SURFACE_OFFSET_SECONDARY_TEXTURE       60 // u16: the opaque material's secondary texture (0xFFFF = none); 62-63 zero
#define LEGACY_SURFACE_OFFSET_EYE_ORIGIN              64 // 12 B: Surface data15.xyz; 76-79 zero

// Texture flags dword: texture-stage ops (bits 0-13), isEye (14) and texgen (17-18) as Surface data13.z.
#define LEGACY_SURFACE_TEXTURE_FLAGS_SURFACE_MASK     (0x7FFFu | (0x3u << 17))
#define LEGACY_SURFACE_FLAG_IS_EYE                    (1u << 14)
#define LEGACY_SURFACE_TEXGEN_SHIFT                   17
#define LEGACY_SURFACE_TEXGEN_MASK                    0x3u
// Flags taken from other Surface words.
#define LEGACY_SURFACE_FLAG_TEXTURE_FACTOR_BLEND      (1u << 19) // Surface data2.w bit 27
#define LEGACY_SURFACE_FLAG_VERTEX_COLOR_BAKED_LIGHTING (1u << 20) // Surface data0b.z bit 1
#define LEGACY_SURFACE_FLAG_CLIP_PLANE                (1u << 21) // Surface data2.w bit 25
#define LEGACY_SURFACE_FLAG_SKIP_SPRITESHEET          (1u << 22) // Surface data2.w bit 29
