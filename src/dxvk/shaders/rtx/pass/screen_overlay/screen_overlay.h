/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
#ifndef SCREEN_OVERLAY_H
#define SCREEN_OVERLAY_H

#include "rtx/utility/shader_types.h"

#define SCREEN_OVERLAY_INPUT_OUTPUT  0
#define SCREEN_OVERLAY_TEXTURE       1

#define SCREEN_OVERLAY_TILE_SIZE 8

// Overlay RGB is already multiplied by its alpha (a host render target
// cleared to transparent black and drawn with SrcAlpha/InvSrcAlpha).
#define SCREEN_OVERLAY_FLAG_PREMULTIPLIED 0x1u

struct ScreenOverlayArgs {
  uint2 imageSize;
  float opacity;
  uint flags;       // SCREEN_OVERLAY_FLAG_*
  // Destination rectangle as fractions of the output (x, y, width, height).
  float rectX;
  float rectY;
  float rectW;
  float rectH;
  // > 0: overlay pixels whose sampled alpha is <= this are dropped (the output
  // keeps its colour), matching a host-side alpha cutoff. 0 = off.
  float alphaCutoff;
};

#endif // SCREEN_OVERLAY_H
