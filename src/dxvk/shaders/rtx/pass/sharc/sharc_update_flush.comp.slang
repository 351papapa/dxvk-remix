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

// rtx.sharc.deferredFlush. One thread per update tile runs the reverse propagation, deposit ceiling and
// SharcAddVoxelData over the vertex record the deferred update variants leave (layout in sharc_update.slangh).
// It runs between the update and the resolve. The slot addressing does not depend on the propagation depth,
// so the depth-8 loop serves the depth-4 update as well, and slots at or above the count are never read.

//!variant sharc_update_flush.comp
//!>       SHARC_PROPAGATION_DEPTH=8

//!end-variants

#define ENABLE_SHARC 1
#define SHARC_UPDATE 1
#define SHARC_DEFERRED_FLUSH 1

#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/sharc/sharc_bindings.slangh"
#include "rtx/pass/sharc/sharc_update.slangh"

[shader("compute")]
[numthreads(16, 8, 1)]
void main(uint2 threadIndex : SV_DispatchThreadID) {
  const uvec2 tileCount = sharcDeferredTileCount();
  if (any(threadIndex >= tileCount)) {
    return;
  }

  const uint tile = threadIndex.y * tileCount.x + threadIndex.x;
  const uint countAddress = sharcDeferredAddress(tile, 0);
  const uint count = SharcDeferredVertices[countAddress];
  if (count == 0) {
    return;
  }

  // The resolved buffer the ceiling reads still holds last frame's resolve, because the update never
  // writes it and the resolve runs after this pass. The ceiling sees what the inline flush would see.
  sharcDepositTileVertices(tile, min(count, uint(SHARC_PROPAGATION_DEPTH)));
  // A tile whose next path stores nothing (a G-buffer miss or no eligible vertex) must read 0.
  SharcDeferredVertices[countAddress] = 0;
}
