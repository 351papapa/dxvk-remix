//!feature wetness ENABLE_SURFACE_WETNESS
//!feature particle_sky ENABLE_PARTICLE_SKY_HEMISPHERE
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

//!variant gbuffer_psr_prepare.comp

/* rtx.gbuffer.slimVariants (levels in gbuffer_rayquery_slim.comp.slang). The lean levels bind these too, because
   GBUFFER_SLIM_NO_SSS compiles to the same binary here, where the SSS decode is already dead code. */

//!variant gbuffer_psr_prepare_slim.comp
//!>       GBUFFER_SLIM_NO_POM
//!>       GBUFFER_SLIM_NO_THIN_FILM
//!>       GBUFFER_SLIM_NO_PORTALS

//!variant gbuffer_psr_prepare_slimpom.comp
//!>       GBUFFER_SLIM_NO_THIN_FILM
//!>       GBUFFER_SLIM_NO_PORTALS

//!end-variants

#define RAY_TRACING_PRIMARY_RAY
#define RESOLVE_OPACITY_LIGHTING_APPROXIMATION 1
#ifndef GBUFFER_SLIM_NO_THIN_FILM
#define OPAQUE_MATERIAL_USE_THIN_FILM 1
#endif
#ifndef GBUFFER_SLIM_NO_POM
#define OPAQUE_MATERIAL_USE_POM 1
#endif
#ifdef GBUFFER_SLIM_NO_PORTALS
#define SURFACE_MATERIAL_RESOLVE_TYPE_ACTIVE_MASK SURFACE_MATERIAL_RESOLVE_TYPE_OPAQUE_TRANSLUCENT
#endif

#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/gbuffer/gbuffer_binding_indices.h"
#include "rtx/pass/gbuffer/gbuffer_bindings.slangh"
#include "rtx/algorithm/geometry_resolver_state.slangh"

#include "rtx/concept/camera/camera.slangh"
#include "rtx/algorithm/geometry_resolver.slangh"

// rtx.gbuffer.compactPsr. Appends this pixel to the lists of the PSR passes it runs (passMask bit 0
// Reflection, bit 1 Transmission), one atomic per wave and list. Reached by every lane of the wave (uniform control flow).
// The first lane also raises the list's thread group count to cover the entries appended so far. A list cannot hold more
// than one entry per pixel and the host sizes it for the whole frame, so it does not fill up. If it ever did, the entry
// would be dropped and the group count raised to a full-screen walk, which the compact PSR kernel then runs instead of
// the list (gbuffer_rayquery.comp.slang), so every PSR pixel is still resolved exactly once.
void compactPsrAppend(uint2 pixelCoordinate, uint passMask)
{
  const uint capacity = push.compactPsrCapacity;
  const uint screenPixels = cb.camera.resolution.x * cb.camera.resolution.y;

  [unroll]
  for (uint list = 0; list < 2; list++)
  {
    const bool append = ((passMask >> list) & 1u) != 0u;
    const uint appendCount = WaveActiveCountBits(append);
    if (appendCount == 0u)
    {
      continue;
    }

    uint base = 0u;
    if (WaveIsFirstLane())
    {
      InterlockedAdd(CompactPsrArgs[GBUFFER_COMPACT_PSR_ARGS_COUNT_OFFSET + list], appendCount, base);
      const uint end = base + appendCount;
      const uint threads = end <= capacity ? end : max(screenPixels, end);
      InterlockedMax(CompactPsrArgs[list * GBUFFER_COMPACT_PSR_ARGS_DWORDS_PER_LIST],
                     (threads + GBUFFER_COMPACT_PSR_GROUP_SIZE - 1u) / GBUFFER_COMPACT_PSR_GROUP_SIZE);
    }
    base = WaveReadLaneFirst(base);

    const uint index = base + WavePrefixCountBits(append);
    if (append && index < capacity)
    {
      CompactPsrList[list * capacity + index] = pixelCoordinate.x | (pixelCoordinate.y << 16);
    }
  }
}

[shader("compute")]
[numthreads(16, 8, 1)]
void main(uint2 dispatchThreadId : SV_DispatchThreadID)
{
  const bool inside = all(dispatchThreadId < cb.camera.resolution);
  uint passMask = 0u;
  if (inside)
  {
    passMask = geometryPSRPreparePass(dispatchThreadId);
  }

  if ((push.compactPsrFlags & GBUFFER_COMPACT_PSR_FLAG_APPEND) != 0u)
  {
    compactPsrAppend(dispatchThreadId, passMask);
  }
}
