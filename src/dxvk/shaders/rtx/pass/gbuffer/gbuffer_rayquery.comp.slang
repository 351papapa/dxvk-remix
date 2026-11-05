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

#include "rtx/concept/surface_material/surface_material_hitgroup.h"

//!variant-matrix gbuffer_rayquery
//!> common RAY_TRACING_PRIMARY_RAY ATMOSPHERE_AVAILABLE
//!> axis features no_psr GBUFFER_FEATURE_DLSSRR=1 GBUFFER_FEATURE_OBJECT_PICKING=0 GBUFFER_FEATURE_RAYTRACED_RENDER_TARGET=0 GBUFFER_FEATURE_STOCHASTIC_ALPHA=0 GBUFFER_FEATURE_PSR_REPLACEMENT=0 GBUFFER_FEATURE_INLINE_PSR_SAMPLING=0
//!> axis features no_psr_raytraced_render_target GBUFFER_FEATURE_DLSSRR=1 GBUFFER_FEATURE_OBJECT_PICKING=0 GBUFFER_FEATURE_RAYTRACED_RENDER_TARGET=1 GBUFFER_FEATURE_STOCHASTIC_ALPHA=0 GBUFFER_FEATURE_PSR_REPLACEMENT=0 GBUFFER_FEATURE_INLINE_PSR_SAMPLING=0
//!> axis features no_psr_stochastic_alpha GBUFFER_FEATURE_DLSSRR=1 GBUFFER_FEATURE_OBJECT_PICKING=0 GBUFFER_FEATURE_RAYTRACED_RENDER_TARGET=0 GBUFFER_FEATURE_STOCHASTIC_ALPHA=1 GBUFFER_FEATURE_PSR_REPLACEMENT=0 GBUFFER_FEATURE_INLINE_PSR_SAMPLING=0
//!> axis features no_psr_raytraced_render_target_stochastic_alpha GBUFFER_FEATURE_DLSSRR=1 GBUFFER_FEATURE_OBJECT_PICKING=0 GBUFFER_FEATURE_RAYTRACED_RENDER_TARGET=1 GBUFFER_FEATURE_STOCHASTIC_ALPHA=1 GBUFFER_FEATURE_PSR_REPLACEMENT=0 GBUFFER_FEATURE_INLINE_PSR_SAMPLING=0
//!> axis features psr_inline GBUFFER_FEATURE_DLSSRR=1 GBUFFER_FEATURE_OBJECT_PICKING=0 GBUFFER_FEATURE_RAYTRACED_RENDER_TARGET=0 GBUFFER_FEATURE_STOCHASTIC_ALPHA=0 GBUFFER_FEATURE_PSR_REPLACEMENT=1 GBUFFER_FEATURE_INLINE_PSR_SAMPLING=1
//!> axis features psr_inline_raytraced_render_target GBUFFER_FEATURE_DLSSRR=1 GBUFFER_FEATURE_OBJECT_PICKING=0 GBUFFER_FEATURE_RAYTRACED_RENDER_TARGET=1 GBUFFER_FEATURE_STOCHASTIC_ALPHA=0 GBUFFER_FEATURE_PSR_REPLACEMENT=1 GBUFFER_FEATURE_INLINE_PSR_SAMPLING=1
//!> axis features psr_inline_stochastic_alpha GBUFFER_FEATURE_DLSSRR=1 GBUFFER_FEATURE_OBJECT_PICKING=0 GBUFFER_FEATURE_RAYTRACED_RENDER_TARGET=0 GBUFFER_FEATURE_STOCHASTIC_ALPHA=1 GBUFFER_FEATURE_PSR_REPLACEMENT=1 GBUFFER_FEATURE_INLINE_PSR_SAMPLING=1
//!> axis features psr_inline_raytraced_render_target_stochastic_alpha GBUFFER_FEATURE_DLSSRR=1 GBUFFER_FEATURE_OBJECT_PICKING=0 GBUFFER_FEATURE_RAYTRACED_RENDER_TARGET=1 GBUFFER_FEATURE_STOCHASTIC_ALPHA=1 GBUFFER_FEATURE_PSR_REPLACEMENT=1 GBUFFER_FEATURE_INLINE_PSR_SAMPLING=1
//!> axis features psr_prepare GBUFFER_FEATURE_DLSSRR=1 GBUFFER_FEATURE_OBJECT_PICKING=0 GBUFFER_FEATURE_RAYTRACED_RENDER_TARGET=0 GBUFFER_FEATURE_STOCHASTIC_ALPHA=0 GBUFFER_FEATURE_PSR_REPLACEMENT=1 GBUFFER_FEATURE_INLINE_PSR_SAMPLING=0
//!> axis features psr_prepare_stochastic_alpha GBUFFER_FEATURE_DLSSRR=1 GBUFFER_FEATURE_OBJECT_PICKING=0 GBUFFER_FEATURE_RAYTRACED_RENDER_TARGET=0 GBUFFER_FEATURE_STOCHASTIC_ALPHA=1 GBUFFER_FEATURE_PSR_REPLACEMENT=1 GBUFFER_FEATURE_INLINE_PSR_SAMPLING=0
//!> axis features debug GBUFFER_FEATURE_DEBUG_VIEW=1 GBUFFER_FEATURE_OBJECT_PICKING=1
//!> axis pass -
//!> axis pass psr GBUFFER_PSR
//!> axis nrc -
//!> axis nrc nrc ENABLE_NRC=1
//!> axis wboit -
//!> axis wboit wboit UNORDERED_RESOLVE_WBOIT
//!> order features pass nrc wboit
//!end-variants

// Note: Opacity lighting approximation used in the GBuffer pass due to it being most visually
// prominent on particles in primary rays and important in PSR.
#define RESOLVE_OPACITY_LIGHTING_APPROXIMATION 1
// rtx.gbuffer.slimVariants. gbuffer_rayquery_slim.comp.slang includes this file with GBUFFER_SLIM_* defines,
// and the host binds those variants only while the compiled-out code cannot run in the frame.
// Note: Use thin film in the GBuffer pass for its usage in PSR on mirrorlike surfaces.
#ifndef GBUFFER_SLIM_NO_THIN_FILM
#define OPAQUE_MATERIAL_USE_THIN_FILM 1
#endif
#ifndef GBUFFER_SLIM_NO_POM
#define OPAQUE_MATERIAL_USE_POM 1
#endif
#ifdef GBUFFER_SLIM_NO_PORTALS
#define SURFACE_MATERIAL_RESOLVE_TYPE_ACTIVE_MASK SURFACE_MATERIAL_RESOLVE_TYPE_OPAQUE_TRANSLUCENT
#endif

#ifdef GBUFFER_PSR
// Note: This is done in primary rays only currently so that attenuation from volumetrics can be factored in
// similar to how it was done previously when it was simply applied in compositing via the total virtual hit
// distance, and ensuring that it does not attenuate secondary rays (since they do not receive inscattering and
// attenuating them would cause large amounts of energy loss as a result).
#define RESOLVER_USE_VOLUMETRIC_ATTENUATION 1
#endif

#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/gbuffer/gbuffer_binding_indices.h"
#include "rtx/pass/gbuffer/gbuffer_bindings.slangh"
#include "rtx/algorithm/geometry_resolver_state.slangh"

#include "rtx/concept/camera/camera.slangh"
#include "rtx/algorithm/geometry_resolver.slangh"

#if defined(GBUFFER_COMPACT_PSR) && defined(GBUFFER_PSR)
// rtx.gbuffer.compactPsr (variants in gbuffer_rayquery_compact.comp.slang). Dispatched indirectly with one thread per
// entry of this pass's list (list 0 Reflection, list 1 Transmission), which PSR Prepare writes. The thread takes its
// pixel from the list, and getDispatchRaysIndex() returns that pixel (rtx/utility/common.slangh), so the
// resolver, its RNG (seeded by pixel and frame) and every pixel-addressed read and write are those of the full-screen
// kernel's thread for the same pixel. The per-pixel early-outs in geometryPSRResolverPass stay as a guard.
[shader("compute")]
[numthreads(GBUFFER_COMPACT_PSR_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
  const uint list = push.isTransmissionPSR != 0 ? 1u : 0u;
  const uint count = CompactPsrArgs[GBUFFER_COMPACT_PSR_ARGS_COUNT_OFFSET + list];
  const uint thread = dispatchThreadId.x;
  uint2 pixelCoordinate;
  if (count <= push.compactPsrCapacity)
  {
    if (thread >= count)
    {
      return;
    }
    const uint packed = CompactPsrList[list * push.compactPsrCapacity + thread];
    pixelCoordinate = uint2(packed & 0xffffu, packed >> 16);
  }
  else
  {
    // The list overflowed (see compactPsrAppend), so this walks the whole screen and the early-outs select the PSR pixels.
    const uint width = cb.camera.resolution.x;
    if (thread >= width * cb.camera.resolution.y)
    {
      return;
    }
    pixelCoordinate = uint2(thread % width, thread / width);
  }

  g_compactPsrPixel = pixelCoordinate;
  geometryPSRResolverPass(pixelCoordinate);
}
#else
[shader("compute")]
[numthreads(16, 8, 1)]
void main(uint2 dispatchThreadId : SV_DispatchThreadID)
{
  if (any(dispatchThreadId >= cb.camera.resolution))
  {
    return;
  }

#ifdef GBUFFER_PSR
  geometryPSRResolverPass(dispatchThreadId);
#else
  geometryResolver(dispatchThreadId);
#endif
}
#endif
