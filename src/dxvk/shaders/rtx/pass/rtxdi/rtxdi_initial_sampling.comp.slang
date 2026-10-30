//!feature wetness ENABLE_SURFACE_WETNESS
/*
* Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
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

//!variant rtxdi_initial_sampling.comp
//!>       RTXDI_NO_PORTALS=0

//!variant rtxdi_initial_sampling_no_portals.comp
//!>       RTXDI_NO_PORTALS=1

//!variant rtxdi_initial_sampling_local_spheres.comp
//!>       RTXDI_NO_PORTALS=0 ENABLE_LOCAL_SPHERES=1

//!variant rtxdi_initial_sampling_no_portals_local_spheres.comp
//!>       RTXDI_NO_PORTALS=1 ENABLE_LOCAL_SPHERES=1

//!end-variants

#ifndef ENABLE_LOCAL_SPHERES
#define ENABLE_LOCAL_SPHERES 0
#endif

#define USE_32BIT_RAY_DIRECTION 1
#define RAB_HAS_CURRENT_GBUFFER 1
#define RAB_HAS_RTXDI_RESERVOIRS 1
#define RTXDI_DEFER_INITIAL_VISIBILITY 1

#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/rtxdi/rtxdi_reuse_bindings.slangh"

#include "rtx/utility/common.slangh"
#include "rtx/utility/noise.slangh"
#include "rtx/utility/math.slangh"
#include "rtx/utility/texture.slangh"
#include "rtx/utility/sampling.slangh"
#include "rtx/utility/packing.slangh"
#include "rtx/concept/ray/ray.slangh"
#include "rtx/concept/surface/surface.slangh"
#include "rtx/concept/surface_material/surface_material.slangh"
#include "rtx/concept/light/light.slangh"
#include "rtx/algorithm/resolve.slangh"
#include "rtx/algorithm/rtxdi/rtxdi.slangh"
#include "rtx/pass/sparse_rendering/compact_active_pixels.slangh"

#if ENABLE_LOCAL_SPHERES
layout(binding = RTXDI_REUSE_BINDING_LOCAL_LIGHTS_INPUT)
StructuredBuffer<uint2> LocalSphereCandidates;

// Streams the nearest sphere candidates while preserving the temporal RNG sequence.
void RAB_SampleLocalLights(inout RTXDI_Reservoir state, inout RAB_RandomSamplerState rng, RAB_Surface surface)
{
  const uint total = min(LocalSphereCandidates[0].x, 64u);
  if (total == 0u)
  {
    return;
  }

  RTXDI_Reservoir localState = RTXDI_EmptyReservoir();
  bool overlapsGlobalSample = false;
  for (uint i = 0; i < total; ++i)
  {
    const uint32_t entry = LocalSphereCandidates[i + 1].y;
    const uint16_t lightIndex = uint16_t(entry >> 3u);
    const uint lightType = entry & 0x7u;

    // Note: always consume the 3 randoms (uv.xy, risRnd) so the RNG stream stays aligned
    // with the skip in rtxdi_temporal_reuse.comp.slang regardless of candidate validity.
    float2 lightUV = float2(RAB_GetNextRandom(rng), RAB_GetNextRandom(rng));
    const float risRnd = RAB_GetNextRandom(rng);

    if (lightIndex == RTXDI_INVALID_LIGHT_INDEX)
    {
      continue;
    }

    overlapsGlobalSample = overlapsGlobalSample ||
      (state.lightIdx == lightIndex && state.portalIdx == RTXDI_INVALID_PORTAL_INDEX);
    RAB_LightSample candidateSample = RAB_GetLightSampleWithTypeHint(lightIndex, lightUV, lightType, surface);
    const float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);

    RTXDI_StreamSample(localState, lightIndex, RTXDI_INVALID_PORTAL_INDEX, lightUV, risRnd, targetPdf, 1.0f);
  }

  // Note: always consume the combine random even when no candidate survived, for RNG alignment.
  const float random = RAB_GetNextRandom(rng);
  if (localState.M == 0u)
  {
    return;
  }

  // One sample per light estimates the sum over distinct direct-light strata.
  RTXDI_FinalizeResampling(localState, 1.0, 1.0);
  localState.M = 1;

  // Both estimators cover these direct lights. Their partition weights sum to one.
  // Portal images are covered only by the global estimator and keep full weight.
  if (overlapsGlobalSample) { state.weightSum *= 0.5f; }
  localState.weightSum *= 0.5f;

  RTXDI_CombineReservoirs(state, localState, random, localState.targetPdf);
}

#endif

// The pass can be launched over the active-pixel list, which counts groups of the consumer group size.
#if COMPACT_ACTIVE_PIXELS_CONSUMER_GROUP_SIZE_X != 16 || COMPACT_ACTIVE_PIXELS_CONSUMER_GROUP_SIZE_Y != 8
#error "The RTXDI passes run 16x8 groups, which must match COMPACT_ACTIVE_PIXELS_CONSUMER_GROUP_SIZE_X / _Y."
#endif

[shader("compute")]
[numthreads(16, 8, 1)]
void main(int2 threadIndex : SV_DispatchThreadID)
{
  uvec2 remapped;
  const bool isActive = loadLaunchPixelCoordinate(uint2(threadIndex), ActiveLocalPixelCoords, ActivePixelList,
    cb.camera.resolution, cb.sparseRenderingArgs, COMPACT_ACTIVE_PIXELS_LAUNCH_RTXDI_INITIAL, remapped);
  int2 pixelCoordinate = int2(remapped);

  // Inactive pixels are handled by the temporal reuse pass (cheap reprojection or skip).
  if (!isActive)
  {
    return;
  }

  RAB_Surface surface = RAB_GetGBufferSurface(pixelCoordinate, false);
  if (!RAB_IsSurfaceValid(surface))
  {
    return;
  }

  RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelCoordinate, cb.frameIdx, 2);

  vec2 screenSpaceMotion = PrimaryScreenSpaceMotionVector[pixelCoordinate];
  ivec2 prevPixelCenter = pixelCoordinate + ivec2(screenSpaceMotion.xy);

  uint16_t previousBestLightIndex = RTXDI_INVALID_LIGHT_INDEX;
  uint8_t previousBestLightPortalIdx = RTXDI_INVALID_PORTAL_INDEX;
  if (all(prevPixelCenter >= 0) && all(prevPixelCenter < cb.camera.resolution) && cb.enableRtxdiBestLightSampling)
  {
    uint2 previousBestLight = RtxdiBestLights[prevPixelCenter / RTXDI_GRAD_FACTOR];

    // Map the previous light index to the current frame, make sure the portal still exists if that light was a cross-portal one
#if RTXDI_NO_PORTALS
    if (previousBestLight.x != RTXDI_INVALID_LIGHT_INDEX && previousBestLight.y == RTXDI_INVALID_PORTAL_INDEX)
#else
    if (previousBestLight.x != RTXDI_INVALID_LIGHT_INDEX && (previousBestLight.y == RTXDI_INVALID_PORTAL_INDEX || previousBestLight.y < cb.numActiveRayPortals))
#endif
    {
      previousBestLightIndex = RAB_TranslateLightIndex(previousBestLight.x, /* currentToPrevious = */ false);
      previousBestLightPortalIdx = previousBestLight.y;
    }
  }

  RTXDI_Reservoir state = RTXDI_EmptyReservoir();
  if (cb.lightCount != 0)
  {
    RTXDI_SampleRandomLights(state, rng, surface, previousBestLightIndex, previousBestLightPortalIdx,
      cb.enableRtxdiInitialVisibility && !RTXDI_DEFER_INITIAL_VISIBILITY);

    RTXDI_Reservoir bestLightReservoir = RTXDI_SampleSingleLight(rng, surface, previousBestLightIndex, previousBestLightPortalIdx,
      cb.enableRtxdiInitialVisibility && !RTXDI_DEFER_INITIAL_VISIBILITY);
    float random = RAB_GetNextRandom(rng);
    RTXDI_CombineReservoirs(state, bestLightReservoir, random, bestLightReservoir.targetPdf);
#if ENABLE_LOCAL_SPHERES
    RAB_SampleLocalLights(state, rng, surface);
#endif

#if RTXDI_DEFER_INITIAL_VISIBILITY
    if (cb.enableRtxdiInitialVisibility)
    {
      RTXDI_UpdateSelectedReservoirVisibility(state, surface);
    }
#endif

    // Note: I have NO IDEA why this works, but it's necessary to get unbiased results with best light sampling.
    state.M = 1;
  }

  RAB_StoreReservoir(state, pixelCoordinate, RTXDI_GetTemporalOutputPage());
}
