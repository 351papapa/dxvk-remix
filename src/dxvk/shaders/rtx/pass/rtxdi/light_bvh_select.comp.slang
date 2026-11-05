//!feature wetness ENABLE_SURFACE_WETNESS
/*
* Light BVH selection pass.
*
* Runs in the RTXDI slot when rtx.lightSampling.mode = LightBVH and bvhSeparateSelectPass is on. It picks
* the primary-surface light sample (BVH traversal + RIS over bvhPrimaryCandidates) in a small, high
* occupancy compute shader and stores it as an RTXDI reservoir in the spatial output page. The direct
* integrator and indirect sample stealing read it exactly like an RTXDI result:
*   lightIdx, uv, weightSum = light-UV RIS weight, M = 1, no portal.
* This keeps the dependent-load BVH walk out of the direct integration shader, which runs at low occupancy
* and so cannot hide the traversal's latency.
*
* Uses the RTXDI reuse bindings (the same parameter block as rtxdi_initial_sampling).
*/

//!variant light_bvh_select.comp

// The pass with the subsurface receiver handling (thin-opaque two-sided importance, diffusion-profile transmission in
// the approximate BSDF weight, the two subsurface reads) and the light types this frame lacks compiled out.
// selectLightBvhSelectShader (rtx_fork_light_bvh.cpp) binds them only on frames where those branches cannot run.
// The estimator is the same, equal to the full pass up to floating-point contraction.
//!variant light_bvh_select_slim.comp
//!>       LIGHT_BVH_SELECT_SLIM=1 ENABLE_RECT_LIGHTS=0 ENABLE_DISK_LIGHTS=0 ENABLE_CYLINDER_LIGHTS=0

//!variant light_bvh_select_slim_area.comp
//!>       LIGHT_BVH_SELECT_SLIM=1 ENABLE_CYLINDER_LIGHTS=0

//!variant light_bvh_select_slim_all.comp
//!>       LIGHT_BVH_SELECT_SLIM=1

//!end-variants

#define USE_32BIT_RAY_DIRECTION 1
#define RAB_HAS_CURRENT_GBUFFER 1
#define RAB_HAS_RTXDI_RESERVOIRS 1
#define RTXDI_NO_PORTALS 1
// Enables the select-pass-only code in light_bvh.slangh.
#define LIGHT_BVH_SELECT_PASS 1
#ifdef LIGHT_BVH_SELECT_SLIM
// No subsurface receiver (rtxdi_application_bridge_ext.slangh RAB_GetGBufferSurface).
#define LIGHT_BVH_SELECT_NO_SUBSURFACE 1
#endif

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
#include "rtx/pass/instance_definitions.h"
#include "rtx/algorithm/light_bvh.slangh"
#include "rtx/pass/sparse_rendering/compact_active_pixels.slangh"

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
  const int2 pixelCoordinate = int2(remapped);

  if (!isActive)
  {
    return;
  }

  RTXDI_Reservoir reservoir = RTXDI_EmptyReservoir();

  // Sky and background pixels get an empty reservoir and no light work.
  const RAB_Surface surface = RAB_GetGBufferSurface(pixelCoordinate, false);
  if (cb.enableDirectLighting && RAB_IsSurfaceValid(surface))
  {
    // The RNG is built like the direct integrator's, with its own offset so the selection does not
    // reuse the numbers the integrator draws afterwards.
    RNG randomState = createRNG(uvec2(pixelCoordinate), cb.frameIdx, 0x4c42u);
    const uint customIndex = surface.isViewModel ? CUSTOM_INDEX_IS_VIEW_MODEL : 0;

    // rtx.lightSampling.bvhSunOcclusionHint. Last frame's reservoir at this pixel holds the distant light with
    // weight 0 when the direct integrator found it occluded. The distant-light probability then drops to its minimum,
    // so shadowed pixels stop spending most of their shadow rays on long sun rays that the unshadowed BVH importance
    // cannot know are blocked. RTXDI gets the same effect from visibility reuse.
    float pInfiniteCap = 1.0f;
    // With the dedicated sun sample the integrator samples distant lights itself, so this reservoir covers
    // local lights only, and the host clears the sun hint.
    const bool excludeDistant = cb.lightBvhDedicatedSun != 0;
    if (cb.lightBvhSunOcclusionHint != 0)
    {
      const RTXDI_Reservoir previous = RAB_LoadReservoir(pixelCoordinate, RTXDI_GetTemporalInputPage());
      const LightRangeInfo distantRange = cb.lightRanges[lightTypeDistant];
      if (previous.lightIdx != RTXDI_INVALID_LIGHT_INDEX && previous.weightSum == 0.0f &&
          uint(previous.lightIdx) >= distantRange.offset && uint(previous.lightIdx) < distantRange.offset + distantRange.count)
      {
        pInfiniteCap = 0.0f;
      }
    }

    uint lightIdx;
    vec2 uv;
    LightSample lightSample;
    float weight;
    if (lightBvhSelect(randomState, surface.minimalSurfaceInteraction, surface.opaqueSurfaceMaterialInteraction,
                       surface.minimalRayInteraction.viewDirection, cb.lightBvhPrimaryCandidateCount, customIndex,
                       lightIdx, uv, lightSample, weight, pInfiniteCap, excludeDistant))
    {
      reservoir.lightIdx = uint16_t(lightIdx);
      reservoir.portalIdx = RTXDI_INVALID_PORTAL_INDEX;
      reservoir.uv = uv;
      reservoir.weightSum = weight;
      reservoir.targetPdf = 1.0f;
      reservoir.M = 1;
      reservoir.lightIdentity = RAB_UseLightIdentityResolution() ? RAB_GetLightIdentity(uint16_t(lightIdx)) : 0u;
    }
  }

  RAB_StoreReservoir(reservoir, pixelCoordinate, RTXDI_GetSpatialOutputPage());
}
