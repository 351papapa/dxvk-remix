#pragma once

// Cheaper visibility (shadow/occlusion) rays through alpha-tested geometry.
//
// Two independent switches, both read by rtx/algorithm/visibility.slangh:
// - firstHitPrimaryNee: the primary-surface NEE shadow ray (evalNEEPrimary) stops at the first confirmed
//   opaque hit instead of searching for the closest one, as every other visibility query does.
// - coarseAlphaMip: cutout (alpha-test) reads on any visibility ray use a fixed coarse mip instead of the
//   ray-cone footprint (which is zero, i.e. mip 0, on RTXDI's visibility rays).

#include <cstdint>

#include "rtx_option.h"

struct RaytraceArgs;

namespace dxvk {

  class ShadowRays {
  public:
    RTX_OPTION("rtx.shadowRays", bool, firstHitPrimaryNee, false,
               "Primary-surface shadow rays (direct lighting, with RTXDI or LightBVH) stop at the first confirmed "
               "opaque hit instead of searching for the closest one, which is cheaper. Visibility and lighting are the "
               "same. Only the hit distance of occluded samples changes, which only NRD reads, so this is ignored while "
               "NRD is the denoiser.");
    RTX_OPTION("rtx.shadowRays", bool, coarseAlphaMip, false,
               "Alpha-tested (cutout) texture reads on visibility rays (RTXDI visibility, shadow rays in direct and "
               "indirect lighting, volumetric and gradient visibility) sample a fixed coarse mip "
               "(rtx.shadowRays.alphaMipTargetResolution) instead of the ray cone's mip. Saves GPU time through "
               "foliage, and shadowed leaf and grass edges may soften slightly. Triangles resolved by opacity micromaps "
               "need no read.");
    RTX_OPTION("rtx.shadowRays", uint32_t, alphaMipTargetResolution, 128,
               "With rtx.shadowRays.coarseAlphaMip: visibility alpha tests read the mip whose larger side has this many "
               "texels (a 1024 texture reads mip 3, a 256 texture mip 1, and textures at or below it mip 0). A coarser "
               "ray-cone mip is kept. Lower values are cheaper and give softer cutout shadows. 4 to 4096.");

    static void setRaytraceArgs(RaytraceArgs& args, bool nrdDenoises);
    static void showImguiSettings();
  };

} // namespace dxvk
