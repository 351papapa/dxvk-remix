#pragma once
#include "rtx/utility/shader_types.h"

#define CAUSTICS_PRIMARY_POSITION 30
#define CAUSTICS_SECONDARY_POSITION 31
#define CAUSTICS_PRIMARY_NORMAL 32
#define CAUSTICS_SECONDARY_NORMAL 33
#define CAUSTICS_PRIMARY_ALBEDO 34
#define CAUSTICS_SECONDARY_ALBEDO 35
#define CAUSTICS_FLAGS 36
#define CAUSTICS_MATERIAL 37
#define CAUSTICS_PDF 38
#define CAUSTICS_PRIMARY_CONE 39
#define CAUSTICS_SECONDARY_CONE 40
#define CAUSTICS_PATTERN 41
#define CAUSTICS_CACHE 42
#define CAUSTICS_PRIMARY_RADIANCE 43
#define CAUSTICS_SECONDARY_RADIANCE 44
#define CAUSTICS_ACTIVE_PIXELS 45

struct WaterCausticsArgs {
  float strength;
  float scale;
  float speed;
  float maxDepth;
  vec3 transmittance;
  float transmittanceDistance;
  uint randomized;
  float reflectedStrength;
  float reflectedHeight;
  // Ocean and authored (pool / fountain) water are separate art domains: a pool is
  // small and still where the ocean is a wide swell, so their caustic webs want
  // different tiling. The shader picks between `scale` and `poolScale` from the
  // water surface the refracted ray actually hit.
  //
  // This occupies the word that used to be an unused `padding` field, so the
  // push-constant block keeps its 128-byte layout and every following member keeps
  // its offset. That matters: reshaping this block without re-verifying the
  // Slang-side offsets has crashed the engine before.
  float poolScale;
  vec3 lightDirection;
  uint lightValid;
  vec3 lightIrradiance;
  float localLightRange;
  uint localLightIndices[8];
  uint localLightCount;
  float fadeStart;
  // Chromatic dispersion. The pattern is evaluated three times, offset before and
  // after along the receiver normal's horizontal projection, and the three results
  // drive the output R / G / B channels separately. That is what makes the filaments
  // grow colour fringes instead of merely smearing.
  //
  // Modelled on ImmersiveVC (see K:\Harness\ImmersiveVC-焦散色散光柱-汇编级完整解析.md):
  // its offset is `normalize(normal.xy) * Disp * 0.08 * saturate(depth * 0.5)` in
  // caustic-tile units, so 0.04 of a tile at the default 0.5.
  //
  // Reuses the word that was an unused `localPadding0`, keeping the 128-byte layout
  // and every other member offset unchanged.
  float dispersion;
  float localPadding1;
};
