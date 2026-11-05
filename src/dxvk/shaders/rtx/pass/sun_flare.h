#pragma once
#include "rtx/utility/shader_types.h"

#define SUN_FLARE_STATE 30
#define SUN_FLARE_COLOR 31
#define SUN_FLARE_VC_STAR 32
#define SUN_FLARE_VC_HEX 33
#define SUN_FLARE_VC_CIRCLE 34
#define SUN_FLARE_VC_RING 35
#define SUN_FLARE_SA_STAR 36

// RGBA32F, 2x1. Texel 0: sun UV, visibility, camera water depth.
// Texel 1: display sun tint RGB, zero for invalid or 1 + resurfacing recovery seconds.
struct SunFlareStateArgs {
  uint resetHistory;
  uint visibilitySamples;
  uint suppressUnderwater;
  uint cloudOcclusion;
  float deltaTime;
  float fadeInSeconds;
  float fadeOutSeconds;
  float waterTransition; // Minimum resurfacing fade-in duration in seconds.
  float maxRayDistance;
  float waterProbeDistance;
  float edgeFade;
  uint cameraMediumState; // 0 unknown, 1 configured dry, 2 configured water.
};

struct SunFlareArgs {
  uint width;
  uint height;
  uint style;
  uint textureSet;
  float brightness;
  float sizeScale;
  float vcAspectRatio;
  float timeSeconds;
  uint scaleWithResolution;
  uint saFlicker;
  uint frameIndex;
  uint reserved;
};
