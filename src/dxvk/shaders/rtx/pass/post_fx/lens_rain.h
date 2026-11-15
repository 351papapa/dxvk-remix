// SPDX-License-Identifier: MIT
#pragma once
#include "rtx/utility/shader_types.h"
#define LENS_RAIN_INPUT 0
#define LENS_RAIN_OUTPUT 1
#define LENS_RAIN_DROPS 2
#define LENS_RAIN_TILES 3
#define LENS_RAIN_STATES 4
#define LENS_RAIN_MAX_DROPS 1024
#define LENS_RAIN_EXTERNAL_DROPS 1024
#define LENS_RAIN_TOTAL_DROPS (LENS_RAIN_MAX_DROPS + LENS_RAIN_EXTERNAL_DROPS)
struct LensRainDrop {
  float4 centerRadii;
  float4 axisOpacityStrength;
  float4 absorptionKind;
};
struct LensRainState {
  float2 position;
  float2 velocity;
  float2 axis;
  float2 motionDirection;
  float radius;
  float age;
  float lifetime;
  float mobility;
  float gravity;
  float responseTime;
  float drift;
  float phase;
  float strength;
  uint seed;
  float2 padding;
};
struct LensRainArgs {
  uint2 imageSize;
  float2 invImageSize;
  uint2 tileCount;
  float amount;
  float refractPixels;
  float dispersion;
  float blurPixels;
  uint externalCount;
  uint tileDropCapacity;
  vec3 flowVelocity;
  float dt;
  float aspect;
  float previousAspect;
  float density;
  float sizeScale;
  float slideSpeed;
  uint dropCount;
  uint previousCount;
  uint allowSpawning;
  // x: sampled values still use display gamma 2.2; remaining components are reserved.
  uint4 colorFlags;
};
#ifdef __cplusplus
static_assert(sizeof(LensRainArgs) == 112);
static_assert(sizeof(LensRainDrop) == 48);
static_assert(sizeof(LensRainState) == 80);
#endif
