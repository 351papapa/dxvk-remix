// SPDX-License-Identifier: MIT
#pragma once
#include "rtx/utility/shader_types.h"
#define LENS_RAIN_INPUT 0
#define LENS_RAIN_OUTPUT 1
#define LENS_RAIN_DROPS 2
#define LENS_RAIN_TILES 3
#define LENS_RAIN_STATES 4
#define LENS_RAIN_MAX_DROPS 1024
struct LensRainDrop {
  float4 centerRadii;
  float4 axisOpacityStrength;
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
  float2 padding;
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
};
#ifdef __cplusplus
static_assert(sizeof(LensRainArgs) == 96);
static_assert(sizeof(LensRainDrop) == 32);
static_assert(sizeof(LensRainState) == 80);
#endif
