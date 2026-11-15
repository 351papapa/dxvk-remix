#pragma once
#include "rtx/utility/shader_types.h"

#define CORONA_SOURCE_LIMIT 4096
#define CORONA_INPUT 30
#define CORONA_SPRITES 31
#define CORONA_COLOR 32

struct LocalCoronaArgs {
  uint width;
  uint height;
  uint count;
  float sizeScale;
  float brightness;
  float endpointBias;
  float reserved0;
  float reserved1;
};

struct LocalCoronaInput {
  vec4 positionSize;
  vec4 colorOpacity;
  float falloff;
  uint kind;
  uint idLow;
  uint idHigh;
};

struct LocalCoronaSprite {
  vec4 centerHalfSize;
  vec4 tint;
};
