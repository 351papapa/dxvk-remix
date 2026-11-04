#pragma once
#include "rtx/utility/shader_types.h"

#define STARBURST_INPUT 30
#define STARBURST_SPRITES 31
#define STARBURST_COLOR 32
#define STARBURST_MASK 33

struct LocalStarburstArgs {
  uint width;
  uint height;
  uint count;
  float sizeScale;
  float brightness;
  float endpointBias;
  float reserved0;
  float reserved1;
};

struct LocalStarburstInput {
  vec4 positionSize;
  vec4 colorOpacity;
  float aspect;
  uint kind;
  uint idLow;
  uint idHigh;
};

struct LocalStarburstSprite {
  vec4 centerHalfSize;
  vec4 tint;
};
