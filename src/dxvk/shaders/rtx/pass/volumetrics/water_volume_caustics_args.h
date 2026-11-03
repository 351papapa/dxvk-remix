#pragma once

#include "rtx/utility/shader_types.h"

#define VOLUME_WATER_CAUSTICS_PATTERN 48
#define VOLUME_WATER_CAUSTICS_SURFACE_INPUT 49

struct WaterVolumeCausticsArgs {
  float strength;
  float scale;
  float speed;
  float maxDepth;
  uint randomized;
  uint sunIndex;
  float surfaceHeight;
  float fadeStart;
  float receiverStrength;
  float textureMaxLod;
  vec2 textureSize;
  uint padding0;
  uint padding1;
  uint padding2;
  uint padding3;
};
