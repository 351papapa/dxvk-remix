#pragma once
#include "rtx/utility/shader_types.h"
#define UNDERWATER_DEPTH 30
#define UNDERWATER_COLOR 31
#define UNDERWATER_PATTERN 32
#define UNDERWATER_SHAFT_FIELD 33
struct WaterUnderwaterArgs {
  uint width;
  uint height;
  float density;
  float shaftStrength;
  vec3 fogColor;
  float transition;
  float shaftScale;
  float shaftMaxDistance;
  float shaftAnisotropy;
  uint shaftSamples;
  uint debugMode;
  uint sceneLightCandidates;
  float sceneLightStrength;
  uint patternAvailable;
};
