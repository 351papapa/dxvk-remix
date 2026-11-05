#pragma once

#include "rtx/utility/shader_types.h"

#define GRASS_CONSTANTS 50
#define GRASS_ROOTS 51
#define GRASS_CANDIDATES 52
#define GRASS_STATE 53
#define GRASS_GROUPS 54
#define GRASS_SELECTED 55
#define GRASS_PREVIOUS 56
#define GRASS_INSTANCES 57
#define GRASS_SURFACES 58
#define GRASS_MATERIALS 59
#define GRASS_MAPPING 60

#define GRASS_GROUP_SIZE 256
#define GRASS_INVALID 0xffffffffu

struct GrassRoot {
  vec4 positionScale;
  vec4 normalAngle;
  uint identityLo;
  uint identityHi;
  uint pad0;
  uint pad1;
};

struct GrassSelection {
  mat4 transform;
  mat4 previousTransform;
  uint previousSlot;
  uint valid;
  uint identityLo;
  uint identityHi;
};

struct GrassConstants {
  vec4 cameraRadius;
  vec4 forwardFade;
  vec4 dimensions;
  vec4 bounds;
  vec4 viewPlanes[5];
  uint rootCount;
  uint budget;
  uint groupCount;
  uint radixShift;
  uint previousBudget;
  uint frontRear;
  uint frustumCull;
  uint zUp;
  mat4 prototypeTransform;
  uint templateSurface;
  uint outputSurface;
  uint previousSurface;
  uint instanceOffset;
  uint customIndexFlags;
  uint instanceMask;
  uint sbtOffsetAndFlags;
  uint blasLo;
  uint blasHi;
  uint previousInstanceCount;
  uint pad2;
  uint pad3;
};
